#!/usr/bin/env python3
"""Fetch exact agkphysics/AudioSet rows through the HF dataset viewer.

The source Parquet files store 100 encoded clips per row group.  Reading one
selected clip through Parquet therefore transfers the complete audio column
chunk.  The Hugging Face dataset viewer exposes the same encoded FLAC as an
individual cached asset.  This sidecar resolves and downloads only the rows in
an already-frozen QCES plan, verifies their identity/labels/revision, and emits
a normal source manifest that the existing crop materializer can consume.

Downloads are resumable per video and never alter the selection plan.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from huggingface_hub import get_token

from mixi_understanding.qces.audioset_plan_materializer import (
    atomic_json,
    atomic_jsonl,
    sha256_file,
    validate_audio_path,
)


FORMAT = "qces_agkphysics_direct_asset_manifest_v1"
SUPPORTED_DATASET = "agkphysics/AudioSet"
SUPPORTED_REVISION = "0c609e8302cf139307f639c57652032af0a88041"
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
PARQUET_RE = re.compile(
    r"/data/(?P<partition>bal_train|eval|unbal_train)/(?P<shard>\d+)\.parquet$"
)
HF_TOKEN = get_token()


class DirectAssetError(RuntimeError):
    pass


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DirectAssetError(f"invalid object at {path}:{line_number}")
            rows.append(value)
    return rows


def load_shard_offsets(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    """Build exact prefix offsets from the pinned availability metadata."""

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for value in _load_jsonl(path):
        if value.get("hf_dataset") != SUPPORTED_DATASET:
            continue
        if value.get("hf_revision") != SUPPORTED_REVISION:
            continue
        match = PARQUET_RE.search(str(value.get("parquet_url") or ""))
        if match is None:
            continue
        grouped.setdefault(match.group("partition"), []).append(
            (int(match.group("shard")), value)
        )
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for partition, shards in grouped.items():
        prefix = 0
        expected_shard = 0
        for shard, metadata in sorted(shards):
            if shard != expected_shard:
                raise DirectAssetError(
                    f"non-contiguous {partition} shard index: {shard} != {expected_shard}"
                )
            row_group_prefix: dict[int, int] = {}
            within = 0
            for group in sorted(metadata.get("row_groups") or [], key=lambda x: x["index"]):
                index = int(group["index"])
                row_group_prefix[index] = within
                within += int(group["num_rows"])
            if within != int(metadata.get("num_rows") or -1):
                raise DirectAssetError(f"row-group count mismatch for {partition}/{shard}")
            output[(partition, shard)] = {
                "prefix": prefix,
                "num_rows": within,
                "row_group_prefix": row_group_prefix,
            }
            prefix += within
            expected_shard += 1
    if not output:
        raise DirectAssetError(f"no pinned agkphysics shard metadata in {path}")
    return output


def viewer_coordinate(
    row: dict[str, Any],
    shard_offsets: dict[tuple[str, int], dict[str, Any]],
) -> tuple[str, int]:
    """Map a pinned source-Parquet location to the viewer's global row."""

    location = row.get("availability_location") or {}
    if location.get("hf_dataset") != SUPPORTED_DATASET:
        raise DirectAssetError(f"unsupported dataset for {row.get('video_id')}")
    if location.get("hf_revision") != SUPPORTED_REVISION:
        raise DirectAssetError(f"unsupported revision for {row.get('video_id')}")
    match = PARQUET_RE.search(str(location.get("parquet_url") or ""))
    if match is None:
        raise DirectAssetError(f"unsupported parquet URL for {row.get('video_id')}")
    partition = match.group("partition")
    shard = int(match.group("shard"))
    row_group = int(location.get("row_group", -1))
    row_index = int(location.get("row_index", -1))
    metadata = shard_offsets.get((partition, shard))
    if metadata is None or row_group not in metadata["row_group_prefix"]:
        raise DirectAssetError(f"missing shard metadata for {row.get('video_id')}")
    group_start = int(metadata["row_group_prefix"][row_group])
    next_starts = [
        int(value)
        for index, value in metadata["row_group_prefix"].items()
        if int(index) > row_group
    ]
    group_end = min(next_starts) if next_starts else int(metadata["num_rows"])
    if row_group < 0 or not 0 <= row_index < group_end - group_start:
        raise DirectAssetError(f"invalid row location for {row.get('video_id')}")
    offset = int(metadata["prefix"]) + group_start + row_index
    return partition, offset


def _request_bytes(url: str, *, attempts: int, timeout: int) -> bytes:
    error: BaseException | None = None
    for attempt in range(attempts):
        try:
            headers = {"User-Agent": "QCES-direct-asset-fetcher/1.0"}
            if HF_TOKEN:
                headers["Authorization"] = f"Bearer {HF_TOKEN}"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as caught:
            error = caught
            # Dataset-viewer quotas are shared at account/IP level.  A long,
            # quiet backoff recovers faster than an exponential retry burst.
            if caught.code == 429:
                delay = 60.0
            elif caught.code in {500, 502, 503, 504}:
                delay = min(30.0, 5.0 * (attempt + 1))
            else:
                delay = min(20.0, 0.75 * (2**attempt))
            if attempt + 1 < attempts:
                time.sleep(delay)
        except (OSError, TimeoutError, urllib.error.URLError) as caught:
            error = caught
            if attempt + 1 < attempts:
                time.sleep(min(20.0, 0.75 * (2**attempt)))
    raise DirectAssetError(f"request failed after {attempts} attempts: {error}")


def _resolve_asset(
    row: dict[str, Any],
    shard_offsets: dict[tuple[str, int], dict[str, Any]],
    *,
    attempts: int,
) -> tuple[str, int, str]:
    partition, offset = viewer_coordinate(row, shard_offsets)
    query = urllib.parse.urlencode(
        {
            "dataset": SUPPORTED_DATASET,
            "config": "full",
            "split": partition,
            "offset": offset,
            "length": 1,
        }
    )
    payload = json.loads(
        _request_bytes(f"{ROWS_ENDPOINT}?{query}", attempts=attempts, timeout=60)
    )
    result_rows = payload.get("rows") or []
    if len(result_rows) != 1:
        raise DirectAssetError(f"viewer returned {len(result_rows)} rows at {partition}:{offset}")
    result = result_rows[0].get("row") or {}
    expected_id = str(row.get("video_id") or "")
    if str(result.get("video_id") or "") != expected_id:
        raise DirectAssetError(
            f"viewer identity mismatch at {partition}:{offset}: "
            f"{result.get('video_id')} != {expected_id}"
        )
    location = row.get("availability_location") or {}
    expected_labels = sorted(str(value) for value in location.get("labels") or [])
    actual_labels = sorted(str(value) for value in result.get("labels") or [])
    if expected_labels != actual_labels:
        raise DirectAssetError(f"viewer label mismatch for {expected_id}")
    audio = result.get("audio") or []
    if not isinstance(audio, list) or len(audio) != 1 or not audio[0].get("src"):
        raise DirectAssetError(f"viewer returned no audio asset for {expected_id}")
    source_url = str(audio[0]["src"])
    if f"/{SUPPORTED_REVISION}/" not in source_url:
        raise DirectAssetError(f"viewer asset revision mismatch for {expected_id}")
    return partition, offset, source_url


def _atomic_download(url: str, output_path: Path, *, attempts: int) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _request_bytes(url, attempts=attempts, timeout=180)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        validate_audio_path(temporary)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return validate_audio_path(output_path)


def _completed_video_ids(materialized_dir: Path | None) -> set[str]:
    if materialized_dir is None:
        return set()
    output: set[str] = set()
    for path in sorted((materialized_dir / "manifest_fragments").glob("*.jsonl")):
        for row in _load_jsonl(path):
            video_id = str(row.get("video_id") or "")
            if video_id:
                output.add(video_id)
    return output


def _reuse_one(row: dict[str, Any], output_dir: Path) -> dict[str, Any] | None:
    video_id = str(row["video_id"])
    audio_path = output_dir / "audio" / f"{video_id}.flac"
    receipt_path = output_dir / "fragments" / f"{video_id}.json"
    if receipt_path.is_file() and audio_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        info = validate_audio_path(audio_path, str(receipt["audio_sha256"]))
        return dict(receipt, audio_bytes=int(info["audio_bytes"]), reused=True)
    return None


def _download_one(
    row: dict[str, Any],
    output_dir: Path,
    resolved: tuple[str, int, str],
    *,
    attempts: int,
) -> dict[str, Any]:
    video_id = str(row["video_id"])
    audio_path = output_dir / "audio" / f"{video_id}.flac"
    receipt_path = output_dir / "fragments" / f"{video_id}.json"
    partition, offset, source_url = resolved
    info = _atomic_download(source_url, audio_path, attempts=attempts)
    location = row.get("availability_location") or {}
    receipt = {
        "format": FORMAT,
        "video_id": video_id,
        "mixture_path": str(audio_path.resolve()),
        "audio_sha256": str(info["audio_sha256"]),
        "audio_bytes": int(info["audio_bytes"]),
        "sample_rate": int(info["sample_rate"]),
        "duration_seconds": float(info["duration_seconds"]),
        "protocol_upstream_split": str(location.get("hf_split") or ""),
        "hf_dataset": SUPPORTED_DATASET,
        "hf_revision": SUPPORTED_REVISION,
        "source_route": str(location.get("source_route") or ""),
        "viewer_config": "full",
        "viewer_split": partition,
        "viewer_offset": offset,
        "identity_verified": True,
        "labels_verified": True,
        "revision_verified": True,
    }
    atomic_json(receipt_path, receipt)
    return dict(receipt, reused=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--materialized-dir", type=Path)
    parser.add_argument("--shard-index", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resolve-workers", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be in [1, 32]")
    if not 1 <= args.resolve_workers <= 4:
        parser.error("--resolve-workers must be in [1, 4]")
    if args.attempts < 1:
        parser.error("--attempts must be positive")

    plan_path = args.plan.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_jsonl(plan_path)
    shard_offsets = load_shard_offsets(args.shard_index.resolve())
    skipped = _completed_video_ids(
        args.materialized_dir.resolve() if args.materialized_dir else None
    )
    pending = [row for row in rows if str(row.get("video_id") or "") not in skipped]
    completed: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for row in pending:
        reused = _reuse_one(row, output_dir)
        if reused is None:
            unresolved.append(row)
        else:
            completed.append(reused)

    with ThreadPoolExecutor(max_workers=args.resolve_workers) as resolver, \
            ThreadPoolExecutor(max_workers=args.workers) as downloader:
        resolve_futures = {
            resolver.submit(
                _resolve_asset,
                row,
                shard_offsets,
                attempts=args.attempts,
            ): row
            for row in unresolved
        }
        download_futures = {}
        for index, future in enumerate(as_completed(resolve_futures), 1):
            row = resolve_futures[future]
            coordinate = future.result()
            download_future = downloader.submit(
                _download_one,
                row,
                output_dir,
                coordinate,
                attempts=args.attempts,
            )
            download_futures[download_future] = row
            if index == 1 or index % 10 == 0 or index == len(resolve_futures):
                print(
                    f"[direct-assets:resolve] {index}/{len(resolve_futures)} "
                    f"(reused direct: {len(completed)}, "
                    f"already materialized: {len(skipped)})",
                    flush=True,
                )
        for index, future in enumerate(as_completed(download_futures), 1):
            completed.append(future.result())
            if index == 1 or index % 10 == 0 or index == len(download_futures):
                print(
                    f"[direct-assets:download] {index}/{len(download_futures)}",
                    flush=True,
                )

    completed.sort(key=lambda row: str(row["video_id"]))
    manifest_path = output_dir / "direct_asset_source_manifest.jsonl"
    atomic_jsonl(manifest_path, completed)
    receipt = {
        "format": FORMAT,
        "complete": True,
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "plan_videos": len(rows),
        "skipped_already_materialized": len(skipped),
        "direct_assets": len(completed),
        "workers": args.workers,
        "resolve_workers": args.resolve_workers,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "audio_bytes": sum(int(row["audio_bytes"]) for row in completed),
        "invariants": {
            "frozen_plan_unchanged": True,
            "viewer_video_identity_verified": all(row["identity_verified"] for row in completed),
            "viewer_labels_verified": all(row["labels_verified"] for row in completed),
            "pinned_revision_verified": all(row["revision_verified"] for row in completed),
        },
    }
    atomic_json(output_dir / "direct_asset_receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
