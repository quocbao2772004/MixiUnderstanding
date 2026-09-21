#!/usr/bin/env python3
"""Download only the AudioTime sources selected by the deterministic QA builder."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf

from mixi_understanding.scripts.build_qa_removal_dataset import (
    DEFAULT_EVENT_DURATION_SECONDS,
    DEFAULT_NON_OVERLAP_INTERFERENCE_DURATION_SECONDS,
    DEFAULT_OVERLAP_INTERFERENCE_DURATION_SECONDS,
    FamilySources,
    SourceClip,
    choose_sources,
    load_sources,
)


DATASET_SERVER = "https://datasets-server.huggingface.co"
DATASET_NAME = "enyoukai/audiotime-timestamps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audiotime-root",
        type=Path,
        default=PROJECT_ROOT / "AudioTime-recovered" / "train5000_timestamp",
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_sources(root: Path, seed: int) -> List[SourceClip]:
    available_from_metadata = load_sources(root, require_audio=False)
    families = choose_sources(
        available_from_metadata,
        event_duration=DEFAULT_EVENT_DURATION_SECONDS,
        overlap_duration=DEFAULT_OVERLAP_INTERFERENCE_DURATION_SECONDS,
        non_overlap_duration=DEFAULT_NON_OVERLAP_INTERFERENCE_DURATION_SECONDS,
        seed=seed,
    )
    selected: Dict[str, SourceClip] = {}
    for family in families:
        _collect_family_sources(family, selected)
    return sorted(selected.values(), key=lambda source: source.source_id)


def _collect_family_sources(
    family: FamilySources, destination: Dict[str, SourceClip]
) -> None:
    for label_sources in family.semantic_clips:
        for source in label_sources:
            destination[source.source_id] = source
    for source in family.interference_clips:
        destination[source.source_id] = source


def request_json(path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{DATASET_SERVER}{path}?{urllib.parse.urlencode(parameters)}"
    error: Exception | None = None
    # The public datasets-server rate-limits long row scans.  A short retry
    # loop made deterministic acquisitions fail around page 35, so honour the
    # server's Retry-After value and allow a longer capped backoff.
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise TypeError(f"dataset server returned non-object payload: {url}")
            return payload
        except Exception as exc:  # network errors vary across Python versions
            error = exc
            if attempt < 9:
                retry_after = None
                if isinstance(exc, urllib.error.HTTPError):
                    retry_after = exc.headers.get("Retry-After")
                try:
                    server_delay = float(retry_after) if retry_after else 0.0
                except ValueError:
                    server_delay = 0.0
                delay = max(server_delay, min(45.0, float(2 ** (attempt + 1))))
                print(
                    f"dataset server retry {attempt + 1}/9 in {delay:.0f}s: "
                    f"{getattr(exc, 'code', type(exc).__name__)}",
                    file=sys.stderr,
                )
                time.sleep(delay)
    raise RuntimeError(f"dataset server request failed: {url}") from error


def request_url_json(url: str) -> Dict[str, Any]:
    error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise TypeError(f"server returned non-object payload: {url}")
            return payload
        except Exception as exc:
            error = exc
            if attempt < 4:
                time.sleep(2**attempt)
    raise RuntimeError(f"request failed: {url}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_assets(
    wanted_ids: Iterable[str], dataset: str, config: str, split: str, page_size: int
) -> Dict[str, str]:
    wanted = set(wanted_ids)
    found: Dict[str, str] = {}
    offset = 0
    while wanted - set(found):
        payload = request_json(
            "/rows",
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": page_size,
            },
        )
        rows = payload.get("rows")
        if not isinstance(rows, list) or not rows:
            break
        for wrapped in rows:
            row = wrapped.get("row", {})
            source_id = row.get("audio_id")
            if source_id not in wanted:
                continue
            audio = row.get("audio")
            if not isinstance(audio, list) or not audio:
                raise ValueError(f"missing audio asset for {source_id}")
            source_url = audio[0].get("src")
            if not isinstance(source_url, str) or not source_url:
                raise ValueError(f"missing audio URL for {source_id}")
            found[source_id] = source_url
        offset += len(rows)
        print(f"indexed {offset} rows; found {len(found)}/{len(wanted)} sources")
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"selected source IDs are absent from Hub dataset: {missing}")
    return found


def download_file(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_audio(path: Path, source: SourceClip) -> None:
    info = sf.info(path)
    duration = info.frames / info.samplerate
    if info.channels <= 0 or info.frames <= 0:
        raise ValueError(f"invalid audio file: {path}")
    if duration + 1e-3 < source.interval_seconds[1]:
        raise ValueError(
            f"{source.source_id} is {duration:.3f}s but metadata needs "
            f"{source.interval_seconds[1]:.3f}s"
        )


def main() -> None:
    args = parse_args()
    if not 1 <= args.page_size <= 100:
        raise SystemExit("--page-size must be in [1, 100]")
    root = args.audiotime_root.resolve()
    sources = selected_sources(root, args.seed)
    output_root = root / "audio"
    output_root.mkdir(parents=True, exist_ok=True)
    pending = [
        source
        for source in sources
        if args.overwrite or not (output_root / f"{source.source_id}.wav").exists()
    ]
    print(f"deterministic selection: {len(sources)} sources; download: {len(pending)}")
    if pending:
        assets = discover_assets(
            (source.source_id for source in pending),
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            page_size=args.page_size,
        )
        for index, source in enumerate(pending, start=1):
            destination = output_root / f"{source.source_id}.wav"
            download_file(assets[source.source_id], destination)
            validate_audio(destination, source)
            print(f"downloaded {index}/{len(pending)}: {source.source_id}")
    for source in sources:
        validate_audio(output_root / f"{source.source_id}.wav", source)
    dataset_info_url = (
        "https://huggingface.co/api/datasets/"
        + urllib.parse.quote(args.dataset, safe="/")
    )
    dataset_info = request_url_json(dataset_info_url)
    revision = dataset_info.get("sha")
    if not isinstance(revision, str) or not revision:
        raise RuntimeError("Hugging Face dataset response is missing its revision")
    metadata_path = root / "timestamp_captions.json"
    receipt = {
        "format": "audiotime_subset_v1",
        "dataset": args.dataset,
        "dataset_revision": revision,
        "config": args.config,
        "split": args.split,
        "selection_seed": args.seed,
        "metadata_sha256": sha256_file(metadata_path),
        "source_count": len(sources),
        "sources": {
            source.source_id: {
                "label": source.label,
                "sha256": sha256_file(output_root / f"{source.source_id}.wav"),
            }
            for source in sources
        },
    }
    receipt_path = root / "subset_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"AudioTime subset ready: {output_root}")
    print(f"Acquisition receipt: {receipt_path}")


if __name__ == "__main__":
    main()
