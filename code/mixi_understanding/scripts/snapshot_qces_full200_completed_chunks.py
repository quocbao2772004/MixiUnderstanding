#!/usr/bin/env python3
"""Freeze an immutable manifest snapshot of completed full-200 clean chunks.

The primary full-200 materializer is intentionally append-only and can keep
running while this command executes.  A chunk is included only when its
terminal ``source_bank_receipt.json`` is internally consistent.  Audio is not
copied; manifest paths are made absolute so the snapshot can be consumed from
its own output directory without changing the live execution tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXECUTION_DIR = PROJECT_ROOT / "outputs/qces_full200_adaptive_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_full200_completed_snapshot_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, default=DEFAULT_EXECUTION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--expected-chunks",
        type=int,
        default=0,
        help="Fail closed unless exactly this many terminal chunks are visible.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            yield value


def atomic_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_audio_path(value: Any) -> str:
    path = Path(str(value or ""))
    if not str(path):
        raise ValueError("accepted source has no audio path")
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def normalize_row(row: Mapping[str, Any], *, chunk_id: str) -> dict[str, Any]:
    copied = dict(row)
    original_audio_path = str(
        copied.get("audio_path") or copied.get("stem_path") or copied.get("source_path") or ""
    )
    absolute_audio_path = resolve_audio_path(original_audio_path)
    copied["audio_path"] = absolute_audio_path
    copied["stem_path"] = absolute_audio_path
    copied["source_path"] = absolute_audio_path
    copied["snapshot_chunk_id"] = chunk_id
    copied["snapshot_original_audio_path"] = original_audio_path
    return copied


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    execution_dir = args.execution_dir.resolve()
    output_dir = args.output_dir.resolve()
    chunk_dirs = sorted(
        path.parent
        for path in (execution_dir / "primary_clean").glob(
            "*/*/source_bank_receipt.json"
        )
    )
    if args.expected_chunks and len(chunk_dirs) != args.expected_chunks:
        raise SystemExit(
            f"expected exactly {args.expected_chunks} completed chunks, found {len(chunk_dirs)}"
        )
    if not chunk_dirs:
        raise SystemExit(f"no completed chunks under {execution_dir / 'primary_clean'}")

    rows: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    seen_item_ids: set[str] = set()
    class_sources: dict[str, set[str]] = defaultdict(set)
    class_videos: dict[str, set[str]] = defaultdict(set)
    class_gold: Counter[str] = Counter()
    class_silver: Counter[str] = Counter()
    class_duration: Counter[str] = Counter()
    total_input = total_accepted = total_rejected = 0

    for chunk_dir in chunk_dirs:
        receipt_path = chunk_dir / "source_bank_receipt.json"
        manifest_path = chunk_dir / "source_bank.jsonl"
        receipt = read_json(receipt_path)
        input_items = int(receipt.get("input_items", -1))
        accepted = int(receipt.get("accepted_items", -1))
        rejected = int(receipt.get("rejected_items", -1))
        if min(input_items, accepted, rejected) < 0 or accepted + rejected != input_items:
            raise ValueError(f"invalid terminal counts: {receipt_path}")
        manifest_receipt = receipt.get("source_bank_manifest") or {}
        if int(manifest_receipt.get("rows", -1)) != accepted:
            raise ValueError(f"accepted row count mismatch: {receipt_path}")
        expected_hash = str(manifest_receipt.get("sha256") or "")
        actual_hash = sha256_file(manifest_path)
        if expected_hash != actual_hash:
            raise ValueError(f"source bank hash mismatch: {manifest_path}")

        chunk_id = chunk_dir.name
        chunk_rows = list(read_jsonl(manifest_path))
        if len(chunk_rows) != accepted:
            raise ValueError(f"manifest length mismatch: {manifest_path}")
        for raw in chunk_rows:
            row = normalize_row(raw, chunk_id=chunk_id)
            source_id = str(row.get("source_id") or "")
            item_id = str(row.get("item_id") or "")
            label = str(row.get("label") or row.get("coverage_label") or "")
            video_id = str(row.get("source_video_id") or row.get("video_id") or "")
            if not source_id or not item_id or not label or not video_id:
                raise ValueError(f"incomplete accepted row in {manifest_path}")
            if source_id in seen_source_ids or item_id in seen_item_ids:
                raise ValueError(f"duplicate accepted source/item: {source_id}/{item_id}")
            seen_source_ids.add(source_id)
            seen_item_ids.add(item_id)
            class_sources[label].add(source_id)
            class_videos[label].add(video_id)
            tier = str(row.get("acceptance_tier") or "").lower()
            if tier == "gold":
                class_gold[label] += 1
            elif tier == "silver":
                class_silver[label] += 1
            else:
                raise ValueError(f"unexpected acceptance tier {tier!r}")
            class_duration[label] += max(
                0.0,
                float(row.get("active_offset_seconds", 0.0))
                - float(row.get("active_onset_seconds", 0.0)),
            )
            rows.append(row)

        split = chunk_dir.parent.name
        chunks.append(
            {
                "chunk_id": chunk_id,
                "split": split,
                "input_items": input_items,
                "accepted_items": accepted,
                "rejected_items": rejected,
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "source_bank_path": str(manifest_path),
                "source_bank_sha256": actual_hash,
            }
        )
        total_input += input_items
        total_accepted += accepted
        total_rejected += rejected

    rows.sort(key=lambda row: (str(row["label"]), str(row["source_id"])))
    labels = sorted(class_sources)
    per_class = [
        {
            "label": label,
            "unique_sources": len(class_sources[label]),
            "unique_videos": len(class_videos[label]),
            "gold": class_gold[label],
            "silver": class_silver[label],
            "active_duration_seconds": class_duration[label],
        }
        for label in labels
    ]
    supports = [entry["unique_videos"] for entry in per_class]
    coverage = {
        "classes_present": len(labels),
        "classes_at_least_20_videos": sum(value >= 20 for value in supports),
        "classes_at_least_40_videos": sum(value >= 40 for value in supports),
        "classes_at_least_50_videos": sum(value >= 50 for value in supports),
        "classes_at_least_80_videos": sum(value >= 80 for value in supports),
        "classes_at_least_100_videos": sum(value >= 100 for value in supports),
        "video_support_min": min(supports, default=0),
        "video_support_p25": percentile(supports, 0.25),
        "video_support_median": percentile(supports, 0.50),
        "video_support_p75": percentile(supports, 0.75),
        "video_support_max": max(supports, default=0),
        "per_class": per_class,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "source_bank.jsonl"
    coverage_path = output_dir / "class_coverage.json"
    receipt_path = output_dir / "snapshot_receipt.json"
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    atomic_text(manifest_path, manifest_text, overwrite=args.overwrite)
    atomic_text(
        coverage_path,
        json.dumps(coverage, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )
    receipt = {
        "format": "qces_full200_completed_chunk_snapshot_v1",
        "complete_full_contract": False,
        "intended_use": "provisional_train_data_only",
        "execution_dir": str(execution_dir),
        "completed_chunks": len(chunks),
        "input_items": total_input,
        "accepted_items": total_accepted,
        "rejected_items": total_rejected,
        "acceptance_rate": total_accepted / max(total_input, 1),
        "source_bank": {
            "path": str(manifest_path),
            "rows": len(rows),
            "sha256": sha256_file(manifest_path),
        },
        "coverage": coverage,
        "chunks": chunks,
        "invariants": {
            "only_terminal_chunks_included": True,
            "terminal_receipts_validated": True,
            "source_bank_hashes_validated": True,
            "audio_files_exist": True,
            "audio_not_copied": True,
            "duplicate_source_ids": 0,
            "duplicate_item_ids": 0,
            "paper_eligible_full_contract": False,
        },
    }
    atomic_text(
        receipt_path,
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "completed_chunks": len(chunks),
                "accepted_items": total_accepted,
                "rejected_items": total_rejected,
                **{key: value for key, value in coverage.items() if key != "per_class"},
                "receipt": str(receipt_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
