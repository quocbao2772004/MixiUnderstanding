#!/usr/bin/env python3
"""Build a no-copy single-event warm-start set from a completed-chunk snapshot.

This is deliberately a provisional detector pretraining artifact, not the
final multi-event QA benchmark.  Official AudioSet-train source identity groups
are split into train/dev using the same leakage-safe partitioner as the final
clean-evidence builder.  The remaining full-200 download may continue because
all inputs are bound by the immutable snapshot manifest and receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from mixi_understanding.qces.clean_evidence_scenes import (
    CleanSource,
    load_source_bank,
    partition_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT = PROJECT_ROOT / "outputs/qces_full200_snapshot_37chunks_20260811_v1"
DEFAULT_ONTOLOGY = (
    PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1/ontology_200_supported.txt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/qces_full200_single_event_pretrain_c37_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2037)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.dev_fraction < 1:
        parser.error("--dev-fraction must be in (0, 1)")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_ontology(path: Path) -> list[str]:
    labels = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(labels) != 200 or len(set(labels)) != 200:
        raise ValueError(f"ontology must contain exactly 200 unique labels: {path}")
    return labels


def scene_row(source: CleanSource, *, split: str, index: int) -> dict[str, Any]:
    provenance = source.provenance
    duration = max(
        source.active_offset_seconds,
        float(provenance.get("duration_seconds") or source.active_offset_seconds),
    )
    sample_rate = int(provenance.get("sample_rate") or 32_000)
    return {
        "format": "qces_full200_single_event_pretrain_scene_v1",
        "scene_id": f"full200_c37_{split}_{index:07d}",
        "split": split,
        "mixture_path": source.audio_path,
        "duration_seconds": duration,
        "sample_rate": sample_rate,
        "events": [
            {
                "event_id": source.source_id,
                "label": source.label,
                "onset_seconds": source.active_onset_seconds,
                "offset_seconds": source.active_offset_seconds,
                "source_id": source.source_id,
                "source_video_id": source.video_id,
                "source_sha256": source.source_sha256,
                "cleanliness_tier": source.cleanliness_tier,
            }
        ],
        "source_route": "full200_completed_chunk_snapshot_single_event",
    }


def identities(rows: Sequence[CleanSource]) -> set[tuple[str, str]]:
    return {identity for source in rows for identity in source.hard_identities}


def support(rows: Sequence[CleanSource], labels: Sequence[str]) -> dict[str, int]:
    counts = Counter(source.label for source in rows)
    return {label: counts[label] for label in labels}


def jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    snapshot_dir = args.snapshot_dir.resolve()
    snapshot_manifest = snapshot_dir / "source_bank.jsonl"
    snapshot_receipt_path = snapshot_dir / "snapshot_receipt.json"
    snapshot_receipt = json.loads(snapshot_receipt_path.read_text(encoding="utf-8"))
    expected_hash = str((snapshot_receipt.get("source_bank") or {}).get("sha256") or "")
    if sha256_file(snapshot_manifest) != expected_hash:
        raise ValueError("snapshot source-bank hash no longer matches its receipt")
    if snapshot_receipt.get("invariants", {}).get("only_terminal_chunks_included") is not True:
        raise ValueError("snapshot does not prove terminal chunks")

    ontology_path = args.ontology.resolve()
    labels = load_ontology(ontology_path)
    sources = load_source_bank(snapshot_manifest, require_audio_file=True)
    observed = {source.label for source in sources}
    if observed != set(labels):
        raise ValueError(
            f"snapshot/ontology mismatch: missing={sorted(set(labels)-observed)[:10]} "
            f"extra={sorted(observed-set(labels))[:10]}"
        )
    split_sources, partition_receipt = partition_sources(
        sources,
        labels,
        seed=args.seed,
        dev_fraction=args.dev_fraction,
    )
    train_sources = split_sources["train"]
    dev_sources = split_sources["dev"]
    if split_sources["test"]:
        raise ValueError("completed train-only snapshot unexpectedly contains eval sources")
    overlap = identities(train_sources) & identities(dev_sources)
    if overlap:
        raise ValueError(f"train/dev hard identity overlap: {sorted(overlap)[:5]}")

    train_rows = [
        scene_row(source, split="train", index=index)
        for index, source in enumerate(train_sources)
    ]
    dev_rows = [
        scene_row(source, split="dev", index=index)
        for index, source in enumerate(dev_sources)
    ]
    train_support = support(train_sources, labels)
    dev_support = support(dev_sources, labels)
    if min(train_support.values()) < 1 or min(dev_support.values()) < 1:
        raise ValueError("every ontology class must remain positive in train and dev")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "detector_scene_manifest_train.jsonl"
    dev_path = output_dir / "detector_scene_manifest_dev.jsonl"
    ontology_output = output_dir / "ontology_200.txt"
    receipt_path = output_dir / "pretrain_data_receipt.json"
    atomic_text(train_path, jsonl(train_rows), overwrite=args.overwrite)
    atomic_text(dev_path, jsonl(dev_rows), overwrite=args.overwrite)
    atomic_text(ontology_output, "\n".join(labels) + "\n", overwrite=args.overwrite)
    receipt = {
        "format": "qces_full200_single_event_pretrain_data_v1",
        "complete_full_contract": False,
        "paper_eligible_final_benchmark": False,
        "intended_use": "warm_start_200_class_detector_head_while_full_bank_downloads",
        "snapshot": {
            "path": str(snapshot_manifest),
            "sha256": expected_hash,
            "completed_chunks": int(snapshot_receipt["completed_chunks"]),
            "accepted_items": int(snapshot_receipt["accepted_items"]),
        },
        "ontology": {
            "path": str(ontology_path),
            "sha256": sha256_file(ontology_path),
            "labels": len(labels),
        },
        "partition": partition_receipt,
        "train": {
            "path": str(train_path),
            "sha256": sha256_file(train_path),
            "scenes": len(train_rows),
            "positive_classes": sum(value > 0 for value in train_support.values()),
            "support_min": min(train_support.values()),
            "support_median": sorted(train_support.values())[len(labels) // 2],
            "support_max": max(train_support.values()),
            "per_class_support": train_support,
        },
        "dev": {
            "path": str(dev_path),
            "sha256": sha256_file(dev_path),
            "scenes": len(dev_rows),
            "positive_classes": sum(value > 0 for value in dev_support.values()),
            "support_min": min(dev_support.values()),
            "support_median": sorted(dev_support.values())[len(labels) // 2],
            "support_max": max(dev_support.values()),
            "per_class_support": dev_support,
        },
        "invariants": {
            "audio_copied": False,
            "source_rows_reused_across_splits": 0,
            "hard_identity_overlap": 0,
            "train_positive_classes": 200,
            "dev_positive_classes": 200,
            "official_eval_used": False,
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
                "train_scenes": len(train_rows),
                "dev_scenes": len(dev_rows),
                "labels": len(labels),
                "train_support_min": min(train_support.values()),
                "dev_support_min": min(dev_support.values()),
                "hard_identity_overlap": len(overlap),
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
