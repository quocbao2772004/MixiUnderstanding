#!/usr/bin/env python3
"""Build the leakage-safe 191-class detector curriculum from data_full.

This sidecar leaves every historical builder untouched.  It keeps only classes
with at least 50 accepted train stems and 5 accepted official-eval stems,
creates hard-identity-disjoint single-event manifests, and optionally renders
the existing balanced 3--6 event Q-DOR scene curriculum.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline" / "PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(value))

from data_util.audioset_classes import as_strong_train_classes
from mixi_understanding.qces.clean_evidence_scenes import (
    CleanSource,
    build_clean_evidence_dataset,
    load_source_bank,
    partition_sources,
)


DEFAULT_CLASS_SUMMARY = PROJECT_ROOT / "data_full" / "index" / "class_summary.csv"
DEFAULT_EXECUTION = PROJECT_ROOT / "outputs" / "qces_full200_adaptive_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "qces_full191_r1_data_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-summary", type=Path, default=DEFAULT_CLASS_SUMMARY)
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument("--min-eval", type=int, default=5)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--render-multievent", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.min_train < 1 or args.min_eval < 1:
        parser.error("minimum support must be positive")
    if not 0 < args.dev_fraction < 1:
        parser.error("--dev-fraction must be in (0,1)")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
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
        temporary.unlink(missing_ok=True)


def jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def eligible_labels(path: Path, *, min_train: int, min_eval: int) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = [
        str(row["label"])
        for row in rows
        if not str(row["status"]).startswith("DROP")
        and int(row["train_accepted"]) >= min_train
        and int(row["eval_accepted"]) >= min_eval
    ]
    if len(labels) != len(set(labels)) or len(labels) < 3:
        raise ValueError("eligible ontology must contain at least three unique labels")
    return sorted(labels)


def aggregate_source_rows(
    execution_root: Path, labels: Sequence[str]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    keep = set(labels)
    rows: list[dict[str, Any]] = []
    display_by_label: dict[str, str] = {}
    seen_ids: set[str] = set()
    paths = sorted(
        Path(path)
        for path in glob.glob(
            str(execution_root / "primary_clean" / "*" / "*" / "source_bank.jsonl")
        )
    )
    if not paths:
        raise ValueError(f"no source banks under {execution_root}")
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                label = str(row.get("label") or "")
                if label not in keep or not row.get("accepted", False):
                    continue
                source_id = str(row.get("source_id") or row.get("item_id") or "")
                if not source_id or source_id in seen_ids:
                    raise ValueError(f"missing or duplicate source id: {source_id}")
                seen_ids.add(source_id)
                audio = Path(str(row["audio_path"]))
                if not audio.is_absolute():
                    audio = PROJECT_ROOT / audio
                if not audio.is_file():
                    raise FileNotFoundError(audio)
                row["audio_path"] = str(audio.resolve())
                row["source_id"] = source_id
                row["official_split"] = str(row.get("metadata_split") or row.get("hf_split"))
                row["cleanliness_tier"] = str(
                    row.get("acceptance_tier") or row.get("cleanliness_tier") or "unknown"
                )
                display = str(row.get("canonical_display_name") or "")
                previous = display_by_label.setdefault(label, display)
                if not display or previous != display:
                    raise ValueError(f"inconsistent canonical display name for {label}")
                rows.append(row)
    observed = {str(row["label"]) for row in rows}
    if observed != keep:
        raise ValueError(f"source/ontology mismatch: missing={sorted(keep-observed)}")
    return rows, display_by_label


def identities(sources: Sequence[CleanSource]) -> set[tuple[str, str]]:
    return {identity for source in sources for identity in source.hard_identities}


def single_scene(source: CleanSource, *, split: str, index: int) -> dict[str, Any]:
    duration = max(
        source.active_offset_seconds,
        float(source.provenance.get("duration_seconds") or source.active_offset_seconds),
    )
    tier = str(source.cleanliness_tier).lower()
    return {
        "format": "qces_full191_r1_single_event_scene_v1",
        "scene_id": f"r1_single_{split}_{index:07d}",
        "split": split,
        "mixture_path": source.audio_path,
        "duration_seconds": duration,
        "sample_rate": int(source.provenance.get("sample_rate") or 32_000),
        "quality_tier": tier,
        "events": [
            {
                "event_id": source.source_id,
                "label": source.label,
                "onset_seconds": source.active_onset_seconds,
                "offset_seconds": source.active_offset_seconds,
                "source_id": source.source_id,
                "source_video_id": source.video_id,
                "source_sha256": source.source_sha256,
                "cleanliness_tier": tier,
            }
        ],
    }


def absolute_multievent_rows(path: Path, root: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            mixture = Path(str(row["mixture_path"]))
            if not mixture.is_absolute():
                mixture = root / mixture
            if not mixture.is_file():
                raise FileNotFoundError(mixture)
            row["mixture_path"] = str(mixture.resolve())
            tiers = [str(event.get("cleanliness_tier") or "unknown").lower() for event in row["events"]]
            row["quality_tier"] = "gold" if tiers and all(tier == "gold" for tier in tiers) else "mixed"
            output.append(row)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = eligible_labels(
        args.class_summary.resolve(), min_train=args.min_train, min_eval=args.min_eval
    )
    raw_sources, display_by_label = aggregate_source_rows(args.execution_root.resolve(), labels)
    pretrained_index = {label: index for index, label in enumerate(as_strong_train_classes)}
    unmatched = [label for label in labels if display_by_label[label] not in pretrained_index]
    if unmatched:
        raise ValueError(f"labels do not exactly match pretrained strong head: {unmatched}")

    source_bank_path = output_dir / "source_bank_accepted.jsonl"
    ontology_path = output_dir / f"ontology_{len(labels)}.txt"
    label_map_path = output_dir / "pretrained_head_label_map.json"
    atomic_text(source_bank_path, jsonl(raw_sources))
    atomic_text(ontology_path, "".join(f"{label}\n" for label in labels))
    atomic_text(
        label_map_path,
        json.dumps(
            {
                label: {
                    "canonical_display_name": display_by_label[label],
                    "pretrained_head_index": pretrained_index[display_by_label[label]],
                }
                for label in labels
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )

    sources = load_source_bank(source_bank_path, require_audio_file=True)
    split_sources, partition_receipt = partition_sources(
        sources, labels, seed=args.seed, dev_fraction=args.dev_fraction
    )
    split_manifests: dict[str, Path] = {}
    split_counts: dict[str, Any] = {}
    for split in ("train", "dev", "test"):
        rows = [
            single_scene(source, split=split, index=index)
            for index, source in enumerate(split_sources[split])
        ]
        path = output_dir / f"detector_scene_manifest_single_{split}.jsonl"
        atomic_text(path, jsonl(rows))
        counts = Counter(event["label"] for row in rows for event in row["events"])
        split_manifests[split] = path
        split_counts[split] = {
            "scenes": len(rows),
            "support_min": min(counts.values()),
            "support_median": sorted(counts.values())[len(labels) // 2],
            "support_max": max(counts.values()),
            "gold": sum(row["quality_tier"] == "gold" for row in rows),
            "silver": sum(row["quality_tier"] == "silver" for row in rows),
        }

    pairwise_overlap = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        count = len(identities(split_sources[left]) & identities(split_sources[right]))
        pairwise_overlap[f"{left}_{right}"] = count
        if count:
            raise ValueError(f"hard identity leakage: {left}/{right}={count}")

    multi_receipt = None
    multi_manifests: dict[str, str] = {}
    if args.render_multievent:
        multi_root = output_dir / "multievent"
        multi_receipt = build_clean_evidence_dataset(
            sources,
            labels,
            output_dir=multi_root,
            seed=args.seed,
            dev_fraction=args.dev_fraction,
            core_rounds=None,
            repeat_rounds=1,
            add_distractors=True,
            max_event_seconds=1.20,
            render_audio=True,
            verify_source_hash=False,
            overwrite=args.overwrite,
            require_ontology_size=len(labels),
        )
        for split in ("train", "dev", "test"):
            rows = absolute_multievent_rows(
                multi_root / f"scene_manifest_{split}.jsonl", multi_root
            )
            path = output_dir / f"detector_scene_manifest_multi_{split}.jsonl"
            atomic_text(path, jsonl(rows))
            multi_manifests[split] = str(path)

    receipt = {
        "format": "qces_full191_r1_data_v1",
        "labels": len(labels),
        "eligibility": {
            "minimum_train_accepted": args.min_train,
            "minimum_eval_accepted": args.min_eval,
            "selection_uses_model_outcomes": False,
        },
        "accepted_sources": len(sources),
        "exact_pretrained_head_matches": len(labels),
        "source_bank": {"path": str(source_bank_path), "sha256": sha256_file(source_bank_path)},
        "ontology": {"path": str(ontology_path), "sha256": sha256_file(ontology_path)},
        "label_map": {"path": str(label_map_path), "sha256": sha256_file(label_map_path)},
        "partition": partition_receipt,
        "single_event": split_counts,
        "single_manifests": {split: str(path) for split, path in split_manifests.items()},
        "multi_manifests": multi_manifests,
        "multievent": multi_receipt,
        "hard_identity_overlap": pairwise_overlap,
        "passes": all(value == 0 for value in pairwise_overlap.values())
        and len(labels) == len(pretrained_index.keys() & set(display_by_label.values())),
    }
    # The exact match assertion above is reported directly as its own count;
    # the pass condition is made explicit to avoid relying on set type mixing.
    receipt["passes"] = all(value == 0 for value in pairwise_overlap.values()) and not unmatched
    receipt_path = output_dir / "data_receipt.json"
    atomic_text(receipt_path, json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "labels": len(labels),
                "accepted_sources": len(sources),
                "exact_pretrained_head_matches": len(labels),
                "single_event": split_counts,
                "multievent_scenes": None
                if multi_receipt is None
                else {split: multi_receipt["schedule"][split]["scenes"] for split in ("train", "dev", "test")},
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
