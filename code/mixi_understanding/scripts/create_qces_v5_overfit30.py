#!/usr/bin/env python3
"""Create a 30-record, counterfactual-complete QCES-v5 micro-overfit view.

This artifact is an anti-collapse engineering test only. It selects ten
question-index groups from one train family and retains all three
base/order-swap/anchor-drop variants for every selected question.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "qces_v5_source_quality_overfit30_v1"
VARIANTS = {"base", "order_swap", "anchor_drop"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-bank", type=Path, required=True)
    parser.add_argument("--listening-queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank row at {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"non-object at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.building-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.building-", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def all_same_status(group: Sequence[Mapping[str, Any]], no_evidence: bool) -> bool:
    return all(bool(row.get("no_evidence")) is no_evidence for row in group)


def first_control_pairs(
    groups: Mapping[int, Sequence[Mapping[str, Any]]], *, no_evidence: bool
) -> list[tuple[int, int]]:
    candidates = [
        index
        for index, group in groups.items()
        if group[0].get("relation") == "first"
        and all_same_status(group, no_evidence)
    ]
    pairs = []
    for left in sorted(candidates):
        for right in sorted(candidates):
            if right <= left:
                continue
            left_by_variant = {str(row["variant_id"]): row for row in groups[left]}
            right_by_variant = {str(row["variant_id"]): row for row in groups[right]}
            if all(
                left_by_variant[variant].get("surface_control_group_id")
                == right_by_variant[variant].get("surface_control_group_id")
                and {
                    left_by_variant[variant].get("mention_order_variant"),
                    right_by_variant[variant].get("mention_order_variant"),
                }
                == {"forward", "reversed"}
                for variant in VARIANTS
            ):
                pairs.append((left, right))
    return pairs


def select_question_indices(
    groups: Mapping[int, Sequence[Mapping[str, Any]]]
) -> list[int]:
    primary_after = [
        index
        for index, group in groups.items()
        if group[0].get("relation") == "after"
        and all(bool(row.get("primary_counterfactual_probe")) for row in group)
    ]
    if len(primary_after) != 1:
        raise ValueError("expected exactly one complete primary after group")
    after_answerable = [
        index
        for index, group in sorted(groups.items())
        if group[0].get("relation") == "after"
        and all_same_status(group, False)
        and index not in primary_after
    ]
    before_answerable = [
        index
        for index, group in sorted(groups.items())
        if group[0].get("relation") == "before"
        and all_same_status(group, False)
    ]
    before_absent = [
        index
        for index, group in sorted(groups.items())
        if group[0].get("relation") == "before"
        and all_same_status(group, True)
    ]
    answerable_first_pairs = first_control_pairs(groups, no_evidence=False)
    absent_first_pairs = first_control_pairs(groups, no_evidence=True)
    if (
        len(after_answerable) < 2
        or len(before_answerable) < 2
        or not before_absent
        or not answerable_first_pairs
        or not absent_first_pairs
    ):
        raise ValueError("family cannot satisfy the registered overfit30 design")
    selected = [
        primary_after[0],
        *after_answerable[:2],
        *before_answerable[:2],
        before_absent[0],
        *answerable_first_pairs[0],
        *absent_first_pairs[0],
    ]
    if len(selected) != 10 or len(set(selected)) != 10:
        raise AssertionError("overfit30 question-index selection is not ten unique groups")
    return sorted(selected)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    receipt_path = args.receipt.expanduser().resolve()
    if output.parent != manifest.parent:
        raise ValueError(
            "overfit manifest must stay beside the source manifest so relative "
            "audio paths remain valid"
        )
    rows = read_jsonl(manifest)
    if len({str(row.get("scene_family_id")) for row in rows}) != 1:
        raise ValueError("overfit30 source manifest must contain exactly one family")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("overfit30 source manifest must be train-only")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        index = row.get("question_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("question_index must be an integer")
        groups[index].append(row)
    if len(groups) != 16:
        raise ValueError("source family must expose 16 question-index groups")
    for index, group in groups.items():
        if len(group) != 3 or {str(row["variant_id"]) for row in group} != VARIANTS:
            raise ValueError(f"question index {index} lacks a complete variant triplet")
    selected_indices = select_question_indices(groups)
    selected = [row for row in rows if int(row["question_index"]) in selected_indices]
    selected.sort(
        key=lambda row: (
            ("base", "order_swap", "anchor_drop").index(str(row["variant_id"])),
            int(row["question_index"]),
        )
    )
    if len(selected) != 30 or len({str(row["id"]) for row in selected}) != 30:
        raise AssertionError("overfit30 selection is not 30 unique records")

    crop_bank_path = args.crop_bank.expanduser().resolve()
    crop_bank = json.loads(crop_bank_path.read_text(encoding="utf-8"))
    if crop_bank.get("format") != "qces_v5_semantic_crop_bank_v2":
        raise ValueError("overfit30 requires the v2 semantic crop bank")
    bank = {
        (str(entry["source_id"]), str(entry["label"])): entry
        for entry in crop_bank["entries"]
    }
    weak_queue_path = args.listening_queue.expanduser().resolve()
    weak = {
        (str(row["source_id"]), str(row["label"]))
        for row in read_jsonl(weak_queue_path)
    }
    source_keys = set()
    for row in selected:
        for event in row.get("events", []):
            if event.get("event_kind") == "semantic":
                source_keys.add((str(event["source_id"]), str(event["label"])))
    if missing := source_keys - set(bank):
        raise ValueError(f"selected semantic sources are missing from crop bank: {missing}")
    if rejected := source_keys & weak:
        raise ValueError(f"selected family contains weakest-queue sources: {rejected}")
    source_scores = [
        float(bank[key]["selected"]["label_probability ↑"]) for key in source_keys
    ]

    relation_counts = Counter(str(row["relation"]) for row in selected)
    variant_counts = Counter(str(row["variant_id"]) for row in selected)
    no_evidence_count = sum(bool(row["no_evidence"]) for row in selected)
    if relation_counts != {"after": 9, "before": 9, "first": 12}:
        raise AssertionError(f"unexpected relation balance: {relation_counts}")
    if any(variant_counts[variant] != 10 for variant in VARIANTS):
        raise AssertionError(f"unexpected variant balance: {variant_counts}")
    if no_evidence_count != 10:
        raise AssertionError("overfit30 requires exactly ten no-evidence records")

    atomic_jsonl(output, selected, args.overwrite)
    receipt = {
        "format": FORMAT,
        "purpose": "micro_overfit_anti_collapse_only_not_heldout_evidence",
        "source_manifest": identity(manifest),
        "output_manifest": identity(output),
        "crop_bank": identity(crop_bank_path),
        "weakest_source_queue": identity(weak_queue_path),
        "selection": {
            "question_indices": selected_indices,
            "complete_counterfactual_triplets ↑": 10,
            "records ↑": len(selected),
            "records_by_variant": dict(sorted(variant_counts.items())),
            "records_by_relation": dict(sorted(relation_counts.items())),
            "answerable_records ↑": len(selected) - no_evidence_count,
            "no_evidence_records ↑": no_evidence_count,
            "complete_first_surface_pairs ↑": 2,
            "primary_counterfactual_triplets ↑": 1,
        },
        "source_quality": {
            "unique_semantic_sources ↑": len(source_keys),
            "weakest_queue_overlap ↓": 0,
            "minimum_crop_bank_probability ↑": min(source_scores),
            "mean_crop_bank_probability ↑": sum(source_scores) / len(source_scores),
            "curator_boundary": (
                "BEATs selected/ranked these crops; source quality is an "
                "engineering filter, not independent evaluation."
            ),
        },
        "authorization": {
            "training_scope": "same-record memorization only",
            "paper_metric_allowed": False,
            "test_or_validation_claim_allowed": False,
            "full_dataset_readiness_implied": False,
        },
    }
    atomic_json(receipt_path, receipt, args.overwrite)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
