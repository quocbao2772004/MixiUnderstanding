#!/usr/bin/env python3
"""Create deterministic, family-complete QCES-v5 train/validation pilot views.

The pilot is an engineering/model-selection accelerator, never a paper result
split.  Families are selected using only their identifiers and a declared seed;
labels, answers, waveform targets, and model metrics do not influence selection.
Every selected family is copied whole so surface controls and the
base/order-swap/anchor-drop counterfactual groups remain valid.
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


FORMAT = "qces_v5_identifier_only_dev_pilot_v1"
EXPECTED_VARIANTS = frozenset({"base", "order_swap", "anchor_drop"})


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-val", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--train-families", type=int, default=12)
    parser.add_argument("--val-families", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _required_text(
    row: Mapping[str, Any], key: str, *, line_number: int, manifest: Path
) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{manifest}: line {line_number}: {key} must be non-empty text"
        )
    return value


def read_manifest(path: Path, expected_split: str) -> list[dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    rows: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{resolved}: line {line_number}: blank row")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"{resolved}: line {line_number}: expected JSON object"
                )
            sample_id = _required_text(
                row, "id", line_number=line_number, manifest=resolved
            )
            if sample_id in sample_ids:
                raise ValueError(f"{resolved}: duplicate sample ID {sample_id}")
            sample_ids.add(sample_id)
            split = _required_text(
                row, "split", line_number=line_number, manifest=resolved
            )
            if split != expected_split:
                raise ValueError(
                    f"{resolved}: {sample_id} has split={split}, expected "
                    f"{expected_split}"
                )
            _required_text(
                row, "scene_family_id", line_number=line_number, manifest=resolved
            )
            _required_text(row, "scene_id", line_number=line_number, manifest=resolved)
            _required_text(
                row, "variant_id", line_number=line_number, manifest=resolved
            )
            _required_text(row, "relation", line_number=line_number, manifest=resolved)
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {resolved}")
    return rows


def group_complete_families(
    rows: Iterable[Mapping[str, Any]], *, manifest: Path
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["scene_family_id"])].append(row)
    for family_id, family_rows in sorted(grouped.items()):
        variants = {str(row["variant_id"]) for row in family_rows}
        scenes = {str(row["scene_id"]) for row in family_rows}
        if variants != EXPECTED_VARIANTS:
            raise ValueError(
                f"{manifest}: family {family_id} variants {sorted(variants)} "
                f"!= {sorted(EXPECTED_VARIANTS)}"
            )
        if len(scenes) != len(EXPECTED_VARIANTS):
            raise ValueError(
                f"{manifest}: family {family_id} must contain exactly three scenes"
            )
        counts = Counter(str(row["variant_id"]) for row in family_rows)
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"{manifest}: family {family_id} has unequal variant row counts"
            )
        if next(iter(counts.values())) < 1:
            raise ValueError(f"{manifest}: family {family_id} has no questions")
    return dict(grouped)


def identifier_rank(seed: int, split: str, family_id: str) -> str:
    payload = f"qces-v5-dev-pilot:{seed}:{split}:{family_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_families(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    split: str,
    count: int,
    seed: int,
) -> list[str]:
    if count <= 0:
        raise ValueError(f"{split} family count must be positive")
    if count > len(grouped):
        raise ValueError(
            f"requested {count} {split} families, but only {len(grouped)} exist"
        )
    ranked = sorted(
        grouped,
        key=lambda family_id: (identifier_rank(seed, split, family_id), family_id),
    )
    return ranked[:count]


def subset_rows(
    rows: Sequence[Mapping[str, Any]], selected_family_ids: Iterable[str]
) -> list[Mapping[str, Any]]:
    selected = set(selected_family_ids)
    result = [row for row in rows if str(row["scene_family_id"]) in selected]
    if {str(row["scene_family_id"]) for row in result} != selected:
        raise RuntimeError("selected family set changed while producing pilot")
    return result


def atomic_write_jsonl(
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool,
) -> None:
    resolved = output.resolve()
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"output exists: {resolved}; use --overwrite")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"output is not a regular file: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.building-", dir=resolved.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)


def split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "families": len({str(row["scene_family_id"]) for row in rows}),
        "scenes": len({str(row["scene_id"]) for row in rows}),
        "by_variant": dict(
            sorted(Counter(str(row["variant_id"]) for row in rows).items())
        ),
        "by_relation": dict(
            sorted(Counter(str(row["relation"]) for row in rows).items())
        ),
        "by_no_evidence": {
            str(key).lower(): value
            for key, value in sorted(
                Counter(bool(row.get("no_evidence")) for row in rows).items()
            )
        },
    }


def atomic_write_json(
    output: Path, payload: Mapping[str, Any], *, overwrite: bool
) -> None:
    resolved = output.resolve()
    if resolved.exists() and not overwrite:
        raise FileExistsError(f"output exists: {resolved}; use --overwrite")
    if resolved.exists() and not resolved.is_file():
        raise ValueError(f"output is not a regular file: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.building-", dir=resolved.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(resolved)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_same_dataset_parent(source: Path, output: Path) -> None:
    if source.resolve().parent != output.resolve().parent:
        raise ValueError(
            "pilot manifest must stay beside its source manifest so unchanged "
            "dataset-relative audio paths remain valid"
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    train_manifest = args.train_manifest.resolve()
    val_manifest = args.val_manifest.resolve()
    output_train = args.output_train.resolve()
    output_val = args.output_val.resolve()
    receipt_path = args.receipt.resolve()
    ensure_same_dataset_parent(train_manifest, output_train)
    ensure_same_dataset_parent(val_manifest, output_val)
    if len({output_train, output_val, receipt_path}) != 3:
        raise ValueError("train, validation, and receipt outputs must be distinct")
    for output in (output_train, output_val, receipt_path):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists: {output}; use --overwrite")

    train_rows = read_manifest(train_manifest, "train")
    val_rows = read_manifest(val_manifest, "val")
    train_grouped = group_complete_families(train_rows, manifest=train_manifest)
    val_grouped = group_complete_families(val_rows, manifest=val_manifest)
    selected_train = select_families(
        train_grouped,
        split="train",
        count=args.train_families,
        seed=args.seed,
    )
    selected_val = select_families(
        val_grouped,
        split="val",
        count=args.val_families,
        seed=args.seed,
    )
    train_pilot = subset_rows(train_rows, selected_train)
    val_pilot = subset_rows(val_rows, selected_val)

    train_sources = {
        str(source_id)
        for row in train_pilot
        for source_id in row.get("source_group_ids", [])
    }
    val_sources = {
        str(source_id)
        for row in val_pilot
        for source_id in row.get("source_group_ids", [])
    }
    source_overlap = sorted(train_sources & val_sources)
    family_overlap = sorted(set(selected_train) & set(selected_val))
    if source_overlap or family_overlap:
        raise ValueError("pilot train/validation isolation check failed")

    atomic_write_jsonl(output_train, train_pilot, overwrite=args.overwrite)
    atomic_write_jsonl(output_val, val_pilot, overwrite=args.overwrite)
    receipt = {
        "format": FORMAT,
        "purpose": "development-only throughput and validation pilot",
        "paper_result_eligible": False,
        "test_split_accessed": False,
        "selection": {
            "method": "sha256(seed, split, scene_family_id); lowest ranks",
            "uses_identifiers_only": True,
            "uses_labels_answers_targets_or_metrics": False,
            "seed": args.seed,
            "train_family_count": args.train_families,
            "val_family_count": args.val_families,
            "selected_train_family_ids": selected_train,
            "selected_val_family_ids": selected_val,
        },
        "inputs": {
            "train_manifest": file_identity(train_manifest),
            "val_manifest": file_identity(val_manifest),
        },
        "outputs": {
            "train_manifest": file_identity(output_train),
            "val_manifest": file_identity(output_val),
        },
        "counts": {
            "train": split_summary(train_pilot),
            "val": split_summary(val_pilot),
        },
        "isolation": {
            "train_val_family_overlap_count ↓": len(family_overlap),
            "train_val_source_group_overlap_count ↓": len(source_overlap),
        },
        "group_completeness": {
            "selected_families_complete ↑": True,
            "required_variants": sorted(EXPECTED_VARIANTS),
            "variant_counts_equal_within_each_family ↑": True,
        },
    }
    atomic_write_json(receipt_path, receipt, overwrite=args.overwrite)
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
