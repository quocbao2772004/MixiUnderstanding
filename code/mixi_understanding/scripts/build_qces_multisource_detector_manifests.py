#!/usr/bin/env python3
"""Merge partial AudioSet-Strong + FUSS/FSD50K manifests for detector training.

The AudioSet-Strong stream can timeout before every planned label reaches quota.
This builder makes the training set robust: it keeps the requested 200-class
ontology when possible, replaces currently missing-positive labels with
FUSS/FSD50K labels that do have positives, and writes clean detector manifests.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "outputs/qces_multisource_200_ontology_v1/ontology_200_multisource.txt"
DEFAULT_AS_ROOT = PROJECT_ROOT / "outputs/qces_audioset_strong_subset_qces200_q40_10"
DEFAULT_FUSS_ROOT = PROJECT_ROOT / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--audioset-root", type=Path, default=DEFAULT_AS_ROOT)
    parser.add_argument("--fuss-root", type=Path, default=DEFAULT_FUSS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-labels", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.target_labels <= 0:
        parser.error("--target-labels must be positive")
    return args


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_lines(path: Path, lines: Sequence[str], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_ontology(path: Path) -> list[str]:
    labels: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        label = raw.strip()
        if label and label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError(f"empty ontology: {path}")
    return labels


def existing_manifest(path: Path) -> Path | None:
    return path if path.exists() else None


def load_rows(paths: Sequence[Path | None], *, route: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path is None:
            continue
        for row in read_jsonl(path):
            copied = dict(row)
            copied.setdefault("source_route", route)
            copied["manifest_source_path"] = str(path)
            rows.append(copied)
    return rows


def row_event_labels(row: Mapping[str, Any]) -> set[str]:
    return {
        str(event.get("label"))
        for event in row.get("events", [])
        if event.get("label") is not None
    }


def label_counts(rows: Sequence[Mapping[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for label in row_event_labels(row):
            counts[label] += 1
    return counts


def select_trainable_ontology(
    base: Sequence[str],
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    *,
    target_labels: int,
) -> tuple[list[str], list[str], list[str]]:
    train_counts = label_counts(train_rows)
    val_counts = label_counts(val_rows)
    selected: list[str] = []
    removed: list[str] = []
    for label in base:
        if train_counts[label] > 0:
            selected.append(label)
        else:
            removed.append(label)
        if len(selected) == target_labels:
            return selected, removed, []

    candidates = [
        label
        for label in sorted(train_counts)
        if label not in selected and train_counts[label] > 0
    ]
    candidates.sort(
        key=lambda label: (
            0 if val_counts[label] > 0 else 1,
            -train_counts[label],
            -val_counts[label],
            label,
        )
    )
    added: list[str] = []
    for label in candidates:
        selected.append(label)
        added.append(label)
        if len(selected) == target_labels:
            break
    if len(selected) != target_labels:
        raise RuntimeError(
            f"only {len(selected)} train-positive labels available; target={target_labels}"
        )
    return selected, removed, added


def filtered_rows(rows: Sequence[Mapping[str, Any]], ontology: set[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        events = [
            dict(event)
            for event in row.get("events", [])
            if str(event.get("label")) in ontology
        ]
        if not events:
            continue
        labels = sorted({str(event["label"]) for event in events})
        copied = dict(row)
        copied["events"] = events
        copied["labels"] = labels
        scene_id = str(copied.get("scene_id") or copied.get("video_id") or len(output))
        if scene_id in seen_ids:
            route = str(copied.get("source_route") or "unknown")
            scene_id = f"{scene_id}_{route}_{len(output):06d}"
            copied["scene_id"] = scene_id
        seen_ids.add(scene_id)
        output.append(copied)
    output.sort(key=lambda item: str(item.get("scene_id")))
    return output


def summarize(rows: Sequence[Mapping[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    counts = label_counts(rows)
    labels_with_positive = [label for label in labels if counts[label] > 0]
    route_counts = Counter(str(row.get("source_route") or "unknown") for row in rows)
    return {
        "rows": len(rows),
        "events": sum(len(row.get("events", [])) for row in rows),
        "labels_with_positive": len(labels_with_positive),
        "missing_labels": [label for label in labels if counts[label] == 0],
        "route_counts": dict(sorted(route_counts.items())),
        "label_count_min": min((counts[label] for label in labels if counts[label] > 0), default=0),
        "label_count_median": sorted([counts[label] for label in labels if counts[label] > 0])[len(labels_with_positive)//2] if labels_with_positive else 0,
        "label_count_max": max((counts[label] for label in labels), default=0),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    base_ontology = load_ontology(args.base_ontology.resolve())
    as_root = args.audioset_root.resolve()
    fuss_root = args.fuss_root.resolve()
    as_train = existing_manifest(as_root / "audioset_strong_detector_manifest_train.jsonl") or existing_manifest(as_root / "audioset_strong_detector_manifest_train.partial.jsonl")
    as_test = existing_manifest(as_root / "audioset_strong_detector_manifest_test.jsonl") or existing_manifest(as_root / "audioset_strong_detector_manifest_test.partial.jsonl")
    train_rows = load_rows(
        [
            as_train,
            fuss_root / "detector_source_manifest_train_clean.jsonl",
        ],
        route="multisource_train",
    )
    val_rows = load_rows(
        [
            as_test,
            fuss_root / "detector_source_manifest_val_clean.jsonl",
        ],
        route="multisource_val",
    )
    test_rows = load_rows(
        [
            as_test,
            fuss_root / "detector_source_manifest_test_clean.jsonl",
        ],
        route="multisource_test",
    )
    ontology, removed, added = select_trainable_ontology(
        base_ontology,
        train_rows,
        val_rows,
        target_labels=args.target_labels,
    )
    ontology_set = set(ontology)
    train_filtered = filtered_rows(train_rows, ontology_set)
    val_filtered = filtered_rows(val_rows, ontology_set)
    test_filtered = filtered_rows(test_rows, ontology_set)

    write_lines(output_dir / "ontology_200_trainable.txt", ontology, overwrite=args.overwrite)
    write_jsonl(output_dir / "detector_manifest_train.jsonl", train_filtered, overwrite=args.overwrite)
    write_jsonl(output_dir / "detector_manifest_val.jsonl", val_filtered, overwrite=args.overwrite)
    write_jsonl(output_dir / "detector_manifest_test.jsonl", test_filtered, overwrite=args.overwrite)
    summary = {
        "format": "qces_multisource_detector_trainset_v1",
        "base_ontology": str(args.base_ontology.resolve()),
        "target_labels": args.target_labels,
        "final_labels": len(ontology),
        "audioset_train_manifest": str(as_train) if as_train else None,
        "audioset_test_manifest": str(as_test) if as_test else None,
        "removed_missing_train_positive_labels": removed,
        "added_replacement_labels": added,
        "train": summarize(train_filtered, ontology),
        "val": summarize(val_filtered, ontology),
        "test": summarize(test_filtered, ontology),
        "paths": {
            "ontology": str((output_dir / "ontology_200_trainable.txt").resolve()),
            "train_manifest": str((output_dir / "detector_manifest_train.jsonl").resolve()),
            "val_manifest": str((output_dir / "detector_manifest_val.jsonl").resolve()),
            "test_manifest": str((output_dir / "detector_manifest_test.jsonl").resolve()),
        },
    }
    write_json(output_dir / "multisource_detector_trainset_summary.json", summary, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
