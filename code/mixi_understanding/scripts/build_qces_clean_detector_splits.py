#!/usr/bin/env python3
"""Build leakage-safe train/dev/test manifests for new QCES experiments.

Historical manifests are read-only inputs.  The command writes a separate
protocol directory and refuses to publish it if any exact audio/source identity
crosses splits or if an ontology label loses all positive training scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mixi_understanding.qces.clean_detector_splits import (
    CANONICAL_SPLITS,
    PROTOCOL_NAME,
    build_clean_detector_splits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_AUDIOSET_ROOT = PROJECT_ROOT / "outputs/qces_audioset_strong_subset_qces200_q40_10"
DEFAULT_PRESERVED_ROOT = PROJECT_ROOT / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1"
DEFAULT_ONTOLOGY = (
    PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/ontology_200_trainable.txt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_clean_detector_protocol_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audioset-train",
        type=Path,
        default=DEFAULT_AUDIOSET_ROOT / "audioset_strong_detector_manifest_train.partial.jsonl",
    )
    parser.add_argument(
        "--audioset-test",
        type=Path,
        default=DEFAULT_AUDIOSET_ROOT / "audioset_strong_detector_manifest_test.partial.jsonl",
    )
    parser.add_argument(
        "--preserved-train",
        type=Path,
        nargs="*",
        default=None,
        help="Official-train manifests to retain in train (default: FUSS/FSD50K train).",
    )
    parser.add_argument(
        "--preserved-dev",
        type=Path,
        nargs="*",
        default=None,
        help="Official-validation manifests to retain in dev (default: FUSS/FSD50K val).",
    )
    parser.add_argument(
        "--preserved-test",
        type=Path,
        nargs="*",
        default=None,
        help="Official-eval manifests to retain in test (default: FUSS/FSD50K test).",
    )
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--no-ontology-filter", action="store_true")
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _default_preserved(args: argparse.Namespace) -> dict[str, list[Path]]:
    return {
        "train": (
            args.preserved_train
            if args.preserved_train is not None
            else [DEFAULT_PRESERVED_ROOT / "detector_source_manifest_train_clean.jsonl"]
        ),
        "dev": (
            args.preserved_dev
            if args.preserved_dev is not None
            else [DEFAULT_PRESERVED_ROOT / "detector_source_manifest_val_clean.jsonl"]
        ),
        "test": (
            args.preserved_test
            if args.preserved_test is not None
            else [DEFAULT_PRESERVED_ROOT / "detector_source_manifest_test_clean.jsonl"]
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _read_many(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(path.resolve()))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str, *, overwrite: bool) -> None:
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


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def _markdown(receipt: Mapping[str, Any]) -> str:
    lines = [
        "# QCES clean detector split receipt",
        "",
        f"Status: **{'PASS' if receipt['passes'] else 'FAIL'}**",
        "",
        "| split | rows | AudioSet | preserved official | positive labels |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in CANONICAL_SPLITS:
        summary = receipt["split_summary"][split]
        lines.append(
            f"| {split} | {summary['rows']} | "
            f"{summary['sources'].get('audioset_strong', 0)} | "
            f"{summary['sources'].get('preserved_official', 0)} | "
            f"{summary['positive_labels']} |"
        )
    invariants = receipt["invariants"]
    stratification = receipt["stratification"]
    lines.extend(
        [
            "",
            "## Invariants",
            "",
            f"- Cross-split exact identity overlaps: **{invariants['cross_split_identity_overlaps']}**",
            f"- AudioSet official-test rows outside test: **{invariants['official_test_rows_outside_test']}**",
            f"- Preserved-source assignment violations: **{invariants['preserved_assignment_violations']}**",
            f"- Ontology labels positive in train: **{invariants['ontology_train_positive_labels']}/{receipt['ontology_labels']}**",
            "",
            "## AudioSet train/dev stratification",
            "",
            f"- Official-train video groups: {stratification['official_train_groups']}",
            f"- Train/dev groups: {stratification['actual_train_groups']}/{stratification['actual_dev_groups']}",
            f"- Labels retained in train: {stratification['labels_retained_in_train']}/{stratification['labels']}",
            f"- Labels represented in dev: {stratification['labels_present_in_dev']}/{stratification['labels']}",
            f"- Mean normalized label-target error: {stratification['mean_normalized_label_target_error']:.6f}",
            "",
            "The official AudioSet test set is never used for model selection.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preserved_paths = {
        split: [path.resolve() for path in paths]
        for split, paths in _default_preserved(args).items()
    }
    audioset_train_path = args.audioset_train.resolve()
    audioset_test_path = args.audioset_test.resolve()
    all_input_paths = [
        audioset_train_path,
        audioset_test_path,
        *(path for split in CANONICAL_SPLITS for path in preserved_paths[split]),
    ]
    missing = [path for path in all_input_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input manifests: {[str(path) for path in missing]}")

    ontology_path = args.ontology.resolve()
    if args.no_ontology_filter:
        ontology: list[str] = []
    else:
        if not ontology_path.is_file():
            raise SystemExit(f"missing ontology: {ontology_path}")
        ontology = list(
            dict.fromkeys(
                line.strip()
                for line in ontology_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        )
        if not ontology:
            raise SystemExit(f"empty ontology: {ontology_path}")

    result = build_clean_detector_splits(
        audioset_train_rows=_read_jsonl(audioset_train_path),
        audioset_test_rows=_read_jsonl(audioset_test_path),
        preserved_rows={
            split: _read_many(preserved_paths[split]) for split in CANONICAL_SPLITS
        },
        ontology=ontology,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
    )

    output_dir = args.output_dir.resolve()
    output_paths = {
        split: output_dir / f"detector_manifest_{split}.jsonl"
        for split in CANONICAL_SPLITS
    }
    receipt_path = output_dir / "clean_split_receipt.json"
    markdown_path = output_dir / "clean_split_receipt.md"
    ontology_output = output_dir / "ontology.txt"
    planned = [*output_paths.values(), receipt_path, markdown_path]
    if ontology:
        planned.append(ontology_output)
    existing = [path for path in planned if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"outputs already exist: {[str(path) for path in existing]}; use --overwrite")

    for split, path in output_paths.items():
        _atomic_text(path, _jsonl_text(result.splits[split]), overwrite=args.overwrite)
    if ontology:
        _atomic_text(ontology_output, "\n".join(ontology) + "\n", overwrite=args.overwrite)

    receipt = dict(result.receipt)
    receipt["inputs"] = [
        {
            "path": str(path),
            "rows": sum(1 for line in path.open("r", encoding="utf-8") if line.strip()),
            "sha256": _sha256(path),
        }
        for path in all_input_paths
    ]
    receipt["ontology"] = (
        {"path": str(ontology_path), "sha256": _sha256(ontology_path)}
        if ontology
        else None
    )
    receipt["outputs"] = {
        split: {
            "path": str(path),
            "rows": len(result.splits[split]),
            "sha256": _sha256(path),
        }
        for split, path in output_paths.items()
    }
    _atomic_text(
        receipt_path,
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )
    _atomic_text(markdown_path, _markdown(receipt), overwrite=args.overwrite)

    print(
        json.dumps(
            {
                "format": PROTOCOL_NAME,
                "passes": receipt["passes"],
                "rows": {
                    split: len(result.splits[split]) for split in CANONICAL_SPLITS
                },
                "cross_split_identity_overlaps": receipt["invariants"][
                    "cross_split_identity_overlaps"
                ],
                "official_test_rows_outside_test": receipt["invariants"][
                    "official_test_rows_outside_test"
                ],
                "ontology_train_positive_labels": receipt["invariants"][
                    "ontology_train_positive_labels"
                ],
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
