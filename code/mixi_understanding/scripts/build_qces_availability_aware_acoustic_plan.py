#!/usr/bin/env python3
"""Reselect the 200-class crop contract from pinned, available audio only."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mixi_understanding.qces.audioset_availability_index import (
    atomic_bytes,
    atomic_json,
    atomic_jsonl,
)
from mixi_understanding.qces.availability_aware_acoustic_plan import (
    AvailabilityAwarePlanError,
    build_availability_aware_plan,
)
from mixi_understanding.qces.supported_ontology import load_strong_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_DIR = (
    PROJECT_ROOT / "code/baseline/PretrainedSED/hf_dataset_gen/metadata"
)
DEFAULT_ONTOLOGY = (
    PROJECT_ROOT
    / "outputs/qces_supported_ontology_200_v1/ontology_200_supported.tsv"
)
DEFAULT_CLEANLINESS_DIR = PROJECT_ROOT / "outputs/qces_acoustic_cleanliness_200_v1"
DEFAULT_AVAILABILITY_INDEX = (
    PROJECT_ROOT
    / "outputs/qces_audioset_official_strong_availability_v1/audioset_availability_index.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/qces_availability_aware_acoustic_plan_v1"
)
DEFAULT_ROUTE_PRIORITY = (
    "enyoukai_audioset_strong_pinned_parquet_v1",
    "agkphysics_audioset_balanced_train_pinned_v1",
    "agkphysics_audioset_unbalanced_train_pinned_v1",
    "agkphysics_audioset_eval_pinned_v1",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--availability-index",
        type=Path,
        action="append",
        default=None,
        help="Repeatable availability index JSONL.",
    )
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--selected-ontology-tsv", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument(
        "--cleanliness-train",
        type=Path,
        default=DEFAULT_CLEANLINESS_DIR / "event_cleanliness_train.jsonl.gz",
    )
    parser.add_argument(
        "--cleanliness-eval",
        type=Path,
        default=DEFAULT_CLEANLINESS_DIR / "event_cleanliness_eval.jsonl.gz",
    )
    parser.add_argument(
        "--route-priority",
        action="append",
        default=None,
        help="Repeatable source route, highest priority first.",
    )
    parser.add_argument("--target-train-videos-per-class", type=int, default=100)
    parser.add_argument("--target-eval-videos-per-class", type=int, default=20)
    parser.add_argument("--reserve-train-videos-per-class", type=int, default=100)
    parser.add_argument("--reserve-eval-videos-per-class", type=int, default=20)
    parser.add_argument("--crop-padding-seconds", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument(
        "--allow-unverified-route-label",
        action="store_true",
        help="Allow a video-ID route whose weak label list does not contain the strong target MID.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.target_train_videos_per_class <= 0:
        parser.error("--target-train-videos-per-class must be positive")
    if args.target_eval_videos_per_class <= 0:
        parser.error("--target-eval-videos-per-class must be positive")
    if args.reserve_train_videos_per_class < 0:
        parser.error("--reserve-train-videos-per-class must be non-negative")
    if args.reserve_eval_videos_per_class < 0:
        parser.error("--reserve-eval-videos-per-class must be non-negative")
    if bool(args.reserve_train_videos_per_class) != bool(
        args.reserve_eval_videos_per_class
    ):
        parser.error("reserve train/eval quotas must both be positive or both be zero")
    if args.crop_padding_seconds < 0:
        parser.error("--crop-padding-seconds must be non-negative")
    return args


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AvailabilityAwarePlanError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise AvailabilityAwarePlanError(
                    f"expected JSON object at {path}:{line_number}"
                )
            yield row


def _gzip_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AvailabilityAwarePlanError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise AvailabilityAwarePlanError(
                    f"expected JSON object at {path}:{line_number}"
                )
            yield row


def _selected_label_to_mid(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            label = str(row.get("label") or "")
            mid = str(row.get("mid") or "")
            if not label or not mid:
                raise AvailabilityAwarePlanError(
                    f"selected ontology row has no label/MID: {row}"
                )
            if label in result and result[label] != mid:
                raise AvailabilityAwarePlanError(
                    f"selected label maps to multiple MIDs: {label}"
                )
            result[label] = mid
    return result


def _tsv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        return b"metadata_split\tlabel\n"
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _markdown(report: Mapping[str, Any]) -> str:
    planned = report["planned_crop_rows"]
    reserve = report["reserve_crop_rows"]
    deficits = report["deficits"]
    lines = [
        "# Availability-aware AudioSet-Strong crop plan",
        "",
        f"- Audit: **{'PASS' if report['audit_passes'] else 'FAIL'}**",
        f"- Classes: {report['selected_classes']}",
        f"- Planned train/eval crops: {planned['train']:,} / {planned['eval']:,}",
        f"- Ranked reserve train/eval crops: {reserve['train']:,} / {reserve['eval']:,}",
        f"- Primary/reserve source overlap: {report['primary_reserve_source_overlap']}",
        f"- Unique source videos: {report['planned_unique_source_videos']['total']:,}",
        f"- Availability locations: {report['availability']['indexed_physical_locations']:,}",
        f"- Route-label mismatches observed: {report['availability'].get('label_mismatched_locations', 0):,}",
        f"- Classes/splits with deficit: {len(deficits)}",
        "",
        "## Route allocation",
        "",
    ]
    for split in ("train", "eval"):
        for route, count in report["route_counts"][split].items():
            lines.append(f"- {split} / `{route}`: {count:,}")
    if deficits:
        lines.extend(["", "## Deficits", ""])
        for row in deficits:
            lines.append(
                f"- {row['metadata_split']} / {row['label']}: "
                f"{row['available_unique_videos']}/{row['target']} "
                f"(deficit {row['deficit']})"
            )
    lines.extend(
        [
            "",
            "Availability is joined before quota selection. No QA answer, detector score, or downstream metric is used.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    availability_paths = tuple(
        path.resolve()
        for path in (args.availability_index or [DEFAULT_AVAILABILITY_INDEX])
    )
    cleanliness_paths = (
        args.cleanliness_train.resolve(),
        args.cleanliness_eval.resolve(),
    )
    ontology_path = args.selected_ontology_tsv.resolve()
    metadata_dir = args.metadata_dir.resolve()
    required = [
        *availability_paths,
        *cleanliness_paths,
        ontology_path,
        metadata_dir / "audioset_train_strong.csv",
        metadata_dir / "audioset_eval_strong.csv",
        metadata_dir / "class_labels_indices_strong.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing inputs: {missing}")
    output_dir = args.output_dir.resolve()
    output_paths = [
        output_dir / "crop_source_plan_train.jsonl",
        output_dir / "crop_source_plan_eval.jsonl",
        output_dir / "crop_source_reserve_train.jsonl",
        output_dir / "crop_source_reserve_eval.jsonl",
        output_dir / "per_class_available_support.tsv",
        output_dir / "per_class_reserve_support.tsv",
        output_dir / "availability_aware_plan_receipt.json",
        output_dir / "availability_aware_plan_receipt.md",
    ]
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"outputs exist; pass --overwrite: {existing}")
    selected = _selected_label_to_mid(ontology_path)
    if len(selected) != 200:
        raise SystemExit(f"expected 200 selected labels, found {len(selected)}")
    _, events_by_split = load_strong_metadata(metadata_dir)
    availability_entries = (
        row for path in availability_paths for row in _jsonl(path)
    )
    cleanliness_rows = (
        row for path in cleanliness_paths for row in _gzip_jsonl(path)
    )
    plans, report = build_availability_aware_plan(
        cleanliness_rows=cleanliness_rows,
        availability_entries=availability_entries,
        events_by_split=events_by_split,
        selected_label_to_mid=selected,
        route_priority=tuple(args.route_priority or DEFAULT_ROUTE_PRIORITY),
        target_train_videos_per_class=int(args.target_train_videos_per_class),
        target_eval_videos_per_class=int(args.target_eval_videos_per_class),
        reserve_train_videos_per_class=int(args.reserve_train_videos_per_class),
        reserve_eval_videos_per_class=int(args.reserve_eval_videos_per_class),
        crop_padding_seconds=float(args.crop_padding_seconds),
        seed=int(args.seed),
        require_label_identity=not bool(args.allow_unverified_route_label),
    )
    report["inputs"] = [
        {"path": str(path), "sha256": _sha256_file(path)}
        for path in [*availability_paths, *cleanliness_paths, ontology_path]
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_paths[0], plans["train"])
    atomic_jsonl(output_paths[1], plans["eval"])
    atomic_jsonl(output_paths[2], plans["reserve_train"])
    atomic_jsonl(output_paths[3], plans["reserve_eval"])
    atomic_bytes(output_paths[4], _tsv(report["per_class_support"]))
    atomic_bytes(
        output_paths[5],
        _tsv(report["per_class_reserve_support_after_primary_source_exclusion"]),
    )
    atomic_json(output_paths[6], report)
    atomic_bytes(output_paths[7], _markdown(report).encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(report["audit_passes"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
