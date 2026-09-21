#!/usr/bin/env python3
"""Audit AudioSet-Strong acoustic cleanliness and plan clean 200-class crops."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from mixi_understanding.qces.acoustic_cleanliness import (
    FORMAT,
    TIER_NAMES,
    audit_video_events,
    best_event_per_label_video,
    build_crop_source_plan,
    build_hierarchy_ancestors,
    event_json,
    load_materialized_audio,
    selected_rows_from_tsv,
    summarize_class_support,
    support_summary,
)
from mixi_understanding.qces.supported_ontology import (
    load_hierarchy,
    load_strong_metadata,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_DIR = PROJECT_ROOT / "code/baseline/PretrainedSED/hf_dataset_gen/metadata"
DEFAULT_ONTOLOGY = (
    PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1/ontology_200_supported.tsv"
)
DEFAULT_HIERARCHY = (
    PROJECT_ROOT
    / "outputs/qces_supported_ontology_official_assets_v1/audioset_ontology.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_acoustic_cleanliness_200_v1"
DEFAULT_MATERIALIZED_MANIFESTS = (
    PROJECT_ROOT
    / "outputs/qces_clean_detector_protocol_v1_current/detector_manifest_train.jsonl",
    PROJECT_ROOT
    / "outputs/qces_clean_detector_protocol_v1_current/detector_manifest_dev.jsonl",
    PROJECT_ROOT
    / "outputs/qces_clean_detector_protocol_v1_current/detector_manifest_test.jsonl",
    PROJECT_ROOT
    / "outputs/qces_audioset_strong_subset_qces200_q40_10/audioset_strong_detector_manifest_train.partial.jsonl",
    PROJECT_ROOT
    / "outputs/qces_audioset_strong_subset_qces200_q40_10/audioset_strong_detector_manifest_test.partial.jsonl",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--selected-ontology-tsv", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--audioset-ontology-json", type=Path, default=DEFAULT_HIERARCHY)
    parser.add_argument(
        "--materialized-manifest",
        action="append",
        type=Path,
        default=None,
        help="Repeatable; defaults to current clean and partial AudioSet manifests.",
    )
    parser.add_argument("--minimum-isolation-margin-seconds", type=float, default=0.25)
    parser.add_argument("--maximum-low-overlap-fraction", type=float, default=0.10)
    parser.add_argument("--crop-padding-seconds", type=float, default=0.25)
    parser.add_argument("--target-train-videos-per-class", type=int, default=100)
    parser.add_argument("--target-eval-videos-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.minimum_isolation_margin_seconds < 0:
        parser.error("--minimum-isolation-margin-seconds must be non-negative")
    if not 0 <= args.maximum_low_overlap_fraction <= 1:
        parser.error("--maximum-low-overlap-fraction must be in [0, 1]")
    if args.crop_padding_seconds < 0:
        parser.error("--crop-padding-seconds must be non-negative")
    if args.target_train_videos_per_class <= 0 or args.target_eval_videos_per_class <= 0:
        parser.error("per-class source targets must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_plain(path: Path, *, overwrite: bool) -> tuple[TextIO, Path]:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; pass --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    return os.fdopen(descriptor, "w", encoding="utf-8", newline=""), Path(temporary_name)


def _write_text(path: Path, text: str, *, overwrite: bool) -> None:
    handle, temporary = _atomic_plain(path, overwrite=overwrite)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class _AtomicDeterministicGzip:
    def __init__(self, path: Path, *, overwrite: bool):
        if path.exists() and not overwrite:
            raise FileExistsError(f"output exists: {path}; pass --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        self.path = path
        self.temporary = Path(temporary_name)
        self.raw = os.fdopen(descriptor, "wb")
        self.gzip = gzip.GzipFile(fileobj=self.raw, mode="wb", filename="", mtime=0)

    def write(self, text: str) -> None:
        self.gzip.write(text.encode("utf-8"))

    def close(self, *, commit: bool) -> None:
        try:
            self.gzip.close()
            self.raw.flush()
            os.fsync(self.raw.fileno())
            self.raw.close()
            if commit:
                os.replace(self.temporary, self.path)
        finally:
            if not self.raw.closed:
                self.raw.close()
            if self.temporary.exists():
                self.temporary.unlink()


def _tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "label\n"
    output = tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        output.seek(0)
        return output.read()
    finally:
        output.close()


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(event_json(row) + "\n" for row in rows)


def _input_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path.resolve())}


def _markdown(receipt: Mapping[str, Any]) -> str:
    support = receipt["support_summary"]
    plan = receipt["source_plan"]
    lines = [
        "# QCES 200-class acoustic-cleanliness audit",
        "",
        f"Audit status: **{'PASS' if receipt['audit_passes'] else 'FAIL'}**",
        "",
        "## Isolation feasibility",
        "",
        "| Check | Classes |",
        "|---|---:|",
        f"| Selected ontology | {support['selected_classes']} |",
        f"| Isolated train support >=100 | {support['classes_with_isolated_train_100']} |",
        f"| Isolated eval support >=20 | {support['classes_with_isolated_eval_20']} |",
        f"| Isolated support >=100/20 | {support['classes_with_isolated_100_20']} |",
        f"| Zero selected-distractor overlap, train >=100 | {support['classes_with_selected_isolated_train_100']} |",
        f"| Zero selected-distractor overlap, eval >=20 | {support['classes_with_selected_isolated_eval_20']} |",
        f"| Zero selected-distractor overlap >=100/20 | {support['classes_with_selected_isolated_100_20']} |",
        f"| Clean (tier 0-2) support >=100/20 | {support['classes_with_clean_100_20']} |",
        f"| Already materialized, selected-isolated >=100/20 | {support['classes_materialized_selected_isolated_100_20_ready']} |",
        f"| Already materialized, clean tier 0-2 >=100/20 | {support['classes_materialized_clean_100_20_ready']} |",
        "",
        "Strict isolation means zero overlap with every different official strong label and at least "
        f"{receipt['configuration']['minimum_isolation_margin_seconds']:.2f}s clearance to the nearest different event.",
        "The detector-oriented support gate is reported separately: zero overlap with another of the selected 200 labels, while unselected hierarchy/context labels remain fully annotated.",
        "",
        "## Deterministic source plan",
        "",
        f"- Planned class-video rows: {plan['planned_rows']}",
        f"- Unique videos: {plan['planned_unique_videos']}",
        f"- Already materialized unique videos: {plan['materialized_unique_videos']}",
        f"- Existing materialized videos audited: {receipt['materialized_availability']['videos_with_existing_audio']}",
        "- Priority is acoustic tier first, then existing local audio, overlap, margin, duration, and a seeded hash.",
        "- Every crop contains all intersecting official strong annotations; `coverage_label` is planning metadata, not a hidden single-label target.",
        "",
        "## Ambiguity tiers",
        "",
        "| Tier | Meaning |",
        "|---:|---|",
    ]
    for tier, name in TIER_NAMES.items():
        lines.append(f"| {tier} | {name} |")
    lines.extend(
        [
            "",
            "This audit does not use QA answers, model predictions, detector scores or downstream benchmark outcomes.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_dir = args.metadata_dir.resolve()
    selected_path = args.selected_ontology_tsv.resolve()
    hierarchy_path = args.audioset_ontology_json.resolve()
    manifest_paths = tuple(args.materialized_manifest or DEFAULT_MATERIALIZED_MANIFESTS)
    required = [
        selected_path,
        hierarchy_path,
        metadata_dir / "class_labels_indices_strong.csv",
        metadata_dir / "audioset_train_strong.csv",
        metadata_dir / "audioset_eval_strong.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required inputs: {missing}")

    selected_rows = selected_rows_from_tsv(selected_path)
    selected_mids = {row["mid"] for row in selected_rows}
    if len(selected_rows) != 200:
        raise SystemExit(f"expected exactly 200 selected labels, found {len(selected_rows)}")
    coverage, events_by_split = load_strong_metadata(metadata_dir)
    hierarchy = load_hierarchy(hierarchy_path)
    ancestors = build_hierarchy_ancestors(hierarchy.descendants)
    materialized = load_materialized_audio(
        manifest_paths,
        project_root=PROJECT_ROOT,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_paths = {
        split: output_dir / f"event_cleanliness_{split}.jsonl.gz"
        for split in ("train", "eval")
    }
    writers = {
        split: _AtomicDeterministicGzip(path, overwrite=args.overwrite)
        for split, path in event_paths.items()
    }
    observed_counts = Counter()
    tier_counts: dict[str, Counter[int]] = {"train": Counter(), "eval": Counter()}
    all_best: dict[tuple[str, str, str], dict[str, Any]] = {}
    train_videos = set(events_by_split["train"])
    eval_videos = set(events_by_split["eval"])
    success = False
    try:
        for split in ("train", "eval"):
            for video_id in sorted(events_by_split[split]):
                rows = audit_video_events(
                    metadata_split=split,
                    video_id=video_id,
                    events=events_by_split[split][video_id],
                    selected_mids=selected_mids,
                    hierarchy_descendants=hierarchy.descendants,
                    hierarchy_ancestors=ancestors,
                    materialized=materialized.get((split, video_id)),
                    minimum_isolation_margin_seconds=(
                        args.minimum_isolation_margin_seconds
                    ),
                    maximum_low_overlap_fraction=args.maximum_low_overlap_fraction,
                )
                for row in rows:
                    writers[split].write(event_json(row) + "\n")
                    observed_counts[(split, row["label"])] += 1
                    tier_counts[split][int(row["ambiguity_tier"])] += 1
                video_best = best_event_per_label_video(rows, seed=args.seed)
                all_best.update(video_best)
        success = True
    finally:
        for writer in writers.values():
            writer.close(commit=success)

    class_rows = summarize_class_support(
        selected_rows=selected_rows,
        best_candidates=all_best,
        train_target=args.target_train_videos_per_class,
        eval_target=args.target_eval_videos_per_class,
    )
    crop_plan, plan_summary = build_crop_source_plan(
        best_candidates=all_best,
        events_by_split=events_by_split,
        selected_mids=selected_mids,
        target_train_videos_per_class=args.target_train_videos_per_class,
        target_eval_videos_per_class=args.target_eval_videos_per_class,
        crop_padding_seconds=args.crop_padding_seconds,
        seed=args.seed,
    )
    plan_paths = {
        split: output_dir / f"crop_source_plan_{split}.jsonl"
        for split in ("train", "eval")
    }
    for split, path in plan_paths.items():
        _write_text(
            path,
            _jsonl([row for row in crop_plan if row["metadata_split"] == split]),
            overwrite=args.overwrite,
        )
    class_path = output_dir / "per_class_cleanliness.tsv"
    _write_text(class_path, _tsv(class_rows), overwrite=args.overwrite)

    expected_counts = {
        (split, row["label"]): int(row[f"{split}_events"])
        for row in selected_rows
        for split in ("train", "eval")
    }
    count_mismatches = [
        {
            "split": split,
            "label": label,
            "expected": expected,
            "observed": observed_counts[(split, label)],
        }
        for (split, label), expected in sorted(expected_counts.items())
        if observed_counts[(split, label)] != expected
    ]
    raw_video_mismatches = []
    selected_by_label = {row["label"]: row for row in selected_rows}
    for row in class_rows:
        source = selected_by_label[row["label"]]
        for split in ("train", "eval"):
            expected = int(source[f"{split}_videos"])
            observed = int(row[f"raw_{split}_unique_videos"])
            if expected != observed:
                raw_video_mismatches.append(
                    {
                        "split": split,
                        "label": row["label"],
                        "expected": expected,
                        "observed": observed,
                    }
                )
    plan_annotation_failures = [
        row["selection_key"]
        for row in crop_plan
        if not any(
            annotation["mid"] == row["coverage_mid"]
            and annotation["source_onset_seconds"] <= row["event_onset_seconds"] + 1e-9
            and annotation["source_offset_seconds"] >= row["event_offset_seconds"] - 1e-9
            for annotation in row["strong_annotations"]
        )
    ]
    support = support_summary(class_rows)
    receipt = {
        "format": FORMAT,
        "audit_passes": not (
            count_mismatches
            or raw_video_mismatches
            or plan_annotation_failures
            or (train_videos & eval_videos)
        ),
        "configuration": {
            "selected_labels": len(selected_rows),
            "minimum_isolation_margin_seconds": args.minimum_isolation_margin_seconds,
            "maximum_low_overlap_fraction": args.maximum_low_overlap_fraction,
            "crop_padding_seconds": args.crop_padding_seconds,
            "target_train_videos_per_class": args.target_train_videos_per_class,
            "target_eval_videos_per_class": args.target_eval_videos_per_class,
            "seed": args.seed,
        },
        "inputs": {
            "selected_ontology": _input_record(selected_path),
            "official_hierarchy": _input_record(hierarchy_path),
            "strong_metadata": {
                filename: _input_record(metadata_dir / filename)
                for filename in (
                    "class_labels_indices_strong.csv",
                    "audioset_train_strong.csv",
                    "audioset_eval_strong.csv",
                )
            },
            "materialized_manifests": [
                _input_record(path.resolve())
                for path in manifest_paths
                if path.is_file()
            ],
        },
        "event_audit": {
            "selected_events_processed": {
                split: sum(
                    count
                    for (candidate_split, _), count in observed_counts.items()
                    if candidate_split == split
                )
                for split in ("train", "eval")
            },
            "ambiguity_tiers": {
                split: {
                    TIER_NAMES[tier]: count
                    for tier, count in sorted(tier_counts[split].items())
                }
                for split in ("train", "eval")
            },
            "event_count_mismatches": count_mismatches,
            "raw_video_count_mismatches": raw_video_mismatches,
        },
        "support_summary": support,
        "source_plan": plan_summary,
        "materialized_availability": {
            "manifest_records_merged": len(materialized),
            "videos_with_existing_audio": sum(
                item.audio_exists for item in materialized.values()
            ),
            "existing_train_videos": sum(
                item.audio_exists and item.metadata_split == "train"
                for item in materialized.values()
            ),
            "existing_eval_videos": sum(
                item.audio_exists and item.metadata_split == "eval"
                for item in materialized.values()
            ),
        },
        "integrity": {
            "official_train_eval_video_overlap": len(train_videos & eval_videos),
            "plan_rows_missing_complete_coverage_annotation": len(
                plan_annotation_failures
            ),
            "uses_downstream_target_or_predictions": False,
            "all_plan_crops_emit_complete_intersecting_strong_annotations": True,
        },
        "infeasible_isolated_100_20_labels": [
            row["label"] for row in class_rows if not row["isolated_100_20_feasible"]
        ],
        "infeasible_selected_isolated_100_20_labels": [
            row["label"]
            for row in class_rows
            if not row["selected_isolated_100_20_feasible"]
        ],
        "outputs": {},
    }
    receipt_path = output_dir / "acoustic_cleanliness_receipt.json"
    markdown_path = output_dir / "acoustic_cleanliness_receipt.md"
    output_records = {
        "per_class_cleanliness": class_path,
        "event_cleanliness_train": event_paths["train"],
        "event_cleanliness_eval": event_paths["eval"],
        "crop_source_plan_train": plan_paths["train"],
        "crop_source_plan_eval": plan_paths["eval"],
    }
    receipt["outputs"] = {
        name: _input_record(path) for name, path in output_records.items()
    }
    _write_text(
        receipt_path,
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )
    _write_text(markdown_path, _markdown(receipt), overwrite=args.overwrite)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if receipt["audit_passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
