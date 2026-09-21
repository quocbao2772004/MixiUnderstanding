#!/usr/bin/env python3
"""Build the one minimal AudioSet plan shared by clean crops and the detector."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from mixi_understanding.qces.acoustic_cleanliness import selected_rows_from_tsv
from mixi_understanding.qces.audioset_plan_materializer import atomic_bytes, atomic_json, atomic_jsonl
from mixi_understanding.qces.joint_materialization_plan import (
    FORMAT,
    JointPlanError,
    build_joint_materialization_plan,
    load_crop_rows,
    sha256_file,
)
from mixi_understanding.qces.supported_ontology import load_strong_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CROP_DIR = PROJECT_ROOT / "outputs/qces_acoustic_cleanliness_200_v1"
DEFAULT_RAW_DIR = PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "code/baseline/PretrainedSED/hf_dataset_gen/metadata"
DEFAULT_ONTOLOGY = DEFAULT_RAW_DIR / "ontology_200_supported.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_joint_acoustic_materialization_plan_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-plan", type=Path, action="append", default=None)
    parser.add_argument("--raw-plan", type=Path, action="append", default=None)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--selected-ontology-tsv", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--target-train-per-class", type=int, default=100)
    parser.add_argument("--target-eval-per-class", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_ids(paths: Sequence[Path]) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    ids = {"train": set(), "eval": set()}
    receipts: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise JointPlanError(f"raw comparison plan missing: {path}")
        rows = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                split = str(row.get("metadata_split") or "")
                if split not in ids:
                    raise JointPlanError(f"invalid raw metadata_split={split!r}")
                ids[split].add(str(row["video_id"]))
                rows += 1
        receipts.append({"path": str(path), "sha256": sha256_file(path), "rows": rows})
    return ids, receipts


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    crop_paths = tuple(args.crop_plan or [
        DEFAULT_CROP_DIR / "crop_source_plan_train.jsonl",
        DEFAULT_CROP_DIR / "crop_source_plan_eval.jsonl",
    ])
    raw_paths = tuple(args.raw_plan or [
        DEFAULT_RAW_DIR / "materialization_plan_train.jsonl",
        DEFAULT_RAW_DIR / "materialization_plan_eval.jsonl",
    ])
    output_dir = args.output_dir.resolve()
    output_paths = {
        "train": output_dir / "joint_materialization_plan_train.jsonl",
        "eval": output_dir / "joint_materialization_plan_eval.jsonl",
        "receipt": output_dir / "joint_materialization_receipt.json",
        "markdown": output_dir / "joint_materialization_receipt.md",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"outputs already exist; pass --overwrite: {existing}")

    rows_by_split, crop_receipts = load_crop_rows(crop_paths)
    raw_ids, raw_receipts = _read_ids(raw_paths)
    selected_rows = selected_rows_from_tsv(args.selected_ontology_tsv.resolve())
    selected_labels = {str(row["label"]) for row in selected_rows}
    selected_mids = {str(row["mid"]) for row in selected_rows}
    _, events_by_split = load_strong_metadata(args.metadata_dir.resolve())
    plans, receipt = build_joint_materialization_plan(
        rows_by_split=rows_by_split,
        events_by_split=events_by_split,
        selected_labels=selected_labels,
        selected_mids=selected_mids,
        target_train_per_class=int(args.target_train_per_class),
        target_eval_per_class=int(args.target_eval_per_class),
        raw_plan_ids=raw_ids,
    )
    route_groups: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: {"train": [], "eval": []}
    )
    for split in ("train", "eval"):
        for row in plans[split]:
            route_groups[str(row["source_route"])][split].append(row)
    route_slugs: dict[str, str] = {}
    for route in route_groups:
        slug = re.sub(r"[^a-z0-9]+", "_", route.lower()).strip("_")
        if not slug or slug in route_slugs.values():
            raise JointPlanError(f"ambiguous materialization route slug: {route!r}")
        route_slugs[route] = slug
    route_output_paths = {
        route: {
            split: output_dir
            / "routes"
            / route_slugs[route]
            / f"joint_materialization_plan_{split}.jsonl"
            for split in ("train", "eval")
            if route_groups[route][split]
        }
        for route in sorted(route_groups)
    }
    route_existing = [
        str(path)
        for paths in route_output_paths.values()
        for path in paths.values()
        if path.exists()
    ]
    if route_existing and not args.overwrite:
        raise SystemExit(
            f"route outputs already exist; pass --overwrite: {route_existing}"
        )
    receipt["inputs"] = {
        "crop_source_plans": crop_receipts,
        "old_raw_plans_for_comparison_only": raw_receipts,
        "selected_ontology": {
            "path": str(args.selected_ontology_tsv.resolve()),
            "sha256": sha256_file(args.selected_ontology_tsv.resolve()),
        },
        "strong_metadata_dir": str(args.metadata_dir.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output_paths["train"], plans["train"])
    atomic_jsonl(output_paths["eval"], plans["eval"])
    route_outputs: dict[str, Any] = {}
    for route in sorted(route_groups):
        route_rows = route_groups[route]
        nonempty = [*route_rows["train"], *route_rows["eval"]]
        datasets = {str(row["hf_dataset"]) for row in nonempty}
        revisions = {str(row["hf_revision"]) for row in nonempty}
        if len(datasets) != 1 or len(revisions) != 1:
            raise JointPlanError(
                f"route has conflicting dataset/revision contracts: {route}"
            )
        split_outputs: dict[str, Any] = {}
        for split, path in route_output_paths[route].items():
            atomic_jsonl(path, route_rows[split])
            split_outputs[split] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(route_rows[split]),
            }
        route_outputs[route] = {
            "slug": route_slugs[route],
            "hf_dataset": next(iter(datasets)),
            "hf_revision": next(iter(revisions)),
            "source_videos": len(nonempty),
            "splits": split_outputs,
        }
    receipt["outputs"] = {
        split: {
            "path": str(output_paths[split]),
            "sha256": sha256_file(output_paths[split]),
            "rows": len(plans[split]),
        }
        for split in ("train", "eval")
    }
    receipt["route_outputs"] = route_outputs
    atomic_json(output_paths["receipt"], receipt)
    estimate = receipt["pre_scan_disk_estimate"]
    comparison = receipt["old_raw_plan_comparison"]
    markdown = "\n".join(
        [
            "# Joint acoustic materialization plan",
            "",
            f"- Audit: **{'PASS' if receipt['audit_passes'] else 'FAIL'}**",
            f"- Selected classes: {receipt['selected_classes']}",
            f"- Train class-video contract: {receipt['coverage_class_video_rows']['train']:,}",
            f"- Eval class-video contract: {receipt['coverage_class_video_rows']['eval']:,}",
            f"- Unique train videos: {receipt['unique_videos']['train']:,}",
            f"- Unique eval videos: {receipt['unique_videos']['eval']:,}",
            f"- Total unique videos: {receipt['total_unique_videos']:,}",
            f"- Cross-split source overlap: {receipt['train_eval_source_overlap']}",
            f"- Naive crop+raw union: {receipt['naive_union_total']:,}",
            f"- Raw-only videos intentionally skipped: {receipt['raw_only_videos_skipped_total']:,}",
            "",
            "## Old raw-plan comparison",
            "",
            f"- Train overlap: {comparison['train']['overlap']:,}",
            f"- Eval overlap: {comparison['eval']['overlap']:,}",
            "",
            "## Pre-scan disk estimate",
            "",
            f"- Existing train audio: {estimate['train']['existing_audio_videos']:,}",
            f"- Existing eval audio: {estimate['eval']['existing_audio_videos']:,}",
            f"- Remaining encoded audio estimate: {estimate['total_remaining_audio_bytes_estimate_from_existing_mean']:,} bytes",
            "",
            "The Parquet row-group network upper bound is computed by the subsequent scan-only materializer receipt.",
            "",
        ]
    )
    atomic_bytes(output_paths["markdown"], markdown.encode("utf-8"))
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
