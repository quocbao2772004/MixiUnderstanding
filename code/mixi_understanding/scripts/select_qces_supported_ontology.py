#!/usr/bin/env python3
"""Select and audit a specific, well-supported 200-class QCES ontology.

The command is metadata-only: it does not download audio or train a model.  It
writes a new sidecar ontology, reports support in already-materialized audio,
and emits exact unseen AudioSet video plans for remaining 100/20 support gaps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mixi_understanding.qces.supported_ontology import (
    FORMAT,
    audit_materialized_support,
    build_missing_materialization_plan,
    load_hierarchy,
    load_strong_metadata,
    materialized_video_assignments,
    read_jsonl,
    select_supported_ontology,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_DIR = PROJECT_ROOT / "code/baseline/PretrainedSED/hf_dataset_gen/metadata"
DEFAULT_HIERARCHY_DIR = PROJECT_ROOT / "outputs/qces_supported_ontology_official_assets_v1"
DEFAULT_CLEAN_ROOT = PROJECT_ROOT / "outputs/qces_clean_detector_protocol_v1_current"
DEFAULT_PRESERVED_ROOT = PROJECT_ROOT / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1"
ONTOLOGY_URL = "https://raw.githubusercontent.com/audioset/ontology/master/ontology.json"
ONTOLOGY_README_URL = "https://raw.githubusercontent.com/audioset/ontology/master/README.md"
ONTOLOGY_LICENSE = "CC BY-SA 4.0"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument(
        "--audioset-ontology-json",
        type=Path,
        default=DEFAULT_HIERARCHY_DIR / "audioset_ontology.json",
    )
    parser.add_argument(
        "--audioset-ontology-readme",
        type=Path,
        default=DEFAULT_HIERARCHY_DIR / "README.md",
    )
    parser.add_argument("--clean-split-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    parser.add_argument("--preserved-root", type=Path, default=DEFAULT_PRESERVED_ROOT)
    parser.add_argument("--target-labels", type=int, default=200)
    parser.add_argument("--minimum-train-videos", type=int, default=100)
    parser.add_argument("--minimum-eval-videos", type=int, default=20)
    parser.add_argument("--materialization-train-target", type=int, default=100)
    parser.add_argument("--materialization-eval-target", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "target_labels",
        "minimum_train_videos",
        "minimum_eval_videos",
        "materialization_train_target",
        "materialization_eval_target",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


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


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def _tsv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "label\n"
    fieldnames = list(rows[0])
    temporary = tempfile.SpooledTemporaryFile(mode="w+", encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(temporary, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        temporary.seek(0)
        return temporary.read()
    finally:
        temporary.close()


def _materialized_reannotation_audit(
    *,
    clean_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    assignments: Mapping[str, set[str]],
    strong_events: Mapping[str, Mapping[str, Sequence[Any]]],
    selected_labels: set[str],
) -> dict[str, Any]:
    manifested_pairs: set[tuple[str, str]] = set()
    for split, rows in clean_rows.items():
        for row in rows:
            source = str(row.get("protocol_source") or row.get("source_route") or "")
            if "audioset" not in source.lower():
                continue
            video_id = str(row.get("video_id") or "")
            for event in row.get("events") or ():
                label = str(event.get("label") or "")
                if label in selected_labels:
                    manifested_pairs.add((video_id, label))
    metadata_pairs: set[tuple[str, str]] = set()
    for split in ("train", "dev", "test"):
        metadata_split = "eval" if split == "test" else "train"
        for video_id in assignments[split]:
            metadata_pairs.update(
                (video_id, event.label)
                for event in strong_events[metadata_split].get(video_id, ())
                if event.label in selected_labels
            )
    recoverable = metadata_pairs - manifested_pairs
    return {
        "selected_label_video_pairs_in_current_manifests": len(manifested_pairs),
        "selected_label_video_pairs_from_full_metadata": len(metadata_pairs),
        "recoverable_label_video_pairs_without_audio_download": len(recoverable),
        "classes_with_recoverable_annotations": len({label for _, label in recoverable}),
        "sample": [
            {"video_id": video_id, "label": label}
            for video_id, label in sorted(recoverable)[:30]
        ],
    }


def _markdown(receipt: Mapping[str, Any]) -> str:
    selection = receipt["selection"]
    materialized = receipt["materialized_support"]
    plan = receipt["materialization_plan"]
    lines = [
        "# QCES supported 200-class ontology receipt",
        "",
        f"Selection status: **{'PASS' if receipt['selection_passes'] else 'FAIL'}**",
        "",
        "| Check | Value |",
        "|---|---:|",
        f"| Metadata labels | {selection['metadata_labels']} |",
        f"| Labels meeting 100/20 metadata gate | {selection['metadata_eligible_labels_before_specificity_filter']} |",
        f"| Specific eligible leaves | {selection['specific_eligible_labels']} |",
        f"| Selected | {selection['selected_labels']}/{selection['target_labels']} |",
        f"| Selected ancestor-descendant pairs | {selection['parent_child_pairs_in_selected']} |",
        f"| Selected classes already materialized at AudioSet 100/20 | {receipt['downloaded_material']['classes_meeting_audioset_100_20']} |",
        "",
        "## Current materialized support",
        "",
        "| split | positive classes | scenes min/median/max | active seconds min/median/max |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "dev", "test"):
        row = materialized["split_summary"][split]
        lines.append(
            f"| {split} | {row['classes_with_positive']} | "
            f"{row['scene_support_min']}/{row['scene_support_median']}/{row['scene_support_max']} | "
            f"{row['active_seconds_min']:.2f}/{row['active_seconds_median']:.2f}/{row['active_seconds_max']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Missing AudioSet materialization plan",
            "",
            f"- Train: {plan['train']['planned_new_videos']} unseen videos; class-video deficit {plan['train']['total_initial_class_video_deficit']} → {plan['train']['total_unresolved_class_video_deficit']}.",
            f"- Eval: {plan['eval']['planned_new_videos']} unseen videos; class-video deficit {plan['eval']['total_initial_class_video_deficit']} → {plan['eval']['total_unresolved_class_video_deficit']}.",
            f"- Audio can be re-annotated without download for {receipt['reannotation_audit']['recoverable_label_video_pairs_without_audio_download']} existing label-video pairs.",
            "",
            "The official AudioSet ontology snapshot is licensed CC BY-SA 4.0.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_dir = args.metadata_dir.resolve()
    hierarchy_path = args.audioset_ontology_json.resolve()
    hierarchy_readme = args.audioset_ontology_readme.resolve()
    clean_root = args.clean_split_root.resolve()
    preserved_root = args.preserved_root.resolve()
    required_paths = [
        metadata_dir / "class_labels_indices_strong.csv",
        metadata_dir / "audioset_train_strong.csv",
        metadata_dir / "audioset_eval_strong.csv",
        hierarchy_path,
        hierarchy_readme,
        *(clean_root / f"detector_manifest_{split}.jsonl" for split in ("train", "dev", "test")),
        preserved_root / "detector_source_manifest_train_clean.jsonl",
        preserved_root / "detector_source_manifest_val_clean.jsonl",
        preserved_root / "detector_source_manifest_test_clean.jsonl",
    ]
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing required inputs: {[str(path) for path in missing]}")

    coverage, strong_events = load_strong_metadata(metadata_dir)
    hierarchy = load_hierarchy(hierarchy_path)
    selection = select_supported_ontology(
        coverage,
        target_labels=args.target_labels,
        minimum_train_videos=args.minimum_train_videos,
        minimum_eval_videos=args.minimum_eval_videos,
        hierarchy_descendants=hierarchy.descendants,
        hierarchy_restrictions=hierarchy.restrictions,
    )
    selected_labels = {row.label for row in selection.selected}

    clean_rows = {
        split: read_jsonl(clean_root / f"detector_manifest_{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    assignments = materialized_video_assignments(clean_rows)
    preserved_rows = {
        "train": read_jsonl(preserved_root / "detector_source_manifest_train_clean.jsonl"),
        "dev": read_jsonl(preserved_root / "detector_source_manifest_val_clean.jsonl"),
        "test": read_jsonl(preserved_root / "detector_source_manifest_test_clean.jsonl"),
    }
    materialized = audit_materialized_support(
        selection.selected,
        strong_events=strong_events,
        audioset_video_ids=assignments,
        preserved_rows=preserved_rows,
    )
    plan = build_missing_materialization_plan(
        selection.selected,
        strong_events=strong_events,
        existing_train_video_ids=assignments["train"] | assignments["dev"],
        existing_eval_video_ids=assignments["test"],
        target_train_videos=args.materialization_train_target,
        target_eval_videos=args.materialization_eval_target,
        seed=args.seed,
    )
    reannotation = _materialized_reannotation_audit(
        clean_rows=clean_rows,
        assignments=assignments,
        strong_events=strong_events,
        selected_labels=selected_labels,
    )

    actual_by_label = {row["label"]: row for row in materialized["per_label"]}
    selection_rows = []
    for row in selection.audit_rows:
        enriched = dict(row)
        if row["label"] in actual_by_label:
            enriched.update(actual_by_label[row["label"]])
        selection_rows.append(enriched)
    selected_rows = [row for row in selection_rows if row["status"] == "selected"]
    train_plan_summary = plan["summary"]["train"]
    eval_plan_summary = plan["summary"]["eval"]
    classes_meeting_downloaded_as_targets = sum(
        train_plan_summary["per_label"][label]["existing"]
        >= args.materialization_train_target
        and eval_plan_summary["per_label"][label]["existing"]
        >= args.materialization_eval_target
        for label in selected_labels
    )
    classes_meeting_combined_actual = sum(
        int(row["materialized_train_scenes"]) >= args.materialization_train_target
        and int(row["materialized_test_scenes"]) >= args.materialization_eval_target
        for row in selected_rows
    )
    missing_rows = [
        {
            "label": label,
            "train_existing_audioset_videos": train_plan_summary["per_label"][label]["existing"],
            "train_target_audioset_videos": args.materialization_train_target,
            "train_missing_before_plan": train_plan_summary["per_label"][label]["missing_before_plan"],
            "train_planned_coverage": train_plan_summary["per_label"][label]["planned_coverage"],
            "eval_existing_audioset_videos": eval_plan_summary["per_label"][label]["existing"],
            "eval_target_audioset_videos": args.materialization_eval_target,
            "eval_missing_before_plan": eval_plan_summary["per_label"][label]["missing_before_plan"],
            "eval_planned_coverage": eval_plan_summary["per_label"][label]["planned_coverage"],
        }
        for label in sorted(selected_labels)
    ]

    selection_passes = (
        selection.receipt["selection_feasible"]
        and selection.receipt["parent_child_audit_complete"]
        and selection.receipt["parent_child_pairs_in_selected"] == 0
    )
    plan_resolves_all = (
        train_plan_summary["total_unresolved_class_video_deficit"] == 0
        and eval_plan_summary["total_unresolved_class_video_deficit"] == 0
    )
    receipt = {
        "format": FORMAT,
        "selection_passes": selection_passes,
        "materialization_plan_resolves_all_metadata_deficits": plan_resolves_all,
        "selection": selection.receipt,
        "official_hierarchy": {
            "ontology_url": ONTOLOGY_URL,
            "ontology_path": str(hierarchy_path),
            "ontology_sha256": _sha256(hierarchy_path),
            "readme_url": ONTOLOGY_README_URL,
            "readme_path": str(hierarchy_readme),
            "readme_sha256": _sha256(hierarchy_readme),
            "license": ONTOLOGY_LICENSE,
            "nodes": len(hierarchy.descendants),
        },
        "metadata_inputs": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in (
                metadata_dir / "class_labels_indices_strong.csv",
                metadata_dir / "audioset_train_strong.csv",
                metadata_dir / "audioset_eval_strong.csv",
            )
        },
        "materialized_support": materialized,
        "downloaded_material": {
            "selected_classes": len(selected_labels),
            "classes_meeting_audioset_100_20": classes_meeting_downloaded_as_targets,
            "classes_meeting_combined_train_test_scene_targets": classes_meeting_combined_actual,
            "all_200_ready_now": classes_meeting_downloaded_as_targets == len(selected_labels),
        },
        "reannotation_audit": reannotation,
        "materialization_plan": {
            "train": {key: value for key, value in train_plan_summary.items() if key != "per_label"},
            "eval": {key: value for key, value in eval_plan_summary.items() if key != "per_label"},
        },
    }

    output_dir = args.output_dir.resolve()
    paths = {
        "ontology": output_dir / "ontology_200_supported.txt",
        "selected_tsv": output_dir / "ontology_200_supported.tsv",
        "audit_tsv": output_dir / "ontology_selection_audit.tsv",
        "missing_tsv": output_dir / "missing_materialization_by_label.tsv",
        "train_plan": output_dir / "materialization_plan_train.jsonl",
        "eval_plan": output_dir / "materialization_plan_eval.jsonl",
        "receipt": output_dir / "selection_receipt.json",
        "markdown": output_dir / "selection_receipt.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"outputs exist: {[str(path) for path in existing]}; use --overwrite")
    _atomic_text(
        paths["ontology"],
        "\n".join(row.label for row in selection.selected) + "\n",
        overwrite=args.overwrite,
    )
    _atomic_text(paths["selected_tsv"], _tsv(selected_rows), overwrite=args.overwrite)
    _atomic_text(paths["audit_tsv"], _tsv(selection_rows), overwrite=args.overwrite)
    _atomic_text(paths["missing_tsv"], _tsv(missing_rows), overwrite=args.overwrite)
    _atomic_text(paths["train_plan"], _jsonl(plan["plans"]["train"]), overwrite=args.overwrite)
    _atomic_text(paths["eval_plan"], _jsonl(plan["plans"]["eval"]), overwrite=args.overwrite)
    receipt["outputs"] = {
        name: {"path": str(path), "sha256": _sha256(path)}
        for name, path in paths.items()
        if name not in {"receipt", "markdown"}
    }
    _atomic_text(
        paths["receipt"],
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        overwrite=args.overwrite,
    )
    _atomic_text(paths["markdown"], _markdown(receipt), overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "selection_passes": selection_passes,
                "selected_labels": len(selection.selected),
                "metadata_eligible": selection.receipt[
                    "metadata_eligible_labels_before_specificity_filter"
                ],
                "specific_eligible": selection.receipt["specific_eligible_labels"],
                "parent_child_pairs": selection.receipt["parent_child_pairs_in_selected"],
                "downloaded_classes_ready_100_20": classes_meeting_downloaded_as_targets,
                "planned_new_train_videos": train_plan_summary["planned_new_videos"],
                "planned_new_eval_videos": eval_plan_summary["planned_new_videos"],
                "unresolved_train_deficit": train_plan_summary[
                    "total_unresolved_class_video_deficit"
                ],
                "unresolved_eval_deficit": eval_plan_summary[
                    "total_unresolved_class_video_deficit"
                ],
                "receipt": str(paths["receipt"]),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if selection_passes and plan_resolves_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
