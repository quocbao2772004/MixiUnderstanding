#!/usr/bin/env python3
"""Audit the metadata contract for a 200-class BEATs-v2 microfit.

This gate deliberately does not open, decode, or inspect any audio file.  It
answers a narrower question before GPU time is spent: do the proposed
train/dev/test manifests, ontology, timestamps, source identities, and native
head mapping form a valid 200-class microfit protocol?

Passing this receipt authorizes the metadata side of a debug microfit only.
It is never paper-eligible and does not authorize training until a separate
audio materialization/duration preflight has succeeded.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.qces.clean_detector_splits import audit_strict_identity_overlap
from mixi_understanding.qces.fixed_grid_detector import (
    NUM_FRAMES,
    build_fixed_grid_boundary_targets,
    build_fixed_grid_targets,
)
from mixi_understanding.scripts.train_qces_supported_beats_detector_v2 import (
    _atomic_json,
    _sha256,
    load_labels,
    load_manifest,
    split_calibration_selection_rows,
    validate_native_initialization,
)


FORMAT = "qces_supported_beats_microfit_metadata_gate_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-train-scenes", type=int, default=200)
    parser.add_argument("--max-train-scenes", type=int, default=500)
    parser.add_argument("--min-dev-scenes", type=int, default=20)
    parser.add_argument("--min-test-scenes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    return parser.parse_args(argv)


def _target_summary(rows: Sequence[Any], labels: Sequence[str]) -> dict[str, Any]:
    """Summarize fixed-grid targets in bounded-memory chunks."""

    activity_per_class = torch.zeros(len(labels), dtype=torch.long)
    onset_per_class = torch.zeros(len(labels), dtype=torch.long)
    offset_per_class = torch.zeros(len(labels), dtype=torch.long)
    valid_frames = 0
    activity_in_padding = 0
    boundary_in_padding = 0
    for start in range(0, len(rows), 32):
        batch = rows[start : start + 32]
        activity, valid = build_fixed_grid_targets(batch, num_labels=len(labels))
        onset, offset, boundary_valid = build_fixed_grid_boundary_targets(
            batch, num_labels=len(labels), dilation_frames=0
        )
        if not torch.equal(valid, boundary_valid):
            raise RuntimeError("activity and boundary builders disagree on valid frames")
        valid_3d = valid.unsqueeze(-1)
        activity_in_padding += int(activity.masked_select(~valid_3d.expand_as(activity)).sum())
        boundary_in_padding += int(
            onset.masked_select(~valid_3d.expand_as(onset)).sum()
            + offset.masked_select(~valid_3d.expand_as(offset)).sum()
        )
        activity_per_class += activity.sum(dim=(0, 1)).long()
        onset_per_class += onset.sum(dim=(0, 1)).long()
        offset_per_class += offset.sum(dim=(0, 1)).long()
        valid_frames += int(valid.sum())
    return {
        "grid_frames": NUM_FRAMES,
        "valid_frames": valid_frames,
        "activity_positive_frames": int(activity_per_class.sum()),
        "onset_positive_frames": int(onset_per_class.sum()),
        "offset_positive_frames": int(offset_per_class.sum()),
        "classes_with_activity": int((activity_per_class > 0).sum()),
        "classes_with_onset": int((onset_per_class > 0).sum()),
        "classes_with_observable_offset": int((offset_per_class > 0).sum()),
        "activity_positive_frames_in_padding": activity_in_padding,
        "boundary_positive_frames_in_padding": boundary_in_padding,
        "passes": activity_in_padding == 0 and boundary_in_padding == 0,
    }


def _split_summary(rows: Sequence[Any], labels: Sequence[str]) -> dict[str, Any]:
    scene_support: Counter[str] = Counter()
    occurrence_support: Counter[str] = Counter()
    positives = 0
    duration_clipped_events = 0
    for row in rows:
        present = {str(event["label"]) for event in row.events}
        if present:
            positives += 1
        scene_support.update(present)
        occurrence_support.update(str(event["label"]) for event in row.events)
        duration_clipped_events += sum(
            bool(event.get("offset_clipped_to_duration")) for event in row.events
        )
    missing = [label for label in labels if scene_support[label] == 0]
    return {
        "scenes": len(rows),
        "positive_scenes": positives,
        "negative_scenes": len(rows) - positives,
        "occurrences": sum(occurrence_support.values()),
        "events_clipped_to_declared_duration_within_one_frame": duration_clipped_events,
        "positive_classes": len(labels) - len(missing),
        "missing_classes": missing,
        "minimum_scene_support": min((scene_support[label] for label in labels), default=0),
        "per_class_scene_support": {label: scene_support[label] for label in labels},
        "targets": _target_summary(rows, labels),
    }


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_train_scenes < 1 or args.max_train_scenes < args.min_train_scenes:
        raise ValueError("invalid train scene range")
    if args.min_dev_scenes < 2 or args.min_test_scenes < 1:
        raise ValueError("microfit requires at least two dev scenes and one test scene")
    if not 0.0 < args.calibration_fraction < 1.0:
        raise ValueError("calibration-fraction must lie in (0,1)")

    ontology = args.ontology.resolve()
    initialization = args.initialization.resolve()
    labels = load_labels(ontology)
    manifest_paths = {
        "train": args.train_manifest.resolve(),
        "dev": args.dev_manifest.resolve(),
        "test": args.test_manifest.resolve(),
    }
    loaded: dict[str, list[Any]] = {}
    raw: dict[str, list[dict[str, Any]]] = {}
    for split, path in manifest_paths.items():
        loaded[split], raw[split] = load_manifest(path, labels, expected_split=split)

    initialization_payload = torch.load(initialization, map_location="cpu", weights_only=False)
    native_audit = validate_native_initialization(
        initialization_payload, labels=labels, ontology=ontology
    )
    identity_audit = audit_strict_identity_overlap(raw)
    calibration, selection = split_calibration_selection_rows(
        loaded["dev"], seed=args.seed, calibration_fraction=args.calibration_fraction
    )
    cal_selection_identity = audit_strict_identity_overlap(
        {
            "train": [dict(row.raw) for row in calibration],
            "dev": [dict(row.raw) for row in selection],
        }
    )
    summaries = {split: _split_summary(rows, labels) for split, rows in loaded.items()}

    failures: list[str] = []
    train_scenes = len(loaded["train"])
    if not args.min_train_scenes <= train_scenes <= args.max_train_scenes:
        failures.append(
            f"train scene count {train_scenes} is outside "
            f"[{args.min_train_scenes},{args.max_train_scenes}]"
        )
    if len(loaded["dev"]) < args.min_dev_scenes:
        failures.append(
            f"dev has {len(loaded['dev'])} scenes; requires >= {args.min_dev_scenes}"
        )
    if len(loaded["test"]) < args.min_test_scenes:
        failures.append(
            f"test has {len(loaded['test'])} scenes; requires >= {args.min_test_scenes}"
        )
    if summaries["train"]["positive_classes"] != len(labels):
        failures.append(
            f"train positives cover {summaries['train']['positive_classes']}/200 classes"
        )
    if summaries["train"]["targets"]["classes_with_onset"] != len(labels):
        failures.append("not all 200 train classes produce a valid fixed-grid onset target")
    for split, summary in summaries.items():
        if not summary["targets"]["passes"]:
            failures.append(f"{split} contains positive targets in right padding")
    if not identity_audit["passes"]:
        failures.append("hard source identity overlaps across train/dev/test")
    if not cal_selection_identity["passes"]:
        failures.append("hard source identity overlaps between calibration and selection")
    if native_audit.get("passes") is not True:
        failures.append("native 200-class initialization mapping failed")

    passes = not failures
    return {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passes": passes,
        "failures": failures,
        "scope": {
            "debug_microfit_only": True,
            "paper_eligible": False,
            "metadata_training_authorized": passes,
            "audio_training_authorized": False,
            "reason_audio_not_authorized": (
                "this gate opens zero audio files; run an exact-path audio "
                "materialization/duration preflight before training"
            ),
        },
        "contract": {
            "train_scene_range": [args.min_train_scenes, args.max_train_scenes],
            "minimum_dev_scenes": args.min_dev_scenes,
            "minimum_test_scenes": args.min_test_scenes,
            "required_train_positive_classes": 200,
            "fixed_grid_frames": NUM_FRAMES,
            "boundary_dilation_frames": 0,
            "calibration_fraction": args.calibration_fraction,
            "test_used_for_selection": False,
        },
        "inputs": {
            "ontology": {"path": str(ontology), "sha256": _sha256(ontology), "labels": len(labels)},
            "initialization": {"path": str(initialization), "sha256": _sha256(initialization)},
            "manifests": {
                split: {"path": str(path), "sha256": _sha256(path)}
                for split, path in manifest_paths.items()
            },
        },
        "native_initialization_audit": native_audit,
        "integrity": identity_audit,
        "calibration_selection": {
            "calibration_scenes": len(calibration),
            "selection_scenes": len(selection),
            "identity_audit": cal_selection_identity,
        },
        "splits": summaries,
        "audio_io": {
            "audio_files_opened": 0,
            "audio_paths_checked_for_existence": False,
            "audio_duration_checked": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    try:
        receipt = build_receipt(args)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        paths = {
            "ontology": args.ontology.resolve(),
            "initialization": args.initialization.resolve(),
            "train_manifest": args.train_manifest.resolve(),
            "dev_manifest": args.dev_manifest.resolve(),
            "test_manifest": args.test_manifest.resolve(),
        }
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "passes": False,
            "failures": [f"{type(exc).__name__}: {exc}"],
            "scope": {
                "debug_microfit_only": True,
                "paper_eligible": False,
                "metadata_training_authorized": False,
                "audio_training_authorized": False,
            },
            "input_files": {
                name: {
                    "path": str(path),
                    "exists": path.is_file(),
                    "sha256": _sha256(path) if path.is_file() else None,
                }
                for name, path in paths.items()
            },
            "audio_io": {
                "audio_files_opened": 0,
                "audio_paths_checked_for_existence": False,
                "audio_duration_checked": False,
            },
        }
    _atomic_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
