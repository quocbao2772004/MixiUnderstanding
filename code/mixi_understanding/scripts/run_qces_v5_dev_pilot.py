#!/usr/bin/env python3
"""Safely orchestrate the frozen QCES-v5 union/dual development pilot.

The default mode is a read-only preflight that verifies every protocol-bound
artifact and prints the exact commands.  Passing ``--execute`` is the only way
to launch the two 512-update FP32 training jobs and their report-only
validation evaluations.  This development pilot never accepts a manifest
argument, so a test manifest cannot be substituted accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAT = "qces_v5_frozen_dev_pilot_orchestration_v1"
SUMMARY_FORMAT = "qces_v5_frozen_dev_pilot_comparison_v1"
SEED = 2026
EXPECTED_COMMON_COMPOSER_TENSORS = 67
EXPECTED_DUAL_ONLY_COMPOSER_TENSORS = 4

TRAIN_MANIFEST = Path("data/qces_v5_paper/qces_devpilot_train_seed2026.jsonl")
VAL_MANIFEST = Path("data/qces_v5_paper/qces_devpilot_val_seed2026.jsonl")
PILOT_CACHE_ROOT = Path("outputs/qces_v5_paper_devpilot_seed2026")
FOUNDATION_TRAIN = PILOT_CACHE_ROOT / "foundation_train"
FOUNDATION_VAL = PILOT_CACHE_ROOT / "foundation_val"
UNION_SEMANTIC_TRAIN = PILOT_CACHE_ROOT / "semantic_union_train.pt"
UNION_SEMANTIC_VAL = PILOT_CACHE_ROOT / "semantic_union_val.pt"
DUAL_SEMANTIC_TRAIN = PILOT_CACHE_ROOT / "semantic_dual_train.pt"
DUAL_SEMANTIC_VAL = PILOT_CACHE_ROOT / "semantic_dual_val.pt"
SELECTION_RECEIPT = PILOT_CACHE_ROOT / "selection_receipt.json"
AUDIOSEP_ROOT = Path("code/baseline/audiosep")
AUDIOSEP_CONFIG = AUDIOSEP_ROOT / "config/audiosep_base.yaml"
AUDIOSEP_CHECKPOINT = AUDIOSEP_ROOT / "checkpoint/hf_audiosep/pytorch_model.bin"
TRAIN_SCRIPT = Path("code/mixi_understanding/scripts/train_qces.py")
EVALUATE_SCRIPT = Path("code/mixi_understanding/scripts/evaluate_qces.py")
DEFAULT_RESULTS_ROOT = Path("outputs/qces_v5_devpilot_screen512_seed2026")


@dataclass(frozen=True)
class BoundArtifact:
    relative_path: Path
    sha256: str


BOUND_ARTIFACTS = (
    BoundArtifact(
        TRAIN_MANIFEST,
        "741c9b0d4285805bc879ba332f1c0278cf7105ed6a2ee59b79675d141064f252",
    ),
    BoundArtifact(
        VAL_MANIFEST,
        "4135f8ce757df30758a83f3d5f44592479778aecd49644c95b736d6df5a7b2d0",
    ),
    BoundArtifact(
        UNION_SEMANTIC_TRAIN,
        "459cd1cad625c63047d1e9c48a8c35db3cd7066e3e68ff4ba02a9e8328747f81",
    ),
    BoundArtifact(
        UNION_SEMANTIC_VAL,
        "b2b6589ec9381eb657704d734fc9d8b43c34866475bf903d207c1b81fec500c9",
    ),
    BoundArtifact(
        DUAL_SEMANTIC_TRAIN,
        "636e7439875e0b027161ba4f0b9033939070ffdcbb40dbb96738d22866052259",
    ),
    BoundArtifact(
        DUAL_SEMANTIC_VAL,
        "f68212299f1e702d95d7cfacf2360b1ed035fb057e6d9d5c914c2c667d1e5a3d",
    ),
    BoundArtifact(
        FOUNDATION_TRAIN / "cache_receipt.json",
        "d21eabfed2bd2a3fdc1707181b8ad1ce6cbca81c18f82a0d7d1f94da54015d05",
    ),
    BoundArtifact(
        FOUNDATION_VAL / "cache_receipt.json",
        "2efa1dccae2809057714e3ce9f16a1ed5cb75c888556b0b3fc62ebd48da0e2e2",
    ),
    BoundArtifact(
        AUDIOSEP_CHECKPOINT,
        "37f1691fb067e2575f1ad1cfbfe44b7b3da18e52f33fcb2b0937b72952f11ba1",
    ),
)

REQUIRED_UNHASHED_FILES = (
    SELECTION_RECEIPT,
    AUDIOSEP_CONFIG,
    TRAIN_SCRIPT,
    EVALUATE_SCRIPT,
    FOUNDATION_TRAIN / "question_features.pt",
    FOUNDATION_TRAIN / "scene_audio_features.pt",
    FOUNDATION_VAL / "question_features.pt",
    FOUNDATION_VAL / "scene_audio_features.pt",
)

COUNTERFACTUAL_ZERO_OPTIONS = (
    ("--surface-semantic-invariance-weight", "0"),
    ("--surface-role-invariance-weight", "0"),
    ("--surface-no-evidence-invariance-weight", "0"),
    ("--surface-evidence-invariance-weight", "0"),
    ("--family-temporal-delta-weight", "0"),
    ("--family-evidence-delta-weight", "0"),
    ("--family-no-evidence-transition-weight", "0"),
    ("--question-temporal-delta-weight", "0"),
    ("--question-evidence-delta-weight", "0"),
)

EVALUATION_METRIC_DIRECTIONS = {
    "evidence_sd_sdr_answerable": "↑",
    "evidence_sd_sdri_answerable": "↑",
    "evidence_si_sdr_answerable": "↑",
    "evidence_si_sdri_answerable": "↑",
    "weakest_role_sd_sdr_answerable": "↑",
    "weakest_role_si_sdr_answerable": "↑",
    "answerable_temporal_iou": "↑",
    "mean_no_evidence_retained_ratio": "↓",
    "maximum_mixture_consistency_l1": "↓",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "launch GPU training/evaluation; without this flag, perform only "
            "artifact validation and print the exact frozen command plan"
        ),
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help="Python executable containing the QCES/AudioSep dependencies",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="new dedicated output directory; existing paths are never overwritten",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_hashes(
    project_root: Path,
    artifacts: Sequence[BoundArtifact] = BOUND_ARTIFACTS,
) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for artifact in artifacts:
        path = (project_root / artifact.relative_path).resolve()
        if not path.is_file():
            raise ValueError(f"protocol-bound artifact is missing: {path}")
        actual = _sha256(path)
        if actual != artifact.sha256:
            raise ValueError(
                "protocol-bound artifact hash mismatch: "
                f"{path}; expected {artifact.sha256}, got {actual}"
            )
        identities.append(
            {
                "path": str(path),
                "sha256": actual,
                "size_bytes ↓": path.stat().st_size,
            }
        )
    return identities


def audit_manifest(
    path: Path,
    *,
    expected_split: str,
    expected_records: int,
    expected_families: int,
    expected_scenes: int,
) -> dict[str, Any]:
    records = 0
    families: set[str] = set()
    scenes: set[str] = set()
    source_groups: set[str] = set()
    family_variant_counts: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank manifest row: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"non-object manifest row: {path}:{line_number}")
            split = str(row.get("split", "")).lower()
            if split == "test":
                raise ValueError(f"test record is forbidden: {path}:{line_number}")
            if split != expected_split:
                raise ValueError(
                    f"unexpected split {split!r}: {path}:{line_number}; "
                    f"expected {expected_split!r}"
                )
            family = row.get("scene_family_id")
            scene = row.get("scene_id")
            variant = row.get("variant_id")
            groups = row.get("source_group_ids")
            if (
                not isinstance(family, str)
                or not isinstance(scene, str)
                or not isinstance(variant, str)
            ):
                raise ValueError(
                    f"missing family/scene/variant identifier: {path}:{line_number}"
                )
            if not isinstance(groups, list) or not all(
                isinstance(group, str) for group in groups
            ):
                raise ValueError(f"invalid source_group_ids: {path}:{line_number}")
            records += 1
            families.add(family)
            scenes.add(scene)
            source_groups.update(groups)
            variants = family_variant_counts.setdefault(family, {})
            variants[variant] = variants.get(variant, 0) + 1
    observed = (records, len(families), len(scenes))
    expected = (expected_records, expected_families, expected_scenes)
    if observed != expected:
        raise ValueError(
            f"manifest counts changed for {path}: expected {expected}, got {observed}"
        )
    required_variants = {"base", "order_swap", "anchor_drop"}
    incomplete = {
        family: counts
        for family, counts in family_variant_counts.items()
        if set(counts) != required_variants or len(set(counts.values())) != 1
    }
    if incomplete:
        raise ValueError(
            f"pilot contains incomplete scene families: {sorted(incomplete)}"
        )
    return {
        "path": str(path.resolve()),
        "split": expected_split,
        "records ↑": records,
        "families ↑": len(families),
        "scenes ↑": len(scenes),
        "source_groups": sorted(source_groups),
        "family_ids": sorted(families),
        "complete_scene_families ↑": True,
        "test_records_accessed ↓": 0,
    }


def _validate_selection_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "paper_result_eligible": False,
        "test_split_accessed": False,
    }
    for key, expected in required.items():
        if payload.get(key) is not expected:
            raise ValueError(f"selection receipt has {key}={payload.get(key)!r}")
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("selection receipt lacks selection metadata")
    if selection.get("seed") != SEED:
        raise ValueError("selection receipt seed changed")
    if selection.get("uses_identifiers_only") is not True:
        raise ValueError("pilot selection was not identifier-only")
    if selection.get("uses_labels_answers_targets_or_metrics") is not False:
        raise ValueError("pilot selection used forbidden supervision")
    return payload


def validate_protocol_artifacts(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    for relative in REQUIRED_UNHASHED_FILES:
        path = (project_root / relative).resolve()
        if not path.is_file():
            raise ValueError(f"required pilot file is missing: {path}")
    if not (project_root / AUDIOSEP_ROOT).resolve().is_dir():
        raise ValueError("AudioSep repository root is missing")

    hashes = validate_hashes(project_root)
    train = audit_manifest(
        (project_root / TRAIN_MANIFEST).resolve(),
        expected_split="train",
        expected_records=576,
        expected_families=12,
        expected_scenes=36,
    )
    val = audit_manifest(
        (project_root / VAL_MANIFEST).resolve(),
        expected_split="val",
        expected_records=288,
        expected_families=6,
        expected_scenes=18,
    )
    source_overlap = set(train.pop("source_groups")) & set(val.pop("source_groups"))
    if source_overlap:
        raise ValueError(
            "pilot train/validation source-group overlap: " f"{sorted(source_overlap)}"
        )
    family_overlap = set(train.pop("family_ids")) & set(val.pop("family_ids"))
    if family_overlap:
        raise ValueError(
            "pilot train/validation scene-family overlap: " f"{sorted(family_overlap)}"
        )
    selection = _validate_selection_receipt(
        (project_root / SELECTION_RECEIPT).resolve()
    )
    return {
        "bound_artifacts": hashes,
        "manifests": {"train": train, "validation": val},
        "selection_receipt": str((project_root / SELECTION_RECEIPT).resolve()),
        "identifier_only_selection ↑": True,
        "train_validation_scene_family_overlap ↓": 0,
        "train_validation_source_group_overlap ↓": 0,
        "test_manifests_opened ↓": 0,
        "paper_result_eligible": bool(selection["paper_result_eligible"]),
    }


def _absolute(project_root: Path, relative: Path) -> str:
    return str((project_root / relative).resolve())


def _append_options(command: list[str], options: Sequence[tuple[str, str]]) -> None:
    for flag, value in options:
        command.extend((flag, value))


def shared_train_options(project_root: Path) -> tuple[tuple[str, str], ...]:
    return (
        ("--manifest", _absolute(project_root, TRAIN_MANIFEST)),
        ("--val-manifest", _absolute(project_root, VAL_MANIFEST)),
        ("--backend", "audiosep"),
        ("--audiosep-root", _absolute(project_root, AUDIOSEP_ROOT)),
        ("--audiosep-config", _absolute(project_root, AUDIOSEP_CONFIG)),
        ("--audiosep-checkpoint", _absolute(project_root, AUDIOSEP_CHECKPOINT)),
        ("--foundation-feature-mode", "audiosep_clap"),
        ("--foundation-feature-cache", _absolute(project_root, FOUNDATION_TRAIN)),
        (
            "--val-foundation-feature-cache",
            _absolute(project_root, FOUNDATION_VAL),
        ),
        ("--temporal-role-mode", "independent_sigmoid"),
        ("--weakest-role-weight", "0.1"),
        ("--role-relative-weight", "0"),
        *COUNTERFACTUAL_ZERO_OPTIONS,
        ("--counterfactual-transition-margin", "0.25"),
        ("--epochs", "1"),
        ("--max-steps", "512"),
        ("--batch-size", "1"),
        ("--learning-rate", "0.0003"),
        ("--dropout", "0.1"),
        ("--crop-seconds", "10"),
        ("--selection-metric", "evidence_sd_sdr"),
        ("--early-stopping-patience", "0"),
        ("--selection-min-delta", "0"),
        ("--num-workers", "0"),
        ("--log-every-epochs", "1"),
        ("--seed", str(SEED)),
        ("--device", "cuda"),
        ("--precision", "fp32"),
    )


def candidate_train_options(
    project_root: Path, candidate: str
) -> tuple[tuple[str, str], ...]:
    if candidate == "union_single":
        return (
            ("--semantic-separation-mode", "union_single"),
            ("--semantic-targets", _absolute(project_root, UNION_SEMANTIC_TRAIN)),
            (
                "--val-semantic-targets",
                _absolute(project_root, UNION_SEMANTIC_VAL),
            ),
            ("--semantic-weight", "0.1"),
            ("--role-semantic-weight", "0"),
            ("--same-semantic-weight", "0"),
        )
    if candidate == "dual_role":
        return (
            ("--semantic-separation-mode", "dual_role"),
            (
                "--role-semantic-targets",
                _absolute(project_root, DUAL_SEMANTIC_TRAIN),
            ),
            (
                "--val-role-semantic-targets",
                _absolute(project_root, DUAL_SEMANTIC_VAL),
            ),
            ("--role-semantic-weight", "0.05"),
            ("--same-semantic-weight", "0.1"),
            ("--semantic-weight", "0"),
        )
    raise ValueError(f"unsupported candidate: {candidate}")


def build_train_command(
    project_root: Path,
    python: Path,
    output_dir: Path,
    candidate: str,
) -> list[str]:
    command = [
        str(python.resolve()),
        _absolute(project_root, TRAIN_SCRIPT),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    _append_options(command, shared_train_options(project_root))
    _append_options(command, candidate_train_options(project_root, candidate))
    command.extend(
        (
            "--no-separator-aware-refiner",
            "--no-save-every-epoch",
            "--deterministic",
        )
    )
    return command


def build_evaluate_command(
    project_root: Path,
    python: Path,
    train_output_dir: Path,
    eval_output_dir: Path,
) -> list[str]:
    return [
        str(python.resolve()),
        _absolute(project_root, EVALUATE_SCRIPT),
        "--checkpoint",
        str((train_output_dir / "checkpoint.pt").resolve()),
        "--audiosep-root",
        _absolute(project_root, AUDIOSEP_ROOT),
        "--audiosep-config",
        _absolute(project_root, AUDIOSEP_CONFIG),
        "--audiosep-checkpoint",
        _absolute(project_root, AUDIOSEP_CHECKPOINT),
        "--foundation-feature-cache",
        _absolute(project_root, FOUNDATION_VAL),
        "--manifest",
        _absolute(project_root, VAL_MANIFEST),
        "--output-dir",
        str(eval_output_dir.resolve()),
        "--batch-size",
        "1",
        "--threshold",
        "0.5",
        "--device",
        "cuda",
        "--no-render-audio",
    ]


def build_plan(
    project_root: Path,
    python: Path,
    results_root: Path,
    artifact_audit: Mapping[str, Any],
) -> dict[str, Any]:
    project_root = project_root.resolve()
    results_root = results_root.resolve()
    allowed_output_root = (project_root / "outputs").resolve()
    try:
        relative_output = results_root.relative_to(allowed_output_root)
    except ValueError as exc:
        raise ValueError(
            f"pilot results must stay under {allowed_output_root}: {results_root}"
        ) from exc
    if not relative_output.parts:
        raise ValueError("the shared outputs root cannot be a pilot results target")
    candidates: dict[str, Any] = {}
    for candidate in ("union_single", "dual_role"):
        train_dir = results_root / f"{candidate}_train"
        eval_dir = results_root / f"{candidate}_eval"
        train_command = build_train_command(project_root, python, train_dir, candidate)
        evaluate_command = build_evaluate_command(
            project_root, python, train_dir, eval_dir
        )
        if "--overwrite" in train_command or "--overwrite" in evaluate_command:
            raise RuntimeError("frozen pilot commands must never overwrite outputs")
        if Path(evaluate_command[evaluate_command.index("--manifest") + 1]).name != (
            VAL_MANIFEST.name
        ):
            raise RuntimeError("evaluation manifest drifted from frozen validation")
        candidates[candidate] = {
            "train_output_dir": str(train_dir),
            "evaluation_output_dir": str(eval_dir),
            "training_command": train_command,
            "training_command_shell": shlex.join(train_command),
            "evaluation_command": evaluate_command,
            "evaluation_command_shell": shlex.join(evaluate_command),
            "effective_separator_evaluations_per_record ↓": (
                1 if candidate == "union_single" else 2
            ),
        }
    return {
        "format": FORMAT,
        "purpose": "development-only union-versus-dual engineering screen",
        "paper_result_eligible": False,
        "project_root": str(project_root),
        "results_root": str(results_root),
        "seed": SEED,
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
        "execution_order": [
            "union_single training",
            "dual_role training",
            "common-initialization/fairness gate",
            "union_single report-only validation evaluation",
            "dual_role report-only validation evaluation",
            "comparison/promotion-gate summary",
        ],
        "environment_overrides": {"PYTHONHASHSEED": str(SEED)},
        "frozen_shared_training_options": dict(shared_train_options(project_root)),
        "candidate_specific_training_options": {
            candidate: dict(candidate_train_options(project_root, candidate))
            for candidate in ("union_single", "dual_role")
        },
        "counterfactual_evidence_equivariance_enabled": False,
        "temporal_refiner_enabled": False,
        "test_manifests_accessed ↓": 0,
        "artifact_audit": dict(artifact_audit),
        "candidates": candidates,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _receipt(summary: Mapping[str, Any], candidate: str) -> Mapping[str, Any]:
    receipt = summary.get("from_scratch_composer_initialization")
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{candidate} lacks from-scratch initialization receipt")
    return receipt


def validate_completed_training(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    union = summaries["union_single"]
    dual = summaries["dual_role"]
    for candidate, summary in summaries.items():
        training = summary.get("training_config")
        if not isinstance(training, Mapping):
            raise ValueError(f"{candidate} summary lacks training_config")
        expected = {
            "epochs": 1,
            "batch_size": 1,
            "max_steps": 512,
            "num_workers": 0,
            "seed": SEED,
            "precision": "fp32",
            "deterministic": True,
            "temporal_role_mode": "independent_sigmoid",
            "foundation_feature_mode": "audiosep_clap",
            "separator_aware_refiner": False,
            "counterfactual_enabled": False,
            "selection_metric": "evidence_sd_sdr",
        }
        changed = {
            key: (training.get(key), value)
            for key, value in expected.items()
            if training.get(key) != value
        }
        if changed:
            raise ValueError(f"{candidate} frozen training config drift: {changed}")
        if summary.get("global_step") != 512:
            raise ValueError(f"{candidate} did not complete exactly 512 updates")
        if summary.get("stop_reason") != "max_steps":
            raise ValueError(f"{candidate} did not stop at max_steps")
        if summary.get("manifest_sha256") != BOUND_ARTIFACTS[0].sha256:
            raise ValueError(f"{candidate} training manifest identity changed")
        if summary.get("val_manifest_sha256") != BOUND_ARTIFACTS[1].sha256:
            raise ValueError(f"{candidate} validation manifest identity changed")

    union_receipt = _receipt(union, "union_single")
    dual_receipt = _receipt(dual, "dual_role")
    union_aggregates = union_receipt.get("aggregates")
    dual_aggregates = dual_receipt.get("aggregates")
    if not isinstance(union_aggregates, Mapping) or not isinstance(
        dual_aggregates, Mapping
    ):
        raise ValueError("composer receipt lacks aggregate hashes")
    union_hash = union_aggregates.get("common_state_tensors_sha256")
    dual_hash = dual_aggregates.get("common_state_tensors_sha256")
    if not isinstance(union_hash, str) or union_hash != dual_hash:
        raise ValueError("common composer initialization hash mismatch")
    union_common = union_receipt.get("common_state_tensor_keys")
    dual_common = dual_receipt.get("common_state_tensor_keys")
    union_only = union_receipt.get("candidate_only_state_tensor_keys")
    dual_only = dual_receipt.get("candidate_only_state_tensor_keys")
    if not isinstance(union_common, list) or not isinstance(dual_common, list):
        raise ValueError("composer receipt lacks common tensor keys")
    if len(union_common) != EXPECTED_COMMON_COMPOSER_TENSORS:
        raise ValueError("union common composer tensor count changed")
    if union_common != dual_common:
        raise ValueError("union/dual common composer tensor keys differ")
    if union_only != []:
        raise ValueError("union unexpectedly has candidate-only tensors")
    if not isinstance(dual_only, list) or len(dual_only) != (
        EXPECTED_DUAL_ONLY_COMPOSER_TENSORS
    ):
        raise ValueError("dual candidate-only composer tensor count changed")
    if not all(str(key).startswith("same_semantic_head.") for key in dual_only):
        raise ValueError("dual has an undeclared candidate-only composer tensor")

    common_training_fields = (
        "epochs",
        "batch_size",
        "learning_rate",
        "dropout",
        "crop_seconds",
        "max_steps",
        "num_workers",
        "seed",
        "precision",
        "deterministic",
        "selection_metric",
        "foundation_feature_mode",
        "temporal_role_mode",
        "counterfactual_enabled",
        "separator_aware_refiner",
    )
    mismatches = [
        key
        for key in common_training_fields
        if union["training_config"].get(key) != dual["training_config"].get(key)
    ]
    if mismatches:
        raise ValueError(f"shared training settings differ: {mismatches}")
    return {
        "common_initialization_tensor_delta ↓": 0,
        "common_initialization_hash": union_hash,
        "common_initialization_tensors ↑": len(union_common),
        "dual_candidate_only_tensors ↓": len(dual_only),
        "shared_training_setting_mismatches ↓": 0,
        "optimizer_update_count_delta ↓": 0,
        "same_training_record_order_inputs ↑": True,
    }


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    return True


def _same_semantic_slice(report: Mapping[str, Any]) -> dict[str, Any]:
    items = report.get("items")
    if not isinstance(items, list):
        raise ValueError("evaluation report lacks item records")
    same = [
        item
        for item in items
        if isinstance(item, Mapping)
        and item.get("same_role_label") is True
        and item.get("no_evidence") is False
    ]
    if not same:
        raise ValueError("pilot validation lacks answerable same-semantic records")
    sd_sdr = [float(item["evidence_sd_sdr"]) for item in same]
    temporal_iou = [float(item["temporal_iou"]) for item in same]
    return {
        "records ↑": len(same),
        "evidence_sd_sdr ↑": mean(sd_sdr),
        "answerable_temporal_iou ↑": mean(temporal_iou),
    }


def _selected_training_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    metrics = summary.get("best_validation_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("training summary lacks best validation metrics")
    requested = {
        "evidence_sd_sdr ↑": "evidence_sd_sdr",
        "answerable_temporal_iou ↑": "answerable_temporal_iou",
        "no_evidence_retained_ratio ↓": "no_evidence_retained_ratio",
        "weakest_role_waveform_error ↓": "weakest_role_waveform",
    }
    selected: dict[str, float] = {}
    for display, key in requested.items():
        value = metrics.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"missing/non-finite validation metric: {key}")
        selected[display] = float(value)
    for display, key in (
        ("same_semantic_brier ↓", "same_semantic_brier"),
        ("same_semantic_accuracy ↑", "same_semantic_accuracy"),
    ):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            selected[display] = float(value)
    return selected


def _evaluation_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    summary = report.get("summary")
    directed = report.get("summary_with_directions")
    if not isinstance(summary, Mapping) or not isinstance(directed, Mapping):
        raise ValueError("evaluation report lacks directed summary")
    result: dict[str, float] = {}
    for key, arrow in EVALUATION_METRIC_DIRECTIONS.items():
        value = summary.get(key)
        directed_key = f"{key}_{arrow}"
        if directed_key not in directed:
            raise ValueError(f"evaluation metric lacks direction: {key}")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"evaluation metric is missing/non-finite: {key}")
        result[f"{key} {arrow}"] = float(value)
    return result


def _directed_resources(resources: Mapping[str, Any]) -> dict[str, Any]:
    """Rename trainer resource scalars so their direction is visible."""

    mapping = {
        "wall_seconds_down": "wall_seconds ↓",
        "startup_seconds_down": "startup_seconds ↓",
        "training_seconds_down": "training_seconds ↓",
        "validation_seconds_down": "validation_seconds ↓",
        "optimizer_steps_per_second_up": "optimizer_steps_per_second ↑",
        "optimizer_steps_per_training_second_up": (
            "optimizer_steps_per_training_second ↑"
        ),
        "cuda_peak_allocated_bytes_down": "cuda_peak_allocated_bytes ↓",
        "cuda_peak_reserved_bytes_down": "cuda_peak_reserved_bytes ↓",
    }
    directed = {
        "scope": resources.get("scope"),
        "optimizer_steps (fixed by protocol)": resources.get("optimizer_steps"),
    }
    for source, display in mapping.items():
        if source not in resources:
            raise ValueError(f"training resource metric is missing: {source}")
        directed[display] = resources[source]
    return directed


def build_comparison_summary(
    plan: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    reports: Mapping[str, Mapping[str, Any]],
    fairness: Mapping[str, Any],
    wav_file_count: int,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    slices: dict[str, dict[str, Any]] = {}
    for candidate in ("union_single", "dual_role"):
        report = reports[candidate]
        if report.get("records") != 288:
            raise ValueError(f"{candidate} evaluation did not cover 288 records")
        if not _finite_numbers(report):
            raise ValueError(f"{candidate} evaluation contains non-finite numbers")
        slices[candidate] = _same_semantic_slice(report)
        run_resources = summaries[candidate].get("run_resources")
        if not isinstance(run_resources, Mapping):
            raise ValueError(f"{candidate} training summary lacks run resources")
        candidates[candidate] = {
            "effective_separator_evaluations_per_record ↓": (
                1 if candidate == "union_single" else 2
            ),
            "selected_training_validation_metrics": _selected_training_metrics(
                summaries[candidate]
            ),
            "report_only_evaluation_metrics": _evaluation_metrics(report),
            "same_semantic_answerable_slice": slices[candidate],
            "training_resources_with_directions": _directed_resources(run_resources),
        }

    union_eval = reports["union_single"]["summary"]
    dual_eval = reports["dual_role"]["summary"]
    union_train = summaries["union_single"]["best_validation_metrics"]
    dual_train = summaries["dual_role"]["best_validation_metrics"]
    evidence_delta = float(dual_eval["evidence_sd_sdr_answerable"]) - float(
        union_eval["evidence_sd_sdr_answerable"]
    )
    temporal_delta = float(dual_eval["answerable_temporal_iou"]) - float(
        union_eval["answerable_temporal_iou"]
    )
    retention_delta = float(dual_eval["mean_no_evidence_retained_ratio"]) - float(
        union_eval["mean_no_evidence_retained_ratio"]
    )
    weakest_error_delta = float(dual_train["weakest_role_waveform"]) - float(
        union_train["weakest_role_waveform"]
    )
    same_evidence_delta = float(slices["dual_role"]["evidence_sd_sdr ↑"]) - float(
        slices["union_single"]["evidence_sd_sdr ↑"]
    )
    same_temporal_delta = float(
        slices["dual_role"]["answerable_temporal_iou ↑"]
    ) - float(slices["union_single"]["answerable_temporal_iou ↑"])

    non_regression = (
        evidence_delta >= -0.25
        and temporal_delta >= -0.01
        and retention_delta <= 0.01
        and weakest_error_delta <= 0.01
    )
    evidence_gain = evidence_delta >= 0.50
    temporal_gain = temporal_delta >= 0.02
    material_gain = evidence_gain or temporal_gain
    triggered_non_reversals: list[bool] = []
    if evidence_gain:
        triggered_non_reversals.append(same_evidence_delta >= 0.0)
    if temporal_gain:
        triggered_non_reversals.append(same_temporal_delta >= 0.0)
    same_non_reversal = bool(triggered_non_reversals) and all(triggered_non_reversals)

    mixture_consistency = max(
        float(union_eval["maximum_mixture_consistency_l1"]),
        float(dual_eval["maximum_mixture_consistency_l1"]),
    )
    hard_valid = (
        mixture_consistency <= 1e-6
        and wav_file_count == 0
        and fairness.get("common_initialization_tensor_delta ↓") == 0
        and fairness.get("shared_training_setting_mismatches ↓") == 0
    )
    return {
        "format": SUMMARY_FORMAT,
        "purpose": "development-only GO/NO-GO screen; not a paper result",
        "paper_result_eligible": False,
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
            "dual_minus_union": (
                "positive favors dual for ↑ metrics; negative favors dual for ↓ metrics"
            ),
        },
        "frozen_plan_format": plan.get("format"),
        "test_manifests_accessed ↓": 0,
        "fairness_and_validity": {
            **dict(fairness),
            "maximum_mixture_consistency_l1 ↓": mixture_consistency,
            "rendered_wav_files ↓": wav_file_count,
            "all_hard_validity_gates_pass ↑": hard_valid,
        },
        "candidates": candidates,
        "dual_minus_union": {
            "evidence_sd_sdr_answerable_delta_dB ↑": evidence_delta,
            "answerable_temporal_iou_delta ↑": temporal_delta,
            "mean_no_evidence_retained_ratio_delta ↓": retention_delta,
            "weakest_role_waveform_error_delta ↓": weakest_error_delta,
            "same_semantic_evidence_sd_sdr_delta_dB ↑": same_evidence_delta,
            "same_semantic_temporal_iou_delta ↑": same_temporal_delta,
        },
        "predeclared_promotion_gate": {
            "condition_1_non_regression_pass ↑": non_regression,
            "condition_2_material_gain_pass ↑": material_gain,
            "condition_2_evidence_gain_at_least_0.50_dB ↑": evidence_gain,
            "condition_2_temporal_gain_at_least_0.02 ↑": temporal_gain,
            "condition_3_same_semantic_non_reversal_pass ↑": same_non_reversal,
            "condition_3_interpretation": (
                "each aggregate metric that clears its material-gain threshold "
                "must have a non-negative dual-minus-union gain on the "
                "answerable same-semantic slice"
            ),
            "dual_authorized_for_full_validation ↑": (
                hard_valid and non_regression and material_gain and same_non_reversal
            ),
            "passing_scope": (
                "authorization for the full-validation comparison only; no "
                "pilot number is eligible for the paper table"
            ),
        },
    }


def _exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _run(command: Sequence[str], *, project_root: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(SEED)
    print(f"running: {shlex.join(command)}", flush=True)
    subprocess.run(
        list(command),
        cwd=project_root,
        env=environment,
        check=True,
    )


def execute_plan(plan: Mapping[str, Any], project_root: Path) -> Path:
    results_root = Path(str(plan["results_root"]))
    if results_root.exists():
        raise ValueError(
            f"results root already exists; refusing any overwrite: {results_root}"
        )
    results_root.mkdir(parents=True, exist_ok=False)
    _exclusive_write_json(results_root / "frozen_run_plan.json", plan)

    candidate_plans = plan["candidates"]
    for candidate in ("union_single", "dual_role"):
        _run(candidate_plans[candidate]["training_command"], project_root=project_root)

    summaries = {
        candidate: _load_json(
            Path(candidate_plans[candidate]["train_output_dir"]) / "summary.json"
        )
        for candidate in ("union_single", "dual_role")
    }
    fairness = validate_completed_training(summaries)

    for candidate in ("union_single", "dual_role"):
        _run(
            candidate_plans[candidate]["evaluation_command"], project_root=project_root
        )
    reports = {
        candidate: _load_json(
            Path(candidate_plans[candidate]["evaluation_output_dir"])
            / "evaluation_report.json"
        )
        for candidate in ("union_single", "dual_role")
    }
    wav_count = sum(
        1
        for candidate in ("union_single", "dual_role")
        for _ in Path(candidate_plans[candidate]["evaluation_output_dir"]).rglob(
            "*.wav"
        )
    )
    summary = build_comparison_summary(plan, summaries, reports, fairness, wav_count)
    summary_path = results_root / "pilot_comparison_summary.json"
    _exclusive_write_json(summary_path, summary)
    print(
        json.dumps(summary["predeclared_promotion_gate"], indent=2, sort_keys=True),
        flush=True,
    )
    print(f"wrote {summary_path}", flush=True)
    return summary_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = PROJECT_ROOT.resolve()
    python = args.python.expanduser().resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise SystemExit(f"Python executable not found or not executable: {python}")
    try:
        artifact_audit = validate_protocol_artifacts(project_root)
        results_root = (
            args.results_root
            if args.results_root.is_absolute()
            else project_root / args.results_root
        ).resolve()
        plan = build_plan(project_root, python, results_root, artifact_audit)
        if args.execute:
            execute_plan(plan, project_root)
        else:
            print(json.dumps(plan, indent=2, sort_keys=True))
            print(
                "validation-only: no training/evaluation process was launched; "
                "pass --execute to run the frozen plan",
                flush=True,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
