#!/usr/bin/env python3
"""Train QCES with optional held-out validation and checkpoint selection."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    DUAL_ROLE_SEMANTIC_MODE,
    FOUNDATION_FEATURE_MODES,
    FOUNDATION_SEMANTIC_MIXING_MODES,
    LEGACY_TEMPORAL_ROLE_MODE,
    LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
    NO_FOUNDATION_FEATURES,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    QUESTION_RESIDUAL_SEMANTIC_MIXING,
    QCESConfig,
    SEMANTIC_SEPARATION_MODES,
    SEPARATOR_AWARE_REFINER_MODES,
    TEMPORAL_ROLE_MODES,
    UNION_SINGLE_SEMANTIC_MODE,
)
from mixi_understanding.qces.counterfactual import (
    COUNTERFACTUAL_METRIC_DIRECTIONS,
    FAMILY_LOSS_NAMES,
    QUESTION_LOSS_NAMES,
    SURFACE_LOSS_NAMES,
    CounterfactualBatchSampler,
    build_counterfactual_group_plan,
    counterfactual_objectives,
)
from mixi_understanding.qces.data import QCESManifestDataset, collate_qces
from mixi_understanding.qces.losses import LossWeights, QCESLoss
from mixi_understanding.qces.metrics import qces_metrics
from mixi_understanding.qces.model import QCESModel
from mixi_understanding.qces.separators import (
    AudioSepConditionedAdapter,
    PhaseAwareComplexMaskSeparator,
)
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.cache_audiosep_clap_features import (
    CANONICAL_DURATION_SECONDS as CLAP_CACHE_DURATION_SECONDS,
    CANONICAL_NUM_SAMPLES as CLAP_CACHE_NUM_SAMPLES,
    CANONICAL_SAMPLE_RATE as CLAP_CACHE_SAMPLE_RATE,
    EFFECTIVE_FRAMES as CLAP_CACHE_EFFECTIVE_FRAMES,
    EXPECTED_QUERY_STATE_TENSORS,
    FEATURE_SPACE as CLAP_CACHE_FEATURE_SPACE,
    FINE_DIM as CLAP_CACHE_FINE_DIM,
    JOINT_DIM as CLAP_CACHE_JOINT_DIM,
    QUESTION_CACHE_FORMAT,
    RAW_FINE_FRAMES as CLAP_CACHE_RAW_FRAMES,
    RECEIPT_FORMAT as FOUNDATION_CACHE_RECEIPT_FORMAT,
    REPEAT_RATIO as CLAP_CACHE_REPEAT_RATIO,
    SCENE_CACHE_FORMAT,
    source_tree_identity as audiosep_source_tree_identity,
    validate_question_payload,
    validate_scene_payload,
)


SELECTION_DIRECTIONS = {
    "total": "minimize",
    "evidence_l1": "minimize",
    "evidence_si_sdr": "maximize",
    "evidence_sd_sdr": "maximize",
    "temporal_iou": "maximize",
    "answerable_temporal_iou": "maximize",
    "no_evidence_retained_ratio": "minimize",
    "role_relative_waveform": "minimize",
    "weakest_role_waveform": "minimize",
    "same_semantic_brier": "minimize",
    "same_semantic_accuracy": "maximize",
}
SELECTION_DIRECTIONS.update(
    {
        name: metadata["direction"]
        for name, metadata in COUNTERFACTUAL_METRIC_DIRECTIONS.items()
    }
)

ANSWERABLE_NORMALIZED_METRICS = {
    "evidence_active_waveform",
    "role_relative_waveform",
    "weakest_role_waveform",
    "minimality",
    "semantic_alignment",
    "anchor_semantic_alignment",
    "answer_semantic_alignment",
    "same_semantic_classification",
    "same_semantic_brier",
    "same_semantic_accuracy",
    "evidence_si_sdr",
    "evidence_sd_sdr",
    "answerable_temporal_iou",
}

SAME_SEMANTIC_NORMALIZED_METRICS = {
    "same_semantic_probability_on_same",
}

DIFFERENT_SEMANTIC_NORMALIZED_METRICS = {
    "same_semantic_probability_on_different",
}

NO_EVIDENCE_NORMALIZED_METRICS = {
    "no_evidence_retained_ratio",
}

DIAGNOSTIC_BEST_DIRECTIONS = {
    "evidence_si_sdr": "maximize",
    "evidence_sd_sdr": "maximize",
    "answerable_temporal_iou": "maximize",
    "no_evidence_retained_ratio": "minimize",
    "weakest_role_waveform": "minimize",
}

SEMANTIC_CACHE_FORMAT = "qces_audiosep_semantic_targets_v1"
SEMANTIC_TARGET_SCOPE = "training_supervision_only"
SEMANTIC_PROMPT_SOURCE = "evidence_role_event_labels_or_absent_label"
V5_UNION_SEMANTIC_CACHE_FORMAT = "qces_audiosep_union_semantic_targets_v2"
V5_UNION_SEMANTIC_TARGET_SCOPE = "training_supervision_only"
V5_UNION_SEMANTIC_PROMPT_SOURCE = "unique_anchor_answer_role_labels_in_timeline_order"
ROLE_SEMANTIC_CACHE_FORMAT = "qces_audiosep_role_semantic_targets_v1"
ROLE_SEMANTIC_TARGET_SCOPE = "training_supervision_only"
ROLE_SEMANTIC_PROMPT_SOURCE = "anchor_and_answer_role_event_labels"

FP32_PRECISION = "fp32"
AMP_FP16_PRECISION = "amp_fp16"
PRECISION_MODES = (FP32_PRECISION, AMP_FP16_PRECISION)
CORE_LOSS_WEIGHT_ARGS = {
    "evidence_waveform": "evidence_waveform_weight",
    "evidence_active_waveform": "evidence_active_waveform_weight",
    "separator_mask": "separator_mask_weight",
    "raw_separator_mask": "raw_separator_mask_weight",
    "residual_waveform": "residual_waveform_weight",
    "multi_resolution_stft": "multi_resolution_stft_weight",
    "reconstruction": "reconstruction_weight",
    "temporal_cross_entropy": "temporal_cross_entropy_weight",
    "temporal_binary_cross_entropy": "temporal_binary_cross_entropy_weight",
    "temporal_dice": "temporal_dice_weight",
    "temporal_role_dice": "temporal_role_dice_weight",
    "no_evidence": "no_evidence_weight",
    "minimality": "minimality_weight",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--val-manifest",
        type=Path,
        help="held-out manifest evaluated deterministically after every epoch",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("mask", "complex", "audiosep"), default="mask"
    )
    parser.add_argument("--audiosep-root", type=Path)
    parser.add_argument("--audiosep-config", type=Path)
    parser.add_argument("--audiosep-checkpoint", type=Path)
    parser.add_argument(
        "--foundation-feature-mode",
        choices=FOUNDATION_FEATURE_MODES,
        default=NO_FOUNDATION_FEATURES,
        help=(
            "none preserves legacy training; audiosep_clap consumes strict "
            "offline frozen-CLAP caches; infer_qces reproduces the same "
            "features online for canonical 10-second clips"
        ),
    )
    parser.add_argument(
        "--foundation-feature-cache",
        type=Path,
        help="training-split cache directory from cache_audiosep_clap_features.py",
    )
    parser.add_argument(
        "--foundation-semantic-mixing-mode",
        choices=FOUNDATION_SEMANTIC_MIXING_MODES,
        default=LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
        help=(
            "bounded_residual preserves old checkpoints; convex_interpolation "
            "lets the learned AudioSep condition reach its pure endpoint; "
            "question_residual starts from full-question CLAP and learns an "
            "unbounded audio-question correction"
        ),
    )
    parser.add_argument(
        "--freeze-semantic-adapter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "freeze the exactly zero-initialized question_residual semantic "
            "adapter so AudioSep receives the normalized full-question CLAP "
            "embedding directly; this is a receipt-bound semantic-composer "
            "ablation, not a general fine-tuning option (default: off)"
        ),
    )
    parser.add_argument(
        "--val-foundation-feature-cache",
        type=Path,
        help="validation-split frozen-CLAP cache directory",
    )
    parser.add_argument(
        "--temporal-role-mode",
        choices=TEMPORAL_ROLE_MODES,
        default=LEGACY_TEMPORAL_ROLE_MODE,
        help=(
            "exclusive_softmax preserves historical checkpoints; "
            "independent_sigmoid permits overlapping anchor/answer frames and "
            "is required for paper-scale QCES-v5 runs"
        ),
    )
    parser.add_argument(
        "--semantic-separation-mode",
        choices=SEMANTIC_SEPARATION_MODES,
        default=UNION_SINGLE_SEMANTIC_MODE,
        help=(
            "union_single preserves the historical one-condition/one-call "
            "path; dual_role predicts and renders separate anchor/answer "
            "conditions through one frozen AudioSep backbone"
        ),
    )
    parser.add_argument(
        "--separator-aware-refiner",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "refine temporal roles after one frozen AudioSep call using the raw "
            "separated stem (AudioSep backend only; default: off)"
        ),
    )
    parser.add_argument(
        "--separator-aware-refiner-mode",
        choices=SEPARATOR_AWARE_REFINER_MODES,
        default="full",
        help=(
            "full uses base frame features plus raw-stem spectrogram/energy; "
            "relative_energy is the low-capacity role-logit + relative-RMS "
            "ablation (requires --separator-aware-refiner; default: full)"
        ),
    )
    parser.add_argument(
        "--init-audiosep-qces-checkpoint",
        type=Path,
        help=(
            "strictly warm-start only the composer from an existing AudioSep "
            "QCES checkpoint bound to the same config and AudioSep assets"
        ),
    )
    parser.add_argument(
        "--freeze-base-composer-steps",
        type=int,
        default=0,
        help=(
            "optimizer updates that train only the separator-aware refiner before "
            "unfreezing the warm-started composer"
        ),
    )
    parser.add_argument("--semantic-targets", type=Path)
    parser.add_argument(
        "--val-semantic-targets",
        type=Path,
        help="semantic-target cache built specifically for --val-manifest",
    )
    parser.add_argument("--semantic-weight", type=float, default=0.0)
    default_loss_weights = LossWeights()
    for field_name, argument_name in CORE_LOSS_WEIGHT_ARGS.items():
        default_weight = getattr(default_loss_weights, field_name)
        parser.add_argument(
            "--" + argument_name.replace("_", "-"),
            type=float,
            default=default_weight,
            help=(
                f"weight for {field_name}; exposed for receipt-bound ablations "
                f"(default: {default_weight})"
            ),
        )
    parser.add_argument(
        "--role-semantic-targets",
        type=Path,
        help=(
            "training-only dual-role CLAP target cache bound to the manifest "
            "and exact frozen AudioSep checkpoint"
        ),
    )
    parser.add_argument(
        "--val-role-semantic-targets",
        type=Path,
        help="dual-role semantic cache built specifically for --val-manifest",
    )
    parser.add_argument("--role-semantic-weight", type=float, default=0.0)
    parser.add_argument("--same-semantic-weight", type=float, default=0.0)
    parser.add_argument("--role-relative-weight", type=float, default=0.0)
    parser.add_argument("--weakest-role-weight", type=float, default=0.0)
    parser.add_argument("--surface-semantic-invariance-weight", type=float, default=0.0)
    parser.add_argument("--surface-role-invariance-weight", type=float, default=0.0)
    parser.add_argument(
        "--surface-no-evidence-invariance-weight", type=float, default=0.0
    )
    parser.add_argument("--surface-evidence-invariance-weight", type=float, default=0.0)
    parser.add_argument("--family-temporal-delta-weight", type=float, default=0.0)
    parser.add_argument("--family-evidence-delta-weight", type=float, default=0.0)
    parser.add_argument(
        "--family-no-evidence-transition-weight", type=float, default=0.0
    )
    parser.add_argument("--question-temporal-delta-weight", type=float, default=0.0)
    parser.add_argument("--question-evidence-delta-weight", type=float, default=0.0)
    parser.add_argument(
        "--counterfactual-transition-margin",
        type=float,
        default=0.25,
        help="minimum no-evidence probability gap for anchor-drop vs base/swap",
    )
    parser.add_argument(
        "--force-counterfactual-batching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "pack complete surface/family/question groups even when every paired "
            "loss weight is zero; this is the schedule-matched CEE-off control"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--crop-seconds", type=float, default=4.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every-epochs", type=int, default=1)
    parser.add_argument(
        "--selection-metric",
        choices=tuple(SELECTION_DIRECTIONS),
        default="total",
        help="held-out validation metric used for best-checkpoint selection",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="epochs without validation improvement; 0 disables early stopping",
    )
    parser.add_argument(
        "--selection-min-delta",
        type=float,
        default=0.0,
        help="minimum absolute validation improvement required to reset patience",
    )
    parser.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "retain an immutable checkpoint and exact metric snapshot after "
            "every evaluated epoch for post-hoc frozen Pareto selection "
            "(default: off)"
        ),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        choices=PRECISION_MODES,
        default=FP32_PRECISION,
        help=(
            "fp32 preserves the historical training/evaluation path; "
            "amp_fp16 enables CUDA float16 autocast with dynamic gradient "
            "scaling while keeping loss and metric reductions in float32"
        ),
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "enable strict deterministic PyTorch/CUDA execution (default: on); "
            "use --no-deterministic only when an unsupported deterministic "
            "kernel prevents training"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def validate_precision_mode(precision: str, device: torch.device) -> None:
    """Reject unsupported precision/device combinations before model loading."""

    if precision not in PRECISION_MODES:
        raise SystemExit(
            f"precision must be one of {PRECISION_MODES}, got {precision!r}"
        )
    if precision == AMP_FP16_PRECISION and device.type != "cuda":
        raise SystemExit("--precision amp_fp16 requires a CUDA device")


def move_tensors(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


@dataclass(frozen=True)
class FoundationFeatureCache:
    """Inference-safe frozen features and their manifest-only ID routing."""

    question_features: Mapping[str, torch.Tensor]
    scene_features: Mapping[str, torch.Tensor]
    sample_to_scene: Mapping[str, str]
    identity: Mapping[str, Any]


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_exact_keys(
    value: Any,
    expected: set[str],
    description: str,
    split_name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise SystemExit(
            f"{split_name} foundation cache has invalid {description} keys: "
            f"{actual}"
        )
    return value


def _aggregate_foundation_mixture_receipt(
    identities: Sequence[Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    for identity in identities:
        for key in ("scene_id", "manifest_path", "sha256", "size_bytes"):
            encoded = str(identity[key]).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def load_foundation_feature_cache(
    cache_dir: Path,
    manifest: Path,
    records: Sequence[object],
    split_name: str,
    *,
    audiosep_checkpoint_identity: Mapping[str, Any],
    audiosep_source_identity: Mapping[str, Any],
) -> FoundationFeatureCache:
    """Load a strict label-free cache bound to exact AudioSep assets and IDs."""

    resolved = cache_dir.resolve()
    if not resolved.is_dir():
        raise SystemExit(
            f"{split_name} foundation-feature cache directory not found: {resolved}"
        )
    receipt_path = resolved / "cache_receipt.json"
    question_path = resolved / "question_features.pt"
    scene_path = resolved / "scene_audio_features.pt"
    if not all(path.is_file() for path in (receipt_path, question_path, scene_path)):
        raise SystemExit(
            f"{split_name} foundation cache must contain cache_receipt.json, "
            "question_features.pt, and scene_audio_features.pt"
        )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"could not read {split_name} foundation cache receipt"
        ) from exc
    top_level = _require_exact_keys(
        receipt,
        {
            "format",
            "purpose",
            "manifest",
            "schema_versions",
            "audiosep_checkpoint",
            "audiosep_source_tree",
            "query_encoder_state",
            "canonical_audio",
            "features",
            "counts",
            "sample_ids",
            "scene_ids",
            "mixtures",
            "mixtures_aggregate_sha256",
            "execution",
            "privacy_contract",
            "artifacts",
        },
        "receipt",
        split_name,
    )
    if top_level["format"] != FOUNDATION_CACHE_RECEIPT_FORMAT:
        raise SystemExit(f"invalid {split_name} foundation cache receipt format")
    if top_level["purpose"] != "frozen_controller_inputs_without_oracle_labels":
        raise SystemExit(f"invalid {split_name} foundation cache purpose")

    expected_manifest = file_identity(manifest)
    declared_manifest = _require_exact_keys(
        top_level["manifest"],
        {"path", "sha256", "size_bytes"},
        "manifest identity",
        split_name,
    )
    if (
        declared_manifest["sha256"] != expected_manifest["sha256"]
        or declared_manifest["size_bytes"] != expected_manifest["size_bytes"]
    ):
        raise SystemExit(f"{split_name} foundation cache does not match its manifest")

    declared_checkpoint = _require_exact_keys(
        top_level["audiosep_checkpoint"],
        {"path", "sha256", "size_bytes"},
        "AudioSep checkpoint identity",
        split_name,
    )
    if declared_checkpoint["sha256"] != audiosep_checkpoint_identity.get(
        "sha256"
    ) or declared_checkpoint["size_bytes"] != audiosep_checkpoint_identity.get(
        "size_bytes"
    ):
        raise SystemExit(
            f"{split_name} foundation cache uses a different AudioSep checkpoint"
        )
    declared_source = _require_exact_keys(
        top_level["audiosep_source_tree"],
        {"path", "sha256", "hashed_file_count", "included_suffixes"},
        "AudioSep source identity",
        split_name,
    )
    if (
        declared_source["sha256"] != audiosep_source_identity.get("sha256")
        or declared_source["hashed_file_count"]
        != audiosep_source_identity.get("hashed_file_count")
        or declared_source["included_suffixes"]
        != audiosep_source_identity.get("included_suffixes")
    ):
        raise SystemExit(
            f"{split_name} foundation cache uses a different AudioSep source tree"
        )

    sample_to_scene: Dict[str, str] = {}
    scene_to_mixture_path: Dict[str, str] = {}
    schema_versions: set[str] = set()
    for record in records:
        sample_id = getattr(record, "sample_id", None)
        scene_id = getattr(record, "scene_id", None)
        schema_version = getattr(record, "schema_version", None)
        mixture_path = getattr(record, "mixture_path", None)
        if not isinstance(sample_id, str) or not sample_id:
            raise SystemExit(f"{split_name} record lacks a valid sample ID")
        if not isinstance(scene_id, str) or not scene_id:
            raise SystemExit(
                f"{split_name} audiosep_clap mode requires scene_id: {sample_id}"
            )
        if not isinstance(schema_version, str) or not schema_version:
            raise SystemExit(f"{split_name} record lacks a schema version: {sample_id}")
        if not isinstance(mixture_path, str) or not mixture_path:
            raise SystemExit(f"{split_name} record lacks a mixture path: {sample_id}")
        if sample_id in sample_to_scene:
            raise SystemExit(f"{split_name} manifest has duplicate ID: {sample_id}")
        sample_to_scene[sample_id] = scene_id
        existing_mixture = scene_to_mixture_path.get(scene_id)
        if existing_mixture is not None and existing_mixture != mixture_path:
            raise SystemExit(
                f"{split_name} scene has inconsistent mixture paths: {scene_id}"
            )
        scene_to_mixture_path[scene_id] = mixture_path
        schema_versions.add(schema_version)
    expected_sample_ids = sorted(sample_to_scene)
    expected_scene_ids = sorted(set(sample_to_scene.values()))
    if top_level["schema_versions"] != sorted(schema_versions):
        raise SystemExit(
            f"{split_name} foundation cache schema versions differ from manifest"
        )
    if top_level["sample_ids"] != expected_sample_ids:
        raise SystemExit(
            f"{split_name} foundation cache sample IDs differ from manifest"
        )
    if top_level["scene_ids"] != expected_scene_ids:
        raise SystemExit(
            f"{split_name} foundation cache scene IDs differ from manifest"
        )

    canonical = _require_exact_keys(
        top_level["canonical_audio"],
        {
            "sample_rate",
            "num_samples",
            "duration_seconds",
            "num_channels",
            "clap_sample_rate",
            "clap_num_samples",
        },
        "canonical audio",
        split_name,
    )
    expected_canonical = {
        "sample_rate": CLAP_CACHE_SAMPLE_RATE,
        "num_samples": CLAP_CACHE_NUM_SAMPLES,
        "duration_seconds": CLAP_CACHE_DURATION_SECONDS,
        "num_channels": 1,
        "clap_sample_rate": 48_000,
        "clap_num_samples": 480_000,
    }
    if dict(canonical) != expected_canonical:
        raise SystemExit(
            f"{split_name} foundation cache is not canonical 10 s / 32 kHz"
        )
    features = _require_exact_keys(
        top_level["features"],
        {
            "feature_space",
            "question_shape",
            "question_dtype",
            "raw_audio_shape",
            "effective_audio_shape",
            "effective_audio_dtype",
            "raw_to_effective_repeat_ratio",
            "audio_normalization",
        },
        "feature declaration",
        split_name,
    )
    expected_features = {
        "feature_space": CLAP_CACHE_FEATURE_SPACE,
        "question_shape": [CLAP_CACHE_JOINT_DIM],
        "question_dtype": "float32",
        "raw_audio_shape": [CLAP_CACHE_RAW_FRAMES, CLAP_CACHE_FINE_DIM],
        "effective_audio_shape": [
            CLAP_CACHE_EFFECTIVE_FRAMES,
            CLAP_CACHE_JOINT_DIM,
        ],
        "effective_audio_dtype": "float16",
        "raw_to_effective_repeat_ratio": CLAP_CACHE_REPEAT_RATIO,
        "audio_normalization": "per_frame_l2_before_fp16_storage",
    }
    if dict(features) != expected_features:
        raise SystemExit(f"{split_name} foundation cache feature declaration drift")

    query_state = _require_exact_keys(
        top_level["query_encoder_state"],
        {
            "extracted_tensor_count",
            "loaded_tensor_count",
            "effective_missing_keys",
            "effective_unexpected_keys",
            "ignored_nonpersistent_checkpoint_keys",
            "fusion_enabled",
        },
        "query encoder state",
        split_name,
    )
    if (
        query_state["extracted_tensor_count"] != EXPECTED_QUERY_STATE_TENSORS
        or not isinstance(query_state["loaded_tensor_count"], int)
        or query_state["loaded_tensor_count"] <= 0
        or query_state["effective_missing_keys"] != []
        or query_state["effective_unexpected_keys"] != []
        or not isinstance(query_state["ignored_nonpersistent_checkpoint_keys"], list)
        or any(
            not isinstance(key, str) or not key.endswith("embeddings.position_ids")
            for key in query_state["ignored_nonpersistent_checkpoint_keys"]
        )
        or query_state["fusion_enabled"] is not False
    ):
        raise SystemExit(
            f"{split_name} foundation cache has incompatible query-encoder state"
        )
    privacy = _require_exact_keys(
        top_level["privacy_contract"],
        {
            "question_text_stored",
            "event_labels_stored",
            "answer_labels_stored",
            "evidence_annotations_stored",
            "allowed_keys",
        },
        "privacy contract",
        split_name,
    )
    if dict(privacy) != {
        "question_text_stored": False,
        "event_labels_stored": False,
        "answer_labels_stored": False,
        "evidence_annotations_stored": False,
        "allowed_keys": "sample_id_and_scene_id_only",
    }:
        raise SystemExit(f"{split_name} foundation cache violates privacy contract")

    counts = _require_exact_keys(
        top_level["counts"],
        {
            "sample_ids",
            "scene_ids",
            "unique_full_question_texts",
            "physical_mixture_encodes",
        },
        "counts",
        split_name,
    )
    if (
        counts["sample_ids"] != len(expected_sample_ids)
        or counts["scene_ids"] != len(expected_scene_ids)
        or counts["physical_mixture_encodes"] != len(expected_scene_ids)
        or not isinstance(counts["unique_full_question_texts"], int)
        or not 1 <= counts["unique_full_question_texts"] <= len(expected_sample_ids)
    ):
        raise SystemExit(f"{split_name} foundation cache counts are inconsistent")

    mixture_rows = top_level["mixtures"]
    if not isinstance(mixture_rows, list) or len(mixture_rows) != len(
        expected_scene_ids
    ):
        raise SystemExit(f"{split_name} foundation cache mixture rows are invalid")
    validated_mixtures: list[Mapping[str, Any]] = []
    for row in mixture_rows:
        mixture = _require_exact_keys(
            row,
            {
                "scene_id",
                "manifest_path",
                "sha256",
                "size_bytes",
                "sample_rate",
                "num_samples",
                "duration_seconds",
                "feature_shape",
                "feature_dtype",
            },
            "mixture identity",
            split_name,
        )
        if (
            not isinstance(mixture["scene_id"], str)
            or not isinstance(mixture["manifest_path"], str)
            or not _valid_sha256(mixture["sha256"])
            or not isinstance(mixture["size_bytes"], int)
            or mixture["size_bytes"] <= 0
            or mixture["sample_rate"] != CLAP_CACHE_SAMPLE_RATE
            or mixture["num_samples"] != CLAP_CACHE_NUM_SAMPLES
            or mixture["duration_seconds"] != CLAP_CACHE_DURATION_SECONDS
            or mixture["feature_shape"]
            != [CLAP_CACHE_EFFECTIVE_FRAMES, CLAP_CACHE_JOINT_DIM]
            or mixture["feature_dtype"] != "float16"
        ):
            raise SystemExit(
                f"{split_name} foundation cache has invalid mixture provenance"
            )
        validated_mixtures.append(mixture)
    if sorted(row["scene_id"] for row in validated_mixtures) != expected_scene_ids:
        raise SystemExit(
            f"{split_name} foundation cache mixture scene IDs differ from manifest"
        )
    mixture_by_scene = {row["scene_id"]: row for row in validated_mixtures}
    dataset_root = manifest.resolve().parent
    current_mixture_identities: Dict[str, Dict[str, Any]] = {}
    for scene_id in expected_scene_ids:
        relative_path = scene_to_mixture_path[scene_id]
        declared = mixture_by_scene[scene_id]
        if declared["manifest_path"] != relative_path:
            raise SystemExit(
                f"{split_name} foundation cache mixture path differs for {scene_id}"
            )
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise SystemExit(
                f"{split_name} manifest mixture path must be relative: {scene_id}"
            )
        waveform_path = (dataset_root / candidate).resolve()
        try:
            waveform_path.relative_to(dataset_root)
        except ValueError as exc:
            raise SystemExit(
                f"{split_name} mixture path escapes dataset root: {scene_id}"
            ) from exc
        if not waveform_path.is_file():
            raise SystemExit(
                f"{split_name} mixture waveform not found: {waveform_path}"
            )
        actual = file_identity(waveform_path)
        if (
            declared["sha256"] != actual["sha256"]
            or declared["size_bytes"] != actual["size_bytes"]
        ):
            raise SystemExit(
                f"{split_name} mixture waveform changed after CLAP caching: "
                f"{scene_id}"
            )
        current_mixture_identities[scene_id] = actual
    if top_level["mixtures_aggregate_sha256"] != _aggregate_foundation_mixture_receipt(
        validated_mixtures
    ):
        raise SystemExit(
            f"{split_name} foundation cache mixture aggregate hash is invalid"
        )

    execution = _require_exact_keys(
        top_level["execution"],
        {"device", "text_batch_size", "determinism", "software"},
        "execution provenance",
        split_name,
    )
    if (
        not isinstance(execution["device"], str)
        or not isinstance(execution["text_batch_size"], int)
        or execution["text_batch_size"] <= 0
        or not isinstance(execution["determinism"], Mapping)
        or not isinstance(execution["software"], Mapping)
    ):
        raise SystemExit(
            f"{split_name} foundation cache execution provenance is invalid"
        )
    determinism = _require_exact_keys(
        execution["determinism"],
        {
            "seed",
            "torch_deterministic_algorithms",
            "cublas_workspace_config",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
        },
        "cache determinism provenance",
        split_name,
    )
    if (
        not isinstance(determinism["seed"], int)
        or determinism["torch_deterministic_algorithms"] is not True
        or determinism["cudnn_benchmark"] is not False
        or determinism["cudnn_deterministic"] is not True
        or determinism["cuda_matmul_allow_tf32"] is not False
        or determinism["cudnn_allow_tf32"] is not False
    ):
        raise SystemExit(
            f"{split_name} foundation cache was not generated deterministically"
        )
    _require_exact_keys(
        execution["software"],
        {"python", "numpy", "soundfile", "torch", "torchaudio", "transformers"},
        "cache software provenance",
        split_name,
    )

    artifacts = _require_exact_keys(
        top_level["artifacts"],
        {"question_features", "scene_audio_features"},
        "artifact declaration",
        split_name,
    )
    for name, path, expected_filename in (
        ("question_features", question_path, "question_features.pt"),
        ("scene_audio_features", scene_path, "scene_audio_features.pt"),
    ):
        declared = _require_exact_keys(
            artifacts[name],
            {"filename", "sha256", "size_bytes"},
            f"{name} artifact",
            split_name,
        )
        actual = file_identity(path)
        if (
            declared["filename"] != expected_filename
            or declared["sha256"] != actual["sha256"]
            or declared["size_bytes"] != actual["size_bytes"]
        ):
            raise SystemExit(
                f"{split_name} foundation cache {name} artifact hash mismatch"
            )

    try:
        question_payload = torch.load(
            question_path, map_location="cpu", weights_only=True
        )
        scene_payload = torch.load(scene_path, map_location="cpu", weights_only=True)
        validate_question_payload(question_payload)
        validate_scene_payload(scene_payload)
    except Exception as exc:
        raise SystemExit(
            f"{split_name} foundation feature tensors are invalid"
        ) from exc
    if (
        question_payload.get("format") != QUESTION_CACHE_FORMAT
        or scene_payload.get("format") != SCENE_CACHE_FORMAT
    ):
        raise SystemExit(f"{split_name} foundation feature payload format mismatch")
    question_features = question_payload["features"]
    scene_features = scene_payload["features"]
    if set(question_features) != set(expected_sample_ids):
        raise SystemExit(f"{split_name} question CLAP IDs differ from manifest")
    if set(scene_features) != set(expected_scene_ids):
        raise SystemExit(f"{split_name} scene CLAP IDs differ from manifest")
    question_matrix = torch.stack(
        [question_features[sample_id] for sample_id in expected_sample_ids]
    )
    question_norms = torch.linalg.vector_norm(question_matrix.float(), dim=-1)
    if not torch.allclose(
        question_norms,
        torch.ones_like(question_norms),
        rtol=1e-3,
        atol=1e-3,
    ):
        raise SystemExit(f"{split_name} question CLAP features are not normalized")

    identity = {
        "format": FOUNDATION_CACHE_RECEIPT_FORMAT,
        "directory": str(resolved),
        "receipt": file_identity(receipt_path),
        "manifest_binding": {
            "cache_declared": dict(declared_manifest),
            "run_expected": expected_manifest,
        },
        "audiosep_checkpoint_binding": {
            "cache_declared": dict(declared_checkpoint),
            "run_expected": dict(audiosep_checkpoint_identity),
        },
        "audiosep_source_binding": {
            "cache_declared": dict(declared_source),
            "run_expected": dict(audiosep_source_identity),
        },
        "question_feature_artifact": file_identity(question_path),
        "scene_feature_artifact": file_identity(scene_path),
        "sample_count": len(expected_sample_ids),
        "scene_count": len(expected_scene_ids),
        "question_shape": [CLAP_CACHE_JOINT_DIM],
        "scene_shape": [CLAP_CACHE_EFFECTIVE_FRAMES, CLAP_CACHE_JOINT_DIM],
        "question_dtype": "float32",
        "scene_dtype": "float16",
        "raw_to_effective_repeat_ratio": CLAP_CACHE_REPEAT_RATIO,
        "current_mixture_identities": current_mixture_identities,
        "contains_oracle_or_label_inputs": False,
        "deployment_status": (
            "offline_cache_training_candidate; online_extractor_pending"
        ),
    }
    return FoundationFeatureCache(
        question_features=question_features,
        scene_features=scene_features,
        sample_to_scene=sample_to_scene,
        identity=identity,
    )


def add_foundation_features(
    batch: Dict[str, Any],
    cache: FoundationFeatureCache | None,
    device: torch.device,
) -> None:
    """Inject only frozen feature tensors using sample-to-scene ID routing."""

    if cache is None:
        return
    if "question_clap" in batch or "scene_clap" in batch:
        raise ValueError("batch already contains foundation feature tensors")
    question_rows = []
    scene_rows = []
    for sample_id in batch["sample_ids"]:
        if sample_id not in cache.question_features:
            raise ValueError(f"foundation cache lacks sample ID: {sample_id}")
        scene_id = cache.sample_to_scene.get(sample_id)
        if scene_id is None or scene_id not in cache.scene_features:
            raise ValueError(f"foundation cache lacks scene for sample: {sample_id}")
        question_rows.append(cache.question_features[sample_id])
        scene_rows.append(cache.scene_features[scene_id])
    batch["question_clap"] = torch.stack(question_rows).to(device)
    batch["scene_clap"] = torch.stack(scene_rows).to(device)


def forward_training_batch(
    model: QCESModel,
    batch: Mapping[str, Any],
):
    """Preserve the exact legacy call while requiring CLAP-mode features."""

    mode = getattr(
        getattr(model, "config", None),
        "foundation_feature_mode",
        NO_FOUNDATION_FEATURES,
    )
    if mode == NO_FOUNDATION_FEATURES:
        if "question_clap" in batch or "scene_clap" in batch:
            raise ValueError("legacy mode received unexpected foundation features")
        return model(batch["mixture"], batch["question_ids"], batch["question_mask"])
    if mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES:
        question_clap = batch.get("question_clap")
        scene_clap = batch.get("scene_clap")
        if not isinstance(question_clap, torch.Tensor) or not isinstance(
            scene_clap, torch.Tensor
        ):
            raise ValueError(
                "audiosep_clap mode refuses a batch missing question_clap or "
                "scene_clap"
            )
        return model(
            batch["mixture"],
            batch["question_ids"],
            batch["question_mask"],
            question_clap=question_clap,
            scene_clap=scene_clap,
        )
    raise ValueError(f"unsupported foundation feature mode: {mode!r}")


def _promote_reduction_value(value: Any) -> Any:
    """Recursively promote model outputs used by losses/metrics to FP32."""

    if isinstance(value, torch.Tensor):
        if torch.is_complex(value):
            return value.to(torch.complex64)
        if value.is_floating_point():
            return value.float()
        return value
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            field.name: _promote_reduction_value(getattr(value, field.name))
            for field in fields(value)
        }
        return replace(value, **updates)
    return value


def forward_training_batch_with_precision(
    model: QCESModel,
    batch: Mapping[str, Any],
    device: torch.device,
    precision: str = FP32_PRECISION,
):
    """Autocast only the model forward; keep reductions numerically stable."""

    if precision == FP32_PRECISION:
        # This deliberately remains the exact historical forward call. In
        # particular, default training never enters an autocast context.
        return forward_training_batch(model, batch)
    validate_precision_mode(precision, device)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = forward_training_batch(model, batch)
    promoted = _promote_reduction_value(output)
    if not isinstance(promoted, type(output)):
        raise TypeError("mixed-precision QCES output promotion changed its type")
    return promoted


def create_gradient_scaler(precision: str):
    """Create a scaler only for the explicitly requested AMP path."""

    if precision == FP32_PRECISION:
        return None
    if precision != AMP_FP16_PRECISION:
        raise ValueError(f"unsupported precision mode: {precision!r}")
    return torch.amp.GradScaler("cuda", enabled=True)


def backward_and_optimizer_step(
    total: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[torch.nn.Parameter],
    *,
    precision: str = FP32_PRECISION,
    scaler: Any | None = None,
    max_grad_norm: float = 5.0,
) -> None:
    """Backpropagate, unscale AMP gradients, clip, and update parameters."""

    if precision == FP32_PRECISION:
        if scaler is not None:
            raise ValueError("fp32 optimizer flow must not receive a GradScaler")
        # Keep the default operation order identical to the pre-AMP path.
        total.backward()
        clip_grad_norm_(parameters, max_grad_norm)
        optimizer.step()
        return
    if precision != AMP_FP16_PRECISION:
        raise ValueError(f"unsupported precision mode: {precision!r}")
    if scaler is None:
        raise ValueError("amp_fp16 optimizer flow requires a GradScaler")
    scaler.scale(total).backward()
    # Clipping scaled gradients changes the effective threshold, so unscale is
    # mandatory and intentionally occurs before clip_grad_norm_.
    scaler.unscale_(optimizer)
    clip_grad_norm_(parameters, max_grad_norm)
    scaler.step(optimizer)
    scaler.update()


def precision_runtime_metadata(precision: str, scaler: Any | None) -> Dict[str, Any]:
    """Return JSON-safe precision and current dynamic-scaler provenance."""

    if precision == FP32_PRECISION:
        if scaler is not None:
            raise ValueError("fp32 metadata must not receive a GradScaler")
        scaler_metadata: Dict[str, Any] = {
            "enabled": False,
            "type": None,
            "state_dict": None,
            "unscale_before_gradient_clip": None,
        }
    elif precision == AMP_FP16_PRECISION:
        if scaler is None:
            raise ValueError("amp_fp16 metadata requires a GradScaler")
        raw_state = scaler.state_dict()
        state = {
            key: (
                value.item()
                if isinstance(value, torch.Tensor) and value.numel() == 1
                else value
            )
            for key, value in raw_state.items()
        }
        scaler_metadata = {
            "enabled": True,
            "type": "torch.amp.GradScaler",
            "state_dict": state,
            "current_scale": float(scaler.get_scale()),
            "unscale_before_gradient_clip": True,
        }
    else:
        raise ValueError(f"unsupported precision mode: {precision!r}")
    return {
        "mode": precision,
        "legacy_default": FP32_PRECISION,
        "autocast": {
            "enabled": precision == AMP_FP16_PRECISION,
            "device_type": "cuda" if precision == AMP_FP16_PRECISION else None,
            "dtype": "float16" if precision == AMP_FP16_PRECISION else None,
            "scope": "model_forward_only",
        },
        "loss_metric_reduction_dtype": "float32",
        "gradient_scaler": scaler_metadata,
        "audiosep_parameter_dtype_mutated": False,
    }


def synchronize_cuda(device: torch.device) -> None:
    """Synchronize only when phase timing a CUDA run."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configure_reproducibility(seed: int, deterministic: bool) -> Dict[str, Any]:
    """Seed every training RNG and configure strict deterministic kernels."""

    if deterministic:
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is not None and workspace not in {":4096:8", ":16:8"}:
            raise SystemExit(
                "deterministic CUDA execution needs CUBLAS_WORKSPACE_CONFIG to be "
                f":4096:8 or :16:8, got {workspace!r}"
            )
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    # Benchmarking can select a different convolution implementation across
    # runs and therefore stays disabled even when strict mode is opted out.
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": seed,
        "deterministic_requested": deterministic,
        "deterministic_algorithms": (torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "python_hash_seed_at_process_start": os.environ.get("PYTHONHASHSEED"),
    }


COMPOSER_INITIALIZATION_RECEIPT_FORMAT = "qces_composer_initialization_v1"
COMPOSER_CANDIDATE_ONLY_PREFIXES = ("same_semantic_head.",)


def _composer_tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Return canonical logical tensor bytes, independent of device/strides."""

    canonical = tensor.detach().to(device="cpu").contiguous().reshape(-1)
    # Viewing a one-dimensional tensor as uint8 also supports scalar, complex,
    # and bfloat16 state without relying on NumPy support for the source dtype.
    return canonical.view(torch.uint8).numpy().tobytes(order="C")


def _aggregate_composer_tensor_receipts(
    tensor_receipts: Mapping[str, Mapping[str, Any]],
    keys: Sequence[str],
) -> str:
    """Hash a sorted, self-describing set of tensor digests."""

    records = [
        {
            "key": key,
            "dtype": tensor_receipts[key]["dtype"],
            "shape": tensor_receipts[key]["shape"],
            "numel": tensor_receipts[key]["numel"],
            "sha256": tensor_receipts[key]["sha256"],
        }
        for key in sorted(keys)
    ]
    canonical_json = json.dumps(
        records,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_json).hexdigest()


def composer_initialization_receipt(
    composer: torch.nn.Module,
    *,
    seed: int,
) -> Dict[str, Any]:
    """Fingerprint a freshly initialized composer before its first update.

    ``same_semantic_head`` is a dual-role candidate-only module.  Its tensors
    are retained in the full receipt but excluded from the common aggregates,
    making the latter directly comparable with a union-single run.
    """

    state = composer.state_dict()
    parameter_keys = {name for name, _ in composer.named_parameters()}
    tensor_receipts: Dict[str, Dict[str, Any]] = {}
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"composer state entry {key!r} is not a tensor; "
                "initialization receipt would be incomplete"
            )
        candidate_only = key.startswith(COMPOSER_CANDIDATE_ONLY_PREFIXES)
        raw = _composer_tensor_bytes(tensor)
        tensor_receipts[key] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "numel": tensor.numel(),
            "state_kind": "parameter" if key in parameter_keys else "buffer",
            "partition": "candidate_only" if candidate_only else "common",
        }

    all_keys = sorted(tensor_receipts)
    candidate_only_keys = [
        key for key in all_keys if tensor_receipts[key]["partition"] == "candidate_only"
    ]
    common_state_keys = [key for key in all_keys if key not in set(candidate_only_keys)]
    common_parameter_keys = [key for key in common_state_keys if key in parameter_keys]
    candidate_only_parameter_keys = [
        key for key in candidate_only_keys if key in parameter_keys
    ]

    mode = getattr(getattr(composer, "config", None), "semantic_separation_mode", None)
    return {
        "format": COMPOSER_INITIALIZATION_RECEIPT_FORMAT,
        "source": "from_scratch_seeded_initialization",
        "capture_point": (
            "after composer construction and device transfer; before optimizer "
            "construction, warm-start loading, or training"
        ),
        "seed": int(seed),
        "semantic_separation_mode": mode,
        "hash_algorithm": "sha256",
        "tensor_byte_encoding": "contiguous_cpu_logical_bytes",
        "candidate_only_prefixes": list(COMPOSER_CANDIDATE_ONLY_PREFIXES),
        "state_tensor_count": len(all_keys),
        "parameter_tensor_count": len(parameter_keys),
        "common_state_tensor_keys": common_state_keys,
        "common_parameter_keys": common_parameter_keys,
        "candidate_only_state_tensor_keys": candidate_only_keys,
        "candidate_only_parameter_keys": candidate_only_parameter_keys,
        "tensor_receipts": tensor_receipts,
        "aggregates": {
            "all_state_tensors_sha256": _aggregate_composer_tensor_receipts(
                tensor_receipts, all_keys
            ),
            "common_state_tensors_sha256": _aggregate_composer_tensor_receipts(
                tensor_receipts, common_state_keys
            ),
            # This is the comparison-safe architecture-ablation identity: it
            # excludes candidate-only parameters and is therefore mode-neutral.
            "common_parameters_sha256": _aggregate_composer_tensor_receipts(
                tensor_receipts, common_parameter_keys
            ),
            "candidate_only_state_tensors_sha256": (
                _aggregate_composer_tensor_receipts(
                    tensor_receipts, candidate_only_keys
                )
                if candidate_only_keys
                else None
            ),
        },
    }


def seed_data_worker(worker_id: int) -> None:
    """Give each DataLoader worker a distinct, reproducible RNG stream."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    worker = torch.utils.data.get_worker_info()
    if worker is not None and hasattr(worker.dataset, "random"):
        worker.dataset.random.seed(worker_seed)


def _protected_output_paths() -> set[Path]:
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        PROJECT_ROOT.resolve(),
        CODE_ROOT.resolve(),
        (PROJECT_ROOT / "outputs").resolve(),
    }
    for broad_path in (
        Path.home(),
        PROJECT_ROOT,
        CODE_ROOT,
        Path("/tmp"),
        Path("/var"),
        Path("/home"),
        Path("/mnt"),
        Path("/media"),
        Path("/data"),
        Path("/workspace"),
    ):
        protected.add(broad_path.resolve())
    # Never allow a run-level overwrite to erase an ancestor containing the
    # repository or the user's home, even if that ancestor has an unusual name.
    protected.update(PROJECT_ROOT.resolve().parents)
    protected.update(Path.home().resolve().parents)
    return protected


def check_output_directory(requested: Path, overwrite: bool) -> tuple[Path, bool]:
    """Validate an output target without mutating an existing run directory."""

    if requested.is_symlink():
        raise SystemExit(f"output directory must not be a symlink: {requested}")
    output_dir = requested.resolve()
    if output_dir in _protected_output_paths():
        raise SystemExit(f"refusing unsafe output directory target: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise SystemExit(f"output directory path is not a directory: {output_dir}")
    nonempty = output_dir.exists() and next(output_dir.iterdir(), None) is not None
    if nonempty and not overwrite:
        raise SystemExit(
            f"output directory is not empty: {output_dir}; use --overwrite"
        )
    return output_dir, bool(nonempty)


def prepare_output_directory(
    output_dir: Path, *, overwrite: bool, was_nonempty: bool
) -> None:
    """Create an empty run directory, deleting only a prevalidated exact target."""

    if output_dir in _protected_output_paths():
        raise SystemExit(f"refusing unsafe output directory target: {output_dir}")
    if output_dir.is_symlink():
        raise SystemExit(f"output directory must not be a symlink: {output_dir}")
    if was_nonempty:
        if not overwrite:
            raise SystemExit(f"refusing to replace output directory: {output_dir}")
        if not output_dir.is_dir():
            raise SystemExit(f"output directory changed type: {output_dir}")
        shutil.rmtree(output_dir)
    elif output_dir.exists() and next(output_dir.iterdir(), None) is not None:
        # The target changed after preflight. Fail rather than deleting files
        # that were not present when --overwrite was authorized.
        raise SystemExit(f"output directory changed after preflight: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def selection_improved(
    candidate: float,
    incumbent: float | None,
    direction: str,
    min_delta: float,
) -> bool:
    """Return whether a finite validation value clears the stated margin."""

    if not math.isfinite(candidate):
        raise ValueError(f"selection metric is not finite: {candidate}")
    if direction not in {"minimize", "maximize"}:
        raise ValueError(f"unsupported selection direction: {direction}")
    if min_delta < 0:
        raise ValueError("selection min_delta must be non-negative")
    if incumbent is None:
        return True
    if direction == "minimize":
        return candidate < incumbent - min_delta
    return candidate > incumbent + min_delta


def load_semantic_target_cache(
    cache_path: Path,
    manifest: Path,
    sample_ids: Sequence[str],
    split_name: str,
    *,
    expected_no_evidence_ids: Sequence[str] | None = None,
    expected_dim: int,
    expected_schema_versions: Sequence[str],
    audiosep_checkpoint_identity: Mapping[str, Any],
) -> tuple[Mapping[str, torch.Tensor], Dict[str, Any]]:
    """Load a strict cache bound to one split and one AudioSep checkpoint."""

    resolved_cache = cache_path.resolve()
    if not resolved_cache.is_file():
        raise SystemExit(
            f"{split_name} semantic-target cache not found: {resolved_cache}"
        )
    payload = torch.load(resolved_cache, map_location="cpu", weights_only=True)
    if (
        isinstance(payload, dict)
        and payload.get("format") == V5_UNION_SEMANTIC_CACHE_FORMAT
    ):
        return _validate_v5_union_semantic_target_cache(
            payload,
            resolved_cache,
            manifest,
            sample_ids,
            expected_no_evidence_ids,
            split_name,
            expected_dim=expected_dim,
            expected_schema_versions=expected_schema_versions,
            audiosep_checkpoint_identity=audiosep_checkpoint_identity,
        )
    if (
        not isinstance(payload, dict)
        or payload.get("format") != SEMANTIC_CACHE_FORMAT
        or not isinstance(payload.get("targets"), dict)
    ):
        raise SystemExit(f"invalid {split_name} AudioSep semantic-target cache")
    expected_schemas = set(expected_schema_versions)
    if len(expected_schemas) != 1:
        raise SystemExit(
            f"{split_name} manifest must declare exactly one schema for semantic "
            f"supervision, got {sorted(expected_schemas)}"
        )
    expected_schema = next(iter(expected_schemas))
    if payload.get("schema_version") != expected_schema:
        raise SystemExit(
            f"{split_name} semantic-target schema mismatch: "
            f"cache={payload.get('schema_version')!r}, manifest={expected_schema!r}"
        )
    if payload.get("target_scope") != SEMANTIC_TARGET_SCOPE:
        raise SystemExit(f"{split_name} semantic-target cache has invalid target_scope")
    if payload.get("prompt_source") != SEMANTIC_PROMPT_SOURCE:
        raise SystemExit(
            f"{split_name} semantic-target cache has invalid prompt_source"
        )
    expected_hash = sha256_file(manifest)
    if payload.get("manifest_sha256") != expected_hash:
        raise SystemExit(
            f"{split_name} semantic-target cache does not match its manifest"
        )

    if expected_dim <= 0:
        raise ValueError("semantic target dimension must be positive")
    if len(set(sample_ids)) != len(sample_ids):
        raise SystemExit(f"{split_name} manifest contains duplicate sample IDs")
    targets = payload["targets"]
    if any(not isinstance(sample_id, str) for sample_id in targets):
        raise SystemExit(f"{split_name} semantic-target cache contains a non-string ID")
    expected_ids = set(sample_ids)
    target_ids = set(targets)
    missing = sorted(expected_ids - target_ids)
    extra = sorted(target_ids - expected_ids)
    if missing or extra:
        raise SystemExit(
            f"{split_name} semantic-target IDs differ from the manifest: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    invalid_targets = []
    dtypes = set()
    for sample_id in sample_ids:
        target = targets[sample_id]
        reason = None
        if not isinstance(target, torch.Tensor):
            reason = "not a tensor"
        elif not target.dtype.is_floating_point:
            reason = f"dtype={target.dtype} is not floating point"
        elif target.ndim != 1:
            reason = f"rank={target.ndim}, expected rank=1"
        elif target.numel() != expected_dim:
            reason = f"dim={target.numel()}, expected dim={expected_dim}"
        elif not bool(torch.isfinite(target).all()):
            reason = "contains NaN or Inf"
        if reason is not None:
            invalid_targets.append((sample_id, reason))
        elif isinstance(target, torch.Tensor):
            dtypes.add(str(target.dtype))
    if invalid_targets:
        raise SystemExit(
            f"{split_name} semantic-target tensors are invalid: "
            f"{invalid_targets[:10]}"
        )

    prompts = payload.get("prompts")
    if not isinstance(prompts, dict) or set(prompts) != expected_ids:
        raise SystemExit(
            f"{split_name} semantic-target prompts must match manifest IDs exactly"
        )
    invalid_prompts = sorted(
        sample_id
        for sample_id, prompt in prompts.items()
        if not isinstance(sample_id, str)
        or not isinstance(prompt, str)
        or not prompt.strip()
    )
    if invalid_prompts:
        raise SystemExit(
            f"{split_name} semantic-target prompts are invalid: "
            f"{invalid_prompts[:10]}"
        )

    expected_checkpoint_hash = audiosep_checkpoint_identity.get("sha256")
    expected_checkpoint_path = audiosep_checkpoint_identity.get("path")
    if not isinstance(expected_checkpoint_hash, str) or not isinstance(
        expected_checkpoint_path, str
    ):
        raise ValueError("AudioSep checkpoint identity must include path and SHA256")
    declared_checkpoint = payload.get("audiosep_checkpoint")
    if not isinstance(declared_checkpoint, str) or not declared_checkpoint.strip():
        raise SystemExit(
            f"{split_name} semantic-target cache lacks AudioSep checkpoint metadata"
        )
    declared_hash = payload.get("audiosep_checkpoint_sha256")
    if declared_hash is None and isinstance(
        payload.get("audiosep_checkpoint_identity"), dict
    ):
        declared_hash = payload["audiosep_checkpoint_identity"].get("sha256")
    if declared_hash is not None:
        if (
            not isinstance(declared_hash, str)
            or len(declared_hash) != 64
            or any(character not in "0123456789abcdef" for character in declared_hash)
        ):
            raise SystemExit(
                f"{split_name} semantic-target cache has malformed checkpoint SHA256"
            )
        if declared_hash != expected_checkpoint_hash:
            raise SystemExit(
                f"{split_name} semantic-target cache was built with a different "
                "AudioSep checkpoint"
            )
        checkpoint_binding_method = "recorded_sha256"
        declared_checkpoint_identity = {
            "path": str(Path(declared_checkpoint).expanduser()),
            "sha256": declared_hash,
        }
    else:
        # Legacy v1 caches recorded only the absolute checkpoint path. Resolve
        # and hash that file now, fail closed if it disappeared, and retain the
        # weaker binding method in provenance so it is never mistaken for a
        # creation-time checksum.
        declared_checkpoint_path = Path(declared_checkpoint).expanduser().resolve()
        if not declared_checkpoint_path.is_file():
            raise SystemExit(
                f"{split_name} cached AudioSep checkpoint no longer exists: "
                f"{declared_checkpoint_path}"
            )
        if str(declared_checkpoint_path) == expected_checkpoint_path:
            declared_checkpoint_identity = dict(audiosep_checkpoint_identity)
        else:
            declared_checkpoint_identity = file_identity(declared_checkpoint_path)
        if declared_checkpoint_identity["sha256"] != expected_checkpoint_hash:
            raise SystemExit(
                f"{split_name} semantic-target cache resolves to a different "
                "AudioSep checkpoint"
            )
        checkpoint_binding_method = "legacy_path_rehashed_at_load"
    identity = {
        **file_identity(resolved_cache),
        "format": payload["format"],
        "manifest_sha256": payload["manifest_sha256"],
        "target_count": len(targets),
        "target_dim": expected_dim,
        "target_dtypes": sorted(dtypes),
        "schema_version": payload["schema_version"],
        "target_scope": payload["target_scope"],
        "prompt_source": payload["prompt_source"],
        "audiosep_checkpoint_binding": {
            "method": checkpoint_binding_method,
            "cache_declared": declared_checkpoint_identity,
            "run_expected": dict(audiosep_checkpoint_identity),
        },
    }
    return targets, identity


def _validate_v5_union_semantic_target_cache(
    payload: Mapping[str, Any],
    resolved_cache: Path,
    manifest: Path,
    sample_ids: Sequence[str],
    expected_no_evidence_ids: Sequence[str] | None,
    split_name: str,
    *,
    expected_dim: int,
    expected_schema_versions: Sequence[str],
    audiosep_checkpoint_identity: Mapping[str, Any],
) -> tuple[Mapping[str, torch.Tensor], Dict[str, Any]]:
    """Validate answerable-only QCES-v5 union semantic supervision."""

    targets = payload.get("targets")
    prompts = payload.get("prompts")
    declared_no_evidence_list = payload.get("no_evidence_ids")
    if (
        not isinstance(targets, dict)
        or not isinstance(prompts, dict)
        or not isinstance(declared_no_evidence_list, list)
    ):
        raise SystemExit(f"invalid {split_name} QCES-v5 union semantic cache")
    schemas = set(expected_schema_versions)
    if len(schemas) != 1:
        raise SystemExit(
            f"{split_name} manifest must declare exactly one semantic-cache "
            f"schema, got {sorted(schemas)}"
        )
    expected_schema = next(iter(schemas))
    if payload.get("schema_version") != expected_schema:
        raise SystemExit(f"{split_name} QCES-v5 union semantic schema mismatch")
    if payload.get("target_scope") != V5_UNION_SEMANTIC_TARGET_SCOPE:
        raise SystemExit(f"{split_name} QCES-v5 union semantic cache has invalid scope")
    if payload.get("prompt_source") != V5_UNION_SEMANTIC_PROMPT_SOURCE:
        raise SystemExit(
            f"{split_name} QCES-v5 union semantic cache has invalid prompt source"
        )
    if payload.get("manifest_sha256") != sha256_file(manifest):
        raise SystemExit(
            f"{split_name} QCES-v5 union semantic cache does not match manifest"
        )
    if expected_dim <= 0:
        raise ValueError("union semantic target dimension must be positive")
    if len(set(sample_ids)) != len(sample_ids):
        raise SystemExit(f"{split_name} manifest contains duplicate sample IDs")
    if expected_no_evidence_ids is None:
        raise SystemExit(
            f"{split_name} QCES-v5 union cache requires the manifest "
            "no-evidence partition"
        )
    if any(not isinstance(item, str) for item in declared_no_evidence_list):
        raise SystemExit(f"{split_name} QCES-v5 union no-evidence IDs must be strings")
    if len(set(declared_no_evidence_list)) != len(declared_no_evidence_list):
        raise SystemExit(
            f"{split_name} QCES-v5 union no-evidence IDs contain duplicates"
        )
    expected_ids = set(sample_ids)
    expected_no_evidence = set(expected_no_evidence_ids)
    declared_no_evidence = set(declared_no_evidence_list)
    if declared_no_evidence != expected_no_evidence:
        raise SystemExit(
            f"{split_name} QCES-v5 union no-evidence partition differs from "
            "the manifest"
        )
    answerable_ids = expected_ids - expected_no_evidence
    if not answerable_ids:
        raise SystemExit(
            f"{split_name} QCES-v5 union cache has no answerable semantic targets"
        )
    if expected_no_evidence - expected_ids:
        raise SystemExit(
            f"{split_name} QCES-v5 union no-evidence partition has unknown IDs"
        )
    if set(targets) != answerable_ids or set(prompts) != answerable_ids:
        raise SystemExit(
            f"{split_name} QCES-v5 union answerable IDs differ from manifest"
        )
    if (
        answerable_ids & declared_no_evidence
        or (answerable_ids | declared_no_evidence) != expected_ids
    ):
        raise SystemExit(
            f"{split_name} QCES-v5 union cache is not an exact ID partition"
        )

    dtypes = set()
    for sample_id in sorted(answerable_ids):
        target = targets[sample_id]
        if (
            not isinstance(target, torch.Tensor)
            or not target.dtype.is_floating_point
            or target.ndim != 1
            or target.numel() != expected_dim
            or not bool(torch.isfinite(target).all())
        ):
            raise SystemExit(
                f"{split_name} invalid QCES-v5 union semantic tensor: {sample_id}"
            )
        dtypes.add(str(target.dtype))
        prompt = prompts[sample_id]
        if not isinstance(prompt, str) or not prompt.strip():
            raise SystemExit(f"{split_name} invalid QCES-v5 union prompt: {sample_id}")

    expected_hash = audiosep_checkpoint_identity.get("sha256")
    declared_hash = payload.get("audiosep_checkpoint_sha256")
    declared_identity = payload.get("audiosep_checkpoint_identity")
    declared_checkpoint = payload.get("audiosep_checkpoint")
    if (
        not isinstance(expected_hash, str)
        or not isinstance(declared_hash, str)
        or declared_hash != expected_hash
        or not isinstance(declared_identity, dict)
        or declared_identity.get("sha256") != expected_hash
        or not isinstance(declared_checkpoint, str)
        or not declared_checkpoint.strip()
    ):
        raise SystemExit(
            f"{split_name} QCES-v5 union cache is not bound to the exact "
            "AudioSep checkpoint"
        )
    identity = {
        **file_identity(resolved_cache),
        "format": V5_UNION_SEMANTIC_CACHE_FORMAT,
        "manifest_sha256": payload["manifest_sha256"],
        "schema_version": expected_schema,
        "target_scope": V5_UNION_SEMANTIC_TARGET_SCOPE,
        "prompt_source": V5_UNION_SEMANTIC_PROMPT_SOURCE,
        "target_count": len(answerable_ids),
        "answerable_target_count": len(answerable_ids),
        "no_evidence_count": len(declared_no_evidence),
        "no_evidence_has_semantic_target": False,
        "target_dim": expected_dim,
        "target_dtypes": sorted(dtypes),
        "audiosep_checkpoint_binding": {
            "method": "creation_time_sha256",
            "cache_declared": dict(declared_identity),
            "run_expected": dict(audiosep_checkpoint_identity),
        },
    }
    return targets, identity


def load_role_semantic_target_cache(
    cache_path: Path,
    manifest: Path,
    sample_ids: Sequence[str],
    no_evidence_ids: Sequence[str],
    expected_same_semantic_targets: Mapping[str, bool],
    split_name: str,
    *,
    expected_dim: int,
    expected_schema_versions: Sequence[str],
    audiosep_checkpoint_identity: Mapping[str, Any],
) -> tuple[Mapping[str, Mapping[str, object]], Dict[str, Any]]:
    """Load role-specific CLAP targets and same-label supervision strictly.

    The cache partitions all manifest IDs into answerable role targets and
    no-evidence IDs.  Thus a missing role target can never silently become a
    fabricated zero target, and the same-label bit is demonstrably sourced
    from training metadata rather than inferred from validation/test labels.
    """

    resolved_cache = cache_path.resolve()
    if not resolved_cache.is_file():
        raise SystemExit(
            f"{split_name} role-semantic cache not found: {resolved_cache}"
        )
    payload = torch.load(resolved_cache, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != ROLE_SEMANTIC_CACHE_FORMAT
        or not isinstance(payload.get("role_targets"), dict)
        or not isinstance(payload.get("role_prompts"), dict)
        or not isinstance(payload.get("no_evidence_ids"), list)
    ):
        raise SystemExit(f"invalid {split_name} role-semantic target cache")
    schemas = set(expected_schema_versions)
    if len(schemas) != 1:
        raise SystemExit(
            f"{split_name} manifest must declare exactly one schema for role "
            f"semantic supervision, got {sorted(schemas)}"
        )
    expected_schema = next(iter(schemas))
    if payload.get("schema_version") != expected_schema:
        raise SystemExit(
            f"{split_name} role-semantic schema mismatch: "
            f"cache={payload.get('schema_version')!r}, "
            f"manifest={expected_schema!r}"
        )
    if payload.get("target_scope") != ROLE_SEMANTIC_TARGET_SCOPE:
        raise SystemExit(f"{split_name} role-semantic cache has invalid scope")
    if payload.get("prompt_source") != ROLE_SEMANTIC_PROMPT_SOURCE:
        raise SystemExit(f"{split_name} role-semantic cache has invalid prompt source")
    if payload.get("manifest_sha256") != sha256_file(manifest):
        raise SystemExit(
            f"{split_name} role-semantic cache does not match its manifest"
        )
    if expected_dim <= 0:
        raise ValueError("role-semantic target dimension must be positive")
    if len(set(sample_ids)) != len(sample_ids):
        raise SystemExit(f"{split_name} manifest contains duplicate sample IDs")
    expected_ids = set(sample_ids)
    expected_no_evidence = set(no_evidence_ids)
    declared_no_evidence_list = payload["no_evidence_ids"]
    if any(not isinstance(item, str) for item in declared_no_evidence_list):
        raise SystemExit(f"{split_name} role-semantic no-evidence IDs must be strings")
    if len(set(declared_no_evidence_list)) != len(declared_no_evidence_list):
        raise SystemExit(
            f"{split_name} role-semantic no-evidence IDs contain duplicates"
        )
    declared_no_evidence = set(declared_no_evidence_list)
    if declared_no_evidence != expected_no_evidence:
        raise SystemExit(
            f"{split_name} role-semantic no-evidence partition differs from "
            "the manifest"
        )

    targets = payload["role_targets"]
    prompts = payload["role_prompts"]
    answerable_ids = expected_ids - expected_no_evidence
    if set(expected_same_semantic_targets) != answerable_ids or any(
        not isinstance(value, bool) for value in expected_same_semantic_targets.values()
    ):
        raise ValueError(
            f"{split_name} expected same-semantic labels must cover every "
            "answerable manifest ID exactly"
        )
    if set(targets) != answerable_ids or set(prompts) != answerable_ids:
        raise SystemExit(
            f"{split_name} role-semantic answerable IDs differ from the manifest"
        )
    if (
        answerable_ids & declared_no_evidence
        or (answerable_ids | declared_no_evidence) != expected_ids
    ):
        raise SystemExit(
            f"{split_name} role-semantic cache is not an exact ID partition"
        )

    same_count = 0
    dtypes = set()
    for sample_id in sorted(answerable_ids):
        target = targets[sample_id]
        prompt = prompts[sample_id]
        if not isinstance(target, dict) or set(target) != {
            "anchor",
            "answer",
            "same_semantic",
        }:
            raise SystemExit(f"{split_name} invalid role target record: {sample_id}")
        if not isinstance(prompt, dict) or set(prompt) != {"anchor", "answer"}:
            raise SystemExit(f"{split_name} invalid role prompt record: {sample_id}")
        for role in ("anchor", "answer"):
            value = target[role]
            if (
                not isinstance(value, torch.Tensor)
                or not value.dtype.is_floating_point
                or value.ndim != 1
                or value.numel() != expected_dim
                or not bool(torch.isfinite(value).all())
            ):
                raise SystemExit(
                    f"{split_name} invalid {role} semantic tensor: {sample_id}"
                )
            dtypes.add(str(value.dtype))
            text = prompt[role]
            if not isinstance(text, str) or not text.strip():
                raise SystemExit(
                    f"{split_name} invalid {role} semantic prompt: {sample_id}"
                )
        same = target["same_semantic"]
        if not isinstance(same, bool):
            raise SystemExit(
                f"{split_name} same-semantic target must be boolean: {sample_id}"
            )
        if same != expected_same_semantic_targets[sample_id]:
            raise SystemExit(
                f"{split_name} same-semantic target disagrees with annotated "
                f"role-event labels: {sample_id}"
            )
        same_count += int(same)

    expected_hash = audiosep_checkpoint_identity.get("sha256")
    declared_hash = payload.get("audiosep_checkpoint_sha256")
    declared_identity = payload.get("audiosep_checkpoint_identity")
    if (
        not isinstance(expected_hash, str)
        or not isinstance(declared_hash, str)
        or declared_hash != expected_hash
        or not isinstance(declared_identity, dict)
        or declared_identity.get("sha256") != expected_hash
    ):
        raise SystemExit(
            f"{split_name} role-semantic cache is not bound to the exact "
            "AudioSep checkpoint"
        )
    identity = {
        **file_identity(resolved_cache),
        "format": ROLE_SEMANTIC_CACHE_FORMAT,
        "manifest_sha256": payload["manifest_sha256"],
        "schema_version": expected_schema,
        "target_scope": ROLE_SEMANTIC_TARGET_SCOPE,
        "prompt_source": ROLE_SEMANTIC_PROMPT_SOURCE,
        "answerable_target_count": len(answerable_ids),
        "no_evidence_count": len(declared_no_evidence),
        "same_semantic_count": same_count,
        "different_semantic_count": len(answerable_ids) - same_count,
        "same_semantic_labels_revalidated_against_manifest": True,
        "target_dim": expected_dim,
        "target_dtypes": sorted(dtypes),
        "audiosep_checkpoint_binding": {
            "method": "creation_time_sha256",
            "cache_declared": dict(declared_identity),
            "run_expected": dict(audiosep_checkpoint_identity),
        },
    }
    return targets, identity


def add_semantic_targets(
    batch: Dict[str, Any],
    semantic_targets: Mapping[str, torch.Tensor] | None,
    device: torch.device,
) -> None:
    if semantic_targets is None:
        return
    if not semantic_targets:
        raise ValueError("semantic target mapping cannot be empty")
    prototype = next(iter(semantic_targets.values()))
    if not isinstance(prototype, torch.Tensor) or prototype.ndim != 1:
        raise ValueError("semantic targets must be rank-one tensors")
    no_evidence = batch.get("no_evidence")
    rows = []
    valid = []
    for index, sample_id in enumerate(batch["sample_ids"]):
        target = semantic_targets.get(sample_id)
        is_no_evidence = bool(
            isinstance(no_evidence, torch.Tensor) and float(no_evidence[index]) >= 0.5
        )
        if target is None:
            if not is_no_evidence:
                raise ValueError(
                    f"answerable sample lacks semantic target: {sample_id}"
                )
            rows.append(torch.zeros_like(prototype))
            valid.append(0.0)
        else:
            rows.append(target)
            # Legacy caches may physically contain an absent-label vector for
            # negatives, but it remains invalid for semantic alignment. New
            # QCES-v5 caches omit such vectors entirely.
            valid.append(0.0 if is_no_evidence else 1.0)
    batch["semantic_target"] = torch.stack(rows).to(device)
    batch["semantic_target_valid"] = torch.tensor(
        valid, dtype=torch.float32, device=device
    )


def same_semantic_targets_from_records(
    records: Sequence[object], split_name: str
) -> Dict[str, bool]:
    """Recompute same-label supervision from parsed role-event metadata."""

    result: Dict[str, bool] = {}
    for record in records:
        sample_id = getattr(record, "sample_id", None)
        if not isinstance(sample_id, str):
            raise SystemExit(
                f"{split_name} dual-role manifest record lacks a sample ID"
            )
        if bool(getattr(record, "no_evidence", False)):
            continue
        anchor_ids = getattr(record, "anchor_event_ids", ())
        answer_ids = getattr(record, "answer_event_ids", ())
        event_by_id = getattr(record, "event_by_id", None)
        if len(anchor_ids) != 1 or len(answer_ids) != 1 or not callable(event_by_id):
            raise SystemExit(
                f"{split_name} dual-role record needs one annotated event per "
                f"role: {sample_id}"
            )
        anchor_label = getattr(event_by_id(anchor_ids[0]), "label", None)
        answer_label = getattr(event_by_id(answer_ids[0]), "label", None)
        if not isinstance(anchor_label, str) or not isinstance(answer_label, str):
            raise SystemExit(
                f"{split_name} role events lack semantic labels: {sample_id}"
            )
        result[sample_id] = anchor_label == answer_label
    return result


def add_role_semantic_targets(
    batch: Dict[str, Any],
    role_targets: Mapping[str, Mapping[str, object]] | None,
    device: torch.device,
    condition_dim: int,
) -> None:
    if role_targets is None:
        return
    batch_size = len(batch["sample_ids"])
    anchor = torch.zeros(batch_size, condition_dim, dtype=torch.float32)
    answer = torch.zeros_like(anchor)
    same = torch.zeros(batch_size, dtype=torch.float32)
    valid = torch.zeros(batch_size, dtype=torch.float32)
    for index, sample_id in enumerate(batch["sample_ids"]):
        target = role_targets.get(sample_id)
        if target is None:
            continue
        anchor[index] = target["anchor"]  # type: ignore[assignment]
        answer[index] = target["answer"]  # type: ignore[assignment]
        same[index] = float(bool(target["same_semantic"]))
        valid[index] = 1.0
    batch["anchor_semantic_target"] = anchor.to(device)
    batch["answer_semantic_target"] = answer.to(device)
    batch["same_semantic_target"] = same.to(device)
    batch["role_semantic_valid"] = valid.to(device)


@torch.inference_mode()
def evaluate_epoch(
    model: QCESModel,
    criterion: QCESLoss,
    loader: DataLoader,
    device: torch.device,
    semantic_targets: Mapping[str, torch.Tensor] | None,
    role_semantic_targets: Mapping[str, Mapping[str, object]] | None = None,
    foundation_feature_cache: FoundationFeatureCache | None = None,
    precision: str = FP32_PRECISION,
) -> Dict[str, float]:
    """Evaluate each record once and aggregate record- or group-level means."""

    model.eval()
    totals: Dict[str, float] = {}
    denominators: Dict[str, int] = {}
    paired_group_totals = {"surface": 0, "family": 0, "question": 0}
    for raw_batch in loader:
        batch = move_tensors(raw_batch, device)
        add_foundation_features(batch, foundation_feature_cache, device)
        add_semantic_targets(batch, semantic_targets, device)
        if role_semantic_targets is not None:
            add_role_semantic_targets(
                batch,
                role_semantic_targets,
                device,
                model.config.condition_dim,
            )
        output = forward_training_batch_with_precision(model, batch, device, precision)
        _, components = criterion(output, batch)
        _, paired_metrics, paired_counts = counterfactual_objectives(
            output,
            batch,
            transition_margin=criterion.counterfactual_transition_margin,
        )
        measured = {**components, **qces_metrics(output, batch), **paired_metrics}
        batch_size = len(batch["sample_ids"])
        if batch_size != 1 and "counterfactual_groups" not in batch:
            raise ValueError(
                "multi-record validation batches require a no-duplication "
                "counterfactual group plan"
            )
        for group_name, count in paired_counts.items():
            paired_group_totals[group_name] += count
        answerable_count = int((batch["no_evidence"] < 0.5).sum().item())
        no_evidence_count = batch_size - answerable_count
        role_valid = batch.get("role_semantic_valid")
        same_target = batch.get("same_semantic_target")
        same_semantic_count = (
            int(((role_valid > 0.5) & (same_target > 0.5)).sum().item())
            if isinstance(role_valid, torch.Tensor)
            and isinstance(same_target, torch.Tensor)
            else 0
        )
        different_semantic_count = (
            int(((role_valid > 0.5) & (same_target <= 0.5)).sum().item())
            if isinstance(role_valid, torch.Tensor)
            and isinstance(same_target, torch.Tensor)
            else 0
        )
        for name, value in measured.items():
            if name == "total":
                continue
            if name in SURFACE_LOSS_NAMES or name.startswith("surface_"):
                weight = paired_counts["surface"]
            elif name in FAMILY_LOSS_NAMES or name.startswith("family_"):
                weight = paired_counts["family"]
            elif name in QUESTION_LOSS_NAMES or name.startswith("question_"):
                weight = paired_counts["question"]
            elif name in SAME_SEMANTIC_NORMALIZED_METRICS:
                weight = same_semantic_count
            elif name in DIFFERENT_SEMANTIC_NORMALIZED_METRICS:
                weight = different_semantic_count
            elif name in ANSWERABLE_NORMALIZED_METRICS:
                weight = answerable_count
            elif name in NO_EVIDENCE_NORMALIZED_METRICS:
                weight = no_evidence_count
            else:
                weight = batch_size
            if weight == 0:
                continue
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * weight
            denominators[name] = denominators.get(name, 0) + weight
    if not denominators:
        raise RuntimeError("validation loader produced no examples")
    result = {name: value / denominators[name] for name, value in totals.items()}
    if any(paired_group_totals.values()):
        result.update(
            counterfactual_surface_pair_count=float(paired_group_totals["surface"]),
            counterfactual_primary_triplet_count=float(paired_group_totals["family"]),
            counterfactual_question_pair_count=float(paired_group_totals["question"]),
        )
    result["total"] = sum(
        getattr(criterion.weights, name) * result[name]
        for name in criterion.weights.__dataclass_fields__
        if name in result
    )
    return result


def git_identity(root: Path) -> Dict[str, Any]:
    """Capture the exact source state without failing outside a Git checkout."""

    def run(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def runtime_identity(device: torch.device) -> Dict[str, Any]:
    device_details: Dict[str, Any] = {
        "requested_device": str(device),
        "type": device.type,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_details.update(
            name=properties.name,
            capability=[properties.major, properties.minor],
            total_memory_bytes=properties.total_memory,
        )
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "device": device_details,
    }


def no_evidence_class_balance(records: Sequence[Any]) -> Dict[str, Any]:
    """Derive a train-split-only BCE weight and record its exact support.

    ``no_evidence=1`` is the positive BCE class.  Weighting it by
    answerable/no-evidence count makes the summed contribution of both classes
    equal without reading validation or test labels.  Degenerate legacy
    manifests keep neutral weighting and declare why balancing was unavailable.
    """

    total = len(records)
    if total <= 0:
        raise ValueError("cannot derive no-evidence balance from no records")
    no_evidence_count = sum(
        bool(getattr(record, "no_evidence", False)) for record in records
    )
    answerable_count = total - no_evidence_count
    both_classes_present = no_evidence_count > 0 and answerable_count > 0
    positive_weight = (
        answerable_count / no_evidence_count if both_classes_present else 1.0
    )
    return {
        "source": "training_manifest_only",
        "positive_class": "no_evidence",
        "answerable_records_↑": answerable_count,
        "no_evidence_records_↑": no_evidence_count,
        "both_classes_present": both_classes_present,
        "positive_weight_descriptive": positive_weight,
        "strategy": (
            "inverse_frequency_equal_total_class_contribution"
            if both_classes_present
            else "neutral_weight_degenerate_single_class_manifest"
        ),
    }


def loaded_project_source_identity(root: Path) -> Dict[str, Any]:
    """Hash every loaded Python source under the project, including untracked files."""

    resolved_root = root.resolve()
    by_path: Dict[Path, set[str]] = {}
    for module_name, module in sorted(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if not isinstance(raw_path, str):
            continue
        candidate = Path(raw_path)
        if candidate.suffix in {".pyc", ".pyo"}:
            try:
                candidate = Path(importlib.util.source_from_cache(str(candidate)))
            except ValueError:
                continue
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file() or (
            resolved != resolved_root and resolved_root not in resolved.parents
        ):
            continue
        by_path.setdefault(resolved, set()).add(module_name)

    trainer_path = Path(__file__).resolve()
    if trainer_path.is_file():
        by_path.setdefault(trainer_path, set()).add("__main__")
    files = []
    for path in sorted(by_path, key=lambda item: item.as_posix()):
        identity = file_identity(path)
        identity["relative_path"] = path.relative_to(resolved_root).as_posix()
        identity["module_names"] = sorted(by_path[path])
        files.append(identity)
    aggregate_payload = [
        (item["relative_path"], item["sha256"], item["size_bytes"]) for item in files
    ]
    aggregate_sha256 = hashlib.sha256(
        json.dumps(aggregate_payload, ensure_ascii=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "method": "loaded_python_sources_under_project_root",
        "root": str(resolved_root),
        "file_count": len(files),
        "aggregate_sha256": aggregate_sha256,
        "files": files,
    }


def dataset_identity(manifest: Path, dataset: QCESManifestDataset) -> Dict[str, Any]:
    identity = file_identity(manifest)
    identity.update(
        example_count=len(dataset),
        sample_rate=dataset.sample_rate,
        schema_versions=sorted(
            {
                str(getattr(record, "schema_version", "unknown"))
                for record in dataset.records
            }
        ),
        declared_splits=sorted(
            {str(getattr(record, "split", "unknown")) for record in dataset.records}
        ),
    )
    return identity


def dataset_build_profile(manifest: Path) -> str | None:
    """Read an adjacent builder profile when the manifest exposes one."""

    config_path = manifest.resolve().parent / "dataset_config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid adjacent dataset config: {config_path}") from exc
    profile = payload.get("profile") if isinstance(payload, dict) else None
    if profile is not None and not isinstance(profile, str):
        raise SystemExit(f"dataset profile must be a string: {config_path}")
    return profile


def split_overlap_audit(
    train_dataset: QCESManifestDataset,
    val_dataset: QCESManifestDataset,
) -> Dict[str, Any]:
    def values(dataset: QCESManifestDataset, name: str) -> set[str]:
        result = set()
        for record in dataset.records:
            value = getattr(record, name, None)
            if value is not None:
                result.add(str(value))
        return result

    def source_ids(dataset: QCESManifestDataset) -> set[str]:
        result = set()
        for record in dataset.records:
            for event in getattr(record, "events", ()):
                source_id = getattr(event, "source_id", None)
                if source_id is not None:
                    result.add(str(source_id))
        return result

    def sample_ids(dataset: QCESManifestDataset) -> set[str]:
        record_ids = [getattr(record, "sample_id", None) for record in dataset.records]
        if all(sample_id is not None for sample_id in record_ids):
            return {str(sample_id) for sample_id in record_ids}
        # Compatibility for lightweight audit-only fixtures and legacy eager
        # datasets. Derived v5 always takes the record branch because examples
        # is intentionally None.
        examples = getattr(dataset, "examples", None)
        if examples is None:
            raise ValueError("dataset records do not expose sample IDs")
        return {str(example.sample_id) for example in examples}

    train_sample_ids = sample_ids(train_dataset)
    val_sample_ids = sample_ids(val_dataset)
    sample_overlap = sorted(train_sample_ids & val_sample_ids)
    scene_overlap = sorted(
        values(train_dataset, "scene_id") & values(val_dataset, "scene_id")
    )
    source_overlap = sorted(source_ids(train_dataset) & source_ids(val_dataset))
    if sample_overlap:
        raise SystemExit(f"train/validation sample IDs overlap: {sample_overlap[:10]}")
    if scene_overlap:
        raise SystemExit(
            "train/validation scenes overlap, so checkpoint selection would leak "
            f"scene content: {scene_overlap[:10]}"
        )
    if source_overlap:
        raise SystemExit(
            "train/validation source IDs overlap, so checkpoint selection would "
            f"leak source recordings: {source_overlap[:10]}"
        )
    return {
        "sample_id_overlap_count": len(sample_overlap),
        "scene_id_overlap_count": len(scene_overlap),
        "source_id_overlap_count": len(source_overlap),
        "source_id_overlap": source_overlap,
    }


def initialize_audiosep_composer(
    model: QCESModel,
    checkpoint_path: Path,
    expected_config: QCESConfig,
    *,
    audiosep_config_identity: Mapping[str, Any],
    audiosep_checkpoint_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    """Strictly warm-start a composer bound to identical AudioSep assets.

    The optional refiner is deliberately not restored here: its residual heads
    must start at zero.  The only permitted model-config transition is enabling
    ``separator_aware_refiner`` on an otherwise identical QCES configuration.
    """

    resolved = checkpoint_path.resolve()
    if not resolved.is_file():
        raise SystemExit(f"initial QCES checkpoint not found: {resolved}")
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SystemExit(f"could not read initial QCES checkpoint: {resolved}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "qces_v1":
        raise SystemExit("initial checkpoint is not a qces_v1 checkpoint")
    if payload.get("backend") != "audiosep":
        raise SystemExit("initial checkpoint backend must be audiosep")
    raw_config = payload.get("config")
    composer_state = payload.get("composer_state_dict")
    if not isinstance(raw_config, dict) or not isinstance(composer_state, dict):
        raise SystemExit("initial AudioSep checkpoint lacks config or composer state")
    try:
        source_config = QCESConfig.from_dict(raw_config)
    except (TypeError, ValueError) as exc:
        raise SystemExit("initial checkpoint has an invalid QCES config") from exc
    source_comparable = source_config.to_dict()
    expected_comparable = expected_config.to_dict()
    source_refiner = source_comparable.pop("separator_aware_refiner")
    expected_refiner = expected_comparable.pop("separator_aware_refiner")
    source_refiner_mode = source_comparable.pop("separator_aware_refiner_mode")
    expected_refiner_mode = expected_comparable.pop("separator_aware_refiner_mode")
    if source_comparable != expected_comparable:
        differences = {
            key: [source_comparable.get(key), expected_comparable.get(key)]
            for key in sorted(set(source_comparable) | set(expected_comparable))
            if source_comparable.get(key) != expected_comparable.get(key)
        }
        raise SystemExit(
            "initial checkpoint QCES config mismatch (source, current): "
            f"{differences}"
        )
    if source_refiner:
        raise SystemExit(
            "initial checkpoint already enables the separator-aware refiner; "
            "--init-audiosep-qces-checkpoint restores only the base composer and "
            "would silently discard the learned refiner. Use a refiner-disabled "
            "base checkpoint when initializing a fresh refiner"
        )

    extra = payload.get("extra")
    audiosep = extra.get("audiosep") if isinstance(extra, dict) else None
    declared_config = audiosep.get("config") if isinstance(audiosep, dict) else None
    declared_checkpoint = (
        audiosep.get("checkpoint") if isinstance(audiosep, dict) else None
    )
    expected_config_hash = audiosep_config_identity.get("sha256")
    expected_checkpoint_hash = audiosep_checkpoint_identity.get("sha256")
    if (
        not isinstance(declared_config, dict)
        or declared_config.get("sha256") != expected_config_hash
    ):
        raise SystemExit(
            "initial checkpoint is not bound to the current AudioSep config"
        )
    if (
        not isinstance(declared_checkpoint, dict)
        or declared_checkpoint.get("sha256") != expected_checkpoint_hash
    ):
        raise SystemExit(
            "initial checkpoint is not bound to the current AudioSep checkpoint"
        )
    try:
        model.composer.load_state_dict(composer_state, strict=True)
    except RuntimeError as exc:
        raise SystemExit("initial composer state is incompatible") from exc
    return {
        "mode": "composer_warm_start_only",
        "checkpoint": file_identity(resolved),
        "source_backend": "audiosep",
        "source_separator_aware_refiner": source_refiner,
        "current_separator_aware_refiner": expected_refiner,
        "source_separator_aware_refiner_mode": source_refiner_mode,
        "current_separator_aware_refiner_mode": expected_refiner_mode,
        "allowed_config_difference": (
            "separator_aware_refiner_false_to_true_with_explicit_mode"
            if expected_refiner
            else "none"
        ),
        "audiosep_config_binding": {
            "source": dict(declared_config),
            "current": dict(audiosep_config_identity),
        },
        "audiosep_checkpoint_binding": {
            "source": dict(declared_checkpoint),
            "current": dict(audiosep_checkpoint_identity),
        },
        "refiner_state_restored": False,
    }


def set_semantic_adapter_trainable(model: QCESModel, trainable: bool) -> None:
    """Toggle only the learned semantic correction used by the composer."""

    for parameter in model.composer.semantic_head.parameters():
        parameter.requires_grad = trainable
    if trainable:
        model.composer.semantic_head.train(model.training)
    else:
        model.composer.semantic_head.eval()


def set_base_composer_trainable(
    model: QCESModel,
    trainable: bool,
    *,
    freeze_semantic_adapter: bool = False,
) -> None:
    """Toggle composer gradients and deterministic warm-up behavior."""

    for parameter in model.composer.parameters():
        parameter.requires_grad = trainable
    if trainable:
        model.composer.train(model.training)
    else:
        # A frozen warm-start must not drift through dropout statistics/noise.
        model.composer.eval()
    if freeze_semantic_adapter:
        # ``model.train()`` and a later base-composer unfreeze must never turn
        # this receipt-bound ablation back into the learned-prompt candidate.
        set_semantic_adapter_trainable(model, False)


def semantic_adapter_provenance(
    model: QCESModel, *, freeze_semantic_adapter: bool
) -> Dict[str, Any]:
    """Prove whether AudioSep receives a learned or direct-question prompt."""

    parameters = list(model.composer.semantic_head.parameters())
    final_layer = model.composer.semantic_head[-1]
    final_tensors = list(final_layer.parameters())
    final_nonzero = sum(
        int(torch.count_nonzero(tensor.detach())) for tensor in final_tensors
    )
    if freeze_semantic_adapter:
        if model.config.foundation_feature_mode != AUDIOSEP_CLAP_FOUNDATION_FEATURES:
            raise RuntimeError("frozen semantic adapter lacks AudioSep-CLAP inputs")
        if (
            model.config.foundation_semantic_mixing_mode
            != QUESTION_RESIDUAL_SEMANTIC_MIXING
        ):
            raise RuntimeError("frozen semantic adapter lacks question-residual mixing")
        if final_nonzero != 0:
            raise RuntimeError("direct-question semantic adapter is not exactly zero")
        if any(parameter.requires_grad for parameter in parameters):
            raise RuntimeError(
                "direct-question semantic adapter is unexpectedly trainable"
            )
    if freeze_semantic_adapter:
        ablation = "direct_full_question_clap"
        audiosep_condition = "normalize(full_question_clap)"
    elif (
        model.config.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES
        and model.config.foundation_semantic_mixing_mode
        == QUESTION_RESIDUAL_SEMANTIC_MIXING
    ):
        ablation = "learned_audio_question_semantic_adapter"
        audiosep_condition = (
            "normalize(full_question_clap + learned_audio_question_delta)"
        )
    elif model.config.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES:
        ablation = "learned_foundation_semantic_adapter"
        audiosep_condition = model.config.foundation_semantic_mixing_mode
    else:
        ablation = "legacy_learned_semantic_head"
        audiosep_condition = "normalize(learned_semantic_condition)"
    return {
        "freeze_semantic_adapter": freeze_semantic_adapter,
        "ablation": ablation,
        "audiosep_condition": audiosep_condition,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        "final_layer_nonzero_parameter_count": final_nonzero,
        "exact_identity_condition_at_initialization": final_nonzero == 0,
    }


def separator_aware_refiner_provenance(model: QCESModel) -> Dict[str, Any]:
    """Describe the live refiner architecture and fail on config drift."""

    refiner = getattr(model.separator, "separator_aware_refiner", None)
    enabled = model.config.separator_aware_refiner
    if enabled and refiner is None:
        raise RuntimeError("enabled separator-aware refiner was not constructed")
    if not enabled and refiner is not None:
        raise RuntimeError("disabled separator-aware refiner was constructed")
    if refiner is None:
        return {
            "enabled": False,
            "mode": None,
            "parameter_count": 0,
            "trainable_parameter_count": 0,
            "input_features": None,
            "rendering": None,
            "zero_initialized_residual_heads": False,
        }
    mode = model.config.separator_aware_refiner_mode
    live_mode = getattr(refiner, "mode", None)
    if live_mode != mode:
        raise RuntimeError(
            "separator-aware refiner mode differs from model config: "
            f"live={live_mode!r}, config={mode!r}"
        )
    return {
        "enabled": True,
        "mode": mode,
        "parameter_count": sum(parameter.numel() for parameter in refiner.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in refiner.parameters()
            if parameter.requires_grad
        ),
        "input_features": (
            "base role logits + raw-stem/mixture frame log-RMS ratio; "
            "no spectrogram encoder"
            if mode == "relative_energy"
            else (
                "base composer frames + raw-stem log-magnitude encoder + "
                "raw-stem/mixture relative energy"
            )
        ),
        "rendering": (
            "one AudioSep call; smooth residual role gate on raw evidence; "
            "residual = mixture - evidence"
        ),
        "zero_initialized_residual_heads": True,
    }


def validate_refiner_training_args(args: argparse.Namespace) -> None:
    """Reject refiner, warm-up, and semantic-mode combinations fail-closed."""

    if args.separator_aware_refiner and args.backend != "audiosep":
        raise SystemExit("--separator-aware-refiner requires --backend audiosep")
    if args.separator_aware_refiner_mode not in SEPARATOR_AWARE_REFINER_MODES:
        raise SystemExit(
            "--separator-aware-refiner-mode must be one of "
            f"{SEPARATOR_AWARE_REFINER_MODES}"
        )
    if not args.separator_aware_refiner and args.separator_aware_refiner_mode != "full":
        raise SystemExit(
            "--separator-aware-refiner-mode relative_energy requires "
            "--separator-aware-refiner"
        )
    if args.init_audiosep_qces_checkpoint is not None and args.backend != "audiosep":
        raise SystemExit("--init-audiosep-qces-checkpoint requires --backend audiosep")
    if args.freeze_base_composer_steps > 0 and not args.separator_aware_refiner:
        raise SystemExit(
            "--freeze-base-composer-steps requires --separator-aware-refiner"
        )
    if (
        args.freeze_base_composer_steps > 0
        and args.init_audiosep_qces_checkpoint is None
    ):
        raise SystemExit(
            "--freeze-base-composer-steps requires "
            "--init-audiosep-qces-checkpoint; freezing a randomly initialized "
            "base composer would train the refiner against arbitrary semantic "
            "and temporal conditions"
        )
    if args.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE:
        if args.backend != "audiosep":
            raise SystemExit(
                "--semantic-separation-mode dual_role requires --backend audiosep"
            )
        if args.temporal_role_mode != OVERLAP_AWARE_TEMPORAL_ROLE_MODE:
            raise SystemExit(
                "dual_role semantic separation requires --temporal-role-mode "
                "independent_sigmoid"
            )
        if args.separator_aware_refiner:
            raise SystemExit(
                "dual_role semantic separation cannot use the single-raw-stem "
                "separator-aware refiner"
            )
        if args.role_semantic_targets is None:
            raise SystemExit(
                "dual_role training requires --role-semantic-targets; role "
                "labels must not be guessed from question text"
            )
        if args.freeze_semantic_adapter:
            if args.role_semantic_weight != 0 or args.same_semantic_weight <= 0:
                raise SystemExit(
                    "dual_role direct-question ablation requires "
                    "--role-semantic-weight 0 and positive "
                    "--same-semantic-weight"
                )
        elif args.role_semantic_weight <= 0 or args.same_semantic_weight <= 0:
            raise SystemExit(
                "dual_role training requires positive --role-semantic-weight "
                "and --same-semantic-weight"
            )
        if args.semantic_targets is not None or args.semantic_weight != 0:
            raise SystemExit(
                "dual_role uses role-specific semantic supervision; legacy "
                "--semantic-targets/--semantic-weight must be disabled"
            )
    elif args.semantic_separation_mode == UNION_SINGLE_SEMANTIC_MODE:
        if (
            args.role_semantic_targets is not None
            or args.val_role_semantic_targets is not None
            or args.role_semantic_weight != 0
            or args.same_semantic_weight != 0
        ):
            raise SystemExit(
                "role-semantic caches/weights require "
                "--semantic-separation-mode dual_role"
            )
    else:
        raise SystemExit(
            f"unsupported semantic separation mode: "
            f"{args.semantic_separation_mode!r}"
        )


def validate_foundation_training_args(args: argparse.Namespace) -> None:
    """Fail closed on incomplete or temporally misaligned CLAP cache usage."""

    mode = args.foundation_feature_mode
    if mode == NO_FOUNDATION_FEATURES:
        if args.foundation_semantic_mixing_mode != LEGACY_BOUNDED_SEMANTIC_RESIDUAL:
            raise SystemExit(
                "a non-legacy --foundation-semantic-mixing-mode requires "
                "--foundation-feature-mode audiosep_clap"
            )
        if (
            args.foundation_feature_cache is not None
            or args.val_foundation_feature_cache is not None
        ):
            raise SystemExit(
                "foundation cache arguments require "
                "--foundation-feature-mode audiosep_clap"
            )
        return
    if mode != AUDIOSEP_CLAP_FOUNDATION_FEATURES:
        raise SystemExit(f"unsupported foundation feature mode: {mode!r}")
    if args.backend != "audiosep":
        raise SystemExit(
            "--foundation-feature-mode audiosep_clap requires --backend audiosep"
        )
    if args.foundation_feature_cache is None:
        raise SystemExit(
            "audiosep_clap mode requires --foundation-feature-cache for training"
        )
    if args.val_manifest is None and args.val_foundation_feature_cache is not None:
        raise SystemExit("--val-foundation-feature-cache requires --val-manifest")
    if args.val_manifest is not None and args.val_foundation_feature_cache is None:
        raise SystemExit(
            "audiosep_clap validation requires --val-foundation-feature-cache"
        )
    if not math.isclose(
        args.crop_seconds,
        CLAP_CACHE_DURATION_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise SystemExit(
            "audiosep_clap cache frames cover the canonical full 10 s scene; "
            "set --crop-seconds 10 to prevent temporal feature misalignment"
        )


def validate_semantic_adapter_training_args(args: argparse.Namespace) -> None:
    """Fail closed unless the direct-question baseline changes one component."""

    if not args.freeze_semantic_adapter:
        return
    if args.foundation_feature_mode != AUDIOSEP_CLAP_FOUNDATION_FEATURES:
        raise SystemExit(
            "--freeze-semantic-adapter requires "
            "--foundation-feature-mode audiosep_clap"
        )
    if args.foundation_semantic_mixing_mode != QUESTION_RESIDUAL_SEMANTIC_MIXING:
        raise SystemExit(
            "--freeze-semantic-adapter requires "
            "--foundation-semantic-mixing-mode question_residual"
        )
    if args.init_audiosep_qces_checkpoint is not None:
        raise SystemExit(
            "--freeze-semantic-adapter forbids a warm-start checkpoint; the "
            "direct-question condition must be proven from exact zero initialization"
        )
    if args.freeze_base_composer_steps != 0:
        raise SystemExit(
            "--freeze-semantic-adapter cannot be combined with base-composer warm-up"
        )
    if (
        args.semantic_targets is not None
        or args.val_semantic_targets is not None
        or args.semantic_weight != 0
    ):
        raise SystemExit(
            "the direct-question ablation requires legacy semantic targets and "
            "--semantic-weight to be disabled"
        )
    if args.role_semantic_weight != 0:
        raise SystemExit(
            "the direct-question ablation requires --role-semantic-weight 0; "
            "the frozen prompt cannot optimize role CLAP alignment"
        )


def make_checkpoint_payload(
    model: QCESModel,
    backend: str,
    config: QCESConfig,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    if model.config != config:
        raise ValueError("checkpoint config differs from the live model config")
    return model.checkpoint_payload(backend=backend, extra=extra)


def atomic_torch_save(
    payload: Dict[str, Any], path: Path, *, overwrite: bool = True
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if overwrite:
        torch.save(payload, temporary)
        temporary.replace(path)
        return
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    created_temporary = False
    try:
        # Exclusive creation makes an interrupted run or a concurrent writer
        # fail closed instead of silently replacing an epoch snapshot.
        with temporary.open("xb") as handle:
            created_temporary = True
            torch.save(payload, handle)
        os.link(temporary, path)
    finally:
        if created_temporary:
            temporary.unlink(missing_ok=True)


def retained_epoch_checkpoint_path(output_dir: Path, epoch: int) -> Path:
    if epoch <= 0:
        raise ValueError("retained checkpoint epoch must be positive")
    return output_dir / "epoch_checkpoints" / f"epoch_{epoch:04d}_checkpoint.pt"


def retained_epoch_checkpoint_extra(
    base: Dict[str, Any],
    *,
    epoch: int,
    global_step: int,
    stop_reason: str,
    best_epoch: int | None,
    best_global_step: int | None,
    best_value: float | None,
    metrics_at_checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Build an exact, self-describing snapshot for one retained epoch."""

    extra = checkpoint_extra(
        base,
        run_epochs_completed=epoch,
        run_global_step=global_step,
        checkpoint_epoch=epoch,
        checkpoint_global_step=global_step,
        checkpoint_role="retained_epoch",
        stop_reason=stop_reason,
        best_epoch=best_epoch,
        best_global_step=best_global_step,
        best_value=best_value,
    )
    validation = metrics_at_checkpoint.get("validation")
    extra["epoch_retention"] = {
        "policy": "save_every_epoch",
        "immutable": True,
        "epoch": epoch,
        "global_step": global_step,
        "checkpoint": str(checkpoint_path),
        "metrics_at_checkpoint": dict(metrics_at_checkpoint),
        "validation_metrics_at_checkpoint": (
            dict(validation) if isinstance(validation, Mapping) else None
        ),
    }
    return extra


def checkpoint_extra(
    base: Dict[str, Any],
    *,
    run_epochs_completed: int,
    run_global_step: int,
    checkpoint_epoch: int,
    checkpoint_global_step: int,
    checkpoint_role: str,
    stop_reason: str,
    best_epoch: int | None,
    best_global_step: int | None,
    best_value: float | None,
) -> Dict[str, Any]:
    return {
        **base,
        "epochs_completed": run_epochs_completed,
        "global_step": run_global_step,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_step": checkpoint_global_step,
        "checkpoint_role": checkpoint_role,
        "stop_reason": stop_reason,
        "best_validation_epoch": best_epoch,
        "best_validation_global_step": best_global_step,
        "best_validation_value": best_value,
    }


def counterfactual_activation(
    weights: Mapping[str, float], force_batching: bool
) -> Dict[str, bool]:
    """Separate paired-objective activation from schedule-matched batching."""

    surface_objective = any(weights[name] > 0 for name in SURFACE_LOSS_NAMES)
    family_objective = any(weights[name] > 0 for name in FAMILY_LOSS_NAMES)
    question_objective = any(weights[name] > 0 for name in QUESTION_LOSS_NAMES)
    objective = surface_objective or family_objective or question_objective
    return {
        "surface_objective": surface_objective,
        "family_objective": family_objective,
        "question_objective": question_objective,
        "objective": objective,
        "surface_batched": surface_objective or force_batching,
        "family_batched": family_objective or force_batching,
        "question_batched": question_objective or force_batching,
        "batching": objective or force_batching,
        "forced_schedule_matched_control": force_batching and not objective,
    }


def main() -> None:
    run_started_monotonic = time.monotonic()
    args = parse_args()
    counterfactual_weights = {
        "surface_semantic_invariance": args.surface_semantic_invariance_weight,
        "surface_role_invariance": args.surface_role_invariance_weight,
        "surface_no_evidence_invariance": (args.surface_no_evidence_invariance_weight),
        "surface_evidence_invariance": args.surface_evidence_invariance_weight,
        "family_temporal_delta": args.family_temporal_delta_weight,
        "family_evidence_delta": args.family_evidence_delta_weight,
        "family_no_evidence_transition": (args.family_no_evidence_transition_weight),
        "question_temporal_delta": args.question_temporal_delta_weight,
        "question_evidence_delta": args.question_evidence_delta_weight,
    }
    counterfactual_modes = counterfactual_activation(
        counterfactual_weights, args.force_counterfactual_batching
    )
    surface_counterfactual_enabled = counterfactual_modes["surface_objective"]
    family_counterfactual_enabled = counterfactual_modes["family_objective"]
    question_counterfactual_enabled = counterfactual_modes["question_objective"]
    counterfactual_enabled = counterfactual_modes["objective"]
    surface_counterfactual_batched = counterfactual_modes["surface_batched"]
    family_counterfactual_batched = counterfactual_modes["family_batched"]
    question_counterfactual_batched = counterfactual_modes["question_batched"]
    counterfactual_batching_enabled = counterfactual_modes["batching"]
    core_loss_weights = {
        field_name: float(getattr(args, argument_name))
        for field_name, argument_name in CORE_LOSS_WEIGHT_ARGS.items()
    }
    if (
        args.epochs <= 0
        or args.batch_size <= 0
        or args.learning_rate <= 0
        or args.crop_seconds <= 0
        or args.max_steps < 0
        or args.num_workers < 0
        or args.semantic_weight < 0
        or args.role_semantic_weight < 0
        or args.same_semantic_weight < 0
        or args.role_relative_weight < 0
        or args.weakest_role_weight < 0
        or any(weight < 0 for weight in core_loss_weights.values())
        or any(weight < 0 for weight in counterfactual_weights.values())
        or not 0.0 <= args.counterfactual_transition_margin <= 1.0
        or args.log_every_epochs <= 0
        or args.early_stopping_patience < 0
        or args.selection_min_delta < 0
        or args.freeze_base_composer_steps < 0
        or not 0.0 <= args.dropout < 1.0
    ):
        raise SystemExit(
            "epochs, batch-size, learning-rate, crop-seconds, and log-every-epochs "
            "must be positive; max-steps, num-workers, patience, min-delta, and loss "
            "weights and freeze-base-composer-steps must be non-negative; "
            "dropout and counterfactual-transition-margin must be in [0, 1]"
        )
    if args.val_semantic_targets is not None and args.val_manifest is None:
        raise SystemExit("--val-semantic-targets requires --val-manifest")
    if args.val_role_semantic_targets is not None and args.val_manifest is None:
        raise SystemExit("--val-role-semantic-targets requires --val-manifest")
    if args.early_stopping_patience > 0 and args.val_manifest is None:
        raise SystemExit("early stopping requires --val-manifest")
    selected_counterfactual_group_enabled = (
        (
            args.selection_metric.startswith("surface_")
            and surface_counterfactual_batched
        )
        or (
            args.selection_metric.startswith("family_")
            and family_counterfactual_batched
        )
        or (
            args.selection_metric.startswith("question_")
            and question_counterfactual_batched
        )
    )
    if args.selection_metric in COUNTERFACTUAL_METRIC_DIRECTIONS and not (
        selected_counterfactual_group_enabled
    ):
        raise SystemExit(
            "a counterfactual selection metric requires at least one positive "
            "paired loss weight so its complete groups are batched"
        )
    if (
        args.selection_metric in {"same_semantic_brier", "same_semantic_accuracy"}
        and args.semantic_separation_mode != DUAL_ROLE_SEMANTIC_MODE
    ):
        raise SystemExit(
            "same-semantic checkpoint selection requires "
            "--semantic-separation-mode dual_role"
        )
    validate_refiner_training_args(args)
    validate_foundation_training_args(args)
    validate_semantic_adapter_training_args(args)
    if (
        args.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE
        and args.val_manifest is not None
        and args.val_role_semantic_targets is None
    ):
        raise SystemExit(
            "dual_role validation requires --val-role-semantic-targets built "
            "for --val-manifest"
        )
    manifest = args.manifest.resolve()
    val_manifest = args.val_manifest.resolve() if args.val_manifest else None
    if not manifest.is_file():
        raise SystemExit(f"manifest not found: {manifest}")
    if val_manifest is not None and not val_manifest.is_file():
        raise SystemExit(f"validation manifest not found: {val_manifest}")
    if val_manifest == manifest:
        raise SystemExit("training and validation manifests must be different files")
    output_dir, output_was_nonempty = check_output_directory(
        args.output_dir, args.overwrite
    )
    reproducibility = configure_reproducibility(args.seed, args.deterministic)
    device = resolve_device(args.device)
    validate_precision_mode(args.precision, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.backend == "audiosep":
        required = (args.audiosep_root, args.audiosep_config, args.audiosep_checkpoint)
        if any(path is None for path in required):
            raise SystemExit(
                "audiosep backend requires --audiosep-root, --audiosep-config, "
                "and --audiosep-checkpoint"
            )
    if (
        args.semantic_targets is not None
        or args.val_semantic_targets is not None
        or args.role_semantic_targets is not None
        or args.val_role_semantic_targets is not None
    ) and args.audiosep_checkpoint is None:
        raise SystemExit(
            "AudioSep semantic-target caches require --audiosep-checkpoint so "
            "their CLAP conditions can be bound to the exact checkpoint"
        )
    audiosep_checkpoint_identity = None
    if args.audiosep_checkpoint is not None:
        checkpoint_path = args.audiosep_checkpoint.resolve()
        if not checkpoint_path.is_file():
            raise SystemExit(f"AudioSep checkpoint not found: {checkpoint_path}")
        audiosep_checkpoint_identity = file_identity(checkpoint_path)
    foundation_audiosep_source_identity = None
    if args.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES:
        assert args.audiosep_root is not None
        foundation_audiosep_source_identity = audiosep_source_tree_identity(
            args.audiosep_root
        )

    probe_dataset = QCESManifestDataset(manifest, crop_samples=None, seed=args.seed)
    build_profile = dataset_build_profile(manifest)
    if (
        build_profile == "paper"
        and args.temporal_role_mode != OVERLAP_AWARE_TEMPORAL_ROLE_MODE
    ):
        raise SystemExit(
            "a paper-profile QCES-v5 manifest requires "
            "--temporal-role-mode independent_sigmoid so overlapping "
            "anchor/answer targets are representable"
        )
    config = QCESConfig(
        sample_rate=probe_dataset.sample_rate,
        dropout=args.dropout,
        temporal_role_mode=args.temporal_role_mode,
        semantic_separation_mode=args.semantic_separation_mode,
        separator_aware_refiner=args.separator_aware_refiner,
        separator_aware_refiner_mode=args.separator_aware_refiner_mode,
        foundation_feature_mode=args.foundation_feature_mode,
        foundation_semantic_mixing_mode=args.foundation_semantic_mixing_mode,
    )
    crop_samples = int(round(args.crop_seconds * config.sample_rate))
    crop_samples = max(crop_samples, config.n_fft)
    dataset = QCESManifestDataset(
        manifest,
        crop_samples=crop_samples,
        random_crop=True,
        seed=args.seed,
        align_v5_family_crops=counterfactual_batching_enabled,
    )
    no_evidence_balance = no_evidence_class_balance(dataset.records)
    val_dataset = (
        QCESManifestDataset(
            val_manifest,
            crop_samples=crop_samples,
            random_crop=False,
            seed=args.seed,
            align_v5_family_crops=counterfactual_batching_enabled,
        )
        if val_manifest is not None
        else None
    )
    if val_dataset is not None:
        declared_val_splits = {
            str(getattr(record, "split", "unknown")).lower()
            for record in val_dataset.records
        }
        if "test" in declared_val_splits:
            raise SystemExit(
                "--val-manifest declares split=test; test data cannot select "
                "checkpoints"
            )
    if val_dataset is not None and val_dataset.sample_rate != dataset.sample_rate:
        raise SystemExit(
            "training and validation sample rates differ: "
            f"{dataset.sample_rate} != {val_dataset.sample_rate}"
        )
    overlap_audit = (
        split_overlap_audit(dataset, val_dataset) if val_dataset is not None else None
    )
    counterfactual_plan = None
    val_counterfactual_plan = None
    if counterfactual_batching_enabled:
        try:
            counterfactual_plan = build_counterfactual_group_plan(
                dataset.records,
                enable_surface=surface_counterfactual_batched,
                enable_family=family_counterfactual_batched,
                enable_question=question_counterfactual_batched,
            )
            if val_dataset is not None:
                val_counterfactual_plan = build_counterfactual_group_plan(
                    val_dataset.records,
                    enable_surface=surface_counterfactual_batched,
                    enable_family=family_counterfactual_batched,
                    enable_question=question_counterfactual_batched,
                )
        except ValueError as exc:
            raise SystemExit(f"invalid counterfactual grouping: {exc}") from exc
    tokenizer = StableHashTokenizer(
        vocab_size=config.vocab_size, max_length=config.max_question_tokens
    )
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_counterfactual_sampler = None
    val_counterfactual_sampler = None
    if counterfactual_plan is not None:
        try:
            train_counterfactual_sampler = CounterfactualBatchSampler(
                counterfactual_plan,
                args.batch_size,
                seed=args.seed,
                shuffle=True,
            )
            loader = DataLoader(
                dataset,
                batch_sampler=train_counterfactual_sampler,
                num_workers=args.num_workers,
                collate_fn=partial(
                    collate_qces,
                    tokenizer=tokenizer,
                    counterfactual_plan=counterfactual_plan,
                ),
                generator=train_generator,
                worker_init_fn=seed_data_worker,
            )
        except ValueError as exc:
            raise SystemExit(f"invalid counterfactual batch plan: {exc}") from exc
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=partial(collate_qces, tokenizer=tokenizer),
            generator=train_generator,
            worker_init_fn=seed_data_worker,
        )
    if val_dataset is not None and val_counterfactual_plan is not None:
        try:
            val_counterfactual_sampler = CounterfactualBatchSampler(
                val_counterfactual_plan,
                args.batch_size,
                seed=args.seed,
                shuffle=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_sampler=val_counterfactual_sampler,
                num_workers=args.num_workers,
                collate_fn=partial(
                    collate_qces,
                    tokenizer=tokenizer,
                    counterfactual_plan=val_counterfactual_plan,
                ),
                worker_init_fn=seed_data_worker,
            )
        except ValueError as exc:
            raise SystemExit(
                f"invalid validation counterfactual batch plan: {exc}"
            ) from exc
    elif val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=partial(collate_qces, tokenizer=tokenizer),
            worker_init_fn=seed_data_worker,
        )
    else:
        val_loader = None

    semantic_targets = None
    semantic_cache_identity = None
    if args.semantic_targets is not None:
        assert audiosep_checkpoint_identity is not None
        semantic_targets, semantic_cache_identity = load_semantic_target_cache(
            args.semantic_targets,
            manifest,
            [record.sample_id for record in dataset.records],
            "training",
            expected_no_evidence_ids=[
                record.sample_id
                for record in dataset.records
                if bool(getattr(record, "no_evidence", False))
            ],
            expected_dim=config.condition_dim,
            expected_schema_versions=[
                str(getattr(record, "schema_version", "unknown"))
                for record in dataset.records
            ],
            audiosep_checkpoint_identity=audiosep_checkpoint_identity,
        )
    if args.semantic_weight > 0 and semantic_targets is None:
        raise SystemExit("positive semantic-weight requires --semantic-targets")
    val_semantic_targets = None
    val_semantic_cache_identity = None
    if args.val_semantic_targets is not None:
        assert val_manifest is not None and val_dataset is not None
        assert audiosep_checkpoint_identity is not None
        val_semantic_targets, val_semantic_cache_identity = load_semantic_target_cache(
            args.val_semantic_targets,
            val_manifest,
            [record.sample_id for record in val_dataset.records],
            "validation",
            expected_no_evidence_ids=[
                record.sample_id
                for record in val_dataset.records
                if bool(getattr(record, "no_evidence", False))
            ],
            expected_dim=config.condition_dim,
            expected_schema_versions=[
                str(getattr(record, "schema_version", "unknown"))
                for record in val_dataset.records
            ],
            audiosep_checkpoint_identity=audiosep_checkpoint_identity,
        )
    if (
        val_dataset is not None
        and args.semantic_weight > 0
        and val_semantic_targets is None
    ):
        raise SystemExit(
            "positive semantic-weight with validation requires "
            "--val-semantic-targets built for --val-manifest"
        )

    role_semantic_targets = None
    role_semantic_cache_identity = None
    if args.role_semantic_targets is not None:
        assert audiosep_checkpoint_identity is not None
        role_semantic_targets, role_semantic_cache_identity = (
            load_role_semantic_target_cache(
                args.role_semantic_targets,
                manifest,
                [record.sample_id for record in dataset.records],
                [
                    record.sample_id
                    for record in dataset.records
                    if bool(getattr(record, "no_evidence", False))
                ],
                same_semantic_targets_from_records(dataset.records, "training"),
                "training",
                expected_dim=config.condition_dim,
                expected_schema_versions=[
                    str(getattr(record, "schema_version", "unknown"))
                    for record in dataset.records
                ],
                audiosep_checkpoint_identity=audiosep_checkpoint_identity,
            )
        )
    val_role_semantic_targets = None
    val_role_semantic_cache_identity = None
    if args.val_role_semantic_targets is not None:
        assert val_manifest is not None and val_dataset is not None
        assert audiosep_checkpoint_identity is not None
        val_role_semantic_targets, val_role_semantic_cache_identity = (
            load_role_semantic_target_cache(
                args.val_role_semantic_targets,
                val_manifest,
                [record.sample_id for record in val_dataset.records],
                [
                    record.sample_id
                    for record in val_dataset.records
                    if bool(getattr(record, "no_evidence", False))
                ],
                same_semantic_targets_from_records(val_dataset.records, "validation"),
                "validation",
                expected_dim=config.condition_dim,
                expected_schema_versions=[
                    str(getattr(record, "schema_version", "unknown"))
                    for record in val_dataset.records
                ],
                audiosep_checkpoint_identity=audiosep_checkpoint_identity,
            )
        )
    if (
        args.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE
        and role_semantic_targets is None
    ):
        raise SystemExit(
            "dual_role training did not load role-specific semantic targets"
        )
    if args.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE:
        assert role_semantic_cache_identity is not None
        if (
            role_semantic_cache_identity["same_semantic_count"] <= 0
            or role_semantic_cache_identity["different_semantic_count"] <= 0
        ):
            raise SystemExit(
                "dual_role same-semantic routing needs both same-label and "
                "different-label training records for identifiable supervision"
            )

    foundation_feature_cache = None
    val_foundation_feature_cache = None
    if args.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES:
        assert args.foundation_feature_cache is not None
        assert audiosep_checkpoint_identity is not None
        assert foundation_audiosep_source_identity is not None
        foundation_feature_cache = load_foundation_feature_cache(
            args.foundation_feature_cache,
            manifest,
            dataset.records,
            "training",
            audiosep_checkpoint_identity=audiosep_checkpoint_identity,
            audiosep_source_identity=foundation_audiosep_source_identity,
        )
        if val_dataset is not None:
            assert val_manifest is not None
            assert args.val_foundation_feature_cache is not None
            val_foundation_feature_cache = load_foundation_feature_cache(
                args.val_foundation_feature_cache,
                val_manifest,
                val_dataset.records,
                "validation",
                audiosep_checkpoint_identity=audiosep_checkpoint_identity,
                audiosep_source_identity=foundation_audiosep_source_identity,
            )

    separator = None
    if args.backend == "complex":
        separator = PhaseAwareComplexMaskSeparator(config)
    if args.backend == "audiosep":
        assert args.audiosep_root is not None
        assert args.audiosep_config is not None
        assert args.audiosep_checkpoint is not None
        separator = AudioSepConditionedAdapter.from_repository(
            repository_root=args.audiosep_root,
            config_path=args.audiosep_config,
            checkpoint_path=args.audiosep_checkpoint,
            device=device,
            freeze_separator=True,
            qces_config=config,
        )
    model = QCESModel(config, separator=separator).to(device)
    initialization_identity = None
    from_scratch_composer_receipt = None
    if args.init_audiosep_qces_checkpoint is not None:
        assert args.audiosep_config is not None
        assert audiosep_checkpoint_identity is not None
        initialization_identity = initialize_audiosep_composer(
            model,
            args.init_audiosep_qces_checkpoint,
            config,
            audiosep_config_identity=file_identity(args.audiosep_config),
            audiosep_checkpoint_identity=audiosep_checkpoint_identity,
        )
    else:
        from_scratch_composer_receipt = composer_initialization_receipt(
            model.composer,
            seed=args.seed,
        )
    refiner_identity = separator_aware_refiner_provenance(model)
    if args.freeze_semantic_adapter:
        # Freeze before optimizer parameter discovery so the baseline cannot
        # silently retain unused trainable semantic-adapter tensors.
        set_semantic_adapter_trainable(model, False)
    semantic_adapter_identity = semantic_adapter_provenance(
        model, freeze_semantic_adapter=args.freeze_semantic_adapter
    )
    criterion = QCESLoss(
        weights=LossWeights(
            **core_loss_weights,
            semantic_alignment=args.semantic_weight,
            anchor_semantic_alignment=args.role_semantic_weight,
            answer_semantic_alignment=args.role_semantic_weight,
            same_semantic_classification=args.same_semantic_weight,
            role_relative_waveform=args.role_relative_weight,
            weakest_role_waveform=args.weakest_role_weight,
            **counterfactual_weights,
        ),
        counterfactual_transition_margin=args.counterfactual_transition_margin,
        no_evidence_positive_weight=no_evidence_balance["positive_weight_descriptive"],
    ).to(device)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    scaler = create_gradient_scaler(args.precision)
    set_base_composer_trainable(
        model,
        trainable=args.freeze_base_composer_steps == 0,
        freeze_semantic_adapter=args.freeze_semantic_adapter,
    )

    train_manifest_identity = dataset_identity(manifest, dataset)
    val_manifest_identity = (
        dataset_identity(val_manifest, val_dataset)
        if val_manifest is not None and val_dataset is not None
        else None
    )
    audiosep_identity = None
    if args.backend == "audiosep":
        assert args.audiosep_root is not None
        assert args.audiosep_config is not None
        assert args.audiosep_checkpoint is not None
        audiosep_identity = {
            "repository_root": str(args.audiosep_root.resolve()),
            "repository_git": git_identity(args.audiosep_root.resolve()),
            "config": file_identity(args.audiosep_config),
            "checkpoint": audiosep_checkpoint_identity,
            "deterministic_reflect_stft_patch": bool(
                getattr(separator, "deterministic_stft_patch", False)
            ),
            "frozen_backbone": True,
            "backbone_trainable_parameter_count": sum(
                parameter.numel()
                for parameter in separator.ss_model.parameters()
                if parameter.requires_grad
            ),
        }
        if audiosep_identity["backbone_trainable_parameter_count"] != 0:
            raise RuntimeError("AudioSep backbone is unexpectedly trainable")
    direction = SELECTION_DIRECTIONS[args.selection_metric]
    direction_arrow = "↓" if direction == "minimize" else "↑"
    base_extra: Dict[str, Any] = {
        # Preserve the original top-level fields for existing consumers.
        "manifest": str(manifest),
        "manifest_sha256": train_manifest_identity["sha256"],
        "val_manifest": str(val_manifest) if val_manifest else None,
        "val_manifest_sha256": (
            val_manifest_identity["sha256"] if val_manifest_identity else None
        ),
        "manifests": {
            "train": train_manifest_identity,
            "validation": val_manifest_identity,
        },
        "dataset_build_profile": build_profile,
        "split_overlap_audit": overlap_audit,
        "epochs_requested": args.epochs,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "crop_seconds": args.crop_seconds,
        "crop_samples": crop_samples,
        "max_steps": args.max_steps,
        "num_workers": args.num_workers,
        "optimizer": {
            "name": "AdamW",
            "weight_decay": 1e-4,
            "gradient_clip_norm": 5.0,
        },
        "training_config": {
            "backend": args.backend,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "dropout": args.dropout,
            "crop_seconds": args.crop_seconds,
            "crop_samples": crop_samples,
            "max_steps": args.max_steps,
            "num_workers": args.num_workers,
            "log_every_epochs": args.log_every_epochs,
            "seed": args.seed,
            "device_requested": args.device,
            "precision": args.precision,
            "deterministic": args.deterministic,
            "selection_metric": args.selection_metric,
            "selection_direction": direction,
            "selection_min_delta": args.selection_min_delta,
            "early_stopping_patience": args.early_stopping_patience,
            "save_every_epoch": args.save_every_epoch,
            "temporal_role_mode": args.temporal_role_mode,
            "semantic_separation_mode": args.semantic_separation_mode,
            "foundation_feature_mode": args.foundation_feature_mode,
            "foundation_semantic_mixing_mode": (args.foundation_semantic_mixing_mode),
            "freeze_semantic_adapter": args.freeze_semantic_adapter,
            "foundation_feature_cache": (
                str(args.foundation_feature_cache.resolve())
                if args.foundation_feature_cache
                else None
            ),
            "val_foundation_feature_cache": (
                str(args.val_foundation_feature_cache.resolve())
                if args.val_foundation_feature_cache
                else None
            ),
            "separator_aware_refiner": args.separator_aware_refiner,
            "separator_aware_refiner_mode": args.separator_aware_refiner_mode,
            "freeze_base_composer_steps": args.freeze_base_composer_steps,
            "counterfactual_enabled": counterfactual_enabled,
            "counterfactual_batching_enabled": counterfactual_batching_enabled,
            "force_counterfactual_batching": args.force_counterfactual_batching,
            "counterfactual_transition_margin": (args.counterfactual_transition_margin),
        },
        "model_config": config.to_dict(),
        "semantic_separation": {
            "mode": config.semantic_separation_mode,
            "legacy_default": UNION_SINGLE_SEMANTIC_MODE,
            "candidate_mode": DUAL_ROLE_SEMANTIC_MODE,
            "frozen_audiosep_backbone": args.backend == "audiosep",
            "physical_backbone_forwards_per_batch": (
                1 if args.backend == "audiosep" else None
            ),
            "effective_separator_evaluations_per_record": (
                2
                if config.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE
                else 1 if args.backend == "audiosep" else None
            ),
            "role_condition_pooling": (
                "independent anchor/answer probability-weighted frame pooling"
                if config.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE
                else "anchor_or_answer evidence-union frame pooling"
            ),
            "shared_target_routing": (
                {
                    "type": "fully_differentiable_learned_interpolation",
                    "formula": (
                        "E=(1-p_same)*(A*p_anchor+B*p_answer)+" "p_same*(A+B)/2*p_union"
                    ),
                    "thresholded_exact_reuse": False,
                    "training_target_source": (
                        "role-event label equality from role-semantic cache"
                    ),
                    "model_forward_inputs": [
                        "mixture",
                        "question_ids",
                        "question_mask",
                        *(
                            ["question_clap", "scene_clap"]
                            if config.foundation_feature_mode
                            == AUDIOSEP_CLAP_FOUNDATION_FEATURES
                            else []
                        ),
                    ],
                    "same_label_target_consumption": (
                        "loss and offline calibration metrics after model forward"
                    ),
                    "validation_test_label_access_at_inference": False,
                }
                if config.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE
                else None
            ),
            "claim_boundary": (
                "learned question-conditioned control of a frozen AudioSep "
                "backbone; not an AudioSep improvement"
            ),
        },
        "foundation_features": {
            "mode": config.foundation_feature_mode,
            "semantic_mixing_mode": config.foundation_semantic_mixing_mode,
            "legacy_default": NO_FOUNDATION_FEATURES,
            "train_cache": (
                dict(foundation_feature_cache.identity)
                if foundation_feature_cache is not None
                else None
            ),
            "validation_cache": (
                dict(val_foundation_feature_cache.identity)
                if val_foundation_feature_cache is not None
                else None
            ),
            "model_inputs": (
                ["question_clap[B,512]", "scene_clap[B,32,512]"]
                if config.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES
                else []
            ),
            "semantic_condition": (
                (
                    (
                        "normalize(full_question_clap)"
                        if args.freeze_semantic_adapter
                        else "normalize(full_question_clap + "
                        "learned_audio_question_delta)"
                    )
                    if config.foundation_semantic_mixing_mode
                    == QUESTION_RESIDUAL_SEMANTIC_MIXING
                    else (
                        "normalize((1-sigmoid(gate_logit))*"
                        "role_pooled_scene_clap + sigmoid(gate_logit)*"
                        "normalize(learned_candidate))"
                        if config.foundation_semantic_mixing_mode
                        == CONVEX_SEMANTIC_INTERPOLATION
                        else "normalize(role_pooled_scene_clap + "
                        "sigmoid(scale_logit)*normalize(learned_delta))"
                    )
                )
                if config.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES
                else "legacy learned semantic head"
            ),
            "answer_base_source": (
                (
                    (
                        "official full-question CLAP inference input; learned "
                        "semantic correction frozen at exact zero; no oracle text"
                        if args.freeze_semantic_adapter
                        else "official full-question CLAP inference input plus a "
                        "learned audio-question delta; no oracle text"
                    )
                    if config.foundation_semantic_mixing_mode
                    == QUESTION_RESIDUAL_SEMANTIC_MIXING
                    else "role-pooled acoustic scene CLAP; never question text"
                )
                if config.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES
                else None
            ),
            "contains_event_answer_or_oracle_inputs": False,
            "deployment_status": (
                "online inference supported by the same frozen AudioSep-CLAP "
                "feature recipe; canonical 10 s clips only"
                if config.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES
                else "legacy"
            ),
            "full_scene_alignment": (
                "canonical 10 s waveform only; cropped training rejected"
                if config.foundation_feature_mode == AUDIOSEP_CLAP_FOUNDATION_FEATURES
                else None
            ),
            "semantic_adapter": semantic_adapter_identity,
        },
        "temporal_role_representation": {
            "mode": config.temporal_role_mode,
            "legacy_default": LEGACY_TEMPORAL_ROLE_MODE,
            "paper_v5_required_mode": OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
            "role_head_output_channels": 3,
            "anchor_answer_overlap_representable": (
                config.temporal_role_mode == OVERLAP_AWARE_TEMPORAL_ROLE_MODE
            ),
            "evidence_union": (
                "1 - (1 - sigmoid(anchor_logit)) * " "(1 - sigmoid(answer_logit))"
                if config.temporal_role_mode == OVERLAP_AWARE_TEMPORAL_ROLE_MODE
                else "1 - softmax(role_logits)[none]"
            ),
            "no_evidence_head": "separate clip-level sigmoid logit",
        },
        "loss_weights": {
            name: getattr(criterion.weights, name)
            for name in criterion.weights.__dataclass_fields__
        },
        "loss_config": {
            "fft_sizes": list(criterion.fft_sizes),
            "sufficiency_margin": criterion.sufficiency_margin,
            "residual_margin": criterion.residual_margin,
            "no_evidence_class_balance": no_evidence_balance,
            "counterfactual_transition_margin": (
                criterion.counterfactual_transition_margin
            ),
        },
        "counterfactual_evidence_equivariance": (
            {
                "train": counterfactual_plan.provenance(),
                "validation": (
                    val_counterfactual_plan.provenance()
                    if val_counterfactual_plan is not None
                    else None
                ),
                "weights": counterfactual_weights,
                "paired_objective_enabled": counterfactual_enabled,
                "forced_schedule_matched_control": (
                    counterfactual_modes["forced_schedule_matched_control"]
                ),
                "sampler": {
                    "type": "deterministic_disjoint_group_batch_sampler",
                    "batch_size": args.batch_size,
                    "one_model_forward_per_batch": True,
                    "physical_separator_forwards_per_batch": (
                        1 if args.backend == "audiosep" else None
                    ),
                    "effective_separator_evaluations_per_record": (
                        2
                        if config.semantic_separation_mode == DUAL_ROLE_SEMANTIC_MODE
                        else 1 if args.backend == "audiosep" else None
                    ),
                    "records_duplicated_within_epoch ↓": 0,
                },
                "temporal_target": (
                    "anchor_or_answer_union; overlap-preserving and compatible "
                    "with independent anchor/answer sigmoid roles"
                ),
                "claim_boundary": (
                    "paired intervention equivariance, not causal proof"
                ),
                "metric_directions": COUNTERFACTUAL_METRIC_DIRECTIONS,
            }
            if counterfactual_plan is not None
            else {
                "enabled": False,
                "backward_compatible_default": "all paired weights are zero",
            }
        ),
        "semantic_targets": (
            str(args.semantic_targets.resolve()) if args.semantic_targets else None
        ),
        "val_semantic_targets": (
            str(args.val_semantic_targets.resolve())
            if args.val_semantic_targets
            else None
        ),
        "semantic_target_caches": {
            "train": semantic_cache_identity,
            "validation": val_semantic_cache_identity,
        },
        "role_semantic_target_caches": {
            "train": role_semantic_cache_identity,
            "validation": val_role_semantic_cache_identity,
        },
        "semantic_weight": args.semantic_weight,
        "role_semantic_weight": args.role_semantic_weight,
        "same_semantic_weight": args.same_semantic_weight,
        "same_semantic_calibration_metrics": {
            "same_semantic_brier ↓": "lower is better",
            "same_semantic_accuracy ↑": "higher is better",
            "p_same_on_same ↑": "descriptive; should approach one",
            "p_same_on_different ↓": "descriptive; should approach zero",
        },
        "role_relative_weight": args.role_relative_weight,
        "weakest_role_weight": args.weakest_role_weight,
        # Keep the historical warm-start identity untouched.  A separate
        # receipt is populated only for a genuinely from-scratch composer.
        "initialization": initialization_identity,
        "from_scratch_composer_initialization": from_scratch_composer_receipt,
        "separator_aware_refiner": refiner_identity,
        "composer_warmup": {
            "freeze_steps_requested": args.freeze_base_composer_steps,
            "updates_completed_while_frozen": 0,
        },
        "selection": {
            "enabled": val_manifest is not None,
            "split": "validation" if val_manifest is not None else None,
            "metric": args.selection_metric if val_manifest is not None else None,
            "direction": direction if val_manifest is not None else None,
            "display": (
                f"{args.selection_metric} {direction_arrow}"
                if val_manifest is not None
                else None
            ),
            "min_delta": args.selection_min_delta,
            "early_stopping_patience": args.early_stopping_patience,
        },
        "diagnostic_checkpoint_tracking": (
            {
                "purpose": (
                    "retain independent validation extrema for predeclared "
                    "Pareto analysis without retraining"
                ),
                "warning": (
                    "Independent extrema are diagnostics, not a joint selection "
                    "rule; the no-evidence minimum alone can select an all-silent "
                    "model"
                ),
                "metrics": {
                    metric: {
                        "direction": metric_direction,
                        "display": (
                            f"{metric} "
                            f"{'↓' if metric_direction == 'minimize' else '↑'}"
                        ),
                    }
                    for metric, metric_direction in (DIAGNOSTIC_BEST_DIRECTIONS.items())
                },
            }
            if val_manifest is not None
            else None
        ),
        "epoch_checkpoint_retention": {
            "enabled": args.save_every_epoch,
            "policy": "save_every_epoch" if args.save_every_epoch else None,
            "immutable": bool(args.save_every_epoch),
            "directory": (
                str(output_dir / "epoch_checkpoints") if args.save_every_epoch else None
            ),
            "filename_pattern": (
                "epoch_{epoch:04d}_checkpoint.pt" if args.save_every_epoch else None
            ),
            "purpose": (
                "retain non-extreme epochs for a frozen post-hoc joint/Pareto rule"
                if args.save_every_epoch
                else None
            ),
        },
        "validation_protocol": (
            {
                "gradient_enabled": False,
                "model_mode": "eval",
                "shuffle": False,
                "random_crop": False,
                "batch_size": (
                    args.batch_size if val_counterfactual_plan is not None else 1
                ),
                "aggregation": (
                    "each record is evaluated once; standard terms are weighted "
                    "by record/answerability counts and paired terms by complete "
                    "group counts"
                ),
            }
            if val_manifest is not None
            else None
        ),
        "runtime": runtime_identity(device),
        "precision": precision_runtime_metadata(args.precision, scaler),
        "reproducibility": reproducibility,
        "git": git_identity(PROJECT_ROOT),
        "source_code": loaded_project_source_identity(PROJECT_ROOT),
        "audiosep": audiosep_identity,
        "command": [sys.executable, *sys.argv],
    }

    # Destructive overwrite is intentionally delayed until manifests, caches,
    # model/checkpoint compatibility, and provenance have all passed preflight.
    prepare_output_directory(
        output_dir,
        overwrite=args.overwrite,
        was_nonempty=output_was_nonempty,
    )
    epoch_checkpoint_dir = output_dir / "epoch_checkpoints"
    if args.save_every_epoch:
        # The parent directory has just passed the destructive-overwrite
        # preflight. Fail if anything appears here afterward.
        epoch_checkpoint_dir.mkdir(exist_ok=False)
    history = []
    retained_epoch_checkpoints: list[Dict[str, Any]] = []
    global_step = 0
    stop_reason = "epochs_completed"
    best_value: float | None = None
    best_epoch: int | None = None
    best_global_step: int | None = None
    best_validation_metrics: Dict[str, float] | None = None
    best_precision_metadata: Dict[str, Any] | None = None
    epochs_without_improvement = 0
    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    diagnostic_bests: Dict[str, Dict[str, Any]] = {}
    diagnostic_checkpoint_paths = {
        metric: output_dir / f"best_{metric}_checkpoint.pt"
        for metric in DIAGNOSTIC_BEST_DIRECTIONS
    }
    composer_frozen_updates = 0
    synchronize_cuda(device)
    startup_seconds = time.monotonic() - run_started_monotonic
    training_seconds = 0.0
    validation_seconds = 0.0
    for epoch in range(1, args.epochs + 1):
        if train_counterfactual_sampler is not None:
            train_counterfactual_sampler.set_epoch(epoch - 1)
        model.train()
        running: Dict[str, float] = {}
        running_denominators: Dict[str, int] = {}
        batch_count = 0
        reached_max_steps = False
        synchronize_cuda(device)
        training_phase_started = time.monotonic()
        for raw_batch in loader:
            composer_is_trainable = global_step >= args.freeze_base_composer_steps
            set_base_composer_trainable(
                model,
                composer_is_trainable,
                freeze_semantic_adapter=args.freeze_semantic_adapter,
            )
            if not composer_is_trainable:
                composer_frozen_updates += 1
            batch = move_tensors(raw_batch, device)
            add_foundation_features(batch, foundation_feature_cache, device)
            add_semantic_targets(batch, semantic_targets, device)
            add_role_semantic_targets(
                batch,
                role_semantic_targets,
                device,
                config.condition_dim,
            )
            optimizer.zero_grad(set_to_none=True)
            output = forward_training_batch_with_precision(
                model, batch, device, args.precision
            )
            total, components = criterion(output, batch)
            backward_and_optimizer_step(
                total,
                optimizer,
                parameters,
                precision=args.precision,
                scaler=scaler,
                max_grad_norm=5.0,
            )
            global_step += 1
            batch_count += 1
            _, paired_metrics, paired_counts = counterfactual_objectives(
                output,
                batch,
                transition_margin=criterion.counterfactual_transition_margin,
            )
            measured = {
                **components,
                **qces_metrics(output, batch),
                **paired_metrics,
            }
            current_batch_size = len(batch["sample_ids"])
            answerable_count = int((batch["no_evidence"] < 0.5).sum().item())
            no_evidence_count = current_batch_size - answerable_count
            role_valid = batch.get("role_semantic_valid")
            same_target = batch.get("same_semantic_target")
            same_semantic_count = (
                int(((role_valid > 0.5) & (same_target > 0.5)).sum().item())
                if isinstance(role_valid, torch.Tensor)
                and isinstance(same_target, torch.Tensor)
                else 0
            )
            different_semantic_count = (
                int(((role_valid > 0.5) & (same_target <= 0.5)).sum().item())
                if isinstance(role_valid, torch.Tensor)
                and isinstance(same_target, torch.Tensor)
                else 0
            )
            for name, value in measured.items():
                if name == "total":
                    weight = 1
                elif name in SURFACE_LOSS_NAMES or name.startswith("surface_"):
                    weight = paired_counts["surface"]
                elif name in FAMILY_LOSS_NAMES or name.startswith("family_"):
                    weight = paired_counts["family"]
                elif name in QUESTION_LOSS_NAMES or name.startswith("question_"):
                    weight = paired_counts["question"]
                elif name in SAME_SEMANTIC_NORMALIZED_METRICS:
                    weight = same_semantic_count
                elif name in DIFFERENT_SEMANTIC_NORMALIZED_METRICS:
                    weight = different_semantic_count
                elif name in ANSWERABLE_NORMALIZED_METRICS:
                    weight = answerable_count
                elif name in NO_EVIDENCE_NORMALIZED_METRICS:
                    weight = no_evidence_count
                else:
                    weight = current_batch_size
                if weight == 0:
                    continue
                running[name] = running.get(name, 0.0) + (
                    float(value.detach().cpu()) * weight
                )
                running_denominators[name] = running_denominators.get(name, 0) + weight
            if args.max_steps and global_step >= args.max_steps:
                reached_max_steps = True
                break
        synchronize_cuda(device)
        training_seconds += time.monotonic() - training_phase_started
        base_extra["precision"] = precision_runtime_metadata(args.precision, scaler)
        epoch_metrics = {
            name: value / running_denominators[name] for name, value in running.items()
        }
        epoch_metrics.update(epoch=epoch, global_step=global_step)
        base_extra["composer_warmup"][
            "updates_completed_while_frozen"
        ] = composer_frozen_updates
        validation_metrics = None
        improved = False
        if val_loader is not None:
            synchronize_cuda(device)
            validation_phase_started = time.monotonic()
            validation_metrics = evaluate_epoch(
                model,
                criterion,
                val_loader,
                device,
                val_semantic_targets,
                val_role_semantic_targets,
                val_foundation_feature_cache,
                args.precision,
            )
            synchronize_cuda(device)
            validation_seconds += time.monotonic() - validation_phase_started
            if args.selection_metric not in validation_metrics:
                raise RuntimeError(
                    "selection metric was not produced by validation: "
                    f"{args.selection_metric}"
                )
            candidate = validation_metrics[args.selection_metric]
            improved = selection_improved(
                candidate,
                best_value,
                direction,
                args.selection_min_delta,
            )
            if improved:
                best_value = candidate
                best_epoch = epoch
                best_global_step = global_step
                best_validation_metrics = dict(validation_metrics)
                best_precision_metadata = precision_runtime_metadata(
                    args.precision, scaler
                )
                epochs_without_improvement = 0
                provisional_extra = checkpoint_extra(
                    base_extra,
                    run_epochs_completed=epoch,
                    run_global_step=global_step,
                    checkpoint_epoch=epoch,
                    checkpoint_global_step=global_step,
                    checkpoint_role="best_validation",
                    stop_reason="training_in_progress",
                    best_epoch=best_epoch,
                    best_global_step=best_global_step,
                    best_value=best_value,
                )
                atomic_torch_save(
                    make_checkpoint_payload(
                        model, args.backend, config, provisional_extra
                    ),
                    best_checkpoint_path,
                )
            else:
                epochs_without_improvement += 1
            diagnostic_updates: Dict[str, bool] = {}
            for metric, metric_direction in DIAGNOSTIC_BEST_DIRECTIONS.items():
                if metric not in validation_metrics:
                    continue
                diagnostic_candidate = validation_metrics[metric]
                incumbent = diagnostic_bests.get(metric, {}).get("value")
                diagnostic_improved = selection_improved(
                    diagnostic_candidate,
                    incumbent,
                    metric_direction,
                    0.0,
                )
                diagnostic_updates[metric] = diagnostic_improved
                if not diagnostic_improved:
                    continue
                diagnostic = {
                    "metric": metric,
                    "direction": metric_direction,
                    "display": (
                        f"{metric} " f"{'↓' if metric_direction == 'minimize' else '↑'}"
                    ),
                    "value": diagnostic_candidate,
                    "epoch": epoch,
                    "global_step": global_step,
                    "checkpoint": str(diagnostic_checkpoint_paths[metric]),
                    "precision": precision_runtime_metadata(args.precision, scaler),
                    "validation_metrics_at_checkpoint": dict(validation_metrics),
                }
                diagnostic_bests[metric] = diagnostic
                diagnostic_extra = checkpoint_extra(
                    base_extra,
                    run_epochs_completed=epoch,
                    run_global_step=global_step,
                    checkpoint_epoch=epoch,
                    checkpoint_global_step=global_step,
                    checkpoint_role=f"diagnostic_best_{metric}",
                    stop_reason="training_in_progress",
                    best_epoch=best_epoch,
                    best_global_step=best_global_step,
                    best_value=best_value,
                )
                diagnostic_extra["diagnostic_selection"] = diagnostic
                atomic_torch_save(
                    make_checkpoint_payload(
                        model, args.backend, config, diagnostic_extra
                    ),
                    diagnostic_checkpoint_paths[metric],
                )
            epoch_metrics["validation"] = validation_metrics
            epoch_metrics["selection"] = {
                "metric": args.selection_metric,
                "direction": direction,
                "display": f"{args.selection_metric} {direction_arrow}",
                "value": candidate,
                "improved": improved,
                "best_value": best_value,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
            }
            epoch_metrics["diagnostic_checkpoint_updates"] = diagnostic_updates
        should_early_stop = bool(
            val_loader is not None
            and args.early_stopping_patience > 0
            and not improved
            and epochs_without_improvement >= args.early_stopping_patience
        )
        if args.save_every_epoch:
            retained_path = retained_epoch_checkpoint_path(output_dir, epoch)
            epoch_metrics["retained_epoch_checkpoint"] = str(retained_path)
            if reached_max_steps:
                retained_stop_reason = "max_steps"
            elif should_early_stop:
                retained_stop_reason = "early_stopping"
            elif epoch == args.epochs:
                retained_stop_reason = "epochs_completed"
            else:
                retained_stop_reason = "training_in_progress"
            retained_extra = retained_epoch_checkpoint_extra(
                base_extra,
                epoch=epoch,
                global_step=global_step,
                stop_reason=retained_stop_reason,
                best_epoch=best_epoch,
                best_global_step=best_global_step,
                best_value=best_value,
                metrics_at_checkpoint=epoch_metrics,
                checkpoint_path=retained_path,
            )
            atomic_torch_save(
                make_checkpoint_payload(model, args.backend, config, retained_extra),
                retained_path,
                overwrite=False,
            )
            retained_epoch_checkpoints.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "checkpoint": file_identity(retained_path),
                    "stop_reason_at_checkpoint": retained_stop_reason,
                    "validation_metrics_at_checkpoint": (
                        dict(validation_metrics)
                        if validation_metrics is not None
                        else None
                    ),
                }
            )
        history.append(epoch_metrics)
        if (
            epoch == 1
            or epoch % args.log_every_epochs == 0
            or reached_max_steps
            or should_early_stop
        ):
            print(json.dumps(epoch_metrics, sort_keys=True), flush=True)
        if reached_max_steps:
            stop_reason = "max_steps"
            break
        if should_early_stop:
            stop_reason = "early_stopping"
            break

    synchronize_cuda(device)
    elapsed_before_checkpoint_seconds = time.monotonic() - run_started_monotonic
    base_extra["precision"] = precision_runtime_metadata(args.precision, scaler)
    base_extra["run_resources"] = {
        "scope": (
            "startup_cache_validation_model_load_training_and_validation; "
            "excludes final checkpoint serialization"
        ),
        "wall_seconds_down": elapsed_before_checkpoint_seconds,
        "startup_seconds_down": startup_seconds,
        "training_seconds_down": training_seconds,
        "validation_seconds_down": validation_seconds,
        "optimizer_steps": global_step,
        "optimizer_steps_per_second_up": (
            global_step / elapsed_before_checkpoint_seconds
            if elapsed_before_checkpoint_seconds > 0
            else None
        ),
        "optimizer_steps_per_training_second_up": (
            global_step / training_seconds if training_seconds > 0 else None
        ),
        "cuda_peak_allocated_bytes_down": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "cuda_peak_reserved_bytes_down": (
            torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
        ),
    }
    base_extra["composer_warmup"][
        "updates_completed_while_frozen"
    ] = composer_frozen_updates
    base_extra["foundation_features"]["semantic_adapter_post_training"] = (
        semantic_adapter_provenance(
            model, freeze_semantic_adapter=args.freeze_semantic_adapter
        )
    )
    final_epoch = int(history[-1]["epoch"])
    final_extra = checkpoint_extra(
        base_extra,
        run_epochs_completed=len(history),
        run_global_step=global_step,
        checkpoint_epoch=final_epoch,
        checkpoint_global_step=global_step,
        checkpoint_role="final" if val_loader is None else "last",
        stop_reason=stop_reason,
        best_epoch=best_epoch,
        best_global_step=best_global_step,
        best_value=best_value,
    )
    checkpoint_path = output_dir / "checkpoint.pt"
    last_checkpoint_path = None
    if val_loader is None:
        atomic_torch_save(
            make_checkpoint_payload(model, args.backend, config, final_extra),
            checkpoint_path,
        )
    else:
        if best_epoch is None or not best_checkpoint_path.is_file():
            raise RuntimeError("validation ran but no finite best checkpoint was saved")
        assert best_global_step is not None
        assert best_precision_metadata is not None
        last_checkpoint_path = output_dir / "last_checkpoint.pt"
        atomic_torch_save(
            make_checkpoint_payload(model, args.backend, config, final_extra),
            last_checkpoint_path,
        )
        best_payload = torch.load(
            best_checkpoint_path, map_location="cpu", weights_only=True
        )
        best_payload["extra"] = checkpoint_extra(
            {**base_extra, "precision": best_precision_metadata},
            run_epochs_completed=len(history),
            run_global_step=global_step,
            checkpoint_epoch=best_epoch,
            checkpoint_global_step=best_global_step,
            checkpoint_role="best_validation",
            stop_reason=stop_reason,
            best_epoch=best_epoch,
            best_global_step=best_global_step,
            best_value=best_value,
        )
        atomic_torch_save(best_payload, best_checkpoint_path)
        atomic_torch_save(best_payload, checkpoint_path)
        for metric, diagnostic in diagnostic_bests.items():
            diagnostic_path = diagnostic_checkpoint_paths[metric]
            diagnostic_payload = torch.load(
                diagnostic_path, map_location="cpu", weights_only=True
            )
            diagnostic_extra = checkpoint_extra(
                {**base_extra, "precision": diagnostic["precision"]},
                run_epochs_completed=len(history),
                run_global_step=global_step,
                checkpoint_epoch=int(diagnostic["epoch"]),
                checkpoint_global_step=int(diagnostic["global_step"]),
                checkpoint_role=f"diagnostic_best_{metric}",
                stop_reason=stop_reason,
                best_epoch=best_epoch,
                best_global_step=best_global_step,
                best_value=best_value,
            )
            diagnostic_extra["diagnostic_selection"] = diagnostic
            diagnostic_payload["extra"] = diagnostic_extra
            atomic_torch_save(diagnostic_payload, diagnostic_path)
    summary = {
        **final_extra,
        "checkpoint_epoch": best_epoch if val_loader is not None else final_epoch,
        "checkpoint_global_step": (
            best_global_step if val_loader is not None else global_step
        ),
        "checkpoint_role": ("best_validation" if val_loader is not None else "final"),
        "backend": args.backend,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "checkpoint_semantics": (
            "final training epoch"
            if val_loader is None
            else "best held-out validation checkpoint"
        ),
        "best_checkpoint": (
            str(best_checkpoint_path) if val_loader is not None else None
        ),
        "last_checkpoint": (
            str(last_checkpoint_path) if last_checkpoint_path is not None else None
        ),
        "best_validation_metrics": best_validation_metrics,
        "selected_checkpoint_precision": (
            best_precision_metadata
            if val_loader is not None
            else base_extra["precision"]
        ),
        "training_final_precision": base_extra["precision"],
        "diagnostic_best_checkpoints": diagnostic_bests,
        "retained_epoch_checkpoints": retained_epoch_checkpoints,
        "diagnostic_checkpoint_warning": (
            "Each file is one independent validation extreme, not an approved "
            "primary checkpoint. In particular, minimizing no-evidence retention "
            "alone can reward silence. Apply a frozen joint/Pareto rule."
            if diagnostic_bests
            else None
        ),
        "final_metrics": history[-1] if history else {},
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if val_loader is None:
        print(f"saved final checkpoint {checkpoint_path}")
    else:
        assert best_value is not None
        print(
            f"saved best validation checkpoint {checkpoint_path}: "
            f"{args.selection_metric} {direction_arrow}={best_value:.6f} "
            f"at epoch {best_epoch}"
        )


if __name__ == "__main__":
    main()
