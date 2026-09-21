#!/usr/bin/env python3
"""Factorize semantic and temporal errors in a frozen-AudioSep QCES model.

This evaluator never updates the prompt composer or AudioSep.  On answerable
examples it renders the complete learned/oracle semantic x learned/oracle
temporal 2x2, plus an ungated learned-semantic output and two temporal-only
mixture controls.  Oracle semantic conditions come from a cache cryptographically
bound to the evaluated manifest.  Oracle temporal windows and all role-wise
waveform scores are annotation-assisted diagnostics, not deployable results.

Temporal threshold selection is deliberately impossible outside a validation
manifest.  Test/train runs can optionally import a threshold from a previously
written validation report, but cannot optimize a threshold on their own labels.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mixi_understanding.data.qces_v4_schema import QCESV4Record
from mixi_understanding.qces.composer import (
    ROLE_ANCHOR,
    ROLE_ANSWER,
    PromptComposition,
)
from mixi_understanding.qces.data import QCESManifestDataset, collate_qces
from mixi_understanding.qces.metrics import (
    scale_dependent_sdr,
    scale_invariant_sdr,
)
from mixi_understanding.qces.model import load_qces_checkpoint
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.evaluate_audiosep_baselines import oracle_prompt


FORMAT = "qces_audiosep_factorization_v1"
VALIDATION_THRESHOLD_STATUS = "fitted_on_validation_only"
ENERGY_CONDITION = "learned_semantic__validation_tuned_raw_rms_temporal"
COMPOSER_BINARY_CONDITION = "learned_semantic__calibrated_binary_temporal"
CLIP_NO_EVIDENCE_CONDITION = "learned_semantic__clip_no_evidence_gate"

CONDITION_PROTOCOL: Dict[str, Dict[str, Any]] = {
    "learned_semantic__learned_soft_temporal": {
        "semantic_source": "audio+question prompt composer",
        "temporal_source": (
            "audio+question composer soft union probability; when enabled by "
            "the checkpoint, separator-aware refinement from the same raw "
            "learned-semantic AudioSep stem"
        ),
        "uses_oracle_annotation": False,
        "deployable": True,
        "description": "The actual QCES inference path.",
    },
    "learned_semantic__oracle_union_window": {
        "semantic_source": "audio+question prompt composer",
        "temporal_source": "ground-truth anchor/answer union window",
        "uses_oracle_annotation": True,
        "deployable": False,
        "description": "Oracle-windowed diagnostic isolating learned semantics.",
    },
    "oracle_semantic__learned_soft_temporal": {
        "semantic_source": "label-derived cached AudioSep CLAP condition",
        "temporal_source": (
            "the same learned temporal probability used by the deployable "
            "learned-semantic path, including its optional separator-aware refiner"
        ),
        "uses_oracle_annotation": True,
        "deployable": False,
        "description": "Oracle-semantic diagnostic isolating temporal prediction.",
    },
    "oracle_semantic__oracle_union_window": {
        "semantic_source": "label-derived cached AudioSep CLAP condition",
        "temporal_source": "ground-truth anchor/answer union window",
        "uses_oracle_annotation": True,
        "deployable": False,
        "description": "Joint semantic+temporal oracle upper-bound diagnostic.",
    },
    "learned_semantic__ungated": {
        "semantic_source": "audio+question prompt composer",
        "temporal_source": "none",
        "uses_oracle_annotation": False,
        "deployable": False,
        "description": "Diagnostic raw AudioSep output before the learned gate.",
    },
    "mixture__learned_soft_temporal": {
        "semantic_source": "none; unseparated mixture",
        "temporal_source": (
            "the same learned temporal probability used by the deployable path, "
            "including its optional separator-aware refiner"
        ),
        "uses_oracle_annotation": False,
        "deployable": False,
        "description": "Temporal-only crop control without AudioSep.",
    },
    "mixture__oracle_union_window": {
        "semantic_source": "none; unseparated mixture",
        "temporal_source": "ground-truth anchor/answer union window",
        "uses_oracle_annotation": True,
        "deployable": False,
        "description": (
            "Oracle temporal-only benchmark-validity control without AudioSep."
        ),
    },
    ENERGY_CONDITION: {
        "semantic_source": "audio+question prompt composer",
        "temporal_source": (
            "per-clip max-normalized frame RMS of ungated learned-semantic "
            "AudioSep output; one global validation-tuned threshold"
        ),
        "uses_oracle_annotation": False,
        "uses_validation_labels_for_global_threshold": True,
        "deployable": True,
        "description": (
            "Separator-aware temporal control. Validation scores use a threshold "
            "selected on validation; test scores must lock that hashed threshold."
        ),
    },
    COMPOSER_BINARY_CONDITION: {
        "semantic_source": "audio+question prompt composer",
        "temporal_source": (
            "composer union probabilities binarized by one global "
            "validation-tuned threshold; fixed linear frame-boundary interpolation"
        ),
        "uses_oracle_annotation": False,
        "uses_validation_labels_for_global_threshold": True,
        "deployable": True,
        "description": (
            "Calibration control separating probability-scale failure from ranking/"
            "localization failure. No dilation or sample-specific tuning is used."
        ),
    },
    CLIP_NO_EVIDENCE_CONDITION: {
        "semantic_source": "audio+question prompt composer",
        "temporal_source": (
            "no frame gate; raw learned-semantic AudioSep stem is silenced only "
            "when clip-level no-evidence probability exceeds one global "
            "validation-tuned threshold"
        ),
        "uses_oracle_annotation": False,
        "uses_validation_labels_for_global_threshold": True,
        "deployable": True,
        "description": (
            "Deployable fallback while the temporal refiner is under development. "
            "The global threshold maximizes validation balanced accuracy."
        ),
    },
}

POSTHOC_CALIBRATED_CONDITIONS = frozenset(
    {
        ENERGY_CONDITION,
        COMPOSER_BINARY_CONDITION,
        CLIP_NO_EVIDENCE_CONDITION,
    }
)
BASE_RENDERED_CONDITIONS = (
    frozenset(CONDITION_PROTOCOL) - POSTHOC_CALIBRATED_CONDITIONS
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle-semantic-cache", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold-grid-size", type=int, default=1001)
    parser.add_argument(
        "--validation-calibration-report",
        type=Path,
        help=(
            "validation factorization report supplying a locked temporal "
            "threshold for a train/test run; never used to refit"
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def source_tree_identity(root: Path) -> Dict[str, Any]:
    """Hash every executable AudioSep source/config file, including dirty edits."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    paths = sorted(
        path
        for path in resolved.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(resolved).parts
        and ".cache" not in path.relative_to(resolved).parts
        and "__pycache__" not in path.relative_to(resolved).parts
        and path.suffix.lower() in {".py", ".yaml", ".yml"}
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(resolved).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "hashed_file_count": len(paths),
        "included_suffixes": [".py", ".yaml", ".yml"],
    }


def dataset_audio_identity(
    dataset_root: Path, records: Sequence[QCESV4Record]
) -> Dict[str, Any]:
    """Aggregate-hash every waveform actually consumed by this evaluator."""

    fields = (
        "mixture_path",
        "evidence_stem_path",
        "residual_stem_path",
        "anchor_stem_path",
        "answer_stem_path",
    )
    relative_paths = sorted(
        {
            str(getattr(record, field))
            for record in records
            for field in fields
        }
    )
    digest = hashlib.sha256()
    total_bytes = 0
    resolved_root = dataset_root.resolve()
    for relative in relative_paths:
        path = (resolved_root / relative).resolve()
        if resolved_root not in path.parents or not path.is_file():
            raise ValueError(f"invalid dataset audio input: {relative}")
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                total_bytes += len(chunk)
                digest.update(chunk)
    return {
        "dataset_root": str(resolved_root),
        "sha256": digest.hexdigest(),
        "unique_waveform_count": len(relative_paths),
        "total_bytes": total_bytes,
        "included_manifest_fields": list(fields),
    }


def git_identity(root: Path) -> Dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root.resolve()), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        status = run("status", "--porcelain")
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status)}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def runtime_identity(device: torch.device) -> Dict[str, Any]:
    package_versions = {}
    for distribution in ("PyYAML", "soundfile", "torchlibrosa"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "package_versions": package_versions,
    }


def load_oracle_semantic_cache(
    cache_path: Path,
    manifest_path: Path,
    sample_ids: Sequence[str],
    condition_dim: int,
    audiosep_checkpoint: Path,
    expected_prompts: Mapping[str, str],
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load an oracle cache only when it is bound to this exact split/input."""

    resolved = cache_path.resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "qces_audiosep_semantic_targets_v1"
        or not isinstance(payload.get("targets"), dict)
        or not isinstance(payload.get("prompts"), dict)
    ):
        raise ValueError("invalid oracle semantic cache format")
    if payload.get("schema_version") != "qces_v4":
        raise ValueError("factorization requires a QCES v4 oracle semantic cache")
    if payload.get("prompt_source") != "evidence_role_event_labels_or_absent_label":
        raise ValueError("oracle semantic cache has an unsupported prompt source")
    manifest_hash = sha256_file(manifest_path.resolve())
    if payload.get("manifest_sha256") != manifest_hash:
        raise ValueError("oracle semantic cache is not bound to this manifest")
    cached_checkpoint = payload.get("audiosep_checkpoint")
    if not isinstance(cached_checkpoint, str) or (
        Path(cached_checkpoint).resolve() != audiosep_checkpoint.resolve()
    ):
        raise ValueError(
            "oracle semantic cache was encoded with a different AudioSep checkpoint"
        )
    expected_ids = set(sample_ids)
    actual_ids = set(payload["targets"])
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_ids)[:10]
        raise ValueError(
            "oracle semantic cache must be split-exact: "
            f"missing={missing}, extra={extra}"
        )
    actual_prompt_ids = set(payload["prompts"])
    if actual_prompt_ids != expected_ids or dict(payload["prompts"]) != dict(
        expected_prompts
    ):
        raise ValueError(
            "oracle semantic cache prompts do not exactly match role-event labels"
        )
    targets: Dict[str, torch.Tensor] = {}
    for sample_id in sample_ids:
        value = payload["targets"][sample_id]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"oracle condition is not a tensor: {sample_id}")
        flattened = value.detach().float().reshape(-1)
        if flattened.numel() != condition_dim or not torch.isfinite(flattened).all():
            raise ValueError(
                f"invalid oracle condition for {sample_id}: {tuple(value.shape)}"
            )
        targets[sample_id] = flattened
    identity = {
        **file_identity(resolved),
        "format": payload["format"],
        "schema_version": payload.get("schema_version"),
        "target_scope": payload.get("target_scope"),
        "prompt_source": payload.get("prompt_source"),
        "manifest_sha256": manifest_hash,
        "target_count": len(targets),
    }
    return targets, identity


def binary_ranking_metrics(
    scores: np.ndarray, targets: np.ndarray
) -> Dict[str, Optional[float]]:
    """Compute tie-aware ROC AUC and average precision without sklearn."""

    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(targets).astype(bool).reshape(-1)
    if score.shape != target.shape or score.size == 0:
        raise ValueError("scores and targets must be non-empty aligned arrays")
    if not np.isfinite(score).all():
        raise ValueError("ranking scores must be finite")
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if positives == 0:
        return {"auroc": None, "auprc": None}

    order = np.argsort(-score, kind="mergesort")
    sorted_score = score[order]
    sorted_target = target[order]
    group_end = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    cumulative_tp = np.cumsum(sorted_target, dtype=np.float64)[group_end]
    cumulative_fp = np.cumsum(~sorted_target, dtype=np.float64)[group_end]
    recall = cumulative_tp / positives
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    previous_recall = np.r_[0.0, recall[:-1]]
    average_precision = float(np.sum((recall - previous_recall) * precision))

    auroc: Optional[float]
    if negatives == 0:
        auroc = None
    else:
        tpr = np.r_[0.0, recall]
        fpr = np.r_[0.0, cumulative_fp / negatives]
        auroc = float(np.trapz(tpr, fpr))
    return {"auroc": auroc, "auprc": average_precision}


def threshold_metrics(
    scores_by_example: Sequence[np.ndarray],
    targets_by_example: Sequence[np.ndarray],
    threshold: float,
) -> Dict[str, float]:
    if len(scores_by_example) != len(targets_by_example) or not scores_by_example:
        raise ValueError("threshold metrics require aligned non-empty examples")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("temporal threshold must be in [0, 1]")
    ious = []
    f1s = []
    total_tp = total_fp = total_fn = 0
    for scores, targets in zip(scores_by_example, targets_by_example):
        score = np.asarray(scores, dtype=np.float64).reshape(-1)
        target = np.asarray(targets).astype(bool).reshape(-1)
        if score.shape != target.shape or score.size == 0:
            raise ValueError("each temporal score/target pair must align")
        predicted = score >= threshold
        tp = int(np.logical_and(predicted, target).sum())
        fp = int(np.logical_and(predicted, ~target).sum())
        fn = int(np.logical_and(~predicted, target).sum())
        union = tp + fp + fn
        ious.append(1.0 if union == 0 else tp / union)
        denominator = 2 * tp + fp + fn
        f1s.append(1.0 if denominator == 0 else 2 * tp / denominator)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_union = total_tp + total_fp + total_fn
    micro_denominator = 2 * total_tp + total_fp + total_fn
    return {
        "answerable_macro_iou_↑": float(np.mean(ious)),
        "answerable_macro_f1_↑": float(np.mean(f1s)),
        "answerable_micro_iou_↑": (
            1.0 if micro_union == 0 else total_tp / micro_union
        ),
        "answerable_micro_f1_↑": (
            1.0 if micro_denominator == 0 else 2 * total_tp / micro_denominator
        ),
    }


def fit_validation_threshold(
    scores_by_example: Sequence[np.ndarray],
    targets_by_example: Sequence[np.ndarray],
    split: str,
    grid_size: int = 1001,
) -> Dict[str, Any]:
    """Fit a union threshold; the split guard prevents train/test leakage."""

    if split != "val":
        raise ValueError("temporal threshold fitting is allowed only on validation")
    if grid_size < 2:
        raise ValueError("threshold grid size must be at least two")
    best_threshold = 0.5
    best_metrics: Optional[Dict[str, float]] = None
    best_key: Optional[tuple[float, ...]] = None
    for raw_threshold in np.linspace(0.0, 1.0, grid_size):
        threshold = float(raw_threshold)
        metrics = threshold_metrics(
            scores_by_example, targets_by_example, threshold
        )
        # Macro IoU is the preregistered selection objective.  The remaining
        # entries make ties deterministic without peeking at downstream QA.
        key = (
            metrics["answerable_macro_iou_↑"],
            metrics["answerable_macro_f1_↑"],
            metrics["answerable_micro_iou_↑"],
            metrics["answerable_micro_f1_↑"],
            -abs(threshold - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    assert best_metrics is not None
    return {
        "status": VALIDATION_THRESHOLD_STATUS,
        "selection_split": "val",
        "selection_objective": "answerable_macro_iou_↑",
        "grid": {
            "minimum": 0.0,
            "maximum": 1.0,
            "size": grid_size,
            "inclusive": True,
        },
        "threshold": best_threshold,
        "metrics": best_metrics,
    }


def no_evidence_threshold_metrics(
    scores: np.ndarray, targets: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Score clip-level no-evidence decisions with answerable safety explicit."""

    score = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(targets).astype(bool).reshape(-1)
    if score.shape != target.shape or score.size == 0:
        raise ValueError("no-evidence scores and targets must be non-empty and aligned")
    if not np.isfinite(score).all() or not 0.0 <= threshold <= 1.0:
        raise ValueError("no-evidence scores/threshold must be finite probabilities")
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("no-evidence calibration requires both target classes")
    predicted = score >= threshold
    true_positive = int(np.logical_and(predicted, target).sum())
    false_positive = int(np.logical_and(predicted, ~target).sum())
    recall = true_positive / positives
    false_silence = false_positive / negatives
    return {
        "balanced_accuracy_↑": 0.5 * (recall + (1.0 - false_silence)),
        "no_evidence_recall_↑": recall,
        "answerable_false_silence_rate_↓": false_silence,
        "overall_accuracy_↑": float(np.mean(predicted == target)),
    }


def fit_validation_no_evidence_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    split: str,
    grid_size: int = 1001,
) -> Dict[str, Any]:
    """Fit one clip gate on validation; never optimize it on train/test."""

    if split != "val":
        raise ValueError("no-evidence threshold fitting is allowed only on validation")
    if grid_size < 2:
        raise ValueError("threshold grid size must be at least two")
    best_threshold = 0.5
    best_metrics: Optional[Dict[str, float]] = None
    best_key: Optional[tuple[float, ...]] = None
    for raw_threshold in np.linspace(0.0, 1.0, grid_size):
        threshold = float(raw_threshold)
        metrics = no_evidence_threshold_metrics(scores, targets, threshold)
        # Balanced accuracy is primary. Ties first protect answerable clips from
        # false silencing, then prefer no-evidence recall, then a central cutoff.
        key = (
            metrics["balanced_accuracy_↑"],
            -metrics["answerable_false_silence_rate_↓"],
            metrics["no_evidence_recall_↑"],
            metrics["overall_accuracy_↑"],
            -abs(threshold - 0.5),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_metrics = metrics
    assert best_metrics is not None
    return {
        "status": VALIDATION_THRESHOLD_STATUS,
        "selection_split": "val",
        "selection_objective": "balanced_accuracy_↑",
        "tie_break_policy": (
            "answerable_false_silence_rate_↓, no_evidence_recall_↑, "
            "overall_accuracy_↑, proximity to 0.5"
        ),
        "grid": {
            "minimum": 0.0,
            "maximum": 1.0,
            "size": grid_size,
            "inclusive": True,
        },
        "threshold": best_threshold,
        "metrics": best_metrics,
    }


def load_locked_validation_threshold(
    path: Path,
    fit_key: str = "validation_only_threshold_fit",
    calibration_section: str = "temporal_calibration",
    expected_checkpoint_sha256: Optional[str] = None,
) -> tuple[float, Dict[str, Any]]:
    resolved = path.resolve()
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if report.get("format") != FORMAT or report.get("split") != "val":
        raise ValueError("locked threshold source must be a validation report")
    if expected_checkpoint_sha256 is not None:
        reported_checkpoint_sha256 = (
            report.get("provenance", {})
            .get("inputs", {})
            .get("learned_qces_checkpoint", {})
            .get("sha256")
        )
        if reported_checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError(
                "validation threshold was fitted for a different QCES checkpoint"
            )
    fit = report.get(calibration_section, {}).get(fit_key)
    if not isinstance(fit, dict) or fit.get("status") != VALIDATION_THRESHOLD_STATUS:
        raise ValueError(
            f"validation report has no validation-fitted threshold: {fit_key}"
        )
    threshold = fit.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ValueError("validation report has an invalid temporal threshold")
    return float(threshold), file_identity(resolved)


def normalized_frame_rms(waveform: torch.Tensor, frames: int) -> torch.Tensor:
    """Return per-clip max-normalized RMS on a fixed temporal frame grid."""

    if waveform.ndim != 2 or frames <= 0:
        raise ValueError("waveform must be [B, N] and frames must be positive")
    mean_square = F.adaptive_avg_pool1d(waveform.square()[:, None], frames).squeeze(1)
    rms = mean_square.clamp_min(0.0).sqrt()
    maximum = rms.amax(dim=-1, keepdim=True)
    return torch.where(maximum > 0.0, rms / maximum.clamp_min(1e-12), rms)


def apply_frame_gate(
    waveform: torch.Tensor, frame_gate: torch.Tensor
) -> torch.Tensor:
    """Linearly lift a frame decision to waveform samples and apply it."""

    if waveform.ndim != 2 or frame_gate.ndim != 2:
        raise ValueError("waveform and frame gate must both be rank two")
    if waveform.size(0) != frame_gate.size(0):
        raise ValueError("waveform and frame gate batch sizes differ")
    sample_gate = F.interpolate(
        frame_gate[:, None].to(waveform.dtype),
        size=waveform.size(-1),
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    return waveform * sample_gate


def apply_clip_no_evidence_gate(
    waveform: torch.Tensor,
    no_evidence_probability: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    if waveform.ndim != 2 or no_evidence_probability.ndim != 1:
        raise ValueError("waveform must be [B, N] and clip probability must be [B]")
    if waveform.size(0) != no_evidence_probability.numel():
        raise ValueError("waveform and clip probability batch sizes differ")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("no-evidence threshold must be in [0, 1]")
    keep = (no_evidence_probability < threshold).to(waveform.dtype)
    return waveform * keep[:, None]


def _separator_waveform(
    separator: torch.nn.Module,
    mixture: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    result = separator({"mixture": mixture[:, None], "condition": condition})
    waveform = result.get("waveform") if isinstance(result, dict) else result
    if not isinstance(waveform, torch.Tensor):
        raise TypeError("AudioSep separator did not return a waveform tensor")
    if waveform.ndim == 3 and waveform.size(1) == 1:
        waveform = waveform[:, 0]
    if waveform.shape != mixture.shape:
        raise ValueError(
            "separator returned "
            f"{tuple(waveform.shape)}, expected {tuple(mixture.shape)}"
        )
    return waveform


def render_factorized_conditions(
    separator: torch.nn.Module,
    mixture: torch.Tensor,
    learned_condition: torch.Tensor,
    oracle_condition: torch.Tensor,
    learned_frame_gate: torch.Tensor,
    oracle_sample_gate: torch.Tensor,
    *,
    learned_raw: Optional[torch.Tensor] = None,
    oracle_raw: Optional[torch.Tensor] = None,
) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Render the 2x2 and controls with at most two AudioSep calls.

    Precomputed raw stems let the caller run the learned stem through the
    optional separator-aware temporal refiner before rendering, without making
    a second learned-semantic AudioSep call.
    """

    if mixture.ndim != 2 or oracle_sample_gate.shape != mixture.shape:
        raise ValueError("mixture and oracle gate must have aligned [B, N] shapes")
    if learned_frame_gate.ndim != 2 or learned_frame_gate.size(0) != mixture.size(0):
        raise ValueError("learned frame gate must have shape [B, T]")
    if learned_condition.shape != oracle_condition.shape:
        raise ValueError("learned and oracle semantic conditions must align")
    if (learned_raw is None) != (oracle_raw is None):
        raise ValueError("precomputed learned/oracle raw stems must be supplied together")
    if learned_raw is None:
        learned_raw = _separator_waveform(separator, mixture, learned_condition)
        oracle_raw = _separator_waveform(separator, mixture, oracle_condition)
    assert oracle_raw is not None
    if learned_raw.shape != mixture.shape or oracle_raw.shape != mixture.shape:
        raise ValueError("precomputed raw stems must match the mixture shape")
    learned_sample_gate = F.interpolate(
        learned_frame_gate[:, None],
        size=mixture.size(-1),
        mode="linear",
        align_corners=False,
    ).squeeze(1)
    evidence = {
        "learned_semantic__learned_soft_temporal": (
            learned_raw * learned_sample_gate
        ),
        "learned_semantic__oracle_union_window": learned_raw * oracle_sample_gate,
        "oracle_semantic__learned_soft_temporal": (
            oracle_raw * learned_sample_gate
        ),
        "oracle_semantic__oracle_union_window": oracle_raw * oracle_sample_gate,
        "learned_semantic__ungated": learned_raw,
        "mixture__learned_soft_temporal": mixture * learned_sample_gate,
        "mixture__oracle_union_window": mixture * oracle_sample_gate,
    }
    if set(evidence) != BASE_RENDERED_CONDITIONS:
        raise AssertionError("factorization condition protocol drifted")
    return {name: (stem, mixture - stem) for name, stem in evidence.items()}


def apply_separator_aware_refiner(
    adapter: torch.nn.Module,
    mixture: torch.Tensor,
    learned_raw: torch.Tensor,
    base_composition: PromptComposition,
) -> PromptComposition:
    """Return the actual learned temporal composition used at inference."""

    refiner = getattr(adapter, "separator_aware_refiner", None)
    if refiner is None:
        return base_composition
    return refiner(mixture, learned_raw, base_composition)


def _safe_float(value: torch.Tensor) -> float:
    result = float(value.detach().cpu())
    if not math.isfinite(result):
        raise ValueError("evaluation produced a non-finite metric")
    return result


def score_condition_item(
    evidence: torch.Tensor,
    residual: torch.Tensor,
    mixture: torch.Tensor,
    evidence_target: torch.Tensor,
    residual_target: torch.Tensor,
    anchor_stem: torch.Tensor,
    answer_stem: torch.Tensor,
    anchor_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    no_evidence: bool,
) -> Dict[str, Optional[float]]:
    """Score one condition; role scores deliberately use oracle windows."""

    mixture_consistency = (mixture - evidence - residual).abs().mean()
    retained = evidence.abs().sum() / mixture.abs().sum().clamp_min(1e-8)
    metrics: Dict[str, Optional[float]] = {
        "evidence_l1_answerable_↓": None,
        "residual_l1_answerable_↓": None,
        "evidence_si_sdr_db_↑": None,
        "evidence_si_sdri_db_↑": None,
        "evidence_sd_sdr_db_↑": None,
        "evidence_sd_sdri_db_↑": None,
        "oracle_windowed_anchor_si_sdr_db_↑": None,
        "oracle_windowed_answer_si_sdr_db_↑": None,
        "oracle_windowed_weakest_role_si_sdr_db_↑": None,
        "no_evidence_retained_ratio_↓": (
            _safe_float(retained) if no_evidence else None
        ),
        "mixture_consistency_l1_sanity_↓": _safe_float(mixture_consistency),
    }
    if no_evidence:
        return metrics
    evidence_si_sdr = _safe_float(
        scale_invariant_sdr(evidence[None], evidence_target[None])[0]
    )
    mixture_si_sdr = _safe_float(
        scale_invariant_sdr(mixture[None], evidence_target[None])[0]
    )
    evidence_sd_sdr = _safe_float(
        scale_dependent_sdr(evidence[None], evidence_target[None])[0]
    )
    mixture_sd_sdr = _safe_float(
        scale_dependent_sdr(mixture[None], evidence_target[None])[0]
    )
    anchor_prediction = evidence * anchor_mask
    answer_prediction = evidence * answer_mask
    anchor_si_sdr = _safe_float(
        scale_invariant_sdr(anchor_prediction[None], anchor_stem[None])[0]
    )
    answer_si_sdr = _safe_float(
        scale_invariant_sdr(answer_prediction[None], answer_stem[None])[0]
    )
    metrics.update(
        {
            "evidence_l1_answerable_↓": _safe_float(
                (evidence - evidence_target).abs().mean()
            ),
            "residual_l1_answerable_↓": _safe_float(
                (residual - residual_target).abs().mean()
            ),
            "evidence_si_sdr_db_↑": evidence_si_sdr,
            "evidence_si_sdri_db_↑": evidence_si_sdr - mixture_si_sdr,
            "evidence_sd_sdr_db_↑": evidence_sd_sdr,
            "evidence_sd_sdri_db_↑": evidence_sd_sdr - mixture_sd_sdr,
            "oracle_windowed_anchor_si_sdr_db_↑": anchor_si_sdr,
            "oracle_windowed_answer_si_sdr_db_↑": answer_si_sdr,
            "oracle_windowed_weakest_role_si_sdr_db_↑": min(
                anchor_si_sdr, answer_si_sdr
            ),
        }
    )
    return metrics


def _mean(values: Iterable[float]) -> Optional[float]:
    materialized = list(values)
    return float(np.mean(materialized)) if materialized else None


def linear_percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    """Return a deterministic linear percentile without NumPy-version drift.

    The interpolation index is ``(n - 1) * quantile``.  Keeping the definition
    here makes acoustic-tail model-selection receipts reproducible across
    environments whose ``numpy.percentile`` defaults may differ.
    """

    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be within [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_condition_items(
    condition_items: Sequence[Mapping[str, Optional[float]]],
) -> Dict[str, Optional[float]]:
    def values(name: str) -> list[float]:
        return [
            float(item[name])
            for item in condition_items
            if item.get(name) is not None
        ]

    evidence_si_sdr = values("evidence_si_sdr_db_↑")
    evidence_sd_sdr = values("evidence_sd_sdr_db_↑")
    summary: Dict[str, Optional[float]] = {
        "evidence_l1_answerable_mean_↓": _mean(
            values("evidence_l1_answerable_↓")
        ),
        "residual_l1_answerable_mean_↓": _mean(
            values("residual_l1_answerable_↓")
        ),
        "evidence_si_sdr_answerable_mean_db_↑": _mean(evidence_si_sdr),
        "evidence_si_sdr_answerable_median_db_↑": (
            float(median(evidence_si_sdr)) if evidence_si_sdr else None
        ),
        "evidence_si_sdr_answerable_minimum_db_↑": (
            min(evidence_si_sdr) if evidence_si_sdr else None
        ),
        "evidence_si_sdr_below_minus20_db_rate_↓": (
            float(np.mean(np.asarray(evidence_si_sdr) < -20.0))
            if evidence_si_sdr
            else None
        ),
        "evidence_si_sdr_floor_minus80_db_rate_↓": (
            float(np.mean(np.asarray(evidence_si_sdr) <= -80.0))
            if evidence_si_sdr
            else None
        ),
        "evidence_si_sdri_answerable_mean_db_↑": _mean(
            values("evidence_si_sdri_db_↑")
        ),
        "evidence_sd_sdr_answerable_mean_db_↑": _mean(
            evidence_sd_sdr
        ),
        "evidence_sd_sdr_answerable_p10_db_↑": linear_percentile(
            evidence_sd_sdr, 0.1
        ),
        "evidence_sd_sdr_answerable_minimum_db_↑": (
            min(evidence_sd_sdr) if evidence_sd_sdr else None
        ),
        "evidence_sd_sdri_answerable_mean_db_↑": _mean(
            values("evidence_sd_sdri_db_↑")
        ),
        "oracle_windowed_anchor_si_sdr_mean_db_↑": _mean(
            values("oracle_windowed_anchor_si_sdr_db_↑")
        ),
        "oracle_windowed_answer_si_sdr_mean_db_↑": _mean(
            values("oracle_windowed_answer_si_sdr_db_↑")
        ),
        "oracle_windowed_weakest_role_si_sdr_mean_db_↑": _mean(
            values("oracle_windowed_weakest_role_si_sdr_db_↑")
        ),
        "no_evidence_retained_ratio_mean_↓": _mean(
            values("no_evidence_retained_ratio_↓")
        ),
        "no_evidence_retained_ratio_maximum_↓": (
            max(values("no_evidence_retained_ratio_↓"))
            if values("no_evidence_retained_ratio_↓")
            else None
        ),
        "mixture_consistency_l1_sanity_maximum_↓": (
            max(values("mixture_consistency_l1_sanity_↓"))
            if values("mixture_consistency_l1_sanity_↓")
            else None
        ),
    }
    return summary


def oracle_gate_validity_metrics(
    mixture: torch.Tensor,
    target: torch.Tensor,
    residual_target: torch.Tensor,
    oracle_gate: torch.Tensor,
) -> Dict[str, float]:
    """Measure contamination retained by annotation-only temporal cropping."""

    gated_mixture = mixture * oracle_gate
    contamination = residual_target * oracle_gate
    target_outside = target * (1.0 - oracle_gate)
    target_energy = target.abs().sum().clamp_min(1e-8)
    gated_si_sdr = _safe_float(
        scale_invariant_sdr(gated_mixture[None], target[None])[0]
    )
    mixture_si_sdr = _safe_float(
        scale_invariant_sdr(mixture[None], target[None])[0]
    )
    return {
        "oracle_window_residual_contamination_l1_↓": _safe_float(
            contamination.abs().mean()
        ),
        "oracle_window_contamination_to_target_l1_ratio_↓": _safe_float(
            contamination.abs().sum() / target_energy
        ),
        "target_outside_oracle_window_l1_↓": _safe_float(
            target_outside.abs().mean()
        ),
        "target_vs_oracle_gated_mixture_si_sdr_db_↑": gated_si_sdr,
        "target_vs_oracle_gated_mixture_si_sdri_db_↑": (
            gated_si_sdr - mixture_si_sdr
        ),
    }


def summarize_validity(
    items: Sequence[Mapping[str, float]],
) -> Dict[str, Optional[float]]:
    if not items:
        return {}
    names = tuple(items[0])
    return {
        name.replace("_↓", "_mean_↓").replace("_↑", "_mean_↑"): _mean(
            float(item[name]) for item in items
        )
        for name in names
    }


def _ranking_report(
    scores: Sequence[np.ndarray], targets: Sequence[np.ndarray], prefix: str
) -> Dict[str, Optional[float]]:
    metrics = binary_ranking_metrics(
        np.concatenate(scores), np.concatenate(targets)
    )
    return {
        f"{prefix}_auroc_↑": metrics["auroc"],
        f"{prefix}_auprc_↑": metrics["auprc"],
    }


def _semantic_cosine_report(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    answerable = [
        float(item["learned_oracle_semantic_cosine_↑"])
        for item in items
        if not item["no_evidence"]
    ]
    negatives = [
        float(item["learned_oracle_semantic_cosine_↑"])
        for item in items
        if item["no_evidence"]
    ]
    return {
        "answerable_learned_oracle_semantic_cosine_mean_↑": _mean(answerable),
        "no_evidence_learned_oracle_semantic_cosine_mean_↑": _mean(negatives),
    }


def _factorial_diagnosis(
    summaries: Mapping[str, Mapping[str, Optional[float]]]
) -> Dict[str, float]:
    metric = "evidence_si_sdr_answerable_mean_db_↑"

    def score(condition: str) -> float:
        value = summaries[condition][metric]
        if value is None:
            raise ValueError("factorial diagnosis needs answerable SI-SDR")
        return float(value)

    learned_learned = score("learned_semantic__learned_soft_temporal")
    learned_oracle = score("learned_semantic__oracle_union_window")
    oracle_learned = score("oracle_semantic__learned_soft_temporal")
    oracle_oracle = score("oracle_semantic__oracle_union_window")
    ungated = score("learned_semantic__ungated")
    semantic_effect = 0.5 * (
        (oracle_learned - learned_learned) + (oracle_oracle - learned_oracle)
    )
    temporal_effect = 0.5 * (
        (learned_oracle - learned_learned) + (oracle_oracle - oracle_learned)
    )
    interaction = oracle_oracle - oracle_learned - learned_oracle + learned_learned
    return {
        "semantic_oracle_headroom_mean_effect_db_↓": semantic_effect,
        "temporal_oracle_headroom_mean_effect_db_↓": temporal_effect,
        "joint_oracle_headroom_db_↓": oracle_oracle - learned_learned,
        "factorial_interaction_absolute_db_↓": abs(interaction),
        "learned_temporal_gain_over_ungated_db_↑": learned_learned - ungated,
    }


def _prepare_output(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("batch-size must be positive")
    if args.threshold_grid_size < 2:
        raise SystemExit("threshold-grid-size must be at least two")
    device = resolve_device(args.device)

    manifest = args.manifest.resolve()
    learned_checkpoint = args.checkpoint.resolve()
    audiosep_root = args.audiosep_root.resolve()
    audiosep_config = args.audiosep_config.resolve()
    audiosep_checkpoint = args.audiosep_checkpoint.resolve()
    for path in (
        manifest,
        learned_checkpoint,
        args.oracle_semantic_cache.resolve(),
        audiosep_config,
        audiosep_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    learned_checkpoint_identity = file_identity(learned_checkpoint)

    dataset = QCESManifestDataset(manifest, crop_samples=None, random_crop=False)
    if not dataset.records or not all(
        isinstance(record, QCESV4Record) for record in dataset.records
    ):
        raise SystemExit("factorization evaluation requires a QCES v4 manifest")
    records = [record for record in dataset.records if isinstance(record, QCESV4Record)]
    splits = {record.split for record in records}
    if len(splits) != 1:
        raise SystemExit(f"manifest mixes evaluation splits: {sorted(splits)}")
    split = next(iter(splits))
    if split == "val" and args.validation_calibration_report is not None:
        raise SystemExit("validation fits its own threshold; do not import one")

    checkpoint_payload = torch.load(
        learned_checkpoint, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint_payload, dict) or checkpoint_payload.get(
        "backend"
    ) != "audiosep":
        raise SystemExit("checkpoint must be a learned QCES AudioSep checkpoint")
    model = load_qces_checkpoint(
        checkpoint_payload,
        map_location=device,
        audiosep_repository_root=str(audiosep_root),
        audiosep_config_path=str(audiosep_config),
        audiosep_checkpoint_path=str(audiosep_checkpoint),
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    oracle_conditions, semantic_cache_identity = load_oracle_semantic_cache(
        args.oracle_semantic_cache,
        manifest,
        [record.sample_id for record in records],
        model.config.condition_dim,
        audiosep_checkpoint,
        {record.sample_id: oracle_prompt(record) for record in records},
    )
    tokenizer = StableHashTokenizer(
        model.config.vocab_size, model.config.max_question_tokens
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_qces, tokenizer=tokenizer),
    )
    records_by_id = {record.sample_id: record for record in records}

    items: list[Dict[str, Any]] = []
    per_condition: Dict[str, list[Mapping[str, Optional[float]]]] = {
        name: [] for name in CONDITION_PROTOCOL
    }
    validity_items: list[Mapping[str, float]] = []
    union_scores_all: list[np.ndarray] = []
    union_targets_all: list[np.ndarray] = []
    union_scores_answerable: list[np.ndarray] = []
    union_targets_answerable: list[np.ndarray] = []
    anchor_scores_answerable: list[np.ndarray] = []
    anchor_targets_answerable: list[np.ndarray] = []
    answer_scores_answerable: list[np.ndarray] = []
    answer_targets_answerable: list[np.ndarray] = []
    separator_energy_scores_all: list[np.ndarray] = []
    separator_energy_scores_answerable: list[np.ndarray] = []
    learned_raw_waveforms: list[torch.Tensor] = []
    no_evidence_scores: list[float] = []
    no_evidence_targets: list[bool] = []

    separator = model.separator.ss_model
    with torch.inference_mode():
        for raw_batch in loader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            composition = model.composer(
                batch["mixture"], batch["question_ids"], batch["question_mask"]
            )
            learned_raw = _separator_waveform(
                separator, batch["mixture"], composition.semantic_condition
            )
            composition = apply_separator_aware_refiner(
                model.separator,
                batch["mixture"],
                learned_raw,
                composition,
            )
            oracle_condition = torch.stack(
                [oracle_conditions[sample_id] for sample_id in batch["sample_ids"]]
            ).to(device)
            oracle_raw = _separator_waveform(
                separator, batch["mixture"], oracle_condition
            )
            oracle_sample_gate = (
                batch["anchor_mask"] + batch["answer_mask"]
            ).clamp_max(1.0)
            rendered = render_factorized_conditions(
                separator,
                batch["mixture"],
                composition.semantic_condition,
                oracle_condition,
                composition.evidence_probability,
                oracle_sample_gate,
                learned_raw=learned_raw,
                oracle_raw=oracle_raw,
            )
            frames = composition.evidence_probability.size(-1)
            anchor_frame_target = F.adaptive_max_pool1d(
                batch["anchor_mask"][:, None], frames
            ).squeeze(1) > 0.5
            answer_frame_target = F.adaptive_max_pool1d(
                batch["answer_mask"][:, None], frames
            ).squeeze(1) > 0.5
            union_frame_target = anchor_frame_target | answer_frame_target
            role_probabilities = composition.role_probabilities
            semantic_cosine = F.cosine_similarity(
                composition.semantic_condition, oracle_condition, dim=-1
            )
            no_evidence_probability = composition.no_evidence_logit.sigmoid()
            learned_raw = rendered["learned_semantic__ungated"][0]
            separator_energy = normalized_frame_rms(learned_raw, frames)

            for local_index, sample_id in enumerate(batch["sample_ids"]):
                record = records_by_id[sample_id]
                no_evidence = bool(record.no_evidence)
                union_score = (
                    composition.evidence_probability[local_index].cpu().numpy()
                )
                union_target = union_frame_target[local_index].cpu().numpy()
                union_scores_all.append(union_score)
                union_targets_all.append(union_target)
                energy_score = separator_energy[local_index].cpu().numpy()
                separator_energy_scores_all.append(energy_score)
                learned_raw_waveforms.append(learned_raw[local_index].cpu())
                no_evidence_scores.append(float(no_evidence_probability[local_index]))
                no_evidence_targets.append(no_evidence)
                if not no_evidence:
                    union_scores_answerable.append(union_score)
                    union_targets_answerable.append(union_target)
                    separator_energy_scores_answerable.append(energy_score)
                    anchor_scores_answerable.append(
                        role_probabilities[local_index, :, ROLE_ANCHOR].cpu().numpy()
                    )
                    anchor_targets_answerable.append(
                        anchor_frame_target[local_index].cpu().numpy()
                    )
                    answer_scores_answerable.append(
                        role_probabilities[local_index, :, ROLE_ANSWER].cpu().numpy()
                    )
                    answer_targets_answerable.append(
                        answer_frame_target[local_index].cpu().numpy()
                    )

                item: Dict[str, Any] = {
                    "id": sample_id,
                    "scene_id": record.scene_id,
                    "question_index": record.question_index,
                    "question_type": record.question_type,
                    "relation": record.relation,
                    "question": record.question,
                    "no_evidence": no_evidence,
                    "learned_oracle_semantic_cosine_↑": float(
                        semantic_cosine[local_index]
                    ),
                    "target_frame_temporal_probability_mean_↑": (
                        float(
                            composition.evidence_probability[local_index][
                                union_frame_target[local_index]
                            ].mean()
                        )
                        if bool(union_frame_target[local_index].any())
                        else None
                    ),
                    "background_frame_temporal_probability_mean_↓": float(
                        composition.evidence_probability[local_index][
                            ~union_frame_target[local_index]
                        ].mean()
                    ),
                    "correct_no_evidence_class_probability_↑": float(
                        no_evidence_probability[local_index]
                        if no_evidence
                        else 1.0 - no_evidence_probability[local_index]
                    ),
                    "raw_separator_target_frame_rms_mean_↑": (
                        float(
                            separator_energy[local_index][
                                union_frame_target[local_index]
                            ].mean()
                        )
                        if bool(union_frame_target[local_index].any())
                        else None
                    ),
                    "raw_separator_background_frame_rms_mean_↓": float(
                        separator_energy[local_index][
                            ~union_frame_target[local_index]
                        ].mean()
                    ),
                    "conditions": {},
                    "oracle_window_benchmark_validity": None,
                }
                for condition_name, (evidence, residual) in rendered.items():
                    condition_metrics = score_condition_item(
                        evidence[local_index],
                        residual[local_index],
                        batch["mixture"][local_index],
                        batch["evidence"][local_index],
                        batch["residual"][local_index],
                        batch["anchor_stem"][local_index],
                        batch["answer_stem"][local_index],
                        batch["anchor_mask"][local_index],
                        batch["answer_mask"][local_index],
                        no_evidence,
                    )
                    item["conditions"][condition_name] = condition_metrics
                    per_condition[condition_name].append(condition_metrics)
                if not no_evidence:
                    validity = oracle_gate_validity_metrics(
                        batch["mixture"][local_index],
                        batch["evidence"][local_index],
                        batch["residual"][local_index],
                        oracle_sample_gate[local_index],
                    )
                    item["oracle_window_benchmark_validity"] = validity
                    validity_items.append(validity)
                items.append(item)

    temporal_threshold_free: Dict[str, Optional[float]] = {}
    temporal_threshold_free.update(
        _ranking_report(union_scores_all, union_targets_all, "all_union_frame")
    )
    temporal_threshold_free.update(
        _ranking_report(
            union_scores_answerable,
            union_targets_answerable,
            "answerable_union_frame",
        )
    )
    temporal_threshold_free.update(
        _ranking_report(
            anchor_scores_answerable,
            anchor_targets_answerable,
            "answerable_anchor_role_frame",
        )
    )
    temporal_threshold_free.update(
        _ranking_report(
            answer_scores_answerable,
            answer_targets_answerable,
            "answerable_answer_role_frame",
        )
    )
    separator_energy_threshold_free: Dict[str, Optional[float]] = {}
    separator_energy_threshold_free.update(
        _ranking_report(
            separator_energy_scores_all,
            union_targets_all,
            "all_raw_separator_normalized_rms_frame",
        )
    )
    separator_energy_threshold_free.update(
        _ranking_report(
            separator_energy_scores_answerable,
            union_targets_answerable,
            "answerable_raw_separator_normalized_rms_frame",
        )
    )

    threshold_fit: Dict[str, Any]
    energy_threshold_fit: Dict[str, Any]
    no_evidence_threshold_fit: Dict[str, Any]
    locked_threshold_evaluation: Optional[Dict[str, Any]] = None
    locked_energy_threshold_evaluation: Optional[Dict[str, Any]] = None
    locked_no_evidence_threshold_evaluation: Optional[Dict[str, Any]] = None
    calibration_source_identity: Optional[Dict[str, Any]] = None
    composer_application_threshold: Optional[float] = None
    energy_application_threshold: Optional[float] = None
    no_evidence_application_threshold: Optional[float] = None
    if split == "val":
        threshold_fit = fit_validation_threshold(
            union_scores_answerable,
            union_targets_answerable,
            split=split,
            grid_size=args.threshold_grid_size,
        )
        energy_threshold_fit = fit_validation_threshold(
            separator_energy_scores_answerable,
            union_targets_answerable,
            split=split,
            grid_size=args.threshold_grid_size,
        )
        energy_threshold_fit["feature"] = (
            "per-clip max-normalized frame RMS of learned-semantic ungated AudioSep"
        )
        no_evidence_threshold_fit = fit_validation_no_evidence_threshold(
            np.asarray(no_evidence_scores),
            np.asarray(no_evidence_targets),
            split=split,
            grid_size=args.threshold_grid_size,
        )
        composer_application_threshold = float(threshold_fit["threshold"])
        energy_application_threshold = float(energy_threshold_fit["threshold"])
        no_evidence_application_threshold = float(
            no_evidence_threshold_fit["threshold"]
        )
    else:
        threshold_fit = {
            "status": "disabled_outside_validation",
            "selection_split": None,
            "reason": "No threshold was optimized on train/test labels.",
        }
        energy_threshold_fit = {
            "status": "disabled_outside_validation",
            "selection_split": None,
            "reason": (
                "No separator-energy threshold was optimized on train/test labels."
            ),
        }
        no_evidence_threshold_fit = {
            "status": "disabled_outside_validation",
            "selection_split": None,
            "reason": (
                "No clip no-evidence threshold was optimized on train/test labels."
            ),
        }
        if args.validation_calibration_report is not None:
            locked_threshold, calibration_source_identity = (
                load_locked_validation_threshold(
                    args.validation_calibration_report,
                    expected_checkpoint_sha256=learned_checkpoint_identity["sha256"],
                )
            )
            locked_threshold_evaluation = {
                "status": "locked_from_validation",
                "threshold": locked_threshold,
                "source": calibration_source_identity,
                "metrics": threshold_metrics(
                    union_scores_answerable,
                    union_targets_answerable,
                    locked_threshold,
                ),
            }
            composer_application_threshold = locked_threshold
            locked_energy_threshold, _ = load_locked_validation_threshold(
                args.validation_calibration_report,
                fit_key="separator_energy_validation_only_threshold_fit",
                expected_checkpoint_sha256=learned_checkpoint_identity["sha256"],
            )
            energy_application_threshold = locked_energy_threshold
            locked_energy_threshold_evaluation = {
                "status": "locked_from_validation",
                "threshold": locked_energy_threshold,
                "source": calibration_source_identity,
                "metrics": threshold_metrics(
                    separator_energy_scores_answerable,
                    union_targets_answerable,
                    locked_energy_threshold,
                ),
            }
            locked_no_evidence_threshold, _ = load_locked_validation_threshold(
                args.validation_calibration_report,
                fit_key="validation_only_threshold_fit",
                calibration_section="no_evidence_calibration",
                expected_checkpoint_sha256=learned_checkpoint_identity["sha256"],
            )
            no_evidence_application_threshold = locked_no_evidence_threshold
            locked_no_evidence_threshold_evaluation = {
                "status": "locked_from_validation",
                "threshold": locked_no_evidence_threshold,
                "source": calibration_source_identity,
                "metrics": no_evidence_threshold_metrics(
                    np.asarray(no_evidence_scores),
                    np.asarray(no_evidence_targets),
                    locked_no_evidence_threshold,
                ),
            }

    posthoc_controls = {
        COMPOSER_BINARY_CONDITION: (
            union_scores_all,
            composer_application_threshold,
        ),
        ENERGY_CONDITION: (
            separator_energy_scores_all,
            energy_application_threshold,
        ),
    }
    for condition_name, (frame_scores, application_threshold) in (
        posthoc_controls.items()
    ):
        if application_threshold is None:
            for item in items:
                item["conditions"][condition_name] = None
            continue
        for item, learned_raw, scores, example in zip(
            items,
            learned_raw_waveforms,
            frame_scores,
            dataset.examples,
        ):
            if item["id"] != example.sample_id:
                raise AssertionError("post-hoc gate evaluation order drifted")
            frame_gate = torch.from_numpy(
                (scores >= application_threshold).astype(np.float32)
            )[None]
            evidence = apply_frame_gate(learned_raw[None], frame_gate)[0]
            residual = example.mixture - evidence
            condition_metrics = score_condition_item(
                evidence,
                residual,
                example.mixture,
                example.evidence,
                example.residual,
                example.anchor_stem,
                example.answer_stem,
                example.anchor_mask,
                example.answer_mask,
                bool(item["no_evidence"]),
            )
            item["conditions"][condition_name] = condition_metrics
            per_condition[condition_name].append(condition_metrics)

    if no_evidence_application_threshold is None:
        for item in items:
            item["conditions"][CLIP_NO_EVIDENCE_CONDITION] = None
    else:
        probability_tensor = torch.tensor(no_evidence_scores, dtype=torch.float32)
        raw_tensor = torch.stack(learned_raw_waveforms)
        clip_gated = apply_clip_no_evidence_gate(
            raw_tensor,
            probability_tensor,
            no_evidence_application_threshold,
        )
        for item, evidence, example in zip(items, clip_gated, dataset.examples):
            if item["id"] != example.sample_id:
                raise AssertionError("clip no-evidence gate evaluation order drifted")
            residual = example.mixture - evidence
            condition_metrics = score_condition_item(
                evidence,
                residual,
                example.mixture,
                example.evidence,
                example.residual,
                example.anchor_stem,
                example.answer_stem,
                example.anchor_mask,
                example.answer_mask,
                bool(item["no_evidence"]),
            )
            item["conditions"][CLIP_NO_EVIDENCE_CONDITION] = condition_metrics
            per_condition[CLIP_NO_EVIDENCE_CONDITION].append(condition_metrics)

    summaries = {
        name: summarize_condition_items(per_condition[name])
        for name in CONDITION_PROTOCOL
    }

    no_evidence_ranking = binary_ranking_metrics(
        np.asarray(no_evidence_scores), np.asarray(no_evidence_targets)
    )
    predicted_no_evidence = np.asarray(no_evidence_scores) >= 0.5
    no_evidence_target_array = np.asarray(no_evidence_targets)
    composer_diagnostics = {
        **_semantic_cosine_report(items),
        "no_evidence_auroc_↑": no_evidence_ranking["auroc"],
        "no_evidence_auprc_↑": no_evidence_ranking["auprc"],
        "no_evidence_accuracy_at_0.5_↑": float(
            np.mean(predicted_no_evidence == no_evidence_target_array)
        ),
        "no_evidence_temporal_probability_mean_↓": _mean(
            float(item["background_frame_temporal_probability_mean_↓"])
            for item in items
            if item["no_evidence"]
        ),
        "answerable_target_frame_temporal_probability_mean_↑": _mean(
            float(item["target_frame_temporal_probability_mean_↑"])
            for item in items
            if not item["no_evidence"]
        ),
        "answerable_raw_separator_target_frame_rms_mean_↑": _mean(
            float(item["raw_separator_target_frame_rms_mean_↑"])
            for item in items
            if not item["no_evidence"]
        ),
        "no_evidence_raw_separator_background_frame_rms_mean_↓": _mean(
            float(item["raw_separator_background_frame_rms_mean_↓"])
            for item in items
            if item["no_evidence"]
        ),
    }

    qces_source_files = [
        Path(__file__),
        CODE_ROOT / "mixi_understanding/qces/model.py",
        CODE_ROOT / "mixi_understanding/qces/config.py",
        CODE_ROOT / "mixi_understanding/qces/composer.py",
        CODE_ROOT / "mixi_understanding/qces/separators.py",
        CODE_ROOT / "mixi_understanding/qces/data.py",
        CODE_ROOT / "mixi_understanding/qces/metrics.py",
        CODE_ROOT / "mixi_understanding/qces/tokenization.py",
        CODE_ROOT / "mixi_understanding/data/qces_v4_schema.py",
        CODE_ROOT / "mixi_understanding/scripts/evaluate_audiosep_baselines.py",
    ]
    report = {
        "format": FORMAT,
        "split": split,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "protocol": {
            "separator_training": "frozen; evaluation-only",
            "composer_training": "frozen; evaluation-only",
            "separator_aware_refiner_enabled": bool(
                model.config.separator_aware_refiner
            ),
            "separator_aware_refiner_mode": (
                model.config.separator_aware_refiner_mode
                if model.config.separator_aware_refiner
                else None
            ),
            "separator_aware_refiner_parameter_count": sum(
                parameter.numel()
                for parameter in (
                    model.separator.separator_aware_refiner.parameters()
                    if model.config.separator_aware_refiner
                    else ()
                )
            ),
            "learned_path_audiosep_calls_per_record": 1,
            "conditions": CONDITION_PROTOCOL,
            "oracle_warning": (
                "Every condition marked uses_oracle_annotation=true is diagnostic "
                "only and must not be presented as a deployable system result."
            ),
            "role_metric_warning": (
                "Anchor/answer SI-SDR masks predictions with ground-truth role "
                "windows. These are explicitly oracle-windowed diagnostics."
            ),
            "mixture_consistency_warning": (
                "Residual is defined arithmetically as X-E, so consistency is only "
                "a numerical sanity check, not evidence faithfulness."
            ),
            "threshold_leakage_policy": (
                "Threshold optimization is executable only for split=val. A "
                "train/test report may evaluate, but never refit, a threshold "
                "imported from a hashed validation report."
            ),
        },
        "dataset_statistics": {
            "record_count": len(items),
            "answerable_count": sum(not item["no_evidence"] for item in items),
            "no_evidence_count": sum(item["no_evidence"] for item in items),
            "scene_count": len({item["scene_id"] for item in items}),
            "composer_frame_count_per_record": int(union_scores_all[0].size),
        },
        "composer_diagnostics": composer_diagnostics,
        "no_evidence_calibration": {
            "validation_only_threshold_fit": no_evidence_threshold_fit,
            "locked_validation_threshold_evaluation": (
                locked_no_evidence_threshold_evaluation
            ),
            "fallback_condition": CLIP_NO_EVIDENCE_CONDITION,
        },
        "temporal_calibration": {
            "soft_threshold_free_metrics": temporal_threshold_free,
            "validation_only_threshold_fit": threshold_fit,
            "locked_validation_threshold_evaluation": locked_threshold_evaluation,
            "separator_energy_threshold_free_metrics": (
                separator_energy_threshold_free
            ),
            "separator_energy_validation_only_threshold_fit": (
                energy_threshold_fit
            ),
            "separator_energy_locked_validation_threshold_evaluation": (
                locked_energy_threshold_evaluation
            ),
        },
        "condition_summaries": summaries,
        "factorial_diagnosis": _factorial_diagnosis(summaries),
        "oracle_window_benchmark_validity": {
            "description": (
                "Residual energy inside oracle union windows measures temporal "
                "overlap contamination. The gated-mixture control tests whether "
                "cropping alone solves the benchmark without AudioSep."
            ),
            "metrics": summarize_validity(validity_items),
        },
        "provenance": {
            "inputs": {
                "learned_qces_checkpoint": learned_checkpoint_identity,
                "manifest": file_identity(manifest),
                "dataset_audio_inputs": dataset_audio_identity(
                    manifest.parent, records
                ),
                "oracle_semantic_cache": semantic_cache_identity,
                "audiosep_config": file_identity(audiosep_config),
                "audiosep_checkpoint": file_identity(audiosep_checkpoint),
                "validation_calibration_report": calibration_source_identity,
            },
            "software": {
                "qces_source_files": [
                    file_identity(path) for path in qces_source_files
                ],
                "audiosep_source_tree": source_tree_identity(audiosep_root),
                "audiosep_git": git_identity(audiosep_root),
                "workspace_git": git_identity(PROJECT_ROOT),
            },
            "checkpoint_metadata": checkpoint_payload.get("extra", {}),
            "runtime": runtime_identity(device),
            "command": [sys.executable, *sys.argv],
        },
        "items": items,
    }
    output_dir = args.output_dir.resolve()
    _prepare_output(output_dir, args.overwrite)
    report_path = output_dir / "factorization_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    console = {
        "split": split,
        "condition_summaries": summaries,
        "factorial_diagnosis": report["factorial_diagnosis"],
        "temporal_calibration": report["temporal_calibration"],
        "no_evidence_calibration": report["no_evidence_calibration"],
        "oracle_window_benchmark_validity": report[
            "oracle_window_benchmark_validity"
        ],
    }
    print(json.dumps(console, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
