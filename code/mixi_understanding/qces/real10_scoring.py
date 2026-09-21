"""Fail-closed non-QA scoring for QCES-Real-10 predictions.

QCES-Real-10 contains natural mixtures and human temporal annotations, but no
clean event waveform.  This module therefore scores temporal grounding,
abstention, compactness, arithmetic reconstruction, and separator cost only.
It deliberately has no SI-SDR/SD-SDR implementation or metric-selection hook.

The scorer is a post-hoc boundary between the strict
``qces_real10_scoring_v2`` view and the label-free renderer artifacts.  Before
examining model quality it independently validates all record bindings,
receipt identities, canonical WAV bytes, hashes, split coverage, and the
sample-exact ``E + R = X`` gate.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import struct
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf

from mixi_understanding.data.qces_real10_schema import (
    DURATION_SECONDS,
    INFERENCE_SCHEMA_VERSION,
    NUM_CHANNELS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    SCORING_SCHEMA_VERSION,
    QCESReal10ScoringRecord,
    canonical_inference_manifest_fingerprint,
    canonical_json_sha256,
    inference_record_fingerprint,
    parse_scoring_manifest,
    project_scoring_manifest,
)
from mixi_understanding.qces.real10_prediction import (
    METHOD_FREEZE_RECEIPT_FORMAT,
    PREDICTION_SCHEMA_VERSION,
    RENDER_RECEIPT_FORMAT,
    pcm_f32le_sha256,
    prediction_manifest_fingerprint,
    validate_prediction_record,
)


REPORT_FORMAT = "qces_real10_nonqa_score_report_v1"
RUN_SPEC_FORMAT = "qces_real10_render_run_spec_v1"
METHOD_IDENTITY_FORMAT = "qces_real10_method_identity_v1"
PREDICTION_MANIFEST_FILENAME = "prediction_manifest.jsonl"
RENDER_RECEIPT_FILENAME = "render_receipt.json"
STEM_AGGREGATE_FORMAT = "qces_real10_stem_aggregate_v1"
CANONICAL_WAV_FORMAT = "WAV"
CANONICAL_WAV_SUBTYPE = "FLOAT"
MAX_RECONSTRUCTION_ABS_ERROR = 1e-6
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 2026
ECE_BINS = 10

_SHA256_HEX = frozenset("0123456789abcdef")

# These are the only paper-facing scalar estimates emitted by this scorer.
# Keeping the arrows in the key makes table direction unambiguous downstream.
METRIC_DIRECTIONS: Mapping[str, str] = {
    "anchor_temporal_iou_answerable_↑": "↑",
    "answer_temporal_iou_answerable_↑": "↑",
    "union_temporal_iou_answerable_↑": "↑",
    "weakest_role_temporal_iou_answerable_↑": "↑",
    "onset_boundary_mae_seconds_answerable_↓": "↓",
    "no_evidence_auroc_↑": "↑",
    "no_evidence_auprc_↑": "↑",
    "no_evidence_brier_↓": "↓",
    "no_evidence_ece_10bin_↓": "↓",
    "no_evidence_retained_energy_ratio_↓": "↓",
    "answerable_retained_duration_ratio_↓": "↓",
    "answerable_retained_energy_ratio_↓": "↓",
    "reconstruction_max_abs_error_mean_↓": "↓",
    "reconstruction_max_abs_error_maximum_↓": "↓",
    "physical_separator_forwards_per_record_↓": "↓",
    "effective_separator_evaluations_per_record_↓": "↓",
}

_ROW_DISTRIBUTION_FIELDS: Mapping[str, str] = {
    "anchor_temporal_iou_answerable_↑": "anchor_iou",
    "answer_temporal_iou_answerable_↑": "answer_iou",
    "union_temporal_iou_answerable_↑": "union_iou",
    "weakest_role_temporal_iou_answerable_↑": "weakest_role_iou",
    "onset_boundary_mae_seconds_answerable_↓": "onset_boundary_mae_seconds",
    "no_evidence_retained_energy_ratio_↓": "no_evidence_energy_ratio",
    "answerable_retained_duration_ratio_↓": "answerable_duration_ratio",
    "answerable_retained_energy_ratio_↓": "answerable_energy_ratio",
    "reconstruction_max_abs_error_mean_↓": "reconstruction_max_abs_error",
    "physical_separator_forwards_per_record_↓": "physical_forwards",
    "effective_separator_evaluations_per_record_↓": "effective_evaluations",
}


class Real10ScoringError(RuntimeError):
    """Raised when a QCES-Real-10 integrity or scoring gate fails closed."""


@dataclass(frozen=True)
class EvaluatedRecord:
    """Validated per-question quantities used to form aggregate metrics."""

    sample_id: str
    scene_id: str
    creator_id: str
    split: str
    relation: str
    upstream_tacos_split: str
    no_evidence: bool
    no_evidence_probability: float
    anchor_iou: float | None
    answer_iou: float | None
    union_iou: float | None
    weakest_role_iou: float | None
    onset_boundary_mae_seconds: float | None
    no_evidence_energy_ratio: float | None
    answerable_duration_ratio: float | None
    answerable_energy_ratio: float | None
    reconstruction_max_abs_error: float
    physical_forwards: float
    effective_evaluations: float


@dataclass(frozen=True)
class ValidatedScoringInputs:
    """Fully content-bound inputs ready for metric aggregation."""

    scoring_manifest_path: Path
    dataset_root: Path
    prediction_root: Path
    scoring_manifest_identity: Mapping[str, Any]
    prediction_manifest_identity: Mapping[str, Any]
    render_receipt_identity: Mapping[str, Any]
    split: str
    records: tuple[QCESReal10ScoringRecord, ...]
    selected_records: tuple[QCESReal10ScoringRecord, ...]
    prediction_rows: tuple[Mapping[str, Any], ...]
    render_receipt: Mapping[str, Any]
    evaluated_records: tuple[EvaluatedRecord, ...]
    independently_observed_max_reconstruction_error: float


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and set(value).issubset(_SHA256_HEX)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise Real10ScoringError(f"artifact is not a regular file: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "soundfile"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _same_file_content(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return first.get("sha256") == second.get("sha256") and first.get(
        "size_bytes"
    ) == second.get("size_bytes")


def _exact_keys(value: Any, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Real10ScoringError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise Real10ScoringError(
            f"{context} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Real10ScoringError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Real10ScoringError(f"{context} must be finite")
    return result


def _positive_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Real10ScoringError(f"{context} must be a positive integer")
    return value


def _read_json_object(
    path: Path, context: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    before = _file_identity(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Real10ScoringError(f"cannot read {context}: {path}") from error
    if not isinstance(payload, Mapping):
        raise Real10ScoringError(f"{context} must contain a JSON object")
    after = _file_identity(path)
    if not _same_file_content(before, after):
        raise Real10ScoringError(f"{context} changed while being read")
    return payload, before


def _read_jsonl_objects(
    path: Path, context: str
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    before = _file_identity(path)
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise Real10ScoringError(
                        f"{context} line {line_number} is blank or lacks a final "
                        "newline"
                    )
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise Real10ScoringError(
                        f"{context} line {line_number} is not an object"
                    )
                rows.append(row)
    except json.JSONDecodeError as error:
        raise Real10ScoringError(f"{context} contains invalid JSON") from error
    if not rows:
        raise Real10ScoringError(f"{context} is empty")
    after = _file_identity(path)
    if not _same_file_content(before, after):
        raise Real10ScoringError(f"{context} changed while being read")
    return rows, before


def _resolve_input_file(root: Path, relative: str, context: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise Real10ScoringError(f"{context} is not a safe relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(
        part in {"", ".", ".."} or ":" in part for part in relative.split("/")
    ):
        raise Real10ScoringError(f"{context} is not a safe relative path")
    candidate = (root / pure).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise Real10ScoringError(f"{context} escapes its declared root") from error
    if not candidate.is_file() or candidate.is_symlink():
        raise Real10ScoringError(f"{context} is not a regular file: {candidate}")
    return candidate


def _pcm_bytes(samples: np.ndarray) -> bytes:
    array = np.asarray(samples)
    if array.shape != (NUM_SAMPLES,) or array.dtype != np.float32:
        raise Real10ScoringError(
            f"canonical waveform must be float32 shape ({NUM_SAMPLES},), got "
            f"{array.dtype} {array.shape}"
        )
    if not np.isfinite(array).all():
        raise Real10ScoringError("canonical waveform contains NaN or Inf")
    return np.ascontiguousarray(array, dtype=np.dtype("<f4")).tobytes(order="C")


def _canonical_float_wav_bytes(samples: np.ndarray) -> bytes:
    pcm = _pcm_bytes(samples)
    fmt_chunk = b"fmt " + struct.pack(
        "<IHHIIHH",
        16,
        3,
        NUM_CHANNELS,
        SAMPLE_RATE,
        SAMPLE_RATE * 4,
        4,
        32,
    )
    fact_chunk = b"fact" + struct.pack("<II", 4, NUM_SAMPLES)
    data_header = b"data" + struct.pack("<I", len(pcm))
    riff_size = 4 + len(fmt_chunk) + len(fact_chunk) + len(data_header) + len(pcm)
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + fmt_chunk
        + fact_chunk
        + data_header
        + pcm
    )


def _decode_float_wav(path: Path, *, deterministic_stem: bool) -> np.ndarray:
    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as error:
        raise Real10ScoringError(f"cannot inspect WAV {path}: {error}") from error
    if (
        info.format != CANONICAL_WAV_FORMAT
        or info.subtype != CANONICAL_WAV_SUBTYPE
        or info.samplerate != SAMPLE_RATE
        or info.channels != NUM_CHANNELS
        or info.frames != NUM_SAMPLES
    ):
        raise Real10ScoringError(
            "audio must be exact mono IEEE-float 32 kHz / 320000 samples: " f"{path}"
        )
    try:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as error:
        raise Real10ScoringError(f"cannot decode WAV {path}: {error}") from error
    array = np.asarray(samples)
    if sample_rate != SAMPLE_RATE or array.shape != (NUM_SAMPLES,):
        raise Real10ScoringError(f"decoded audio shape changed: {path}")
    array = np.ascontiguousarray(array, dtype=np.float32)
    _pcm_bytes(array)
    if deterministic_stem and path.read_bytes() != _canonical_float_wav_bytes(array):
        raise Real10ScoringError(
            f"prediction stem is not the deterministic canonical FLOAT WAV: {path}"
        )
    return array


def _validate_declared_file_identity(
    identity: Mapping[str, Any], actual: Mapping[str, Any], context: str
) -> None:
    if not _is_sha256(identity.get("sha256")):
        raise Real10ScoringError(f"{context} has an invalid SHA256")
    if (
        isinstance(identity.get("size_bytes"), bool)
        or not isinstance(identity.get("size_bytes"), int)
        or identity["size_bytes"] <= 0
    ):
        raise Real10ScoringError(f"{context} has an invalid size")
    if not _same_file_content(identity, actual):
        raise Real10ScoringError(f"{context} file identity mismatch")


def _union_intervals(
    intervals: Iterable[Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    ordered = sorted(
        ((float(interval[0]), float(interval[1])) for interval in intervals),
        key=lambda item: (item[0], item[1]),
    )
    merged: list[list[float]] = []
    for onset, offset in ordered:
        if not (math.isfinite(onset) and math.isfinite(offset)):
            raise Real10ScoringError("temporal interval is not finite")
        if not 0.0 <= onset < offset <= DURATION_SECONDS:
            raise Real10ScoringError("temporal interval lies outside [0, 10]")
        if not merged or onset > merged[-1][1]:
            merged.append([onset, offset])
        else:
            merged[-1][1] = max(merged[-1][1], offset)
    return tuple((item[0], item[1]) for item in merged)


def _interval_length(intervals: Sequence[Sequence[float]]) -> float:
    return float(sum(float(offset) - float(onset) for onset, offset in intervals))


def temporal_iou(
    predicted: Sequence[Sequence[float]], gold: Sequence[Sequence[float]]
) -> float:
    """Compute continuous-time IoU after independently unioning both inputs."""

    left = _union_intervals(predicted)
    right = _union_intervals(gold)
    if not left and not right:
        return 1.0
    intersection = 0.0
    i = j = 0
    while i < len(left) and j < len(right):
        intersection += max(
            0.0,
            min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]),
        )
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    union = _interval_length(left) + _interval_length(right) - intersection
    if union <= 0.0:
        return 1.0
    return float(intersection / union)


def _onset_boundary_mae(
    predicted: Sequence[Sequence[float]], gold: Sequence[Sequence[float]]
) -> float:
    """Match all ordered onsets with a 10 s penalty for an unmatched boundary.

    In one dimension, monotone dynamic-programming alignment is the optimal
    assignment under absolute error.  Unlike an earliest-onset-only score, it
    also covers the two candidate events in ``first`` questions and penalizes
    fragmented predictions which invent extra starts.
    """

    gold_union = _union_intervals(gold)
    if not gold_union:
        raise Real10ScoringError("answerable onset metric requires gold evidence")
    predicted_union = _union_intervals(predicted)
    if not predicted_union:
        return DURATION_SECONDS
    predicted_onsets = [interval[0] for interval in predicted_union]
    gold_onsets = [interval[0] for interval in gold_union]
    rows = len(predicted_onsets)
    columns = len(gold_onsets)
    costs = np.empty((rows + 1, columns + 1), dtype=np.float64)
    costs[0, :] = np.arange(columns + 1, dtype=np.float64) * DURATION_SECONDS
    costs[:, 0] = np.arange(rows + 1, dtype=np.float64) * DURATION_SECONDS
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            costs[row, column] = min(
                costs[row - 1, column - 1]
                + abs(predicted_onsets[row - 1] - gold_onsets[column - 1]),
                costs[row - 1, column] + DURATION_SECONDS,
                costs[row, column - 1] + DURATION_SECONDS,
            )
    return float(costs[rows, columns] / max(rows, columns))


def _binary_ranking_metrics(
    probabilities: Sequence[float], labels: Sequence[bool]
) -> tuple[float | None, float | None]:
    scores = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=bool)
    if scores.ndim != 1 or scores.shape != targets.shape or scores.size == 0:
        raise Real10ScoringError("abstention scores and labels are not aligned")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise Real10ScoringError("abstention probabilities must lie in [0, 1]")
    positives = int(targets.sum())
    negatives = int(targets.size - positives)
    if positives == 0:
        return None, None

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    group_end = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    cumulative_tp = np.cumsum(sorted_targets, dtype=np.float64)[group_end]
    cumulative_fp = np.cumsum(~sorted_targets, dtype=np.float64)[group_end]
    recall = cumulative_tp / positives
    precision = cumulative_tp / (cumulative_tp + cumulative_fp)
    previous_recall = np.r_[0.0, recall[:-1]]
    auprc = float(np.sum((recall - previous_recall) * precision))
    if negatives == 0:
        return None, auprc
    tpr = np.r_[0.0, recall]
    fpr = np.r_[0.0, cumulative_fp / negatives]
    auroc = float(np.trapz(tpr, fpr))
    return auroc, auprc


def _expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[bool]
) -> float:
    scores = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if scores.size == 0 or scores.shape != targets.shape:
        raise Real10ScoringError("calibration scores and labels are not aligned")
    # Equal-width [0, .1), ..., [.9, 1] bins; 1.0 is kept in the final bin.
    bin_ids = np.minimum((scores * ECE_BINS).astype(np.int64), ECE_BINS - 1)
    ece = 0.0
    for bin_index in range(ECE_BINS):
        active = bin_ids == bin_index
        if active.any():
            ece += float(active.mean()) * abs(
                float(scores[active].mean()) - float(targets[active].mean())
            )
    return float(ece)


def _mean(values: Sequence[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return float(np.mean(np.asarray(selected, dtype=np.float64))) if selected else None


def _median(values: Sequence[float | None]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return (
        float(np.median(np.asarray(selected, dtype=np.float64))) if selected else None
    )


def summarize_records(records: Sequence[EvaluatedRecord]) -> dict[str, Any]:
    """Aggregate the frozen non-QA endpoints over a row collection."""

    if not records:
        raise Real10ScoringError("cannot summarize an empty record collection")
    probabilities = [record.no_evidence_probability for record in records]
    labels = [record.no_evidence for record in records]
    auroc, auprc = _binary_ranking_metrics(probabilities, labels)
    targets = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(probabilities, dtype=np.float64)
    metrics: dict[str, float | None] = {
        "anchor_temporal_iou_answerable_↑": _mean(
            [record.anchor_iou for record in records]
        ),
        "answer_temporal_iou_answerable_↑": _mean(
            [record.answer_iou for record in records]
        ),
        "union_temporal_iou_answerable_↑": _mean(
            [record.union_iou for record in records]
        ),
        "weakest_role_temporal_iou_answerable_↑": _mean(
            [record.weakest_role_iou for record in records]
        ),
        "onset_boundary_mae_seconds_answerable_↓": _mean(
            [record.onset_boundary_mae_seconds for record in records]
        ),
        "no_evidence_auroc_↑": auroc,
        "no_evidence_auprc_↑": auprc,
        "no_evidence_brier_↓": float(np.mean(np.square(scores - targets))),
        "no_evidence_ece_10bin_↓": _expected_calibration_error(probabilities, labels),
        "no_evidence_retained_energy_ratio_↓": _mean(
            [record.no_evidence_energy_ratio for record in records]
        ),
        "answerable_retained_duration_ratio_↓": _mean(
            [record.answerable_duration_ratio for record in records]
        ),
        "answerable_retained_energy_ratio_↓": _mean(
            [record.answerable_energy_ratio for record in records]
        ),
        "reconstruction_max_abs_error_mean_↓": _mean(
            [record.reconstruction_max_abs_error for record in records]
        ),
        "reconstruction_max_abs_error_maximum_↓": max(
            record.reconstruction_max_abs_error for record in records
        ),
        "physical_separator_forwards_per_record_↓": _mean(
            [record.physical_forwards for record in records]
        ),
        "effective_separator_evaluations_per_record_↓": _mean(
            [record.effective_evaluations for record in records]
        ),
    }
    if set(metrics) != set(METRIC_DIRECTIONS):
        raise AssertionError("metric registry and summary diverged")
    medians = {
        metric: _median([getattr(record, field) for record in records])
        for metric, field in _ROW_DISTRIBUTION_FIELDS.items()
    }
    return {
        "counts": {
            "records_↑": len(records),
            "unique_scenes_↑": len({record.scene_id for record in records}),
            "authoritative_creator_clusters_↑": len(
                {record.creator_id for record in records}
            ),
            "answerable_records_↑": sum(not record.no_evidence for record in records),
            "no_evidence_records_↑": sum(record.no_evidence for record in records),
            "physical_separator_forwards_total_↓": int(
                sum(record.physical_forwards for record in records)
            ),
            "effective_separator_evaluations_total_↓": int(
                sum(record.effective_evaluations for record in records)
            ),
        },
        "metrics": metrics,
        "row_metric_medians": medians,
    }


def _bootstrap_seed(seed: int, label: str) -> int:
    encoded = f"qces-real10-bootstrap-v1\0{seed}\0{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def creator_cluster_bootstrap(
    records: Sequence[EvaluatedRecord],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    label: str = "overall",
) -> dict[str, Any]:
    """Bootstrap creator clusters, never individual question rows."""

    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
    ):
        raise Real10ScoringError("bootstrap replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise Real10ScoringError("bootstrap seed must be an integer")
    if not records:
        raise Real10ScoringError("bootstrap record collection is empty")
    clusters: dict[str, list[EvaluatedRecord]] = {}
    for record in records:
        clusters.setdefault(record.creator_id, []).append(record)
    cluster_ids = sorted(clusters)
    generator = np.random.default_rng(_bootstrap_seed(seed, label))
    values: dict[str, list[float]] = {metric: [] for metric in METRIC_DIRECTIONS}
    for _ in range(replicates):
        indices = generator.integers(0, len(cluster_ids), size=len(cluster_ids))
        sampled: list[EvaluatedRecord] = []
        for index in indices:
            sampled.extend(clusters[cluster_ids[int(index)]])
        metrics = summarize_records(sampled)["metrics"]
        for metric, value in metrics.items():
            if value is not None:
                values[metric].append(float(value))
    intervals: dict[str, Any] = {}
    for metric in METRIC_DIRECTIONS:
        observed = values[metric]
        intervals[metric] = {
            "bootstrap_95ci": (
                [
                    float(value)
                    for value in np.quantile(
                        np.asarray(observed, dtype=np.float64), (0.025, 0.975)
                    )
                ]
                if observed
                else None
            ),
            "valid_replicates_↑": len(observed),
        }
    return {
        "method": "nonparametric bootstrap over authoritative creator clusters",
        "cluster_unit": "creator_id (one scene per creator verified before scoring)",
        "row_bootstrap_used": False,
        "seed": seed,
        "domain_separated_slice_seed": _bootstrap_seed(seed, label),
        "replicates_↑": replicates,
        "creator_clusters_↑": len(cluster_ids),
        "scene_clusters_↑": len({record.scene_id for record in records}),
        "confidence_intervals": intervals,
    }


def _stem_aggregate(rows: Sequence[Mapping[str, Any]]) -> str:
    artifacts = [
        {
            "id": row["id"],
            "evidence": row["stems"]["evidence"],
            "residual": row["stems"]["residual"],
        }
        for row in sorted(rows, key=lambda item: item["id"])
    ]
    return canonical_json_sha256(
        {"format": STEM_AGGREGATE_FORMAT, "artifacts": artifacts}
    )


def _validate_scoring_invariants(records: Sequence[QCESReal10ScoringRecord]) -> None:
    creator_to_scene: dict[str, str] = {}
    creator_to_split: dict[str, str] = {}
    creator_to_upstream: dict[str, str] = {}
    scene_to_creator: dict[str, str] = {}
    family_to_split: dict[str, str] = {}
    for record in records:
        previous = creator_to_scene.setdefault(record.creator_id, record.scene_id)
        if previous != record.scene_id:
            raise Real10ScoringError(
                "creator_id maps to multiple scenes; creator cluster invariant failed"
            )
        previous = creator_to_split.setdefault(record.creator_id, record.split)
        if previous != record.split:
            raise Real10ScoringError("creator_id crosses QCES real splits")
        previous = creator_to_upstream.setdefault(
            record.creator_id, record.upstream_tacos_split
        )
        if previous != record.upstream_tacos_split:
            raise Real10ScoringError("creator_id has inconsistent upstream TACOS split")
        previous = scene_to_creator.setdefault(record.scene_id, record.creator_id)
        if previous != record.creator_id:
            raise Real10ScoringError("scene_id has inconsistent creator_id")
        previous = family_to_split.setdefault(record.scene_family_id, record.split)
        if previous != record.split:
            raise Real10ScoringError("scene_family_id crosses QCES real splits")


def _validate_prediction_receipt(
    *,
    receipt: Mapping[str, Any],
    receipt_identity: Mapping[str, Any],
    prediction_rows: Sequence[Mapping[str, Any]],
    prediction_manifest_identity: Mapping[str, Any],
    records: Sequence[QCESReal10ScoringRecord],
    selected_records: Sequence[QCESReal10ScoringRecord],
    selected_mixtures: Sequence[Mapping[str, Any]],
    split: str,
    allow_real_test: bool,
) -> None:
    del receipt_identity  # The caller retains this immutable identity in the report.
    top = _exact_keys(
        receipt,
        {
            "format",
            "purpose",
            "run_spec",
            "run_identity_sha256",
            "execution",
            "counts",
            "gates",
            "artifacts",
            "input_boundary",
        },
        "render receipt",
    )
    if (
        top["format"] != RENDER_RECEIPT_FORMAT
        or top["purpose"] != "label_free_qces_real10_evidence_and_residual_render"
    ):
        raise Real10ScoringError("unsupported render receipt")
    run_spec = _exact_keys(
        top["run_spec"],
        {
            "format",
            "split",
            "source_inference_manifest",
            "selected_inference_view",
            "selected_mixtures",
            "foundation_cache",
            "method",
            "method_identity_sha256",
            "method_freeze_receipt",
        },
        "render run spec",
    )
    if run_spec["format"] != RUN_SPEC_FORMAT or run_spec["split"] != split:
        raise Real10ScoringError("render receipt belongs to a different split")
    if top["run_identity_sha256"] != canonical_json_sha256(run_spec):
        raise Real10ScoringError("render run identity SHA256 is invalid")
    method = run_spec["method"]
    if (
        not isinstance(method, Mapping)
        or method.get("format") != METHOD_IDENTITY_FORMAT
    ):
        raise Real10ScoringError("render method identity is malformed")
    if (
        run_spec["method_identity_sha256"] != canonical_json_sha256(method)
        or method.get("inference_schema_version") != INFERENCE_SCHEMA_VERSION
        or method.get("prediction_schema_version") != PREDICTION_SCHEMA_VERSION
    ):
        raise Real10ScoringError("render method identity binding is invalid")

    freeze_identity = run_spec["method_freeze_receipt"]
    if split == "real_test":
        if not allow_real_test:
            raise Real10ScoringError(
                "real_test scoring requires explicit allow_real_test authorization"
            )
        freeze = _exact_keys(
            freeze_identity,
            {"path", "sha256", "size_bytes"},
            "real-test method-freeze file identity",
        )
        if not isinstance(freeze["path"], str) or not freeze["path"]:
            raise Real10ScoringError("real-test method-freeze path is invalid")
        freeze_path = Path(freeze["path"]).resolve()
        freeze_payload, actual_freeze_identity = _read_json_object(
            freeze_path, "method-freeze receipt"
        )
        _validate_declared_file_identity(
            freeze, actual_freeze_identity, "method freeze"
        )
        parsed_freeze = _exact_keys(
            freeze_payload,
            {
                "format",
                "purpose",
                "source_split",
                "method",
                "method_identity_sha256",
                "declaration",
            },
            "method-freeze receipt",
        )
        if (
            parsed_freeze["format"] != METHOD_FREEZE_RECEIPT_FORMAT
            or parsed_freeze["purpose"]
            != "freeze_method_before_single_real_test_render"
            or parsed_freeze["source_split"] != "real_dev"
            or parsed_freeze["method"] != method
            or parsed_freeze["method_identity_sha256"] != canonical_json_sha256(method)
        ):
            raise Real10ScoringError("real-test method-freeze gate is not proven")
        declaration = _exact_keys(
            parsed_freeze["declaration"],
            {
                "method_selected_without_real_test_metrics",
                "real_test_model_selection_prohibited",
                "single_real_test_render_after_freeze",
                "real_test_render_count_before_freeze_↓",
            },
            "method-freeze declaration",
        )
        if dict(declaration) != {
            "method_selected_without_real_test_metrics": True,
            "real_test_model_selection_prohibited": True,
            "single_real_test_render_after_freeze": True,
            "real_test_render_count_before_freeze_↓": 0,
        }:
            raise Real10ScoringError("real-test method-freeze declaration failed")
    elif freeze_identity is not None or allow_real_test:
        raise Real10ScoringError(
            "real-test authorization or freeze identity is invalid for real_dev"
        )

    full_projection = project_scoring_manifest(records)
    selected_projection = project_scoring_manifest(selected_records)
    source = _exact_keys(
        run_spec["source_inference_manifest"],
        {"path", "sha256", "size_bytes", "canonical_fingerprint", "record_count"},
        "source inference manifest identity",
    )
    if (
        not isinstance(source["path"], str)
        or not _is_sha256(source["sha256"])
        or isinstance(source["size_bytes"], bool)
        or not isinstance(source["size_bytes"], int)
        or source["size_bytes"] <= 0
        or source["record_count"] != len(records)
        or source["canonical_fingerprint"]
        != canonical_inference_manifest_fingerprint(full_projection)
    ):
        raise Real10ScoringError(
            "scoring-to-full-inference manifest fingerprint mismatch"
        )
    selected = _exact_keys(
        run_spec["selected_inference_view"],
        {
            "canonical_fingerprint",
            "record_count",
            "scene_count",
            "sample_ids",
        },
        "selected inference view",
    )
    selected_ids = sorted(record.sample_id for record in selected_records)
    if (
        selected["canonical_fingerprint"]
        != canonical_inference_manifest_fingerprint(selected_projection)
        or selected["record_count"] != len(selected_records)
        or selected["scene_count"]
        != len({record.scene_id for record in selected_records})
        or selected["sample_ids"] != selected_ids
    ):
        raise Real10ScoringError(
            "scoring-to-selected-inference manifest fingerprint mismatch"
        )
    if run_spec["selected_mixtures"] != list(selected_mixtures):
        raise Real10ScoringError(
            "render selected-mixture identities do not match audio"
        )

    execution = _exact_keys(
        top["execution"],
        {"device", "batch_size", "precision", "determinism", "software"},
        "render execution",
    )
    if (
        execution["batch_size"] != 1
        or execution["precision"] != "float32_no_autocast"
        or execution["device"] not in {"cpu", "cuda"}
        or not isinstance(execution["determinism"], Mapping)
        or execution["determinism"].get("torch_deterministic_algorithms") is not True
        or execution["determinism"].get("cudnn_deterministic") is not True
        or execution["determinism"].get("cudnn_benchmark") is not False
        or execution["determinism"].get("cuda_matmul_allow_tf32") is not False
        or execution["determinism"].get("cudnn_allow_tf32") is not False
        or not isinstance(execution["software"], Mapping)
    ):
        raise Real10ScoringError("render determinism/precision declaration failed")

    expected_physical = sum(
        row["separator_calls"]["physical_forwards"] for row in prediction_rows
    )
    expected_effective = sum(
        row["separator_calls"]["effective_evaluations"] for row in prediction_rows
    )
    declared_maximum = max(
        row["max_evidence_plus_residual_minus_mixture_abs_error"]
        for row in prediction_rows
    )
    counts = _exact_keys(
        top["counts"],
        {
            "prediction_records_↑",
            "unique_scenes_↑",
            "physical_separator_forwards_↓",
            "effective_separator_evaluations_↓",
            "maximum_evidence_plus_residual_minus_mixture_abs_error_↓",
        },
        "render counts",
    )
    if dict(counts) != {
        "prediction_records_↑": len(prediction_rows),
        "unique_scenes_↑": len({row["scene_id"] for row in prediction_rows}),
        "physical_separator_forwards_↓": expected_physical,
        "effective_separator_evaluations_↓": expected_effective,
        "maximum_evidence_plus_residual_minus_mixture_abs_error_↓": declared_maximum,
    }:
        raise Real10ScoringError("render receipt counts disagree with predictions")
    gates = _exact_keys(
        top["gates"],
        {
            "maximum_evidence_plus_residual_minus_mixture_abs_error_threshold_↓",
            "observed_maximum_↓",
            "reconstruction_gate_passed",
        },
        "render reconstruction gate",
    )
    if dict(gates) != {
        "maximum_evidence_plus_residual_minus_mixture_abs_error_threshold_↓": (
            MAX_RECONSTRUCTION_ABS_ERROR
        ),
        "observed_maximum_↓": declared_maximum,
        "reconstruction_gate_passed": True,
    }:
        raise Real10ScoringError("render receipt reconstruction gate is invalid")
    artifacts = _exact_keys(
        top["artifacts"],
        {"prediction_manifest", "stem_aggregate_sha256"},
        "render artifacts",
    )
    manifest = _exact_keys(
        artifacts["prediction_manifest"],
        {"filename", "sha256", "size_bytes", "schema_version", "canonical_fingerprint"},
        "prediction manifest artifact",
    )
    if (
        manifest["filename"] != PREDICTION_MANIFEST_FILENAME
        or manifest["schema_version"] != PREDICTION_SCHEMA_VERSION
        or manifest["canonical_fingerprint"]
        != prediction_manifest_fingerprint(prediction_rows)
    ):
        raise Real10ScoringError("prediction manifest fingerprint is invalid")
    _validate_declared_file_identity(
        manifest, prediction_manifest_identity, "prediction manifest"
    )
    if artifacts["stem_aggregate_sha256"] != _stem_aggregate(prediction_rows):
        raise Real10ScoringError("prediction stem aggregate fingerprint is invalid")
    input_boundary = _exact_keys(
        top["input_boundary"],
        {
            "accepted_manifest_schema",
            "scoring_or_gold_manifest_opened",
            "human_answer_fields_consumed",
            "human_temporal_fields_consumed",
            "clean_reference_metrics_emitted",
        },
        "renderer input boundary",
    )
    if dict(input_boundary) != {
        "accepted_manifest_schema": INFERENCE_SCHEMA_VERSION,
        "scoring_or_gold_manifest_opened": False,
        "human_answer_fields_consumed": False,
        "human_temporal_fields_consumed": False,
        "clean_reference_metrics_emitted": False,
    }:
        raise Real10ScoringError("renderer input boundary admits scoring/oracle data")


def _validate_stem(
    prediction_root: Path,
    identity: Mapping[str, Any],
    context: str,
) -> np.ndarray:
    parsed = _exact_keys(
        identity,
        {
            "path",
            "file_sha256",
            "pcm_f32le_sha256",
            "size_bytes",
            "sample_rate",
            "num_channels",
            "num_samples",
            "duration_seconds",
            "dtype",
            "wav_subtype",
        },
        context,
    )
    path = _resolve_input_file(prediction_root, parsed["path"], f"{context}.path")
    actual = _file_identity(path)
    if (
        actual["sha256"] != parsed["file_sha256"]
        or actual["size_bytes"] != parsed["size_bytes"]
        or not _is_sha256(parsed["pcm_f32le_sha256"])
        or parsed["sample_rate"] != SAMPLE_RATE
        or parsed["num_channels"] != NUM_CHANNELS
        or parsed["num_samples"] != NUM_SAMPLES
        or parsed["duration_seconds"] != DURATION_SECONDS
        or parsed["dtype"] != "float32"
        or parsed["wav_subtype"] != CANONICAL_WAV_SUBTYPE
    ):
        raise Real10ScoringError(f"{context} identity is invalid")
    samples = _decode_float_wav(path, deterministic_stem=True)
    if pcm_f32le_sha256(samples) != parsed["pcm_f32le_sha256"]:
        raise Real10ScoringError(f"{context} PCM SHA256 mismatch")
    return samples


def validate_scoring_inputs(
    *,
    scoring_manifest_path: Path,
    dataset_root: Path,
    prediction_root: Path,
    split: str = "real_dev",
    allow_real_test: bool = False,
) -> ValidatedScoringInputs:
    """Validate every dataset, render, audio, and split identity before scoring."""

    if split not in {"real_dev", "real_test"}:
        raise Real10ScoringError("split must be real_dev or real_test")
    resolved_dataset_root = dataset_root.resolve()
    resolved_prediction_root = prediction_root.resolve()
    if (
        not resolved_dataset_root.is_dir()
        or dataset_root.is_symlink()
        or not resolved_prediction_root.is_dir()
        or prediction_root.is_symlink()
    ):
        raise Real10ScoringError(
            "dataset and prediction roots must be regular directories"
        )
    resolved_scoring_manifest = scoring_manifest_path.resolve()
    try:
        resolved_scoring_manifest.relative_to(resolved_dataset_root)
    except ValueError as error:
        raise Real10ScoringError("scoring manifest escapes dataset root") from error

    raw_scoring, scoring_identity = _read_jsonl_objects(
        resolved_scoring_manifest, "QCES-Real-10 scoring manifest"
    )
    try:
        records = parse_scoring_manifest(raw_scoring)
    except (TypeError, ValueError) as error:
        raise Real10ScoringError(
            f"invalid {SCORING_SCHEMA_VERSION} scoring manifest: {error}"
        ) from error
    _validate_scoring_invariants(records)
    selected_records = tuple(
        sorted(
            (record for record in records if record.split == split),
            key=lambda record: record.sample_id,
        )
    )
    if not selected_records:
        raise Real10ScoringError(f"scoring manifest has no {split} records")

    prediction_manifest_path = resolved_prediction_root / PREDICTION_MANIFEST_FILENAME
    raw_predictions, prediction_identity = _read_jsonl_objects(
        prediction_manifest_path, "QCES-Real-10 prediction manifest"
    )
    try:
        prediction_rows = tuple(
            dict(validate_prediction_record(row)) for row in raw_predictions
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise Real10ScoringError(f"invalid prediction manifest: {error}") from error
    prediction_ids = [row["id"] for row in prediction_rows]
    selected_ids = [record.sample_id for record in selected_records]
    if len(prediction_ids) != len(set(prediction_ids)):
        raise Real10ScoringError("prediction manifest contains duplicate IDs")
    if set(prediction_ids) != set(selected_ids):
        raise Real10ScoringError(
            "prediction ID coverage is not exact for the requested split: "
            f"missing={sorted(set(selected_ids) - set(prediction_ids))[:10]}, "
            f"extra={sorted(set(prediction_ids) - set(selected_ids))[:10]}"
        )
    if any(row["split"] != split for row in prediction_rows):
        raise Real10ScoringError("prediction manifest mixes or mislabels splits")
    prediction_by_id = {row["id"]: row for row in prediction_rows}

    # Verify unique mixture bytes once per scene and bind them to both manifests.
    mixture_by_scene: dict[str, np.ndarray] = {}
    mixture_identity_by_scene: dict[str, dict[str, Any]] = {}
    for record in selected_records:
        if record.scene_id in mixture_by_scene:
            continue
        mixture_path = _resolve_input_file(
            resolved_scoring_manifest.parent,
            record.mixture_path,
            f"mixture_path for {record.scene_id}",
        )
        try:
            mixture_path.relative_to(resolved_dataset_root)
        except ValueError as error:
            raise Real10ScoringError("mixture escapes dataset root") from error
        identity = _file_identity(mixture_path)
        if identity["sha256"] != record.mixture_sha256:
            raise Real10ScoringError(
                f"scoring mixture file SHA256 mismatch for {record.scene_id}"
            )
        samples = _decode_float_wav(mixture_path, deterministic_stem=False)
        if float(np.max(np.abs(samples))) <= 1e-7:
            raise Real10ScoringError(f"canonical mixture is silent: {record.scene_id}")
        mixture_by_scene[record.scene_id] = samples
        mixture_identity_by_scene[record.scene_id] = {
            "scene_id": record.scene_id,
            "manifest_path": record.mixture_path,
            "file_sha256": identity["sha256"],
            "pcm_f32le_sha256": pcm_f32le_sha256(samples),
            "size_bytes": identity["size_bytes"],
        }

    evaluated: list[EvaluatedRecord] = []
    expected_output_files = {
        prediction_manifest_path.resolve(),
        (resolved_prediction_root / RENDER_RECEIPT_FILENAME).resolve(),
    }
    independent_maximum = 0.0
    for record in selected_records:
        row = prediction_by_id[record.sample_id]
        if (
            row["scene_id"] != record.scene_id
            or row["question_index"] != record.question_index
            or row["relation"] != record.relation
            or row["inference_record_sha256"]
            != inference_record_fingerprint(record.to_inference())
        ):
            raise Real10ScoringError(
                f"prediction/scoring inference binding mismatch: {record.sample_id}"
            )
        mixture_identity = mixture_identity_by_scene[record.scene_id]
        declared_mixture = row["mixture"]
        if dict(declared_mixture) != {
            "manifest_path": record.mixture_path,
            "file_sha256": record.mixture_sha256,
            "pcm_f32le_sha256": mixture_identity["pcm_f32le_sha256"],
            "size_bytes": mixture_identity["size_bytes"],
            "sample_rate": SAMPLE_RATE,
            "num_channels": NUM_CHANNELS,
            "num_samples": NUM_SAMPLES,
            "duration_seconds": DURATION_SECONDS,
        }:
            raise Real10ScoringError(
                f"prediction mixture identity mismatch: {record.sample_id}"
            )
        evidence = _validate_stem(
            resolved_prediction_root,
            row["stems"]["evidence"],
            f"evidence stem for {record.sample_id}",
        )
        residual = _validate_stem(
            resolved_prediction_root,
            row["stems"]["residual"],
            f"residual stem for {record.sample_id}",
        )
        for stem_name in ("evidence", "residual"):
            expected_output_files.add(
                _resolve_input_file(
                    resolved_prediction_root,
                    row["stems"][stem_name]["path"],
                    f"{stem_name} stem path",
                ).resolve()
            )
        mixture = mixture_by_scene[record.scene_id]
        reconstruction = evidence + residual - mixture
        if not np.isfinite(reconstruction).all():
            raise Real10ScoringError("independent reconstruction contains NaN/Inf")
        reconstruction_error = float(np.max(np.abs(reconstruction)))
        independent_maximum = max(independent_maximum, reconstruction_error)
        declared_error = _finite_float(
            row["max_evidence_plus_residual_minus_mixture_abs_error"],
            "declared reconstruction error",
        )
        if reconstruction_error != declared_error:
            raise Real10ScoringError(
                f"independent reconstruction differs from renderer declaration: "
                f"{record.sample_id}"
            )
        if reconstruction_error > MAX_RECONSTRUCTION_ABS_ERROR:
            raise Real10ScoringError(
                f"independent E+R-X gate failed for {record.sample_id}: "
                f"{reconstruction_error} > {MAX_RECONSTRUCTION_ABS_ERROR}"
            )

        prediction = row["prediction"]
        anchor_iou: float | None = None
        answer_iou: float | None = None
        union_iou: float | None = None
        weakest_iou: float | None = None
        onset_mae: float | None = None
        no_evidence_energy: float | None = None
        answerable_duration: float | None = None
        answerable_energy: float | None = None
        mixture_energy = float(
            np.dot(mixture.astype(np.float64), mixture.astype(np.float64))
        )
        if mixture_energy <= 0.0:
            raise Real10ScoringError("mixture energy denominator is zero")
        evidence_energy = float(
            np.dot(evidence.astype(np.float64), evidence.astype(np.float64))
        )
        energy_ratio = evidence_energy / mixture_energy
        if record.no_evidence:
            # No residual semantic endpoint is defined for negatives: R should
            # remain close to X and still contain the scene.
            no_evidence_energy = energy_ratio
        else:
            anchor_iou = temporal_iou(
                prediction["anchor_intervals"], record.anchor_intervals
            )
            answer_iou = temporal_iou(
                prediction["answer_intervals"], record.answer_intervals
            )
            union_iou = temporal_iou(
                prediction["union_intervals"], record.evidence_intervals
            )
            weakest_iou = min(anchor_iou, answer_iou)
            onset_mae = 0.5 * (
                _onset_boundary_mae(
                    prediction["anchor_intervals"], record.anchor_intervals
                )
                + _onset_boundary_mae(
                    prediction["answer_intervals"], record.answer_intervals
                )
            )
            answerable_duration = (
                _interval_length(_union_intervals(prediction["union_intervals"]))
                / DURATION_SECONDS
            )
            answerable_energy = energy_ratio
        evaluated.append(
            EvaluatedRecord(
                sample_id=record.sample_id,
                scene_id=record.scene_id,
                creator_id=record.creator_id,
                split=record.split,
                relation=record.relation,
                upstream_tacos_split=record.upstream_tacos_split,
                no_evidence=record.no_evidence,
                no_evidence_probability=float(prediction["no_evidence_probability"]),
                anchor_iou=anchor_iou,
                answer_iou=answer_iou,
                union_iou=union_iou,
                weakest_role_iou=weakest_iou,
                onset_boundary_mae_seconds=onset_mae,
                no_evidence_energy_ratio=no_evidence_energy,
                answerable_duration_ratio=answerable_duration,
                answerable_energy_ratio=answerable_energy,
                reconstruction_max_abs_error=reconstruction_error,
                physical_forwards=float(
                    _positive_int(
                        row["separator_calls"]["physical_forwards"],
                        "physical separator forwards",
                    )
                ),
                effective_evaluations=float(
                    _positive_int(
                        row["separator_calls"]["effective_evaluations"],
                        "effective separator evaluations",
                    )
                ),
            )
        )

    actual_output_files: set[Path] = set()
    for path in resolved_prediction_root.rglob("*"):
        if path.is_symlink():
            raise Real10ScoringError(f"prediction output contains a symlink: {path}")
        if path.is_file():
            actual_output_files.add(path.resolve())
    if actual_output_files != expected_output_files:
        raise Real10ScoringError(
            "prediction root has missing or undeclared artifacts: "
            "missing="
            f"{sorted(map(str, expected_output_files - actual_output_files))[:5]}, "
            f"extra={sorted(map(str, actual_output_files - expected_output_files))[:5]}"
        )

    receipt, receipt_identity = _read_json_object(
        resolved_prediction_root / RENDER_RECEIPT_FILENAME, "render receipt"
    )
    selected_mixture_rows = [
        mixture_identity_by_scene[scene_id]
        for scene_id in sorted(mixture_identity_by_scene)
    ]
    _validate_prediction_receipt(
        receipt=receipt,
        receipt_identity=receipt_identity,
        prediction_rows=prediction_rows,
        prediction_manifest_identity=prediction_identity,
        records=records,
        selected_records=selected_records,
        selected_mixtures=selected_mixture_rows,
        split=split,
        allow_real_test=allow_real_test,
    )
    declared_receipt_maximum = receipt["gates"]["observed_maximum_↓"]
    if independent_maximum != declared_receipt_maximum:
        raise Real10ScoringError(
            "independently observed reconstruction maximum differs from receipt"
        )
    return ValidatedScoringInputs(
        scoring_manifest_path=resolved_scoring_manifest,
        dataset_root=resolved_dataset_root,
        prediction_root=resolved_prediction_root,
        scoring_manifest_identity=scoring_identity,
        prediction_manifest_identity=prediction_identity,
        render_receipt_identity=receipt_identity,
        split=split,
        records=records,
        selected_records=selected_records,
        prediction_rows=prediction_rows,
        render_receipt=receipt,
        evaluated_records=tuple(evaluated),
        independently_observed_max_reconstruction_error=independent_maximum,
    )


def _slice_report(
    records: Sequence[EvaluatedRecord],
    *,
    label: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    return {
        **summarize_records(records),
        "creator_clustered_bootstrap": creator_cluster_bootstrap(
            records,
            replicates=bootstrap_replicates,
            seed=seed,
            label=label,
        ),
    }


def _grouped_reports(
    records: Sequence[EvaluatedRecord],
    *,
    field: str,
    label_prefix: str,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    values = sorted({str(getattr(record, field)) for record in records})
    return {
        value: _slice_report(
            [record for record in records if str(getattr(record, field)) == value],
            label=f"{label_prefix}:{value}",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
        for value in values
    }


def _worst_relation(by_relation: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        candidates = [
            (relation, report["metrics"][metric])
            for relation, report in by_relation.items()
            if report["metrics"][metric] is not None
        ]
        if not candidates:
            result[metric] = None
            continue

        def value_key(item: tuple[str, Any]) -> Any:
            return item[1]

        relation, value = (
            min(candidates, key=value_key)
            if direction == "↑"
            else max(candidates, key=value_key)
        )
        result[metric] = {"relation": relation, "value": value}
    return result


def build_score_report(
    validated: ValidatedScoringInputs,
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build the deterministic aggregate report after all integrity gates pass."""

    records = validated.evaluated_records
    overall = _slice_report(
        records,
        label=f"split:{validated.split}:overall",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    by_relation = _grouped_reports(
        records,
        field="relation",
        label_prefix=f"split:{validated.split}:relation",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    by_upstream = _grouped_reports(
        records,
        field="upstream_tacos_split",
        label_prefix=f"split:{validated.split}:upstream_tacos_split",
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    by_split = {
        validated.split: _slice_report(
            records,
            label=f"split:{validated.split}",
            bootstrap_replicates=bootstrap_replicates,
            seed=seed,
        )
    }
    receipt = validated.render_receipt
    report = {
        "format": REPORT_FORMAT,
        "scope": (
            "QCES-Real-10 non-QA temporal, abstention, compactness, "
            "consistency, and cost scoring"
        ),
        "split": validated.split,
        "schema_versions": {
            "scoring": SCORING_SCHEMA_VERSION,
            "inference_projection": INFERENCE_SCHEMA_VERSION,
            "prediction": PREDICTION_SCHEMA_VERSION,
            "render_receipt": RENDER_RECEIPT_FORMAT,
        },
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
        "metric_directions": dict(METRIC_DIRECTIONS),
        "scoring_settings": {
            "bootstrap_replicates_↑": bootstrap_replicates,
            "bootstrap_seed": seed,
            "ece_equal_width_bins_↑": ECE_BINS,
            "maximum_reconstruction_abs_error_threshold_↓": (
                MAX_RECONSTRUCTION_ABS_ERROR
            ),
        },
        "identities": {
            "scoring_manifest": dict(validated.scoring_manifest_identity),
            "prediction_manifest": dict(validated.prediction_manifest_identity),
            "render_receipt": dict(validated.render_receipt_identity),
            "projected_full_inference_manifest_fingerprint": (
                canonical_inference_manifest_fingerprint(
                    project_scoring_manifest(validated.records)
                )
            ),
            "projected_selected_inference_manifest_fingerprint": (
                canonical_inference_manifest_fingerprint(
                    project_scoring_manifest(validated.selected_records)
                )
            ),
            "prediction_manifest_fingerprint": prediction_manifest_fingerprint(
                validated.prediction_rows
            ),
            "render_run_identity_sha256": receipt["run_identity_sha256"],
            "render_method_identity_sha256": receipt["run_spec"][
                "method_identity_sha256"
            ],
            "scorer_source": _file_identity(Path(__file__)),
            "scoring_software": _software_versions(),
        },
        "integrity_gates": {
            "all_integrity_gates_passed": True,
            "exact_prediction_id_coverage": True,
            "exact_split_binding": True,
            "scoring_to_inference_bindings_verified": True,
            "creator_is_authoritative_cluster": True,
            "one_scene_per_creator_verified": True,
            "upstream_tacos_split_invariants_verified": True,
            "prediction_manifest_and_receipt_hashes_verified": True,
            "mixture_and_stem_file_and_pcm_hashes_verified": True,
            "deterministic_float_stem_bytes_verified": True,
            "independent_reconstruction_gate_passed": True,
            "independently_observed_max_abs_E_plus_R_minus_X_↓": (
                validated.independently_observed_max_reconstruction_error
            ),
            "maximum_allowed_abs_E_plus_R_minus_X_↓": (MAX_RECONSTRUCTION_ABS_ERROR),
            "oracle_or_clean_waveform_references_consumed_↓": 0,
            "waveform_distortion_metrics_computed_↓": 0,
            "row_bootstrap_replicates_↓": 0,
            "real_test_method_freeze_gate_verified": (
                True if validated.split == "real_test" else None
            ),
        },
        "metric_definitions": {
            "point_aggregation": (
                "arithmetic question-row mean; the frozen design gives every "
                "scene the same eight slots and every relation slice the same "
                "number of slots per scene"
            ),
            "temporal_iou": (
                "continuous-time intersection-over-union after interval "
                "unioning; answerable rows only"
            ),
            "weakest_role": (
                "per answerable row min(anchor tIoU, answer tIoU), then averaged"
            ),
            "onset_boundary_mae": (
                "monotone minimum-cost alignment of every predicted/gold role "
                "onset, averaged over anchor and answer; each unmatched "
                "boundary receives a 10 s penalty"
            ),
            "no_evidence_calibration": (
                "positive class=no_evidence; Brier plus 10 equal-width-bin ECE"
            ),
            "retained_energy_ratio": "sum(E^2) / sum(X^2), computed in float64",
            "retained_duration_ratio": "predicted union interval duration / 10 s",
            "reconstruction": (
                "maximum samplewise abs(float32 E + R - X), recomputed from "
                "stored WAVs"
            ),
            "bootstrap": (
                "fixed-seed nonparametric creator-cluster bootstrap; all "
                "questions from a sampled scene remain together"
            ),
        },
        "overall": overall,
        "slices": {
            "by_relation": by_relation,
            "by_upstream_tacos_split": by_upstream,
            "by_split": by_split,
        },
        "worst_relation": _worst_relation(by_relation),
        "claim_boundary": [
            "Natural TACOS recordings have no clean event waveform reference.",
            "E+R=X is arithmetic consistency, not semantic faithfulness.",
            "No-evidence rows use evidence-energy retention and abstention only.",
            "Answerable compactness ratios are interpretable only at matched "
            "frozen-QA sufficiency.",
            "QA sufficiency is evaluated by a separate frozen-auditor protocol.",
        ],
    }
    # Fail closed against accidental NaN/Infinity or non-JSON additions.
    json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return report


def score_qces_real10(
    *,
    scoring_manifest_path: Path,
    dataset_root: Path,
    prediction_root: Path,
    split: str = "real_dev",
    allow_real_test: bool = False,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate and score one exact QCES-Real-10 render split."""

    validated = validate_scoring_inputs(
        scoring_manifest_path=scoring_manifest_path,
        dataset_root=dataset_root,
        prediction_root=prediction_root,
        split=split,
        allow_real_test=allow_real_test,
    )
    return build_score_report(
        validated, bootstrap_replicates=bootstrap_replicates, seed=seed
    )


def write_score_report_exclusive(path: Path, report: Mapping[str, Any]) -> None:
    """Exclusively write one canonical JSON report without overwrite."""

    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"refusing to overwrite score report: {resolved}")
    descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        resolved.unlink(missing_ok=True)
        raise


__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "ECE_BINS",
    "EvaluatedRecord",
    "MAX_RECONSTRUCTION_ABS_ERROR",
    "METRIC_DIRECTIONS",
    "REPORT_FORMAT",
    "Real10ScoringError",
    "ValidatedScoringInputs",
    "build_score_report",
    "creator_cluster_bootstrap",
    "score_qces_real10",
    "summarize_records",
    "temporal_iou",
    "validate_scoring_inputs",
    "write_score_report_exclusive",
]
