#!/usr/bin/env python3
"""Select a retained QCES refiner epoch with a frozen joint safety rule.

This is deliberately an offline selector.  It never trusts ``checkpoint.pt``
or a trainer's independent best-metric files.  Every candidate must be an
immutable ``--save-every-epoch`` checkpoint, must have a factorization report
cryptographically bound to that checkpoint, and must be comparable to the base
report on exactly the same validation records.

The frozen pilot rule is:

1. no-evidence retained-ratio mean ↓ <= base + tolerance;
2. answerable false-silence rate ↓ and SI-SDR-below--20-dB rate ↓ do not
   exceed the base;
3. evidence SD-SDR p10 ↑ and worst ↑ do not fall below the base;
4. among feasible epochs, maximize answerable temporal IoU ↑, then evidence
   SD-SDR mean ↑, then minimize weakest-role waveform error ↓.

Old factorization reports need not contain pre-aggregated SD-SDR p10/worst
fields.  Those tails are recomputed from answerable per-item values using the
declared linear percentile ``(n - 1) * 0.1``.  Missing or mismatched item
coverage fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RECEIPT_FORMAT = "qces_frozen_joint_checkpoint_selection_v1"
FACTORIZATION_FORMAT = "qces_audiosep_factorization_v1"
DEPLOYABLE_CONDITION = "learned_semantic__learned_soft_temporal"
SD_SDR_ITEM_KEY = "evidence_sd_sdr_db_↑"
SI_SDR_ITEM_KEY = "evidence_si_sdr_db_↑"
SD_SDR_MEAN_KEY = "evidence_sd_sdr_answerable_mean_db_↑"
SD_SDR_P10_KEY = "evidence_sd_sdr_answerable_p10_db_↑"
SD_SDR_MIN_KEY = "evidence_sd_sdr_answerable_minimum_db_↑"
SI_SDR_FAILURE_KEY = "evidence_si_sdr_below_minus20_db_rate_↓"
NE_RETAINED_KEY = "no_evidence_retained_ratio_mean_↓"
FALSE_SILENCE_KEY = "answerable_false_silence_rate_↓"
TEMPORAL_IOU_KEY = "answerable_temporal_iou_↑"
WEAKEST_ERROR_KEY = "weakest_role_waveform_error_↓"


class SelectionInputError(ValueError):
    """An input cannot support an auditable frozen selection."""


@dataclass(frozen=True)
class Thresholds:
    no_evidence_tolerance: float = 0.01
    false_silence_tolerance: float = 0.0
    failure_rate_tolerance: float = 0.0
    sd_sdr_tail_tolerance_db: float = 0.0
    tie_tolerance: float = 1e-12

    def validate(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0.0:
                raise SelectionInputError(
                    f"{name} must be a finite non-negative number"
                )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> Dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SelectionInputError(f"missing file: {resolved}")
    try:
        portable_path = resolved.relative_to(PROJECT_ROOT).as_posix()
        path_base = "project_root"
    except ValueError:
        portable_path = str(resolved)
        path_base = "absolute"
    return {
        "path": portable_path,
        "path_base": path_base,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def load_json(path: Path, label: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    identity = file_identity(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SelectionInputError(f"invalid {label} JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise SelectionInputError(f"{label} must contain a JSON object: {path}")
    return payload, identity


def require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SelectionInputError(f"missing or invalid mapping: {label}")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SelectionInputError(f"missing or invalid list: {label}")
    return value


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise SelectionInputError(f"{label} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SelectionInputError(f"missing or invalid metric: {label}") from error
    if not math.isfinite(result):
        raise SelectionInputError(f"non-finite metric: {label}")
    return result


def exact_int(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise SelectionInputError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise SelectionInputError(f"missing or invalid integer: {label}") from error
    if result != value or result < minimum:
        raise SelectionInputError(f"invalid integer: {label}={value!r}")
    return result


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    """Deterministic linear interpolation at index ``(n - 1) * quantile``."""

    if not values:
        raise SelectionInputError("cannot compute a percentile from no values")
    if not 0.0 <= quantile <= 1.0:
        raise SelectionInputError("quantile must be within [0, 1]")
    ordered = sorted(finite_float(value, "percentile value") for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def identities_match(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left.get("sha256") == right.get("sha256")
        and left.get("size_bytes") == right.get("size_bytes")
    )


def verify_declared_file_identity(
    declared: Any, label: str
) -> Dict[str, Any]:
    identity = require_mapping(declared, label)
    path_value = identity.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise SelectionInputError(f"missing provenance path: {label}.path")
    expected_hash = identity.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise SelectionInputError(f"missing provenance hash: {label}.sha256")
    expected_size = exact_int(identity.get("size_bytes"), f"{label}.size_bytes")
    actual = file_identity(Path(path_value))
    if actual["sha256"] != expected_hash or actual["size_bytes"] != expected_size:
        raise SelectionInputError(
            f"declared identity does not match file bytes: {label}"
        )
    return actual


def require_close(left: float, right: float, label: str, tolerance: float = 1e-5) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
        raise SelectionInputError(
            f"metric/provenance disagreement for {label}: {left} != {right}"
        )


def _identity_signature(value: Any, label: str) -> tuple[Any, ...]:
    identity = require_mapping(value, label)
    sha256 = identity.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise SelectionInputError(f"missing provenance hash: {label}.sha256")
    size = identity.get("size_bytes")
    if size is not None:
        size = exact_int(size, f"{label}.size_bytes")
    return sha256, size


def _source_file_signature(report: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    software = require_mapping(
        require_mapping(report.get("provenance"), "provenance").get("software"),
        "provenance.software",
    )
    rows = require_list(
        software.get("qces_source_files"),
        "provenance.software.qces_source_files",
    )
    signature: list[tuple[str, str]] = []
    for index, row_value in enumerate(rows):
        row = require_mapping(row_value, f"qces_source_files[{index}]")
        path = row.get("path")
        digest = row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
            raise SelectionInputError(
                f"invalid source provenance at qces_source_files[{index}]"
            )
        signature.append((Path(path).name, digest))
    if not signature or len({name for name, _ in signature}) != len(signature):
        raise SelectionInputError("empty or duplicate QCES source provenance")
    return tuple(sorted(signature))


def _audio_tree_signature(report: Mapping[str, Any]) -> tuple[Any, ...]:
    software = require_mapping(
        require_mapping(report.get("provenance"), "provenance").get("software"),
        "provenance.software",
    )
    tree = require_mapping(
        software.get("audiosep_source_tree"),
        "provenance.software.audiosep_source_tree",
    )
    digest = tree.get("sha256")
    count = tree.get("hashed_file_count")
    if not isinstance(digest, str) or len(digest) != 64:
        raise SelectionInputError("invalid AudioSep source-tree provenance")
    return digest, exact_int(count, "audiosep_source_tree.hashed_file_count")


def _dataset_signature(report: Mapping[str, Any]) -> Dict[str, Any]:
    provenance = require_mapping(report.get("provenance"), "provenance")
    inputs = require_mapping(provenance.get("inputs"), "provenance.inputs")
    manifest = require_mapping(inputs.get("manifest"), "provenance.inputs.manifest")
    audio = require_mapping(
        inputs.get("dataset_audio_inputs"),
        "provenance.inputs.dataset_audio_inputs",
    )
    statistics = require_mapping(report.get("dataset_statistics"), "dataset_statistics")
    return {
        "manifest": _identity_signature(manifest, "provenance.inputs.manifest"),
        "dataset_audio_sha256": _identity_signature(
            audio, "provenance.inputs.dataset_audio_inputs"
        )[0],
        "dataset_audio_total_bytes": exact_int(
            audio.get("total_bytes"), "dataset_audio_inputs.total_bytes"
        ),
        "dataset_audio_unique_waveform_count": exact_int(
            audio.get("unique_waveform_count"),
            "dataset_audio_inputs.unique_waveform_count",
        ),
        "record_count": exact_int(statistics.get("record_count"), "record_count", 1),
        "answerable_count": exact_int(
            statistics.get("answerable_count"), "answerable_count", 1
        ),
        "no_evidence_count": exact_int(
            statistics.get("no_evidence_count"), "no_evidence_count", 1
        ),
    }


def _report_items(
    report: Mapping[str, Any], expected_coverage: Optional[Mapping[str, bool]] = None
) -> tuple[Dict[str, bool], list[float], list[float]]:
    items = require_list(report.get("items"), "items")
    coverage: Dict[str, bool] = {}
    sd_values: list[float] = []
    si_values: list[float] = []
    for index, item_value in enumerate(items):
        item = require_mapping(item_value, f"items[{index}]")
        sample_id = item.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SelectionInputError(f"invalid items[{index}].id")
        if sample_id in coverage:
            raise SelectionInputError(f"duplicate report item id: {sample_id}")
        no_evidence = item.get("no_evidence")
        if not isinstance(no_evidence, bool):
            raise SelectionInputError(f"invalid no_evidence flag for {sample_id}")
        coverage[sample_id] = no_evidence
        condition = require_mapping(
            require_mapping(item.get("conditions"), f"{sample_id}.conditions").get(
                DEPLOYABLE_CONDITION
            ),
            f"{sample_id}.{DEPLOYABLE_CONDITION}",
        )
        sd_value = condition.get(SD_SDR_ITEM_KEY)
        si_value = condition.get(SI_SDR_ITEM_KEY)
        if no_evidence:
            if sd_value is not None or si_value is not None:
                raise SelectionInputError(
                    f"no-evidence item has answerable SDR metrics: {sample_id}"
                )
        else:
            sd_values.append(finite_float(sd_value, f"{sample_id}.{SD_SDR_ITEM_KEY}"))
            si_values.append(finite_float(si_value, f"{sample_id}.{SI_SDR_ITEM_KEY}"))
    if expected_coverage is not None and coverage != dict(expected_coverage):
        missing = sorted(set(expected_coverage) - set(coverage))
        extra = sorted(set(coverage) - set(expected_coverage))
        changed = sorted(
            key
            for key in set(coverage) & set(expected_coverage)
            if coverage[key] != expected_coverage[key]
        )
        raise SelectionInputError(
            "report item coverage/no-evidence flags differ from base: "
            f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}"
        )
    signature = _dataset_signature(report)
    if len(coverage) != signature["record_count"]:
        raise SelectionInputError("dataset record_count does not match report items")
    if len(sd_values) != signature["answerable_count"]:
        raise SelectionInputError("answerable_count does not match per-item SDR coverage")
    if sum(coverage.values()) != signature["no_evidence_count"]:
        raise SelectionInputError("no_evidence_count does not match report items")
    return coverage, sd_values, si_values


def _validate_factorization_protocol(report: Mapping[str, Any], label: str) -> None:
    if report.get("format") != FACTORIZATION_FORMAT:
        raise SelectionInputError(f"{label} has unsupported factorization format")
    if report.get("split") != "val":
        raise SelectionInputError(f"{label} must evaluate split=val")
    protocol = require_mapping(report.get("protocol"), f"{label}.protocol")
    if protocol.get("separator_training") != "frozen; evaluation-only":
        raise SelectionInputError(f"{label} does not certify a frozen separator")
    if protocol.get("composer_training") != "frozen; evaluation-only":
        raise SelectionInputError(f"{label} does not certify evaluation-only composer")
    condition = require_mapping(
        require_mapping(protocol.get("conditions"), f"{label}.protocol.conditions").get(
            DEPLOYABLE_CONDITION
        ),
        f"{label}.{DEPLOYABLE_CONDITION} protocol",
    )
    if condition.get("deployable") is not True or condition.get("uses_oracle_annotation") is not False:
        raise SelectionInputError(f"{label} deployable condition is absent or oracle")


def _extract_report_metrics(
    report: Mapping[str, Any], expected_coverage: Optional[Mapping[str, bool]] = None
) -> tuple[Dict[str, float], Dict[str, bool], Dict[str, str]]:
    coverage, sd_values, si_values = _report_items(report, expected_coverage)
    summaries = require_mapping(report.get("condition_summaries"), "condition_summaries")
    summary = require_mapping(
        summaries.get(DEPLOYABLE_CONDITION),
        f"condition_summaries.{DEPLOYABLE_CONDITION}",
    )
    no_evidence = finite_float(summary.get(NE_RETAINED_KEY), NE_RETAINED_KEY)
    sd_mean = finite_float(summary.get(SD_SDR_MEAN_KEY), SD_SDR_MEAN_KEY)
    failure_rate = finite_float(summary.get(SI_SDR_FAILURE_KEY), SI_SDR_FAILURE_KEY)
    derived_sd_mean = sum(sd_values) / len(sd_values)
    derived_failure_rate = sum(value < -20.0 for value in si_values) / len(si_values)
    require_close(sd_mean, derived_sd_mean, SD_SDR_MEAN_KEY)
    require_close(failure_rate, derived_failure_rate, SI_SDR_FAILURE_KEY)

    derived_p10 = linear_percentile(sd_values, 0.1)
    derived_minimum = min(sd_values)
    metric_sources = {
        SD_SDR_P10_KEY: "derived_from_answerable_items_linear_(n-1)*0.1",
        SD_SDR_MIN_KEY: "derived_from_answerable_items",
    }
    if summary.get(SD_SDR_P10_KEY) is not None:
        require_close(
            finite_float(summary.get(SD_SDR_P10_KEY), SD_SDR_P10_KEY),
            derived_p10,
            SD_SDR_P10_KEY,
        )
        metric_sources[SD_SDR_P10_KEY] = "summary_cross_checked_against_items"
    if summary.get(SD_SDR_MIN_KEY) is not None:
        require_close(
            finite_float(summary.get(SD_SDR_MIN_KEY), SD_SDR_MIN_KEY),
            derived_minimum,
            SD_SDR_MIN_KEY,
        )
        metric_sources[SD_SDR_MIN_KEY] = "summary_cross_checked_against_items"

    calibration = require_mapping(
        report.get("no_evidence_calibration"), "no_evidence_calibration"
    )
    threshold_fit = require_mapping(
        calibration.get("validation_only_threshold_fit"),
        "no_evidence_calibration.validation_only_threshold_fit",
    )
    if (
        threshold_fit.get("status") != "fitted_on_validation_only"
        or threshold_fit.get("selection_split") != "val"
    ):
        raise SelectionInputError("false-silence metric is not validation-only calibrated")
    false_silence = finite_float(
        require_mapping(threshold_fit.get("metrics"), "no-evidence fit metrics").get(
            FALSE_SILENCE_KEY
        ),
        FALSE_SILENCE_KEY,
    )
    for label, value in (
        (NE_RETAINED_KEY, no_evidence),
        (SI_SDR_FAILURE_KEY, failure_rate),
        (FALSE_SILENCE_KEY, false_silence),
    ):
        if not 0.0 <= value <= 1.0:
            raise SelectionInputError(f"out-of-range rate: {label}={value}")
    return (
        {
            NE_RETAINED_KEY: no_evidence,
            FALSE_SILENCE_KEY: false_silence,
            SI_SDR_FAILURE_KEY: failure_rate,
            SD_SDR_MEAN_KEY: sd_mean,
            SD_SDR_P10_KEY: derived_p10,
            SD_SDR_MIN_KEY: derived_minimum,
        },
        coverage,
        metric_sources,
    )


def _validate_report_comparability(
    base: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    if _dataset_signature(base) != _dataset_signature(candidate):
        raise SelectionInputError("candidate dataset provenance differs from base")
    base_inputs = require_mapping(
        require_mapping(base.get("provenance"), "base provenance").get("inputs"),
        "base provenance.inputs",
    )
    candidate_inputs = require_mapping(
        require_mapping(candidate.get("provenance"), "candidate provenance").get(
            "inputs"
        ),
        "candidate provenance.inputs",
    )
    for key in ("audiosep_config", "audiosep_checkpoint", "oracle_semantic_cache"):
        if _identity_signature(base_inputs.get(key), f"base {key}") != _identity_signature(
            candidate_inputs.get(key), f"candidate {key}"
        ):
            raise SelectionInputError(f"candidate {key} provenance differs from base")
    if _source_file_signature(base) != _source_file_signature(candidate):
        raise SelectionInputError("candidate QCES evaluator/source provenance differs from base")
    if _audio_tree_signature(base) != _audio_tree_signature(candidate):
        raise SelectionInputError("candidate AudioSep source provenance differs from base")


def _retained_candidates(summary: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    retention = require_mapping(
        summary.get("epoch_checkpoint_retention"), "epoch_checkpoint_retention"
    )
    if (
        retention.get("enabled") is not True
        or retention.get("immutable") is not True
        or retention.get("policy") != "save_every_epoch"
    ):
        raise SelectionInputError(
            "run summary does not certify immutable --save-every-epoch retention"
        )
    rows = require_list(
        summary.get("retained_epoch_checkpoints"), "retained_epoch_checkpoints"
    )
    if not rows:
        raise SelectionInputError("run summary retained no epoch checkpoints")
    result: Dict[int, Mapping[str, Any]] = {}
    for index, row_value in enumerate(rows):
        row = require_mapping(row_value, f"retained_epoch_checkpoints[{index}]")
        epoch = exact_int(row.get("epoch"), f"retained epoch {index}", 1)
        if epoch in result:
            raise SelectionInputError(f"duplicate retained checkpoint epoch: {epoch}")
        checkpoint = verify_declared_file_identity(
            row.get("checkpoint"), f"retained epoch {epoch} checkpoint"
        )
        metrics = require_mapping(
            row.get("validation_metrics_at_checkpoint"),
            f"retained epoch {epoch} validation metrics",
        )
        # Required even before a matching report is supplied: a missing value in
        # the immutable run ledger invalidates model selection provenance.
        finite_float(metrics.get("answerable_temporal_iou"), TEMPORAL_IOU_KEY)
        finite_float(metrics.get("evidence_sd_sdr"), SD_SDR_MEAN_KEY)
        finite_float(metrics.get("no_evidence_retained_ratio"), NE_RETAINED_KEY)
        finite_float(metrics.get("weakest_role_waveform"), WEAKEST_ERROR_KEY)
        _validate_retained_checkpoint_metadata(
            Path(require_mapping(row.get("checkpoint"), "checkpoint")["path"]),
            epoch,
            exact_int(row.get("global_step"), f"retained epoch {epoch} global_step"),
            metrics,
        )
        normalized = dict(row)
        normalized["checkpoint"] = checkpoint
        result[epoch] = normalized
    return result


def _validate_retained_checkpoint_metadata(
    path: Path,
    epoch: int,
    global_step: int,
    summary_metrics: Mapping[str, Any],
) -> None:
    """Cross-check the immutable summary against metadata inside checkpoint bytes."""

    try:
        import torch

        payload = torch.load(path.resolve(), map_location="cpu", weights_only=True)
    except Exception as error:  # torch uses several format-specific exceptions
        raise SelectionInputError(
            f"cannot safely load retained epoch {epoch} checkpoint metadata: {error}"
        ) from error
    checkpoint = require_mapping(payload, f"epoch {epoch} checkpoint payload")
    extra = require_mapping(checkpoint.get("extra"), f"epoch {epoch} checkpoint extra")
    if extra.get("checkpoint_role") != "retained_epoch":
        raise SelectionInputError(
            f"epoch {epoch} checkpoint bytes do not declare role=retained_epoch"
        )
    if exact_int(extra.get("checkpoint_epoch"), "checkpoint extra epoch", 1) != epoch:
        raise SelectionInputError(f"epoch {epoch} checkpoint-byte epoch mismatch")
    if exact_int(extra.get("checkpoint_global_step"), "checkpoint extra global_step") != global_step:
        raise SelectionInputError(f"epoch {epoch} checkpoint-byte global-step mismatch")
    retention = require_mapping(
        extra.get("epoch_retention"), f"epoch {epoch} checkpoint epoch_retention"
    )
    if (
        retention.get("policy") != "save_every_epoch"
        or retention.get("immutable") is not True
        or exact_int(retention.get("epoch"), "checkpoint retention epoch", 1) != epoch
        or exact_int(retention.get("global_step"), "checkpoint retention global_step")
        != global_step
    ):
        raise SelectionInputError(
            f"epoch {epoch} checkpoint bytes lack immutable retention provenance"
        )
    checkpoint_metrics = require_mapping(
        retention.get("validation_metrics_at_checkpoint"),
        f"epoch {epoch} checkpoint validation metrics",
    )
    for key in (
        "answerable_temporal_iou",
        "evidence_sd_sdr",
        "no_evidence_retained_ratio",
        "weakest_role_waveform",
    ):
        require_close(
            finite_float(summary_metrics.get(key), f"summary epoch {epoch} {key}"),
            finite_float(checkpoint_metrics.get(key), f"checkpoint epoch {epoch} {key}"),
            f"epoch {epoch} summary/checkpoint {key}",
        )


def _candidate_epoch(report: Mapping[str, Any], label: str) -> int:
    metadata = require_mapping(
        require_mapping(report.get("provenance"), f"{label}.provenance").get(
            "checkpoint_metadata"
        ),
        f"{label}.checkpoint_metadata",
    )
    return exact_int(metadata.get("checkpoint_epoch"), f"{label}.checkpoint_epoch", 1)


def discover_reports(
    explicit: Sequence[str], reports_dir: Optional[Path]
) -> Dict[int, Path]:
    paths: Dict[int, Path] = {}
    for specification in explicit:
        if "=" not in specification:
            raise SelectionInputError(
                f"--epoch-report must be EPOCH=PATH, got: {specification}"
            )
        raw_epoch, raw_path = specification.split("=", 1)
        try:
            epoch = int(raw_epoch)
        except ValueError as error:
            raise SelectionInputError(f"invalid epoch in {specification}") from error
        if epoch < 1 or epoch in paths:
            raise SelectionInputError(f"duplicate/invalid explicit epoch: {epoch}")
        paths[epoch] = Path(raw_path)
    if reports_dir is not None:
        resolved = reports_dir.resolve()
        if not resolved.is_dir():
            raise SelectionInputError(f"reports directory does not exist: {resolved}")
        for path in sorted(resolved.rglob("factorization_report.json")):
            report, _ = load_json(path, "candidate factorization report")
            epoch = _candidate_epoch(report, str(path))
            if epoch in paths:
                raise SelectionInputError(
                    f"multiple candidate reports declare epoch {epoch}: "
                    f"{paths[epoch]} and {path}"
                )
            paths[epoch] = path
    return dict(sorted(paths.items()))


def _validate_candidate_binding(
    report: Mapping[str, Any],
    report_epoch: int,
    retained: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, float]]:
    provenance = require_mapping(report.get("provenance"), "candidate provenance")
    report_checkpoint = verify_declared_file_identity(
        require_mapping(provenance.get("inputs"), "candidate provenance.inputs").get(
            "learned_qces_checkpoint"
        ),
        f"epoch {report_epoch} factorization checkpoint",
    )
    retained_checkpoint = require_mapping(retained.get("checkpoint"), "retained checkpoint")
    if not identities_match(report_checkpoint, retained_checkpoint):
        raise SelectionInputError(
            f"epoch {report_epoch} report checkpoint hash differs from run summary"
        )
    metadata = require_mapping(
        provenance.get("checkpoint_metadata"), "candidate checkpoint_metadata"
    )
    if metadata.get("checkpoint_role") != "retained_epoch":
        raise SelectionInputError(
            f"epoch {report_epoch} is not an immutable retained_epoch checkpoint"
        )
    if exact_int(metadata.get("checkpoint_epoch"), "checkpoint_epoch", 1) != report_epoch:
        raise SelectionInputError(f"epoch {report_epoch} metadata epoch mismatch")
    global_step = exact_int(metadata.get("checkpoint_global_step"), "checkpoint_global_step")
    if global_step != exact_int(retained.get("global_step"), "summary global_step"):
        raise SelectionInputError(f"epoch {report_epoch} global-step mismatch")
    epoch_retention = require_mapping(metadata.get("epoch_retention"), "epoch_retention")
    if (
        epoch_retention.get("policy") != "save_every_epoch"
        or epoch_retention.get("immutable") is not True
        or exact_int(epoch_retention.get("epoch"), "epoch_retention.epoch", 1)
        != report_epoch
    ):
        raise SelectionInputError(
            f"epoch {report_epoch} checkpoint lacks immutable retention provenance"
        )
    validation_metrics = require_mapping(
        retained.get("validation_metrics_at_checkpoint"), "summary validation metrics"
    )
    checkpoint_validation = require_mapping(
        epoch_retention.get("validation_metrics_at_checkpoint"),
        "checkpoint validation metrics",
    )
    extracted: Dict[str, float] = {}
    for summary_key, receipt_key in (
        ("answerable_temporal_iou", TEMPORAL_IOU_KEY),
        ("evidence_sd_sdr", SD_SDR_MEAN_KEY),
        ("no_evidence_retained_ratio", NE_RETAINED_KEY),
        ("weakest_role_waveform", WEAKEST_ERROR_KEY),
    ):
        value = finite_float(validation_metrics.get(summary_key), receipt_key)
        checkpoint_value = finite_float(
            checkpoint_validation.get(summary_key), f"checkpoint {receipt_key}"
        )
        require_close(value, checkpoint_value, f"epoch {report_epoch} {receipt_key}")
        extracted[receipt_key] = value
    return report_checkpoint, extracted


def _retained_trainer_metrics(retained: Mapping[str, Any]) -> Dict[str, float]:
    validation_metrics = require_mapping(
        retained.get("validation_metrics_at_checkpoint"),
        "retained validation_metrics_at_checkpoint",
    )
    return {
        TEMPORAL_IOU_KEY: finite_float(
            validation_metrics.get("answerable_temporal_iou"), TEMPORAL_IOU_KEY
        ),
        SD_SDR_MEAN_KEY: finite_float(
            validation_metrics.get("evidence_sd_sdr"), SD_SDR_MEAN_KEY
        ),
        NE_RETAINED_KEY: finite_float(
            validation_metrics.get("no_evidence_retained_ratio"), NE_RETAINED_KEY
        ),
        WEAKEST_ERROR_KEY: finite_float(
            validation_metrics.get("weakest_role_waveform"), WEAKEST_ERROR_KEY
        ),
    }


def _constraint_evaluation(
    metrics: Mapping[str, float],
    base: Mapping[str, float],
    thresholds: Thresholds,
) -> tuple[Dict[str, Any], list[str]]:
    checks = {
        NE_RETAINED_KEY: {
            "candidate": metrics[NE_RETAINED_KEY],
            "limit": base[NE_RETAINED_KEY] + thresholds.no_evidence_tolerance,
            "pass": metrics[NE_RETAINED_KEY]
            <= base[NE_RETAINED_KEY] + thresholds.no_evidence_tolerance,
        },
        FALSE_SILENCE_KEY: {
            "candidate": metrics[FALSE_SILENCE_KEY],
            "limit": base[FALSE_SILENCE_KEY] + thresholds.false_silence_tolerance,
            "pass": metrics[FALSE_SILENCE_KEY]
            <= base[FALSE_SILENCE_KEY] + thresholds.false_silence_tolerance,
        },
        SI_SDR_FAILURE_KEY: {
            "candidate": metrics[SI_SDR_FAILURE_KEY],
            "limit": base[SI_SDR_FAILURE_KEY] + thresholds.failure_rate_tolerance,
            "pass": metrics[SI_SDR_FAILURE_KEY]
            <= base[SI_SDR_FAILURE_KEY] + thresholds.failure_rate_tolerance,
        },
        SD_SDR_P10_KEY: {
            "candidate": metrics[SD_SDR_P10_KEY],
            "limit": base[SD_SDR_P10_KEY] - thresholds.sd_sdr_tail_tolerance_db,
            "pass": metrics[SD_SDR_P10_KEY]
            >= base[SD_SDR_P10_KEY] - thresholds.sd_sdr_tail_tolerance_db,
        },
        SD_SDR_MIN_KEY: {
            "candidate": metrics[SD_SDR_MIN_KEY],
            "limit": base[SD_SDR_MIN_KEY] - thresholds.sd_sdr_tail_tolerance_db,
            "pass": metrics[SD_SDR_MIN_KEY]
            >= base[SD_SDR_MIN_KEY] - thresholds.sd_sdr_tail_tolerance_db,
        },
    }
    reasons = [
        f"{key} violates frozen limit: {value['candidate']} vs {value['limit']}"
        for key, value in checks.items()
        if not value["pass"]
    ]
    return checks, reasons


def _tie_break(
    feasible: Sequence[Mapping[str, Any]], tie_tolerance: float
) -> Mapping[str, Any]:
    pool = list(feasible)
    best_iou = max(row["metrics"][TEMPORAL_IOU_KEY] for row in pool)
    pool = [
        row
        for row in pool
        if row["metrics"][TEMPORAL_IOU_KEY] >= best_iou - tie_tolerance
    ]
    best_sd_sdr = max(row["metrics"][SD_SDR_MEAN_KEY] for row in pool)
    pool = [
        row
        for row in pool
        if row["metrics"][SD_SDR_MEAN_KEY] >= best_sd_sdr - tie_tolerance
    ]
    best_weakest_error = min(row["metrics"][WEAKEST_ERROR_KEY] for row in pool)
    pool = [
        row
        for row in pool
        if row["metrics"][WEAKEST_ERROR_KEY]
        <= best_weakest_error + tie_tolerance
    ]
    # An exact final tie is resolved toward the earlier epoch.  This is merely
    # deterministic; it is not an additional performance claim.
    return min(pool, key=lambda row: row["epoch"])


def build_selection_receipt(
    base_report_path: Path,
    run_summary_path: Path,
    epoch_reports: Mapping[int, Path],
    thresholds: Thresholds = Thresholds(),
    *,
    allow_partial_evaluation: bool = False,
) -> Dict[str, Any]:
    thresholds.validate()
    base_report, base_identity = load_json(base_report_path, "base factorization report")
    summary, summary_identity = load_json(run_summary_path, "run summary")
    _validate_factorization_protocol(base_report, "base report")
    base_inputs = require_mapping(
        require_mapping(base_report.get("provenance"), "base provenance").get("inputs"),
        "base provenance.inputs",
    )
    base_checkpoint = verify_declared_file_identity(
        base_inputs.get("learned_qces_checkpoint"), "base checkpoint"
    )
    base_manifest = verify_declared_file_identity(
        base_inputs.get("manifest"), "base validation manifest"
    )
    base_metrics, base_coverage, base_metric_sources = _extract_report_metrics(
        base_report
    )
    retained = _retained_candidates(summary)
    summary_validation = require_mapping(
        require_mapping(summary.get("manifests"), "summary.manifests").get(
            "validation"
        ),
        "summary.manifests.validation",
    )
    if not identities_match(summary_validation, base_manifest):
        raise SelectionInputError("run-summary validation manifest differs from base report")

    provided_epochs = sorted(epoch_reports)
    unknown_epochs = sorted(set(provided_epochs) - set(retained))
    if unknown_epochs:
        raise SelectionInputError(
            f"reports are not retained epochs from this run: {unknown_epochs}"
        )
    gate1_limit = base_metrics[NE_RETAINED_KEY] + thresholds.no_evidence_tolerance
    retained_trainer_metrics = {
        epoch: _retained_trainer_metrics(row) for epoch, row in retained.items()
    }
    pre_rejected_gate1_epochs = sorted(
        epoch
        for epoch, metrics in retained_trainer_metrics.items()
        if metrics[NE_RETAINED_KEY] > gate1_limit
    )
    factorization_required_epochs = sorted(
        set(retained) - set(pre_rejected_gate1_epochs)
    )
    missing_factorization_epochs = sorted(set(retained) - set(provided_epochs))
    missing_required_epochs = sorted(
        set(factorization_required_epochs) - set(provided_epochs)
    )
    if missing_required_epochs and not allow_partial_evaluation:
        raise SelectionInputError(
            "missing factorization reports for epochs that passed gate 1: "
            f"{missing_required_epochs}; pass --allow-partial-evaluation only "
            "for a non-selecting diagnostic receipt"
        )

    candidates: list[Dict[str, Any]] = []
    invalid_candidates = False
    for declared_epoch in sorted(retained):
        trainer_summary_metrics = retained_trainer_metrics[declared_epoch]
        report_path = epoch_reports.get(declared_epoch)
        if report_path is None and declared_epoch in pre_rejected_gate1_epochs:
            candidates.append(
                {
                    "epoch": declared_epoch,
                    "evaluation_stage": "pre_rejected_gate1",
                    "report": None,
                    "checkpoint": retained[declared_epoch]["checkpoint"],
                    "input_valid": True,
                    "feasible": False,
                    "metrics": {
                        NE_RETAINED_KEY: trainer_summary_metrics[NE_RETAINED_KEY]
                    },
                    "trainer_diagnostics_not_used_after_gate1_rejection": {
                        TEMPORAL_IOU_KEY: trainer_summary_metrics[TEMPORAL_IOU_KEY],
                        SD_SDR_MEAN_KEY: trainer_summary_metrics[SD_SDR_MEAN_KEY],
                        WEAKEST_ERROR_KEY: trainer_summary_metrics[WEAKEST_ERROR_KEY],
                    },
                    "metric_sources": {
                        NE_RETAINED_KEY: (
                            "immutable_run_summary_cross_checked_with_checkpoint_bytes"
                        )
                    },
                    "constraints": {
                        NE_RETAINED_KEY: {
                            "candidate": trainer_summary_metrics[NE_RETAINED_KEY],
                            "limit": gate1_limit,
                            "pass": False,
                        }
                    },
                    "rejected_reasons": [
                        f"{NE_RETAINED_KEY} violates frozen gate-1 limit: "
                        f"{trainer_summary_metrics[NE_RETAINED_KEY]} vs {gate1_limit}"
                    ],
                }
            )
            continue
        if report_path is None:
            candidates.append(
                {
                    "epoch": declared_epoch,
                    "evaluation_stage": "missing_required_factorization",
                    "report": None,
                    "checkpoint": retained[declared_epoch]["checkpoint"],
                    "input_valid": True,
                    "feasible": None,
                    "metrics": {
                        NE_RETAINED_KEY: trainer_summary_metrics[NE_RETAINED_KEY]
                    },
                    "metric_sources": {
                        NE_RETAINED_KEY: (
                            "immutable_run_summary_cross_checked_with_checkpoint_bytes"
                        )
                    },
                    "constraints": {
                        NE_RETAINED_KEY: {
                            "candidate": trainer_summary_metrics[NE_RETAINED_KEY],
                            "limit": gate1_limit,
                            "pass": True,
                        }
                    },
                    "rejected_reasons": [
                        "factorization report required after passing gate 1; "
                        "checkpoint remains unevaluated and cannot be selected"
                    ],
                }
            )
            continue
        row: Dict[str, Any] = {
            "epoch": declared_epoch,
            "evaluation_stage": "full_factorization",
            "report": file_identity(report_path),
            "input_valid": False,
            "feasible": False,
            "metrics": None,
            "constraints": None,
            "rejected_reasons": [],
        }
        try:
            report, report_identity = load_json(
                report_path, f"epoch {declared_epoch} factorization report"
            )
            row["report"] = report_identity
            _validate_factorization_protocol(report, f"epoch {declared_epoch} report")
            actual_epoch = _candidate_epoch(report, f"epoch {declared_epoch} report")
            if actual_epoch != declared_epoch:
                raise SelectionInputError(
                    f"mapping declares epoch {declared_epoch}, report declares {actual_epoch}"
                )
            _validate_report_comparability(base_report, report)
            checkpoint, trainer_metrics = _validate_candidate_binding(
                report, declared_epoch, retained[declared_epoch]
            )
            report_metrics, _, metric_sources = _extract_report_metrics(
                report, base_coverage
            )
            for receipt_key, trainer_key in (
                (NE_RETAINED_KEY, NE_RETAINED_KEY),
                (SD_SDR_MEAN_KEY, SD_SDR_MEAN_KEY),
            ):
                require_close(
                    report_metrics[receipt_key],
                    trainer_metrics[trainer_key],
                    f"epoch {declared_epoch} factorization/trainer {receipt_key}",
                )
            metrics = {
                **report_metrics,
                TEMPORAL_IOU_KEY: trainer_metrics[TEMPORAL_IOU_KEY],
                WEAKEST_ERROR_KEY: trainer_metrics[WEAKEST_ERROR_KEY],
            }
            constraints, reasons = _constraint_evaluation(
                metrics, base_metrics, thresholds
            )
            row.update(
                {
                    "checkpoint": checkpoint,
                    "input_valid": True,
                    "feasible": not reasons,
                    "metrics": metrics,
                    "metric_sources": {
                        **metric_sources,
                        TEMPORAL_IOU_KEY: "immutable_run_summary_cross_checked_with_checkpoint_metadata",
                        WEAKEST_ERROR_KEY: "immutable_run_summary_cross_checked_with_checkpoint_metadata",
                    },
                    "constraints": constraints,
                    "rejected_reasons": reasons,
                }
            )
        except SelectionInputError as error:
            invalid_candidates = True
            row["rejected_reasons"] = [f"invalid input/provenance: {error}"]
        candidates.append(row)

    feasible = [
        row
        for row in candidates
        if row["input_valid"] and row["feasible"] is True
    ]
    coverage_complete = not missing_required_epochs
    selected: Optional[Mapping[str, Any]] = None
    if invalid_candidates:
        status = "invalid_candidate_input"
    elif not feasible:
        status = (
            "no_feasible_checkpoint"
            if coverage_complete
            else "no_feasible_evaluated_checkpoint"
        )
    elif not coverage_complete:
        status = "incomplete_candidate_coverage"
    else:
        selected = _tie_break(feasible, thresholds.tie_tolerance)
        status = "selected"

    decision_authoritative = status in {"selected", "no_feasible_checkpoint"}
    if status == "selected":
        decision = {
            "action": "promote_refiner_checkpoint",
            "reason": "one retained epoch passed all frozen constraints",
            "checkpoint": selected["checkpoint"] if selected is not None else None,
        }
    elif status == "no_feasible_checkpoint":
        decision = {
            "action": "retain_base_no_refiner",
            "reason": "every retained epoch was either gate-1 rejected or failed full constraints",
            "checkpoint": base_checkpoint,
        }
    else:
        decision = {
            "action": "inconclusive_no_checkpoint_promotion",
            "reason": "invalid input or missing factorization for a gate-1-passing epoch",
            "checkpoint": None,
        }

    receipt = {
        "format": RECEIPT_FORMAT,
        "status": status,
        "selection_authoritative": status == "selected",
        "decision_authoritative": decision_authoritative,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "policy": {
            "constraints": [
                f"{NE_RETAINED_KEY} <= base + no_evidence_tolerance",
                f"{FALSE_SILENCE_KEY} <= base + false_silence_tolerance",
                f"{SI_SDR_FAILURE_KEY} <= base + failure_rate_tolerance",
                f"{SD_SDR_P10_KEY} >= base - sd_sdr_tail_tolerance_db",
                f"{SD_SDR_MIN_KEY} >= base - sd_sdr_tail_tolerance_db",
            ],
            "gate_order": (
                "Gate 1 uses no-evidence retention from the immutable run summary "
                "cross-checked against checkpoint bytes. Factorization reports "
                "are required only for epochs that pass gate 1."
            ),
            "objective_order": [
                f"maximize {TEMPORAL_IOU_KEY}",
                f"maximize {SD_SDR_MEAN_KEY}",
                f"minimize {WEAKEST_ERROR_KEY}",
                "earlier epoch only for an exact remaining tie",
            ],
            "no_primary_or_independent_extrema_checkpoint_is_implicitly_eligible": True,
            "sd_sdr_p10_definition": "sorted linear interpolation at (n - 1) * 0.1",
            "thresholds": {
                "no_evidence_tolerance_absolute_↓": thresholds.no_evidence_tolerance,
                "false_silence_tolerance_absolute_↓": thresholds.false_silence_tolerance,
                "below_minus20_db_failure_tolerance_absolute_↓": thresholds.failure_rate_tolerance,
                "sd_sdr_tail_tolerance_db_↓": thresholds.sd_sdr_tail_tolerance_db,
                "tie_tolerance_absolute_↓": thresholds.tie_tolerance,
            },
        },
        "inputs": {
            "base_factorization_report": base_identity,
            "base_checkpoint": base_checkpoint,
            "run_summary": summary_identity,
            "validation_manifest": base_manifest,
        },
        "software": {
            "selector": file_identity(Path(__file__)),
        },
        "base_metrics": base_metrics,
        "base_metric_sources": base_metric_sources,
        "candidate_coverage": {
            "retained_epoch_count": len(retained),
            "evaluated_epoch_count": len(epoch_reports),
            "coverage_complete": coverage_complete,
            "evaluated_epochs": provided_epochs,
            "factorization_required_epochs": factorization_required_epochs,
            "pre_rejected_gate1_epochs": pre_rejected_gate1_epochs,
            "missing_factorization_epochs": missing_factorization_epochs,
            "missing_required_factorization_epochs": missing_required_epochs,
            "partial_evaluation_explicitly_allowed": allow_partial_evaluation,
            "missing_required_factorization_can_never_select": True,
            "gate1_pre_rejection_without_factorization_is_complete": True,
        },
        "candidates": candidates,
        "decision": decision,
        "selected": (
            {
                "epoch": selected["epoch"],
                "checkpoint": selected["checkpoint"],
                "factorization_report": selected["report"],
                "metrics": selected["metrics"],
            }
            if selected is not None
            else None
        ),
        "failure_policy": (
            "Any missing/non-finite metric, provenance mismatch, checkpoint/hash "
            "mismatch, item coverage mismatch, invalid candidate, or missing "
            "factorization for an epoch that passes gate 1 yields no selected "
            "checkpoint. A checkpoint-byte-verified gate-1 rejection needs no "
            "factorization report."
        ),
        "exit_code_semantics": {
            "0": "authoritative checkpoint selected",
            "2": "invalid input/tooling failure; invalid receipt emitted when possible",
            "3": "valid scientific receipt but no authoritative selection",
        },
    }
    return receipt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select only an immutable retained QCES epoch that passes the frozen "
            "joint acoustic-safety rule. Metric arrows are included in the receipt."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Exit codes: 0 = selected; 2 = invalid input/tool error; "
            "3 = valid scientific receipt with no authoritative selection "
            "(for example no feasible epoch or partial coverage)."
        ),
    )
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--run-summary", type=Path, required=True)
    parser.add_argument(
        "--epoch-report",
        action="append",
        default=[],
        metavar="EPOCH=PATH",
        help="factorization report for one immutable retained epoch; repeatable",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="recursively discover factorization_report.json files",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-evidence-tolerance", type=float, default=0.01)
    parser.add_argument("--false-silence-tolerance", type=float, default=0.0)
    parser.add_argument("--failure-rate-tolerance", type=float, default=0.0)
    parser.add_argument("--sd-sdr-tail-tolerance-db", type=float, default=0.0)
    parser.add_argument("--tie-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--allow-partial-evaluation",
        action="store_true",
        help=(
            "emit a diagnostic receipt for a subset of retained epochs; partial "
            "coverage can never produce a selected checkpoint"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _write_receipt(path: Path, receipt: Mapping[str, Any], overwrite: bool) -> Dict[str, Any]:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not overwrite:
        raise SelectionInputError(
            f"output already exists (pass --overwrite explicitly): {resolved}"
        )
    temporary = resolved.parent / f".{resolved.name}.{os.getpid()}.tmp"
    if temporary.exists():
        raise SelectionInputError(f"refusing to overwrite temporary file: {temporary}")
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_identity(resolved)


def _best_effort_identity(path: Path) -> Dict[str, Any]:
    try:
        return file_identity(path)
    except SelectionInputError as error:
        resolved = path.resolve()
        try:
            portable_path = resolved.relative_to(PROJECT_ROOT).as_posix()
            path_base = "project_root"
        except ValueError:
            portable_path = str(resolved)
            path_base = "absolute"
        return {
            "path": portable_path,
            "path_base": path_base,
            "unavailable_reason": str(error),
        }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        reports = discover_reports(args.epoch_report, args.reports_dir)
        receipt = build_selection_receipt(
            args.base_report,
            args.run_summary,
            reports,
            Thresholds(
                no_evidence_tolerance=args.no_evidence_tolerance,
                false_silence_tolerance=args.false_silence_tolerance,
                failure_rate_tolerance=args.failure_rate_tolerance,
                sd_sdr_tail_tolerance_db=args.sd_sdr_tail_tolerance_db,
                tie_tolerance=args.tie_tolerance,
            ),
            allow_partial_evaluation=args.allow_partial_evaluation,
        )
        output_identity = _write_receipt(args.output, receipt, args.overwrite)
    except SelectionInputError as error:
        invalid_receipt = {
            "format": RECEIPT_FORMAT,
            "status": "invalid_input",
            "selection_authoritative": False,
            "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
            "inputs": {
                "base_factorization_report": _best_effort_identity(args.base_report),
                "run_summary": _best_effort_identity(args.run_summary),
                "candidate_reports": [
                    _best_effort_identity(path)
                    for path in (
                        reports.values() if "reports" in locals() else []
                    )
                ],
            },
            "software": {"selector": file_identity(Path(__file__))},
            "rejected_reasons": [str(error)],
            "selected": None,
            "failure_policy": "Invalid or incomplete input can never select a checkpoint.",
            "exit_code_semantics": {
                "0": "authoritative checkpoint selected",
                "2": "invalid input/tooling failure; invalid receipt emitted when possible",
                "3": "valid scientific receipt but no authoritative selection",
            },
        }
        try:
            output_identity = _write_receipt(
                args.output, invalid_receipt, args.overwrite
            )
        except SelectionInputError as write_error:
            print(
                json.dumps(
                    {
                        "status": "invalid_input",
                        "error": str(error),
                        "receipt_write_error": str(write_error),
                    }
                ),
                file=sys.stderr,
            )
            return 2
        print(
            json.dumps(
                {
                    "status": "invalid_input",
                    "error": str(error),
                    "receipt": output_identity,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "selection_authoritative": receipt["selection_authoritative"],
                "selected": receipt["selected"],
                "receipt": output_identity,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if receipt["status"] == "selected" else 3


if __name__ == "__main__":
    raise SystemExit(main())
