#!/usr/bin/env python3
"""Validate QA removal v2 records, paired scenes, audio, and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import scipy
import soundfile as sf


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.data.qa_schema import (  # noqa: E402
    EDIT_RATIONALES,
    NON_OVERLAP_SNR_MEASUREMENT,
    ORDINAL_CANDIDATE_ORDER_RULE,
    ORDINAL_SEMANTIC_ORDER_RULE,
    OVERLAP_SNR_MEASUREMENT,
    QUESTION_TYPES,
    RELATIVE_SEMANTIC_ORDER_RULE,
    SCHEMA_VERSION,
    EventAnnotation,
    QARecord,
    parse_record,
)


EPSILON = 1e-12
RECONSTRUCTION_TOLERANCE = 1.1e-4
SILENCE_TOLERANCE = 1.1e-4
SNR_TOLERANCE_DB = 0.15


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Manifest row {line_number} must be an object")
            rows.append(row)
    return rows


def read_audio(path: Path) -> Tuple[np.ndarray, int, str, int, int, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing audio file: {path}")
    info = sf.info(path)
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.shape[1] != 1:
        raise AssertionError(f"Expected mono audio: {path}")
    waveform = np.asarray(waveform[:, 0], dtype=np.float32)
    if not np.all(np.isfinite(waveform)):
        raise AssertionError(f"Non-finite samples: {path}")
    return (
        waveform,
        int(sample_rate),
        str(info.subtype),
        int(info.channels),
        int(info.frames),
        float(info.duration),
    )


def rms(waveform: np.ndarray) -> float:
    if waveform.size == 0:
        raise AssertionError("Cannot measure RMS of an empty waveform")
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def outside_interval_peak(waveform: np.ndarray, start: int, end: int) -> float:
    left_peak = float(np.max(np.abs(waveform[:start]))) if start else 0.0
    right_peak = float(np.max(np.abs(waveform[end:]))) if end < waveform.size else 0.0
    return max(left_peak, right_peak)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under(root: Path, relative_path: str, context: str) -> Path:
    """Resolve an existing path and reject traversal or symlink escape."""

    candidate = (root / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AssertionError(f"{context} escapes project root: {relative_path}") from exc
    return candidate


def validate_config_identity(
    config: Mapping[str, Any], project_root: Path
) -> Tuple[Dict[str, str], Dict[str, Any]]:
    """Verify the recorded runtime and source-code build identities."""

    expected_runtime_identity = {
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "scipy_version": str(scipy.__version__),
        "soundfile_version": str(sf.__version__),
    }
    if config.get("runtime_identity") != expected_runtime_identity:
        raise AssertionError(
            "Config runtime identity does not match the validating environment"
        )

    source_paths = {
        "builder": (
            CODE_ROOT
            / "mixi_understanding"
            / "scripts"
            / "build_qa_removal_dataset.py"
        ),
        "schema": CODE_ROOT / "mixi_understanding" / "data" / "qa_schema.py",
        "validator": Path(__file__).resolve(),
    }
    expected_build_identity = {
        "source_files": {
            role: {
                "path": path.resolve().relative_to(project_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for role, path in sorted(source_paths.items())
        }
    }
    if config.get("build_identity") != expected_build_identity:
        raise AssertionError(
            "Config build identity does not match builder/schema/validator sources"
        )
    return expected_runtime_identity, expected_build_identity


def intervals_equal(
    left: Sequence[Tuple[float, float]], right: Sequence[Tuple[float, float]]
) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a[0], b[0], abs_tol=1e-6)
        and math.isclose(a[1], b[1], abs_tol=1e-6)
        for a, b in zip(left, right)
    )


def event_identity(event: EventAnnotation) -> Tuple[Any, ...]:
    """Return immutable physical identity, intentionally excluding QA role."""

    return (
        event.event_id,
        event.label,
        event.source_dataset,
        event.source_id,
        event.source_path,
        event.source_sha256,
        event.source_interval_seconds,
        event.source_crop_interval_seconds,
        event.onset_seconds,
        event.offset_seconds,
    )


def semantic_events(record: QARecord) -> Tuple[EventAnnotation, EventAnnotation]:
    events = sorted(
        (event for event in record.events if event.role != "interference"),
        key=lambda event: (event.onset_seconds, event.event_id),
    )
    if len(events) != 2:
        raise AssertionError(f"{record.sample_id} must have two semantic events")
    return events[0], events[1]


def validate_audio_record(
    root: Path,
    record: QARecord,
    expected_rate: int,
    expected_channels: int,
    expected_samples: int,
    expected_duration: float,
    expected_subtype: str,
    expected_snr_by_rationale: Mapping[str, float],
) -> Dict[str, float]:
    if record.sample_rate != expected_rate:
        raise AssertionError(
            f"{record.sample_id} record sample_rate: "
            f"{record.sample_rate} != {expected_rate}"
        )
    if record.num_channels != expected_channels:
        raise AssertionError(
            f"{record.sample_id} record num_channels: "
            f"{record.num_channels} != {expected_channels}"
        )
    if record.num_samples != expected_samples:
        raise AssertionError(
            f"{record.sample_id} record num_samples: "
            f"{record.num_samples} != {expected_samples}"
        )
    if not math.isclose(record.duration_seconds, expected_duration, abs_tol=1e-9):
        raise AssertionError(
            f"{record.sample_id} record duration: "
            f"{record.duration_seconds} != {expected_duration}"
        )
    expected_requested_snr = float(expected_snr_by_rationale[record.edit_rationale])
    if not math.isclose(record.snr_db_requested, expected_requested_snr, abs_tol=1e-9):
        raise AssertionError(
            f"{record.sample_id} requested SNR does not match config rationale"
        )

    arrays: Dict[str, np.ndarray] = {}
    for field_name, relative_path in (
        ("mixture", record.mixture_path),
        ("clean", record.clean_path),
        ("interference", record.interference_stem_path),
    ):
        audio_path = resolve_under(root, relative_path, f"{record.sample_id} audio")
        waveform, sample_rate, subtype, channels, frames, duration = read_audio(
            audio_path
        )
        if sample_rate != record.sample_rate or sample_rate != expected_rate:
            raise AssertionError(
                f"{record.sample_id} {field_name} sample rate mismatch: {sample_rate}"
            )
        if frames != record.num_samples or waveform.size != expected_samples:
            raise AssertionError(
                f"{record.sample_id} {field_name} frame count mismatch: {frames}"
            )
        if channels != record.num_channels or channels != expected_channels:
            raise AssertionError(
                f"{record.sample_id} {field_name} channel mismatch: {channels}"
            )
        if subtype != expected_subtype:
            raise AssertionError(
                f"{record.sample_id} {field_name} subtype: {subtype} != {expected_subtype}"
            )
        if not math.isclose(duration, record.duration_seconds, abs_tol=1e-9):
            raise AssertionError(
                f"{record.sample_id} {field_name} WAV duration mismatch: {duration}"
            )
        arrays[field_name] = waveform

    reconstruction_error = float(
        np.max(np.abs(arrays["mixture"] - arrays["clean"] - arrays["interference"]))
    )
    if reconstruction_error > RECONSTRUCTION_TOLERANCE:
        raise AssertionError(
            f"{record.sample_id} reconstruction error: {reconstruction_error}"
        )
    peak = max(float(np.max(np.abs(waveform))) for waveform in arrays.values())
    if peak > 1.0:
        raise AssertionError(f"{record.sample_id} has samples outside [-1, 1]")
    stored_peak_error = abs(
        float(np.max(np.abs(arrays["mixture"]))) - record.mixture_peak
    )
    if stored_peak_error > RECONSTRUCTION_TOLERANCE:
        raise AssertionError(
            f"{record.sample_id} stored mixture_peak mismatch: {stored_peak_error}"
        )

    interference_interval = record.interference_intervals[0]
    interference_start = int(round(interference_interval[0] * expected_rate))
    interference_end = int(round(interference_interval[1] * expected_rate))
    active_interference = arrays["interference"][interference_start:interference_end]
    outside_peak = outside_interval_peak(
        arrays["interference"], interference_start, interference_end
    )
    if outside_peak > SILENCE_TOLERANCE:
        raise AssertionError(
            f"{record.sample_id} interference is active outside its annotation: "
            f"{outside_peak}"
        )
    if rms(active_interference) < 1e-5:
        raise AssertionError(f"{record.sample_id} interference interval is silent")

    preserve_mask = np.zeros(expected_samples, dtype=bool)
    for interval in record.anchor_intervals + record.answer_intervals:
        start = int(round(interval[0] * expected_rate))
        end = int(round(interval[1] * expected_rate))
        preserve_mask[start:end] = True
        if rms(arrays["clean"][start:end]) < 1e-5:
            raise AssertionError(
                f"{record.sample_id} clean evidence interval {interval} is silent"
            )
    clean_outside_peak = (
        float(np.max(np.abs(arrays["clean"][~preserve_mask])))
        if np.any(~preserve_mask)
        else 0.0
    )
    if clean_outside_peak > SILENCE_TOLERANCE:
        raise AssertionError(
            f"{record.sample_id} clean audio is active outside evidence: "
            f"{clean_outside_peak}"
        )

    interference_mask = np.zeros(expected_samples, dtype=bool)
    interference_mask[interference_start:interference_end] = True
    intersection_mask = preserve_mask & interference_mask
    if record.snr_measurement == OVERLAP_SNR_MEASUREMENT:
        if not np.any(intersection_mask):
            raise AssertionError(f"{record.sample_id} overlap SNR has empty intersection")
        clean_reference = arrays["clean"][intersection_mask]
        interference_reference = arrays["interference"][intersection_mask]
    elif record.snr_measurement == NON_OVERLAP_SNR_MEASUREMENT:
        if np.any(intersection_mask):
            raise AssertionError(f"{record.sample_id} non-overlap masks intersect")
        clean_reference = arrays["clean"][preserve_mask]
        interference_reference = arrays["interference"][interference_mask]
    else:
        raise AssertionError(f"{record.sample_id} unknown SNR measurement")
    measured_snr = 20.0 * math.log10(
        (rms(clean_reference) + EPSILON)
        / (rms(interference_reference) + EPSILON)
    )
    stored_snr_error = abs(measured_snr - record.snr_db)
    requested_snr_error = abs(measured_snr - record.snr_db_requested)
    if stored_snr_error > SNR_TOLERANCE_DB:
        raise AssertionError(
            f"{record.sample_id} realized SNR mismatch: {stored_snr_error:.6f} dB"
        )
    if requested_snr_error > SNR_TOLERANCE_DB:
        raise AssertionError(
            f"{record.sample_id} requested SNR mismatch: "
            f"{requested_snr_error:.6f} dB"
        )

    timeline_path = root / "timelines" / f"{record.sample_id}.svg"
    if not timeline_path.exists() or timeline_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing timeline: {timeline_path}")

    return {
        "reconstruction_error": reconstruction_error,
        "peak": peak,
        "stored_peak_error": stored_peak_error,
        "outside_interference_peak": outside_peak,
        "outside_clean_peak": clean_outside_peak,
        "stored_snr_error_db": stored_snr_error,
        "requested_snr_error_db": requested_snr_error,
    }


def validate_source_event(
    event: EventAnnotation,
    project_root: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    hash_cache: Dict[Path, str],
) -> Path:
    if event.source_dataset != "AudioTime":
        raise AssertionError(f"{event.source_id} source_dataset must be AudioTime")
    item = metadata.get(event.source_id)
    if not isinstance(item, dict):
        raise AssertionError(f"Missing source ID in AudioTime metadata: {event.source_id}")
    events = item.get("event")
    if not isinstance(events, dict) or len(events) != 1:
        raise AssertionError(f"Source metadata is not single-event: {event.source_id}")
    label, intervals = next(iter(events.items()))
    if str(label).strip() != event.label:
        raise AssertionError(f"Source label mismatch: {event.source_id}")
    if not isinstance(intervals, list) or len(intervals) != 1:
        raise AssertionError(f"Source interval count mismatch: {event.source_id}")
    metadata_interval = intervals[0]
    if not isinstance(metadata_interval, list) or len(metadata_interval) != 2:
        raise AssertionError(f"Malformed source interval: {event.source_id}")
    if not (
        math.isclose(float(metadata_interval[0]), event.source_interval_seconds[0], abs_tol=1e-6)
        and math.isclose(float(metadata_interval[1]), event.source_interval_seconds[1], abs_tol=1e-6)
    ):
        raise AssertionError(f"Source interval mismatch: {event.source_id}")

    source_path = resolve_under(
        project_root, event.source_path, f"source {event.source_id}"
    )
    expected_source_path = (
        metadata_path.parent / "audio" / f"{event.source_id}.wav"
    ).resolve(strict=True)
    if source_path != expected_source_path:
        raise AssertionError(
            f"Source path does not match canonical metadata path: {event.source_id}"
        )
    if source_path not in hash_cache:
        hash_cache[source_path] = sha256_file(source_path)
    if hash_cache[source_path] != event.source_sha256:
        raise AssertionError(f"Source SHA256 mismatch: {event.source_id}")
    return source_path


def validate_scene_pair(scene_id: str, records: Sequence[QARecord]) -> Dict[str, Any]:
    if len(records) != 2:
        raise AssertionError(f"{scene_id} must contain exactly two records")
    by_type = {record.question_type: record for record in records}
    question_types = set(by_type)
    if question_types == {"temporal_before", "temporal_after"}:
        early_type = "temporal_before"
        late_type = "temporal_after"
        family = "relative"
    elif question_types == {"temporal_first", "temporal_last"}:
        early_type = "temporal_first"
        late_type = "temporal_last"
        family = "ordinal"
    else:
        raise AssertionError(f"{scene_id} has invalid paired question types: {question_types}")
    early_record = by_type[early_type]
    late_record = by_type[late_type]

    shared_fields = (
        "split",
        "sample_rate",
        "num_channels",
        "num_samples",
        "duration_seconds",
        "mixture_path",
        "clean_path",
        "interference_stem_path",
        "event_presence_labels",
        "edit_needed",
        "edit_rationale",
        "selector_target",
        "snr_measurement",
        "snr_db_requested",
        "snr_db",
        "mixture_peak",
        "source_group_ids",
        "generation_seed",
    )
    for field_name in shared_fields:
        if getattr(early_record, field_name) != getattr(late_record, field_name):
            raise AssertionError(f"{scene_id} pair differs in {field_name}")
    if early_record.anchor_event_ids != late_record.answer_event_ids:
        raise AssertionError(f"{scene_id} anchor IDs are not swapped to answer IDs")
    if early_record.answer_event_ids != late_record.anchor_event_ids:
        raise AssertionError(f"{scene_id} answer IDs are not swapped to anchor IDs")
    if not intervals_equal(early_record.anchor_intervals, late_record.answer_intervals):
        raise AssertionError(f"{scene_id} anchor masks are not complementary")
    if not intervals_equal(early_record.answer_intervals, late_record.anchor_intervals):
        raise AssertionError(f"{scene_id} answer masks are not complementary")

    early_events = {event.event_id: event for event in early_record.events}
    late_events = {event.event_id: event for event in late_record.events}
    if set(early_events) != {"event_early", "event_late", "event_interference"}:
        raise AssertionError(f"{scene_id} event IDs do not identify physical events")
    if set(early_events) != set(late_events):
        raise AssertionError(f"{scene_id} paired event IDs differ")
    for event_id in early_events:
        if event_identity(early_events[event_id]) != event_identity(late_events[event_id]):
            raise AssertionError(f"{scene_id} physical event changed: {event_id}")
    if early_events["event_early"].role != "answer":
        raise AssertionError(f"{scene_id} early-question answer is not event_early")
    if early_events["event_late"].role != "anchor":
        raise AssertionError(f"{scene_id} early-question anchor is not event_late")
    if late_events["event_early"].role != "anchor":
        raise AssertionError(f"{scene_id} late-question anchor is not event_early")
    if late_events["event_late"].role != "answer":
        raise AssertionError(f"{scene_id} late-question answer is not event_late")
    if (
        early_events["event_interference"].role != "interference"
        or late_events["event_interference"].role != "interference"
    ):
        raise AssertionError(f"{scene_id} interference role changed")

    physical_semantic_order = (
        early_events["event_early"].label,
        early_events["event_late"].label,
    )
    semantic_labels = tuple(sorted(physical_semantic_order))
    semantic_reversed = physical_semantic_order == (
        semantic_labels[1],
        semantic_labels[0],
    )
    candidate_order: Tuple[str, str] | None = None
    if family == "ordinal":
        candidate_order = semantic_labels
        expected_first = (
            f"Which sound occurs first, {candidate_order[0]} "
            f"or {candidate_order[1]}?"
        )
        expected_last = (
            f"Which sound occurs last, {candidate_order[0]} "
            f"or {candidate_order[1]}?"
        )
        if early_record.question != expected_first or late_record.question != expected_last:
            raise AssertionError(
                f"{scene_id} ordinal candidate order must be family-level sorted labels"
            )

    interference_event = early_events["event_interference"]
    return {
        "family": family,
        "semantic_pair": semantic_labels,
        "physical_semantic_order": physical_semantic_order,
        "semantic_reversed": semantic_reversed,
        "candidate_order": candidate_order,
        "interference_label": interference_event.label,
        "interference_source_id": interference_event.source_id,
        "interference_source_path": interference_event.source_path,
        "interference_source_sha256": interference_event.source_sha256,
        "edit_state": "edit_needed" if early_record.edit_needed else "no_edit",
        "edit_rationale": early_record.edit_rationale,
        "questions": {
            early_record.question_type: early_record.question,
            late_record.question_type: late_record.question,
        },
        "answers": {
            early_record.question_type: early_record.answer,
            late_record.question_type: late_record.answer,
        },
        "onsets": (
            early_events["event_early"].onset_seconds,
            early_events["event_late"].onset_seconds,
        ),
        "audio_paths": (
            early_record.mixture_path,
            early_record.clean_path,
            early_record.interference_stem_path,
        ),
    }


def artifact_identity(
    root: Path, relative_paths: Iterable[str]
) -> Tuple[str, Dict[str, str]]:
    per_file: Dict[str, str] = {}
    digest = hashlib.sha256()
    for relative_path in sorted(set(relative_paths)):
        file_path = resolve_under(root, relative_path, "artifact")
        file_hash = sha256_file(file_path)
        per_file[relative_path] = file_hash
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), per_file


def validate_dataset(root: Path, write_report: bool = False) -> Dict[str, Any]:
    """Validate a built QA v2 dataset and return its independent report."""

    root = root.resolve()
    project_root = root.parents[1].resolve()
    config_path = root / "dataset_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Dataset config must be an object")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("Dataset config schema_version mismatch")
    runtime_identity, build_identity = validate_config_identity(config, project_root)
    expected_rate = int(config["sample_rate"])
    expected_channels = int(config["num_channels"])
    expected_samples = int(config["num_samples"])
    expected_duration = float(config["duration_seconds"])
    expected_subtype = str(config["audio_format"]["subtype"])
    expected_count = int(config["splits"]["qa_overfit16"])
    if expected_rate != 32_000 or expected_channels != 1 or expected_subtype != "PCM_16":
        raise AssertionError("Dataset must be 32 kHz mono PCM-16")
    if expected_samples != int(round(expected_rate * expected_duration)):
        raise AssertionError("Config sample count and duration disagree")
    if expected_count != 16:
        raise AssertionError("qa_overfit16 must contain exactly 16 records")
    if config.get("scenes") != {
        "count": 8,
        "records_per_scene": 2,
        "shared_audio_with_complementary_roles": True,
    }:
        raise AssertionError("Config must declare eight paired shared-audio scenes")

    quota = config.get("quota", {})
    expected_quota_values = {
        "per_question_type": 4,
        "per_question_type_and_edit_state": 2,
        "exact_question_count": 4,
        "per_exact_question": 4,
        "per_exact_question_and_edit_state": 2,
        "distinct_edit_selector_targets_per_exact_question": 2,
        "interference_label_count": 2,
        "sources_per_interference_label": 4,
        "edit_needed": 8,
        "no_edit": 8,
    }
    for key, expected_value in expected_quota_values.items():
        if quota.get(key) != expected_value:
            raise AssertionError(
                f"Config quota {key} must be {expected_value}, got {quota.get(key)!r}"
            )
    if quota.get("question_types") != list(QUESTION_TYPES):
        raise AssertionError("Config question-type quota order mismatch")

    expected_question_control = {
        "relative_semantic_order_rule": RELATIVE_SEMANTIC_ORDER_RULE,
        "ordinal_semantic_order_rule": ORDINAL_SEMANTIC_ORDER_RULE,
        "ordinal_candidate_order_rule": ORDINAL_CANDIDATE_ORDER_RULE,
    }
    if config.get("question_control") != expected_question_control:
        raise AssertionError("Config question-control rules mismatch")
    if config.get("answer_usage", {}).get("model_input_fields") != [
        "mixture_path",
        "question",
    ]:
        raise AssertionError("Config must exclude answer from model input fields")
    if config.get("answer_usage", {}).get("supervision_only_fields") != ["answer"]:
        raise AssertionError("Config must mark answer as supervision-only")
    edit_labeling = config.get("edit_labeling", {})
    if edit_labeling.get("synthetic_proxy") is not True:
        raise AssertionError("Config must identify edit labels as synthetic proxies")
    notice = str(edit_labeling.get("notice", "")).lower()
    if "not measurements of causal qa degradation" not in notice:
        raise AssertionError("Config must disclaim measured causal QA impact")

    snr_config = config.get("snr_measurement", {})
    if snr_config.get("overlap") != OVERLAP_SNR_MEASUREMENT:
        raise AssertionError("Config overlap SNR definition mismatch")
    if snr_config.get("non_overlap") != NON_OVERLAP_SNR_MEASUREMENT:
        raise AssertionError("Config non-overlap SNR definition mismatch")
    configured_snr_tolerance = float(
        snr_config.get("requested_realized_tolerance_db", -1.0)
    )
    if not math.isclose(configured_snr_tolerance, SNR_TOLERANCE_DB, abs_tol=1e-9):
        raise AssertionError("Config SNR tolerance mismatch")
    expected_snr_by_rationale = {
        key: float(value)
        for key, value in config["composition"]["rationale_snr_db"].items()
    }
    if set(expected_snr_by_rationale) != set(EDIT_RATIONALES):
        raise AssertionError("Config must define SNR for every edit rationale")
    if not -4.0 <= expected_snr_by_rationale["overlap_obstructive_proxy"] <= -2.0:
        raise AssertionError("Obstructive proxy SNR must be about -3 dB")
    if not 17.0 <= expected_snr_by_rationale["overlap_benign_proxy"] <= 19.0:
        raise AssertionError("Benign overlap proxy SNR must be about +18 dB")
    if not 5.0 <= expected_snr_by_rationale["non_overlap"] <= 7.0:
        raise AssertionError("Non-overlap SNR must be about +6 dB")

    source_config = config.get("source", {})
    metadata_relative_path = str(source_config.get("metadata_path", ""))
    metadata_path = resolve_under(project_root, metadata_relative_path, "source metadata")
    metadata_sha256 = sha256_file(metadata_path)
    if source_config.get("metadata_sha256") != metadata_sha256:
        raise AssertionError("AudioTime metadata SHA256 mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError("AudioTime metadata must be an object")
    receipt_relative_path = source_config.get("acquisition_receipt_path")
    receipt_expected_hash = source_config.get("acquisition_receipt_sha256")
    if not isinstance(receipt_relative_path, str) or not isinstance(
        receipt_expected_hash, str
    ):
        raise AssertionError("Config must pin an AudioTime acquisition receipt")
    receipt_path = resolve_under(
        project_root, receipt_relative_path, "source acquisition receipt"
    )
    receipt_hash = sha256_file(receipt_path)
    if receipt_hash != receipt_expected_hash:
        raise AssertionError("AudioTime acquisition receipt SHA256 mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("format") != "audiotime_subset_v1":
        raise AssertionError("Unsupported AudioTime acquisition receipt")
    if receipt.get("metadata_sha256") != metadata_sha256:
        raise AssertionError("Receipt and dataset use different AudioTime metadata")
    receipt_sources = receipt.get("sources")
    if not isinstance(receipt_sources, dict) or len(receipt_sources) != 24:
        raise AssertionError("AudioTime receipt must contain exactly 24 sources")
    for source_id, source_receipt in receipt_sources.items():
        if not isinstance(source_receipt, dict):
            raise TypeError(f"Invalid receipt entry for {source_id}")
        source_path = metadata_path.parent / "audio" / f"{source_id}.wav"
        if sha256_file(source_path) != source_receipt.get("sha256"):
            raise AssertionError(f"Receipt audio SHA256 mismatch for {source_id}")
    acquisition_revision = receipt.get("dataset_revision")
    if not isinstance(acquisition_revision, str) or len(acquisition_revision) != 40:
        raise AssertionError("Receipt must pin a 40-character Hub revision")

    manifest_path = root / "qa_overfit16.jsonl"
    rows = read_jsonl(manifest_path)
    if len(rows) != expected_count:
        raise AssertionError(
            f"qa_overfit16 count mismatch: {len(rows)} != {expected_count}"
        )

    records: List[QARecord] = []
    records_by_scene: Dict[str, List[QARecord]] = defaultdict(list)
    sample_ids: Set[str] = set()
    split_counts: Counter = Counter()
    question_counts: Counter = Counter()
    edit_counts: Counter = Counter()
    matrix_counts: Counter = Counter()
    rationale_counts: Counter = Counter()
    exact_question_counts: Counter = Counter()
    exact_question_state_counts: Counter = Counter()
    exact_question_edit_selectors: Dict[str, Set[str]] = defaultdict(set)
    ordinal_question_state_answers: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    maximum_reconstruction_error = 0.0
    maximum_peak = 0.0
    maximum_stored_peak_error = 0.0
    maximum_outside_interference_peak = 0.0
    maximum_outside_clean_peak = 0.0
    maximum_stored_snr_error_db = 0.0
    maximum_requested_snr_error_db = 0.0
    source_hash_cache: Dict[Path, str] = {}
    source_occurrences: Dict[str, Counter] = {
        "id": Counter(),
        "path": Counter(),
        "hash": Counter(),
    }
    source_scene_owner: Dict[str, Dict[str, str]] = {
        "id": {},
        "path": {},
        "hash": {},
    }
    cross_scene_source_collisions: List[Tuple[str, str, str, str]] = []

    for row_index, row in enumerate(rows):
        try:
            record = parse_record(row)
        except Exception as exc:
            raise ValueError(f"Invalid record at row {row_index + 1}: {exc}") from exc
        if record.sample_id in sample_ids:
            raise AssertionError(f"Duplicate sample ID: {record.sample_id}")
        sample_ids.add(record.sample_id)
        records_by_scene[record.scene_id].append(record)

        record_source_paths: Set[str] = set()
        record_source_hashes: Set[str] = set()
        for event in record.events:
            canonical_source_path = validate_source_event(
                event,
                project_root,
                metadata_path,
                metadata,
                source_hash_cache,
            )
            path_key = canonical_source_path.as_posix()
            record_source_paths.add(path_key)
            record_source_hashes.add(event.source_sha256)
            dimensions = {
                "id": event.source_id,
                "path": path_key,
                "hash": event.source_sha256,
            }
            for dimension, value in dimensions.items():
                source_occurrences[dimension][value] += 1
                owner = source_scene_owner[dimension].get(value)
                if owner is None:
                    source_scene_owner[dimension][value] = record.scene_id
                elif owner != record.scene_id:
                    cross_scene_source_collisions.append(
                        (dimension, value, owner, record.scene_id)
                    )
        if len(record_source_paths) != 3 or len(record_source_hashes) != 3:
            raise AssertionError(
                f"{record.sample_id} must use three distinct source paths and hashes"
            )

        audio_metrics = validate_audio_record(
            root,
            record,
            expected_rate,
            expected_channels,
            expected_samples,
            expected_duration,
            expected_subtype,
            expected_snr_by_rationale,
        )
        maximum_reconstruction_error = max(
            maximum_reconstruction_error, audio_metrics["reconstruction_error"]
        )
        maximum_peak = max(maximum_peak, audio_metrics["peak"])
        maximum_stored_peak_error = max(
            maximum_stored_peak_error, audio_metrics["stored_peak_error"]
        )
        maximum_outside_interference_peak = max(
            maximum_outside_interference_peak,
            audio_metrics["outside_interference_peak"],
        )
        maximum_outside_clean_peak = max(
            maximum_outside_clean_peak, audio_metrics["outside_clean_peak"]
        )
        maximum_stored_snr_error_db = max(
            maximum_stored_snr_error_db,
            audio_metrics["stored_snr_error_db"],
        )
        maximum_requested_snr_error_db = max(
            maximum_requested_snr_error_db,
            audio_metrics["requested_snr_error_db"],
        )
        split_counts[record.split] += 1
        question_counts[record.question_type] += 1
        edit_state = "edit_needed" if record.edit_needed else "no_edit"
        edit_counts[edit_state] += 1
        matrix_counts[(record.question_type, edit_state)] += 1
        rationale_counts[record.edit_rationale] += 1
        exact_question_counts[record.question] += 1
        exact_question_state_counts[(record.question, edit_state)] += 1
        if record.edit_needed:
            exact_question_edit_selectors[record.question].add(record.selector_target)
        if record.question_type in {"temporal_first", "temporal_last"}:
            ordinal_question_state_answers[(record.question, edit_state)].add(
                record.answer
            )
        records.append(record)

    if split_counts != Counter({"train": expected_count}):
        raise AssertionError(
            f"QA v2 must be a single train split, got {dict(split_counts)}"
        )

    cross_scene_source_reuse = bool(cross_scene_source_collisions)
    if cross_scene_source_reuse:
        raise AssertionError(
            "Cross-scene source ID/path/hash reuse: "
            f"{cross_scene_source_collisions[:3]}"
        )
    for dimension, occurrences in source_occurrences.items():
        bad_counts = {
            value: count for value, count in occurrences.items() if count != 2
        }
        if bad_counts:
            raise AssertionError(
                f"Source {dimension} reuse must be exactly the two records of one scene: "
                f"{bad_counts}"
            )
    if len(source_occurrences["id"]) != 24:
        raise AssertionError("Eight scenes must use exactly 24 unique sources")

    if len(records_by_scene) != 8:
        raise AssertionError(f"Expected 8 scenes, found {len(records_by_scene)}")
    scene_summaries: Dict[str, Dict[str, Any]] = {}
    semantic_pair_states: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    semantic_pair_interference: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    semantic_pair_edit_interference: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    interference_label_states: Dict[str, Set[str]] = defaultdict(set)
    interference_family_state_scene_counts: Counter = Counter()
    ordinal_semantic_order_state_counts: Counter = Counter()
    family_semantic_pairs: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    family_physical_semantic_orders: Dict[
        str, Set[Tuple[str, str]]
    ] = defaultdict(set)
    audio_path_scene_owner: Dict[str, str] = {}
    onset_pairs: Set[Tuple[float, float]] = set()
    family_counts: Counter = Counter()
    scene_rationale_counts: Counter = Counter()
    for scene_id, scene_records in sorted(records_by_scene.items()):
        summary = validate_scene_pair(scene_id, scene_records)
        scene_summaries[scene_id] = summary
        family_counts[summary["family"]] += 1
        scene_rationale_counts[summary["edit_rationale"]] += 1
        semantic_pair = summary["semantic_pair"]
        edit_state = summary["edit_state"]
        interference_label = summary["interference_label"]
        semantic_pair_states[semantic_pair].add(edit_state)
        semantic_pair_interference[semantic_pair].add(interference_label)
        if edit_state == "edit_needed":
            semantic_pair_edit_interference[semantic_pair].add(interference_label)
        interference_label_states[interference_label].add(edit_state)
        interference_family_state_scene_counts[
            (summary["family"], edit_state, interference_label)
        ] += 1
        family_semantic_pairs[summary["family"]].add(semantic_pair)
        family_physical_semantic_orders[summary["family"]].add(
            summary["physical_semantic_order"]
        )
        if summary["family"] == "ordinal":
            ordinal_semantic_order_state_counts[
                (edit_state, bool(summary["semantic_reversed"]))
            ] += 1
        onset_pairs.add(summary["onsets"])
        for audio_path in summary["audio_paths"]:
            owner = audio_path_scene_owner.get(audio_path)
            if owner is not None and owner != scene_id:
                raise AssertionError(
                    f"Audio path shared across scenes: {audio_path}, {owner}, {scene_id}"
                )
            audio_path_scene_owner[audio_path] = scene_id

    if family_counts != Counter({"relative": 4, "ordinal": 4}):
        raise AssertionError(f"Scene family balance mismatch: {dict(family_counts)}")
    if set(family_semantic_pairs) != {"relative", "ordinal"} or any(
        len(pairs) != 1 for pairs in family_semantic_pairs.values()
    ):
        raise AssertionError(
            "Each question family must use exactly one semantic label pair"
        )
    if len(family_physical_semantic_orders["relative"]) != 1:
        raise AssertionError("Relative semantic order must be fixed across scenes")
    ordinal_semantic_pair = next(iter(family_semantic_pairs["ordinal"]))
    if family_physical_semantic_orders["ordinal"] != {
        ordinal_semantic_pair,
        tuple(reversed(ordinal_semantic_pair)),
    }:
        raise AssertionError(
            "Ordinal scenes must cover canonical and reversed semantic orders"
        )
    if len(onset_pairs) != 8:
        raise AssertionError("Every scene must have a distinct deterministic onset layout")

    required_states = {"edit_needed", "no_edit"}
    for semantic_pair, states in semantic_pair_states.items():
        if states != required_states:
            raise AssertionError(
                f"Semantic pair lacks both edit states: {semantic_pair}, {states}"
            )
        if len(semantic_pair_interference[semantic_pair]) != 2:
            raise AssertionError(
                f"Semantic pair must use exactly two interference labels: {semantic_pair}"
            )
        if len(semantic_pair_edit_interference[semantic_pair]) != 2:
            raise AssertionError(
                f"Edit selector cannot map one semantic pair to one class: {semantic_pair}"
            )

    interference_labels = set(interference_label_states)
    if len(interference_labels) != 2:
        raise AssertionError(
            f"Expected exactly two shared interference labels, got "
            f"{sorted(interference_labels)}"
        )
    for interference_label, states in interference_label_states.items():
        if states != required_states:
            raise AssertionError(
                f"Interference label lacks both edit states: {interference_label}, {states}"
            )
    for family in ("relative", "ordinal"):
        for edit_state in ("edit_needed", "no_edit"):
            actual_label_counts = Counter(
                {
                    label: interference_family_state_scene_counts[
                        (family, edit_state, label)
                    ]
                    for label in interference_labels
                    if interference_family_state_scene_counts[
                        (family, edit_state, label)
                    ]
                }
            )
            expected_label_counts = Counter(
                {label: 1 for label in interference_labels}
            )
            if actual_label_counts != expected_label_counts:
                raise AssertionError(
                    "Interference-label family/state coverage mismatch for "
                    f"{family}/{edit_state}: {dict(actual_label_counts)}"
                )

    expected_ordinal_semantic_order_balance = Counter(
        {
            ("edit_needed", False): 1,
            ("edit_needed", True): 1,
            ("no_edit", False): 1,
            ("no_edit", True): 1,
        }
    )
    if (
        ordinal_semantic_order_state_counts
        != expected_ordinal_semantic_order_balance
    ):
        raise AssertionError(
            "Ordinal semantic-order/edit balance mismatch: "
            f"{dict(ordinal_semantic_order_state_counts)}"
        )

    expected_question_types = set(QUESTION_TYPES)
    if set(question_counts) != expected_question_types:
        raise AssertionError(
            f"Question type coverage mismatch: {sorted(question_counts)}"
        )
    for question_type in QUESTION_TYPES:
        if question_counts[question_type] != 4:
            raise AssertionError(f"{question_type} must have 4 records")
        for edit_state in ("edit_needed", "no_edit"):
            if matrix_counts[(question_type, edit_state)] != 2:
                raise AssertionError(
                    f"{question_type}/{edit_state} must have 2 records"
                )

    questions_by_type = {
        question_type: {
            record.question
            for record in records
            if record.question_type == question_type
        }
        for question_type in QUESTION_TYPES
    }
    if any(len(questions) != 1 for questions in questions_by_type.values()):
        raise AssertionError(
            "Each question type must use one exact question string: "
            f"{questions_by_type}"
        )
    if len(exact_question_counts) != 4:
        raise AssertionError(
            f"Expected four exact question strings, got {len(exact_question_counts)}"
        )
    for question, count in exact_question_counts.items():
        if count != 4:
            raise AssertionError(
                f"Exact question must occur four times: {question!r} has {count}"
            )
        for edit_state in ("edit_needed", "no_edit"):
            state_count = exact_question_state_counts[(question, edit_state)]
            if state_count != 2:
                raise AssertionError(
                    f"Exact question {question!r}/{edit_state} must have 2 records, "
                    f"got {state_count}"
                )
        if exact_question_edit_selectors[question] != interference_labels:
            raise AssertionError(
                f"Exact question {question!r} must have both edit selector targets: "
                f"{sorted(exact_question_edit_selectors[question])}"
            )

    ordinal_exact_questions = (
        questions_by_type["temporal_first"]
        | questions_by_type["temporal_last"]
    )
    expected_ordinal_answers = set(ordinal_semantic_pair)
    for question in ordinal_exact_questions:
        for edit_state in ("edit_needed", "no_edit"):
            actual_answers = ordinal_question_state_answers[(question, edit_state)]
            if actual_answers != expected_ordinal_answers:
                raise AssertionError(
                    f"Ordinal exact question {question!r}/{edit_state} must have "
                    f"both answers {sorted(expected_ordinal_answers)}, got "
                    f"{sorted(actual_answers)}"
                )

    if edit_counts != Counter({"edit_needed": 8, "no_edit": 8}):
        raise AssertionError(f"Edit-state balance mismatch: {dict(edit_counts)}")
    expected_rationale_counts = Counter(
        {"overlap_obstructive_proxy": 8, "overlap_benign_proxy": 4, "non_overlap": 4}
    )
    if rationale_counts != expected_rationale_counts:
        raise AssertionError(
            f"Edit-rationale balance mismatch: {dict(rationale_counts)}"
        )
    if scene_rationale_counts != Counter(
        {"overlap_obstructive_proxy": 4, "overlap_benign_proxy": 2, "non_overlap": 2}
    ):
        raise AssertionError(
            f"Scene rationale balance mismatch: {dict(scene_rationale_counts)}"
        )

    unique_mixture_paths = {record.mixture_path for record in records}
    unique_clean_paths = {record.clean_path for record in records}
    unique_interference_paths = {
        record.interference_stem_path for record in records
    }
    if not (
        len(unique_mixture_paths)
        == len(unique_clean_paths)
        == len(unique_interference_paths)
        == 8
    ):
        raise AssertionError("Expected eight files in each audio stem directory")
    expected_wav_paths = (
        unique_mixture_paths | unique_clean_paths | unique_interference_paths
    )
    actual_wav_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*.wav")
    }
    if actual_wav_paths != expected_wav_paths:
        raise AssertionError(
            "WAV artifact set mismatch: "
            f"missing={sorted(expected_wav_paths - actual_wav_paths)}, "
            f"extra={sorted(actual_wav_paths - expected_wav_paths)}"
        )
    expected_timeline_paths = {
        f"timelines/{record.sample_id}.svg" for record in records
    }
    actual_timeline_paths = {
        path.relative_to(root).as_posix() for path in (root / "timelines").glob("*.svg")
    }
    if actual_timeline_paths != expected_timeline_paths:
        raise AssertionError(
            "Timeline artifact set mismatch: "
            f"missing={sorted(expected_timeline_paths - actual_timeline_paths)}, "
            f"extra={sorted(actual_timeline_paths - expected_timeline_paths)}"
        )

    if set(source_occurrences["id"]) != set(receipt_sources):
        raise AssertionError(
            "Manifest source IDs do not exactly match the acquisition receipt"
        )

    fingerprint_paths = {
        "dataset_config.json",
        "qa_overfit16.jsonl",
    } | expected_wav_paths | expected_timeline_paths
    artifact_fingerprint_sha256, artifact_file_sha256 = artifact_identity(
        root, fingerprint_paths
    )
    if len(artifact_file_sha256) != 42:
        raise AssertionError("Artifact fingerprint must cover config, manifest, WAVs, and SVGs")

    report = {
        "status": "passed",
        "checked_samples": len(records),
        "checked_scenes": len(records_by_scene),
        "schema_version": SCHEMA_VERSION,
        "sample_rate": expected_rate,
        "num_channels": expected_channels,
        "num_samples": expected_samples,
        "duration_seconds": expected_duration,
        "audio_format": expected_subtype,
        "runtime_identity": runtime_identity,
        "build_identity": build_identity,
        "config_identity_verified": True,
        "split_counts": dict(sorted(split_counts.items())),
        "question_type_counts": dict(sorted(question_counts.items())),
        "edit_state_counts": dict(sorted(edit_counts.items())),
        "edit_rationale_counts": dict(sorted(rationale_counts.items())),
        "question_edit_matrix": {
            question_type: {
                edit_state: matrix_counts[(question_type, edit_state)]
                for edit_state in ("edit_needed", "no_edit")
            }
            for question_type in QUESTION_TYPES
        },
        "exact_question_counts": dict(sorted(exact_question_counts.items())),
        "exact_question_edit_matrix": {
            question: {
                edit_state: exact_question_state_counts[(question, edit_state)]
                for edit_state in ("edit_needed", "no_edit")
            }
            for question in sorted(exact_question_counts)
        },
        "exact_question_edit_selector_targets": {
            question: sorted(exact_question_edit_selectors[question])
            for question in sorted(exact_question_counts)
        },
        "ordinal_exact_question_state_answers": {
            question: {
                edit_state: sorted(
                    ordinal_question_state_answers[(question, edit_state)]
                )
                for edit_state in ("edit_needed", "no_edit")
            }
            for question in sorted(ordinal_exact_questions)
        },
        "scene_family_counts": dict(sorted(family_counts.items())),
        "unique_onset_layouts": len(onset_pairs),
        "semantic_pair_count": len(semantic_pair_states),
        "interference_labels": sorted(interference_labels),
        "interference_label_count": len(interference_labels),
        "interference_label_family_state_scene_counts": {
            family: {
                edit_state: {
                    label: interference_family_state_scene_counts[
                        (family, edit_state, label)
                    ]
                    for label in sorted(interference_labels)
                }
                for edit_state in ("edit_needed", "no_edit")
            }
            for family in ("relative", "ordinal")
        },
        "ordinal_semantic_order_edit_matrix": {
            edit_state: {
                "canonical": ordinal_semantic_order_state_counts[
                    (edit_state, False)
                ],
                "reversed": ordinal_semantic_order_state_counts[
                    (edit_state, True)
                ],
            }
            for edit_state in ("edit_needed", "no_edit")
        },
        "unique_source_count": len(source_occurrences["id"]),
        "source_occurrence_count": sum(source_occurrences["id"].values()),
        "within_scene_source_reuse": True,
        "cross_scene_source_reuse": cross_scene_source_reuse,
        "source_leakage": None,
        "source_leakage_checked": False,
        "source_leakage_scope": "not_applicable_single_train_split",
        "source_metadata_sha256": metadata_sha256,
        "source_acquisition_revision": acquisition_revision,
        "source_acquisition_receipt_sha256": receipt_hash,
        "mixture_file_count": len(unique_mixture_paths),
        "clean_file_count": len(unique_clean_paths),
        "interference_file_count": len(unique_interference_paths),
        "timeline_count": len(expected_timeline_paths),
        "maximum_peak": maximum_peak,
        "maximum_reconstruction_error": maximum_reconstruction_error,
        "maximum_stored_peak_error": maximum_stored_peak_error,
        "maximum_outside_interference_peak": maximum_outside_interference_peak,
        "maximum_outside_clean_peak": maximum_outside_clean_peak,
        "maximum_stored_snr_error_db": maximum_stored_snr_error_db,
        "maximum_requested_snr_error_db": maximum_requested_snr_error_db,
        "config_sha256": artifact_file_sha256["dataset_config.json"],
        "manifest_sha256": artifact_file_sha256["qa_overfit16.jsonl"],
        "artifact_fingerprint_sha256": artifact_fingerprint_sha256,
        "artifact_file_count": len(artifact_file_sha256),
        "artifact_file_sha256": artifact_file_sha256,
    }
    if write_report:
        (root / "validation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "removal_qa_synthetic_v2",
    )
    args = parser.parse_args()
    validate_dataset(args.dataset_root, write_report=True)


if __name__ == "__main__":
    main()
