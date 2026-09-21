"""Typed schema and semantic validation for QA removal dataset v2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Sequence, Tuple


SCHEMA_VERSION = "qa_removal_v2"
NO_EDIT_TARGET = "no_edit"
QUESTION_TYPES = (
    "temporal_before",
    "temporal_after",
    "temporal_first",
    "temporal_last",
)
RELATIVE_SEMANTIC_ORDER_RULE = "fixed_across_family_scenes"
ORDINAL_SEMANTIC_ORDER_RULE = "canonical_slots_0_2_reversed_slots_1_3"
ORDINAL_CANDIDATE_ORDER_RULE = "family_sorted_labels_fixed_across_scenes"
EVENT_ROLES = ("anchor", "answer", "interference")
EDIT_RATIONALES = (
    "overlap_obstructive_proxy",
    "overlap_benign_proxy",
    "non_overlap",
)
OVERLAP_SNR_MEASUREMENT = "preserve_interference_intersection"
NON_OVERLAP_SNR_MEASUREMENT = "evidence_vs_active_interference"
SNR_MEASUREMENTS = (
    OVERLAP_SNR_MEASUREMENT,
    NON_OVERLAP_SNR_MEASUREMENT,
)
SNR_METADATA_TOLERANCE_DB = 0.15
Interval = Tuple[float, float]

_RECORD_FIELDS = {
    "schema_version",
    "id",
    "scene_id",
    "split",
    "sample_rate",
    "num_channels",
    "num_samples",
    "duration_seconds",
    "mixture_path",
    "clean_path",
    "interference_stem_path",
    "question",
    "answer",
    "question_type",
    "events",
    "anchor_event_ids",
    "answer_event_ids",
    "anchor_intervals",
    "answer_intervals",
    "interference_event_ids",
    "interference_intervals",
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
}
_EVENT_FIELDS = {
    "event_id",
    "label",
    "source_dataset",
    "source_id",
    "source_path",
    "source_sha256",
    "source_interval_seconds",
    "source_crop_interval_seconds",
    "onset_seconds",
    "offset_seconds",
    "role",
}


def _require_exact_fields(
    value: Mapping[str, Any], required: set, context: str
) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing or extra:
        raise ValueError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return value


def _sha256(value: Any, context: str) -> str:
    result = _string(value, context)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a boolean")
    return value


def _interval(value: Any, context: str) -> Interval:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{context} must be a two-element JSON array")
    start = _number(value[0], f"{context}[0]")
    end = _number(value[1], f"{context}[1]")
    if start < 0.0 or end <= start:
        raise ValueError(f"{context} must satisfy 0 <= start < end")
    return start, end


def _intervals(value: Any, context: str) -> Tuple[Interval, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be a non-empty JSON array")
    return tuple(
        _interval(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )


def _string_list(value: Any, context: str) -> Tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TypeError(f"{context} must be a non-empty JSON array")
    result = tuple(
        _string(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{context} must not contain duplicates")
    return result


def _relative_wav_path(value: Any, context: str) -> str:
    result = _string(value, context)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".wav":
        raise ValueError(f"{context} must be a safe relative WAV path")
    return result


def _relative_source_path(value: Any, context: str) -> str:
    result = _string(value, context)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must be a safe relative path")
    return result


def intervals_overlap(left: Interval, right: Interval) -> bool:
    """Return whether two half-open temporal intervals overlap."""

    return max(left[0], right[0]) < min(left[1], right[1])


def interval_contains(outer: Interval, inner: Interval) -> bool:
    """Return whether ``outer`` contains ``inner`` within annotation tolerance."""

    return outer[0] <= inner[0] + 1e-6 and outer[1] >= inner[1] - 1e-6


def canonical_questions(
    question_type: str, anchor_label: str, answer_label: str
) -> Tuple[str, ...]:
    """Return every exact template allowed for the supplied structured roles."""

    if question_type == "temporal_before":
        return (f"What sound occurs immediately before {anchor_label}?",)
    if question_type == "temporal_after":
        return (f"What sound occurs immediately after {anchor_label}?",)
    if question_type not in {"temporal_first", "temporal_last"}:
        raise ValueError(f"Unsupported question type: {question_type}")
    operator = "first" if question_type == "temporal_first" else "last"
    return (
        f"Which sound occurs {operator}, {answer_label} or {anchor_label}?",
        f"Which sound occurs {operator}, {anchor_label} or {answer_label}?",
    )


@dataclass(frozen=True)
class EventAnnotation:
    """One rendered event and its source provenance."""

    event_id: str
    label: str
    source_dataset: str
    source_id: str
    source_path: str
    source_sha256: str
    source_interval_seconds: Interval
    source_crop_interval_seconds: Interval
    onset_seconds: float
    offset_seconds: float
    role: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], context: str) -> "EventAnnotation":
        if not isinstance(payload, dict):
            raise TypeError(f"{context} must be an object")
        _require_exact_fields(payload, _EVENT_FIELDS, context)
        role = _string(payload["role"], f"{context}.role")
        if role not in EVENT_ROLES:
            raise ValueError(f"{context}.role must be one of {EVENT_ROLES}")
        source_interval = _interval(
            payload["source_interval_seconds"],
            f"{context}.source_interval_seconds",
        )
        crop_interval = _interval(
            payload["source_crop_interval_seconds"],
            f"{context}.source_crop_interval_seconds",
        )
        if (
            crop_interval[0] < source_interval[0] - 1e-6
            or crop_interval[1] > source_interval[1] + 1e-6
        ):
            raise ValueError(f"{context} crop must lie inside its source interval")
        onset = _number(payload["onset_seconds"], f"{context}.onset_seconds")
        offset = _number(payload["offset_seconds"], f"{context}.offset_seconds")
        if onset < 0.0 or offset <= onset:
            raise ValueError(f"{context} rendered interval is invalid")
        rendered_duration = offset - onset
        crop_duration = crop_interval[1] - crop_interval[0]
        if not math.isclose(rendered_duration, crop_duration, abs_tol=1e-5):
            raise ValueError(
                f"{context} source crop and rendered durations differ: "
                f"{crop_duration} vs {rendered_duration}"
            )
        return cls(
            event_id=_string(payload["event_id"], f"{context}.event_id"),
            label=_string(payload["label"], f"{context}.label"),
            source_dataset=_string(
                payload["source_dataset"], f"{context}.source_dataset"
            ),
            source_id=_string(payload["source_id"], f"{context}.source_id"),
            source_path=_relative_source_path(
                payload["source_path"], f"{context}.source_path"
            ),
            source_sha256=_sha256(
                payload["source_sha256"], f"{context}.source_sha256"
            ),
            source_interval_seconds=source_interval,
            source_crop_interval_seconds=crop_interval,
            onset_seconds=onset,
            offset_seconds=offset,
            role=role,
        )

    @property
    def rendered_interval(self) -> Interval:
        return self.onset_seconds, self.offset_seconds


@dataclass(frozen=True)
class QARecord:
    """Validated top-level QA v2 manifest record."""

    schema_version: str
    sample_id: str
    scene_id: str
    split: str
    sample_rate: int
    num_channels: int
    num_samples: int
    duration_seconds: float
    mixture_path: str
    clean_path: str
    interference_stem_path: str
    question: str
    answer: str
    question_type: str
    events: Tuple[EventAnnotation, ...]
    anchor_event_ids: Tuple[str, ...]
    answer_event_ids: Tuple[str, ...]
    anchor_intervals: Tuple[Interval, ...]
    answer_intervals: Tuple[Interval, ...]
    interference_event_ids: Tuple[str, ...]
    interference_intervals: Tuple[Interval, ...]
    event_presence_labels: Tuple[str, ...]
    edit_needed: bool
    edit_rationale: str
    selector_target: str
    snr_measurement: str
    snr_db_requested: float
    snr_db: float
    mixture_peak: float
    source_group_ids: Tuple[str, ...]
    generation_seed: int

    def event_by_id(self, event_id: str) -> EventAnnotation:
        for event in self.events:
            if event.event_id == event_id:
                return event
        raise KeyError(event_id)


def _interval_lists_match(
    actual: Sequence[Interval], expected: Sequence[Interval]
) -> bool:
    if len(actual) != len(expected):
        return False
    return all(
        math.isclose(left[0], right[0], abs_tol=1e-6)
        and math.isclose(left[1], right[1], abs_tol=1e-6)
        for left, right in zip(actual, expected)
    )


def parse_record(payload: Dict[str, Any]) -> QARecord:
    """Parse and validate one JSON-compatible QA v2 record."""

    if not isinstance(payload, dict):
        raise TypeError("record must be an object")
    _require_exact_fields(payload, _RECORD_FIELDS, "record")

    schema_version = _string(payload["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {schema_version!r}"
        )
    sample_id = _string(payload["id"], "id")
    scene_id = _string(payload["scene_id"], "scene_id")
    if not scene_id.startswith("scene_"):
        raise ValueError("scene_id must start with 'scene_'")
    split = _string(payload["split"], "split")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if not sample_id.startswith(f"{split}_"):
        raise ValueError("id must start with its split name")

    sample_rate = _integer(payload["sample_rate"], "sample_rate")
    num_channels = _integer(payload["num_channels"], "num_channels")
    num_samples = _integer(payload["num_samples"], "num_samples")
    duration_seconds = _number(payload["duration_seconds"], "duration_seconds")
    if sample_rate <= 0 or num_samples <= 0 or duration_seconds <= 0.0:
        raise ValueError("audio dimensions must be positive")
    if num_channels != 1:
        raise ValueError("QA v2 currently requires mono audio")
    if int(round(sample_rate * duration_seconds)) != num_samples:
        raise ValueError("sample_rate × duration_seconds must equal num_samples")

    events_payload = payload["events"]
    if not isinstance(events_payload, list) or len(events_payload) != 3:
        raise ValueError("events must contain exactly anchor, answer, and interference")
    events = tuple(
        EventAnnotation.from_dict(item, f"events[{index}]")
        for index, item in enumerate(events_payload)
    )
    event_ids = [event.event_id for event in events]
    event_labels = [event.label for event in events]
    source_ids = [event.source_id for event in events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event IDs must be unique")
    if len(set(event_labels)) != len(event_labels):
        raise ValueError("event labels must be unique within a sample")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source IDs must be unique within a sample")
    for event in events:
        if event.offset_seconds > duration_seconds + 1e-6:
            raise ValueError(f"event {event.event_id} exceeds audio duration")

    anchor_event_ids = _string_list(payload["anchor_event_ids"], "anchor_event_ids")
    answer_event_ids = _string_list(payload["answer_event_ids"], "answer_event_ids")
    interference_event_ids = _string_list(
        payload["interference_event_ids"], "interference_event_ids"
    )
    anchor_intervals = _intervals(payload["anchor_intervals"], "anchor_intervals")
    answer_intervals = _intervals(payload["answer_intervals"], "answer_intervals")
    interference_intervals = _intervals(
        payload["interference_intervals"], "interference_intervals"
    )

    role_ids = {
        role: tuple(event.event_id for event in events if event.role == role)
        for role in EVENT_ROLES
    }
    supplied_ids = {
        "anchor": anchor_event_ids,
        "answer": answer_event_ids,
        "interference": interference_event_ids,
    }
    if any(len(role_ids[role]) != 1 for role in EVENT_ROLES):
        raise ValueError("exactly one event is required for each role")
    for role in EVENT_ROLES:
        if supplied_ids[role] != role_ids[role]:
            raise ValueError(f"{role}_event_ids do not match event roles")

    supplied_intervals = {
        "anchor": anchor_intervals,
        "answer": answer_intervals,
        "interference": interference_intervals,
    }
    for role in EVENT_ROLES:
        expected = tuple(
            event.rendered_interval for event in events if event.role == role
        )
        if not _interval_lists_match(supplied_intervals[role], expected):
            raise ValueError(f"{role}_intervals do not match event annotations")

    event_presence_labels = _string_list(
        payload["event_presence_labels"], "event_presence_labels"
    )
    if set(event_presence_labels) != set(event_labels):
        raise ValueError("event_presence_labels must exactly match rendered events")
    source_group_ids = _string_list(payload["source_group_ids"], "source_group_ids")
    if set(source_group_ids) != set(source_ids):
        raise ValueError("source_group_ids must exactly match event source IDs")

    question = _string(payload["question"], "question")
    answer = _string(payload["answer"], "answer")
    question_type = _string(payload["question_type"], "question_type")
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"question_type must be one of {QUESTION_TYPES}")
    answer_event = next(event for event in events if event.role == "answer")
    anchor_event = next(event for event in events if event.role == "anchor")
    interference_event = next(
        event for event in events if event.role == "interference"
    )
    if answer != answer_event.label:
        raise ValueError("answer must equal the answer-role event label")
    if question not in canonical_questions(
        question_type, anchor_event.label, answer_event.label
    ):
        raise ValueError("question does not match its exact temporal template and roles")

    if question_type in {"temporal_before", "temporal_first"}:
        if answer_event.offset_seconds >= anchor_event.onset_seconds:
            raise ValueError(f"{question_type} requires answer strictly before anchor")
    else:
        if anchor_event.offset_seconds >= answer_event.onset_seconds:
            raise ValueError(f"{question_type} requires answer strictly after anchor")

    edit_needed = _boolean(payload["edit_needed"], "edit_needed")
    edit_rationale = _string(payload["edit_rationale"], "edit_rationale")
    if edit_rationale not in EDIT_RATIONALES:
        raise ValueError(f"edit_rationale must be one of {EDIT_RATIONALES}")
    selector_target = _string(payload["selector_target"], "selector_target")
    if edit_needed and selector_target != interference_event.label:
        raise ValueError("edit samples must select the interference label")
    if not edit_needed and selector_target != NO_EDIT_TARGET:
        raise ValueError("no-edit samples must use selector_target='no_edit'")

    preserve_intervals = anchor_intervals + answer_intervals
    has_overlap = any(
        intervals_overlap(interference, preserve)
        for interference in interference_intervals
        for preserve in preserve_intervals
    )
    fully_covers_evidence = all(
        any(interval_contains(interference, preserve) for interference in interference_intervals)
        for preserve in preserve_intervals
    )
    if edit_rationale == "overlap_obstructive_proxy":
        if not edit_needed or not fully_covers_evidence:
            raise ValueError(
                "overlap_obstructive_proxy requires edit_needed and strong evidence overlap"
            )
    elif edit_rationale == "overlap_benign_proxy":
        if edit_needed or not fully_covers_evidence:
            raise ValueError(
                "overlap_benign_proxy requires no_edit and strong evidence overlap"
            )
    elif edit_needed or has_overlap:
        raise ValueError("non_overlap requires no_edit and disjoint interference")

    snr_measurement = _string(payload["snr_measurement"], "snr_measurement")
    if snr_measurement not in SNR_MEASUREMENTS:
        raise ValueError(f"snr_measurement must be one of {SNR_MEASUREMENTS}")
    expected_measurement = (
        NON_OVERLAP_SNR_MEASUREMENT
        if edit_rationale == "non_overlap"
        else OVERLAP_SNR_MEASUREMENT
    )
    if snr_measurement != expected_measurement:
        raise ValueError("snr_measurement does not match edit_rationale")
    snr_db_requested = _number(payload["snr_db_requested"], "snr_db_requested")
    snr_db = _number(payload["snr_db"], "snr_db")
    if abs(snr_db_requested - snr_db) > SNR_METADATA_TOLERANCE_DB:
        raise ValueError("requested and realized SNR differ beyond tolerance")

    mixture_peak = _number(payload["mixture_peak"], "mixture_peak")
    if not 0.0 < mixture_peak <= 1.0:
        raise ValueError("mixture_peak must be in (0, 1]")
    generation_seed = _integer(payload["generation_seed"], "generation_seed")
    if generation_seed < 0:
        raise ValueError("generation_seed must be non-negative")

    return QARecord(
        schema_version=schema_version,
        sample_id=sample_id,
        scene_id=scene_id,
        split=split,
        sample_rate=sample_rate,
        num_channels=num_channels,
        num_samples=num_samples,
        duration_seconds=duration_seconds,
        mixture_path=_relative_wav_path(payload["mixture_path"], "mixture_path"),
        clean_path=_relative_wav_path(payload["clean_path"], "clean_path"),
        interference_stem_path=_relative_wav_path(
            payload["interference_stem_path"], "interference_stem_path"
        ),
        question=question,
        answer=answer,
        question_type=question_type,
        events=events,
        anchor_event_ids=anchor_event_ids,
        answer_event_ids=answer_event_ids,
        anchor_intervals=anchor_intervals,
        answer_intervals=answer_intervals,
        interference_event_ids=interference_event_ids,
        interference_intervals=interference_intervals,
        event_presence_labels=event_presence_labels,
        edit_needed=edit_needed,
        edit_rationale=edit_rationale,
        selector_target=selector_target,
        snr_measurement=snr_measurement,
        snr_db_requested=snr_db_requested,
        snr_db=snr_db,
        mixture_peak=mixture_peak,
        source_group_ids=source_group_ids,
        generation_seed=generation_seed,
    )
