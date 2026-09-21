"""Strict schema for the question-contrastive QCES v3 diagnostic dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Optional, Tuple


SCHEMA_VERSION = "qces_v3"
NO_EVIDENCE_ANSWER = "no_evidence"
QUESTION_TYPES = (
    "temporal_after",
    "temporal_before",
    "temporal_first",
    "no_evidence_after",
)
Interval = Tuple[float, float]

_RECORD_FIELDS = {
    "schema_version",
    "id",
    "scene_id",
    "question_family_id",
    "question_index",
    "split",
    "sample_rate",
    "num_channels",
    "num_samples",
    "duration_seconds",
    "mixture_path",
    "evidence_stem_path",
    "residual_stem_path",
    "anchor_stem_path",
    "answer_stem_path",
    "question",
    "answer",
    "question_type",
    "no_evidence",
    "absent_label",
    "events",
    "anchor_event_ids",
    "answer_event_ids",
    "evidence_event_ids",
    "anchor_intervals",
    "answer_intervals",
    "event_presence_labels",
    "source_group_ids",
    "nuisance_snr_db_requested",
    "nuisance_snr_db",
    "mixture_peak",
    "generation_seed",
}
_EVENT_FIELDS = {
    "event_id",
    "label",
    "event_kind",
    "source_dataset",
    "source_id",
    "source_path",
    "source_sha256",
    "source_interval_seconds",
    "source_crop_interval_seconds",
    "onset_seconds",
    "offset_seconds",
    "stem_path",
}


def _exact(payload: Mapping[str, Any], fields: set[str], context: str) -> None:
    missing = sorted(fields - set(payload))
    extra = sorted(set(payload) - fields)
    if missing or extra:
        raise ValueError(f"{context} fields mismatch: missing={missing}, extra={extra}")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _relative_wav(value: Any, context: str) -> str:
    result = _string(value, context)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".wav":
        raise ValueError(f"{context} must be a safe relative WAV path")
    return result


def _interval(value: Any, context: str) -> Interval:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{context} must be a two-element list")
    start = _number(value[0], f"{context}[0]")
    end = _number(value[1], f"{context}[1]")
    if start < 0.0 or end <= start:
        raise ValueError(f"{context} must satisfy 0 <= start < end")
    return start, end


def _interval_list(value: Any, context: str) -> Tuple[Interval, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return tuple(_interval(item, f"{context}[{i}]") for i, item in enumerate(value))


def _string_list(value: Any, context: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = tuple(_string(item, f"{context}[{i}]") for i, item in enumerate(value))
    if len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicates")
    return result


@dataclass(frozen=True)
class QCESEvent:
    event_id: str
    label: str
    event_kind: str
    source_dataset: str
    source_id: str
    source_path: str
    source_sha256: str
    source_interval_seconds: Interval
    source_crop_interval_seconds: Interval
    onset_seconds: float
    offset_seconds: float
    stem_path: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], context: str) -> "QCESEvent":
        if not isinstance(payload, dict):
            raise TypeError(f"{context} must be an object")
        _exact(payload, _EVENT_FIELDS, context)
        event_kind = _string(payload["event_kind"], f"{context}.event_kind")
        if event_kind not in {"semantic", "nuisance"}:
            raise ValueError(f"{context}.event_kind is invalid")
        source_interval = _interval(
            payload["source_interval_seconds"], f"{context}.source_interval_seconds"
        )
        crop_interval = _interval(
            payload["source_crop_interval_seconds"],
            f"{context}.source_crop_interval_seconds",
        )
        onset = _number(payload["onset_seconds"], f"{context}.onset_seconds")
        offset = _number(payload["offset_seconds"], f"{context}.offset_seconds")
        if onset < 0.0 or offset <= onset:
            raise ValueError(f"{context} has an invalid rendered interval")
        if not math.isclose(
            offset - onset, crop_interval[1] - crop_interval[0], abs_tol=1e-5
        ):
            raise ValueError(f"{context} crop and rendered durations differ")
        source_sha256 = _string(payload["source_sha256"], f"{context}.source_sha256")
        if len(source_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in source_sha256
        ):
            raise ValueError(f"{context}.source_sha256 is invalid")
        return cls(
            event_id=_string(payload["event_id"], f"{context}.event_id"),
            label=_string(payload["label"], f"{context}.label"),
            event_kind=event_kind,
            source_dataset=_string(
                payload["source_dataset"], f"{context}.source_dataset"
            ),
            source_id=_string(payload["source_id"], f"{context}.source_id"),
            source_path=_string(payload["source_path"], f"{context}.source_path"),
            source_sha256=source_sha256,
            source_interval_seconds=source_interval,
            source_crop_interval_seconds=crop_interval,
            onset_seconds=onset,
            offset_seconds=offset,
            stem_path=_relative_wav(payload["stem_path"], f"{context}.stem_path"),
        )

    @property
    def interval(self) -> Interval:
        return self.onset_seconds, self.offset_seconds


@dataclass(frozen=True)
class QCESRecord:
    schema_version: str
    sample_id: str
    scene_id: str
    question_family_id: str
    question_index: int
    split: str
    sample_rate: int
    num_channels: int
    num_samples: int
    duration_seconds: float
    mixture_path: str
    evidence_stem_path: str
    residual_stem_path: str
    anchor_stem_path: str
    answer_stem_path: str
    question: str
    answer: str
    question_type: str
    no_evidence: bool
    absent_label: Optional[str]
    events: Tuple[QCESEvent, ...]
    anchor_event_ids: Tuple[str, ...]
    answer_event_ids: Tuple[str, ...]
    evidence_event_ids: Tuple[str, ...]
    anchor_intervals: Tuple[Interval, ...]
    answer_intervals: Tuple[Interval, ...]
    event_presence_labels: Tuple[str, ...]
    source_group_ids: Tuple[str, ...]
    nuisance_snr_db_requested: float
    nuisance_snr_db: float
    mixture_peak: float
    generation_seed: int

    def event_by_id(self, event_id: str) -> QCESEvent:
        return next(event for event in self.events if event.event_id == event_id)


def parse_qces_record(payload: Dict[str, Any]) -> QCESRecord:
    """Parse and semantically validate one QCES v3 record."""

    if not isinstance(payload, dict):
        raise TypeError("record must be an object")
    _exact(payload, _RECORD_FIELDS, "record")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    events_payload = payload["events"]
    if not isinstance(events_payload, list) or len(events_payload) != 5:
        raise ValueError("events must contain four semantic events and one nuisance")
    events = tuple(
        QCESEvent.from_dict(item, f"events[{i}]")
        for i, item in enumerate(events_payload)
    )
    event_ids = {event.event_id for event in events}
    if len(event_ids) != 5 or len({event.source_id for event in events}) != 5:
        raise ValueError("event and source IDs must be unique within a scene")
    if sum(event.event_kind == "semantic" for event in events) != 4:
        raise ValueError("exactly four semantic events are required")
    if sum(event.event_kind == "nuisance" for event in events) != 1:
        raise ValueError("exactly one nuisance event is required")

    no_evidence = payload["no_evidence"]
    if not isinstance(no_evidence, bool):
        raise TypeError("no_evidence must be boolean")
    absent_label = payload["absent_label"]
    if absent_label is not None:
        absent_label = _string(absent_label, "absent_label")
    anchor_ids = _string_list(payload["anchor_event_ids"], "anchor_event_ids")
    answer_ids = _string_list(payload["answer_event_ids"], "answer_event_ids")
    evidence_ids = _string_list(payload["evidence_event_ids"], "evidence_event_ids")
    if not set(anchor_ids + answer_ids + evidence_ids).issubset(event_ids):
        raise ValueError("role IDs must reference present events")
    anchor_intervals = _interval_list(payload["anchor_intervals"], "anchor_intervals")
    answer_intervals = _interval_list(payload["answer_intervals"], "answer_intervals")
    answer = _string(payload["answer"], "answer")
    question_type = _string(payload["question_type"], "question_type")
    if question_type not in QUESTION_TYPES:
        raise ValueError("unsupported question_type")
    if no_evidence:
        if (
            anchor_ids
            or answer_ids
            or evidence_ids
            or anchor_intervals
            or answer_intervals
        ):
            raise ValueError("no-evidence records must have empty role annotations")
        if answer != NO_EVIDENCE_ANSWER or absent_label is None:
            raise ValueError("no-evidence answer/absent_label mismatch")
        if absent_label in {event.label for event in events}:
            raise ValueError("absent_label is actually present")
    else:
        if len(anchor_ids) != 1 or len(answer_ids) != 1:
            raise ValueError("answerable records need one anchor and one answer")
        if set(evidence_ids) != set(anchor_ids + answer_ids):
            raise ValueError("evidence must be the anchor/answer union")
        if len(anchor_intervals) != 1 or len(answer_intervals) != 1:
            raise ValueError("answerable records need anchor/answer intervals")
        anchor_event = next(
            event for event in events if event.event_id == anchor_ids[0]
        )
        answer_event = next(
            event for event in events if event.event_id == answer_ids[0]
        )
        if (
            anchor_intervals[0] != anchor_event.interval
            or answer_intervals[0] != answer_event.interval
        ):
            raise ValueError("role intervals do not match events")
        if answer != answer_event.label or absent_label is not None:
            raise ValueError("answerable answer/absent_label mismatch")

    sample_rate = _integer(payload["sample_rate"], "sample_rate")
    num_samples = _integer(payload["num_samples"], "num_samples")
    duration = _number(payload["duration_seconds"], "duration_seconds")
    if int(round(sample_rate * duration)) != num_samples:
        raise ValueError("sample count and duration disagree")
    presence = _string_list(payload["event_presence_labels"], "event_presence_labels")
    source_groups = _string_list(payload["source_group_ids"], "source_group_ids")
    if set(presence) != {event.label for event in events}:
        raise ValueError("event_presence_labels mismatch")
    if set(source_groups) != {event.source_id for event in events}:
        raise ValueError("source_group_ids mismatch")
    peak = _number(payload["mixture_peak"], "mixture_peak")
    if not 0.0 < peak <= 1.0:
        raise ValueError("mixture_peak must be in (0, 1]")
    return QCESRecord(
        schema_version=SCHEMA_VERSION,
        sample_id=_string(payload["id"], "id"),
        scene_id=_string(payload["scene_id"], "scene_id"),
        question_family_id=_string(
            payload["question_family_id"], "question_family_id"
        ),
        question_index=_integer(payload["question_index"], "question_index"),
        split=_string(payload["split"], "split"),
        sample_rate=sample_rate,
        num_channels=_integer(payload["num_channels"], "num_channels"),
        num_samples=num_samples,
        duration_seconds=duration,
        mixture_path=_relative_wav(payload["mixture_path"], "mixture_path"),
        evidence_stem_path=_relative_wav(
            payload["evidence_stem_path"], "evidence_stem_path"
        ),
        residual_stem_path=_relative_wav(
            payload["residual_stem_path"], "residual_stem_path"
        ),
        anchor_stem_path=_relative_wav(
            payload["anchor_stem_path"], "anchor_stem_path"
        ),
        answer_stem_path=_relative_wav(
            payload["answer_stem_path"], "answer_stem_path"
        ),
        question=_string(payload["question"], "question"),
        answer=answer,
        question_type=question_type,
        no_evidence=no_evidence,
        absent_label=absent_label,
        events=events,
        anchor_event_ids=anchor_ids,
        answer_event_ids=answer_ids,
        evidence_event_ids=evidence_ids,
        anchor_intervals=anchor_intervals,
        answer_intervals=answer_intervals,
        event_presence_labels=presence,
        source_group_ids=source_groups,
        nuisance_snr_db_requested=_number(
            payload["nuisance_snr_db_requested"], "nuisance_snr_db_requested"
        ),
        nuisance_snr_db=_number(payload["nuisance_snr_db"], "nuisance_snr_db"),
        mixture_peak=peak,
        generation_seed=_integer(payload["generation_seed"], "generation_seed"),
    )
