"""Strict record-local schema for the QCES v4 benchmark.

Cross-record invariants (for example, the membership and coverage of a
counterfactual group) belong in the dataset validator.  This module validates
everything that can be proved from one manifest record without opening audio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "qces_v4"
NO_EVIDENCE_ANSWER = "no_evidence"
RELATIONS = ("after", "before", "first")
NO_EVIDENCE_REASONS = ("absent_anchor",)
QUESTION_TYPE_TO_RELATION = {
    "temporal_after": "after",
    "temporal_before": "before",
    "temporal_first": "first",
    "no_evidence_after": "after",
    "no_evidence_before": "before",
}
Interval = Tuple[float, float]

_RECORD_FIELDS = {
    "schema_version",
    "id",
    "scene_id",
    "question_family_id",
    "counterfactual_group_id",
    "paraphrase_family_id",
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
    "answer_options",
    "answer_option_index",
    "question_type",
    "relation",
    "no_evidence",
    "no_evidence_reason",
    "absent_label",
    "query_labels",
    "query_event_ids",
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
    """Reject missing and unknown keys instead of silently accepting drift."""

    actual = set(payload.keys())
    missing = sorted(fields - actual)
    extra = sorted(actual - fields, key=repr)
    if missing or extra:
        raise ValueError(
            f"{context} fields mismatch: missing={missing}, extra={extra}"
        )


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{context} must not have surrounding whitespace")
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


def _relative_path(value: Any, context: str) -> str:
    result = _string(value, context)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must be a safe relative path")
    return result


def _relative_wav(value: Any, context: str) -> str:
    result = _relative_path(value, context)
    if PurePosixPath(result).suffix.lower() != ".wav":
        raise ValueError(f"{context} must be a WAV path")
    return result


def _sha256(value: Any, context: str) -> str:
    result = _string(value, context)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
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
    return tuple(
        _interval(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )


def _string_list(value: Any, context: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = tuple(
        _string(item, f"{context}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicates")
    return result


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


@dataclass(frozen=True)
class QCESEvent:
    """One rendered event together with its immutable source provenance."""

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
            raise ValueError(f"{context}.event_kind must be semantic or nuisance")
        source_interval = _interval(
            payload["source_interval_seconds"], f"{context}.source_interval_seconds"
        )
        crop_interval = _interval(
            payload["source_crop_interval_seconds"],
            f"{context}.source_crop_interval_seconds",
        )
        if (
            crop_interval[0] < source_interval[0] - 1e-6
            or crop_interval[1] > source_interval[1] + 1e-6
        ):
            raise ValueError(f"{context} crop must lie within its source interval")

        onset = _number(payload["onset_seconds"], f"{context}.onset_seconds")
        offset = _number(payload["offset_seconds"], f"{context}.offset_seconds")
        if onset < 0.0 or offset <= onset:
            raise ValueError(f"{context} has an invalid rendered interval")
        if not math.isclose(
            offset - onset, crop_interval[1] - crop_interval[0], abs_tol=1e-5
        ):
            raise ValueError(f"{context} crop and rendered durations differ")

        return cls(
            event_id=_string(payload["event_id"], f"{context}.event_id"),
            label=_string(payload["label"], f"{context}.label"),
            event_kind=event_kind,
            source_dataset=_string(
                payload["source_dataset"], f"{context}.source_dataset"
            ),
            source_id=_string(payload["source_id"], f"{context}.source_id"),
            source_path=_relative_path(
                payload["source_path"], f"{context}.source_path"
            ),
            source_sha256=_sha256(
                payload["source_sha256"], f"{context}.source_sha256"
            ),
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
class QCESV4Record:
    """A fully validated QCES v4 manifest record."""

    schema_version: str
    sample_id: str
    scene_id: str
    question_family_id: str
    counterfactual_group_id: str
    paraphrase_family_id: str
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
    answer_options: Tuple[str, ...]
    answer_option_index: int
    question_type: str
    relation: str
    no_evidence: bool
    no_evidence_reason: Optional[str]
    absent_label: Optional[str]
    query_labels: Tuple[str, ...]
    query_event_ids: Tuple[str, ...]
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
        for event in self.events:
            if event.event_id == event_id:
                return event
        raise KeyError(event_id)


def _validate_answerable_relations(
    *,
    relation: str,
    events: Tuple[QCESEvent, ...],
    anchor_ids: Tuple[str, ...],
    answer_ids: Tuple[str, ...],
    evidence_ids: Tuple[str, ...],
    anchor_intervals: Tuple[Interval, ...],
    answer_intervals: Tuple[Interval, ...],
    query_labels: Tuple[str, ...],
    query_ids: Tuple[str, ...],
    answer: str,
) -> None:
    if len(anchor_ids) != 1 or len(answer_ids) != 1:
        raise ValueError("answerable records need exactly one anchor and one answer")
    if anchor_ids[0] == answer_ids[0]:
        raise ValueError("anchor and answer must be different events")
    if set(evidence_ids) != set(anchor_ids + answer_ids) or len(evidence_ids) != 2:
        raise ValueError("evidence must be exactly the anchor/answer union")

    event_map = {event.event_id: event for event in events}
    anchor = event_map[anchor_ids[0]]
    answer_event = event_map[answer_ids[0]]
    if anchor.event_kind != "semantic" or answer_event.event_kind != "semantic":
        raise ValueError("anchor and answer must reference semantic events")
    if not _interval_lists_match(anchor_intervals, (anchor.interval,)):
        raise ValueError("anchor_intervals do not match the anchor event")
    if not _interval_lists_match(answer_intervals, (answer_event.interval,)):
        raise ValueError("answer_intervals do not match the answer event")
    if answer != answer_event.label:
        raise ValueError("answer must equal the answer-role event label")

    semantic_timeline = sorted(
        (event for event in events if event.event_kind == "semantic"),
        key=lambda event: (event.onset_seconds, event.offset_seconds, event.event_id),
    )
    semantic_onsets = [event.onset_seconds for event in semantic_timeline]
    if len(set(semantic_onsets)) != len(semantic_onsets):
        raise ValueError("semantic events must have unique onsets")
    timeline_ids = [event.event_id for event in semantic_timeline]

    if relation in {"after", "before"}:
        if query_ids != anchor_ids or query_labels != (anchor.label,):
            raise ValueError(
                "after/before query fields must identify exactly the anchor event"
            )
        anchor_index = timeline_ids.index(anchor.event_id)
        answer_index = timeline_ids.index(answer_event.event_id)
        if relation == "after":
            if answer_index != anchor_index + 1:
                raise ValueError("after answer must immediately follow the anchor")
            if anchor.offset_seconds > answer_event.onset_seconds + 1e-6:
                raise ValueError("after anchor and answer must not overlap")
        else:
            if answer_index != anchor_index - 1:
                raise ValueError("before answer must immediately precede the anchor")
            if answer_event.offset_seconds > anchor.onset_seconds + 1e-6:
                raise ValueError("before answer and anchor must not overlap")
        return

    if len(query_ids) != 2 or set(query_ids) != {
        anchor.event_id,
        answer_event.event_id,
    }:
        raise ValueError("first queries must contain exactly the two role events")
    expected_query_labels = tuple(event_map[event_id].label for event_id in query_ids)
    if query_labels != expected_query_labels:
        raise ValueError("query_labels must align positionally with query_event_ids")
    if answer_event.onset_seconds >= anchor.onset_seconds:
        raise ValueError("first answer must start before the other candidate")


def parse_qces_v4_record(payload: Dict[str, Any]) -> QCESV4Record:
    """Parse and semantically validate one JSON-compatible QCES v4 record."""

    if not isinstance(payload, dict):
        raise TypeError("record must be an object")
    _exact(payload, _RECORD_FIELDS, "record")

    schema_version = _string(payload["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")

    sample_id = _string(payload["id"], "id")
    scene_id = _string(payload["scene_id"], "scene_id")
    split = _string(payload["split"], "split")
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if not sample_id.startswith(f"{split}_"):
        raise ValueError("id must start with its split name")
    if not scene_id.startswith("scene_"):
        raise ValueError("scene_id must start with 'scene_'")

    sample_rate = _integer(payload["sample_rate"], "sample_rate")
    num_channels = _integer(payload["num_channels"], "num_channels")
    num_samples = _integer(payload["num_samples"], "num_samples")
    duration = _number(payload["duration_seconds"], "duration_seconds")
    if sample_rate <= 0 or num_samples <= 0 or duration <= 0.0:
        raise ValueError("audio dimensions must be positive")
    if num_channels != 1:
        raise ValueError("QCES v4 currently requires mono audio")
    if int(round(sample_rate * duration)) != num_samples:
        raise ValueError("sample count and duration disagree")

    events_payload = payload["events"]
    if not isinstance(events_payload, list):
        raise TypeError("events must be a list")
    events = tuple(
        QCESEvent.from_dict(item, f"events[{index}]")
        for index, item in enumerate(events_payload)
    )
    if sum(event.event_kind == "semantic" for event in events) < 3:
        raise ValueError("at least three semantic events are required")
    if sum(event.event_kind == "nuisance" for event in events) < 1:
        raise ValueError("at least one nuisance event is required")
    event_ids = tuple(event.event_id for event in events)
    event_labels = tuple(event.label for event in events)
    source_ids = tuple(event.source_id for event in events)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("event IDs must be unique within a scene")
    if len(set(event_labels)) != len(event_labels):
        raise ValueError("event labels must be unique within a scene")
    if NO_EVIDENCE_ANSWER in event_labels:
        raise ValueError("no_evidence is reserved and cannot be an event label")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source IDs must be unique within a scene")
    if any(event.offset_seconds > duration + 1e-6 for event in events):
        raise ValueError("an event exceeds the audio duration")
    event_id_set = set(event_ids)

    no_evidence = payload["no_evidence"]
    if not isinstance(no_evidence, bool):
        raise TypeError("no_evidence must be boolean")
    reason = payload["no_evidence_reason"]
    if reason is not None:
        reason = _string(reason, "no_evidence_reason")
        if reason not in NO_EVIDENCE_REASONS:
            raise ValueError("no_evidence_reason must be absent_anchor or null")
    absent_label = payload["absent_label"]
    if absent_label is not None:
        absent_label = _string(absent_label, "absent_label")
        if absent_label == NO_EVIDENCE_ANSWER:
            raise ValueError("absent_label cannot be the reserved no_evidence token")

    relation = _string(payload["relation"], "relation")
    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {RELATIONS}")
    question_type = _string(payload["question_type"], "question_type")
    if question_type not in QUESTION_TYPE_TO_RELATION:
        raise ValueError("unsupported question_type")
    if QUESTION_TYPE_TO_RELATION[question_type] != relation:
        raise ValueError("question_type and relation disagree")

    query_labels = _string_list(payload["query_labels"], "query_labels")
    query_ids = _string_list(payload["query_event_ids"], "query_event_ids")
    anchor_ids = _string_list(payload["anchor_event_ids"], "anchor_event_ids")
    answer_ids = _string_list(payload["answer_event_ids"], "answer_event_ids")
    evidence_ids = _string_list(payload["evidence_event_ids"], "evidence_event_ids")
    referenced_ids = set(query_ids + anchor_ids + answer_ids + evidence_ids)
    if not referenced_ids.issubset(event_id_set):
        raise ValueError("query and role IDs must reference present events")
    anchor_intervals = _interval_list(payload["anchor_intervals"], "anchor_intervals")
    answer_intervals = _interval_list(payload["answer_intervals"], "answer_intervals")

    answer = _string(payload["answer"], "answer")
    answer_options = _string_list(payload["answer_options"], "answer_options")
    if len(answer_options) != 5 or NO_EVIDENCE_ANSWER not in answer_options:
        raise ValueError(
            "answer_options must contain exactly five unique choices "
            "including no_evidence"
        )
    answer_option_index = _integer(
        payload["answer_option_index"], "answer_option_index"
    )
    if not 0 <= answer_option_index < len(answer_options):
        raise ValueError("answer_option_index is out of range")
    if answer_options[answer_option_index] != answer:
        raise ValueError("answer_option_index does not select answer")

    if no_evidence:
        if question_type not in {"no_evidence_after", "no_evidence_before"}:
            raise ValueError("no-evidence question_type must encode no evidence")
        if relation == "first":
            raise ValueError("absent-anchor no-evidence is undefined for first")
        if reason != "absent_anchor" or absent_label is None:
            raise ValueError("no-evidence records require an absent anchor label")
        if answer != NO_EVIDENCE_ANSWER:
            raise ValueError("no-evidence answer must be no_evidence")
        if query_labels != (absent_label,) or query_ids:
            raise ValueError(
                "absent-anchor query must retain its label and have no event IDs"
            )
        if absent_label in set(event_labels):
            raise ValueError("the queried absent anchor is actually present")
        if (
            anchor_ids
            or answer_ids
            or evidence_ids
            or anchor_intervals
            or answer_intervals
        ):
            raise ValueError("no-evidence records must have empty role annotations")
    else:
        if question_type.startswith("no_evidence_"):
            raise ValueError("answerable question_type cannot encode no evidence")
        if reason is not None or absent_label is not None:
            raise ValueError("answerable records cannot have no-evidence metadata")
        _validate_answerable_relations(
            relation=relation,
            events=events,
            anchor_ids=anchor_ids,
            answer_ids=answer_ids,
            evidence_ids=evidence_ids,
            anchor_intervals=anchor_intervals,
            answer_intervals=answer_intervals,
            query_labels=query_labels,
            query_ids=query_ids,
            answer=answer,
        )

    presence = _string_list(payload["event_presence_labels"], "event_presence_labels")
    if set(presence) != set(event_labels) or len(presence) != len(events):
        raise ValueError("event_presence_labels must exactly match rendered events")
    source_groups = _string_list(payload["source_group_ids"], "source_group_ids")
    if set(source_groups) != set(source_ids) or len(source_groups) != len(events):
        raise ValueError("source_group_ids must exactly match event source IDs")

    peak = _number(payload["mixture_peak"], "mixture_peak")
    if not 0.0 < peak <= 1.0:
        raise ValueError("mixture_peak must be in (0, 1]")
    generation_seed = _integer(payload["generation_seed"], "generation_seed")
    question_index = _integer(payload["question_index"], "question_index")
    if generation_seed < 0 or question_index < 0:
        raise ValueError("generation_seed and question_index must be non-negative")

    return QCESV4Record(
        schema_version=schema_version,
        sample_id=sample_id,
        scene_id=scene_id,
        question_family_id=_string(
            payload["question_family_id"], "question_family_id"
        ),
        counterfactual_group_id=_string(
            payload["counterfactual_group_id"], "counterfactual_group_id"
        ),
        paraphrase_family_id=_string(
            payload["paraphrase_family_id"], "paraphrase_family_id"
        ),
        question_index=question_index,
        split=split,
        sample_rate=sample_rate,
        num_channels=num_channels,
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
        answer_options=answer_options,
        answer_option_index=answer_option_index,
        question_type=question_type,
        relation=relation,
        no_evidence=no_evidence,
        no_evidence_reason=reason,
        absent_label=absent_label,
        query_labels=query_labels,
        query_event_ids=query_ids,
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
        generation_seed=generation_seed,
    )


# A short alias keeps generic manifest consumers ergonomic without weakening the
# version-specific public name used by benchmark code.
parse_record = parse_qces_v4_record


__all__ = [
    "NO_EVIDENCE_ANSWER",
    "NO_EVIDENCE_REASONS",
    "QCESEvent",
    "QCESV4Record",
    "RELATIONS",
    "SCHEMA_VERSION",
    "parse_qces_v4_record",
    "parse_record",
]
