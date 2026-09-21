"""Strict record-local schema for the paper-scale QCES v5 benchmark.

Unlike v4, v5 deliberately allows repeated event labels and overlapping
semantic events.  Relations are therefore defined over unique *onset order*;
an ordinal selector (for example, the second ``Croak`` occurrence) identifies
the anchor instance.  Dataset-wide counterfactual, split, template and source
isolation invariants are checked by ``validate_qces_v5_dataset.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


# ``qces_v5`` is the original, fully materialized representation.  Keep it
# stable so existing manifests remain readable.  The derived representation is
# explicitly versioned because omitting four per-question WAV paths must never
# be mistaken for a permissive optional-field rule.
SCHEMA_VERSION = "qces_v5"
DERIVED_SCHEMA_VERSION = "qces_v5_scene_event_derived_v1"
MATERIALIZED_STORAGE_MODE = "materialized"
DERIVED_STORAGE_MODE = "scene_event_derived"
SCHEMA_VERSIONS = (SCHEMA_VERSION, DERIVED_SCHEMA_VERSION)
NO_EVIDENCE_ANSWER = "no_evidence"
SPLITS = (
    "train",
    "val",
    "test_iid",
    "test_compositional_ood",
    "test_label_ood",
)
RELATIONS = ("after", "before", "first")
QUESTION_TYPE_TO_RELATION = {
    "temporal_after": "after",
    "temporal_before": "before",
    "temporal_first": "first",
}
NO_EVIDENCE_REASONS = ("absent_anchor", "absent_candidates")
VARIANTS = ("base", "order_swap", "anchor_drop")
EVALUATION_AXIS_BY_SPLIT = {
    "train": "development",
    "val": "development",
    "test_iid": "iid",
    "test_compositional_ood": "compositional_ood",
    "test_label_ood": "label_ood",
}
TEMPLATE_PARTITION_BY_SPLIT = {
    "train": "train",
    "val": "validation",
    "test_iid": "evaluation",
    "test_compositional_ood": "evaluation",
    "test_label_ood": "evaluation",
}
Interval = Tuple[float, float]


_RECORD_FIELDS = {
    "schema_version",
    "id",
    "scene_id",
    "scene_family_id",
    "variant_id",
    "counterfactual_intervention",
    "question_semantics_id",
    "counterfactual_group_id",
    "paraphrase_family_id",
    "template_partition",
    "question_index",
    "split",
    "evaluation_axis",
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
    "absent_labels",
    "query_label",
    "query_instance_ordinal",
    "query_candidate_labels",
    "query_event_ids",
    "surface_control_group_id",
    "mention_order_variant",
    "events",
    "anchor_event_ids",
    "answer_event_ids",
    "evidence_event_ids",
    "anchor_intervals",
    "answer_intervals",
    "source_group_ids",
    "primary_counterfactual_probe",
    "composition_pair_signature",
    "composition_triplet_signature",
    "same_label_repeat",
    "semantic_overlap",
    "max_polyphony",
    "hard_case_tags",
    "render_recipe_id",
    "mixture_peak",
    "family_gain",
    "generation_seed",
}
_MATERIALIZED_STEM_FIELDS = {
    "evidence_stem_path",
    "residual_stem_path",
    "anchor_stem_path",
    "answer_stem_path",
}
_DERIVED_RECORD_FIELDS = (
    _RECORD_FIELDS - _MATERIALIZED_STEM_FIELDS
) | {"storage_mode"}
_EVENT_FIELDS = {
    "event_id",
    "label",
    "event_kind",
    "source_dataset",
    "dataset_version",
    "source_id",
    "creator_id",
    "uploader_id",
    "attribution",
    "source_license_spdx",
    "source_license_url",
    "source_partition",
    "source_path",
    "source_sha256",
    "license_record_id",
    "source_interval_seconds",
    "source_crop_interval_seconds",
    "onset_seconds",
    "offset_seconds",
    "gain_db",
    "occurrence_index",
    "stem_path",
}
_INTERVENTION_FIELDS = {
    "kind",
    "parent_variant_id",
    "intervened_event_ids",
}


def _exact(payload: Mapping[str, Any], fields: set[str], context: str) -> None:
    actual = set(payload)
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


def _optional_string(value: Any, context: str) -> Optional[str]:
    return None if value is None else _string(value, context)


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _relative(value: Any, context: str, suffix: str | None = None) -> str:
    result = _string(value, context)
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context} must be a safe relative path")
    if suffix is not None and path.suffix.lower() != suffix:
        raise ValueError(f"{context} must end with {suffix}")
    return result


def _sha256(value: Any, context: str) -> str:
    result = _string(value, context)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return result


def _string_list(value: Any, context: str, *, unique: bool = True) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = tuple(_string(item, f"{context}[{i}]") for i, item in enumerate(value))
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicates")
    return result


def _interval(value: Any, context: str) -> Interval:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{context} must be a two-element list")
    start = _number(value[0], f"{context}[0]")
    end = _number(value[1], f"{context}[1]")
    if start < 0.0 or end <= start:
        raise ValueError(f"{context} must satisfy 0 <= start < end")
    return start, end


def _intervals(value: Any, context: str) -> Tuple[Interval, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    return tuple(_interval(item, f"{context}[{i}]") for i, item in enumerate(value))


def _same_intervals(left: Sequence[Interval], right: Sequence[Interval]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a[0], b[0], abs_tol=1e-6)
        and math.isclose(a[1], b[1], abs_tol=1e-6)
        for a, b in zip(left, right)
    )


@dataclass(frozen=True)
class QCESV5Intervention:
    kind: str
    parent_variant_id: Optional[str]
    intervened_event_ids: Tuple[str, ...]

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], variant_id: str
    ) -> "QCESV5Intervention":
        if not isinstance(payload, dict):
            raise TypeError("counterfactual_intervention must be an object")
        _exact(payload, _INTERVENTION_FIELDS, "counterfactual_intervention")
        kind = _string(payload["kind"], "counterfactual_intervention.kind")
        parent = _optional_string(
            payload["parent_variant_id"],
            "counterfactual_intervention.parent_variant_id",
        )
        ids = _string_list(
            payload["intervened_event_ids"],
            "counterfactual_intervention.intervened_event_ids",
        )
        expected = {
            "base": ("none", None, 0),
            "order_swap": ("onset_swap", "base", 2),
            "anchor_drop": ("event_drop", "base", 1),
        }[variant_id]
        if (kind, parent, len(ids)) != expected:
            raise ValueError(
                f"intervention does not match variant {variant_id}: "
                f"expected kind/parent/count={expected}"
            )
        return cls(kind=kind, parent_variant_id=parent, intervened_event_ids=ids)


@dataclass(frozen=True)
class QCESV5Event:
    event_id: str
    label: str
    event_kind: str
    source_dataset: str
    dataset_version: str
    source_id: str
    creator_id: str
    uploader_id: str
    attribution: str
    source_license_spdx: str
    source_license_url: str
    source_partition: str
    source_path: str
    source_sha256: str
    license_record_id: str
    source_interval_seconds: Interval
    source_crop_interval_seconds: Interval
    onset_seconds: float
    offset_seconds: float
    gain_db: float
    occurrence_index: int
    stem_path: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], context: str) -> "QCESV5Event":
        if not isinstance(payload, dict):
            raise TypeError(f"{context} must be an object")
        _exact(payload, _EVENT_FIELDS, context)
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
            raise ValueError(f"{context} crop lies outside the source interval")
        onset = _number(payload["onset_seconds"], f"{context}.onset_seconds")
        offset = _number(payload["offset_seconds"], f"{context}.offset_seconds")
        if onset < 0.0 or offset <= onset:
            raise ValueError(f"{context} has an invalid rendered interval")
        if not math.isclose(
            offset - onset, crop_interval[1] - crop_interval[0], abs_tol=5e-5
        ):
            raise ValueError(f"{context} crop/render duration mismatch")
        kind = _string(payload["event_kind"], f"{context}.event_kind")
        if kind not in {"semantic", "nuisance"}:
            raise ValueError(f"{context}.event_kind must be semantic or nuisance")
        partition = _string(
            payload["source_partition"], f"{context}.source_partition"
        )
        if partition not in SPLITS:
            raise ValueError(f"{context}.source_partition is unsupported")
        occurrence = _integer(
            payload["occurrence_index"], f"{context}.occurrence_index"
        )
        if occurrence <= 0:
            raise ValueError(f"{context}.occurrence_index must be positive")
        return cls(
            event_id=_string(payload["event_id"], f"{context}.event_id"),
            label=_string(payload["label"], f"{context}.label"),
            event_kind=kind,
            source_dataset=_string(
                payload["source_dataset"], f"{context}.source_dataset"
            ),
            dataset_version=_string(
                payload["dataset_version"], f"{context}.dataset_version"
            ),
            source_id=_string(payload["source_id"], f"{context}.source_id"),
            creator_id=_string(payload["creator_id"], f"{context}.creator_id"),
            uploader_id=_string(payload["uploader_id"], f"{context}.uploader_id"),
            attribution=_string(payload["attribution"], f"{context}.attribution"),
            source_license_spdx=_string(
                payload["source_license_spdx"], f"{context}.source_license_spdx"
            ),
            source_license_url=_string(
                payload["source_license_url"], f"{context}.source_license_url"
            ),
            source_partition=partition,
            source_path=_relative(payload["source_path"], f"{context}.source_path"),
            source_sha256=_sha256(
                payload["source_sha256"], f"{context}.source_sha256"
            ),
            license_record_id=_string(
                payload["license_record_id"], f"{context}.license_record_id"
            ),
            source_interval_seconds=source_interval,
            source_crop_interval_seconds=crop_interval,
            onset_seconds=onset,
            offset_seconds=offset,
            gain_db=_number(payload["gain_db"], f"{context}.gain_db"),
            occurrence_index=occurrence,
            stem_path=_relative(payload["stem_path"], f"{context}.stem_path", ".wav"),
        )

    @property
    def interval(self) -> Interval:
        return self.onset_seconds, self.offset_seconds


@dataclass(frozen=True)
class QCESV5Record:
    schema_version: str
    storage_mode: str
    sample_id: str
    scene_id: str
    scene_family_id: str
    variant_id: str
    intervention: QCESV5Intervention
    question_semantics_id: str
    counterfactual_group_id: str
    paraphrase_family_id: str
    template_partition: str
    question_index: int
    split: str
    evaluation_axis: str
    sample_rate: int
    num_channels: int
    num_samples: int
    duration_seconds: float
    mixture_path: str
    evidence_stem_path: Optional[str]
    residual_stem_path: Optional[str]
    anchor_stem_path: Optional[str]
    answer_stem_path: Optional[str]
    question: str
    answer: str
    answer_options: Tuple[str, ...]
    answer_option_index: int
    question_type: str
    relation: str
    no_evidence: bool
    no_evidence_reason: Optional[str]
    absent_labels: Tuple[str, ...]
    query_label: Optional[str]
    query_instance_ordinal: Optional[int]
    query_candidate_labels: Tuple[str, ...]
    query_event_ids: Tuple[str, ...]
    surface_control_group_id: Optional[str]
    mention_order_variant: str
    events: Tuple[QCESV5Event, ...]
    anchor_event_ids: Tuple[str, ...]
    answer_event_ids: Tuple[str, ...]
    evidence_event_ids: Tuple[str, ...]
    anchor_intervals: Tuple[Interval, ...]
    answer_intervals: Tuple[Interval, ...]
    source_group_ids: Tuple[str, ...]
    primary_counterfactual_probe: bool
    composition_pair_signature: str
    composition_triplet_signature: str
    same_label_repeat: bool
    semantic_overlap: bool
    max_polyphony: int
    hard_case_tags: Tuple[str, ...]
    render_recipe_id: str
    mixture_peak: float
    family_gain: float
    generation_seed: int

    def event_by_id(self, event_id: str) -> QCESV5Event:
        for event in self.events:
            if event.event_id == event_id:
                return event
        raise KeyError(event_id)


def _max_polyphony(events: Sequence[QCESV5Event]) -> int:
    points = []
    for event in events:
        # End before start at an exact boundary: touching events do not overlap.
        points.append((event.onset_seconds, 1))
        points.append((event.offset_seconds, -1))
    active = maximum = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _semantic_overlap(events: Sequence[QCESV5Event]) -> bool:
    semantic = [event for event in events if event.event_kind == "semantic"]
    return any(
        max(left.onset_seconds, right.onset_seconds)
        < min(left.offset_seconds, right.offset_seconds) - 1e-9
        for i, left in enumerate(semantic)
        for right in semantic[i + 1 :]
    )


def _validate_occurrences(events: Sequence[QCESV5Event]) -> None:
    labels = sorted(set(event.label for event in events))
    for label in labels:
        ordered = sorted(
            (event for event in events if event.label == label),
            key=lambda event: (event.onset_seconds, event.event_id),
        )
        if [event.occurrence_index for event in ordered] != list(
            range(1, len(ordered) + 1)
        ):
            raise ValueError(f"occurrence indices are invalid for label {label!r}")


def _validate_roles(
    *, relation: str, no_evidence: bool, reason: Optional[str],
    absent_labels: Tuple[str, ...], query_label: Optional[str],
    query_ordinal: Optional[int], candidate_labels: Tuple[str, ...],
    query_ids: Tuple[str, ...], anchor_ids: Tuple[str, ...],
    answer_ids: Tuple[str, ...], evidence_ids: Tuple[str, ...],
    anchor_intervals: Tuple[Interval, ...], answer_intervals: Tuple[Interval, ...],
    answer: str, events: Tuple[QCESV5Event, ...],
) -> None:
    event_map = {event.event_id: event for event in events}
    if no_evidence:
        if answer != NO_EVIDENCE_ANSWER:
            raise ValueError("no-evidence answer must be no_evidence")
        if relation in {"after", "before"}:
            if reason != "absent_anchor" or len(absent_labels) != 1:
                raise ValueError("after/before negatives require one absent anchor")
            if query_label != absent_labels[0] or query_ordinal is None:
                raise ValueError("absent query must preserve label and ordinal")
            occurrences = sum(event.label == query_label for event in events)
            if occurrences >= query_ordinal:
                raise ValueError("the declared absent anchor occurrence is present")
            if candidate_labels:
                raise ValueError("after/before negatives cannot have candidates")
        else:
            if reason != "absent_candidates" or len(absent_labels) != 2:
                raise ValueError("first negatives require two absent candidate labels")
            if query_label is not None or query_ordinal is not None:
                raise ValueError("first negatives cannot have an anchor selector")
            if candidate_labels != absent_labels:
                raise ValueError("first absent labels must align with candidate order")
            present = {event.label for event in events}
            if present & set(absent_labels):
                raise ValueError("a declared absent first candidate is present")
        if query_ids or anchor_ids or answer_ids or evidence_ids:
            raise ValueError("no-evidence records must have empty role IDs")
        if anchor_intervals or answer_intervals:
            raise ValueError("no-evidence records must have empty role intervals")
        return

    if reason is not None or absent_labels:
        raise ValueError("answerable records cannot carry no-evidence metadata")
    if len(anchor_ids) != 1 or len(answer_ids) != 1:
        raise ValueError("answerable records need one anchor and one answer event")
    if anchor_ids[0] == answer_ids[0]:
        raise ValueError("anchor and answer events must differ")
    if set(evidence_ids) != set(anchor_ids + answer_ids) or len(evidence_ids) != 2:
        raise ValueError("evidence must equal the two role-event union")
    anchor = event_map[anchor_ids[0]]
    answer_event = event_map[answer_ids[0]]
    if anchor.event_kind != "semantic" or answer_event.event_kind != "semantic":
        raise ValueError("role events must be semantic")
    if answer != answer_event.label:
        raise ValueError("answer must equal the answer-role label")
    if not _same_intervals(anchor_intervals, (anchor.interval,)):
        raise ValueError("anchor_intervals mismatch")
    if not _same_intervals(answer_intervals, (answer_event.interval,)):
        raise ValueError("answer_intervals mismatch")

    timeline = sorted(
        (event for event in events if event.event_kind == "semantic"),
        key=lambda event: (event.onset_seconds, event.event_id),
    )
    if len({event.onset_seconds for event in timeline}) != len(timeline):
        raise ValueError("semantic event onsets must be unique")
    timeline_ids = [event.event_id for event in timeline]
    if relation in {"after", "before"}:
        if candidate_labels:
            raise ValueError("after/before records cannot have candidate labels")
        if query_label != anchor.label or query_ordinal != anchor.occurrence_index:
            raise ValueError("query selector does not identify the anchor instance")
        if query_ids != anchor_ids:
            raise ValueError("query_event_ids must equal anchor_event_ids")
        anchor_index = timeline_ids.index(anchor.event_id)
        answer_index = timeline_ids.index(answer_event.event_id)
        step = 1 if relation == "after" else -1
        if answer_index != anchor_index + step:
            raise ValueError(
                f"{relation} answer must be adjacent by semantic onset order"
            )
        return

    if query_label is not None or query_ordinal is not None:
        raise ValueError("first questions use candidate labels, not one anchor selector")
    if len(candidate_labels) != 2 or len(query_ids) != 2:
        raise ValueError("first questions require two candidate labels/event IDs")
    if tuple(event_map[event_id].label for event_id in query_ids) != candidate_labels:
        raise ValueError("first candidates do not align with query event IDs")
    if set(query_ids) != {anchor.event_id, answer_event.event_id}:
        raise ValueError("first role events must equal the candidate events")
    if answer_event.onset_seconds >= anchor.onset_seconds:
        raise ValueError("first answer must begin before the other candidate")


def parse_qces_v5_record(payload: Dict[str, Any]) -> QCESV5Record:
    if not isinstance(payload, dict):
        raise TypeError("record must be an object")
    schema = _string(payload["schema_version"], "schema_version")
    if schema == SCHEMA_VERSION:
        _exact(payload, _RECORD_FIELDS, "record")
        storage_mode = MATERIALIZED_STORAGE_MODE
    elif schema == DERIVED_SCHEMA_VERSION:
        _exact(payload, _DERIVED_RECORD_FIELDS, "record")
        storage_mode = _string(payload["storage_mode"], "storage_mode")
        if storage_mode != DERIVED_STORAGE_MODE:
            raise ValueError(
                f"{DERIVED_SCHEMA_VERSION} requires storage_mode="
                f"{DERIVED_STORAGE_MODE}"
            )
    else:
        raise ValueError(f"schema_version must be one of {SCHEMA_VERSIONS}")
    split = _string(payload["split"], "split")
    if split not in SPLITS:
        raise ValueError("unsupported split")
    sample_id = _string(payload["id"], "id")
    scene_id = _string(payload["scene_id"], "scene_id")
    family_id = _string(payload["scene_family_id"], "scene_family_id")
    if not sample_id.startswith(f"{split}_") or not scene_id.startswith(f"scene_{split}_"):
        raise ValueError("sample/scene IDs must expose their split")
    if not family_id.startswith(f"family_{split}_"):
        raise ValueError("scene_family_id must expose its split")
    variant = _string(payload["variant_id"], "variant_id")
    if variant not in VARIANTS:
        raise ValueError("unsupported counterfactual variant")
    intervention = QCESV5Intervention.from_dict(
        payload["counterfactual_intervention"], variant
    )

    evaluation_axis = _string(payload["evaluation_axis"], "evaluation_axis")
    if evaluation_axis != EVALUATION_AXIS_BY_SPLIT[split]:
        raise ValueError("evaluation_axis does not match split")
    template_partition = _string(
        payload["template_partition"], "template_partition"
    )
    if template_partition != TEMPLATE_PARTITION_BY_SPLIT[split]:
        raise ValueError("template_partition does not match split")

    sample_rate = _integer(payload["sample_rate"], "sample_rate")
    channels = _integer(payload["num_channels"], "num_channels")
    num_samples = _integer(payload["num_samples"], "num_samples")
    duration = _number(payload["duration_seconds"], "duration_seconds")
    if sample_rate <= 0 or channels != 1 or num_samples <= 0 or duration <= 0:
        raise ValueError("invalid mono audio dimensions")
    if int(round(sample_rate * duration)) != num_samples:
        raise ValueError("duration and sample count disagree")

    event_payloads = payload["events"]
    if not isinstance(event_payloads, list):
        raise TypeError("events must be a list")
    events = tuple(
        QCESV5Event.from_dict(item, f"events[{i}]")
        for i, item in enumerate(event_payloads)
    )
    if sum(event.event_kind == "semantic" for event in events) < 4:
        raise ValueError("v5 requires at least four semantic events")
    ids = tuple(event.event_id for event in events)
    if len(ids) != len(set(ids)):
        raise ValueError("event IDs must be unique")
    if any(event.offset_seconds > duration + 1e-6 for event in events):
        raise ValueError("an event exceeds scene duration")
    if any(event.source_partition != split for event in events):
        raise ValueError("event source partition does not match record split")
    _validate_occurrences(events)
    event_ids = set(ids)

    relation = _string(payload["relation"], "relation")
    question_type = _string(payload["question_type"], "question_type")
    if relation not in RELATIONS or QUESTION_TYPE_TO_RELATION.get(question_type) != relation:
        raise ValueError("question_type/relation mismatch")
    no_evidence = payload["no_evidence"]
    if not isinstance(no_evidence, bool):
        raise TypeError("no_evidence must be boolean")
    reason = _optional_string(payload["no_evidence_reason"], "no_evidence_reason")
    if reason is not None and reason not in NO_EVIDENCE_REASONS:
        raise ValueError("unsupported no_evidence_reason")
    absent_labels = _string_list(payload["absent_labels"], "absent_labels")
    query_label = _optional_string(payload["query_label"], "query_label")
    query_ordinal = payload["query_instance_ordinal"]
    if query_ordinal is not None:
        query_ordinal = _integer(query_ordinal, "query_instance_ordinal")
        if query_ordinal <= 0:
            raise ValueError("query_instance_ordinal must be positive")
    candidate_labels = _string_list(
        payload["query_candidate_labels"], "query_candidate_labels"
    )
    query_ids = _string_list(payload["query_event_ids"], "query_event_ids")
    surface_control_group_id = _optional_string(
        payload["surface_control_group_id"], "surface_control_group_id"
    )
    mention_order_variant = _string(
        payload["mention_order_variant"], "mention_order_variant"
    )
    if relation == "first":
        if surface_control_group_id is None or mention_order_variant not in {
            "forward",
            "reversed",
        }:
            raise ValueError("first questions require a paired mention-order control")
    elif surface_control_group_id is not None or mention_order_variant != "not_applicable":
        raise ValueError("after/before questions cannot carry mention-order controls")
    anchor_ids = _string_list(payload["anchor_event_ids"], "anchor_event_ids")
    answer_ids = _string_list(payload["answer_event_ids"], "answer_event_ids")
    evidence_ids = _string_list(payload["evidence_event_ids"], "evidence_event_ids")
    if not set(query_ids + anchor_ids + answer_ids + evidence_ids).issubset(event_ids):
        raise ValueError("role IDs reference absent events")
    anchor_intervals = _intervals(payload["anchor_intervals"], "anchor_intervals")
    answer_intervals = _intervals(payload["answer_intervals"], "answer_intervals")

    answer = _string(payload["answer"], "answer")
    options = _string_list(payload["answer_options"], "answer_options")
    if len(options) != 5 or NO_EVIDENCE_ANSWER not in options:
        raise ValueError("answer_options require five choices including no_evidence")
    if relation == "first" and not set(candidate_labels).issubset(options):
        raise ValueError("first answer options must expose both named candidates")
    answer_index = _integer(payload["answer_option_index"], "answer_option_index")
    if not 0 <= answer_index < 5 or options[answer_index] != answer:
        raise ValueError("answer_option_index does not select the answer")
    _validate_roles(
        relation=relation,
        no_evidence=no_evidence,
        reason=reason,
        absent_labels=absent_labels,
        query_label=query_label,
        query_ordinal=query_ordinal,
        candidate_labels=candidate_labels,
        query_ids=query_ids,
        anchor_ids=anchor_ids,
        answer_ids=answer_ids,
        evidence_ids=evidence_ids,
        anchor_intervals=anchor_intervals,
        answer_intervals=answer_intervals,
        answer=answer,
        events=events,
    )

    source_groups = _string_list(payload["source_group_ids"], "source_group_ids")
    if set(source_groups) != {event.source_id for event in events}:
        raise ValueError("source_group_ids must equal the unique event sources")
    repeated = payload["same_label_repeat"]
    overlap = payload["semantic_overlap"]
    primary = payload["primary_counterfactual_probe"]
    for value, name in (
        (repeated, "same_label_repeat"),
        (overlap, "semantic_overlap"),
        (primary, "primary_counterfactual_probe"),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean")
    semantic_labels = [event.label for event in events if event.event_kind == "semantic"]
    if repeated != (len(semantic_labels) != len(set(semantic_labels))):
        raise ValueError("same_label_repeat flag mismatch")
    if overlap != _semantic_overlap(events):
        raise ValueError("semantic_overlap flag mismatch")
    polyphony = _integer(payload["max_polyphony"], "max_polyphony")
    if polyphony != _max_polyphony(events):
        raise ValueError("max_polyphony mismatch")
    tags = _string_list(payload["hard_case_tags"], "hard_case_tags")

    mixture_peak = _number(payload["mixture_peak"], "mixture_peak")
    family_gain = _number(payload["family_gain"], "family_gain")
    generation_seed = _integer(payload["generation_seed"], "generation_seed")
    question_index = _integer(payload["question_index"], "question_index")
    if not 0 < mixture_peak <= 1 or not 0 < family_gain <= 1:
        raise ValueError("mixture_peak/family_gain must be in (0, 1]")
    if generation_seed < 0 or question_index < 0:
        raise ValueError("seed and question index must be non-negative")

    return QCESV5Record(
        schema_version=schema,
        storage_mode=storage_mode,
        sample_id=sample_id,
        scene_id=scene_id,
        scene_family_id=family_id,
        variant_id=variant,
        intervention=intervention,
        question_semantics_id=_string(
            payload["question_semantics_id"], "question_semantics_id"
        ),
        counterfactual_group_id=_string(
            payload["counterfactual_group_id"], "counterfactual_group_id"
        ),
        paraphrase_family_id=_string(
            payload["paraphrase_family_id"], "paraphrase_family_id"
        ),
        template_partition=template_partition,
        question_index=question_index,
        split=split,
        evaluation_axis=evaluation_axis,
        sample_rate=sample_rate,
        num_channels=channels,
        num_samples=num_samples,
        duration_seconds=duration,
        mixture_path=_relative(payload["mixture_path"], "mixture_path", ".wav"),
        evidence_stem_path=(
            _relative(payload["evidence_stem_path"], "evidence_stem_path", ".wav")
            if storage_mode == MATERIALIZED_STORAGE_MODE
            else None
        ),
        residual_stem_path=(
            _relative(payload["residual_stem_path"], "residual_stem_path", ".wav")
            if storage_mode == MATERIALIZED_STORAGE_MODE
            else None
        ),
        anchor_stem_path=(
            _relative(payload["anchor_stem_path"], "anchor_stem_path", ".wav")
            if storage_mode == MATERIALIZED_STORAGE_MODE
            else None
        ),
        answer_stem_path=(
            _relative(payload["answer_stem_path"], "answer_stem_path", ".wav")
            if storage_mode == MATERIALIZED_STORAGE_MODE
            else None
        ),
        question=_string(payload["question"], "question"),
        answer=answer,
        answer_options=options,
        answer_option_index=answer_index,
        question_type=question_type,
        relation=relation,
        no_evidence=no_evidence,
        no_evidence_reason=reason,
        absent_labels=absent_labels,
        query_label=query_label,
        query_instance_ordinal=query_ordinal,
        query_candidate_labels=candidate_labels,
        query_event_ids=query_ids,
        surface_control_group_id=surface_control_group_id,
        mention_order_variant=mention_order_variant,
        events=events,
        anchor_event_ids=anchor_ids,
        answer_event_ids=answer_ids,
        evidence_event_ids=evidence_ids,
        anchor_intervals=anchor_intervals,
        answer_intervals=answer_intervals,
        source_group_ids=source_groups,
        primary_counterfactual_probe=primary,
        composition_pair_signature=_string(
            payload["composition_pair_signature"], "composition_pair_signature"
        ),
        composition_triplet_signature=_string(
            payload["composition_triplet_signature"], "composition_triplet_signature"
        ),
        same_label_repeat=repeated,
        semantic_overlap=overlap,
        max_polyphony=polyphony,
        hard_case_tags=tags,
        render_recipe_id=_string(payload["render_recipe_id"], "render_recipe_id"),
        mixture_peak=mixture_peak,
        family_gain=family_gain,
        generation_seed=generation_seed,
    )


parse_record = parse_qces_v5_record


__all__ = [
    "DERIVED_SCHEMA_VERSION",
    "DERIVED_STORAGE_MODE",
    "EVALUATION_AXIS_BY_SPLIT",
    "MATERIALIZED_STORAGE_MODE",
    "NO_EVIDENCE_ANSWER",
    "NO_EVIDENCE_REASONS",
    "QCESV5Event",
    "QCESV5Intervention",
    "QCESV5Record",
    "RELATIONS",
    "SCHEMA_VERSION",
    "SCHEMA_VERSIONS",
    "SPLITS",
    "TEMPLATE_PARTITION_BY_SPLIT",
    "VARIANTS",
    "parse_qces_v5_record",
    "parse_record",
]
