"""Deterministic contracts for deriving QCES questions from RealDESED.

RealDESED supplies real mixtures and reviewed temporal event annotations, not
isolated sources.  The contracts in this module therefore permit temporal and
model-behaviour evaluation while explicitly forbidding clean-stem waveform
metrics such as SI-SDR and SD-SDR.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCENE_SCHEMA_VERSION = "qces_realdesed_scene_v1"
INFERENCE_SCHEMA_VERSION = "qces_realdesed_inference_v1"
SCORING_SCHEMA_VERSION = "qces_realdesed_scoring_v1"
INFERENCE_FINGERPRINT_FORMAT = "qces_realdesed_inference_fingerprint_v1"

DATASET_NAME = "RealDESED"
DATASET_RECORD_ID = "20056072"
SAMPLE_RATE = 32_000
NUM_CHANNELS = 1
DURATION_SECONDS = 10.0
NUM_SAMPLES = SAMPLE_RATE * 10
NO_EVIDENCE_ANSWER = "no_evidence"

REALDESED_CLASSES = (
    "bell_ringing",
    "coffee_machine",
    "cutlery_dishes",
    "door_open_close",
    "footsteps",
    "keyboard_typing",
    "keychain",
    "light_switch",
    "microwave",
    "phone_ringing",
    "running_water",
    "toilet_flushing",
    "vacuum_cleaner",
    "wardrobe_drawer_open_close",
    "window_open_close",
)

DISPLAY_NAME = {
    "bell_ringing": "bell ringing",
    "coffee_machine": "coffee machine",
    "cutlery_dishes": "cutlery and dishes",
    "door_open_close": "door opening or closing",
    "footsteps": "footsteps",
    "keyboard_typing": "keyboard typing",
    "keychain": "keys or a keychain",
    "light_switch": "light switch",
    "microwave": "microwave",
    "phone_ringing": "phone ringing or notification",
    "running_water": "running water",
    "toilet_flushing": "toilet flushing",
    "vacuum_cleaner": "vacuum cleaner",
    "wardrobe_drawer_open_close": "wardrobe or drawer opening or closing",
    "window_open_close": "window opening or closing",
}

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, order=True)
class Event:
    onset: float
    offset: float
    label: str
    event_id: str = ""

    def __post_init__(self) -> None:
        if self.label not in REALDESED_CLASSES:
            raise ValueError(f"unknown RealDESED class: {self.label!r}")
        if not (
            math.isfinite(self.onset)
            and math.isfinite(self.offset)
            and 0.0 <= self.onset < self.offset
        ):
            raise ValueError("event interval must be finite and positive")

    @property
    def interval(self) -> list[float]:
        return [round(self.onset, 6), round(self.offset, 6)]


def merge_reviewed_events(
    events: Iterable[Event], *, maximum_gap_seconds: float = 0.08
) -> tuple[Event, ...]:
    """Deduplicate and merge touching same-class reviewed regions.

    The operation never merges different classes.  A short gap is allowed to
    collapse boundary jitter or one event split into adjacent annotation
    regions.  Every resulting instance receives a stable onset-ordered ID.
    """

    if maximum_gap_seconds < 0:
        raise ValueError("maximum_gap_seconds must be non-negative")
    unique = {
        (event.label, round(event.onset, 6), round(event.offset, 6)): event
        for event in events
    }
    by_label: dict[str, list[Event]] = {}
    for event in unique.values():
        by_label.setdefault(event.label, []).append(event)
    merged = []
    for label, label_events in by_label.items():
        current: Event | None = None
        for event in sorted(label_events):
            if current is None:
                current = event
            elif event.onset <= current.offset + maximum_gap_seconds:
                current = Event(
                    onset=current.onset,
                    offset=max(current.offset, event.offset),
                    label=label,
                )
            else:
                merged.append(current)
                current = event
        if current is not None:
            merged.append(current)
    ordered = sorted(merged, key=lambda item: (item.onset, item.offset, item.label))
    return tuple(
        Event(
            onset=event.onset,
            offset=event.offset,
            label=event.label,
            event_id=f"event_{index:03d}",
        )
        for index, event in enumerate(ordered)
    )


def choose_ten_second_crop(
    events: Sequence[Event],
    *,
    source_duration_seconds: float,
    boundary_margin_seconds: float = 0.05,
    minimum_visible_duration_seconds: float = 0.05,
) -> tuple[float, tuple[Event, ...]]:
    """Choose one deterministic event-rich 10-second crop.

    Event onsets must lie inside the crop with a small boundary margin, because
    the relational questions concern onset order.  An event continuing beyond
    the right crop edge is clipped at that edge (right-censored); events whose
    onset precedes the crop are never retained.  The objective first maximizes
    distinct classes, then retained instances, then relational adjacent pairs,
    and finally prefers the earlier crop.  It does not inspect waveform energy
    or model predictions.
    """

    if source_duration_seconds < DURATION_SECONDS:
        raise ValueError("source recording is shorter than ten seconds")
    if minimum_visible_duration_seconds <= 0:
        raise ValueError("minimum_visible_duration_seconds must be positive")
    maximum_start = source_duration_seconds - DURATION_SECONDS

    def clamp(value: float) -> float:
        return min(maximum_start, max(0.0, value))

    candidates = {0.0, maximum_start}
    for event in events:
        candidates.add(clamp(event.onset - boundary_margin_seconds))
        candidates.add(
            clamp(event.offset + boundary_margin_seconds - DURATION_SECONDS)
        )
        candidates.add(clamp((event.onset + event.offset - DURATION_SECONDS) / 2))

    best: tuple[tuple[int, int, int, float], float, tuple[Event, ...]] | None = None
    for raw_start in sorted(candidates):
        start = round(raw_start, 6)
        end = start + DURATION_SECONDS
        onset_visible_absolute = tuple(
            event
            for event in events
            if event.onset >= start + boundary_margin_seconds
            and min(event.offset, end - boundary_margin_seconds) - event.onset
            >= minimum_visible_duration_seconds
        )
        label_counts: dict[str, int] = {}
        for event in onset_visible_absolute:
            label_counts[event.label] = label_counts.get(event.label, 0) + 1
        unique_instances = tuple(
            event
            for event in onset_visible_absolute
            if label_counts[event.label] == 1
        )
        adjacent_pairs = sum(
            first.label != second.label
            for first, second in zip(unique_instances, unique_instances[1:])
        )
        score = (
            len(label_counts),
            len(onset_visible_absolute),
            adjacent_pairs,
            -start,
        )
        shifted = tuple(
            Event(
                onset=event.onset - start,
                offset=min(event.offset, end - boundary_margin_seconds) - start,
                label=event.label,
                event_id=event.event_id,
            )
            for event in onset_visible_absolute
        )
        candidate = (score, start, shifted)
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    return best[1], best[2]


def _safe_identifier(value: str, context: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{context} is not a safe identifier: {value!r}")
    return value


def _safe_audio_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("mixture_path must use POSIX separators")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.suffix.casefold() != ".wav"
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise ValueError("mixture_path must be a safe relative WAV path")
    return value


def _validate_common(payload: Mapping[str, Any]) -> None:
    for field in ("id", "scene_id", "scene_family_id"):
        value = payload.get(field)
        if not isinstance(value, str):
            raise TypeError(f"{field} must be a string")
        _safe_identifier(value, field)
    if payload.get("dataset") != DATASET_NAME:
        raise ValueError(f"dataset must equal {DATASET_NAME!r}")
    if payload.get("dataset_record_id") != DATASET_RECORD_ID:
        raise ValueError(f"dataset_record_id must equal {DATASET_RECORD_ID!r}")
    if payload.get("upstream_split") not in {"validation", "test"}:
        raise ValueError("upstream_split must be validation or test")
    expected_split = {
        "validation": "real_dev",
        "test": "real_test",
    }[str(payload["upstream_split"])]
    if payload.get("split") != expected_split:
        raise ValueError("upstream_split/QCES split mismatch")
    audio_contract = {
        "sample_rate": SAMPLE_RATE,
        "num_channels": NUM_CHANNELS,
        "num_samples": NUM_SAMPLES,
        "duration_seconds": DURATION_SECONDS,
    }
    for field, expected in audio_contract.items():
        if payload.get(field) != expected:
            raise ValueError(f"{field} must equal {expected!r}")
    _safe_audio_path(str(payload.get("mixture_path")))
    digest = payload.get("mixture_sha256")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise ValueError("mixture_sha256 must be a lowercase SHA256")
    relation = payload.get("relation")
    if relation not in {"after", "before", "first", "no_evidence"}:
        raise ValueError("invalid relation")
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")


def inference_fingerprint(payload: Mapping[str, Any]) -> str:
    validate_inference_record(payload)
    return canonical_json_sha256(
        {"format": INFERENCE_FINGERPRINT_FORMAT, "record": dict(payload)}
    )


def validate_inference_record(payload: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "id",
        "scene_id",
        "scene_family_id",
        "dataset",
        "dataset_record_id",
        "upstream_record_id",
        "upstream_split",
        "split",
        "question_index",
        "question_type",
        "relation",
        "question",
        "sample_rate",
        "num_channels",
        "num_samples",
        "duration_seconds",
        "mixture_path",
        "mixture_sha256",
    }
    if set(payload) != expected:
        raise ValueError(
            f"inference fields mismatch: missing={sorted(expected - set(payload))}, "
            f"extra={sorted(set(payload) - expected)}"
        )
    if payload.get("schema_version") != INFERENCE_SCHEMA_VERSION:
        raise ValueError("invalid inference schema_version")
    _validate_common(payload)
    if not isinstance(payload.get("question_index"), int):
        raise TypeError("question_index must be an integer")
    if not isinstance(payload.get("upstream_record_id"), str):
        raise TypeError("upstream_record_id must be a string")


def validate_scoring_record(payload: Mapping[str, Any]) -> None:
    inference_fields = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "inference_record_sha256",
            "answer_options",
            "answer",
            "answer_option_index",
            "no_evidence",
            "source_license",
            "anchor_event_ids",
            "answer_event_ids",
            "evidence_event_ids",
            "anchor_intervals",
            "answer_intervals",
            "evidence_intervals",
            "annotation_kind",
            "clean_reference_stems_available",
            "waveform_sdr_evaluation_allowed",
        }
    }
    inference_fields["schema_version"] = INFERENCE_SCHEMA_VERSION
    validate_inference_record(inference_fields)
    expected_extra = {
        "inference_record_sha256",
        "answer_options",
        "answer",
        "answer_option_index",
        "no_evidence",
        "source_license",
        "anchor_event_ids",
        "answer_event_ids",
        "evidence_event_ids",
        "anchor_intervals",
        "answer_intervals",
        "evidence_intervals",
        "annotation_kind",
        "clean_reference_stems_available",
        "waveform_sdr_evaluation_allowed",
    }
    if payload.get("schema_version") != SCORING_SCHEMA_VERSION:
        raise ValueError("invalid scoring schema_version")
    if set(payload) != (set(inference_fields) | expected_extra):
        raise ValueError("scoring fields mismatch")
    if payload.get("inference_record_sha256") != inference_fingerprint(
        inference_fields
    ):
        raise ValueError("scoring row is not bound to its inference projection")
    options = payload.get("answer_options")
    if not isinstance(options, list) or len(options) != 5 or len(set(options)) != 5:
        raise ValueError("answer_options must contain five unique values")
    if NO_EVIDENCE_ANSWER not in options:
        raise ValueError("answer_options must include no_evidence")
    answer = payload.get("answer")
    answer_index = payload.get("answer_option_index")
    if (
        not isinstance(answer_index, int)
        or not 0 <= answer_index < 5
        or options[answer_index] != answer
    ):
        raise ValueError("answer_option_index does not identify answer")
    no_evidence = payload.get("no_evidence")
    if not isinstance(no_evidence, bool) or no_evidence != (answer == NO_EVIDENCE_ANSWER):
        raise ValueError("no_evidence/answer mismatch")
    if payload.get("annotation_kind") != "reviewed_strong_temporal":
        raise ValueError("annotation_kind must be reviewed_strong_temporal")
    if payload.get("clean_reference_stems_available") is not False:
        raise ValueError("real recordings have no clean reference stems")
    if payload.get("waveform_sdr_evaluation_allowed") is not False:
        raise ValueError("waveform SDR evaluation is forbidden on RealDESED")

    roles = []
    for role in ("anchor", "answer", "evidence"):
        event_ids = payload.get(f"{role}_event_ids")
        intervals = payload.get(f"{role}_intervals")
        if not isinstance(event_ids, list) or not isinstance(intervals, list):
            raise TypeError(f"{role} fields must be lists")
        if len(event_ids) != len(intervals):
            raise ValueError(f"{role} event/interval cardinality mismatch")
        for interval in intervals:
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not 0 <= interval[0] < interval[1] <= DURATION_SECONDS
            ):
                raise ValueError(f"invalid {role} interval")
        roles.append((event_ids, intervals))
    anchor_ids, answer_ids, evidence_ids = (item[0] for item in roles)
    if set(evidence_ids) != set(anchor_ids) | set(answer_ids):
        raise ValueError("evidence IDs must equal the anchor/answer union")
    if no_evidence and any(item for pair in roles for item in pair):
        raise ValueError("no-evidence rows cannot contain oracle event evidence")
    if not no_evidence and (not anchor_ids or not answer_ids):
        raise ValueError("answerable rows need anchor and answer evidence")


def _ordered_options(
    *, answer: str, sample_id: str, question_index: int
) -> tuple[list[str], int]:
    labels = [DISPLAY_NAME[label] for label in REALDESED_CLASSES]
    seed = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16], 16)
    generator = random.Random(seed)
    pool = [label for label in labels if label != answer]
    generator.shuffle(pool)
    if answer == NO_EVIDENCE_ANSWER:
        selected = pool[:4]
    else:
        selected = pool[:3] + [NO_EVIDENCE_ANSWER]
    target_index = question_index % 5
    selected.insert(target_index, answer)
    return selected, target_index


def build_question_records(
    *,
    scene_id: str,
    upstream_record_id: str,
    upstream_split: str,
    mixture_path: str,
    mixture_sha256: str,
    source_license: str,
    events: Sequence[Event],
    max_answerable_per_relation: int = 2,
    no_evidence_questions: int = 2,
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Build label-free inference rows and bound post-hoc scoring rows."""

    if upstream_split not in {"validation", "test"}:
        raise ValueError("upstream_split must be validation or test")
    split = {"validation": "real_dev", "test": "real_test"}[upstream_split]
    label_counts: dict[str, int] = {}
    for event in events:
        label_counts[event.label] = label_counts.get(event.label, 0) + 1
    unique_events = tuple(event for event in events if label_counts[event.label] == 1)
    pairs = [
        (first, second)
        for first, second in zip(unique_events, unique_events[1:])
        if first.label != second.label
    ]
    specifications: list[dict[str, Any]] = []
    for first, second in pairs[:max_answerable_per_relation]:
        specifications.append(
            {
                "relation": "after",
                "question_type": "temporal_after",
                "question": (
                    f"Which sound starts next after the {DISPLAY_NAME[first.label]} "
                    "starts?"
                ),
                "answer": DISPLAY_NAME[second.label],
                "anchor": (first,),
                "answer_events": (second,),
            }
        )
        specifications.append(
            {
                "relation": "before",
                "question_type": "temporal_before",
                "question": (
                    f"Which sound starts immediately before the "
                    f"{DISPLAY_NAME[second.label]} starts?"
                ),
                "answer": DISPLAY_NAME[first.label],
                "anchor": (second,),
                "answer_events": (first,),
            }
        )
    for first, second in pairs[:max_answerable_per_relation]:
        specifications.append(
            {
                "relation": "first",
                "question_type": "temporal_first",
                "question": (
                    f"Which starts first: {DISPLAY_NAME[first.label]} or "
                    f"{DISPLAY_NAME[second.label]}?"
                ),
                "answer": DISPLAY_NAME[first.label],
                # ``first`` has two named candidates rather than a linguistic
                # anchor.  Keep the later candidate in the anchor slot and the
                # gold, earlier candidate in the answer slot so role labels do
                # not contradict the semantic answer.
                "anchor": (second,),
                "answer_events": (first,),
            }
        )
    present = set(label_counts)
    absent = [label for label in REALDESED_CLASSES if label not in present]
    absent_seed = int(hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:16], 16)
    random.Random(absent_seed).shuffle(absent)
    for label in absent[:no_evidence_questions]:
        specifications.append(
            {
                "relation": "no_evidence",
                "question_type": "temporal_no_evidence",
                "question": (
                    f"Which sound starts next after the {DISPLAY_NAME[label]} starts?"
                ),
                "answer": NO_EVIDENCE_ANSWER,
                "anchor": (),
                "answer_events": (),
            }
        )

    inference_rows = []
    scoring_rows = []
    for question_index, spec in enumerate(specifications):
        sample_id = f"{scene_id}__q{question_index:02d}"
        inference = {
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "id": sample_id,
            "scene_id": scene_id,
            "scene_family_id": scene_id,
            "dataset": DATASET_NAME,
            "dataset_record_id": DATASET_RECORD_ID,
            "upstream_record_id": upstream_record_id,
            "upstream_split": upstream_split,
            "split": split,
            "question_index": question_index,
            "question_type": spec["question_type"],
            "relation": spec["relation"],
            "question": spec["question"],
            "sample_rate": SAMPLE_RATE,
            "num_channels": NUM_CHANNELS,
            "num_samples": NUM_SAMPLES,
            "duration_seconds": DURATION_SECONDS,
            "mixture_path": mixture_path,
            "mixture_sha256": mixture_sha256,
        }
        validate_inference_record(inference)
        anchor = tuple(spec["anchor"])
        answer_events = tuple(spec["answer_events"])
        evidence = tuple(
            {event.event_id: event for event in (*anchor, *answer_events)}.values()
        )
        options, answer_index = _ordered_options(
            answer=str(spec["answer"]),
            sample_id=sample_id,
            question_index=question_index,
        )
        scoring = {
            **inference,
            "schema_version": SCORING_SCHEMA_VERSION,
            "inference_record_sha256": inference_fingerprint(inference),
            "answer_options": options,
            "answer": spec["answer"],
            "answer_option_index": answer_index,
            "no_evidence": spec["answer"] == NO_EVIDENCE_ANSWER,
            "source_license": source_license,
            "anchor_event_ids": [event.event_id for event in anchor],
            "answer_event_ids": [event.event_id for event in answer_events],
            "evidence_event_ids": [event.event_id for event in evidence],
            "anchor_intervals": [event.interval for event in anchor],
            "answer_intervals": [event.interval for event in answer_events],
            "evidence_intervals": [event.interval for event in evidence],
            "annotation_kind": "reviewed_strong_temporal",
            "clean_reference_stems_available": False,
            "waveform_sdr_evaluation_allowed": False,
        }
        validate_scoring_record(scoring)
        inference_rows.append(inference)
        scoring_rows.append(scoring)
    return tuple(inference_rows), tuple(scoring_rows)
