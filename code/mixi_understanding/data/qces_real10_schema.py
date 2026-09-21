"""Strict, leakage-aware contracts for the canonical QCES-Real-10 set.

QCES-Real-10 uses one immutable ten-second mixture for every question about a
scene.  It deliberately has no clean source stems: human event intervals are
valid temporal scoring annotations, but they are not oracle waveforms and must
never authorize SI-SDR or SD-SDR claims.

Two record views make the model-input boundary explicit:

``qces_real10_inference_v2``
    Contains only the question and canonical mixture identity needed by QCES.

``qces_real10_scoring_v2``
    Adds multiple-choice and human temporal labels for post-hoc scoring.  Each
    row is bound to the SHA256 of its deterministic inference projection.

The module is intentionally independent of TACOS packet/planner internals so
the acquisition pipeline can change without weakening the released contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, Tuple


INFERENCE_SCHEMA_VERSION = "qces_real10_inference_v2"
SCORING_SCHEMA_VERSION = "qces_real10_scoring_v2"
INFERENCE_RECORD_FINGERPRINT_FORMAT = "qces_real10_inference_record_fingerprint_v1"
INFERENCE_MANIFEST_FINGERPRINT_FORMAT = "qces_real10_inference_manifest_fingerprint_v1"

SAMPLE_RATE = 32_000
NUM_CHANNELS = 1
NUM_SAMPLES = 320_000
DURATION_SECONDS = 10.0
NO_EVIDENCE_ANSWER = "no_evidence"

SPLITS = ("real_dev", "real_test")
ORIGINAL_TACOS_SPLITS = ("development", "test")
RELATIONS = ("after", "before", "first")
QUESTION_TYPE_BY_RELATION = {
    "after": "temporal_after",
    "before": "temporal_before",
    "first": "temporal_first",
}

Interval = Tuple[float, float]

_COMMON_FIELDS = frozenset(
    {
        "id",
        "scene_id",
        "scene_family_id",
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
)
INFERENCE_FIELDS = frozenset({"schema_version", *_COMMON_FIELDS})
_SCORING_ONLY_FIELDS = frozenset(
    {
        "inference_record_sha256",
        "answer_options",
        "answer",
        "answer_option_index",
        "no_evidence",
        "creator_id",
        "anchor_event_ids",
        "answer_event_ids",
        "evidence_event_ids",
        "anchor_intervals",
        "answer_intervals",
        "evidence_intervals",
        "upstream_tacos_split",
        "clean_reference_stems_available",
        "waveform_sdr_evaluation_allowed",
    }
)
SCORING_FIELDS = frozenset({"schema_version", *_COMMON_FIELDS, *_SCORING_ONLY_FIELDS})

# Exact-field validation rejects every undeclared key.  Keeping this explicit
# set gives particularly clear failures for tempting but scientifically invalid
# additions to a real-recording manifest.
FORBIDDEN_ORACLE_OR_SDR_FIELDS = frozenset(
    {
        "evidence_stem_path",
        "residual_stem_path",
        "anchor_stem_path",
        "answer_stem_path",
        "oracle_evidence_path",
        "oracle_residual_path",
        "si_sdr",
        "si_sdri",
        "sd_sdr",
        "sd_sdri",
    }
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_sha256(payload: Any) -> str:
    """Hash one canonical UTF-8 JSON representation."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_forbidden_claim_fields(payload: Mapping[str, Any], context: str) -> None:
    forbidden = sorted(set(payload) & FORBIDDEN_ORACLE_OR_SDR_FIELDS)
    if forbidden:
        raise ValueError(
            f"{context} cannot contain oracle clean-stem or waveform-SDR fields: "
            f"{forbidden}"
        )


def _exact(payload: Mapping[str, Any], fields: frozenset[str], context: str) -> None:
    _reject_forbidden_claim_fields(payload, context)
    actual = set(payload)
    missing = sorted(fields - actual)
    extra = sorted(actual - fields, key=repr)
    if missing or extra:
        raise ValueError(f"{context} fields mismatch: missing={missing}, extra={extra}")


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{context} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{context} must not have surrounding whitespace")
    return value


def _identifier(value: Any, context: str) -> str:
    result = _string(value, context)
    if _IDENTIFIER_RE.fullmatch(result) is None:
        raise ValueError(f"{context} is not a safe identifier")
    return result


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


def _sha256(value: Any, context: str) -> str:
    result = _string(value, context)
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{context} must be a lowercase SHA256 digest")
    return result


def _safe_mixture_path(value: Any, context: str) -> str:
    result = _string(value, context)
    if "\\" in result:
        raise ValueError(f"{context} must use POSIX separators")
    raw_parts = result.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{context} must be a normalized safe relative path")
    if any(":" in part for part in raw_parts):
        raise ValueError(f"{context} must not contain a drive or URI scheme")
    path = PurePosixPath(result)
    if path.is_absolute() or path.suffix.casefold() != ".wav":
        raise ValueError(f"{context} must be a relative .wav path")
    return result


def _intervals(value: Any, context: str) -> Tuple[Interval, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = []
    for index, raw in enumerate(value):
        if not isinstance(raw, list) or len(raw) != 2:
            raise TypeError(f"{context}[{index}] must be [onset, offset]")
        onset = _number(raw[0], f"{context}[{index}][0]")
        offset = _number(raw[1], f"{context}[{index}][1]")
        if not 0.0 <= onset < offset <= DURATION_SECONDS:
            raise ValueError(
                f"{context}[{index}] must lie inside [0, {DURATION_SECONDS}]"
            )
        result.append((onset, offset))
    return tuple(result)


def _string_tuple(value: Any, context: str) -> Tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a list")
    result = tuple(
        _string(item, f"{context}[{index}]") for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ValueError(f"{context} contains duplicates")
    return result


@dataclass(frozen=True)
class QCESReal10InferenceRecord:
    schema_version: str
    sample_id: str
    scene_id: str
    scene_family_id: str
    split: str
    question_index: int
    question_type: str
    relation: str
    question: str
    sample_rate: int
    num_channels: int
    num_samples: int
    duration_seconds: float
    mixture_path: str
    mixture_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.sample_id,
            "scene_id": self.scene_id,
            "scene_family_id": self.scene_family_id,
            "split": self.split,
            "question_index": self.question_index,
            "question_type": self.question_type,
            "relation": self.relation,
            "question": self.question,
            "sample_rate": self.sample_rate,
            "num_channels": self.num_channels,
            "num_samples": self.num_samples,
            "duration_seconds": self.duration_seconds,
            "mixture_path": self.mixture_path,
            "mixture_sha256": self.mixture_sha256,
        }


@dataclass(frozen=True)
class QCESReal10ScoringRecord:
    schema_version: str
    sample_id: str
    scene_id: str
    scene_family_id: str
    split: str
    question_index: int
    question_type: str
    relation: str
    question: str
    sample_rate: int
    num_channels: int
    num_samples: int
    duration_seconds: float
    mixture_path: str
    mixture_sha256: str
    inference_record_sha256: str
    answer_options: Tuple[str, ...]
    answer: str
    answer_option_index: int
    no_evidence: bool
    creator_id: str
    anchor_event_ids: Tuple[str, ...]
    answer_event_ids: Tuple[str, ...]
    evidence_event_ids: Tuple[str, ...]
    anchor_intervals: Tuple[Interval, ...]
    answer_intervals: Tuple[Interval, ...]
    evidence_intervals: Tuple[Interval, ...]
    upstream_tacos_split: str
    clean_reference_stems_available: bool
    waveform_sdr_evaluation_allowed: bool

    def to_inference(self) -> QCESReal10InferenceRecord:
        return QCESReal10InferenceRecord(
            schema_version=INFERENCE_SCHEMA_VERSION,
            sample_id=self.sample_id,
            scene_id=self.scene_id,
            scene_family_id=self.scene_family_id,
            split=self.split,
            question_index=self.question_index,
            question_type=self.question_type,
            relation=self.relation,
            question=self.question,
            sample_rate=self.sample_rate,
            num_channels=self.num_channels,
            num_samples=self.num_samples,
            duration_seconds=self.duration_seconds,
            mixture_path=self.mixture_path,
            mixture_sha256=self.mixture_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_inference().to_dict()
        payload.update(
            {
                "schema_version": self.schema_version,
                "inference_record_sha256": self.inference_record_sha256,
                "answer_options": list(self.answer_options),
                "answer": self.answer,
                "answer_option_index": self.answer_option_index,
                "no_evidence": self.no_evidence,
                "creator_id": self.creator_id,
                "anchor_event_ids": list(self.anchor_event_ids),
                "answer_event_ids": list(self.answer_event_ids),
                "evidence_event_ids": list(self.evidence_event_ids),
                "anchor_intervals": [
                    list(interval) for interval in self.anchor_intervals
                ],
                "answer_intervals": [
                    list(interval) for interval in self.answer_intervals
                ],
                "evidence_intervals": [
                    list(interval) for interval in self.evidence_intervals
                ],
                "upstream_tacos_split": self.upstream_tacos_split,
                "clean_reference_stems_available": self.clean_reference_stems_available,
                "waveform_sdr_evaluation_allowed": self.waveform_sdr_evaluation_allowed,
            }
        )
        return payload


QCESReal10Record = QCESReal10InferenceRecord | QCESReal10ScoringRecord


def _parse_common(
    payload: Mapping[str, Any], *, schema_version: str
) -> QCESReal10InferenceRecord:
    if payload.get("schema_version") != schema_version:
        raise ValueError(f"schema_version must equal {schema_version!r}")
    sample_id = _identifier(payload["id"], "id")
    scene_id = _identifier(payload["scene_id"], "scene_id")
    family_id = _identifier(payload["scene_family_id"], "scene_family_id")
    split = _string(payload["split"], "split")
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    question_index = _integer(payload["question_index"], "question_index")
    if question_index < 0:
        raise ValueError("question_index must be non-negative")
    relation = _string(payload["relation"], "relation")
    if relation not in RELATIONS:
        raise ValueError(f"relation must be one of {RELATIONS}")
    question_type = _string(payload["question_type"], "question_type")
    if question_type != QUESTION_TYPE_BY_RELATION[relation]:
        raise ValueError("question_type/relation mismatch")
    sample_rate = _integer(payload["sample_rate"], "sample_rate")
    channels = _integer(payload["num_channels"], "num_channels")
    samples = _integer(payload["num_samples"], "num_samples")
    duration = _number(payload["duration_seconds"], "duration_seconds")
    if (
        sample_rate != SAMPLE_RATE
        or channels != NUM_CHANNELS
        or samples != NUM_SAMPLES
        or not math.isclose(duration, DURATION_SECONDS, rel_tol=0.0, abs_tol=1e-9)
    ):
        raise ValueError(
            "QCES-Real-10 audio must be exactly mono, 32 kHz, 320000 samples, "
            "and 10.0 seconds"
        )
    return QCESReal10InferenceRecord(
        schema_version=INFERENCE_SCHEMA_VERSION,
        sample_id=sample_id,
        scene_id=scene_id,
        scene_family_id=family_id,
        split=split,
        question_index=question_index,
        question_type=question_type,
        relation=relation,
        question=_string(payload["question"], "question"),
        sample_rate=sample_rate,
        num_channels=channels,
        num_samples=samples,
        duration_seconds=duration,
        mixture_path=_safe_mixture_path(payload["mixture_path"], "mixture_path"),
        mixture_sha256=_sha256(payload["mixture_sha256"], "mixture_sha256"),
    )


def parse_inference_record(payload: Mapping[str, Any]) -> QCESReal10InferenceRecord:
    """Parse one exact canonical model-input record."""

    if not isinstance(payload, Mapping):
        raise TypeError("inference record must be an object")
    _exact(payload, INFERENCE_FIELDS, "QCES-Real-10 inference record")
    return _parse_common(payload, schema_version=INFERENCE_SCHEMA_VERSION)


def project_scoring_fields_to_inference(
    payload: Mapping[str, Any],
) -> QCESReal10InferenceRecord:
    """Project common scoring fields without reading any scoring label field.

    This helper can construct the binding before a scoring row has its
    ``inference_record_sha256`` field.  Full scoring validity is established by
    :func:`parse_scoring_record`.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("scoring projection source must be an object")
    missing = sorted(_COMMON_FIELDS - set(payload))
    if missing:
        raise ValueError(f"scoring projection lacks common fields: {missing}")
    projected = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        **{field: payload[field] for field in _COMMON_FIELDS},
    }
    return parse_inference_record(projected)


def inference_record_fingerprint(record: QCESReal10InferenceRecord) -> str:
    """Return a domain-separated canonical binding for one inference row."""

    if not isinstance(record, QCESReal10InferenceRecord):
        raise TypeError("record must be QCESReal10InferenceRecord")
    validated = parse_inference_record(record.to_dict())
    return canonical_json_sha256(
        {
            "format": INFERENCE_RECORD_FINGERPRINT_FORMAT,
            "record": validated.to_dict(),
        }
    )


def parse_scoring_record(payload: Mapping[str, Any]) -> QCESReal10ScoringRecord:
    """Parse a post-hoc scoring row and verify its inference binding."""

    if not isinstance(payload, Mapping):
        raise TypeError("scoring record must be an object")
    _exact(payload, SCORING_FIELDS, "QCES-Real-10 scoring record")
    if payload.get("schema_version") != SCORING_SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCORING_SCHEMA_VERSION!r}")
    inference = project_scoring_fields_to_inference(payload)
    declared_binding = _sha256(
        payload["inference_record_sha256"], "inference_record_sha256"
    )
    expected_binding = inference_record_fingerprint(inference)
    if declared_binding != expected_binding:
        raise ValueError("scoring row does not match its inference-record binding")

    options = _string_tuple(payload["answer_options"], "answer_options")
    normalized_options = {" ".join(option.casefold().split()) for option in options}
    if len(options) != 5 or len(normalized_options) != 5:
        raise ValueError(
            "answer_options must contain five case-insensitively unique values"
        )
    if NO_EVIDENCE_ANSWER not in options:
        raise ValueError("answer_options must include no_evidence")
    answer = _string(payload["answer"], "answer")
    answer_index = _integer(payload["answer_option_index"], "answer_option_index")
    if not 0 <= answer_index < len(options) or options[answer_index] != answer:
        raise ValueError("answer_option_index does not identify answer")
    no_evidence = payload["no_evidence"]
    if not isinstance(no_evidence, bool):
        raise TypeError("no_evidence must be boolean")
    creator_id = _string(payload["creator_id"], "creator_id")
    anchor_ids = _string_tuple(payload["anchor_event_ids"], "anchor_event_ids")
    answer_ids = _string_tuple(payload["answer_event_ids"], "answer_event_ids")
    evidence_ids = _string_tuple(payload["evidence_event_ids"], "evidence_event_ids")
    anchor_intervals = _intervals(payload["anchor_intervals"], "anchor_intervals")
    answer_intervals = _intervals(payload["answer_intervals"], "answer_intervals")
    evidence_intervals = _intervals(payload["evidence_intervals"], "evidence_intervals")
    role_pairs = (
        ("anchor", anchor_ids, anchor_intervals),
        ("answer", answer_ids, answer_intervals),
        ("evidence", evidence_ids, evidence_intervals),
    )
    for role, event_ids, intervals in role_pairs:
        if len(event_ids) != len(intervals):
            raise ValueError(
                f"{role} event IDs and intervals must have equal cardinality"
            )
    interval_by_event: dict[str, Interval] = {}
    for role, event_ids, intervals in role_pairs:
        for event_id, interval in zip(event_ids, intervals):
            previous = interval_by_event.setdefault(event_id, interval)
            if previous != interval:
                raise ValueError(
                    f"event {event_id!r} has inconsistent {role} interval pairing"
                )
    role_union = set(anchor_ids) | set(answer_ids)
    if set(evidence_ids) != role_union:
        raise ValueError(
            "evidence event IDs/intervals must equal the deduplicated "
            "anchor-answer union"
        )
    if no_evidence:
        if answer != NO_EVIDENCE_ANSWER or any(
            event_ids or intervals for _, event_ids, intervals in role_pairs
        ):
            raise ValueError(
                "no-evidence rows require answer=no_evidence and all role labels empty"
            )
    elif answer == NO_EVIDENCE_ANSWER or not anchor_ids or not answer_ids:
        raise ValueError(
            "answerable rows require nonempty anchor and answer temporal evidence"
        )

    original_split = _string(payload["upstream_tacos_split"], "upstream_tacos_split")
    if original_split not in ORIGINAL_TACOS_SPLITS:
        raise ValueError(f"upstream_tacos_split must be one of {ORIGINAL_TACOS_SPLITS}")

    clean_stems = payload["clean_reference_stems_available"]
    sdr_allowed = payload["waveform_sdr_evaluation_allowed"]
    if not isinstance(clean_stems, bool) or not isinstance(sdr_allowed, bool):
        raise TypeError("clean-stem and waveform-SDR flags must be boolean")
    if clean_stems or sdr_allowed:
        raise ValueError(
            "QCES-Real-10 has no oracle clean stems and forbids waveform-SDR claims"
        )

    return QCESReal10ScoringRecord(
        schema_version=SCORING_SCHEMA_VERSION,
        sample_id=inference.sample_id,
        scene_id=inference.scene_id,
        scene_family_id=inference.scene_family_id,
        split=inference.split,
        question_index=inference.question_index,
        question_type=inference.question_type,
        relation=inference.relation,
        question=inference.question,
        sample_rate=inference.sample_rate,
        num_channels=inference.num_channels,
        num_samples=inference.num_samples,
        duration_seconds=inference.duration_seconds,
        mixture_path=inference.mixture_path,
        mixture_sha256=inference.mixture_sha256,
        inference_record_sha256=declared_binding,
        answer_options=options,
        answer=answer,
        answer_option_index=answer_index,
        no_evidence=no_evidence,
        creator_id=creator_id,
        anchor_event_ids=anchor_ids,
        answer_event_ids=answer_ids,
        evidence_event_ids=evidence_ids,
        anchor_intervals=anchor_intervals,
        answer_intervals=answer_intervals,
        evidence_intervals=evidence_intervals,
        upstream_tacos_split=original_split,
        clean_reference_stems_available=clean_stems,
        waveform_sdr_evaluation_allowed=sdr_allowed,
    )


def parse_record(payload: Mapping[str, Any]) -> QCESReal10Record:
    """Dispatch one record while failing closed on unknown versions."""

    schema_version = (
        payload.get("schema_version") if isinstance(payload, Mapping) else None
    )
    if schema_version == INFERENCE_SCHEMA_VERSION:
        return parse_inference_record(payload)
    if schema_version == SCORING_SCHEMA_VERSION:
        return parse_scoring_record(payload)
    raise ValueError(f"unsupported QCES-Real-10 schema_version: {schema_version!r}")


def scoring_to_inference(
    record: QCESReal10ScoringRecord,
) -> QCESReal10InferenceRecord:
    """Return the validated, label-free projection of one scoring record."""

    if not isinstance(record, QCESReal10ScoringRecord):
        raise TypeError("record must be QCESReal10ScoringRecord")
    validated = parse_scoring_record(record.to_dict())
    return validated.to_inference()


def _validate_manifest_records(
    records: Sequence[QCESReal10Record], *, context: str
) -> None:
    if not records:
        raise ValueError(f"{context} must not be empty")
    sample_ids = [record.sample_id for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"{context} contains duplicate sample IDs")
    scene_identity: dict[str, tuple[Any, ...]] = {}
    scene_question_indices: set[tuple[str, int]] = set()
    for record in records:
        identity = (
            record.scene_family_id,
            record.split,
            record.sample_rate,
            record.num_channels,
            record.num_samples,
            record.duration_seconds,
            record.mixture_path,
            record.mixture_sha256,
        )
        previous = scene_identity.setdefault(record.scene_id, identity)
        if previous != identity:
            raise ValueError(f"{context} has inconsistent scene audio identity")
        question_key = (record.scene_id, record.question_index)
        if question_key in scene_question_indices:
            raise ValueError(f"{context} reuses a question index within one scene")
        scene_question_indices.add(question_key)


def parse_inference_manifest(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[QCESReal10InferenceRecord, ...]:
    records = tuple(parse_inference_record(row) for row in rows)
    _validate_manifest_records(records, context="QCES-Real-10 inference manifest")
    return records


def parse_scoring_manifest(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[QCESReal10ScoringRecord, ...]:
    records = tuple(parse_scoring_record(row) for row in rows)
    _validate_manifest_records(records, context="QCES-Real-10 scoring manifest")
    upstream_split_by_scene: dict[str, str] = {}
    creator_by_scene: dict[str, str] = {}
    for record in records:
        previous = upstream_split_by_scene.setdefault(
            record.scene_id, record.upstream_tacos_split
        )
        if previous != record.upstream_tacos_split:
            raise ValueError(
                "QCES-Real-10 scoring manifest has inconsistent upstream "
                "TACOS split within one scene"
            )
        previous_creator = creator_by_scene.setdefault(
            record.scene_id, record.creator_id
        )
        if previous_creator != record.creator_id:
            raise ValueError(
                "QCES-Real-10 scoring manifest has inconsistent creator ID "
                "within one scene"
            )
    return records


def project_scoring_manifest(
    records: Sequence[QCESReal10ScoringRecord],
) -> Tuple[QCESReal10InferenceRecord, ...]:
    projected = tuple(scoring_to_inference(record) for record in records)
    _validate_manifest_records(projected, context="projected QCES-Real-10 manifest")
    return projected


def canonical_inference_manifest_fingerprint(
    records: Sequence[QCESReal10InferenceRecord],
) -> str:
    """Fingerprint a validated manifest independent of JSONL row order."""

    validated = []
    for record in records:
        if not isinstance(record, QCESReal10InferenceRecord):
            raise TypeError("manifest records must be QCESReal10InferenceRecord")
        validated.append(parse_inference_record(record.to_dict()))
    materialized = tuple(validated)
    _validate_manifest_records(
        materialized, context="QCES-Real-10 manifest fingerprint"
    )
    ordered = sorted(
        (record.to_dict() for record in materialized), key=lambda row: row["id"]
    )
    return canonical_json_sha256(
        {
            "format": INFERENCE_MANIFEST_FINGERPRINT_FORMAT,
            "records": ordered,
        }
    )


def resolve_manifest_mixture_path(
    manifest_path: Path, record: QCESReal10Record
) -> Path:
    """Resolve a mixture below the manifest directory, including symlink checks."""

    root = manifest_path.resolve().parent
    candidate = (root / PurePosixPath(record.mixture_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("mixture_path escapes the manifest directory") from error
    return candidate


__all__ = [
    "DURATION_SECONDS",
    "FORBIDDEN_ORACLE_OR_SDR_FIELDS",
    "INFERENCE_FIELDS",
    "INFERENCE_SCHEMA_VERSION",
    "NO_EVIDENCE_ANSWER",
    "NUM_CHANNELS",
    "NUM_SAMPLES",
    "ORIGINAL_TACOS_SPLITS",
    "QCESReal10InferenceRecord",
    "QCESReal10Record",
    "QCESReal10ScoringRecord",
    "RELATIONS",
    "SAMPLE_RATE",
    "SCORING_FIELDS",
    "SCORING_SCHEMA_VERSION",
    "SPLITS",
    "canonical_inference_manifest_fingerprint",
    "canonical_json_sha256",
    "inference_record_fingerprint",
    "parse_inference_manifest",
    "parse_inference_record",
    "parse_record",
    "parse_scoring_manifest",
    "parse_scoring_record",
    "project_scoring_fields_to_inference",
    "project_scoring_manifest",
    "resolve_manifest_mixture_path",
    "scoring_to_inference",
]
