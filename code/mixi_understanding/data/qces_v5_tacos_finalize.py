"""Adjudicate TACOS annotations and emit the real QCES QA manifest."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from mixi_understanding.data.qces_real10_schema import (
    DURATION_SECONDS,
    INFERENCE_SCHEMA_VERSION,
    NUM_CHANNELS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    SCORING_SCHEMA_VERSION,
    canonical_inference_manifest_fingerprint,
    inference_record_fingerprint,
    parse_inference_manifest,
    parse_scoring_manifest,
    project_scoring_fields_to_inference,
    project_scoring_manifest,
)
from mixi_understanding.data.qces_v5_tacos import (
    PACKET_FORMAT,
    QUESTION_SLOTS,
    TacosAuditError,
    atomic_json,
    atomic_jsonl,
    canonical_json_sha256,
    normalize_caption,
    sha256_file,
)
from mixi_understanding.data.qces_v5_tacos_annotation import (
    RATER_ID_RE,
    load_bound_tasks,
    response_path,
    validate_response,
)
from mixi_understanding.data.qces_v5_tacos_audio import AUDIO_RECEIPT_FORMAT


# Kept local because the real manifest is intentionally independent of the
# synthetic storage schema (TACOS has no clean source stems).
NO_EVIDENCE_ANSWER = "no_evidence"
REAL_SCENE_FORMAT = "qces_v5_tacos_verified_scene_v2"
REAL_QA_FORMAT = "qces_v5_tacos_real_qa_v2"
FINALIZATION_FORMAT = "qces_v5_tacos_human_finalization_v4"
ADJUDICATION_AUDIT_FORMAT = "qces_v5_tacos_adjudication_audit_v1"
SCORING_MANIFEST_FINGERPRINT_FORMAT = "qces_real10_scoring_manifest_fingerprint_v1"
AUDIO_MANIFEST_FINGERPRINT_FORMAT = "qces_real10_audio_manifest_fingerprint_v1"

LEGACY_OUTPUT_NAMES = (
    "qces_v5_tacos_verified_scenes.jsonl",
    "qces_v5_tacos_real_qa.jsonl",
    "qces_v5_tacos_adjudication_audit.jsonl",
)
AUTHORITATIVE_OUTPUT_NAMES = (
    "qces_real10_inference.jsonl",
    "qces_real10_scoring.jsonl",
)
SPLIT_INFERENCE_OUTPUT_NAMES = {
    "real_dev": "qces_real10_inference_real_dev.jsonl",
    "real_test": "qces_real10_inference_real_test.jsonl",
}


def _validate_resolution_thresholds(
    maximum_boundary_difference_seconds: float,
    minimum_interval_iou: float,
) -> None:
    if (
        isinstance(maximum_boundary_difference_seconds, bool)
        or not isinstance(maximum_boundary_difference_seconds, (int, float))
        or not math.isfinite(float(maximum_boundary_difference_seconds))
        or maximum_boundary_difference_seconds < 0.0
    ):
        raise TacosAuditError(
            "maximum boundary difference must be finite and non-negative"
        )
    if (
        isinstance(minimum_interval_iou, bool)
        or not isinstance(minimum_interval_iou, (int, float))
        or not math.isfinite(float(minimum_interval_iou))
    ):
        raise TacosAuditError("minimum interval IoU must be finite")
    if not 0.0 <= minimum_interval_iou <= 1.0:
        raise TacosAuditError("minimum interval IoU must lie in [0, 1]")


def _validate_bound_task(task: Mapping[str, Any]) -> None:
    """Reject anything outside the current packet/canonical-audio boundary."""

    scene_id = task.get("scene_id", "unknown scene")
    if task.get("schema_version") != PACKET_FORMAT:
        raise TacosAuditError(f"{scene_id} is not a {PACKET_FORMAT} task")
    partition = task.get("selection_partition")
    if partition not in {"real_dev", "real_test"}:
        raise TacosAuditError(f"{scene_id} has an invalid QCES real partition")
    source = task.get("source")
    audio = task.get("audio")
    window = task.get("benchmark_window")
    if not all(isinstance(value, Mapping) for value in (source, audio, window)):
        raise TacosAuditError(f"{scene_id} lacks packet source/audio/window data")
    assert isinstance(source, Mapping)
    assert isinstance(audio, Mapping)
    assert isinstance(window, Mapping)
    if source.get("custom_qces_partition") != partition:
        raise TacosAuditError(f"{scene_id} source/QCES partition mismatch")
    if source.get("upstream_tacos_split") not in {"development", "test"}:
        raise TacosAuditError(f"{scene_id} lacks its original TACOS split")
    if source.get("clip_duration_seconds") != DURATION_SECONDS:
        raise TacosAuditError(f"{scene_id} is not an exact ten-second source crop")

    start = window.get("start_sample")
    end = window.get("end_sample")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or window.get("sample_rate_hz") != SAMPLE_RATE
        or window.get("duration_seconds") != DURATION_SECONDS
        or end - start != NUM_SAMPLES
        or audio.get("benchmark_window_start_sample_32k") != start
        or audio.get("benchmark_window_end_sample_32k") != end
        or audio.get("benchmark_sample_rate_hz") != SAMPLE_RATE
    ):
        raise TacosAuditError(f"{scene_id} violates the packet 10 s audio window")
    local_path = task.get("audio_receipt_local_path")
    local_sha256 = task.get("audio_receipt_sha256")
    if not isinstance(local_path, str) or not local_path.endswith(".wav"):
        raise TacosAuditError(f"{scene_id} lacks a derived receipt-v2 WAV path")
    if (
        not isinstance(local_sha256, str)
        or len(local_sha256) != 64
        or any(character not in "0123456789abcdef" for character in local_sha256)
    ):
        raise TacosAuditError(f"{scene_id} lacks a derived receipt-v2 SHA256")


def interval_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0.0 else 0.0


def _immutable_query_phrase_by_region(
    task: Mapping[str, Any],
) -> dict[str, str]:
    """Return exact packet captions keyed by region, never human rewrites."""

    proposals = task.get("proposal_regions")
    if not isinstance(proposals, list):
        raise TacosAuditError("task lacks immutable proposal captions")
    result: dict[str, str] = {}
    for proposal in proposals:
        if not isinstance(proposal, Mapping):
            raise TacosAuditError("task has an invalid proposal region")
        region_id = proposal.get("region_id")
        caption = proposal.get("caption")
        if (
            not isinstance(region_id, str)
            or not region_id
            or region_id in result
            or not isinstance(caption, str)
            or not caption
            or caption != caption.strip()
        ):
            raise TacosAuditError("task has an invalid immutable proposal caption")
        result[region_id] = caption
    return result


def _quoted_query_phrase(value: str) -> str:
    """Quote arbitrary upstream prose without turning it into a new caption."""

    return json.dumps(value, ensure_ascii=False)


def _verified_absent_entries(
    response: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Project human inaudibility judgments onto immutable planner support."""

    contract = task.get("question_contract")
    if not isinstance(contract, Mapping):
        raise TacosAuditError("task lacks an absent-anchor question contract")
    candidates = contract.get("absent_anchor_candidates")
    support_rows = contract.get("absent_anchor_cross_scene_support")
    judgments = response.get("absent_anchor_judgments")
    if (
        not isinstance(candidates, list)
        or not isinstance(support_rows, list)
        or not isinstance(judgments, list)
        or len(candidates) != len(support_rows)
        or len(candidates) != len(judgments)
    ):
        raise TacosAuditError("absent-anchor candidate/support/judgment mismatch")
    verified: list[dict[str, Any]] = []
    for caption, support, judgment in zip(
        candidates, support_rows, judgments, strict=True
    ):
        support_scene_ids = (
            support.get("cross_scene_support_scene_ids")
            if isinstance(support, Mapping)
            else None
        )
        if (
            not isinstance(caption, str)
            or not caption
            or not isinstance(support, Mapping)
            or support.get("caption") != caption
            or not isinstance(support_scene_ids, list)
            or not support_scene_ids
            or len(set(support_scene_ids)) != len(support_scene_ids)
            or any(
                not isinstance(scene_id, str)
                or not scene_id
                or scene_id == task.get("scene_id")
                for scene_id in support_scene_ids
            )
            or not isinstance(judgment, Mapping)
            or judgment.get("caption") != caption
        ):
            raise TacosAuditError("invalid immutable absent-anchor support binding")
        if judgment.get("confirmed_inaudible") == "yes":
            verified.append(
                {
                    "caption": caption,
                    "cross_scene_support_scene_ids": list(support_scene_ids),
                }
            )
    return verified


def _read_response(
    *,
    response_root: Path,
    rater_id: str,
    task: Mapping[str, Any],
    packet_fingerprint: str,
    audio_fingerprint: str,
) -> Mapping[str, Any] | None:
    path = response_path(response_root, rater_id, task["scene_id"])
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TacosAuditError(f"cannot parse human response: {path}") from error
    if not isinstance(payload, dict) or payload.get("rater_id") != rater_id:
        raise TacosAuditError(f"response/rater binding mismatch: {path}")
    return validate_response(
        payload,
        task=task,
        packet_fingerprint=packet_fingerprint,
        audio_receipt_fingerprint=audio_fingerprint,
    )


def resolve_rater_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    maximum_boundary_difference_seconds: float,
    minimum_interval_iou: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Resolve exact semantic agreement and bounded temporal disagreement."""

    _validate_resolution_thresholds(
        maximum_boundary_difference_seconds,
        minimum_interval_iou,
    )
    conflicts: list[str] = []
    if left["scene_decision"] != "accept" or right["scene_decision"] != "accept":
        conflicts.append("rater_rejected_or_uncertain")
        return None, conflicts
    if left["selected_region_ids"] != right["selected_region_ids"]:
        conflicts.append("selected_region_ids_disagree")
    left_absent = _verified_absent_entries(left, task=task)
    right_absent = _verified_absent_entries(right, task=task)
    right_absent_captions = {entry["caption"] for entry in right_absent}
    jointly_verified_absent = [
        entry for entry in left_absent if entry["caption"] in right_absent_captions
    ]
    if not jointly_verified_absent:
        conflicts.append("no_jointly_verified_absent_anchor")
    if any(
        region.get("proposal_caption_accurate") != "yes"
        for response in (left, right)
        for region in response["regions"]
    ):
        conflicts.append("proposal_caption_not_verified_accurate")

    resolved_regions: list[dict[str, Any]] = []
    interval_ious: list[float] = []
    maximum_boundary_difference = 0.0
    if not conflicts:
        immutable_query_phrases = _immutable_query_phrase_by_region(task)
        for index, (left_region, right_region) in enumerate(
            zip(left["regions"], right["regions"])
        ):
            left_phrase = normalize_caption(left_region["canonical_event_phrase"])
            right_phrase = normalize_caption(right_region["canonical_event_phrase"])
            if left_phrase != right_phrase:
                conflicts.append(f"phrase_disagrees:{index}")
                continue
            proposal_region_id = left_region["region_id"]
            if proposal_region_id not in immutable_query_phrases:
                conflicts.append(f"proposal_caption_missing:{index}")
                continue
            left_interval = (
                float(left_region["verified_onset_seconds"]),
                float(left_region["verified_offset_seconds"]),
            )
            right_interval = (
                float(right_region["verified_onset_seconds"]),
                float(right_region["verified_offset_seconds"]),
            )
            boundary_difference = max(
                abs(left_interval[0] - right_interval[0]),
                abs(left_interval[1] - right_interval[1]),
            )
            iou = interval_iou(left_interval, right_interval)
            maximum_boundary_difference = max(
                maximum_boundary_difference, boundary_difference
            )
            interval_ious.append(iou)
            if boundary_difference > maximum_boundary_difference_seconds:
                conflicts.append(f"boundary_disagrees:{index}")
            if iou < minimum_interval_iou:
                conflicts.append(f"interval_iou_below_threshold:{index}")
            resolved_regions.append(
                {
                    "event_id": f"{task['scene_id']}__event_{index}",
                    "proposal_region_id": proposal_region_id,
                    "query_event_phrase": immutable_query_phrases[proposal_region_id],
                    "query_event_phrase_human_verified_accurate": True,
                    "canonical_event_phrase": left_region[
                        "canonical_event_phrase"
                    ].strip(),
                    "onset_seconds": (left_interval[0] + right_interval[0]) / 2.0,
                    "offset_seconds": (left_interval[1] + right_interval[1]) / 2.0,
                    "contamination_rating": (
                        "mixed"
                        if "mixed"
                        in {
                            left_region["contamination_rating"],
                            right_region["contamination_rating"],
                        }
                        else "clean"
                    ),
                    "minimum_salience_1_to_5": min(
                        left_region["salience_1_to_5"],
                        right_region["salience_1_to_5"],
                    ),
                    "rater_interval_iou_↑": iou,
                    "rater_maximum_boundary_difference_seconds_↓": boundary_difference,
                }
            )
    if conflicts:
        return None, sorted(set(conflicts))
    onsets = [region["onset_seconds"] for region in resolved_regions]
    if any(
        right_onset <= left_onset for left_onset, right_onset in zip(onsets, onsets[1:])
    ):
        return None, ["resolved_onset_order_invalid"]
    return (
        {
            "format": REAL_SCENE_FORMAT,
            "scene_id": task["scene_id"],
            "selection_partition": task["selection_partition"],
            "selection_tier": task["selection_tier"],
            "selection_ordinal": task["selection_ordinal"],
            "source": task["source"],
            "audio": {
                "local_path": task["audio_receipt_local_path"],
                "local_sha256": task["audio_receipt_sha256"],
            },
            "events": resolved_regions,
            "jointly_verified_absent_anchor_candidates": jointly_verified_absent,
            "selected_absent_anchor_query_phrase": jointly_verified_absent[0][
                "caption"
            ],
            "selected_absent_anchor_positive_support_scene_id": (
                jointly_verified_absent[0]["cross_scene_support_scene_ids"][0]
            ),
            "human_verification": {
                "rater_ids": [left["rater_id"], right["rater_id"]],
                "resolved_by": "independent_pair_agreement",
                "minimum_interval_iou_↑": min(interval_ious),
                "mean_interval_iou_↑": sum(interval_ious) / len(interval_ious),
                "maximum_boundary_difference_seconds_↓": maximum_boundary_difference,
                "model_outputs_visible": False,
            },
        },
        [],
    )


def scene_from_adjudicator(
    response: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    conflict_reasons: Sequence[str],
) -> dict[str, Any]:
    if response["scene_decision"] != "accept":
        raise TacosAuditError("adjudicator did not accept the conflicted scene")
    if any(
        region.get("proposal_caption_accurate") != "yes"
        for region in response["regions"]
    ):
        raise TacosAuditError(
            "adjudicator did not verify every immutable proposal caption"
        )
    immutable_query_phrases = _immutable_query_phrase_by_region(task)
    verified_absent = _verified_absent_entries(response, task=task)
    if not verified_absent:
        raise TacosAuditError("adjudicator did not verify any absent-anchor candidate")
    events = []
    for index, region in enumerate(response["regions"]):
        proposal_region_id = region["region_id"]
        if proposal_region_id not in immutable_query_phrases:
            raise TacosAuditError("adjudicator selected an unknown proposal region")
        events.append(
            {
                "event_id": f"{task['scene_id']}__event_{index}",
                "proposal_region_id": proposal_region_id,
                "query_event_phrase": immutable_query_phrases[proposal_region_id],
                "query_event_phrase_human_verified_accurate": True,
                "canonical_event_phrase": region["canonical_event_phrase"],
                "onset_seconds": float(region["verified_onset_seconds"]),
                "offset_seconds": float(region["verified_offset_seconds"]),
                "contamination_rating": region["contamination_rating"],
                "minimum_salience_1_to_5": region["salience_1_to_5"],
                "rater_interval_iou_↑": None,
                "rater_maximum_boundary_difference_seconds_↓": None,
            }
        )
    onsets = [event["onset_seconds"] for event in events]
    if any(
        right_onset <= left_onset for left_onset, right_onset in zip(onsets, onsets[1:])
    ):
        raise TacosAuditError("adjudicator verified onset order is invalid")
    return {
        "format": REAL_SCENE_FORMAT,
        "scene_id": task["scene_id"],
        "selection_partition": task["selection_partition"],
        "selection_tier": task["selection_tier"],
        "selection_ordinal": task["selection_ordinal"],
        "source": task["source"],
        "audio": {
            "local_path": task["audio_receipt_local_path"],
            "local_sha256": task["audio_receipt_sha256"],
        },
        "events": events,
        "jointly_verified_absent_anchor_candidates": verified_absent,
        "selected_absent_anchor_query_phrase": verified_absent[0]["caption"],
        "selected_absent_anchor_positive_support_scene_id": verified_absent[0][
            "cross_scene_support_scene_ids"
        ][0],
        "human_verification": {
            "rater_ids": [response["rater_id"]],
            "resolved_by": "third_rater_adjudication",
            "conflict_reasons": list(conflict_reasons),
            "minimum_interval_iou_↑": None,
            "mean_interval_iou_↑": None,
            "maximum_boundary_difference_seconds_↓": None,
            "model_outputs_visible": False,
        },
    }


def _options_with_answer_at(
    event_phrases: Sequence[str], answer: str, answer_index: int
) -> list[str]:
    universe = list(event_phrases) + [NO_EVIDENCE_ANSWER]
    if len(universe) != 5 or len(set(map(normalize_caption, universe))) != 5:
        raise TacosAuditError("real QA requires four distinct event options")
    distractors = [value for value in universe if value != answer]
    if len(distractors) != 4 or not 0 <= answer_index < 5:
        raise TacosAuditError("invalid answer/options contract")
    return distractors[:answer_index] + [answer] + distractors[answer_index:]


def build_scene_questions(
    scene: Mapping[str, Any], *, global_scene_ordinal: int
) -> list[dict[str, Any]]:
    events = scene["events"]
    if len(events) != 4:
        raise TacosAuditError("verified real scene must contain four events")
    phrases = [event["canonical_event_phrase"] for event in events]
    query_phrases = [event["query_event_phrase"] for event in events]
    if any(
        event.get("query_event_phrase_human_verified_accurate") is not True
        for event in events
    ) or any(not isinstance(value, str) or not value for value in query_phrases):
        raise TacosAuditError(
            "verified real scene lacks human-confirmed immutable query captions"
        )
    absent = scene.get("selected_absent_anchor_query_phrase")
    absent_support_scene_id = scene.get(
        "selected_absent_anchor_positive_support_scene_id"
    )
    if (
        not isinstance(absent, str)
        or not absent
        or not isinstance(absent_support_scene_id, str)
        or absent_support_scene_id == scene.get("scene_id")
    ):
        raise TacosAuditError(
            "verified real scene lacks a cross-scene-supported absent query"
        )
    rows: list[dict[str, Any]] = []
    for slot_index, slot in enumerate(QUESTION_SLOTS):
        slot_id = slot["slot_id"]
        no_evidence = bool(slot["no_evidence"])
        query_event_ids: list[str] = []
        anchor_event_ids: list[str] = []
        answer_event_ids: list[str] = []
        query_candidates: list[str] = []
        mention_order = slot["option_order_variant"]
        if slot_id == "after_internal_0":
            anchor_index, answer_index = 0, 1
        elif slot_id == "after_internal_1":
            anchor_index, answer_index = 1, 2
        elif slot_id == "before_internal_0":
            anchor_index, answer_index = 2, 1
        elif slot_id == "before_internal_1":
            anchor_index, answer_index = 3, 2
        elif slot_id in ("first_pair_forward", "first_pair_reversed"):
            first, second = 0, 2
            shown = (first, second) if slot_id.endswith("forward") else (second, first)
            question = (
                "Which described sound begins first in the clip, "
                f"{_quoted_query_phrase(query_phrases[shown[0]])} or "
                f"{_quoted_query_phrase(query_phrases[shown[1]])}?"
            )
            answer = phrases[first]
            query_candidates = [
                query_phrases[shown[0]],
                query_phrases[shown[1]],
            ]
            query_event_ids = [events[first]["event_id"], events[second]["event_id"]]
            anchor_event_ids = list(query_event_ids)
            answer_event_ids = [events[first]["event_id"]]
            anchor_index = answer_index = -1
        elif slot_id == "after_absent_anchor_no_evidence":
            question = (
                f"The described sound, {_quoted_query_phrase(absent)}, begins; "
                "what sound begins immediately after it?"
            )
            answer = NO_EVIDENCE_ANSWER
            anchor_index = answer_index = -1
        elif slot_id == "before_absent_anchor_no_evidence":
            question = (
                f"The described sound, {_quoted_query_phrase(absent)}, begins; "
                "what sound begins immediately before it?"
            )
            answer = NO_EVIDENCE_ANSWER
            anchor_index = answer_index = -1
        else:
            raise TacosAuditError(f"unknown real question slot: {slot_id}")

        if slot["relation"] in ("after", "before") and not no_evidence:
            anchor = events[anchor_index]
            answer_event = events[answer_index]
            direction = "after" if slot["relation"] == "after" else "before"
            question = (
                "The described sound, "
                f"{_quoted_query_phrase(anchor['query_event_phrase'])}, begins; "
                f"what sound begins immediately {direction} it?"
            )
            answer = answer_event["canonical_event_phrase"]
            query_event_ids = [anchor["event_id"]]
            anchor_event_ids = [anchor["event_id"]]
            answer_event_ids = [answer_event["event_id"]]

        desired_answer_index = (
            global_scene_ordinal * len(QUESTION_SLOTS) + slot_index
        ) % 5
        options = _options_with_answer_at(phrases, answer, desired_answer_index)
        event_by_id = {event["event_id"]: event for event in events}

        def intervals(ids: Sequence[str]) -> list[list[float]]:
            return [
                [
                    event_by_id[event_id]["onset_seconds"],
                    event_by_id[event_id]["offset_seconds"],
                ]
                for event_id in ids
            ]

        evidence_ids = sorted(
            set(anchor_event_ids + answer_event_ids),
            key=lambda event_id: event_by_id[event_id]["onset_seconds"],
        )
        rows.append(
            {
                "schema_version": REAL_QA_FORMAT,
                "id": f"{scene['scene_id']}__{slot_id}",
                "scene_id": scene["scene_id"],
                "scene_family_id": scene["scene_id"],
                "split": scene["selection_partition"],
                "question_index": slot_index,
                "question_type": f"temporal_{slot['relation']}",
                "relation": slot["relation"],
                "question": question,
                "answer": answer,
                "answer_options": options,
                "answer_option_index": options.index(answer),
                "no_evidence": no_evidence,
                "no_evidence_reason": "absent_anchor" if no_evidence else None,
                "query_label": (
                    absent
                    if no_evidence
                    else (
                        query_phrases[anchor_index]
                        if slot["relation"] in ("after", "before")
                        else None
                    )
                ),
                "absent_anchor_positive_support_scene_id": (
                    absent_support_scene_id if no_evidence else None
                ),
                "query_candidate_labels": query_candidates,
                "query_event_ids": query_event_ids,
                "anchor_event_ids": anchor_event_ids,
                "answer_event_ids": answer_event_ids,
                "evidence_event_ids": evidence_ids,
                "anchor_intervals": intervals(anchor_event_ids),
                "answer_intervals": intervals(answer_event_ids),
                "evidence_intervals": intervals(evidence_ids),
                "mention_order_variant": mention_order,
                "audio_path": scene["audio"]["local_path"],
                "audio_sha256": scene["audio"]["local_sha256"],
                "sample_rate": SAMPLE_RATE,
                "num_channels": NUM_CHANNELS,
                "num_samples": NUM_SAMPLES,
                "duration_seconds": DURATION_SECONDS,
                "source": scene["source"],
                "events": events,
                "human_verification": scene["human_verification"],
                "clean_reference_stems_available": False,
                "waveform_sdr_evaluation_allowed": False,
            }
        )
    return rows


def build_real10_manifests(
    qa_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create the authoritative, label-separated QCES-Real-10 views."""

    scoring_payloads: list[dict[str, Any]] = []
    try:
        for row in qa_rows:
            scene_id = row["scene_id"]
            source = row["source"]
            scoring_payload: dict[str, Any] = {
                "schema_version": SCORING_SCHEMA_VERSION,
                "id": row["id"],
                "scene_id": scene_id,
                "scene_family_id": row["scene_family_id"],
                "split": row["split"],
                "question_index": row["question_index"],
                "question_type": row["question_type"],
                "relation": row["relation"],
                "question": row["question"],
                "sample_rate": SAMPLE_RATE,
                "num_channels": NUM_CHANNELS,
                "num_samples": NUM_SAMPLES,
                "duration_seconds": DURATION_SECONDS,
                "mixture_path": f"audio/{scene_id}.wav",
                "mixture_sha256": row["audio_sha256"],
                "inference_record_sha256": "0" * 64,
                "answer_options": list(row["answer_options"]),
                "answer": row["answer"],
                "answer_option_index": row["answer_option_index"],
                "no_evidence": row["no_evidence"],
                "creator_id": source["creator_id"],
                "anchor_event_ids": list(row["anchor_event_ids"]),
                "answer_event_ids": list(row["answer_event_ids"]),
                "evidence_event_ids": list(row["evidence_event_ids"]),
                "anchor_intervals": [
                    list(interval) for interval in row["anchor_intervals"]
                ],
                "answer_intervals": [
                    list(interval) for interval in row["answer_intervals"]
                ],
                "evidence_intervals": [
                    list(interval) for interval in row["evidence_intervals"]
                ],
                "upstream_tacos_split": source["upstream_tacos_split"],
                "clean_reference_stems_available": False,
                "waveform_sdr_evaluation_allowed": False,
            }
            projection = project_scoring_fields_to_inference(scoring_payload)
            scoring_payload["inference_record_sha256"] = inference_record_fingerprint(
                projection
            )
            scoring_payloads.append(scoring_payload)
        scoring_records = parse_scoring_manifest(scoring_payloads)
        inference_records = project_scoring_manifest(scoring_records)
    except (KeyError, TypeError, ValueError) as error:
        raise TacosAuditError(
            f"cannot construct strict QCES-Real-10 manifests: {error}"
        ) from error
    inference_rows = [record.to_dict() for record in inference_records]
    scoring_rows = [record.to_dict() for record in scoring_records]
    if any(
        inference != scoring.to_inference().to_dict()
        for inference, scoring in zip(inference_rows, scoring_records)
    ):
        raise AssertionError("QCES-Real-10 inference projection changed unexpectedly")
    return inference_rows, scoring_rows


def _bind_final_absent_query_support(
    scenes: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only a fixed point whose negative vocabulary is positive in-set."""

    working = [dict(scene) for scene in scenes]
    dropped: list[str] = []
    while working:
        by_id = {scene["scene_id"]: scene for scene in working}
        if len(by_id) != len(working):
            raise TacosAuditError("final real scene IDs are not unique")
        positive_phrases = {
            scene_id: {event["query_event_phrase"] for event in scene.get("events", [])}
            for scene_id, scene in by_id.items()
        }
        assignments: dict[str, tuple[str, str]] = {}
        unsupported: list[str] = []
        for scene in working:
            scene_id = scene["scene_id"]
            assignment: tuple[str, str] | None = None
            candidates = scene.get("jointly_verified_absent_anchor_candidates")
            if not isinstance(candidates, list):
                raise TacosAuditError(
                    f"{scene_id} lacks jointly verified absent candidates"
                )
            for candidate in candidates:
                if not isinstance(candidate, Mapping):
                    raise TacosAuditError(
                        f"{scene_id} has an invalid verified absent candidate"
                    )
                caption = candidate.get("caption")
                support_ids = candidate.get("cross_scene_support_scene_ids")
                if not isinstance(caption, str) or not isinstance(support_ids, list):
                    raise TacosAuditError(
                        f"{scene_id} has invalid absent candidate support"
                    )
                valid_supporters = sorted(
                    support_id
                    for support_id in support_ids
                    if support_id != scene_id
                    and support_id in by_id
                    and by_id[support_id]["selection_partition"]
                    == scene["selection_partition"]
                    and caption in positive_phrases[support_id]
                )
                if valid_supporters:
                    assignment = (caption, valid_supporters[0])
                    break
            if assignment is None:
                unsupported.append(scene_id)
            else:
                assignments[scene_id] = assignment
        if not unsupported:
            bound: list[dict[str, Any]] = []
            for scene in working:
                caption, support_scene_id = assignments[scene["scene_id"]]
                bound.append(
                    {
                        **scene,
                        "selected_absent_anchor_query_phrase": caption,
                        "selected_absent_anchor_positive_support_scene_id": (
                            support_scene_id
                        ),
                    }
                )
            return bound, dropped
        unsupported_set = set(unsupported)
        dropped.extend(unsupported)
        working = [
            scene for scene in working if scene["scene_id"] not in unsupported_set
        ]
    return [], dropped


def canonical_scoring_manifest_fingerprint(
    scoring_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Hash a validated scoring manifest independently of row order."""

    try:
        records = parse_scoring_manifest(scoring_rows)
    except (TypeError, ValueError) as error:
        raise TacosAuditError(
            f"invalid QCES-Real-10 scoring manifest: {error}"
        ) from error
    ordered = sorted(
        (record.to_dict() for record in records), key=lambda row: row["id"]
    )
    return canonical_json_sha256(
        {
            "format": SCORING_MANIFEST_FINGERPRINT_FORMAT,
            "records": ordered,
        }
    )


def canonical_audio_manifest_fingerprint(
    inference_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Bind every released scene to its one manifest-relative waveform."""

    try:
        records = parse_inference_manifest(inference_rows)
    except (TypeError, ValueError) as error:
        raise TacosAuditError(
            f"invalid QCES-Real-10 inference manifest: {error}"
        ) from error
    by_scene: dict[str, dict[str, str]] = {}
    for record in records:
        by_scene.setdefault(
            record.scene_id,
            {
                "scene_id": record.scene_id,
                "mixture_path": record.mixture_path,
                "mixture_sha256": record.mixture_sha256,
            },
        )
    return canonical_json_sha256(
        {
            "format": AUDIO_MANIFEST_FINGERPRINT_FORMAT,
            "scenes": [by_scene[scene_id] for scene_id in sorted(by_scene)],
        }
    )


def _split_inference_rows(
    inference_rows: Sequence[Mapping[str, Any]], split: str
) -> list[dict[str, Any]]:
    if split not in SPLIT_INFERENCE_OUTPUT_NAMES:
        raise TacosAuditError(f"invalid QCES-Real-10 inference split: {split}")
    rows = [dict(row) for row in inference_rows if row.get("split") == split]
    if not rows:
        raise TacosAuditError(f"QCES-Real-10 {split} inference split is empty")
    parse_inference_manifest(rows)
    return rows


def finalize_human_annotations(
    *,
    packet_path: Path,
    audio_receipt_path: Path,
    response_root: Path,
    rater_a_id: str,
    rater_b_id: str,
    adjudicator_id: str | None,
    project_root: Path,
    target_real_dev_scenes: int,
    target_real_test_scenes: int,
    maximum_boundary_difference_seconds: float,
    minimum_interval_iou: float,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    rater_ids = [rater_a_id, rater_b_id] + (
        [adjudicator_id] if adjudicator_id is not None else []
    )
    if any(
        not isinstance(rater_id, str) or not RATER_ID_RE.fullmatch(rater_id)
        for rater_id in rater_ids
    ):
        raise TacosAuditError("rater IDs must match [A-Za-z0-9][A-Za-z0-9_-]{1,31}")
    if len(rater_ids) != len(set(rater_ids)):
        raise TacosAuditError("rater and adjudicator IDs must be distinct")
    if (
        isinstance(target_real_dev_scenes, bool)
        or isinstance(target_real_test_scenes, bool)
        or not isinstance(target_real_dev_scenes, int)
        or not isinstance(target_real_test_scenes, int)
        or target_real_dev_scenes <= 0
        or target_real_test_scenes <= 0
    ):
        raise TacosAuditError("real dev/test scene targets must be positive integers")
    _validate_resolution_thresholds(
        maximum_boundary_difference_seconds,
        minimum_interval_iou,
    )
    packet_fp, audio_fp, tasks = load_bound_tasks(
        packet_path=packet_path,
        audio_receipt_path=audio_receipt_path,
        project_root=project_root,
    )
    for task in tasks:
        _validate_bound_task(task)
    resolved: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for task in tasks:
        left = _read_response(
            response_root=response_root,
            rater_id=rater_a_id,
            task=task,
            packet_fingerprint=packet_fp,
            audio_fingerprint=audio_fp,
        )
        right = _read_response(
            response_root=response_root,
            rater_id=rater_b_id,
            task=task,
            packet_fingerprint=packet_fp,
            audio_fingerprint=audio_fp,
        )
        audit_row: dict[str, Any] = {
            "format": ADJUDICATION_AUDIT_FORMAT,
            "scene_id": task["scene_id"],
            "selection_partition": task["selection_partition"],
            "selection_tier": task["selection_tier"],
            "selection_ordinal": task["selection_ordinal"],
            "independent_rater_ids": [rater_a_id, rater_b_id],
            "independent_responses_present": [
                left is not None,
                right is not None,
            ],
            # `_read_response` returns only schema-v3 responses whose explicit
            # full-listen record is bound to this task's final WAV and PCM SHA.
            "independent_full_listen_confirmed": [
                left is not None,
                right is not None,
            ],
            "independent_scene_decisions": [
                left["scene_decision"] if left is not None else None,
                right["scene_decision"] if right is not None else None,
            ],
            "pair_conflicts": [],
            "adjudicator_id": adjudicator_id,
            "adjudicator_response_present": False,
            "adjudicator_full_listen_confirmed": False,
            "adjudicator_scene_decision": None,
            "resolution_status": "pending",
            "included_in_final_manifest": False,
        }
        if left is None or right is None:
            counters["missing_pair"] += 1
            audit_row["resolution_status"] = "missing_independent_response"
            audit_rows.append(audit_row)
            continue
        counters["paired"] += 1
        counters["validated_independent_full_listen_judgments"] += 2
        scene, conflicts = resolve_rater_pair(
            left,
            right,
            task=task,
            maximum_boundary_difference_seconds=maximum_boundary_difference_seconds,
            minimum_interval_iou=minimum_interval_iou,
        )
        audit_row["pair_conflicts"] = conflicts
        if scene is None and conflicts and adjudicator_id is not None:
            adjudication = _read_response(
                response_root=response_root,
                rater_id=adjudicator_id,
                task=task,
                packet_fingerprint=packet_fp,
                audio_fingerprint=audio_fp,
            )
            audit_row["adjudicator_response_present"] = adjudication is not None
            audit_row["adjudicator_full_listen_confirmed"] = adjudication is not None
            counters["validated_adjudicator_full_listen_judgments"] += int(
                adjudication is not None
            )
            audit_row["adjudicator_scene_decision"] = (
                adjudication["scene_decision"] if adjudication is not None else None
            )
            if adjudication is not None and adjudication["scene_decision"] == "accept":
                scene = scene_from_adjudicator(
                    adjudication, task=task, conflict_reasons=conflicts
                )
                counters["adjudicated"] += 1
        if scene is None:
            counters["unresolved"] += 1
            audit_row["resolution_status"] = "unresolved_or_rejected"
            audit_rows.append(audit_row)
            continue
        resolved.append(scene)
        counters["resolved"] += 1
        if scene["human_verification"]["resolved_by"] == "independent_pair_agreement":
            counters["resolved_independent"] += 1
        audit_row["resolution_status"] = (
            "resolved_by_third_rater_adjudication"
            if scene["human_verification"]["resolved_by"] == "third_rater_adjudication"
            else "resolved_by_independent_pair"
        )
        audit_rows.append(audit_row)

    def choose(partition: str, target: int) -> list[dict[str, Any]]:
        candidates = [
            row for row in resolved if row["selection_partition"] == partition
        ]
        candidates.sort(
            key=lambda row: (
                0 if row["selection_tier"] == "core" else 1,
                row["selection_ordinal"],
                row["scene_id"],
            )
        )
        return candidates[:target]

    selected_dev = choose("real_dev", target_real_dev_scenes)
    selected_test = choose("real_test", target_real_test_scenes)
    final_dev, unsupported_dev_scene_ids = _bind_final_absent_query_support(
        selected_dev
    )
    final_test, unsupported_test_scene_ids = _bind_final_absent_query_support(
        selected_test
    )
    unsupported_absent_support_scene_ids = (
        unsupported_dev_scene_ids + unsupported_test_scene_ids
    )
    final_scenes = final_dev + final_test
    final_scene_ids = {scene["scene_id"] for scene in final_scenes}
    for audit_row in audit_rows:
        audit_row["included_in_final_manifest"] = (
            audit_row["scene_id"] in final_scene_ids
        )
    qa_rows: list[dict[str, Any]] = []
    for global_scene_ordinal, scene in enumerate(final_scenes):
        qa_rows.extend(
            build_scene_questions(scene, global_scene_ordinal=global_scene_ordinal)
        )
    inference_rows, scoring_rows = build_real10_manifests(qa_rows)
    inference_records = parse_inference_manifest(inference_rows)
    inference_manifest_fingerprint = canonical_inference_manifest_fingerprint(
        inference_records
    )
    scoring_manifest_fingerprint = canonical_scoring_manifest_fingerprint(scoring_rows)
    audio_manifest_fingerprint = canonical_audio_manifest_fingerprint(inference_rows)
    split_inference_rows = {
        split: _split_inference_rows(inference_rows, split)
        for split in SPLIT_INFERENCE_OUTPUT_NAMES
    }
    split_inference_fingerprints = {
        split: canonical_inference_manifest_fingerprint(parse_inference_manifest(rows))
        for split, rows in split_inference_rows.items()
    }
    split_union_rows = sorted(
        [row for rows in split_inference_rows.values() for row in rows],
        key=lambda row: row["id"],
    )
    full_sorted_rows = sorted(inference_rows, key=lambda row: row["id"])
    split_union_is_exact = split_union_rows == full_sorted_rows
    final_source_ids = [scene["source"]["freesound_id"] for scene in final_scenes]
    final_creator_ids = [scene["source"]["creator_id"] for scene in final_scenes]
    dev_sources = {scene["source"]["freesound_id"] for scene in final_dev}
    test_sources = {scene["source"]["freesound_id"] for scene in final_test}
    dev_creators = {scene["source"]["creator_id"] for scene in final_dev}
    test_creators = {scene["source"]["creator_id"] for scene in final_test}
    dev_test_source_overlap = len(dev_sources & test_sources)
    dev_test_creator_overlap = len(dev_creators & test_creators)
    expected_question_count = (target_real_dev_scenes + target_real_test_scenes) * len(
        QUESTION_SLOTS
    )
    gate_failures: list[str] = []
    if len(final_dev) != target_real_dev_scenes:
        gate_failures.append("real_dev_scene_target_not_met")
    if len(final_test) != target_real_test_scenes:
        gate_failures.append("real_test_scene_target_not_met")
    if unsupported_absent_support_scene_ids:
        gate_failures.append("final_negative_query_lacks_positive_in_split_support")
    if len(final_source_ids) != len(set(final_source_ids)):
        gate_failures.append("final_source_ids_not_unique")
    if len(final_creator_ids) != len(set(final_creator_ids)):
        gate_failures.append("final_creator_ids_not_unique")
    if dev_test_source_overlap:
        gate_failures.append("real_dev_test_source_overlap")
    if dev_test_creator_overlap:
        gate_failures.append("real_dev_test_creator_overlap")
    if len(qa_rows) != expected_question_count:
        gate_failures.append("real_question_target_not_met")
    if len(inference_rows) != expected_question_count:
        gate_failures.append("real10_inference_question_target_not_met")
    if len(scoring_rows) != expected_question_count:
        gate_failures.append("real10_scoring_question_target_not_met")
    if not split_union_is_exact:
        gate_failures.append("split_inference_union_mismatch")
    gate_passed = not gate_failures
    compliance = {
        "format": FINALIZATION_FORMAT,
        "input_contract": {
            "annotation_packet_schema_version": PACKET_FORMAT,
            "derived_audio_receipt_format": AUDIO_RECEIPT_FORMAT,
        },
        "packet_fingerprint": packet_fp,
        "audio_receipt_fingerprint": audio_fp,
        "scene_manifest_fingerprint": canonical_json_sha256(final_scenes),
        "qa_manifest_fingerprint": canonical_json_sha256(qa_rows),
        "adjudication_audit_fingerprint": canonical_json_sha256(audit_rows),
        "qces_real10_inference_manifest_fingerprint": (inference_manifest_fingerprint),
        "qces_real10_scoring_manifest_fingerprint": scoring_manifest_fingerprint,
        "qces_real10_audio_manifest_fingerprint": audio_manifest_fingerprint,
        "qces_real10_real_dev_inference_manifest_fingerprint": (
            split_inference_fingerprints["real_dev"]
        ),
        "qces_real10_real_test_inference_manifest_fingerprint": (
            split_inference_fingerprints["real_test"]
        ),
        "qces_real10_split_inference_union_is_exact_full_manifest": (
            split_union_is_exact
        ),
        "qces_real10_inference_is_exact_scoring_projection": True,
        "authoritative_manifests": {
            "inference": {
                "filename": AUTHORITATIVE_OUTPUT_NAMES[0],
                "schema_version": INFERENCE_SCHEMA_VERSION,
            },
            "scoring": {
                "filename": AUTHORITATIVE_OUTPUT_NAMES[1],
                "schema_version": SCORING_SCHEMA_VERSION,
            },
            "inference_by_split": {
                split: {
                    "filename": filename,
                    "schema_version": INFERENCE_SCHEMA_VERSION,
                    "manifest_fingerprint": split_inference_fingerprints[split],
                }
                for split, filename in SPLIT_INFERENCE_OUTPUT_NAMES.items()
            },
        },
        "human_verification_gate_passed": gate_passed,
        "submission_real_data_gate_passed": gate_passed,
        "paper_result_eligible": gate_passed,
        "gate_failures": gate_failures,
        "thresholds": {
            "maximum_boundary_difference_seconds_↓": maximum_boundary_difference_seconds,
            "minimum_interval_iou_↑": minimum_interval_iou,
            "required_independent_raters_↑": 2,
            "exact_final_wav_full_listen_required": True,
        },
        "metrics": {
            "candidate_scenes_↑": len(tasks),
            "dual_rater_response_scenes_↑": counters["paired"],
            "validated_independent_full_listen_judgments_↑": counters[
                "validated_independent_full_listen_judgments"
            ],
            "validated_adjudicator_full_listen_judgments_↑": counters[
                "validated_adjudicator_full_listen_judgments"
            ],
            "resolved_pair_scenes_↑": counters["resolved_independent"],
            "resolved_total_scenes_↑": counters["resolved"],
            "third_rater_adjudications_↓": counters["adjudicated"],
            "missing_response_pairs_↓": counters["missing_pair"],
            "unresolved_or_rejected_scenes_↓": counters["unresolved"],
            "final_real_dev_scenes_↑": len(final_dev),
            "final_real_test_scenes_↑": len(final_test),
            "final_real_dev_questions_↑": len(final_dev) * len(QUESTION_SLOTS),
            "final_real_test_questions_↑": len(final_test) * len(QUESTION_SLOTS),
            "authoritative_inference_rows_↑": len(inference_rows),
            "authoritative_scoring_rows_↑": len(scoring_rows),
            "authoritative_real_dev_inference_rows_↑": len(
                split_inference_rows["real_dev"]
            ),
            "authoritative_real_test_inference_rows_↑": len(
                split_inference_rows["real_test"]
            ),
            "authoritative_unique_audio_files_↑": len(final_scenes),
            "questions_per_authoritative_scene_↑": len(QUESTION_SLOTS),
            "final_unique_sources_↑": len(set(final_source_ids)),
            "final_unique_creators_↑": len(set(final_creator_ids)),
            "final_dev_test_source_overlap_↓": dev_test_source_overlap,
            "final_dev_test_creator_overlap_↓": dev_test_creator_overlap,
            "resolved_scenes_dropped_for_negative_support_↓": len(
                unsupported_absent_support_scene_ids
            ),
            "final_negative_query_positive_support_violations_↓": 0,
            "final_question_only_negative_vocabulary_shortcut_violations_↓": 0,
            "model_outputs_visible_during_annotation_↓": 0,
            "waveform_sdr_claims_on_tacos_↓": 0,
        },
        "gate_reason": (
            "Two-pass human, source, creator, audio, and question-count gates passed."
            if gate_passed
            else "Real-data gate failed: " + ", ".join(gate_failures) + "."
        ),
    }
    return compliance, final_scenes, qa_rows, audit_rows


def _source_audio_rows(
    qa_rows: Sequence[Mapping[str, Any]],
    inference_rows: Sequence[Mapping[str, Any]],
    *,
    project_root: Path,
    output_dir: Path,
) -> list[tuple[Path, Path, str]]:
    inference_by_scene: dict[str, Mapping[str, Any]] = {}
    for row in inference_rows:
        inference_by_scene.setdefault(row["scene_id"], row)
    source_by_scene: dict[str, tuple[str, str]] = {}
    for row in qa_rows:
        scene_id = row.get("scene_id")
        local_path = row.get("audio_path")
        sha256 = row.get("audio_sha256")
        if not all(isinstance(value, str) for value in (scene_id, local_path, sha256)):
            raise TacosAuditError("legacy QA row lacks receipt-v2 audio provenance")
        assert isinstance(scene_id, str)
        assert isinstance(local_path, str)
        assert isinstance(sha256, str)
        identity = (local_path, sha256)
        previous = source_by_scene.setdefault(scene_id, identity)
        if previous != identity:
            raise TacosAuditError(f"{scene_id} has inconsistent source audio identity")

    root = project_root.resolve()
    destination_root = output_dir.resolve()
    materializations: list[tuple[Path, Path, str]] = []
    for scene_id in sorted(inference_by_scene):
        inference = inference_by_scene[scene_id]
        if scene_id not in source_by_scene:
            raise TacosAuditError(f"{scene_id} lacks source audio provenance")
        local_path, source_sha256 = source_by_scene[scene_id]
        parts = local_path.split("/")
        relative = PurePosixPath(local_path)
        if (
            "\\" in local_path
            or relative.is_absolute()
            or relative.suffix.casefold() != ".wav"
            or any(part in {"", ".", ".."} or ":" in part for part in parts)
        ):
            raise TacosAuditError(f"{scene_id} receipt audio path is unsafe")
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise TacosAuditError(
                f"{scene_id} receipt audio escapes project root"
            ) from error
        mixture_sha256 = inference["mixture_sha256"]
        if source_sha256 != mixture_sha256:
            raise TacosAuditError(f"{scene_id} receipt/manifest SHA256 mismatch")
        destination = destination_root / PurePosixPath(inference["mixture_path"])
        resolved_parent = destination.parent.resolve()
        try:
            resolved_parent.relative_to(destination_root)
        except ValueError as error:
            raise TacosAuditError(
                f"{scene_id} released audio destination escapes output directory"
            ) from error
        if destination.parent.is_symlink():
            raise TacosAuditError(
                f"{scene_id} released audio directory must not be a symlink"
            )
        materializations.append((source, destination, source_sha256))
    return materializations


def _verify_audio_materialization(
    source: Path, destination: Path, expected_sha256: str
) -> None:
    if not source.is_file() or sha256_file(source) != expected_sha256:
        raise TacosAuditError(f"source audio hash mismatch: {source}")
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file():
            raise TacosAuditError(
                f"released audio destination is not a regular file: {destination}"
            )
        if sha256_file(destination) != expected_sha256:
            raise TacosAuditError(
                f"refusing to overwrite mismatched released audio: {destination}"
            )


def _copy_audio_exclusive(
    source: Path, destination: Path, expected_sha256: str
) -> None:
    _verify_audio_materialization(source, destination, expected_sha256)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, output)
            output.flush()
            os.fsync(output.fileno())
        if sha256_file(temporary) != expected_sha256:
            raise TacosAuditError(f"staged released audio hash mismatch: {destination}")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _verify_audio_materialization(source, destination, expected_sha256)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_finalization_artifacts(
    *,
    output_dir: Path,
    project_root: Path,
    compliance: Mapping[str, Any],
    scenes: Sequence[Mapping[str, Any]],
    qa_rows: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    overwrite: bool,
) -> None:
    inference_rows, scoring_rows = build_real10_manifests(qa_rows)
    inference_records = parse_inference_manifest(inference_rows)
    split_rows = {
        split: _split_inference_rows(inference_rows, split)
        for split in SPLIT_INFERENCE_OUTPUT_NAMES
    }
    fingerprints = {
        "scene_manifest_fingerprint": canonical_json_sha256(scenes),
        "qa_manifest_fingerprint": canonical_json_sha256(qa_rows),
        "adjudication_audit_fingerprint": canonical_json_sha256(audit_rows),
        "qces_real10_inference_manifest_fingerprint": (
            canonical_inference_manifest_fingerprint(inference_records)
        ),
        "qces_real10_scoring_manifest_fingerprint": (
            canonical_scoring_manifest_fingerprint(scoring_rows)
        ),
        "qces_real10_audio_manifest_fingerprint": (
            canonical_audio_manifest_fingerprint(inference_rows)
        ),
        "qces_real10_real_dev_inference_manifest_fingerprint": (
            canonical_inference_manifest_fingerprint(
                parse_inference_manifest(split_rows["real_dev"])
            )
        ),
        "qces_real10_real_test_inference_manifest_fingerprint": (
            canonical_inference_manifest_fingerprint(
                parse_inference_manifest(split_rows["real_test"])
            )
        ),
    }
    if compliance.get("format") != FINALIZATION_FORMAT:
        raise TacosAuditError("finalization compliance format mismatch")
    if compliance.get("qces_real10_inference_is_exact_scoring_projection") is not True:
        raise TacosAuditError("finalization compliance lacks exact projection proof")
    for field, actual in fingerprints.items():
        if compliance.get(field) != actual:
            raise TacosAuditError(f"{field} does not match finalization rows")
    if (
        compliance.get("qces_real10_split_inference_union_is_exact_full_manifest")
        is not True
    ):
        raise TacosAuditError("finalization lacks exact split-inference union proof")
    paths = (
        output_dir / "qces_v5_tacos_human_compliance.json",
        *(output_dir / name for name in LEGACY_OUTPUT_NAMES),
        *(output_dir / name for name in AUTHORITATIVE_OUTPUT_NAMES),
        *(
            output_dir / SPLIT_INFERENCE_OUTPUT_NAMES[split]
            for split in ("real_dev", "real_test")
        ),
    )
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError(
                "finalization outputs exist; use --overwrite: "
                + ", ".join(map(str, existing))
            )
    materializations = _source_audio_rows(
        qa_rows,
        inference_rows,
        project_root=project_root,
        output_dir=output_dir,
    )
    for source, destination, expected_sha256 in materializations:
        _verify_audio_materialization(source, destination, expected_sha256)
    for source, destination, expected_sha256 in materializations:
        _copy_audio_exclusive(source, destination, expected_sha256)

    atomic_json(paths[0], compliance, overwrite=overwrite)
    atomic_jsonl(paths[1], scenes, overwrite=overwrite)
    atomic_jsonl(paths[2], qa_rows, overwrite=overwrite)
    atomic_jsonl(paths[3], audit_rows, overwrite=overwrite)
    atomic_jsonl(paths[4], inference_rows, overwrite=overwrite)
    atomic_jsonl(paths[5], scoring_rows, overwrite=overwrite)
    atomic_jsonl(paths[6], split_rows["real_dev"], overwrite=overwrite)
    atomic_jsonl(paths[7], split_rows["real_test"], overwrite=overwrite)
