"""First-relation candidate utilities for QCES-v6 evidence grounding.

The original QCES-v6 pair reranker represents a ``first`` question candidate
with a single event: the event whose label is predicted as the answer.  That is
enough for answer accuracy, but it is not enough for evidence IoU because the
gold evidence for questions such as "Which begins sooner, A or B?" contains
both compared events.

This module keeps the existing candidate representation for ``after`` and
``before`` questions, but represents ``first`` candidates as a pair of event
proposals: one proposal from each compared label.  The answer label is the
label of the earlier proposal in the pair.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.event_proposals import EventProposal
from mixi_understanding.scripts.calibrate_qces_v6_pair_reranker import (
    top_candidate_by_record,
)
from mixi_understanding.scripts.calibrate_qces_v6_span_iou_refiner import (
    PreparedSplit,
)
from mixi_understanding.scripts.train_qces_v6_pair_reranker import (
    Candidate,
    candidate_features as base_candidate_features,
    prepare_split,
    scores_for,
)
from mixi_understanding.scripts.train_qces_v6_spanaware_pair_reranker import (
    candidate_span_iou,
)


def candidate_features_firstpair(
    record: QCESV5Record,
    parsed: Any,
    proposals: Sequence[EventProposal],
    *,
    max_neighbors: int,
) -> list[tuple[str, tuple[EventProposal, ...], tuple[float, ...]]]:
    """Build candidates, using two-event evidence for ``first`` questions."""

    if getattr(parsed, "relation", None) != "first":
        return base_candidate_features(
            record,
            parsed,
            proposals,
            max_neighbors=max_neighbors,
        )

    labels = list(dict.fromkeys(getattr(parsed, "candidate_labels", ()) or ()))
    if len(labels) < 2:
        return base_candidate_features(
            record,
            parsed,
            proposals,
            max_neighbors=max_neighbors,
        )

    rel_after = 0.0
    rel_before = 0.0
    rel_first = 1.0
    total = len(proposals)

    label_props: dict[str, list[EventProposal]] = {}
    for label in labels:
        props = sorted(
            [item for item in proposals if item.label == label],
            key=lambda item: (item.onset_seconds, item.offset_seconds),
        )
        label_props[label] = props[:3]

    proto_rows: list[
        tuple[str, tuple[EventProposal, ...], tuple[float, ...], float]
    ] = []
    for left_idx, left_label in enumerate(labels):
        for right_label in labels[left_idx + 1 :]:
            left_props = label_props.get(left_label, [])
            right_props = label_props.get(right_label, [])
            for left_rank, left_event in enumerate(left_props, start=1):
                for right_rank, right_event in enumerate(right_props, start=1):
                    if left_event.onset_seconds <= right_event.onset_seconds:
                        earlier = left_event
                        later = right_event
                        answer_rank = left_rank
                        other_rank = right_rank
                        answer_label = left_label
                    else:
                        earlier = right_event
                        later = left_event
                        answer_rank = right_rank
                        other_rank = left_rank
                        answer_label = right_label

                    gap = abs(later.onset_seconds - earlier.onset_seconds)
                    events = tuple(
                        sorted((left_event, right_event), key=lambda item: item.onset_seconds)
                    )
                    both_first_occurrences = float(left_rank == 1 and right_rank == 1)
                    rank_abs_error = float(abs(left_rank - 1) + abs(right_rank - 1))
                    proto_rows.append(
                        (
                            answer_label,
                            events,
                            (
                                rel_after,
                                rel_before,
                                rel_first,
                                float(earlier.confidence),
                                float(later.confidence),
                                float(earlier.confidence * later.confidence),
                                float(earlier.onset_seconds / 10.0),
                                float(later.onset_seconds / 10.0),
                                float(earlier.duration_seconds),
                                float(later.duration_seconds),
                                float(answer_rank),
                                rank_abs_error,
                                both_first_occurrences,
                                float(len(left_props) + len(right_props)),
                                float(total),
                                0.0,  # filled with gap rank below
                                float(gap),
                                float(np.log1p(gap)),
                                0.0,
                            ),
                            float(gap),
                        )
                    )

    if not proto_rows:
        return []

    gap_rank = {
        index: rank + 1
        for rank, (index, _row) in enumerate(
            sorted(enumerate(proto_rows), key=lambda item: item[1][3])
        )
    }
    rows: list[tuple[str, tuple[EventProposal, ...], tuple[float, ...]]] = []
    for index, (answer_label, events, features, _gap) in enumerate(proto_rows):
        values = list(features)
        values[15] = float(gap_rank[index])
        rows.append((answer_label, events, tuple(values)))
    return rows


def split_candidates_firstpair(
    prepared: tuple[
        list[QCESV5Record],
        dict[str, Any],
        dict[str, tuple[str, ...]],
        dict[str, dict[tuple[str, tuple[str, ...]], list[EventProposal]]],
    ],
    *,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, list[Candidate], dict[str, QCESV5Record]]:
    records, parsed, labels_by_id, decoded_by_relation = prepared
    features: list[tuple[float, ...]] = []
    targets: list[int] = []
    candidates: list[Candidate] = []
    records_by_id = {record.sample_id: record for record in records}
    for record in records:
        question = parsed[record.sample_id]
        if question.relation not in decoded_by_relation:
            continue
        proposals = decoded_by_relation[question.relation][
            (record.scene_id, labels_by_id[record.sample_id])
        ]
        for answer_label, events, feature_row in candidate_features_firstpair(
            record,
            question,
            proposals,
            max_neighbors=max_neighbors,
        ):
            target = int((not record.no_evidence) and answer_label == record.answer)
            candidates.append(
                Candidate(
                    sample_id=record.sample_id,
                    answer_label=answer_label,
                    events=events,
                    features=feature_row,
                    target=target,
                )
            )
            features.append(feature_row)
            targets.append(target)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.int8),
        candidates,
        records_by_id,
    )


def split_candidates_spanaware_firstpair(
    prepared: tuple[
        list[QCESV5Record],
        dict[str, Any],
        dict[str, tuple[str, ...]],
        dict[str, dict[tuple[str, tuple[str, ...]], list[EventProposal]]],
    ],
    *,
    max_neighbors: int,
    positive_iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[Candidate], dict[str, QCESV5Record], np.ndarray]:
    records, parsed, labels_by_id, decoded_by_relation = prepared
    features: list[tuple[float, ...]] = []
    targets: list[int] = []
    candidate_ious: list[float] = []
    candidates: list[Candidate] = []
    records_by_id = {record.sample_id: record for record in records}
    grouped_indices: dict[str, list[int]] = {}

    for record in records:
        question = parsed[record.sample_id]
        if question.relation not in decoded_by_relation:
            continue
        proposals = decoded_by_relation[question.relation][
            (record.scene_id, labels_by_id[record.sample_id])
        ]
        for answer_label, events, feature_row in candidate_features_firstpair(
            record,
            question,
            proposals,
            max_neighbors=max_neighbors,
        ):
            iou = 0.0 if record.no_evidence else candidate_span_iou(record, events)
            label_ok = (not record.no_evidence) and answer_label == record.answer
            target = int(label_ok and iou >= positive_iou_threshold)
            candidate = Candidate(
                sample_id=record.sample_id,
                answer_label=answer_label,
                events=events,
                features=feature_row,
                target=target,
            )
            grouped_indices.setdefault(record.sample_id, []).append(len(candidates))
            candidates.append(candidate)
            features.append(feature_row)
            targets.append(target)
            candidate_ious.append(iou)

    targets_arr = np.asarray(targets, dtype=np.int8)
    ious_arr = np.asarray(candidate_ious, dtype=np.float32)
    for record in records:
        if record.no_evidence:
            continue
        indices = grouped_indices.get(record.sample_id, [])
        if not indices or bool(targets_arr[indices].any()):
            continue
        label_indices = [
            idx for idx in indices if candidates[idx].answer_label == record.answer
        ]
        if not label_indices:
            continue
        best = max(label_indices, key=lambda idx: float(ious_arr[idx]))
        if float(ious_arr[best]) > 0.0:
            targets_arr[best] = 1
            old = candidates[best]
            candidates[best] = Candidate(
                sample_id=old.sample_id,
                answer_label=old.answer_label,
                events=old.events,
                features=old.features,
                target=1,
            )

    return (
        np.asarray(features, dtype=np.float32),
        targets_arr,
        candidates,
        records_by_id,
        ious_arr,
    )


def prepare_named_split_firstpair(
    *,
    name: str,
    manifest: Path,
    cache: Path,
    dataset_config: Path,
    proposal_head: Path,
    relation_thresholds: Mapping[str, float],
    model: Any,
    max_neighbors: int,
    device: torch.device,
) -> PreparedSplit:
    prepared = prepare_split(
        manifest=manifest.resolve(),
        cache_path=cache.resolve(),
        dataset_config=dataset_config.resolve(),
        proposal_head=proposal_head.resolve(),
        thresholds=relation_thresholds,
        device=device,
    )
    features, _targets, candidates, records_by_id = split_candidates_firstpair(
        prepared,
        max_neighbors=max_neighbors,
    )
    scores = scores_for(model, features)
    return PreparedSplit(
        name=name,
        records=list(prepared[0]),
        candidates=candidates,
        scores=scores,
        top=top_candidate_by_record(candidates, scores),
        records_by_id=records_by_id,
    )
