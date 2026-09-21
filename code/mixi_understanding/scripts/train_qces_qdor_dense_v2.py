#!/usr/bin/env python3
"""Train Q-DOR-v2 with dense-gold binding and leakage-safe model selection.

This file is a sidecar.  It deliberately leaves the historical Q-DOR model
and trainer untouched.  The protocol enforced here is stricter in four ways:

* every explicit QA row is bound back to the dense scene's gold events;
* all scene event onsets supervise a frame-by-class onset head;
* threshold calibration and checkpoint selection use disjoint scene sets;
* checkpoints are selected by strict positive/NONE evidence performance at
  IoU >= 0.50; the historical IoU >= 0.30 score is diagnostic only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.benchmark_integrity import BalancedQAItem, semantic_events
from mixi_understanding.qces.dense_event_qa import (
    FRAME_HOP_SECONDS,
    RELATION_AFTER,
    RELATION_BEFORE,
    intervals_to_40ms_mask,
)
from mixi_understanding.qces.dense_event_qa_v2 import (
    DenseTemporalReasonerV2,
    DenseTemporalReasonerV2Config,
    DenseTemporalThresholdsV2,
    dense_temporal_reasoner_v2_loss,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    FIXED_AUDIO_SECONDS,
    NUM_FRAMES,
    DenseFeatureStore,
    _atomic_json,
    _atomic_torch,
    _device,
    _mask_padding,
    _sha256_file,
    load_explicit_qa_manifest,
    load_oracle_gate,
    load_scene_list,
    teacher_forcing_for_epoch,
)


RECEIPT_FORMAT_V2 = "qces_qdor_dense_training_receipt_v2"
CHECKPOINT_FORMAT_V2 = "qces_qdor_dense_checkpoint_v2"
DIAGNOSTIC_STRICT_IOU = 0.30
PRIMARY_STRICT_IOU = 0.50
STRICT_METRIC_SCHEMA_V2 = "qces_qdor_strict_evidence_metrics_v2_iou30_iou50"


@dataclass(frozen=True)
class BoundQAItemV2:
    """QA row whose target identities were verified against dense gold."""

    qa: BalancedQAItem
    anchor_event_id: str
    anchor_event_index: int
    answer_event_id: str | None
    answer_event_index: int | None


def _event_interval(event: Mapping[str, Any]) -> tuple[float, float]:
    return float(event["onset_seconds"]), float(event["offset_seconds"])


def _interval_equal(
    left: tuple[float, float] | None,
    right: tuple[float, float] | None,
    *,
    tolerance: float = 1e-7,
) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left[0], right[0], abs_tol=tolerance) and math.isclose(
        left[1], right[1], abs_tol=tolerance
    )


def bind_qa_items_to_dense_gold(
    items: Sequence[BalancedQAItem],
    store: DenseFeatureStore,
    *,
    max_ordinal: int,
) -> list[BoundQAItemV2]:
    """Fail closed unless every QA target is entailed by dense scene gold.

    In particular, ``immediately before/after`` must point to the adjacent
    onset-sorted semantic event.  An ordinal beyond the actual number of
    occurrences is rejected; it is never clamped to a valid occurrence.
    """

    if max_ordinal < 1:
        raise ValueError("max_ordinal must be positive")
    label_to_id = {label: index for index, label in enumerate(store.labels or [])}
    bound: list[BoundQAItemV2] = []
    for item in items:
        events = semantic_events(store.metadata(item.scene_id))
        if not events:
            raise ValueError(f"{item.item_id}: dense scene has no semantic events")
        for event in events:
            label = str(event["label"])
            if label not in label_to_id:
                raise ValueError(f"{item.item_id}: dense event label is outside ontology: {label}")
            if event.get("label_id") is not None and int(event["label_id"]) != label_to_id[label]:
                raise ValueError(f"{item.item_id}: dense label_id disagrees with ontology")

        matching_anchor = [
            index
            for index, event in enumerate(events)
            if str(event["label"]) == item.anchor_label
            and _interval_equal(_event_interval(event), item.gold_anchor_interval)
        ]
        if len(matching_anchor) != 1:
            raise ValueError(
                f"{item.item_id}: anchor must match exactly one dense gold event; "
                f"matched={len(matching_anchor)}"
            )
        anchor_index = matching_anchor[0]
        actual_ordinal = 1 + sum(
            str(event["label"]) == item.anchor_label for event in events[:anchor_index]
        )
        total_occurrences = sum(str(event["label"]) == item.anchor_label for event in events)
        if item.anchor_ordinal > total_occurrences:
            raise ValueError(
                f"{item.item_id}: requested ordinal {item.anchor_ordinal} exceeds "
                f"{total_occurrences}; ordinal cannot be silently clamped"
            )
        if item.anchor_ordinal > max_ordinal:
            raise ValueError(
                f"{item.item_id}: ordinal {item.anchor_ordinal} exceeds model max_ordinal={max_ordinal}"
            )
        if actual_ordinal != item.anchor_ordinal:
            raise ValueError(
                f"{item.item_id}: ordinal mismatch QA={item.anchor_ordinal} dense={actual_ordinal}"
            )

        if item.relation == "before":
            expected_index = anchor_index - 1 if anchor_index > 0 else None
            expected_verification = (0.0, item.gold_anchor_interval[0])
            expected_none_evidence = (0.0, item.gold_anchor_interval[1])
        elif item.relation == "after":
            expected_index = anchor_index + 1 if anchor_index + 1 < len(events) else None
            expected_verification = (item.gold_anchor_interval[1], FIXED_AUDIO_SECONDS)
            expected_none_evidence = (item.gold_anchor_interval[0], FIXED_AUDIO_SECONDS)
        else:
            raise ValueError(f"{item.item_id}: unsupported relation {item.relation!r}")

        if item.no_evidence:
            if expected_index is not None:
                raise ValueError(
                    f"{item.item_id}: marked NONE but an immediate dense event exists"
                )
            if item.answer_label is not None or item.gold_answer_interval is not None:
                raise ValueError(f"{item.item_id}: NONE row carries an answer target")
            if not _interval_equal(item.gold_verification_interval, expected_verification):
                raise ValueError(f"{item.item_id}: NONE verification window disagrees with dense gold")
            if len(item.gold_evidence_intervals) != 1 or not _interval_equal(
                item.gold_evidence_intervals[0], expected_none_evidence
            ):
                raise ValueError(f"{item.item_id}: NONE evidence is not anchor+verification")
            answer_index = None
        else:
            if expected_index is None:
                raise ValueError(f"{item.item_id}: positive row has no immediate dense event")
            answer = events[expected_index]
            if str(answer["label"]) != str(item.answer_label):
                raise ValueError(
                    f"{item.item_id}: answer label is not the immediate dense event"
                )
            if not _interval_equal(_event_interval(answer), item.gold_answer_interval):
                raise ValueError(
                    f"{item.item_id}: answer interval is not the immediate dense event"
                )
            if item.gold_verification_interval is not None:
                raise ValueError(f"{item.item_id}: positive row carries a NONE verification window")
            if tuple(item.gold_evidence_intervals) != (
                item.gold_anchor_interval,
                item.gold_answer_interval,
            ):
                raise ValueError(f"{item.item_id}: positive evidence is not anchor+answer")
            answer_index = expected_index

        bound.append(
            BoundQAItemV2(
                qa=item,
                anchor_event_id=str(events[anchor_index].get("event_id") or anchor_index),
                anchor_event_index=anchor_index,
                answer_event_id=(
                    None
                    if answer_index is None
                    else str(events[answer_index].get("event_id") or answer_index)
                ),
                answer_event_index=answer_index,
            )
        )
    return bound


def load_bound_qa_manifest_v2(
    path: Path,
    *,
    allowed_scene_ids: Sequence[str],
    store: DenseFeatureStore,
    max_ordinal: int,
) -> list[BoundQAItemV2]:
    items = load_explicit_qa_manifest(path, allowed_scene_ids=allowed_scene_ids)
    return bind_qa_items_to_dense_gold(items, store, max_ordinal=max_ordinal)


def dense_scene_onset_targets(
    events: Sequence[Mapping[str, Any]],
    *,
    label_to_id: Mapping[str, int],
    valid_frames: int,
    num_frames: int = NUM_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create exact frame/class onset supervision from every scene event."""

    if not 1 <= valid_frames <= num_frames:
        raise ValueError("valid_frames is outside the dense grid")
    class_mask = torch.zeros(num_frames, len(label_to_id), dtype=torch.float32)
    for event in semantic_events({"events": events}):
        label = str(event["label"])
        if label not in label_to_id:
            raise ValueError(f"gold onset label is outside ontology: {label}")
        onset = float(event["onset_seconds"])
        frame = int(math.floor(onset / FRAME_HOP_SECONDS + 1e-8))
        if frame < 0 or frame >= valid_frames:
            raise ValueError(f"gold onset {onset:.6f}s is outside valid audio")
        class_mask[frame, label_to_id[label]] = 1.0
    return class_mask, class_mask.amax(dim=1)


class DenseQADatasetV2(Dataset[dict[str, Any]]):
    """Dense feature dataset with bound QA and scene-level onset targets."""

    def __init__(
        self,
        store: DenseFeatureStore,
        items: Sequence[BoundQAItemV2],
        *,
        preload: bool,
    ) -> None:
        if not items:
            raise ValueError("Q-DOR-v2 dataset cannot be empty")
        self.store = store
        self.items = list(items)
        self.label_to_id = {label: index for index, label in enumerate(store.labels or [])}
        used_scene_ids = list(dict.fromkeys(item.qa.scene_id for item in self.items))
        missing = [scene_id for scene_id in used_scene_ids if scene_id not in store.scene_ids]
        if missing:
            raise ValueError(f"QA scenes absent from dense indexes: {missing[:5]}")
        self.preloaded = store.preload(used_scene_ids) if preload else None
        self.onset_targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for scene_id in used_scene_ids:
            metadata = store.metadata(scene_id)
            self.onset_targets[scene_id] = dense_scene_onset_targets(
                metadata["gold_events"],
                label_to_id=self.label_to_id,
                valid_frames=int(metadata["valid_frames"]),
            )

    def __len__(self) -> int:
        return len(self.items)

    def _dense(self, scene_id: str) -> dict[str, Any]:
        if self.preloaded is not None:
            return self.preloaded[scene_id]
        return self.store.get(scene_id)

    def __getitem__(self, index: int) -> dict[str, Any]:
        bound = self.items[index]
        item = bound.qa
        dense = self._dense(item.scene_id)
        valid_frames = int(dense["valid_frames"])
        valid = torch.arange(NUM_FRAMES) < valid_frames
        anchor = intervals_to_40ms_mask([item.gold_anchor_interval], NUM_FRAMES).bool() & valid
        answer_intervals = [] if item.gold_answer_interval is None else [item.gold_answer_interval]
        answer = intervals_to_40ms_mask(answer_intervals, NUM_FRAMES).bool() & valid
        verification_intervals = (
            [] if item.gold_verification_interval is None else [item.gold_verification_interval]
        )
        verification = intervals_to_40ms_mask(verification_intervals, NUM_FRAMES).bool() & valid
        evidence = intervals_to_40ms_mask(item.gold_evidence_intervals, NUM_FRAMES).bool() & valid
        class_onset, union_onset = self.onset_targets[item.scene_id]
        return {
            "features": dense["features"].to(torch.float32),
            "detector_logits": dense["logits"].to(torch.float32),
            "valid_mask": valid,
            "anchor_label": self.label_to_id[item.anchor_label],
            "relation_id": RELATION_BEFORE if item.relation == "before" else RELATION_AFTER,
            "ordinal": int(item.anchor_ordinal),
            "gold_answer_label": (
                -1 if item.no_evidence else self.label_to_id[str(item.answer_label)]
            ),
            "gold_anchor_mask": anchor.to(torch.float32),
            "gold_answer_mask": answer.to(torch.float32),
            "gold_verification_mask": verification.to(torch.float32),
            "gold_evidence_mask": evidence.to(torch.float32),
            "gold_class_onset_mask": class_onset,
            "gold_union_onset_mask": union_onset,
            "item_id": item.item_id,
            "scene_id": item.scene_id,
        }


_TENSOR_KEYS_V2 = (
    "features",
    "detector_logits",
    "valid_mask",
    "anchor_label",
    "relation_id",
    "ordinal",
    "gold_answer_label",
    "gold_anchor_mask",
    "gold_answer_mask",
    "gold_verification_mask",
    "gold_evidence_mask",
    "gold_class_onset_mask",
    "gold_union_onset_mask",
)


def collate_dense_qa_v2(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty batch")
    batch: dict[str, Any] = {}
    for key in _TENSOR_KEYS_V2:
        values = [row[key] for row in rows]
        batch[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)
    batch["item_id"] = [str(row["item_id"]) for row in rows]
    batch["scene_id"] = [str(row["scene_id"]) for row in rows]
    return batch


def class_statistics_v2(
    dataset: DenseQADatasetV2,
    num_classes: int,
    *,
    beta: float,
    smoothing: float,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if not 0.0 <= beta < 1.0:
        raise ValueError("class-balance beta must be in [0,1)")
    if smoothing <= 0:
        raise ValueError("prior smoothing must be positive")
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for bound in dataset.items:
        if not bound.qa.no_evidence:
            counts[dataset.label_to_id[str(bound.qa.answer_label)]] += 1
    if counts.sum() == 0:
        raise ValueError("training QA has no answerable examples")
    priors = (counts + smoothing) / (counts.sum() + smoothing * num_classes)
    if beta == 0:
        weights = torch.ones_like(counts)
    else:
        effective = 1.0 - torch.pow(torch.full_like(counts, beta), counts.clamp_min(1.0))
        weights = (1.0 - beta) / effective.clamp_min(1e-12)
    weights = torch.where(counts > 0, weights, torch.zeros_like(weights))
    nonzero = weights > 0
    weights[nonzero] = weights[nonzero] / weights[nonzero].mean()
    return (
        priors.to(torch.float32),
        weights.to(torch.float32),
        [int(value) for value in counts.tolist()],
    )


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _binary_iou(predicted: torch.Tensor, gold: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    predicted = predicted.bool() & valid.bool()
    gold = gold.bool() & valid.bool()
    intersection = (predicted & gold).sum(dim=1).to(torch.float32)
    union = (predicted | gold).sum(dim=1).to(torch.float32)
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


@dataclass(frozen=True)
class EvaluationCacheV2:
    answerability_probability: torch.Tensor
    raw_answer_label: torch.Tensor
    gold_answer_label: torch.Tensor
    anchor_probability: torch.Tensor
    answer_probability: torch.Tensor
    verification_probability: torch.Tensor
    gold_anchor_mask: torch.Tensor
    gold_answer_mask: torch.Tensor
    gold_verification_mask: torch.Tensor
    gold_evidence_mask: torch.Tensor
    valid_mask: torch.Tensor

    def __post_init__(self) -> None:
        rows = int(self.gold_answer_label.shape[0])
        # Do not use dataclasses.asdict here: it deep-copies every potentially
        # large [N,T] tensor in an evaluation cache.
        for name, tensor in vars(self).items():
            if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != rows:
                raise ValueError(f"evaluation cache field {name} has inconsistent rows")
        if rows == 0:
            raise ValueError("evaluation cache cannot be empty")


@torch.no_grad()
def collect_evaluation_cache_v2(
    model: DenseTemporalReasonerV2,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> EvaluationCacheV2:
    model.eval()
    columns: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "answerability_probability",
            "raw_answer_label",
            "gold_answer_label",
            "anchor_probability",
            "answer_probability",
            "verification_probability",
            "gold_anchor_mask",
            "gold_answer_mask",
            "gold_verification_mask",
            "gold_evidence_mask",
            "valid_mask",
        )
    }
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        _mask_padding(batch)
        outputs = model(
            batch["features"],
            batch["detector_logits"],
            batch["anchor_label"].long(),
            batch["relation_id"].long(),
            batch["ordinal"].long(),
            teacher_forcing=0.0,
            valid_frame_mask=batch["valid_mask"],
        )
        valid = batch["valid_mask"].bool()
        columns["answerability_probability"].append(
            outputs["answerability_logit"].sigmoid().cpu()
        )
        columns["raw_answer_label"].append(outputs["answer_logits"].argmax(dim=1).cpu())
        columns["gold_answer_label"].append(batch["gold_answer_label"].long().cpu())
        columns["anchor_probability"].append(
            (outputs["anchor_mask_logits"].sigmoid() * valid).cpu()
        )
        columns["answer_probability"].append(
            (outputs["answer_mask_logits"].sigmoid() * valid).cpu()
        )
        # relation_mask is an explicit probability-like CDF geometry, not a
        # learned verifier gate.  NONE evidence uses it directly.
        columns["verification_probability"].append((outputs["relation_mask"] * valid).cpu())
        for name in (
            "gold_anchor_mask",
            "gold_answer_mask",
            "gold_verification_mask",
            "gold_evidence_mask",
            "valid_mask",
        ):
            columns[name].append(batch[name].cpu())
    if not columns["gold_answer_label"]:
        raise ValueError("evaluation loader emitted no rows")
    return EvaluationCacheV2(
        **{name: torch.cat(parts, dim=0) for name, parts in columns.items()}
    )


def score_evaluation_cache_v2(
    cache: EvaluationCacheV2,
    thresholds: DenseTemporalThresholdsV2,
) -> dict[str, Any]:
    valid = cache.valid_mask.bool()
    gold_label = cache.gold_answer_label.long()
    positive = gold_label >= 0
    negative = ~positive
    predicted_positive = cache.answerability_probability >= thresholds.answerability
    label_correct = cache.raw_answer_label == gold_label

    anchor_mask = cache.anchor_probability >= thresholds.anchor_mask
    answer_mask = cache.answer_probability >= thresholds.answer_mask
    verification_mask = cache.verification_probability >= thresholds.verification
    anchor_iou = _binary_iou(anchor_mask, cache.gold_anchor_mask, valid)
    answer_iou = _binary_iou(answer_mask, cache.gold_answer_mask, valid)
    verification_iou = _binary_iou(
        verification_mask, cache.gold_verification_mask, valid
    )
    evidence_mask = anchor_mask | torch.where(
        predicted_positive[:, None], answer_mask, verification_mask
    )
    evidence_iou = _binary_iou(evidence_mask, cache.gold_evidence_mask, valid)

    def strict_metrics(iou_threshold: float) -> tuple[float, float, float, float]:
        # Positive rows require the correct answer decision and exact label,
        # plus independently adequate anchor, answer, and union-evidence
        # localization.  NONE rows require an explicit NONE decision plus
        # adequate anchor, verification-region, and union-evidence masks.
        strict_positive_rows = (
            predicted_positive
            & positive
            & label_correct
            & (anchor_iou >= iou_threshold)
            & (answer_iou >= iou_threshold)
            & (evidence_iou >= iou_threshold)
        )
        strict_none_rows = (
            (~predicted_positive)
            & negative
            & (anchor_iou >= iou_threshold)
            & (verification_iou >= iou_threshold)
            & (evidence_iou >= iou_threshold)
        )
        positive_score = (
            float(strict_positive_rows[positive].float().mean())
            if bool(positive.any())
            else 0.0
        )
        none_score = (
            float(strict_none_rows[negative].float().mean())
            if bool(negative.any())
            else 0.0
        )
        return (
            positive_score,
            none_score,
            0.5 * (positive_score + none_score),
            min(positive_score, none_score),
        )

    (
        strict_positive_iou30,
        strict_none_iou30,
        strict_balanced_iou30,
        strict_min_iou30,
    ) = strict_metrics(DIAGNOSTIC_STRICT_IOU)
    (
        strict_positive_iou50,
        strict_none_iou50,
        strict_balanced_iou50,
        strict_min_iou50,
    ) = strict_metrics(PRIMARY_STRICT_IOU)
    direct_positive = (
        float((predicted_positive & label_correct)[positive].float().mean())
        if bool(positive.any())
        else 0.0
    )
    none_decision = (
        float((~predicted_positive)[negative].float().mean())
        if bool(negative.any())
        else 0.0
    )
    conditional_label = (
        float(label_correct[positive].float().mean()) if bool(positive.any()) else 0.0
    )
    correct_answer = (predicted_positive & positive & label_correct) | (
        (~predicted_positive) & negative
    )
    return {
        "strict_metric_schema": STRICT_METRIC_SCHEMA_V2,
        "primary_strict_iou_threshold": PRIMARY_STRICT_IOU,
        "diagnostic_strict_iou_threshold": DIAGNOSTIC_STRICT_IOU,
        "thresholds": asdict(thresholds),
        "positive_count": int(positive.sum()),
        "none_count": int(negative.sum()),
        "conditional_answer_label_top1": conditional_label,
        "direct_positive_answer_accuracy": direct_positive,
        "none_decision_accuracy": none_decision,
        "answer_balanced_accuracy": 0.5 * (direct_positive + none_decision),
        "overall_answer_accuracy": float(correct_answer.float().mean()),
        "mean_anchor_iou": float(anchor_iou.mean()),
        "mean_answer_iou_positive": (
            float(answer_iou[positive].mean()) if bool(positive.any()) else 0.0
        ),
        "mean_verification_iou_none": (
            float(verification_iou[negative].mean()) if bool(negative.any()) else 0.0
        ),
        "mean_union_evidence_iou": float(evidence_iou.mean()),
        "mean_union_evidence_iou_positive": (
            float(evidence_iou[positive].mean()) if bool(positive.any()) else 0.0
        ),
        "mean_union_evidence_iou_none": (
            float(evidence_iou[negative].mean()) if bool(negative.any()) else 0.0
        ),
        # IoU@0.30 is retained only as an explicitly named diagnostic for
        # comparison with the historical Q-DOR-v2 metric.  It is never used
        # for threshold calibration or checkpoint selection.
        "strict_positive_iou30_diagnostic": strict_positive_iou30,
        "strict_none_iou30_diagnostic": strict_none_iou30,
        "strict_balanced_iou30_diagnostic": strict_balanced_iou30,
        "strict_min_iou30_diagnostic": strict_min_iou30,
        # These IoU@0.50 metrics are the only primary strict metrics.
        "strict_positive_iou50": strict_positive_iou50,
        "strict_none_iou50": strict_none_iou50,
        "strict_balanced_iou50": strict_balanced_iou50,
        "strict_min_iou50": strict_min_iou50,
        "mean_predicted_answer_mask_probability_positive": (
            float(cache.answer_probability[positive].mean()) if bool(positive.any()) else 0.0
        ),
    }


def calibrate_thresholds_v2(
    cache: EvaluationCacheV2,
    *,
    answerability_values: Sequence[float] | None = None,
    anchor_values: Sequence[float] | None = None,
    answer_values: Sequence[float] | None = None,
    verification: float = 0.5,
) -> tuple[DenseTemporalThresholdsV2, dict[str, Any]]:
    """Calibrate only on the dedicated calibration cache."""

    answerability_values = answerability_values or tuple(index / 10 for index in range(1, 10))
    anchor_values = anchor_values or tuple(index / 10 for index in range(2, 9))
    answer_values = answer_values or tuple(index / 10 for index in range(2, 9))
    best: tuple[tuple[float, ...], DenseTemporalThresholdsV2, dict[str, Any]] | None = None
    for answerability in answerability_values:
        for anchor in anchor_values:
            for answer in answer_values:
                thresholds = DenseTemporalThresholdsV2(
                    answerability=float(answerability),
                    anchor_mask=float(anchor),
                    answer_mask=float(answer),
                    verification=float(verification),
                )
                metrics = score_evaluation_cache_v2(cache, thresholds)
                key = (
                    float(metrics["strict_balanced_iou50"]),
                    float(metrics["strict_min_iou50"]),
                    float(metrics["strict_positive_iou50"]),
                    float(metrics["strict_none_iou50"]),
                    float(metrics["answer_balanced_accuracy"]),
                    -abs(float(answerability) - 0.5),
                    -abs(float(anchor) - 0.5),
                    -abs(float(answer) - 0.5),
                )
                if best is None or key > best[0]:
                    best = (key, thresholds, metrics)
    if best is None:
        raise ValueError("threshold calibration grid is empty")
    return best[1], best[2]


def checkpoint_selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    """Primary strict IoU@0.50 balance dominates every checkpoint tie break."""

    return (
        float(metrics["strict_balanced_iou50"]),
        float(metrics["strict_min_iou50"]),
        float(metrics["strict_positive_iou50"]),
        float(metrics["strict_none_iou50"]),
        float(metrics["answer_balanced_accuracy"]),
        float(metrics["mean_union_evidence_iou"]),
    )


def split_calibration_selection_scenes(
    scene_ids: Sequence[str],
    *,
    seed: int,
    calibration_fraction: float,
) -> tuple[list[str], list[str]]:
    """Deterministic scene-level split; never split QA rows from one scene."""

    unique = list(dict.fromkeys(str(value) for value in scene_ids))
    if len(unique) < 2:
        raise ValueError("at least two dev scenes are required for calibration/selection")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie in (0,1)")
    ranked = sorted(
        unique,
        key=lambda scene_id: hashlib.sha256(
            f"{seed}:qdor-v2-calibration:{scene_id}".encode("utf-8")
        ).hexdigest(),
    )
    count = min(len(ranked) - 1, max(1, int(round(len(ranked) * calibration_fraction))))
    calibration = ranked[:count]
    selection = ranked[count:]
    if set(calibration) & set(selection):
        raise RuntimeError("calibration/selection scene split overlaps")
    return calibration, selection


def assert_dense_identity_disjoint_v2(
    store: DenseFeatureStore,
    split_scene_ids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Reject source-audio leakage even when scene identifiers differ."""

    identity_fields = (
        "source_sha256",
        "source_path",
        "source_id",
        "source_video_id",
    )
    identities: dict[str, set[tuple[str, str]]] = {}
    for split, scene_ids in split_scene_ids.items():
        values: set[tuple[str, str]] = set()
        for scene_id in scene_ids:
            metadata = store.metadata(scene_id)
            for event in semantic_events(metadata):
                for field in identity_fields:
                    value = str(event.get(field) or "").strip()
                    if value:
                        values.add((field, value))
        identities[split] = values
    overlaps: dict[str, int] = {}
    names = list(split_scene_ids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = identities[left] & identities[right]
            overlaps[f"{left}__{right}"] = len(shared)
            if shared:
                preview = sorted(shared)[:5]
                raise ValueError(
                    f"{left}/{right} dense source identity leakage: {preview}"
                )
    return {
        "fields": list(identity_fields),
        "identity_counts": {
            split: len(values) for split, values in identities.items()
        },
        "overlap_counts": overlaps,
        "passes": True,
    }


def run_toy_microfit_v2(*, steps: int = 20, seed: int = 2031) -> dict[str, Any]:
    """Deterministic two-case microfit for the complete strict metric.

    Dense detector tracks are intentionally perfect so this test isolates the
    learned positive/NONE verifier while still executing the real anchor,
    nearest-onset, hard-label, occurrence-mask, and NONE-evidence operators.
    """

    if steps < 1:
        raise ValueError("microfit steps must be positive")
    torch.manual_seed(seed)
    model = DenseTemporalReasonerV2(
        DenseTemporalReasonerV2Config(
            feature_dim=4,
            num_classes=4,
            hidden_dim=8,
            max_ordinal=3,
            dropout=0.0,
        )
    )
    with torch.no_grad():
        model.frame_class_feature_scale.zero_()
        model.onset_feature_scale.zero_()
        model.anchor_feature_scale.zero_()
        model.ordinal_prior_scale.fill_(1.0)
        model.answer_bias.zero_()
        model.answer_mask_residual[-1].weight.zero_()
        model.answer_mask_residual[-1].bias.zero_()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.answerability_scorer.parameters():
        parameter.requires_grad_(True)

    batch, frames, classes = 8, 125, 4
    features = torch.zeros(batch, frames, 4)
    detector = torch.full((batch, frames, classes), -10.0)
    gold_label = torch.tensor([1, -1] * (batch // 2), dtype=torch.long)
    gold_anchor = torch.zeros(batch, frames)
    gold_answer = torch.zeros(batch, frames)
    gold_verification = torch.zeros(batch, frames)
    gold_evidence = torch.zeros(batch, frames)
    for row in range(batch):
        if row % 2 == 0:
            # A -> B -> a stronger but farther C.  Immediate B must win.
            detector[row, 20:30, 0] = 10.0
            detector[row, 60:70, 1] = 10.0
            detector[row, 100:110, 2] = 12.0
            gold_anchor[row, 20:30] = 1.0
            gold_answer[row, 60:70] = 1.0
            gold_evidence[row, 20:30] = 1.0
            gold_evidence[row, 60:70] = 1.0
        else:
            # B -> C -> A: nothing occurs after the anchor.
            detector[row, 20:30, 1] = 10.0
            detector[row, 60:70, 2] = 10.0
            detector[row, 100:110, 0] = 10.0
            gold_anchor[row, 100:110] = 1.0
            gold_verification[row, 110:] = 1.0
            gold_evidence[row, 100:] = 1.0
    anchor_label = torch.zeros(batch, dtype=torch.long)
    relation = torch.full((batch,), RELATION_AFTER, dtype=torch.long)
    ordinal = torch.ones(batch, dtype=torch.long)
    optimizer = torch.optim.Adam(model.answerability_scorer.parameters(), lr=0.03)
    for _ in range(steps):
        outputs = model(
            features,
            detector,
            anchor_label,
            relation,
            ordinal,
        )
        loss = F.binary_cross_entropy_with_logits(
            outputs["answerability_logit"], (gold_label >= 0).to(torch.float32)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        outputs = model(
            features,
            detector,
            anchor_label,
            relation,
            ordinal,
        )
    cache = EvaluationCacheV2(
        answerability_probability=outputs["answerability_logit"].sigmoid(),
        raw_answer_label=outputs["answer_logits"].argmax(dim=1),
        gold_answer_label=gold_label,
        anchor_probability=outputs["anchor_mask_logits"].sigmoid(),
        answer_probability=outputs["answer_mask_logits"].sigmoid(),
        verification_probability=outputs["relation_mask"],
        gold_anchor_mask=gold_anchor,
        gold_answer_mask=gold_answer,
        gold_verification_mask=gold_verification,
        gold_evidence_mask=gold_evidence,
        valid_mask=torch.ones(batch, frames, dtype=torch.bool),
    )
    return score_evaluation_cache_v2(cache, DenseTemporalThresholdsV2())


def _require_positive_and_none(items: Sequence[BoundQAItemV2], name: str) -> None:
    positive = sum(not item.qa.no_evidence for item in items)
    none = len(items) - positive
    if positive == 0 or none == 0:
        raise ValueError(
            f"{name} must contain both positive and NONE rows; positive={positive} none={none}"
        )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-index", type=Path, action="append", required=True)
    parser.add_argument("--train-scene-list", type=Path, required=True)
    parser.add_argument(
        "--dev-scene-list",
        type=Path,
        required=True,
        help="Selection scenes, or the full dev pool when calibration list is omitted.",
    )
    parser.add_argument(
        "--calibration-scene-list",
        type=Path,
        help="Optional disjoint calibration scenes. Otherwise dev scenes are split by hash.",
    )
    parser.add_argument("--train-qa-manifest", type=Path, required=True)
    parser.add_argument(
        "--dev-qa-manifest",
        type=Path,
        required=True,
        help="Must contain every requested calibration and selection scene.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--oracle-gate-receipt", type=Path)
    parser.add_argument("--min-oracle-answer-top1", type=float, default=0.70)
    parser.add_argument("--allow-failed-oracle-gate", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-ordinal", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs-stage-a", type=int, default=4)
    parser.add_argument("--epochs-stage-b", type=int, default=12)
    parser.add_argument("--stage-b-teacher-start", type=float, default=0.9)
    parser.add_argument("--stage-b-teacher-end", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--logit-adjustment-tau", type=float, default=0.5)
    parser.add_argument("--class-balance-beta", type=float, default=0.999)
    parser.add_argument("--prior-smoothing", type=float, default=1.0)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument(
        "--preload", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--shard-cache-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.num_workers < 0:
        raise SystemExit("batch-size must be positive and num-workers non-negative")
    if args.max_ordinal < 1:
        raise SystemExit("max-ordinal must be positive")
    if (
        args.epochs_stage_a < 0
        or args.epochs_stage_b < 0
        or args.epochs_stage_a + args.epochs_stage_b < 1
    ):
        raise SystemExit("at least one stage epoch is required")
    for value, name in (
        (args.stage_b_teacher_start, "stage-b-teacher-start"),
        (args.stage_b_teacher_end, "stage-b-teacher-end"),
    ):
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"{name} must be in [0,1]")

    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "qdor_v2_best.pt"
    receipt_path = output_dir / "receipt.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_gate = load_oracle_gate(
        args.oracle_gate_receipt,
        minimum_top1=args.min_oracle_answer_top1,
        allow_failed=args.allow_failed_oracle_gate,
    )
    _seed_everything(args.seed)
    train_scene_ids = load_scene_list(args.train_scene_list)
    dev_scene_ids = load_scene_list(args.dev_scene_list)
    if args.calibration_scene_list is None:
        calibration_scene_ids, selection_scene_ids = split_calibration_selection_scenes(
            dev_scene_ids,
            seed=args.seed,
            calibration_fraction=args.calibration_fraction,
        )
        calibration_source = "deterministic_hash_split_of_dev_scenes"
    else:
        calibration_scene_ids = load_scene_list(args.calibration_scene_list)
        selection_scene_ids = dev_scene_ids
        calibration_source = "explicit_scene_list"
    split_sets = {
        "train": set(train_scene_ids),
        "calibration": set(calibration_scene_ids),
        "selection": set(selection_scene_ids),
    }
    for left, right in (("train", "calibration"), ("train", "selection"), ("calibration", "selection")):
        overlap = sorted(split_sets[left] & split_sets[right])
        if overlap:
            raise ValueError(f"{left}/{right} scene lists overlap: {overlap[:10]}")

    store = DenseFeatureStore(args.dense_index, cache_size=args.shard_cache_size)
    missing = sorted(set().union(*split_sets.values()) - store.scene_ids)
    if missing:
        raise ValueError(f"requested scenes absent from dense indexes: {missing[:10]}")
    identity_audit = assert_dense_identity_disjoint_v2(
        store,
        {
            "train": train_scene_ids,
            "calibration": calibration_scene_ids,
            "selection": selection_scene_ids,
        },
    )
    train_items = load_bound_qa_manifest_v2(
        args.train_qa_manifest,
        allowed_scene_ids=train_scene_ids,
        store=store,
        max_ordinal=args.max_ordinal,
    )
    all_dev_ids = list(dict.fromkeys(calibration_scene_ids + selection_scene_ids))
    dev_items = load_bound_qa_manifest_v2(
        args.dev_qa_manifest,
        allowed_scene_ids=all_dev_ids,
        store=store,
        max_ordinal=args.max_ordinal,
    )
    calibration_set = set(calibration_scene_ids)
    selection_set = set(selection_scene_ids)
    calibration_items = [item for item in dev_items if item.qa.scene_id in calibration_set]
    selection_items = [item for item in dev_items if item.qa.scene_id in selection_set]
    _require_positive_and_none(train_items, "train")
    _require_positive_and_none(calibration_items, "calibration")
    _require_positive_and_none(selection_items, "selection")

    train_dataset = DenseQADatasetV2(store, train_items, preload=args.preload)
    calibration_dataset = DenseQADatasetV2(store, calibration_items, preload=args.preload)
    selection_dataset = DenseQADatasetV2(store, selection_items, preload=args.preload)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_dense_qa_v2,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_kwargs
    )
    calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_kwargs)
    selection_loader = DataLoader(selection_dataset, shuffle=False, **loader_kwargs)

    labels = store.labels or []
    priors, class_weights, class_counts = class_statistics_v2(
        train_dataset,
        len(labels),
        beta=args.class_balance_beta,
        smoothing=args.prior_smoothing,
    )
    config = DenseTemporalReasonerV2Config(
        feature_dim=int(store.feature_dim or 0),
        num_classes=len(labels),
        hidden_dim=args.hidden_dim,
        max_ordinal=args.max_ordinal,
        dropout=args.dropout,
    )
    device = _device(args.device)
    model = DenseTemporalReasonerV2(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    log_prior = priors.log().to(device)
    class_weights_device = class_weights.to(device)

    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_selection: dict[str, Any] | None = None
    global_epoch = 0
    for stage, stage_epochs in (("A", args.epochs_stage_a), ("B", args.epochs_stage_b)):
        for stage_epoch in range(stage_epochs):
            global_epoch += 1
            teacher_forcing = teacher_forcing_for_epoch(
                stage,
                stage_epoch,
                stage_epochs,
                args.stage_b_teacher_start,
                args.stage_b_teacher_end,
            )
            model.train()
            totals: Counter[str] = Counter()
            examples = 0
            for raw_batch in train_loader:
                batch = _to_device(raw_batch, device)
                _mask_padding(batch)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    batch["features"],
                    batch["detector_logits"],
                    batch["anchor_label"].long(),
                    batch["relation_id"].long(),
                    batch["ordinal"].long(),
                    gold_anchor_mask=batch["gold_anchor_mask"],
                    gold_answer_label=batch["gold_answer_label"].long(),
                    teacher_forcing=teacher_forcing,
                    valid_frame_mask=batch["valid_mask"],
                )
                losses = dense_temporal_reasoner_v2_loss(
                    outputs,
                    batch,
                    log_prior=log_prior,
                    class_weights=class_weights_device,
                    logit_adjustment_tau=args.logit_adjustment_tau,
                )
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"non-finite loss at epoch {global_epoch}")
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                count = int(batch["features"].shape[0])
                examples += count
                for name, value in losses.items():
                    totals[name] += float(value.detach()) * count
            train_metrics = {
                name: value / max(examples, 1) for name, value in totals.items()
            }

            calibration_cache = collect_evaluation_cache_v2(
                model, calibration_loader, device
            )
            thresholds, calibration_metrics = calibrate_thresholds_v2(
                calibration_cache
            )
            # Frozen application: selection labels never participate in the
            # threshold search above.
            selection_cache = collect_evaluation_cache_v2(model, selection_loader, device)
            selection_metrics = score_evaluation_cache_v2(selection_cache, thresholds)
            epoch_row = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch + 1,
                "teacher_forcing": teacher_forcing,
                "train": train_metrics,
                "calibration": calibration_metrics,
                "selection": selection_metrics,
            }
            history.append(epoch_row)
            print(json.dumps(epoch_row, ensure_ascii=False, sort_keys=True), flush=True)

            key = checkpoint_selection_key(selection_metrics)
            if best_key is None or key > best_key:
                best_key = key
                best_selection = dict(selection_metrics)
                _atomic_torch(
                    {
                        "format": CHECKPOINT_FORMAT_V2,
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "model_state_dict": model.state_dict(),
                        "config": asdict(config),
                        "labels": labels,
                        "thresholds": asdict(thresholds),
                        "epoch": global_epoch,
                        "stage": stage,
                        "selection_metrics": selection_metrics,
                        "calibration_metrics": calibration_metrics,
                        "selection_metric": (
                            "strict_balanced_iou50_then_strict_min_iou50"
                        ),
                        "strict_metric_schema": STRICT_METRIC_SCHEMA_V2,
                        "class_priors": priors,
                        "class_weights": class_weights,
                    },
                    checkpoint_path,
                )
            _atomic_json(
                {
                    "format": RECEIPT_FORMAT_V2,
                    "status": "running",
                    "history": history,
                    "best_selection": best_selection,
                },
                output_dir / "training_history.json",
            )

    assert best_selection is not None
    receipt = {
        "format": RECEIPT_FORMAT_V2,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Q-DOR-v2 dense nearest-onset reasoner",
        "data_protocol": {
            "qa_binding": "exact dense-gold event identity, ordinal, adjacency and interval",
            "onset_supervision": "all semantic scene events on fixed 40 ms class grid",
            "evidence_policy": {
                "positive": "anchor+answer occurrence",
                "none": "anchor+complete relation verification region",
            },
            "train_scene_count": len(train_scene_ids),
            "calibration_scene_count": len(calibration_scene_ids),
            "selection_scene_count": len(selection_scene_ids),
            "scene_overlap": 0,
            "dense_source_identity_audit": identity_audit,
            "calibration_source": calibration_source,
            "thresholds_calibrated_on_selection": False,
            "train_scene_list_sha256": _sha256_file(args.train_scene_list.resolve()),
            "dev_scene_list_sha256": _sha256_file(args.dev_scene_list.resolve()),
            "calibration_scene_list_sha256": (
                None
                if args.calibration_scene_list is None
                else _sha256_file(args.calibration_scene_list.resolve())
            ),
            "train_qa_manifest_sha256": _sha256_file(args.train_qa_manifest.resolve()),
            "dev_qa_manifest_sha256": _sha256_file(args.dev_qa_manifest.resolve()),
            "dense_indexes": store.index_sha256,
            "oracle_representation_gate": oracle_gate,
        },
        "dataset": {
            "train_qa_count": len(train_dataset),
            "calibration_qa_count": len(calibration_dataset),
            "selection_qa_count": len(selection_dataset),
            "num_classes": len(labels),
            "answer_class_counts": class_counts,
        },
        "model_config": asdict(config),
        "evaluation_protocol": {
            "strict_metric_schema": STRICT_METRIC_SCHEMA_V2,
            "primary_strict_iou_threshold": PRIMARY_STRICT_IOU,
            "diagnostic_strict_iou_threshold": DIAGNOSTIC_STRICT_IOU,
            "positive_strict_requirements": [
                "correct_positive_decision",
                "correct_answer_label",
                "anchor_iou_at_threshold",
                "answer_iou_at_threshold",
                "union_evidence_iou_at_threshold",
            ],
            "none_strict_requirements": [
                "correct_none_decision",
                "anchor_iou_at_threshold",
                "verification_iou_at_threshold",
                "union_evidence_iou_at_threshold",
            ],
            "iou30_role": "diagnostic_only_never_calibration_or_selection",
            "threshold_calibration_split": "scene_disjoint_calibration",
            "checkpoint_selection_split": "scene_disjoint_selection",
        },
        "selection_metric": (
            "strict_balanced_iou50_then_strict_min_iou50_on_"
            "scene_disjoint_selection"
        ),
        "best_selection": best_selection,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    _atomic_json(receipt, receipt_path)
    return receipt


if __name__ == "__main__":
    main()
