"""Fixed-grid utilities for the QCES BEATs-Strong detector.

The historical detector trainer projected timestamps with ``time / duration``.
That silently stretched a five-second recording over all 250 output frames.
This module defines the corrected contract: every output frame is always a
40-ms interval on a left-aligned ten-second grid and right padding is excluded
from both training losses and evaluation metrics.

The functions in this file are deliberately independent of the heavy BEATs
wrapper so their temporal and metric contracts can be unit tested with small
synthetic tensors.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F


NUM_FRAMES = 250
FRAME_HOP_SECONDS = 0.04
FIXED_AUDIO_SECONDS = NUM_FRAMES * FRAME_HOP_SECONDS


def valid_frames_for_duration(duration_seconds: float) -> int:
    """Return non-padding frames on the fixed 40-ms grid."""

    clipped = min(max(float(duration_seconds), 0.0), FIXED_AUDIO_SECONDS)
    return min(NUM_FRAMES, max(0, int(math.ceil(clipped / FRAME_HOP_SECONDS - 1e-5))))


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def build_fixed_grid_targets(
    rows: Sequence[Any],
    *,
    num_labels: int,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build multi-label frame targets and a valid-frame mask.

    Frame ``i`` always covers ``[i*0.04, (i+1)*0.04)``.  An event activates
    every frame whose interval intersects the clipped event interval.  The
    mapping never depends on the recording duration except for clipping and
    masking right padding.
    """

    targets = torch.zeros(
        (len(rows), NUM_FRAMES, int(num_labels)), dtype=torch.float32, device=device
    )
    valid = torch.zeros((len(rows), NUM_FRAMES), dtype=torch.bool, device=device)
    for batch_index, row in enumerate(rows):
        duration = min(
            max(float(_row_value(row, "duration_seconds", FIXED_AUDIO_SECONDS)), 0.0),
            FIXED_AUDIO_SECONDS,
        )
        valid_frames = valid_frames_for_duration(duration)
        if valid_frames > 0:
            valid[batch_index, :valid_frames] = True
        for event in _row_value(row, "events", ()) or ():
            label_id = int(_row_value(event, "label_id", -1))
            onset = max(0.0, float(_row_value(event, "onset_seconds", 0.0)))
            offset = min(duration, float(_row_value(event, "offset_seconds", 0.0)))
            if label_id < 0 or label_id >= num_labels or offset <= onset or valid_frames <= 0:
                continue
            start = max(0, min(valid_frames - 1, int(math.floor(onset / FRAME_HOP_SECONDS))))
            end = max(
                start + 1,
                min(valid_frames, int(math.ceil(offset / FRAME_HOP_SECONDS - 1e-8))),
            )
            targets[batch_index, start:end, label_id] = 1.0
    return targets, valid


def build_fixed_grid_boundary_targets(
    rows: Sequence[Any],
    *,
    num_labels: int,
    dilation_frames: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build activity-edge onset/offset targets on the fixed 40-ms grid.

    These targets supervise boundaries derived from a single activity logit,
    so they must be the rising/falling edges of the *union* activity mask for
    each class.  Touching or overlapping same-class annotations therefore form
    one activity region.  Inventing an offset and onset at their shared frame
    would be structurally impossible for ``a[t] * (1-a[t-1])`` and
    ``a[t-1] * (1-a[t])`` to represent simultaneously.  Occurrence-specific
    onsets remain a downstream Q-DOR target rather than this auxiliary target.

    An event ending at the valid-audio boundary has no observable inactive
    offset frame.  ``dilation_frames`` is retained for standalone diagnostics;
    the detector trainer enforces zero dilation for learning.
    """

    if dilation_frames < 0:
        raise ValueError("dilation_frames must be non-negative")
    activity, valid = build_fixed_grid_targets(rows, num_labels=num_labels, device=device)
    active = activity > 0.5
    previous = F.pad(active[:, :-1, :], (0, 0, 1, 0), value=False)
    onset_targets = (active & ~previous).to(dtype=torch.float32)
    offset_targets = (previous & ~active).to(dtype=torch.float32)

    dilation = int(dilation_frames)
    if dilation:
        kernel_size = 2 * dilation + 1
        onset_targets = F.max_pool1d(
            onset_targets.transpose(1, 2),
            kernel_size=kernel_size,
            stride=1,
            padding=dilation,
        ).transpose(1, 2)
        offset_targets = F.max_pool1d(
            offset_targets.transpose(1, 2),
            kernel_size=kernel_size,
            stride=1,
            padding=dilation,
        ).transpose(1, 2)

    valid_float = valid.unsqueeze(-1).to(dtype=torch.float32)
    onset_targets = onset_targets * valid_float
    offset_targets = offset_targets * valid_float
    return onset_targets, offset_targets, valid


def boundary_probabilities_from_activity_logits(
    frame_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive differentiable onset/offset probabilities from activity logits."""

    if frame_logits.ndim != 3:
        raise ValueError("frame_logits must have shape [B,T,C]")
    activity = frame_logits.sigmoid()
    previous = F.pad(activity[:, :-1, :], (0, 0, 1, 0))
    onset = activity * (1.0 - previous)
    offset = previous * (1.0 - activity)
    return onset, offset


def masked_probability_bce(
    probability: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Positive-weighted BCE for probability heads with padding masked out."""

    if probability.shape != targets.shape:
        raise ValueError("probability/targets shape mismatch")
    if valid_mask.shape != probability.shape[:2]:
        raise ValueError("valid_mask shape mismatch")
    if pos_weight.shape != (probability.shape[-1],):
        raise ValueError("pos_weight must contain one value per class")
    if not bool(torch.isfinite(probability).all()) or bool(
        ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("boundary probability must be finite and lie in [0,1]")
    clipped = probability.clamp(1e-6, 1.0 - 1e-6)
    error = F.binary_cross_entropy(clipped, targets, reduction="none")
    positive_scale = 1.0 + targets * (pos_weight.to(error) - 1.0).view(1, 1, -1)
    error = error * positive_scale
    if class_weight is not None:
        if class_weight.shape != (probability.shape[-1],):
            raise ValueError("class_weight must contain one value per class")
        error = error * class_weight.to(error).view(1, 1, -1)
    mask = valid_mask.to(error.dtype).unsqueeze(-1)
    denominator = mask.sum().clamp_min(1.0) * probability.shape[-1]
    return (error * mask).sum() / denominator


def boundary_pos_weights(
    rows: Sequence[Any],
    *,
    num_labels: int,
    dilation_frames: int,
    max_pos_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute classwise onset/offset imbalance without materializing all rows."""

    onset_count = torch.zeros(num_labels, dtype=torch.float64)
    offset_count = torch.zeros(num_labels, dtype=torch.float64)
    total_valid = 0
    for row in rows:
        onset, offset, valid = build_fixed_grid_boundary_targets(
            [row],
            num_labels=num_labels,
            dilation_frames=dilation_frames,
        )
        onset_count += onset[0].sum(dim=0).to(torch.float64)
        offset_count += offset[0].sum(dim=0).to(torch.float64)
        total_valid += int(valid.sum())
    maximum = float(max_pos_weight)
    onset_negative = float(total_valid) - onset_count
    offset_negative = float(total_valid) - offset_count
    onset_weight = (onset_negative / onset_count.clamp_min(1.0)).clamp(1.0, maximum)
    offset_weight = (offset_negative / offset_count.clamp_min(1.0)).clamp(1.0, maximum)
    return onset_weight.to(torch.float32), offset_weight.to(torch.float32)


def masked_balanced_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Balanced multi-label BCE that cannot learn from padded frames."""

    if logits.shape != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: {logits.shape} vs {targets.shape}")
    if valid_mask.shape != logits.shape[:2]:
        raise ValueError(f"valid_mask shape mismatch: {valid_mask.shape} vs {logits.shape[:2]}")
    if pos_weight.shape != (logits.shape[-1],):
        raise ValueError("pos_weight must contain one value per class")
    per_element = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight.to(logits), reduction="none"
    )
    mask = valid_mask.to(dtype=per_element.dtype).unsqueeze(-1)
    if class_weight is not None:
        if class_weight.shape != (logits.shape[-1],):
            raise ValueError("class_weight must contain one value per class")
        per_element = per_element * class_weight.to(logits).view(1, 1, -1)
    denominator = mask.sum().clamp_min(1.0) * logits.shape[-1]
    return (per_element * mask).sum() / denominator


def class_coverage(rows: Sequence[Any], labels: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Return scene, occurrence, and active-frame supervision per class."""

    index = {label: label_id for label_id, label in enumerate(labels)}
    scene_count: Counter[str] = Counter()
    event_count: Counter[str] = Counter()
    active_frames: Counter[str] = Counter()
    for row in rows:
        seen: set[str] = set()
        duration = min(
            max(float(_row_value(row, "duration_seconds", FIXED_AUDIO_SECONDS)), 0.0),
            FIXED_AUDIO_SECONDS,
        )
        valid_frames = valid_frames_for_duration(duration)
        for event in _row_value(row, "events", ()) or ():
            label = str(_row_value(event, "label", ""))
            if label not in index:
                continue
            onset = max(0.0, float(_row_value(event, "onset_seconds", 0.0)))
            offset = min(duration, float(_row_value(event, "offset_seconds", 0.0)))
            if offset <= onset or valid_frames <= 0:
                continue
            start = max(0, min(valid_frames - 1, int(math.floor(onset / FRAME_HOP_SECONDS))))
            end = max(
                start + 1,
                min(valid_frames, int(math.ceil(offset / FRAME_HOP_SECONDS - 1e-8))),
            )
            event_count[label] += 1
            active_frames[label] += end - start
            seen.add(label)
        scene_count.update(seen)
    return {
        label: {
            "label_id": label_id,
            "scenes": int(scene_count[label]),
            "events": int(event_count[label]),
            "active_frames": int(active_frames[label]),
            "active_seconds": float(active_frames[label] * FRAME_HOP_SECONDS),
        }
        for label_id, label in enumerate(labels)
    }


def balancing_tensors(
    rows: Sequence[Any], labels: Sequence[str], *, max_pos_weight: float = 80.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return positive BCE weights, class weights, and scene sampling weights.

    BCE positive weights compensate active/inactive frame imbalance.  A milder
    inverse-square-root class weight and scene sampler prevent frequent speech
    classes from dominating batches without exploding gradients for rare
    classes.
    """

    coverage = class_coverage(rows, labels)
    total_valid_frames = sum(
        valid_frames_for_duration(float(_row_value(row, "duration_seconds", FIXED_AUDIO_SECONDS)))
        for row in rows
    )
    active = torch.tensor(
        [float(coverage[label]["active_frames"]) for label in labels], dtype=torch.float64
    )
    negative = float(total_valid_frames) - active
    pos_weight = (negative / active.clamp_min(1.0)).clamp(1.0, float(max_pos_weight)).float()

    scene_support = torch.tensor(
        [float(coverage[label]["scenes"]) for label in labels], dtype=torch.float64
    )
    inverse_sqrt = (scene_support.clamp_min(1.0).max() / scene_support.clamp_min(1.0)).sqrt()
    inverse_sqrt = inverse_sqrt / inverse_sqrt.mean().clamp_min(1e-12)
    class_weight = inverse_sqrt.clamp(0.25, 4.0).float()

    label_to_id = {label: index for index, label in enumerate(labels)}
    scene_weights: list[float] = []
    for row in rows:
        ids = {
            label_to_id[str(_row_value(event, "label", ""))]
            for event in _row_value(row, "events", ()) or ()
            if str(_row_value(event, "label", "")) in label_to_id
        }
        # Negative-only scenes remain represented instead of disappearing.
        value = 1.0 if not ids else float(class_weight[list(ids)].mean())
        scene_weights.append(value)
    sampler_weight = torch.tensor(scene_weights, dtype=torch.double)
    sampler_weight /= sampler_weight.mean().clamp_min(1e-12)
    return pos_weight, class_weight, sampler_weight


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0.0 else 0.0


def decode_fixed_grid_events(
    probs: torch.Tensor,
    *,
    labels: Sequence[str],
    valid_frames: int,
    threshold: float,
    min_duration_seconds: float = 0.08,
    merge_gap_seconds: float = 0.08,
) -> list[dict[str, Any]]:
    """Decode multi-label events without stretching time to clip duration."""

    if probs.ndim != 2 or probs.shape[0] != NUM_FRAMES or probs.shape[1] != len(labels):
        raise ValueError(f"expected probs [250,{len(labels)}], got {tuple(probs.shape)}")
    valid_frames = min(NUM_FRAMES, max(0, int(valid_frames)))
    active = (probs[:valid_frames] >= float(threshold)).cpu()
    max_merge_frames = int(math.floor(merge_gap_seconds / FRAME_HOP_SECONDS + 1e-8))
    min_frames = max(1, int(math.ceil(min_duration_seconds / FRAME_HOP_SECONDS - 1e-8)))
    output: list[dict[str, Any]] = []
    for label_id, label in enumerate(labels):
        raw_segments: list[tuple[int, int]] = []
        start: int | None = None
        flags = active[:, label_id].tolist() + [False]
        for frame, flag in enumerate(flags):
            if flag and start is None:
                start = frame
            elif not flag and start is not None:
                raw_segments.append((start, frame))
                start = None
        merged: list[tuple[int, int]] = []
        for start, end in raw_segments:
            if merged and start - merged[-1][1] <= max_merge_frames:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        for start, end in merged:
            if end - start < min_frames:
                continue
            output.append(
                {
                    "label": label,
                    "label_id": label_id,
                    "onset_seconds": start * FRAME_HOP_SECONDS,
                    "offset_seconds": end * FRAME_HOP_SECONDS,
                    "confidence": float(probs[start:end, label_id].max()),
                }
            )
    output.sort(key=lambda item: (item["onset_seconds"], item["offset_seconds"], item["label"]))
    return output


def _safe_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def _match_events(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    """Maximum-cardinality, label-exact event matching at an IoU threshold.

    Confidence-greedy matching can undercount true positives: a broad,
    high-confidence prediction may consume the only target available to a
    second precise prediction.  The augmenting-path matcher below maximizes
    the number of valid one-to-one matches independently for each label.
    """

    tp: Counter[str] = Counter()
    predicted_count: Counter[str] = Counter(str(event["label"]) for event in predicted)
    gold_count: Counter[str] = Counter(str(event["label"]) for event in gold)
    labels = sorted(set(predicted_count) | set(gold_count))
    for label in labels:
        pred_rows = [event for event in predicted if str(event["label"]) == label]
        gold_rows = [event for event in gold if str(event["label"]) == label]
        adjacency: list[list[int]] = []
        for event in pred_rows:
            scored: list[tuple[float, int]] = []
            for index, target in enumerate(gold_rows):
                score = interval_iou(
                    (float(event["onset_seconds"]), float(event["offset_seconds"])),
                    (float(target["onset_seconds"]), float(target["offset_seconds"])),
                )
                if score >= iou_threshold:
                    scored.append((score, index))
            adjacency.append([index for _, index in sorted(scored, reverse=True)])

        gold_to_prediction: dict[int, int] = {}

        def augment(prediction_index: int, seen_gold: set[int]) -> bool:
            for gold_index in adjacency[prediction_index]:
                if gold_index in seen_gold:
                    continue
                seen_gold.add(gold_index)
                previous = gold_to_prediction.get(gold_index)
                if previous is None or augment(previous, seen_gold):
                    gold_to_prediction[gold_index] = prediction_index
                    return True
            return False

        order = sorted(
            range(len(pred_rows)),
            key=lambda index: float(pred_rows[index].get("confidence", 0.0)),
            reverse=True,
        )
        matches = sum(int(augment(index, set())) for index in order)
        tp[label] = matches
    fp = Counter(
        {label: predicted_count[label] - tp[label] for label in labels if predicted_count[label] - tp[label]}
    )
    fn = Counter({label: gold_count[label] - tp[label] for label in labels if gold_count[label] - tp[label]})
    return tp, fp, fn


def evaluate_fixed_grid_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str],
    threshold: float,
    event_iou_threshold: float = 0.30,
    min_duration_seconds: float = 0.08,
    merge_gap_seconds: float = 0.08,
    boundary_diagnostic_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate frame/event F1 and exact-label oracle-window top-1/top-5."""

    label_to_id = {label: index for index, label in enumerate(labels)}
    frame_tp: Counter[str] = Counter()
    frame_fp: Counter[str] = Counter()
    frame_fn: Counter[str] = Counter()
    event_tp: Counter[str] = Counter()
    event_fp: Counter[str] = Counter()
    event_fn: Counter[str] = Counter()
    onset_tp: Counter[str] = Counter()
    onset_fp: Counter[str] = Counter()
    onset_fn: Counter[str] = Counter()
    offset_tp: Counter[str] = Counter()
    offset_fp: Counter[str] = Counter()
    offset_fn: Counter[str] = Counter()
    oracle_total: Counter[str] = Counter()
    oracle_top1: Counter[str] = Counter()
    oracle_top5: Counter[str] = Counter()

    for item in predictions:
        logits = torch.as_tensor(item["logits"]).float()
        if tuple(logits.shape) != (NUM_FRAMES, len(labels)):
            raise ValueError(
                f"prediction logits must have shape {(NUM_FRAMES, len(labels))}, "
                f"got {tuple(logits.shape)}"
            )
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("prediction logits contain non-finite values")
        probs = torch.sigmoid(logits)
        gold_events = list(item.get("gold_events") or ())
        duration = float(item.get("duration_seconds", FIXED_AUDIO_SECONDS))
        expected_valid_frames = valid_frames_for_duration(duration)
        valid_frames = int(item.get("valid_frames", expected_valid_frames))
        if valid_frames != expected_valid_frames:
            raise ValueError(
                "prediction valid_frames disagrees with duration/fixed-grid padding"
            )
        target_rows = [{"duration_seconds": duration, "events": gold_events}]
        targets, valid_mask = build_fixed_grid_targets(target_rows, num_labels=len(labels))
        onset_targets, offset_targets, boundary_valid = build_fixed_grid_boundary_targets(
            target_rows, num_labels=len(labels), dilation_frames=0
        )
        targets = targets[0].bool()
        valid_mask = valid_mask[0]
        if not torch.equal(valid_mask, boundary_valid[0]):
            raise RuntimeError("activity/boundary valid masks disagree")
        predicted_frames = probs >= threshold
        onset_probability, offset_probability = boundary_probabilities_from_activity_logits(
            logits.unsqueeze(0)
        )
        if not 0.0 <= boundary_diagnostic_threshold <= 1.0:
            raise ValueError("boundary_diagnostic_threshold must lie in [0,1]")
        predicted_onsets = onset_probability[0] >= boundary_diagnostic_threshold
        predicted_offsets = offset_probability[0] >= boundary_diagnostic_threshold
        for label_id, label in enumerate(labels):
            pred = predicted_frames[:, label_id] & valid_mask
            gold = targets[:, label_id] & valid_mask
            frame_tp[label] += int((pred & gold).sum())
            frame_fp[label] += int((pred & ~gold & valid_mask).sum())
            frame_fn[label] += int((~pred & gold & valid_mask).sum())
            pred_onset = predicted_onsets[:, label_id] & valid_mask
            gold_onset = onset_targets[0, :, label_id].bool() & valid_mask
            onset_tp[label] += int((pred_onset & gold_onset).sum())
            onset_fp[label] += int((pred_onset & ~gold_onset & valid_mask).sum())
            onset_fn[label] += int((~pred_onset & gold_onset & valid_mask).sum())
            pred_offset = predicted_offsets[:, label_id] & valid_mask
            gold_offset = offset_targets[0, :, label_id].bool() & valid_mask
            offset_tp[label] += int((pred_offset & gold_offset).sum())
            offset_fp[label] += int((pred_offset & ~gold_offset & valid_mask).sum())
            offset_fn[label] += int((~pred_offset & gold_offset & valid_mask).sum())

        decoded = decode_fixed_grid_events(
            probs,
            labels=labels,
            valid_frames=valid_frames,
            threshold=threshold,
            min_duration_seconds=min_duration_seconds,
            merge_gap_seconds=merge_gap_seconds,
        )
        tp, fp, fn = _match_events(decoded, gold_events, iou_threshold=event_iou_threshold)
        event_tp.update(tp)
        event_fp.update(fp)
        event_fn.update(fn)

        for event in gold_events:
            label = str(event.get("label") or "")
            if label not in label_to_id:
                continue
            start = max(
                0,
                min(valid_frames - 1, int(math.floor(float(event["onset_seconds"]) / FRAME_HOP_SECONDS))),
            )
            end = max(
                start + 1,
                min(valid_frames, int(math.ceil(float(event["offset_seconds"]) / FRAME_HOP_SECONDS - 1e-8))),
            )
            if valid_frames <= 0 or end <= start:
                continue
            pooled = logits[start:end].amax(dim=0)
            ranking = pooled.argsort(descending=True)
            gold_id = label_to_id[label]
            oracle_total[label] += 1
            oracle_top1[label] += int(int(ranking[0]) == gold_id)
            oracle_top5[label] += int(bool((ranking[: min(5, len(labels))] == gold_id).any()))

    def aggregate(tp: Counter[str], fp: Counter[str], fn: Counter[str]) -> dict[str, Any]:
        micro = _safe_prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
        per_class = {
            label: _safe_prf(tp[label], fp[label], fn[label]) for label in labels
        }
        observed = [per_class[label]["f1"] for label in labels if tp[label] + fn[label] > 0]
        return {
            **micro,
            "macro_f1_observed_classes": sum(observed) / len(observed) if observed else 0.0,
            "per_class": per_class,
        }

    oracle_count = sum(oracle_total.values())
    observed_labels = [label for label in labels if oracle_total[label] > 0]
    oracle = {
        "events": oracle_count,
        "observed_classes": len(observed_labels),
        "top1_micro": sum(oracle_top1.values()) / oracle_count if oracle_count else 0.0,
        "top5_micro": sum(oracle_top5.values()) / oracle_count if oracle_count else 0.0,
        "top1_macro_observed_classes": (
            sum(oracle_top1[label] / oracle_total[label] for label in observed_labels)
            / len(observed_labels)
            if observed_labels
            else 0.0
        ),
        "top5_macro_observed_classes": (
            sum(oracle_top5[label] / oracle_total[label] for label in observed_labels)
            / len(observed_labels)
            if observed_labels
            else 0.0
        ),
        "per_class": {
            label: {
                "events": oracle_total[label],
                "top1": oracle_top1[label] / oracle_total[label] if oracle_total[label] else None,
                "top5": oracle_top5[label] / oracle_total[label] if oracle_total[label] else None,
            }
            for label in labels
        },
    }
    return {
        "threshold": float(threshold),
        "frame": aggregate(frame_tp, frame_fp, frame_fn),
        "event": {
            **aggregate(event_tp, event_fp, event_fn),
            "iou_threshold": float(event_iou_threshold),
        },
        "onset_frame": {
            **aggregate(onset_tp, onset_fp, onset_fn),
            "threshold": float(boundary_diagnostic_threshold),
            "diagnostic_only": True,
        },
        "offset_frame": {
            **aggregate(offset_tp, offset_fp, offset_fn),
            "threshold": float(boundary_diagnostic_threshold),
            "diagnostic_only": True,
        },
        "oracle_window_exact_label": oracle,
    }
