"""Class-agnostic temporal event slots for relational acoustic evidence.

The model deliberately predicts *when an event exists* without predicting its
class.  A frozen detector may score a question's anchor class inside the
resulting slots later, but neither the answer label nor answer waveform is an
input to this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class RelationalEventSlotsV1Config:
    feature_dim: int = 768
    hidden_dim: int = 256
    num_slots: int = 8
    num_heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 2
    feedforward_dim: int = 768
    dropout: float = 0.1
    max_audio_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name in (
            "feature_dim",
            "hidden_dim",
            "num_slots",
            "num_heads",
            "encoder_layers",
            "decoder_layers",
            "feedforward_dim",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not math.isfinite(self.max_audio_seconds) or self.max_audio_seconds <= 0:
            raise ValueError("max_audio_seconds must be finite and positive")


@dataclass(frozen=True)
class RelationalEventSlotsV1LossWeights:
    objectness: float = 1.0
    box_l1: float = 5.0
    box_iou: float = 2.0
    frame_bce: float = 1.0
    frame_dice: float = 1.0
    onset_bce: float = 0.5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"loss weight {name} must be finite and non-negative")


def _sinusoidal_position_encoding(
    length: int, dimension: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(
        torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    result = torch.zeros(length, dimension, device=device, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * frequency)
    result[:, 1::2] = torch.cos(position * frequency[: result[:, 1::2].shape[1]])
    return result.to(dtype=dtype)


def center_width_to_intervals(center_width: torch.Tensor) -> torch.Tensor:
    """Convert normalized center/width to ordered [start,end] intervals."""

    if center_width.shape[-1] != 2:
        raise ValueError("center_width must end in width 2")
    center = center_width[..., 0]
    width = center_width[..., 1]
    start = (center - 0.5 * width).clamp(0.0, 1.0)
    end = (center + 0.5 * width).clamp(0.0, 1.0)
    return torch.stack((start, end.maximum(start + 1e-5).clamp_max(1.0)), dim=-1)


def interval_iou_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Pairwise one-dimensional IoU for normalized [start,end] intervals."""

    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != (2,) or right.shape[1:] != (2,):
        raise ValueError("interval inputs must have shape [N,2] and [M,2]")
    intersection = (
        torch.minimum(left[:, None, 1], right[None, :, 1])
        - torch.maximum(left[:, None, 0], right[None, :, 0])
    ).clamp_min(0.0)
    left_width = (left[:, 1] - left[:, 0]).clamp_min(0.0)[:, None]
    right_width = (right[:, 1] - right[:, 0]).clamp_min(0.0)[None, :]
    union = left_width + right_width - intersection
    return intersection / union.clamp_min(1e-8)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index + 1 < len(self.layers):
                value = F.gelu(value)
        return value


class RelationalEventSlotsV1(nn.Module):
    """DETR-style, class-agnostic temporal interval predictor."""

    def __init__(self, config: RelationalEventSlotsV1Config) -> None:
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.feature_dim)
        self.input_projection = nn.Linear(config.feature_dim, config.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.encoder_layers, norm=nn.LayerNorm(config.hidden_dim)
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.decoder_layers, norm=nn.LayerNorm(config.hidden_dim)
        )
        self.slot_queries = nn.Embedding(config.num_slots, config.hidden_dim)
        self.objectness_head = nn.Linear(config.hidden_dim, 1)
        self.box_head = _MLP(config.hidden_dim, config.hidden_dim, 2, 3)
        self.frame_event_head = nn.Linear(config.hidden_dim, 1)
        self.frame_onset_head = nn.Linear(config.hidden_dim, 1)

        # An evenly spread reference center breaks slot symmetry, while the
        # learned delta can move every slot anywhere in the recording.
        reference = (torch.arange(config.num_slots, dtype=torch.float32) + 0.5) / config.num_slots
        self.register_buffer("reference_center", reference, persistent=True)
        # QCES events are commonly around one second in a ten-second scene.
        # Starting at width 0.10 avoids DETR's otherwise destructive 0.50
        # interval prior while leaving width fully learnable.
        nn.init.zeros_(self.box_head.layers[-1].weight)
        nn.init.zeros_(self.box_head.layers[-1].bias)
        with torch.no_grad():
            self.box_head.layers[-1].bias[1] = math.log(0.10 / 0.90)

    def forward(
        self, features: torch.Tensor, valid_frame_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.config.feature_dim:
            raise ValueError("features must have shape [B,T,feature_dim]")
        batch, frames, _ = features.shape
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(batch, frames, dtype=torch.bool, device=features.device)
        if valid_frame_mask.shape != (batch, frames):
            raise ValueError("valid_frame_mask must have shape [B,T]")
        valid_frame_mask = valid_frame_mask.bool()
        position = _sinusoidal_position_encoding(
            frames, self.config.hidden_dim, device=features.device, dtype=features.dtype
        )
        memory = self.input_projection(self.input_norm(features)) + position[None]
        memory = self.encoder(memory, src_key_padding_mask=~valid_frame_mask)
        queries = self.slot_queries.weight[None].expand(batch, -1, -1)
        decoded = self.decoder(
            queries,
            memory,
            memory_key_padding_mask=~valid_frame_mask,
        )
        raw_box = self.box_head(decoded)
        reference_logits = torch.logit(
            self.reference_center.clamp(1e-4, 1.0 - 1e-4)
        )[None]
        center = torch.sigmoid(reference_logits + raw_box[..., 0])
        width = torch.sigmoid(raw_box[..., 1])
        center_width = torch.stack((center, width), dim=-1)
        intervals = center_width_to_intervals(center_width)
        return {
            "objectness_logits": self.objectness_head(decoded).squeeze(-1),
            "center_width": center_width,
            "intervals": intervals,
            "frame_event_logits": self.frame_event_head(memory).squeeze(-1),
            "frame_onset_logits": self.frame_onset_head(memory).squeeze(-1),
        }


def hungarian_match_event_slots(
    objectness_logits: torch.Tensor,
    predicted_intervals: torch.Tensor,
    target_intervals: Sequence[torch.Tensor],
    *,
    objectness_cost: float = 1.0,
    l1_cost: float = 5.0,
    iou_cost: float = 2.0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Match slots to gold intervals independently for each batch item."""

    if objectness_logits.ndim != 2 or predicted_intervals.shape != (*objectness_logits.shape, 2):
        raise ValueError("invalid slot prediction shapes")
    if len(target_intervals) != objectness_logits.shape[0]:
        raise ValueError("target batch length mismatch")
    matches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch_index, targets in enumerate(target_intervals):
        targets = targets.to(device=predicted_intervals.device, dtype=predicted_intervals.dtype)
        if targets.ndim != 2 or targets.shape[-1] != 2:
            raise ValueError("every target interval tensor must have shape [N,2]")
        if targets.shape[0] == 0:
            empty = torch.empty(0, dtype=torch.long, device=predicted_intervals.device)
            matches.append((empty, empty))
            continue
        prediction = predicted_intervals[batch_index]
        l1 = torch.cdist(prediction, targets, p=1)
        iou = interval_iou_matrix(prediction, targets)
        confidence = objectness_logits[batch_index].sigmoid()[:, None]
        cost = l1_cost * l1 + iou_cost * (1.0 - iou) - objectness_cost * confidence
        row, column = linear_sum_assignment(cost.detach().float().cpu().numpy())
        matches.append(
            (
                torch.as_tensor(row, dtype=torch.long, device=prediction.device),
                torch.as_tensor(column, dtype=torch.long, device=prediction.device),
            )
        )
    return matches


def _masked_dice_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid() * valid
    target = target * valid
    numerator = 2.0 * (probability * target).sum(dim=1) + 1.0
    denominator = probability.sum(dim=1) + target.sum(dim=1) + 1.0
    return (1.0 - numerator / denominator).mean()


def relational_event_slots_v1_loss(
    outputs: Mapping[str, torch.Tensor],
    target_intervals: Sequence[torch.Tensor],
    frame_event_target: torch.Tensor,
    frame_onset_target: torch.Tensor,
    valid_frame_mask: torch.Tensor,
    *,
    weights: RelationalEventSlotsV1LossWeights | None = None,
    no_object_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    weights = weights or RelationalEventSlotsV1LossWeights()
    objectness_logits = outputs["objectness_logits"]
    predicted_intervals = outputs["intervals"]
    matches = hungarian_match_event_slots(
        objectness_logits,
        predicted_intervals,
        target_intervals,
        objectness_cost=weights.objectness,
        l1_cost=weights.box_l1,
        iou_cost=weights.box_iou,
    )
    objectness_target = torch.zeros_like(objectness_logits)
    matched_prediction: list[torch.Tensor] = []
    matched_target: list[torch.Tensor] = []
    for batch_index, (prediction_index, target_index) in enumerate(matches):
        objectness_target[batch_index, prediction_index] = 1.0
        if prediction_index.numel():
            matched_prediction.append(predicted_intervals[batch_index, prediction_index])
            matched_target.append(
                target_intervals[batch_index]
                .to(device=predicted_intervals.device, dtype=predicted_intervals.dtype)[target_index]
            )
    objectness_element = F.binary_cross_entropy_with_logits(
        objectness_logits, objectness_target, reduction="none"
    )
    objectness_weight = torch.where(
        objectness_target > 0.5,
        torch.ones_like(objectness_target),
        torch.full_like(objectness_target, float(no_object_weight)),
    )
    objectness_loss = (objectness_element * objectness_weight).sum() / objectness_weight.sum().clamp_min(1)
    if matched_prediction:
        prediction_all = torch.cat(matched_prediction, dim=0)
        target_all = torch.cat(matched_target, dim=0)
        box_l1 = F.l1_loss(prediction_all, target_all)
        box_iou = 1.0 - torch.diagonal(interval_iou_matrix(prediction_all, target_all)).mean()
    else:
        box_l1 = predicted_intervals.sum() * 0.0
        box_iou = predicted_intervals.sum() * 0.0

    valid = valid_frame_mask.to(frame_event_target.dtype)
    frame_bce_element = F.binary_cross_entropy_with_logits(
        outputs["frame_event_logits"], frame_event_target, reduction="none"
    )
    frame_bce = (frame_bce_element * valid).sum() / valid.sum().clamp_min(1)
    frame_dice = _masked_dice_loss(outputs["frame_event_logits"], frame_event_target, valid)
    # Onsets occupy very few frames.  A bounded positive weight avoids both
    # trivial all-negative predictions and unstable gradients.
    positives = (frame_onset_target * valid).sum()
    negatives = valid.sum() - positives
    onset_pos_weight = (negatives / positives.clamp_min(1)).clamp(1.0, 50.0)
    onset_bce_element = F.binary_cross_entropy_with_logits(
        outputs["frame_onset_logits"],
        frame_onset_target,
        reduction="none",
        pos_weight=onset_pos_weight,
    )
    onset_bce = (onset_bce_element * valid).sum() / valid.sum().clamp_min(1)
    total = (
        weights.objectness * objectness_loss
        + weights.box_l1 * box_l1
        + weights.box_iou * box_iou
        + weights.frame_bce * frame_bce
        + weights.frame_dice * frame_dice
        + weights.onset_bce * onset_bce
    )
    return {
        "loss": total,
        "objectness": objectness_loss,
        "box_l1": box_l1,
        "box_iou": box_iou,
        "frame_bce": frame_bce,
        "frame_dice": frame_dice,
        "onset_bce": onset_bce,
    }


def decode_event_slots_v1(
    outputs: Mapping[str, torch.Tensor], *, objectness_threshold: float
) -> list[list[dict[str, float | int]]]:
    """Decode slots, sorted by onset, without applying a class query."""

    if not 0.0 <= objectness_threshold <= 1.0:
        raise ValueError("objectness_threshold must be in [0,1]")
    confidence = outputs["objectness_logits"].sigmoid().detach().cpu()
    intervals = outputs["intervals"].detach().cpu()
    decoded: list[list[dict[str, float | int]]] = []
    for batch_index in range(confidence.shape[0]):
        rows: list[dict[str, float | int]] = []
        for slot_index in range(confidence.shape[1]):
            score = float(confidence[batch_index, slot_index])
            if score < objectness_threshold:
                continue
            start, end = intervals[batch_index, slot_index].tolist()
            rows.append(
                {
                    "slot_index": slot_index,
                    "score": score,
                    "start": float(start),
                    "end": float(end),
                }
            )
        rows.sort(key=lambda row: (float(row["start"]), float(row["end"])))
        decoded.append(rows)
    return decoded
