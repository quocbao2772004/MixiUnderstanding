"""Overlap-aware, class-agnostic temporal event slots.

Unlike the historical relational-slot decoder, this module never collapses
all active frames into one union mask.  Every DETR query predicts its own
interval and frame mask, so temporally overlapping events remain distinct.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
    RelationalEventSlotsV1LossWeights,
    _sinusoidal_position_encoding,
    center_width_to_intervals,
    hungarian_match_event_slots,
    relational_event_slots_v1_loss,
)


@dataclass(frozen=True)
class OverlapEventSlotsV2LossWeights:
    slot_mask_bce: float = 1.0
    slot_mask_dice: float = 2.0

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"loss weight {name} must be finite and non-negative")


class OverlapEventSlotsV2(RelationalEventSlotsV1):
    """DETR-style event instances with a separate frame mask per slot."""

    def __init__(self, config: RelationalEventSlotsV1Config) -> None:
        super().__init__(config)
        self.slot_mask_query = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.slot_mask_memory = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

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
        decoded = self.decoder(queries, memory, memory_key_padding_mask=~valid_frame_mask)

        raw_box = self.box_head(decoded)
        reference_logits = torch.logit(self.reference_center.clamp(1e-4, 1.0 - 1e-4))[None]
        center = torch.sigmoid(reference_logits + raw_box[..., 0])
        width = torch.sigmoid(raw_box[..., 1])
        center_width = torch.stack((center, width), dim=-1)

        mask_query = self.slot_mask_query(decoded)
        mask_memory = self.slot_mask_memory(memory)
        slot_mask_logits = torch.einsum("bsh,bth->bst", mask_query, mask_memory)
        slot_mask_logits = slot_mask_logits / math.sqrt(self.config.hidden_dim)
        slot_mask_logits = slot_mask_logits.masked_fill(~valid_frame_mask[:, None], -20.0)

        return {
            "objectness_logits": self.objectness_head(decoded).squeeze(-1),
            "center_width": center_width,
            "intervals": center_width_to_intervals(center_width),
            "frame_event_logits": self.frame_event_head(memory).squeeze(-1),
            "frame_onset_logits": self.frame_onset_head(memory).squeeze(-1),
            "slot_mask_logits": slot_mask_logits,
            "slot_embeddings": decoded,
        }


def _matched_slot_mask_loss(
    outputs: Mapping[str, torch.Tensor],
    target_intervals: Sequence[torch.Tensor],
    target_event_masks: Sequence[torch.Tensor],
    valid_frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = outputs["slot_mask_logits"]
    matches = hungarian_match_event_slots(
        outputs["objectness_logits"], outputs["intervals"], target_intervals
    )
    bce_terms: list[torch.Tensor] = []
    dice_terms: list[torch.Tensor] = []
    for batch_index, (prediction_index, target_index) in enumerate(matches):
        if prediction_index.numel() == 0:
            continue
        prediction = logits[batch_index, prediction_index]
        target = target_event_masks[batch_index].to(
            device=prediction.device, dtype=prediction.dtype
        )[target_index]
        valid = valid_frame_mask[batch_index].to(prediction.dtype)[None]

        # Normalize positive and negative regions independently.  This avoids
        # the long inactive tail driving every event mask toward zero.
        element = F.binary_cross_entropy_with_logits(prediction, target, reduction="none")
        positive = (element * target * valid).sum(dim=1) / (target * valid).sum(dim=1).clamp_min(1)
        negative_mask = (1.0 - target) * valid
        negative = (element * negative_mask).sum(dim=1) / negative_mask.sum(dim=1).clamp_min(1)
        bce_terms.append((positive + negative).mean())

        probability = prediction.sigmoid() * valid
        masked_target = target * valid
        numerator = 2.0 * (probability * masked_target).sum(dim=1) + 1.0
        denominator = probability.sum(dim=1) + masked_target.sum(dim=1) + 1.0
        dice_terms.append((1.0 - numerator / denominator).mean())

    zero = logits.sum() * 0.0
    return (
        torch.stack(bce_terms).mean() if bce_terms else zero,
        torch.stack(dice_terms).mean() if dice_terms else zero,
    )


def overlap_event_slots_v2_loss(
    outputs: Mapping[str, torch.Tensor],
    target_intervals: Sequence[torch.Tensor],
    target_event_masks: Sequence[torch.Tensor],
    frame_event_target: torch.Tensor,
    frame_onset_target: torch.Tensor,
    valid_frame_mask: torch.Tensor,
    *,
    base_weights: RelationalEventSlotsV1LossWeights | None = None,
    overlap_weights: OverlapEventSlotsV2LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    base_weights = base_weights or RelationalEventSlotsV1LossWeights()
    overlap_weights = overlap_weights or OverlapEventSlotsV2LossWeights()
    losses = relational_event_slots_v1_loss(
        outputs,
        target_intervals,
        frame_event_target,
        frame_onset_target,
        valid_frame_mask,
        weights=base_weights,
    )
    mask_bce, mask_dice = _matched_slot_mask_loss(
        outputs, target_intervals, target_event_masks, valid_frame_mask
    )
    losses["slot_mask_bce"] = mask_bce
    losses["slot_mask_dice"] = mask_dice
    losses["loss"] = (
        losses["loss"]
        + overlap_weights.slot_mask_bce * mask_bce
        + overlap_weights.slot_mask_dice * mask_dice
    )
    return losses


def decode_overlap_event_slots_v2(
    outputs: Mapping[str, torch.Tensor], *, objectness_threshold: float
) -> list[list[dict[str, float | int]]]:
    """Decode query-specific intervals without temporal NMS.

    Temporal NMS is deliberately absent: two real sounds are allowed to have
    highly overlapping intervals.  Duplicate suppression is learned through
    DETR's one-to-one matching and the no-object targets.
    """

    if not 0.0 <= objectness_threshold <= 1.0:
        raise ValueError("objectness_threshold must be in [0,1]")
    confidence = outputs["objectness_logits"].sigmoid().detach().cpu()
    intervals = outputs["intervals"].detach().cpu()
    mask_probability = outputs["slot_mask_logits"].sigmoid().detach().cpu()
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
                    "mask_peak": float(mask_probability[batch_index, slot_index].max()),
                }
            )
        rows.sort(key=lambda row: (float(row["start"]), float(row["end"])))
        decoded.append(rows)
    return decoded
