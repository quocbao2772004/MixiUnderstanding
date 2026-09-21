"""Class-aware boundary proposals for polyphonic QCES scenes.

This module is deliberately separate from the historical class-agnostic DETR
slots.  Every proposal keeps its event-class identity from frame prediction to
boundary decoding, so two different sounds are allowed to occupy the same
temporal interval.

The implementation follows the useful parts of RED/EPN while retaining the
frozen QCES detector as a safe semantic prior:

* detector logits plus a trainable residual predict direct frame presence;
* conditional start/end probabilities are converted into recurrent event
  presence, onset and offset probabilities by a parameter-free RED scan;
* a single bidirectional GRU predicts time-since-onset and time-to-offset for
  every class and frame;
* proposals are suppressed only within a class, never across classes.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ClassAwareRedEpnV1Config:
    feature_dim: int = 768
    num_classes: int = 191
    context_dim: int = 256
    epn_hidden_dim: int = 128
    context_kernel_size: int = 5
    dropout: float = 0.10
    direct_presence_weight: float = 0.75
    minimum_duration: float = 1.0 / 250.0

    def __post_init__(self) -> None:
        for name in ("feature_dim", "num_classes", "context_dim", "epn_hidden_dim"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.context_kernel_size <= 0 or self.context_kernel_size % 2 == 0:
            raise ValueError("context_kernel_size must be a positive odd integer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not 0.0 <= self.direct_presence_weight <= 1.0:
            raise ValueError("direct_presence_weight must be in [0,1]")
        if not math.isfinite(self.minimum_duration) or self.minimum_duration <= 0:
            raise ValueError("minimum_duration must be finite and positive")


@dataclass(frozen=True)
class ClassAwareRedEpnV1LossWeights:
    direct_presence: float = 1.0
    red_presence: float = 0.5
    onset: float = 5.0
    offset: float = 5.0
    duration_iou: float = 2.0
    clip_presence: float = 0.5
    residual_l2: float = 0.01

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"loss weight {name} must be finite and non-negative")


def red_event_probabilities(
    conditional_start_logits: torch.Tensor,
    conditional_end_logits: torch.Tensor,
    valid_frame_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run the differentiable RED recurrence independently for every class.

    Args:
        conditional_start_logits: ``[B,T,C]`` start-given-inactive logits.
        conditional_end_logits: ``[B,T,C]`` end-given-active logits.
        valid_frame_mask: optional ``[B,T]`` mask. Invalid frames are forced
            inactive and cannot emit boundaries.
    """

    if conditional_start_logits.shape != conditional_end_logits.shape:
        raise ValueError("start/end logits must have identical shape")
    if conditional_start_logits.ndim != 3:
        raise ValueError("start/end logits must have shape [B,T,C]")
    batch, frames, _ = conditional_start_logits.shape
    if valid_frame_mask is None:
        valid_frame_mask = torch.ones(
            batch, frames, dtype=torch.bool, device=conditional_start_logits.device
        )
    if valid_frame_mask.shape != (batch, frames):
        raise ValueError("valid_frame_mask must have shape [B,T]")
    valid = valid_frame_mask.bool()
    start = conditional_start_logits.sigmoid()
    end = conditional_end_logits.sigmoid()
    previous = torch.zeros_like(start[:, 0])
    presence_rows: list[torch.Tensor] = []
    onset_rows: list[torch.Tensor] = []
    offset_rows: list[torch.Tensor] = []
    for frame in range(frames):
        frame_valid = valid[:, frame, None]
        onset = start[:, frame] * (1.0 - previous)
        offset = end[:, frame] * previous
        current = onset + (1.0 - end[:, frame]) * previous
        current = torch.where(frame_valid, current, torch.zeros_like(current))
        onset = torch.where(frame_valid, onset, torch.zeros_like(onset))
        offset = torch.where(frame_valid, offset, torch.zeros_like(offset))
        presence_rows.append(current)
        onset_rows.append(onset)
        offset_rows.append(offset)
        previous = current
    return {
        "red_presence_probability": torch.stack(presence_rows, dim=1),
        "onset_probability": torch.stack(onset_rows, dim=1),
        "offset_probability": torch.stack(offset_rows, dim=1),
    }


class ClassAwareRedEpnV1(nn.Module):
    """Boundary-aware event inventory on frozen BEATs frame features/logits."""

    def __init__(self, config: ClassAwareRedEpnV1Config) -> None:
        super().__init__()
        self.config = config
        padding = config.context_kernel_size // 2
        self.input_norm = nn.LayerNorm(config.feature_dim)
        self.context_projection = nn.Linear(config.feature_dim, config.context_dim)
        self.context_conv = nn.Sequential(
            nn.Conv1d(
                config.context_dim,
                config.context_dim,
                config.context_kernel_size,
                padding=padding,
                groups=config.context_dim,
            ),
            nn.GELU(),
            nn.Conv1d(config.context_dim, config.context_dim, 1),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.presence_residual_head = nn.Linear(config.context_dim, config.num_classes)
        self.conditional_start_head = nn.Linear(config.context_dim, config.num_classes)
        self.conditional_end_head = nn.Linear(config.context_dim, config.num_classes)
        self.epn = nn.GRU(
            input_size=3 * config.num_classes,
            hidden_size=config.epn_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.duration_head = nn.Linear(2 * config.epn_hidden_dim, 2 * config.num_classes)

        # Epoch zero behaves like the frozen detector for semantic presence.
        nn.init.zeros_(self.presence_residual_head.weight)
        nn.init.zeros_(self.presence_residual_head.bias)
        # Events are sparse, so RED transitions should begin rare rather than
        # at the unstable 0.5/0.5 equilibrium.
        nn.init.zeros_(self.conditional_start_head.weight)
        nn.init.zeros_(self.conditional_end_head.weight)
        nn.init.constant_(self.conditional_start_head.bias, -4.0)
        nn.init.constant_(self.conditional_end_head.bias, -4.0)
        nn.init.zeros_(self.duration_head.weight)
        initial_duration = 0.05
        nn.init.constant_(
            self.duration_head.bias,
            math.log(math.expm1(initial_duration)),
        )

    def forward(
        self,
        features: torch.Tensor,
        detector_logits: torch.Tensor,
        valid_frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        config = self.config
        if features.ndim != 3 or features.shape[-1] != config.feature_dim:
            raise ValueError("features must have shape [B,T,feature_dim]")
        if detector_logits.shape != (*features.shape[:2], config.num_classes):
            raise ValueError("detector_logits must have shape [B,T,num_classes]")
        batch, frames, _ = features.shape
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(
                batch, frames, dtype=torch.bool, device=features.device
            )
        if valid_frame_mask.shape != (batch, frames):
            raise ValueError("valid_frame_mask must have shape [B,T]")
        valid_frame_mask = valid_frame_mask.bool()

        context = self.context_projection(self.input_norm(features))
        context = context + self.context_conv(context.transpose(1, 2)).transpose(1, 2)
        presence_residual = self.presence_residual_head(context)
        direct_logits = detector_logits + presence_residual
        conditional_start_logits = self.conditional_start_head(context)
        conditional_end_logits = self.conditional_end_head(context)
        red = red_event_probabilities(
            conditional_start_logits,
            conditional_end_logits,
            valid_frame_mask,
        )
        direct_probability = direct_logits.sigmoid()
        weight = config.direct_presence_weight
        fused_probability = (
            weight * direct_probability + (1.0 - weight) * red["red_presence_probability"]
        )
        valid_float = valid_frame_mask[:, :, None].to(fused_probability.dtype)
        fused_probability = fused_probability * valid_float

        epn_input = torch.cat(
            (
                fused_probability,
                red["onset_probability"],
                red["offset_probability"],
            ),
            dim=-1,
        )
        epn_output, _ = self.epn(epn_input)
        duration_raw = self.duration_head(epn_output).view(
            batch, frames, config.num_classes, 2
        )
        durations = F.softplus(duration_raw).clamp_min(config.minimum_duration)
        durations = durations * valid_float[..., None]

        masked_direct = direct_logits.masked_fill(~valid_frame_mask[:, :, None], -1e4)
        clip_presence_logits = masked_direct.amax(dim=1)
        frame_center = (
            (torch.arange(frames, device=features.device, dtype=features.dtype) + 0.5)
            / float(frames)
        )[None, :, None]
        proposal_start = (frame_center - durations[..., 0]).clamp(0.0, 1.0)
        proposal_end = (frame_center + durations[..., 1]).clamp(0.0, 1.0)
        proposal_end = proposal_end.maximum(
            (proposal_start + config.minimum_duration).clamp_max(1.0)
        )
        return {
            "presence_residual": presence_residual,
            "direct_presence_logits": direct_logits,
            "direct_presence_probability": direct_probability,
            "conditional_start_logits": conditional_start_logits,
            "conditional_end_logits": conditional_end_logits,
            "red_presence_probability": red["red_presence_probability"],
            "onset_probability": red["onset_probability"],
            "offset_probability": red["offset_probability"],
            "fused_presence_probability": fused_probability,
            "duration_since_onset": durations[..., 0],
            "duration_to_offset": durations[..., 1],
            "proposal_start": proposal_start,
            "proposal_end": proposal_end,
            "clip_presence_logits": clip_presence_logits,
        }


def _masked_focal_probability_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    valid_frame_mask: torch.Tensor,
    *,
    positive_alpha: float,
    gamma: float = 2.0,
) -> torch.Tensor:
    if probability.shape != target.shape or probability.ndim != 3:
        raise ValueError("probability and target must have shape [B,T,C]")
    valid = valid_frame_mask[:, :, None].to(probability.dtype)
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    positive = -(1.0 - probability).pow(gamma) * target * probability.log()
    negative = -probability.pow(gamma) * (1.0 - target) * (1.0 - probability).log()

    # A 191-class scene has only a few active labels and boundary frames are
    # sparser still. Averaging over B*T*C lets the easy negatives dominate and
    # recreates the silence-collapse failure seen in the old temporal gate.
    # Normalize positive and negative mass independently before combining them.
    positive_mass = (target * valid).sum().clamp_min(1.0)
    negative_mass = ((1.0 - target) * valid).sum().clamp_min(1.0)
    positive_loss = (positive * valid).sum() / positive_mass
    negative_loss = (negative * valid).sum() / negative_mass
    return positive_alpha * positive_loss + (1.0 - positive_alpha) * negative_loss


def _masked_focal_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_frame_mask: torch.Tensor,
    *,
    positive_alpha: float,
    gamma: float = 2.0,
) -> torch.Tensor:
    return _masked_focal_probability_loss(
        logits.sigmoid(),
        target,
        valid_frame_mask,
        positive_alpha=positive_alpha,
        gamma=gamma,
    )


def class_aware_red_epn_v1_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    valid_frame_mask: torch.Tensor,
    *,
    weights: ClassAwareRedEpnV1LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    """Boundary-aware objective with separately normalized active regions."""

    weights = weights or ClassAwareRedEpnV1LossWeights()
    presence_target = targets["frame_presence"]
    onset_target = targets["frame_onset"]
    offset_target = targets["frame_offset"]
    direct_presence = _masked_focal_logits_loss(
        outputs["direct_presence_logits"],
        presence_target,
        valid_frame_mask,
        positive_alpha=0.75,
    )
    red_presence = _masked_focal_probability_loss(
        outputs["red_presence_probability"],
        presence_target,
        valid_frame_mask,
        positive_alpha=0.75,
    )
    onset = _masked_focal_probability_loss(
        outputs["onset_probability"],
        onset_target,
        valid_frame_mask,
        positive_alpha=0.95,
    )
    offset = _masked_focal_probability_loss(
        outputs["offset_probability"],
        offset_target,
        valid_frame_mask,
        positive_alpha=0.95,
    )

    active = (presence_target > 0.5) & valid_frame_mask[:, :, None]
    if active.any():
        center = targets["frame_center"]
        target_start = (center - targets["duration_since_onset"]).clamp(0.0, 1.0)
        target_end = (center + targets["duration_to_offset"]).clamp(0.0, 1.0)
        predicted_start = outputs["proposal_start"]
        predicted_end = outputs["proposal_end"]
        intersection = (
            torch.minimum(predicted_end, target_end)
            - torch.maximum(predicted_start, target_start)
        ).clamp_min(0.0)
        union = (
            (predicted_end - predicted_start).clamp_min(0.0)
            + (target_end - target_start).clamp_min(0.0)
            - intersection
        )
        iou_loss = 1.0 - intersection / union.clamp_min(1e-6)
        inverse_duration = 1.0 / (target_end - target_start).clamp_min(1.0 / 250.0)
        duration_iou = (iou_loss[active] * inverse_duration[active]).sum() / inverse_duration[
            active
        ].sum().clamp_min(1.0)
    else:
        duration_iou = outputs["proposal_start"].sum() * 0.0

    clip_logits = outputs["clip_presence_logits"]
    clip_target = targets["clip_presence"]
    clip_positive = -(clip_target * F.logsigmoid(clip_logits)).sum() / clip_target.sum().clamp_min(1.0)
    clip_negative_target = 1.0 - clip_target
    clip_negative = -(
        clip_negative_target * F.logsigmoid(-clip_logits)
    ).sum() / clip_negative_target.sum().clamp_min(1.0)
    clip_presence = 0.75 * clip_positive + 0.25 * clip_negative
    residual_l2 = outputs["presence_residual"].square().mean()
    loss = (
        weights.direct_presence * direct_presence
        + weights.red_presence * red_presence
        + weights.onset * onset
        + weights.offset * offset
        + weights.duration_iou * duration_iou
        + weights.clip_presence * clip_presence
        + weights.residual_l2 * residual_l2
    )
    return {
        "loss": loss,
        "direct_presence": direct_presence,
        "red_presence": red_presence,
        "onset": onset,
        "offset": offset,
        "duration_iou": duration_iou,
        "clip_presence": clip_presence,
        "residual_l2": residual_l2,
    }


def interval_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(
        0.0,
        min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0])),
    )
    union = (
        max(0.0, float(left[1]) - float(left[0]))
        + max(0.0, float(right[1]) - float(right[0]))
        - intersection
    )
    return intersection / max(union, 1e-8)


def decode_class_aware_proposals(
    outputs: Mapping[str, torch.Tensor],
    batch_index: int,
    *,
    max_classes: int = 32,
    candidate_frames_per_class: int = 16,
    max_events_per_class: int = 3,
    max_events: int = 20,
    nms_iou: float = 0.50,
    minimum_duration: float = 0.02,
) -> list[dict[str, float | int]]:
    """Decode class-labelled regions with class-wise NMS.

    NMS never compares different labels.  This is the critical polyphonic
    behavior absent from the historical class-agnostic slot decoder.
    """

    presence = outputs["fused_presence_probability"][batch_index].detach().float().cpu()
    start = outputs["proposal_start"][batch_index].detach().float().cpu()
    end = outputs["proposal_end"][batch_index].detach().float().cpu()
    clip = outputs["clip_presence_logits"][batch_index].detach().float().cpu().sigmoid()
    frames, classes = presence.shape
    if start.shape != (frames, classes) or end.shape != (frames, classes):
        raise ValueError("proposal tensors disagree with presence shape")
    max_classes = min(max(int(max_classes), 1), classes)
    class_ids = clip.topk(max_classes).indices.tolist()
    decoded: list[dict[str, float | int]] = []
    for class_id in class_ids:
        frame_count = min(max(int(candidate_frames_per_class), 1), frames)
        frame_ids = presence[:, class_id].topk(frame_count).indices.tolist()
        candidates = []
        for frame_id in frame_ids:
            interval = [float(start[frame_id, class_id]), float(end[frame_id, class_id])]
            if interval[1] - interval[0] < minimum_duration:
                continue
            left = max(0, min(frames - 1, int(math.floor(interval[0] * frames))))
            right = max(left + 1, min(frames, int(math.ceil(interval[1] * frames))))
            region_score = float(presence[left:right, class_id].mean())
            score = region_score * math.sqrt(max(float(clip[class_id]), 1e-8))
            candidates.append(
                {
                    "label_id": int(class_id),
                    "start": interval[0],
                    "end": interval[1],
                    "score": score,
                    "clip_score": float(clip[class_id]),
                    "center_frame": int(frame_id),
                }
            )
        candidates.sort(key=lambda row: float(row["score"]), reverse=True)
        kept: list[dict[str, float | int]] = []
        for candidate in candidates:
            candidate_interval = [float(candidate["start"]), float(candidate["end"])]
            if any(
                interval_iou(candidate_interval, [float(row["start"]), float(row["end"])])
                >= nms_iou
                for row in kept
            ):
                continue
            kept.append(candidate)
            if len(kept) >= max_events_per_class:
                break
        decoded.extend(kept)
    decoded.sort(key=lambda row: float(row["score"]), reverse=True)
    return decoded[: max(int(max_events), 1)]
