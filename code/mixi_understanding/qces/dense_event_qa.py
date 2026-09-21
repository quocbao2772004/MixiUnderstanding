"""Dense question-conditioned temporal reasoning over detector features.

This module implements the Q-DOR core without changing the historical QCES
models.  It deliberately reasons over the complete ``[time, class]`` detector
grid instead of thresholding it into a lossy proposal list.  Its only temporal
unit is a fixed 40 ms half-open frame interval.

The two supported relations are onset relations:

``before``
    Candidate frames whose onset is earlier than the selected anchor onset.
``after``
    Candidate frames whose onset is later than the selected anchor onset.

The relation mask is the CDF (or survival function) of a soft anchor-onset
distribution.  Applying it to all class tracks is therefore ``O(T * C)`` in
time and memory; no ``T * T`` pair matrix is constructed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


FRAME_HOP_SECONDS = 0.04
RELATION_BEFORE = 0
RELATION_AFTER = 1
RELATION_NAMES = ("before", "after")


@dataclass(frozen=True)
class DenseTemporalReasonerConfig:
    """Shape and optimization-independent settings for :class:`DenseTemporalReasoner`.

    ``feature_dim`` is the width of the frame features exported by the frozen
    (or lightly fine-tuned) SED encoder.  ``num_classes`` defaults to the QCES
    200-label ontology.  The frame hop is intentionally not configurable:
    manifests, masks, decoding, and reported boundary metrics must share the
    same 40 ms grid.
    """

    feature_dim: int
    num_classes: int = 200
    hidden_dim: int = 256
    max_ordinal: int = 10
    dropout: float = 0.1
    pooling_temperature: float = 0.5
    ordinal_sigma: float = 0.65
    detector_logit_scale: float = 1.0
    feature_logit_scale: float = 0.1
    ordinal_prior_scale: float = 1.0
    frame_hop_seconds: float = FRAME_HOP_SECONDS

    def __post_init__(self) -> None:
        positive = {
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "max_ordinal": self.max_ordinal,
            "pooling_temperature": self.pooling_temperature,
            "ordinal_sigma": self.ordinal_sigma,
            "detector_logit_scale": self.detector_logit_scale,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"reasoner dimensions/scales must be positive: {invalid}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.feature_logit_scale < 0 or self.ordinal_prior_scale < 0:
            raise ValueError("feature and ordinal scales must be non-negative")
        if not math.isclose(
            self.frame_hop_seconds,
            FRAME_HOP_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Q-DOR uses a fixed 40 ms frame grid")


@dataclass(frozen=True)
class DenseTemporalLossWeights:
    """Weights for the directly supervised answer and evidence objectives."""

    answer_choice: float = 1.0
    answerability: float = 0.5
    anchor_bce: float = 1.0
    anchor_dice: float = 1.0
    answer_bce: float = 1.0
    answer_dice: float = 1.0
    relation_consistency: float = 0.2

    def __post_init__(self) -> None:
        invalid = {
            name: value
            for name, value in self.__dict__.items()
            if not math.isfinite(value) or value < 0
        }
        if invalid:
            raise ValueError(f"loss weights must be finite and non-negative: {invalid}")


def intervals_to_40ms_mask(
    intervals: torch.Tensor | Sequence[Sequence[float]],
    num_frames: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rasterize half-open second intervals on the fixed 40 ms frame grid.

    ``intervals`` may have shape ``[N, 2]`` or ``[B, N, 2]``.  A frame is
    active when its half-open interval overlaps an annotation by a positive
    duration.  ``[nan, nan]`` rows are accepted as batch padding; other
    malformed rows raise an error instead of silently corrupting boundaries.
    """

    if not isinstance(num_frames, int) or num_frames <= 0:
        raise ValueError("num_frames must be a positive integer")
    tensor = torch.as_tensor(intervals, device=device, dtype=torch.float32)
    # ``[]`` is the natural annotation for a no-evidence answer mask.
    if tensor.ndim == 1 and tensor.numel() == 0:
        tensor = tensor.reshape(0, 2)
    if tensor.ndim not in (2, 3) or tensor.shape[-1] != 2:
        raise ValueError("intervals must have shape [N, 2] or [B, N, 2]")
    unbatched = tensor.ndim == 2
    if unbatched:
        tensor = tensor.unsqueeze(0)

    finite = torch.isfinite(tensor).all(dim=-1)
    all_nan = torch.isnan(tensor).all(dim=-1)
    partially_nonfinite = ~(finite | all_nan)
    if bool(partially_nonfinite.any()):
        raise ValueError("an interval must be finite or an all-NaN padding row")
    onset, offset = tensor.unbind(dim=-1)
    invalid = finite & ((onset < 0) | (offset <= onset))
    if bool(invalid.any()):
        raise ValueError("finite intervals require 0 <= onset < offset")

    frame_start = (
        torch.arange(num_frames, device=tensor.device, dtype=tensor.dtype)
        * FRAME_HOP_SECONDS
    )
    frame_end = frame_start + FRAME_HOP_SECONDS
    overlap = (
        finite.unsqueeze(-1)
        & (onset.unsqueeze(-1) < frame_end)
        & (offset.unsqueeze(-1) > frame_start)
    )
    mask = overlap.any(dim=1).to(dtype=dtype)
    return mask[0] if unbatched else mask


def masks_to_40ms_intervals(
    masks: torch.Tensor,
    *,
    threshold: float = 0.5,
    valid_frame_mask: torch.Tensor | None = None,
) -> list[tuple[float, float]] | list[list[tuple[float, float]]]:
    """Decode contiguous mask regions into fixed-grid half-open intervals."""

    if masks.ndim not in (1, 2):
        raise ValueError("masks must have shape [T] or [B, T]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    unbatched = masks.ndim == 1
    binary = (masks.detach().cpu() >= threshold)
    if unbatched:
        binary = binary.unsqueeze(0)
    if valid_frame_mask is not None:
        valid = valid_frame_mask.detach().cpu()
        if unbatched and valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != binary.shape:
            raise ValueError("valid_frame_mask shape must match masks")
        binary = binary & valid.to(torch.bool)

    decoded: list[list[tuple[float, float]]] = []
    for row in binary:
        intervals: list[tuple[float, float]] = []
        start: int | None = None
        for index, active in enumerate(row.tolist() + [False]):
            if active and start is None:
                start = index
            elif not active and start is not None:
                intervals.append(
                    (start * FRAME_HOP_SECONDS, index * FRAME_HOP_SECONDS)
                )
                start = None
        decoded.append(intervals)
    return decoded[0] if unbatched else decoded


def _valid_frame_mask(
    valid_frame_mask: torch.Tensor | None,
    *,
    batch: int,
    frames: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate and normalize an optional frame mask to boolean ``[B, T]``."""

    if valid_frame_mask is None:
        return torch.ones(batch, frames, dtype=torch.bool, device=device)
    if valid_frame_mask.shape != (batch, frames):
        raise ValueError("valid_frame_mask must have shape [B, T]")
    valid = valid_frame_mask.to(device=device)
    if valid.dtype != torch.bool:
        if valid.is_floating_point() and not bool(torch.isfinite(valid).all()):
            raise ValueError("valid_frame_mask must be finite")
        if bool(((valid != 0) & (valid != 1)).any()):
            raise ValueError("valid_frame_mask entries must be binary")
        valid = valid.to(torch.bool)
    if bool((~valid.any(dim=1)).any()):
        raise ValueError("every example must contain at least one valid frame")
    return valid


def soft_onset_distribution(
    mask_probability: torch.Tensor,
    valid_frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert ``[B, T]`` activity probabilities to normalized onset mass."""

    if mask_probability.ndim != 2:
        raise ValueError("mask_probability must have shape [B, T]")
    if mask_probability.shape[1] < 1:
        raise ValueError("mask_probability needs at least one frame")
    valid = _valid_frame_mask(
        valid_frame_mask,
        batch=mask_probability.shape[0],
        frames=mask_probability.shape[1],
        device=mask_probability.device,
    )
    valid_float = valid.to(mask_probability.dtype)
    activity = mask_probability * valid_float
    previous = F.pad(activity[:, :-1], (1, 0))
    onset = F.relu(activity - previous) * valid_float
    # The tiny uniform fallback keeps the CDF finite at initialization without
    # breaking gradients through genuine onset mass.
    # ``finfo.eps`` is about 1e-3 in float16 and would contribute substantial
    # fake onset mass over a long clip.  1e-6 remains representable in the
    # mixed-precision dtypes used by the detector while staying negligible.
    eps = 1e-6
    onset = onset + eps * valid_float
    return onset / onset.sum(dim=1, keepdim=True).clamp_min(eps)


def onset_relation_mask(
    anchor_probability: torch.Tensor,
    relation_id: torch.Tensor,
    valid_frame_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return differentiable strict-before/strict-after masks and onset mass.

    For a one-hot onset at frame ``k``, ``before`` activates exactly frames
    ``t < k`` and ``after`` exactly frames ``t > k``.  The computation uses
    cumulative sums only and is consequently linear in ``T``.
    """

    if relation_id.ndim != 1 or relation_id.shape[0] != anchor_probability.shape[0]:
        raise ValueError("relation_id must have shape [B]")
    if relation_id.dtype == torch.bool or relation_id.is_floating_point():
        raise TypeError("relation_id must be an integer tensor")
    if bool(((relation_id < RELATION_BEFORE) | (relation_id > RELATION_AFTER)).any()):
        raise ValueError(f"relation_id must index {RELATION_NAMES}")

    valid = _valid_frame_mask(
        valid_frame_mask,
        batch=anchor_probability.shape[0],
        frames=anchor_probability.shape[1],
        device=anchor_probability.device,
    )
    onset = soft_onset_distribution(anchor_probability, valid)
    cdf_inclusive = onset.cumsum(dim=1)
    before = (1.0 - cdf_inclusive).clamp(0.0, 1.0)
    after = F.pad(cdf_inclusive[:, :-1], (1, 0)).clamp(0.0, 1.0)
    relation = torch.where(
        relation_id[:, None] == RELATION_BEFORE,
        before,
        after,
    )
    relation = relation * valid.to(relation.dtype)
    return relation, onset


class DenseTemporalReasoner(nn.Module):
    """Question-conditioned dense answer/NONE scorer and temporal grounder."""

    def __init__(self, config: DenseTemporalReasonerConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_dim

        self.frame_encoder = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.position_encoder = nn.Linear(4, hidden, bias=False)
        self.class_embeddings = nn.Embedding(config.num_classes, hidden)
        self.relation_embeddings = nn.Embedding(len(RELATION_NAMES), hidden)
        self.ordinal_embeddings = nn.Embedding(config.max_ordinal + 1, hidden)
        self.query_norm = nn.LayerNorm(hidden)
        self.frame_norm = nn.LayerNorm(hidden)

        self.anchor_detector_scale = nn.Parameter(
            torch.tensor(float(config.detector_logit_scale))
        )
        self.anchor_feature_scale = nn.Parameter(torch.tensor(1.0))
        self.ordinal_prior_scale = nn.Parameter(
            torch.tensor(float(config.ordinal_prior_scale))
        )
        self.frame_class_feature_scale = nn.Parameter(
            torch.tensor(float(config.feature_logit_scale))
        )
        self.answer_bias = nn.Parameter(torch.zeros(config.num_classes))
        self.none_scorer = nn.Sequential(
            nn.LayerNorm(2 * hidden + 3),
            nn.Linear(2 * hidden + 3, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        # At initialization C equally likely answer coordinates collectively
        # have roughly log(C) more mass than one NONE coordinate.
        nn.init.constant_(self.none_scorer[-1].bias, math.log(config.num_classes))
        self.answer_mask_residual = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _validate_inputs(
        self,
        features: torch.Tensor,
        detector_logits: torch.Tensor,
        anchor_label: torch.Tensor,
        relation_id: torch.Tensor,
        ordinal: torch.Tensor,
        gold_anchor_mask: torch.Tensor | None,
        gold_answer_label: torch.Tensor | None,
        teacher_forcing: float,
        valid_frame_mask: torch.Tensor | None,
    ) -> tuple[int, int]:
        if features.ndim != 3:
            raise ValueError("features must have shape [B, T, D]")
        batch, frames, width = features.shape
        if width != self.config.feature_dim:
            raise ValueError(
                f"feature width {width} != configured {self.config.feature_dim}"
            )
        if detector_logits.shape != (batch, frames, self.config.num_classes):
            raise ValueError(
                "detector_logits must have shape "
                f"[B, T, {self.config.num_classes}]"
            )
        for name, value in (
            ("anchor_label", anchor_label),
            ("relation_id", relation_id),
            ("ordinal", ordinal),
        ):
            if value.shape != (batch,):
                raise ValueError(f"{name} must have shape [B]")
            if value.dtype == torch.bool or value.is_floating_point():
                raise TypeError(f"{name} must be an integer tensor")
        if bool(((anchor_label < 0) | (anchor_label >= self.config.num_classes)).any()):
            raise ValueError("anchor_label is outside the class ontology")
        if bool(((relation_id < 0) | (relation_id >= len(RELATION_NAMES))).any()):
            raise ValueError(f"relation_id must index {RELATION_NAMES}")
        if bool(((ordinal < 1) | (ordinal > self.config.max_ordinal)).any()):
            raise ValueError("ordinal must be one-indexed and within max_ordinal")
        if gold_anchor_mask is not None and gold_anchor_mask.shape != (batch, frames):
            raise ValueError("gold_anchor_mask must have shape [B, T]")
        if gold_answer_label is not None and gold_answer_label.shape != (batch,):
            raise ValueError("gold_answer_label must have shape [B]")
        if valid_frame_mask is not None and valid_frame_mask.shape != (batch, frames):
            raise ValueError("valid_frame_mask must have shape [B, T]")
        if not isinstance(teacher_forcing, (float, int)) or not math.isfinite(
            float(teacher_forcing)
        ):
            raise TypeError("teacher_forcing must be a finite scalar")
        if not 0.0 <= float(teacher_forcing) <= 1.0:
            raise ValueError("teacher_forcing must be in [0, 1]")
        return batch, frames

    @staticmethod
    def _position_features(
        batch: int,
        frames: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.linspace(0.0, 1.0, frames, device=device, dtype=dtype)
        values = torch.stack(
            (
                position,
                position.square(),
                torch.sin(math.pi * position),
                torch.cos(math.pi * position),
            ),
            dim=-1,
        )
        return values.unsqueeze(0).expand(batch, -1, -1)

    def _ordinal_prior(
        self,
        anchor_detector_logits: torch.Tensor,
        ordinal: torch.Tensor,
        valid_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_frame_mask.to(anchor_detector_logits.dtype)
        activity = anchor_detector_logits.sigmoid() * valid
        previous = F.pad(activity[:, :-1], (1, 0))
        onset = F.relu(activity - previous) * valid
        # Normalizing by the strongest rise makes a confident occurrence count
        # approximately one while retaining weaker occurrences as soft counts.
        scale = onset.amax(dim=1, keepdim=True).clamp_min(1e-4)
        occurrence_position = (onset / scale).cumsum(dim=1)
        target = ordinal.to(activity.dtype).unsqueeze(1)
        prior = -0.5 * (
            (occurrence_position - target) / self.config.ordinal_sigma
        ).square()
        return prior * valid

    def _frame_class_logits(
        self,
        frame_hidden: torch.Tensor,
        detector_logits: torch.Tensor,
        valid_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        frame_vector = F.normalize(self.frame_norm(frame_hidden), dim=-1)
        class_vector = F.normalize(self.class_embeddings.weight, dim=-1)
        feature_logits = torch.einsum("bth,ch->btc", frame_vector, class_vector)
        feature_logits = feature_logits * math.sqrt(self.config.hidden_dim)
        logits = detector_logits + self.frame_class_feature_scale * feature_logits
        return logits.masked_fill(~valid_frame_mask.unsqueeze(-1), -30.0)

    def _pool_class_tracks(
        self,
        frame_class_logits: torch.Tensor,
        relation_mask: torch.Tensor,
        valid_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        temperature = self.config.pooling_temperature
        eps = 1e-6
        log_weight = relation_mask.clamp_min(eps).log().unsqueeze(-1)
        log_weight = log_weight.masked_fill(
            ~valid_frame_mask.unsqueeze(-1), -torch.inf
        )
        numerator = torch.logsumexp(
            frame_class_logits / temperature + log_weight,
            dim=1,
        )
        denominator = torch.logsumexp(log_weight, dim=1)
        pooled = temperature * (numerator - denominator)
        # When the anchor is at the relevant clip boundary, a strict before or
        # after region can be empty.  Without this support term, clamping zero
        # weights would accidentally turn the operation into global pooling.
        support = relation_mask.amax(dim=1).clamp_min(eps)
        return pooled + support.log().unsqueeze(1)

    def forward(
        self,
        features: torch.Tensor,
        detector_logits: torch.Tensor,
        anchor_label: torch.Tensor,
        relation_id: torch.Tensor,
        ordinal: torch.Tensor,
        gold_anchor_mask: torch.Tensor | None = None,
        gold_answer_label: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
        valid_frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run dense temporal reasoning.

        Args follow the public Q-DOR contract.  A gold answer label uses
        ``0..C-1`` for an answer and either ``-1`` or ``C`` for ``NONE``.
        Teacher forcing blends probabilities rather than making a discrete
        branch, so gradients remain available to the predicted path whenever
        its weight is non-zero.
        """

        batch, frames = self._validate_inputs(
            features,
            detector_logits,
            anchor_label,
            relation_id,
            ordinal,
            gold_anchor_mask,
            gold_answer_label,
            teacher_forcing,
            valid_frame_mask,
        )
        tf = float(teacher_forcing)
        valid = _valid_frame_mask(
            valid_frame_mask,
            batch=batch,
            frames=frames,
            device=features.device,
        )
        position = self._position_features(
            batch,
            frames,
            device=features.device,
            dtype=features.dtype,
        )
        frame_hidden = self.frame_encoder(features) + self.position_encoder(position)
        frame_hidden = self.frame_norm(frame_hidden)

        anchor_class = self.class_embeddings(anchor_label)
        query = self.query_norm(
            anchor_class
            + self.relation_embeddings(relation_id)
            + self.ordinal_embeddings(ordinal)
        )
        feature_pointer = torch.einsum("bth,bh->bt", frame_hidden, query)
        feature_pointer = feature_pointer / math.sqrt(self.config.hidden_dim)
        anchor_detector = detector_logits.gather(
            2,
            anchor_label[:, None, None].expand(-1, frames, 1),
        ).squeeze(-1)
        anchor_mask_logits = (
            self.anchor_detector_scale * anchor_detector
            + self.anchor_feature_scale * feature_pointer
            + self.ordinal_prior_scale
            * self._ordinal_prior(anchor_detector, ordinal, valid)
        )
        anchor_mask_logits = anchor_mask_logits.masked_fill(~valid, -30.0)
        predicted_anchor = anchor_mask_logits.sigmoid()
        anchor_for_relation = predicted_anchor
        if gold_anchor_mask is not None and tf > 0.0:
            gold_anchor = gold_anchor_mask.to(
                device=predicted_anchor.device,
                dtype=predicted_anchor.dtype,
            ).clamp(0.0, 1.0) * valid.to(predicted_anchor.dtype)
            anchor_for_relation = (1.0 - tf) * predicted_anchor + tf * gold_anchor

        relation_mask, anchor_onset = onset_relation_mask(
            anchor_for_relation,
            relation_id,
            valid,
        )
        frame_class_logits = self._frame_class_logits(
            frame_hidden, detector_logits, valid
        )
        answer_logits = self._pool_class_tracks(
            frame_class_logits, relation_mask, valid
        )
        answer_logits = answer_logits + self.answer_bias

        relation_weight = relation_mask / relation_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        relation_context = torch.einsum("bt,bth->bh", relation_weight, frame_hidden)
        availability = relation_mask.sum(dim=1) / valid.sum(dim=1).clamp_min(1).to(
            relation_mask.dtype
        )
        score_summary = torch.stack(
            (
                answer_logits.amax(dim=1),
                answer_logits.mean(dim=1),
                availability,
            ),
            dim=1,
        )
        none_input = torch.cat((query, relation_context, score_summary), dim=1)
        none_logit = self.none_scorer(none_input).squeeze(-1)
        choice_logits = torch.cat((answer_logits, none_logit[:, None]), dim=1)
        answerability_logit = torch.logsumexp(answer_logits, dim=1) - none_logit

        predicted_label_weight = answer_logits.softmax(dim=1)
        answerable_gate = answerability_logit.sigmoid()
        label_weight = predicted_label_weight
        if gold_answer_label is not None and tf > 0.0:
            gold = gold_answer_label.to(device=answer_logits.device, dtype=torch.long)
            gold_is_answerable = (gold >= 0) & (gold < self.config.num_classes)
            one_hot = F.one_hot(
                gold.clamp(0, self.config.num_classes - 1),
                num_classes=self.config.num_classes,
            ).to(answer_logits.dtype)
            one_hot = one_hot * gold_is_answerable[:, None].to(answer_logits.dtype)
            label_weight = (1.0 - tf) * predicted_label_weight + tf * one_hot
            answerable_gate = (
                (1.0 - tf) * answerable_gate
                + tf * gold_is_answerable.to(answer_logits.dtype)
            )

        selected_activity = torch.einsum(
            "bc,btc->bt",
            label_weight,
            frame_class_logits.sigmoid(),
        )
        base_answer_probability = (
            selected_activity * relation_mask * answerable_gate[:, None]
        )
        eps = 1e-6
        base_answer_logit = torch.logit(
            base_answer_probability.clamp(eps, 1.0 - eps)
        )
        answer_query = torch.einsum(
            "bc,ch->bh", label_weight, self.class_embeddings.weight
        )
        answer_residual_input = torch.cat(
            (frame_hidden, answer_query[:, None, :].expand(-1, frames, -1)),
            dim=-1,
        )
        answer_mask_logits = base_answer_logit + 0.1 * self.answer_mask_residual(
            answer_residual_input
        ).squeeze(-1)
        answer_mask_logits = answer_mask_logits.masked_fill(~valid, -30.0)

        return {
            "answer_logits": answer_logits,
            "answerability_logit": answerability_logit,
            "anchor_mask_logits": anchor_mask_logits,
            "answer_mask_logits": answer_mask_logits,
            "relation_mask": relation_mask,
            # Extra tensors make loss computation and scientific diagnostics
            # explicit while preserving the required public output fields.
            "choice_logits": choice_logits,
            "none_logit": none_logit,
            "anchor_onset_distribution": anchor_onset,
            "frame_class_logits": frame_class_logits,
            "valid_frame_mask": valid,
        }


def _balanced_mask_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-example foreground/background-balanced binary cross entropy."""

    error = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    valid = (
        torch.ones_like(target)
        if valid_frame_mask is None
        else valid_frame_mask.to(device=target.device, dtype=target.dtype)
    )
    positive = target * valid
    negative = (1.0 - target) * valid
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    positive_loss = (error * positive).sum(dim=1) / positive_count.clamp_min(1.0)
    negative_loss = (error * negative).sum(dim=1) / negative_count.clamp_min(1.0)
    has_positive = positive_count > 0
    has_negative = negative_count > 0
    combined = torch.where(
        has_positive & has_negative,
        0.5 * (positive_loss + negative_loss),
        torch.where(has_positive, positive_loss, negative_loss),
    )
    return combined.mean()


def _soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = (
        torch.ones_like(target)
        if valid_frame_mask is None
        else valid_frame_mask.to(device=target.device, dtype=target.dtype)
    )
    probability = logits.sigmoid() * valid
    target = target * valid
    intersection = (probability * target).sum(dim=1)
    denominator = probability.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def dense_temporal_reasoner_loss(
    outputs: Mapping[str, torch.Tensor],
    *,
    gold_answer_label: torch.Tensor,
    gold_anchor_mask: torch.Tensor,
    gold_answer_mask: torch.Tensor,
    weights: DenseTemporalLossWeights | None = None,
    valid_frame_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute answer/NONE, anchor, answer-mask, and relation losses.

    Gold answer labels use ``0..C-1`` for answerable examples.  ``-1`` or
    ``C`` denotes ``NONE``.  Empty answer masks are supervised normally, so
    no-evidence examples explicitly teach the temporal head to remain silent.
    """

    weights = weights or DenseTemporalLossWeights()
    required = {
        "answer_logits",
        "answerability_logit",
        "anchor_mask_logits",
        "answer_mask_logits",
        "relation_mask",
        "choice_logits",
    }
    missing = required - set(outputs)
    if missing:
        raise KeyError(f"reasoner outputs are missing {sorted(missing)}")
    answer_logits = outputs["answer_logits"]
    batch, classes = answer_logits.shape
    if gold_answer_label.shape != (batch,):
        raise ValueError("gold_answer_label must have shape [B]")
    if gold_anchor_mask.shape != outputs["anchor_mask_logits"].shape:
        raise ValueError("gold_anchor_mask shape does not match anchor logits")
    if gold_answer_mask.shape != outputs["answer_mask_logits"].shape:
        raise ValueError("gold_answer_mask shape does not match answer logits")

    gold = gold_answer_label.to(device=answer_logits.device, dtype=torch.long)
    answerable = (gold >= 0) & (gold < classes)
    invalid = (gold < -1) | (gold > classes)
    if bool(invalid.any()):
        raise ValueError("gold labels must be 0..C-1, -1, or C for NONE")
    choice_target = torch.where(answerable, gold, torch.full_like(gold, classes))
    anchor_target = gold_anchor_mask.to(
        device=answer_logits.device, dtype=answer_logits.dtype
    ).clamp(0.0, 1.0)
    answer_target = gold_answer_mask.to(
        device=answer_logits.device, dtype=answer_logits.dtype
    ).clamp(0.0, 1.0)
    if valid_frame_mask is None:
        valid_frame_mask = outputs.get("valid_frame_mask")
    valid = _valid_frame_mask(
        valid_frame_mask,
        batch=batch,
        frames=outputs["anchor_mask_logits"].shape[1],
        device=answer_logits.device,
    )

    components = {
        "answer_choice": F.cross_entropy(outputs["choice_logits"], choice_target),
        "answerability": F.binary_cross_entropy_with_logits(
            outputs["answerability_logit"], answerable.to(answer_logits.dtype)
        ),
        "anchor_bce": _balanced_mask_bce(
            outputs["anchor_mask_logits"], anchor_target, valid
        ),
        "anchor_dice": _soft_dice_loss(
            outputs["anchor_mask_logits"], anchor_target, valid
        ),
        "answer_bce": _balanced_mask_bce(
            outputs["answer_mask_logits"], answer_target, valid
        ),
        "answer_dice": _soft_dice_loss(
            outputs["answer_mask_logits"], answer_target, valid
        ),
    }
    answer_probability = outputs["answer_mask_logits"].sigmoid()
    valid_float = valid.to(answer_probability.dtype)
    outside_relation = (
        1.0 - outputs["relation_mask"].clamp(0.0, 1.0)
    ) * valid_float
    components["relation_consistency"] = (
        answer_probability * outside_relation
    ).sum(dim=1).div(
        (answer_probability * valid_float).sum(dim=1).clamp_min(1e-6)
    ).mean()

    total = answer_logits.new_zeros(())
    for name, component in components.items():
        total = total + getattr(weights, name) * component
    return {"loss": total, **components}


@torch.no_grad()
def decode_dense_temporal_outputs(
    outputs: Mapping[str, torch.Tensor],
    *,
    answerability_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    valid_frame_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Decode labels, NONE decisions, and audible intervals on the 40 ms grid."""

    if not 0.0 <= answerability_threshold <= 1.0:
        raise ValueError("answerability_threshold must be in [0, 1]")
    answer_logits = outputs["answer_logits"]
    answerability_probability = outputs["answerability_logit"].sigmoid()
    answerable = answerability_probability >= answerability_threshold
    answer_label = answer_logits.argmax(dim=1)
    answer_label = torch.where(answerable, answer_label, -torch.ones_like(answer_label))
    if valid_frame_mask is None:
        valid_frame_mask = outputs.get("valid_frame_mask")
    valid = _valid_frame_mask(
        valid_frame_mask,
        batch=answer_logits.shape[0],
        frames=outputs["anchor_mask_logits"].shape[1],
        device=answer_logits.device,
    )
    valid_float = valid.to(outputs["anchor_mask_logits"].dtype)
    anchor_probability = outputs["anchor_mask_logits"].sigmoid() * valid_float
    answer_probability = outputs["answer_mask_logits"].sigmoid() * valid_float
    return {
        "answer_label": answer_label,
        "answerable": answerable,
        "answerability_probability": answerability_probability,
        "anchor_probability": anchor_probability,
        "answer_probability": answer_probability,
        "anchor_intervals": masks_to_40ms_intervals(
            anchor_probability,
            threshold=mask_threshold,
            valid_frame_mask=valid,
        ),
        "answer_intervals": masks_to_40ms_intervals(
            answer_probability,
            threshold=mask_threshold,
            valid_frame_mask=valid,
        ),
    }


__all__ = [
    "FRAME_HOP_SECONDS",
    "RELATION_AFTER",
    "RELATION_BEFORE",
    "RELATION_NAMES",
    "DenseTemporalLossWeights",
    "DenseTemporalReasoner",
    "DenseTemporalReasonerConfig",
    "decode_dense_temporal_outputs",
    "dense_temporal_reasoner_loss",
    "intervals_to_40ms_mask",
    "masks_to_40ms_intervals",
    "onset_relation_mask",
    "soft_onset_distribution",
]
