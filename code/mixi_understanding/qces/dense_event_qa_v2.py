"""Q-DOR-v2: dense nearest-onset reasoning with faithful occurrence masks.

Version 2 fixes three structural failure modes of the first dense reasoner:

* anchor localization depends on class and ordinal, never on relation;
* ``immediately before/after`` is a differentiable first-passage operator over
  class-onset hazards instead of global pooling over half of the recording;
* answer evidence is conditioned on one hard top-1 class (straight-through in
  training), localized to one occurrence, and never multiplied by the NONE
  verifier probability.

All operators remain O(T*C); no proposal list or T-by-T matrix is constructed.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from mixi_understanding.qces.dense_event_qa import (
    FRAME_HOP_SECONDS,
    RELATION_AFTER,
    RELATION_BEFORE,
    RELATION_NAMES,
    _valid_frame_mask,
    intervals_to_40ms_mask,
    masks_to_40ms_intervals,
)


@dataclass(frozen=True)
class DenseTemporalReasonerV2Config:
    feature_dim: int
    num_classes: int = 200
    hidden_dim: int = 256
    max_ordinal: int = 10
    dropout: float = 0.1
    ordinal_sigma: float = 0.65
    detector_logit_scale: float = 1.0
    feature_logit_scale: float = 0.1
    ordinal_prior_scale: float = 1.0
    onset_feature_scale: float = 0.1
    frame_hop_seconds: float = FRAME_HOP_SECONDS

    def __post_init__(self) -> None:
        for name in ("feature_dim", "num_classes", "hidden_dim", "max_ordinal"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "ordinal_sigma",
            "detector_logit_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if (
            self.feature_logit_scale < 0
            or self.ordinal_prior_scale < 0
            or self.onset_feature_scale < 0
        ):
            raise ValueError("feature/ordinal/onset scales must be non-negative")
        if not math.isclose(self.frame_hop_seconds, FRAME_HOP_SECONDS, abs_tol=1e-12):
            raise ValueError("Q-DOR-v2 requires the fixed 40 ms grid")


@dataclass(frozen=True)
class DenseTemporalReasonerV2LossWeights:
    answer_label: float = 1.0
    answerability: float = 0.75
    anchor_bce: float = 1.0
    anchor_dice: float = 1.0
    anchor_onset: float = 0.5
    answer_bce: float = 1.0
    answer_dice: float = 1.0
    answer_onset: float = 0.5
    class_onset_bce: float = 0.75
    class_onset_dice: float = 0.5
    union_onset_bce: float = 0.5
    union_onset_dice: float = 0.25
    relation_consistency: float = 0.2

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"loss weight {name} must be finite and non-negative")


@dataclass(frozen=True)
class DenseTemporalThresholdsV2:
    answerability: float = 0.5
    anchor_mask: float = 0.5
    answer_mask: float = 0.5
    verification: float = 0.5

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not 0 <= value <= 1:
                raise ValueError(f"threshold {name} must be in [0, 1]")


def _strict_relation_ids(relation_id: torch.Tensor, batch: int) -> None:
    if relation_id.shape != (batch,):
        raise ValueError("relation_id must have shape [B]")
    if relation_id.dtype == torch.bool or relation_id.is_floating_point():
        raise TypeError("relation_id must be an integer tensor")
    if bool(((relation_id < RELATION_BEFORE) | (relation_id > RELATION_AFTER)).any()):
        raise ValueError(f"relation_id must index {RELATION_NAMES}")


def asymmetric_anchor_relation_mask(
    anchor_probability: torch.Tensor,
    relation_id: torch.Tensor,
    valid_frame_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return relation mask plus anchor onset/offset distributions.

    ``before`` is strictly before anchor onset.  ``after`` begins at the first
    inactive frame after anchor offset.  Thus an anchor active on inclusive
    frame indices 3..6 permits before frames 0..2 and after frames 7 onward;
    no candidate onset inside the anchor can win first passage.
    """

    if anchor_probability.ndim != 2:
        raise ValueError("anchor_probability must have shape [B,T]")
    batch, frames = anchor_probability.shape
    _strict_relation_ids(relation_id, batch)
    valid = _valid_frame_mask(
        valid_frame_mask,
        batch=batch,
        frames=frames,
        device=anchor_probability.device,
    )
    probability = anchor_probability.clamp(0.0, 1.0) * valid.to(
        anchor_probability.dtype
    )
    previous = F.pad(probability[:, :-1], (1, 0))
    rising = F.relu(probability - previous) * valid.to(probability.dtype)
    rising_total = rising.sum(dim=1, keepdim=True)
    uniform_fallback = valid.to(probability.dtype) / valid.sum(
        dim=1, keepdim=True
    ).clamp_min(1)
    onset = torch.where(
        rising_total > 1e-6,
        rising / rising_total.clamp_min(1e-6),
        uniform_fallback,
    )
    falling = F.relu(previous - probability) * valid.to(probability.dtype)
    offset = falling / falling.sum(dim=1, keepdim=True).clamp_min(1e-6)
    onset_cdf = onset.cumsum(dim=1)
    before = (1.0 - onset_cdf).clamp(0.0, 1.0)
    # A fall at frame k means frame k is the first frame after the anchor.
    after = offset.cumsum(dim=1).clamp(0.0, 1.0)
    relation = torch.where(
        relation_id[:, None] == RELATION_BEFORE,
        before,
        after,
    )
    relation = relation * valid.to(relation.dtype)
    return relation, onset, offset


def first_passage_nearest_onset(
    event_onset_probability: torch.Tensor,
    anchor_probability: torch.Tensor,
    relation_id: torch.Tensor,
    valid_frame_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return relation geometry, nearest-event mass, and no-event mass.

    For ``after`` the hazard is scanned from left to right; for ``before`` it
    is scanned from right to left.  At a candidate frame ``t`` the first-
    passage mass is ``h[t] * product(1-h[u])`` over every closer candidate.
    Consequently a plausible near event suppresses a stronger far event.
    """

    if event_onset_probability.ndim != 2:
        raise ValueError("event_onset_probability must have shape [B,T]")
    if anchor_probability.shape != event_onset_probability.shape:
        raise ValueError("anchor/event-onset probabilities must share [B,T]")
    batch, frames = event_onset_probability.shape
    _strict_relation_ids(relation_id, batch)
    valid = _valid_frame_mask(
        valid_frame_mask,
        batch=batch,
        frames=frames,
        device=event_onset_probability.device,
    )
    if not bool(torch.isfinite(event_onset_probability).all()):
        raise ValueError("event onset probability must be finite")
    if bool(((event_onset_probability < 0) | (event_onset_probability > 1)).any()):
        raise ValueError("event onset probability must lie in [0,1]")
    relation_mask, _, _ = asymmetric_anchor_relation_mask(
        anchor_probability, relation_id, valid
    )
    hazard = (
        event_onset_probability
        * relation_mask
        * valid.to(event_onset_probability.dtype)
    ).clamp(0.0, 1.0 - 1e-6)

    def scan(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        survival_factor = (1.0 - values).clamp_min(1e-6)
        survival_before = F.pad(
            torch.cumprod(survival_factor[:, :-1], dim=1), (1, 0), value=1.0
        )
        mass = values * survival_before
        no_event = torch.prod(survival_factor, dim=1)
        return mass, no_event

    after_mass, after_none = scan(hazard)
    before_reversed, before_none = scan(torch.flip(hazard, dims=(1,)))
    before_mass = torch.flip(before_reversed, dims=(1,))
    is_before = relation_id[:, None] == RELATION_BEFORE
    nearest = torch.where(is_before, before_mass, after_mass)
    no_event = torch.where(relation_id == RELATION_BEFORE, before_none, after_none)
    nearest = nearest * valid.to(nearest.dtype)
    return relation_mask, nearest, no_event


def localized_occurrence_state(
    selected_activity_probability: torch.Tensor,
    selected_class_onset_probability: torch.Tensor,
    selected_answer_onset_distribution: torch.Tensor,
    valid_frame_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only the occurrence initiated by the selected answer onset.

    The O(T) recurrent state starts at selected onset mass and is reset by a
    later onset of the same class.  Multiplication by class activity removes
    silence between onset and offset without opening a later occurrence.
    """

    if selected_activity_probability.ndim != 2:
        raise ValueError("selected activity must have shape [B,T]")
    expected = selected_activity_probability.shape
    if (
        selected_class_onset_probability.shape != expected
        or selected_answer_onset_distribution.shape != expected
    ):
        raise ValueError("occurrence tensors must share [B,T]")
    batch, frames = expected
    valid = _valid_frame_mask(
        valid_frame_mask,
        batch=batch,
        frames=frames,
        device=selected_activity_probability.device,
    )
    state = selected_activity_probability.new_zeros(batch)
    states: list[torch.Tensor] = []
    for frame in range(frames):
        reset = selected_class_onset_probability[:, frame].clamp(0.0, 1.0)
        start = selected_answer_onset_distribution[:, frame].clamp(0.0, 1.0)
        state = (state * (1.0 - reset) + start).clamp(0.0, 1.0)
        state = state * valid[:, frame].to(state.dtype)
        states.append(state)
    occurrence_state = torch.stack(states, dim=1)
    probability = (
        selected_activity_probability.clamp(0.0, 1.0)
        * occurrence_state
        * valid.to(selected_activity_probability.dtype)
    )
    return probability, occurrence_state


def _hard_top1_straight_through(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    soft = logits.softmax(dim=1)
    index = logits.argmax(dim=1)
    hard = F.one_hot(index, num_classes=logits.shape[1]).to(logits.dtype)
    return hard + soft - soft.detach(), index


class DenseTemporalReasonerV2(nn.Module):
    def __init__(self, config: DenseTemporalReasonerV2Config) -> None:
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
        self.anchor_query_norm = nn.LayerNorm(hidden)
        self.reasoning_query_norm = nn.LayerNorm(hidden)
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
        # A class/frame-specific onset residual can correct false detector
        # derivatives.  A scalar temperature+bias cannot.  The residual sees
        # both current semantics and the local temporal change.
        self.onset_frame_encoder = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )
        self.onset_frame_norm = nn.LayerNorm(hidden)
        self.onset_class_embeddings = nn.Embedding(config.num_classes, hidden)
        self.onset_feature_scale = nn.Parameter(
            torch.tensor(float(config.onset_feature_scale))
        )
        self.onset_class_bias = nn.Parameter(torch.zeros(config.num_classes))
        self.answer_bias = nn.Parameter(torch.zeros(config.num_classes))
        self.answerability_scorer = nn.Sequential(
            nn.LayerNorm(2 * hidden + 4),
            nn.Linear(2 * hidden + 4, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, 1),
        )
        self.answer_mask_residual = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _position_features(
        batch: int, frames: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        position = torch.linspace(0.0, 1.0, frames, device=device, dtype=dtype)
        values = torch.stack(
            (position, position.square(), torch.sin(math.pi * position), torch.cos(math.pi * position)),
            dim=-1,
        )
        return values.unsqueeze(0).expand(batch, -1, -1)

    def _validate(
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
    ) -> tuple[int, int, torch.Tensor]:
        if features.ndim != 3 or features.shape[2] != self.config.feature_dim:
            raise ValueError("features must have shape [B,T,feature_dim]")
        batch, frames, _ = features.shape
        if detector_logits.shape != (batch, frames, self.config.num_classes):
            raise ValueError("detector_logits must have shape [B,T,C]")
        for name, tensor in (("anchor_label", anchor_label), ("ordinal", ordinal)):
            if tensor.shape != (batch,) or tensor.dtype == torch.bool or tensor.is_floating_point():
                raise TypeError(f"{name} must be an integer [B] tensor")
        _strict_relation_ids(relation_id, batch)
        if bool(((anchor_label < 0) | (anchor_label >= self.config.num_classes)).any()):
            raise ValueError("anchor_label is outside the ontology")
        if bool(((ordinal < 1) | (ordinal > self.config.max_ordinal)).any()):
            raise ValueError("ordinal must be one-indexed and cannot be silently clamped")
        if gold_anchor_mask is not None and gold_anchor_mask.shape != (batch, frames):
            raise ValueError("gold_anchor_mask must have shape [B,T]")
        if gold_answer_label is not None and gold_answer_label.shape != (batch,):
            raise ValueError("gold_answer_label must have shape [B]")
        if not math.isfinite(float(teacher_forcing)) or not 0 <= teacher_forcing <= 1:
            raise ValueError("teacher_forcing must lie in [0,1]")
        valid = _valid_frame_mask(
            valid_frame_mask, batch=batch, frames=frames, device=features.device
        )
        return batch, frames, valid

    def _ordinal_prior(
        self,
        anchor_detector_logits: torch.Tensor,
        ordinal: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        activity = anchor_detector_logits.sigmoid() * valid.to(anchor_detector_logits.dtype)
        previous = F.pad(activity[:, :-1], (1, 0))
        rise = F.relu(activity - previous) * valid.to(activity.dtype)
        scale = rise.amax(dim=1, keepdim=True).clamp_min(1e-4)
        position = (rise / scale).cumsum(dim=1)
        target = ordinal.to(activity.dtype).unsqueeze(1)
        return -0.5 * ((position - target) / self.config.ordinal_sigma).square()

    def _frame_class_logits(
        self,
        frame_hidden: torch.Tensor,
        detector_logits: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        frame_vector = F.normalize(self.frame_norm(frame_hidden), dim=-1)
        class_vector = F.normalize(self.class_embeddings.weight, dim=-1)
        feature_logits = torch.einsum("bth,ch->btc", frame_vector, class_vector)
        feature_logits = feature_logits * math.sqrt(self.config.hidden_dim)
        logits = detector_logits + self.frame_class_feature_scale * feature_logits
        return logits.masked_fill(~valid.unsqueeze(-1), -30.0)

    def forward(
        self,
        features: torch.Tensor,
        detector_logits: torch.Tensor,
        anchor_label: torch.Tensor,
        relation_id: torch.Tensor,
        ordinal: torch.Tensor,
        *,
        gold_anchor_mask: torch.Tensor | None = None,
        gold_answer_label: torch.Tensor | None = None,
        teacher_forcing: float = 0.0,
        valid_frame_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, frames, valid = self._validate(
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
        use_gold_context = torch.zeros(batch, dtype=torch.bool, device=features.device)
        if tf > 0:
            use_gold_context = torch.rand(batch, device=features.device) < tf
        position = self._position_features(
            batch, frames, device=features.device, dtype=features.dtype
        )
        frame_hidden = self.frame_norm(
            self.frame_encoder(features) + self.position_encoder(position)
        )

        # Relation is deliberately absent from the anchor query.
        anchor_query = self.anchor_query_norm(
            self.class_embeddings(anchor_label) + self.ordinal_embeddings(ordinal)
        )
        reasoning_query = self.reasoning_query_norm(
            anchor_query + self.relation_embeddings(relation_id)
        )
        pointer = torch.einsum("bth,bh->bt", frame_hidden, anchor_query)
        pointer = pointer / math.sqrt(self.config.hidden_dim)
        anchor_detector = detector_logits.gather(
            2, anchor_label[:, None, None].expand(-1, frames, 1)
        ).squeeze(-1)
        anchor_mask_logits = (
            self.anchor_detector_scale * anchor_detector
            + self.anchor_feature_scale * pointer
            + self.ordinal_prior_scale
            * self._ordinal_prior(anchor_detector, ordinal, valid)
        ).masked_fill(~valid, -30.0)
        predicted_anchor = anchor_mask_logits.sigmoid()
        anchor_for_relation = predicted_anchor
        if gold_anchor_mask is not None and tf > 0:
            gold_anchor = (
                gold_anchor_mask.to(predicted_anchor.dtype).clamp(0.0, 1.0)
                * valid.to(predicted_anchor.dtype)
            )
            anchor_for_relation = torch.where(
                use_gold_context[:, None], gold_anchor, predicted_anchor
            )

        frame_class_logits = self._frame_class_logits(
            frame_hidden, detector_logits, valid
        )
        class_activity = frame_class_logits.sigmoid() * valid.unsqueeze(-1).to(
            frame_class_logits.dtype
        )
        previous_activity = F.pad(class_activity[:, :-1, :], (0, 0, 1, 0))
        raw_rise = F.relu(class_activity - previous_activity)
        raw_rise_logit = torch.logit(raw_rise.clamp(1e-6, 1.0 - 1e-6))
        previous_hidden = F.pad(frame_hidden[:, :-1, :], (0, 0, 1, 0))
        onset_hidden = self.onset_frame_encoder(
            torch.cat((frame_hidden, frame_hidden - previous_hidden), dim=-1)
        )
        onset_frame_vector = F.normalize(self.onset_frame_norm(onset_hidden), dim=-1)
        onset_class_vector = F.normalize(self.onset_class_embeddings.weight, dim=-1)
        onset_feature_logits = torch.einsum(
            "bth,ch->btc", onset_frame_vector, onset_class_vector
        ) * math.sqrt(self.config.hidden_dim)
        class_onset_logits = (
            raw_rise_logit
            + self.onset_feature_scale * onset_feature_logits
            + self.onset_class_bias[None, None, :]
        )
        class_onset_logits = class_onset_logits.masked_fill(~valid.unsqueeze(-1), -30.0)
        class_onset_probability = class_onset_logits.sigmoid()
        event_onset_probability = class_onset_probability.amax(dim=2)

        relation_mask, nearest_mass, no_event_probability = first_passage_nearest_onset(
            event_onset_probability,
            anchor_for_relation,
            relation_id,
            valid,
        )
        class_share = class_onset_probability / class_onset_probability.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)
        answer_mass = torch.einsum("bt,btc->bc", nearest_mass, class_share)
        answer_logits = (answer_mass + 1e-8).log() + self.answer_bias

        relation_weight = relation_mask / relation_mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        relation_context = torch.einsum("bt,bth->bh", relation_weight, frame_hidden)
        summary = torch.stack(
            (
                answer_logits.amax(dim=1),
                answer_logits.mean(dim=1),
                nearest_mass.sum(dim=1),
                no_event_probability,
            ),
            dim=1,
        )
        answerability_logit = self.answerability_scorer(
            torch.cat((reasoning_query, relation_context, summary), dim=1)
        ).squeeze(-1)

        label_weight, predicted_answer_label = _hard_top1_straight_through(answer_logits)
        if gold_answer_label is not None and tf > 0:
            gold = gold_answer_label.to(answer_logits.device, dtype=torch.long)
            positive = (gold >= 0) & (gold < self.config.num_classes)
            gold_one_hot = F.one_hot(
                gold.clamp(0, self.config.num_classes - 1),
                num_classes=self.config.num_classes,
            ).to(answer_logits.dtype)
            # Scheduled teacher forcing chooses one hard condition per row;
            # it never interpolates two class masks.  This preserves the
            # one-occurrence evidence contract at every training stage.
            use_gold = use_gold_context & positive
            # NONE rows retain predicted hard conditioning; their answer mask
            # is ignored by decoding and excluded from the positive mask loss.
            label_weight = torch.where(use_gold[:, None], gold_one_hot, label_weight)

        selected_activity = torch.einsum("bc,btc->bt", label_weight, class_activity)
        selected_class_onset = torch.einsum(
            "bc,btc->bt", label_weight, class_onset_probability
        )
        selected_onset_mass = nearest_mass * selected_class_onset
        selected_answer_onset = selected_onset_mass / selected_onset_mass.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        occurrence_probability, occurrence_state = localized_occurrence_state(
            selected_activity,
            selected_class_onset,
            selected_answer_onset,
            valid,
        )
        base_answer_probability = occurrence_probability * relation_mask
        base_logit = torch.logit(base_answer_probability.clamp(1e-6, 1.0 - 1e-6))
        answer_query = torch.einsum(
            "bc,ch->bh", label_weight, self.class_embeddings.weight
        )
        residual_input = torch.cat(
            (frame_hidden, answer_query[:, None, :].expand(-1, frames, -1)), dim=-1
        )
        answer_mask_logits = base_logit + 0.1 * self.answer_mask_residual(
            residual_input
        ).squeeze(-1)
        answer_mask_logits = answer_mask_logits.masked_fill(~valid, -30.0)

        return {
            "answer_logits": answer_logits,
            "answerability_logit": answerability_logit,
            "anchor_mask_logits": anchor_mask_logits,
            "anchor_condition_probability": anchor_for_relation,
            "answer_mask_logits": answer_mask_logits,
            "relation_mask": relation_mask,
            "nearest_event_onset_mass": nearest_mass,
            "no_event_probability": no_event_probability,
            "class_onset_logits": class_onset_logits,
            "class_onset_probability": class_onset_probability,
            "union_onset_probability": event_onset_probability,
            "answer_onset_distribution": selected_answer_onset,
            "anchor_onset_distribution": asymmetric_anchor_relation_mask(
                anchor_for_relation, relation_id, valid
            )[1],
            "occurrence_state": occurrence_state,
            "hard_answer_label": predicted_answer_label,
            "label_condition_weight": label_weight,
            "teacher_forcing_gold_context": use_gold_context,
            "frame_class_logits": frame_class_logits,
            "valid_frame_mask": valid,
        }


def _balanced_bce_logits(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    error = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    valid_float = valid.to(error.dtype)
    while valid_float.ndim < error.ndim:
        valid_float = valid_float.unsqueeze(-1)
    positive = target * valid_float
    negative = (1.0 - target) * valid_float
    reduce_dims = tuple(range(1, error.ndim))
    pos_count = positive.sum(dim=reduce_dims)
    neg_count = negative.sum(dim=reduce_dims)
    pos_loss = (error * positive).sum(dim=reduce_dims) / pos_count.clamp_min(1.0)
    neg_loss = (error * negative).sum(dim=reduce_dims) / neg_count.clamp_min(1.0)
    return torch.where(
        (pos_count > 0) & (neg_count > 0),
        0.5 * (pos_loss + neg_loss),
        torch.where(pos_count > 0, pos_loss, neg_loss),
    ).mean()


def _balanced_bce_probability(
    probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    logits = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))
    return _balanced_bce_logits(logits, target, valid)


def _dice_probability(
    probability: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    valid_float = valid.to(probability.dtype)
    while valid_float.ndim < probability.ndim:
        valid_float = valid_float.unsqueeze(-1)
    probability = probability * valid_float
    target = target * valid_float
    reduce_dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=reduce_dims)
    denominator = probability.sum(dim=reduce_dims) + target.sum(dim=reduce_dims)
    return (1.0 - (2 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def _onset_nll(distribution: torch.Tensor, gold_mask: torch.Tensor) -> torch.Tensor:
    previous = F.pad(gold_mask[:, :-1], (1, 0))
    gold_onset = F.relu(gold_mask - previous)
    gold_onset = gold_onset / gold_onset.sum(dim=1, keepdim=True).clamp_min(1.0)
    return -(gold_onset * distribution.clamp_min(1e-8).log()).sum(dim=1).mean()


def _balanced_answerability_bce(logit: torch.Tensor, answerable: torch.Tensor) -> torch.Tensor:
    error = F.binary_cross_entropy_with_logits(
        logit, answerable.to(logit.dtype), reduction="none"
    )
    positive, negative = answerable, ~answerable
    if bool(positive.any()) and bool(negative.any()):
        return 0.5 * (error[positive].mean() + error[negative].mean())
    return error.mean()


def dense_temporal_reasoner_v2_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    *,
    log_prior: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    logit_adjustment_tau: float = 0.0,
    weights: DenseTemporalReasonerV2LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    """Directly supervise answer, evidence, and the full scene onset grid."""

    weights = weights or DenseTemporalReasonerV2LossWeights()
    gold_label = batch["gold_answer_label"].long()
    answerable = gold_label >= 0
    answer_logits = outputs["answer_logits"]
    if bool(answerable.any()):
        adjusted = answer_logits[answerable]
        if log_prior is not None:
            adjusted = adjusted + logit_adjustment_tau * log_prior[None, :]
        per_row = F.cross_entropy(adjusted, gold_label[answerable], reduction="none")
        if class_weights is not None:
            row_weight = class_weights[gold_label[answerable]]
            answer_label_loss = (per_row * row_weight).sum() / row_weight.sum().clamp_min(1e-6)
        else:
            answer_label_loss = per_row.mean()
    else:
        answer_label_loss = answer_logits.sum() * 0.0
    valid = batch["valid_mask"].bool()
    anchor_target = batch["gold_anchor_mask"].to(answer_logits.dtype)
    answer_target = batch["gold_answer_mask"].to(answer_logits.dtype)
    class_onset_target = batch["gold_class_onset_mask"].to(answer_logits.dtype)
    union_onset_target = batch["gold_union_onset_mask"].to(answer_logits.dtype)

    components: dict[str, torch.Tensor] = {
        "answer_label": answer_label_loss,
        "answerability": _balanced_answerability_bce(
            outputs["answerability_logit"], answerable
        ),
        "anchor_bce": _balanced_bce_logits(
            outputs["anchor_mask_logits"], anchor_target, valid
        ),
        "anchor_dice": _dice_probability(
            outputs["anchor_mask_logits"].sigmoid(), anchor_target, valid
        ),
        "anchor_onset": _onset_nll(
            outputs["anchor_onset_distribution"], anchor_target
        ),
        "class_onset_bce": _balanced_bce_probability(
            outputs["class_onset_probability"], class_onset_target, valid
        ),
        "class_onset_dice": _dice_probability(
            outputs["class_onset_probability"], class_onset_target, valid
        ),
        "union_onset_bce": _balanced_bce_probability(
            outputs["union_onset_probability"], union_onset_target, valid
        ),
        "union_onset_dice": _dice_probability(
            outputs["union_onset_probability"], union_onset_target, valid
        ),
    }
    if bool(answerable.any()):
        positive_valid = valid[answerable]
        components["answer_bce"] = _balanced_bce_logits(
            outputs["answer_mask_logits"][answerable],
            answer_target[answerable],
            positive_valid,
        )
        components["answer_dice"] = _dice_probability(
            outputs["answer_mask_logits"][answerable].sigmoid(),
            answer_target[answerable],
            positive_valid,
        )
        components["answer_onset"] = _onset_nll(
            outputs["answer_onset_distribution"][answerable],
            answer_target[answerable],
        )
        probability = outputs["answer_mask_logits"][answerable].sigmoid()
        outside = (1.0 - outputs["relation_mask"][answerable]).clamp(0.0, 1.0)
        components["relation_consistency"] = (
            probability * outside * positive_valid.to(probability.dtype)
        ).sum(dim=1).div(
            (probability * positive_valid).sum(dim=1).clamp_min(1e-6)
        ).mean()
    else:
        zero = outputs["answer_mask_logits"].sum() * 0.0
        components.update(
            {
                "answer_bce": zero,
                "answer_dice": zero,
                "answer_onset": zero,
                "relation_consistency": zero,
            }
        )
    total = answer_logits.new_zeros(())
    for name, value in components.items():
        total = total + getattr(weights, name) * value
    return {"loss": total, **components}


@torch.no_grad()
def decode_dense_temporal_outputs_v2(
    outputs: Mapping[str, torch.Tensor],
    *,
    thresholds: DenseTemporalThresholdsV2 | None = None,
    valid_frame_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or DenseTemporalThresholdsV2()
    answer_logits = outputs["answer_logits"]
    valid = _valid_frame_mask(
        valid_frame_mask if valid_frame_mask is not None else outputs.get("valid_frame_mask"),
        batch=answer_logits.shape[0],
        frames=outputs["anchor_mask_logits"].shape[1],
        device=answer_logits.device,
    )
    answerability_probability = outputs["answerability_logit"].sigmoid()
    answerable = answerability_probability >= thresholds.answerability
    raw_label = answer_logits.argmax(dim=1)
    answer_label = torch.where(answerable, raw_label, -torch.ones_like(raw_label))
    anchor_probability = outputs["anchor_mask_logits"].sigmoid() * valid
    raw_answer_probability = outputs["answer_mask_logits"].sigmoid() * valid
    verification_probability = outputs["relation_mask"] * valid
    anchor_mask = anchor_probability >= thresholds.anchor_mask
    raw_answer_mask = raw_answer_probability >= thresholds.answer_mask
    verification_mask = verification_probability >= thresholds.verification
    answer_mask = raw_answer_mask & answerable[:, None]
    evidence_mask = anchor_mask | torch.where(
        answerable[:, None], raw_answer_mask, verification_mask
    )
    return {
        "answer_label": answer_label,
        "raw_answer_label": raw_label,
        "answerable": answerable,
        "answerability_probability": answerability_probability,
        "anchor_probability": anchor_probability,
        "raw_answer_probability": raw_answer_probability,
        "verification_probability": verification_probability,
        "anchor_mask": anchor_mask,
        "answer_mask": answer_mask,
        "verification_mask": verification_mask & (~answerable[:, None]),
        "evidence_mask": evidence_mask,
        "anchor_intervals": masks_to_40ms_intervals(
            anchor_mask.to(torch.float32), valid_frame_mask=valid
        ),
        "answer_intervals": masks_to_40ms_intervals(
            answer_mask.to(torch.float32), valid_frame_mask=valid
        ),
        "verification_intervals": masks_to_40ms_intervals(
            (verification_mask & (~answerable[:, None])).to(torch.float32),
            valid_frame_mask=valid,
        ),
        "evidence_intervals": masks_to_40ms_intervals(
            evidence_mask.to(torch.float32), valid_frame_mask=valid
        ),
    }


__all__ = [
    "asymmetric_anchor_relation_mask",
    "DenseTemporalReasonerV2",
    "DenseTemporalReasonerV2Config",
    "DenseTemporalReasonerV2LossWeights",
    "DenseTemporalThresholdsV2",
    "decode_dense_temporal_outputs_v2",
    "dense_temporal_reasoner_v2_loss",
    "first_passage_nearest_onset",
    "intervals_to_40ms_mask",
    "localized_occurrence_state",
]
