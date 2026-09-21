"""Training objectives for faithful question-conditioned evidence separation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mixi_understanding.qces.counterfactual import (
    COUNTERFACTUAL_LOSS_NAMES,
    FAMILY_LOSS_NAMES,
    QUESTION_LOSS_NAMES,
    SURFACE_LOSS_NAMES,
    counterfactual_objectives,
)
from mixi_understanding.qces.config import (
    LEGACY_TEMPORAL_ROLE_MODE,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
)
from mixi_understanding.qces.model import QCESOutput
from mixi_understanding.qces.signal import qces_stft


QAAuditor = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class LossWeights:
    evidence_waveform: float = 1.0
    evidence_active_waveform: float = 2.0
    role_relative_waveform: float = 0.0
    weakest_role_waveform: float = 0.0
    separator_mask: float = 4.0
    raw_separator_mask: float = 2.0
    residual_waveform: float = 0.5
    multi_resolution_stft: float = 0.5
    reconstruction: float = 1.0
    temporal_cross_entropy: float = 0.5
    temporal_binary_cross_entropy: float = 0.5
    temporal_dice: float = 0.5
    temporal_role_dice: float = 0.5
    no_evidence: float = 0.1
    minimality: float = 0.02
    qa_sufficiency: float = 0.0
    qa_necessity: float = 0.0
    semantic_alignment: float = 0.0
    anchor_semantic_alignment: float = 0.0
    answer_semantic_alignment: float = 0.0
    same_semantic_classification: float = 0.0
    surface_semantic_invariance: float = 0.0
    surface_role_invariance: float = 0.0
    surface_no_evidence_invariance: float = 0.0
    surface_evidence_invariance: float = 0.0
    family_temporal_delta: float = 0.0
    family_evidence_delta: float = 0.0
    family_no_evidence_transition: float = 0.0
    question_temporal_delta: float = 0.0
    question_evidence_delta: float = 0.0


def _multi_resolution_stft_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: Sequence[int],
) -> torch.Tensor:
    terms = []
    for n_fft in fft_sizes:
        if n_fft > prediction.size(-1):
            continue
        window = torch.hann_window(
            n_fft, device=prediction.device, dtype=prediction.dtype
        )
        predicted = qces_stft(
            prediction,
            n_fft=n_fft,
            hop_length=max(1, n_fft // 4),
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        expected = qces_stft(
            target,
            n_fft=n_fft,
            hop_length=max(1, n_fft // 4),
            win_length=n_fft,
            window=window,
            return_complex=True,
        ).abs()
        expected_norm = torch.linalg.vector_norm(expected, dim=(-2, -1))
        spectral_convergence = torch.linalg.vector_norm(
            predicted - expected, dim=(-2, -1)
        ) / expected_norm.clamp_min(1e-6)
        log_magnitude = (
            (torch.log(predicted.clamp_min(1e-5)) - torch.log(expected.clamp_min(1e-5)))
            .abs()
            .mean(dim=(-2, -1))
        )
        # Spectral convergence is undefined for a silent no-evidence target.
        # Penalize predicted magnitude directly in that case instead of
        # dividing by epsilon, which otherwise dominates the whole batch.
        silent_magnitude = predicted.mean(dim=(-2, -1))
        per_example = torch.where(
            expected_norm > 1e-4,
            spectral_convergence + log_magnitude,
            silent_magnitude,
        )
        terms.append(per_example.mean())
    if not terms:
        return prediction.new_zeros(())
    return torch.stack(terms).mean()


def _frame_role_targets(
    anchor_mask: torch.Tensor, answer_mask: torch.Tensor, frames: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor = F.adaptive_max_pool1d(anchor_mask[:, None], frames).squeeze(1) > 0.5
    answer = F.adaptive_max_pool1d(answer_mask[:, None], frames).squeeze(1) > 0.5
    union = anchor | answer
    independent = torch.stack([~union, anchor, answer], dim=-1).to(anchor_mask.dtype)
    # Historical categorical targets deliberately retain answer-overwrites-
    # anchor semantics. They are used only in the legacy mode so historical
    # losses and checkpoints remain exact.
    labels = torch.zeros_like(anchor, dtype=torch.long)
    labels[anchor] = 1
    labels[answer] = 2
    return labels, independent, union.to(anchor_mask.dtype)


def _separator_mask_loss(
    predicted_mask: Optional[torch.Tensor],
    evidence_target: torch.Tensor,
    residual_target: torch.Tensor,
    sample_union: torch.Tensor,
    *,
    balance_temporal_classes: bool = False,
) -> torch.Tensor:
    """Supervise a bounded ideal-ratio mask without separator internals."""

    if predicted_mask is None:
        return evidence_target.new_zeros(())
    if predicted_mask.ndim == 2:
        target = F.adaptive_max_pool1d(
            sample_union[:, None], predicted_mask.size(-1)
        ).squeeze(1)
        if balance_temporal_classes:
            return _balanced_binary_frame_error(
                (predicted_mask - target).abs(), target
            ).mean()
        return F.l1_loss(predicted_mask, target)
    if predicted_mask.ndim != 3:
        raise ValueError("separator mask must have shape [B, T] or [B, F, T]")

    frequency_bins, frames = predicted_mask.shape[-2:]
    n_fft = max(2, 2 * (frequency_bins - 1))
    hop_length = max(1, round(evidence_target.size(-1) / max(frames - 1, 1)))
    window = torch.hann_window(
        n_fft, device=evidence_target.device, dtype=evidence_target.dtype
    )

    def magnitude(waveform: torch.Tensor) -> torch.Tensor:
        spectrum = qces_stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).abs()
        if spectrum.shape[-2:] != (frequency_bins, frames):
            spectrum = F.interpolate(
                spectrum[:, None],
                size=(frequency_bins, frames),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        return spectrum

    if torch.is_complex(predicted_mask):

        def complex_spectrum(waveform: torch.Tensor) -> torch.Tensor:
            result = qces_stft(
                waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True,
            )
            if result.shape[-2:] != (frequency_bins, frames):
                real = F.interpolate(
                    result.real[:, None],
                    size=(frequency_bins, frames),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                imaginary = F.interpolate(
                    result.imag[:, None],
                    size=(frequency_bins, frames),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                result = torch.complex(real, imaginary)
            return result

        evidence_spectrum = complex_spectrum(evidence_target)
        residual_spectrum = complex_spectrum(residual_target)
        mixture_spectrum = evidence_spectrum + residual_spectrum
        valid = mixture_spectrum.abs() > 1e-5
        safe_mixture = torch.where(
            valid, mixture_spectrum, torch.ones_like(mixture_spectrum)
        )
        target_mask = evidence_spectrum / safe_mixture
        target_mask = torch.where(valid, target_mask, torch.zeros_like(target_mask))
        target_magnitude = target_mask.abs()
        target_mask = target_mask * (2.0 / target_magnitude.clamp_min(1e-8)).clamp_max(
            1.0
        )
        mask_error = (predicted_mask - target_mask).abs()
    else:
        evidence_magnitude = magnitude(evidence_target)
        residual_magnitude = magnitude(residual_target)
        ideal_ratio = evidence_magnitude / (
            evidence_magnitude + residual_magnitude
        ).clamp_min(1e-6)
        mask_error = (predicted_mask - ideal_ratio).abs()
    active_frames = F.adaptive_max_pool1d(sample_union[:, None], frames).squeeze(1)
    frame_weight = 0.25 + 2.75 * active_frames
    return (mask_error * frame_weight[:, None]).sum() / (
        frame_weight.sum() * frequency_bins
    ).clamp_min(1.0)


def _dice_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (probability * target).sum(dim=-1)
    denominator = probability.sum(dim=-1) + target.sum(dim=-1)
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _balanced_binary_frame_error(
    elementwise_error: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Balance positive/negative frame means for every example and channel.

    Sparse role windows otherwise make the all-background solution cheaper
    than retaining a short event.  A class that is absent in one example still
    contributes its available side, which keeps no-evidence records useful.
    The returned tensor has the input shape with the frame dimension removed.
    """

    if elementwise_error.shape != target.shape or target.ndim not in {2, 3}:
        raise ValueError("balanced frame error needs matching [B,T] or [B,T,C] tensors")
    positive = target.to(elementwise_error.dtype)
    negative = 1.0 - positive
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    positive_mean = (elementwise_error * positive).sum(dim=1) / (
        positive_count.clamp_min(1.0)
    )
    negative_mean = (elementwise_error * negative).sum(dim=1) / (
        negative_count.clamp_min(1.0)
    )
    positive_present = (positive_count > 0).to(elementwise_error.dtype)
    negative_present = (negative_count > 0).to(elementwise_error.dtype)
    return (positive_mean * positive_present + negative_mean * negative_present) / (
        positive_present + negative_present
    ).clamp_min(1.0)


def _answer_log_probability(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    return logits.log_softmax(dim=-1).gather(1, targets[:, None]).squeeze(1)


def _relative_role_waveform_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_mask: torch.Tensor,
) -> torch.Tensor:
    """Return per-example role error normalized by that role's target energy."""

    absolute_error = ((prediction - target).abs() * sample_mask).sum(dim=-1)
    target_energy = (target.abs() * sample_mask).sum(dim=-1)
    return absolute_error / target_energy.clamp_min(1e-5)


class QCESLoss(nn.Module):
    """Stem, grounding, minimality, and optional counterfactual QA losses."""

    def __init__(
        self,
        weights: LossWeights = LossWeights(),
        fft_sizes: Sequence[int] = (256, 512, 1024),
        sufficiency_margin: float = 0.0,
        residual_margin: float = -0.5,
        counterfactual_transition_margin: float = 0.25,
        no_evidence_positive_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.fft_sizes = tuple(fft_sizes)
        self.sufficiency_margin = sufficiency_margin
        self.residual_margin = residual_margin
        if not 0.0 <= counterfactual_transition_margin <= 1.0:
            raise ValueError("counterfactual_transition_margin must be in [0, 1]")
        self.counterfactual_transition_margin = counterfactual_transition_margin
        if (
            not isinstance(no_evidence_positive_weight, (int, float))
            or not torch.isfinite(torch.tensor(float(no_evidence_positive_weight)))
            or no_evidence_positive_weight <= 0.0
        ):
            raise ValueError("no_evidence_positive_weight must be finite and positive")
        self.register_buffer(
            "no_evidence_positive_weight",
            torch.tensor(float(no_evidence_positive_weight)),
        )

    def forward(
        self,
        output: QCESOutput,
        batch: Dict[str, torch.Tensor],
        qa_auditor: Optional[QAAuditor] = None,
        answer_targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mixture = batch["mixture"]
        evidence_target = batch["evidence"]
        residual_target = batch["residual"]
        predicted_evidence = output.evidence
        predicted_residual = output.residual

        role_labels, independent_role_targets, union_target = _frame_role_targets(
            batch["anchor_mask"],
            batch["answer_mask"],
            output.composition.role_logits.size(1),
        )
        evidence_probability = output.composition.evidence_probability
        temporal_role_mode = output.composition.temporal_role_mode
        role_probabilities = output.composition.role_probabilities
        zero = mixture.new_zeros(())
        if temporal_role_mode == LEGACY_TEMPORAL_ROLE_MODE:
            temporal_cross_entropy = F.cross_entropy(
                output.composition.role_logits.reshape(
                    -1, output.composition.role_logits.size(-1)
                ),
                role_labels.reshape(-1),
                weight=output.composition.role_logits.new_tensor([0.25, 1.0, 1.0]),
            )
            temporal_binary_cross_entropy = zero
            anchor_role_target = (role_labels == 1).to(evidence_probability.dtype)
            answer_role_target = (role_labels == 2).to(evidence_probability.dtype)
        elif temporal_role_mode == OVERLAP_AWARE_TEMPORAL_ROLE_MODE:
            temporal_cross_entropy = zero
            binary_error = F.binary_cross_entropy_with_logits(
                output.composition.role_logits,
                independent_role_targets.to(output.composition.role_logits.dtype),
                reduction="none",
            )
            role_weights = output.composition.role_logits.new_tensor([0.25, 1.0, 1.0])
            balanced_binary_error = _balanced_binary_frame_error(
                binary_error,
                independent_role_targets.to(binary_error.dtype),
            )
            temporal_binary_cross_entropy = (
                balanced_binary_error * role_weights
            ).sum() / (binary_error.size(0) * role_weights.sum())
            anchor_role_target = independent_role_targets[..., 1].to(
                evidence_probability.dtype
            )
            answer_role_target = independent_role_targets[..., 2].to(
                evidence_probability.dtype
            )
        else:  # PromptComposition validates through its probability property.
            raise ValueError(f"unsupported temporal role mode: {temporal_role_mode!r}")
        answerable = 1.0 - batch["no_evidence"].to(evidence_probability.dtype)
        sample_union = (batch["anchor_mask"] + batch["answer_mask"]).clamp_max(1.0)
        active_error = (
            (predicted_evidence - evidence_target).abs() * sample_union
        ).sum(dim=-1) / sample_union.sum(dim=-1).clamp_min(1.0)
        anchor_target = batch.get("anchor_stem", evidence_target * batch["anchor_mask"])
        answer_target = batch.get("answer_stem", evidence_target * batch["answer_mask"])
        anchor_role_error = _relative_role_waveform_error(
            predicted_evidence,
            anchor_target,
            batch["anchor_mask"],
        )
        answer_role_error = _relative_role_waveform_error(
            predicted_evidence,
            answer_target,
            batch["answer_mask"],
        )
        mean_role_error = 0.5 * (anchor_role_error + answer_role_error)
        weakest_role_error = torch.maximum(anchor_role_error, answer_role_error)

        losses: Dict[str, torch.Tensor] = {
            "evidence_waveform": F.l1_loss(predicted_evidence, evidence_target),
            "evidence_active_waveform": (active_error * answerable).sum()
            / answerable.sum().clamp_min(1.0),
            "role_relative_waveform": (mean_role_error * answerable).sum()
            / answerable.sum().clamp_min(1.0),
            "weakest_role_waveform": (weakest_role_error * answerable).sum()
            / answerable.sum().clamp_min(1.0),
            "separator_mask": _separator_mask_loss(
                output.separation.mask,
                evidence_target,
                residual_target,
                sample_union,
                balance_temporal_classes=(
                    temporal_role_mode == OVERLAP_AWARE_TEMPORAL_ROLE_MODE
                ),
            ),
            "raw_separator_mask": _separator_mask_loss(
                output.separation.raw_mask,
                evidence_target,
                residual_target,
                sample_union,
                balance_temporal_classes=(
                    temporal_role_mode == OVERLAP_AWARE_TEMPORAL_ROLE_MODE
                ),
            ),
            "residual_waveform": F.l1_loss(predicted_residual, residual_target),
            "multi_resolution_stft": _multi_resolution_stft_loss(
                predicted_evidence, evidence_target, self.fft_sizes
            ),
            "reconstruction": F.l1_loss(
                predicted_evidence + predicted_residual, mixture
            ),
            # Legacy CE is flattened to avoid PyTorch's nondeterministic CUDA
            # nll_loss2d kernel. Overlap-aware mode instead uses independent
            # binary targets in which anchor and answer may both equal one.
            "temporal_cross_entropy": temporal_cross_entropy,
            "temporal_binary_cross_entropy": temporal_binary_cross_entropy,
            "temporal_dice": _dice_loss(evidence_probability, union_target),
            "temporal_role_dice": 0.5
            * (
                _dice_loss(
                    role_probabilities[..., 1],
                    anchor_role_target,
                )
                + _dice_loss(
                    role_probabilities[..., 2],
                    answer_role_target,
                )
            ),
            "no_evidence": F.binary_cross_entropy_with_logits(
                output.composition.no_evidence_logit,
                batch["no_evidence"].to(output.composition.no_evidence_logit.dtype),
                pos_weight=self.no_evidence_positive_weight.to(
                    output.composition.no_evidence_logit.dtype
                ),
            ),
            "minimality": (
                (
                    (
                        (evidence_probability * (1.0 - union_target)).sum(dim=-1)
                        / (1.0 - union_target).sum(dim=-1).clamp_min(1.0)
                    )
                    * answerable
                ).sum()
                / answerable.sum().clamp_min(1.0)
                if temporal_role_mode == OVERLAP_AWARE_TEMPORAL_ROLE_MODE
                else (evidence_probability.mean(dim=-1) * answerable).sum()
                / answerable.sum().clamp_min(1.0)
            ),
        }

        losses["qa_sufficiency"] = zero
        losses["qa_necessity"] = zero
        losses["semantic_alignment"] = zero
        losses["anchor_semantic_alignment"] = zero
        losses["answer_semantic_alignment"] = zero
        losses["same_semantic_classification"] = zero
        paired_requested = any(
            getattr(self.weights, name) > 0 for name in COUNTERFACTUAL_LOSS_NAMES
        )
        if paired_requested and "counterfactual_groups" not in batch:
            raise ValueError(
                "positive counterfactual loss weights require validated paired "
                "group metadata"
            )
        if paired_requested:
            metadata = batch["counterfactual_groups"]
            declared = metadata.get("plan_enabled", {})
            requirements = (
                (
                    "surface",
                    SURFACE_LOSS_NAMES,
                    "surface_pairs",
                ),
                (
                    "family",
                    FAMILY_LOSS_NAMES,
                    "primary_triplets",
                ),
                (
                    "question",
                    QUESTION_LOSS_NAMES,
                    "question_pairs",
                ),
            )
            for group_name, names, local_key in requirements:
                requested = any(getattr(self.weights, name) > 0 for name in names)
                locally_complete = bool(metadata.get(local_key))
                if requested and not (
                    bool(declared.get(group_name)) or locally_complete
                ):
                    raise ValueError(
                        f"{group_name} counterfactual loss requested without a "
                        "validated complete group plan"
                    )
        paired_losses, _, _ = counterfactual_objectives(
            output,
            batch,
            transition_margin=self.counterfactual_transition_margin,
        )
        losses.update(paired_losses)
        semantic_target = batch.get("semantic_target")
        if semantic_target is not None:
            semantic_error = 1.0 - F.cosine_similarity(
                output.composition.semantic_condition,
                semantic_target.to(output.composition.semantic_condition.dtype),
                dim=-1,
            )
            semantic_valid = batch.get("semantic_target_valid")
            semantic_mask = answerable
            if semantic_valid is not None:
                semantic_mask = semantic_mask * semantic_valid.to(semantic_error.dtype)
            losses["semantic_alignment"] = (
                semantic_error * semantic_mask
            ).sum() / semantic_mask.sum().clamp_min(1.0)
        role_semantic_keys = {
            "anchor_semantic_target",
            "answer_semantic_target",
            "same_semantic_target",
            "role_semantic_valid",
        }
        present_role_semantic_keys = role_semantic_keys & set(batch)
        if (
            present_role_semantic_keys
            and present_role_semantic_keys != role_semantic_keys
        ):
            raise ValueError(
                "role-semantic supervision is incomplete: expected anchor, "
                "answer, same-semantic, and validity targets"
            )
        dual_semantic_requested = any(
            getattr(self.weights, name) > 0
            for name in (
                "anchor_semantic_alignment",
                "answer_semantic_alignment",
                "same_semantic_classification",
            )
        )
        if dual_semantic_requested and not present_role_semantic_keys:
            raise ValueError(
                "positive dual-role semantic weights require role-specific "
                "semantic cache targets"
            )
        if present_role_semantic_keys:
            anchor_condition = output.composition.anchor_semantic_condition
            answer_condition = output.composition.answer_semantic_condition
            same_logit = output.composition.same_semantic_logit
            if (
                anchor_condition is None
                or answer_condition is None
                or same_logit is None
            ):
                raise ValueError(
                    "role-specific semantic targets require a dual_role composer"
                )
            valid = batch["role_semantic_valid"].to(anchor_condition.dtype)
            if valid.ndim != 1 or valid.shape != same_logit.shape:
                raise ValueError("role_semantic_valid must have shape [B]")
            denominator = valid.sum().clamp_min(1.0)
            anchor_error = 1.0 - F.cosine_similarity(
                anchor_condition,
                batch["anchor_semantic_target"].to(anchor_condition.dtype),
                dim=-1,
            )
            answer_error = 1.0 - F.cosine_similarity(
                answer_condition,
                batch["answer_semantic_target"].to(answer_condition.dtype),
                dim=-1,
            )
            same_error = F.binary_cross_entropy_with_logits(
                same_logit,
                batch["same_semantic_target"].to(same_logit.dtype),
                reduction="none",
            )
            losses["anchor_semantic_alignment"] = (
                anchor_error * valid
            ).sum() / denominator
            losses["answer_semantic_alignment"] = (
                answer_error * valid
            ).sum() / denominator
            losses["same_semantic_classification"] = (
                same_error * valid
            ).sum() / denominator
        if qa_auditor is not None:
            if answer_targets is None:
                raise ValueError("answer_targets are required with a QA auditor")
            question_ids = batch["question_ids"]
            question_mask = batch["question_mask"]
            original_logp = _answer_log_probability(
                qa_auditor(mixture, question_ids, question_mask), answer_targets
            )
            evidence_logp = _answer_log_probability(
                qa_auditor(predicted_evidence, question_ids, question_mask),
                answer_targets,
            )
            residual_logp = _answer_log_probability(
                qa_auditor(predicted_residual, question_ids, question_mask),
                answer_targets,
            )
            evidence_gain = evidence_logp - original_logp
            residual_gain = residual_logp - original_logp
            losses["qa_sufficiency"] = F.relu(
                self.sufficiency_margin - evidence_gain
            ).mean()
            losses["qa_necessity"] = F.relu(residual_gain - self.residual_margin).mean()

        total = sum(
            getattr(self.weights, name) * value for name, value in losses.items()
        )
        losses["total"] = total
        return total, losses
