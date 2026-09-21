"""Small dependency-free evaluation metrics for QCES experiments."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from mixi_understanding.qces.model import QCESOutput


def scale_invariant_sdr(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (
        (prediction * target).sum(dim=-1, keepdim=True)
        * target
        / target.square().sum(dim=-1, keepdim=True).clamp_min(epsilon)
    )
    noise = prediction - projection
    ratio = projection.square().sum(dim=-1) / noise.square().sum(dim=-1).clamp_min(
        epsilon
    )
    return 10.0 * torch.log10(ratio.clamp_min(epsilon))


def scale_dependent_sdr(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8
) -> torch.Tensor:
    """Scale-dependent SDR from Le Roux et al., ICASSP 2019.

    With ``alpha = <prediction,target> / ||target||^2``, SD-SDR is
    ``10 log10(||alpha target||^2 / ||target - prediction||^2)``.  Unlike
    SI-SDR, attenuating an otherwise perfect waveform is therefore penalized.
    Signals are zero-mean along time, matching :func:`scale_invariant_sdr`.
    """

    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    target_energy = target.square().sum(dim=-1, keepdim=True)
    alpha = (prediction * target).sum(dim=-1, keepdim=True) / (
        target_energy.clamp_min(epsilon)
    )
    scaled_target = alpha * target
    numerator = scaled_target.square().sum(dim=-1)
    distortion = (target - prediction).square().sum(dim=-1)
    ratio = numerator / distortion.clamp_min(epsilon)
    return 10.0 * torch.log10(ratio.clamp_min(epsilon))


def qces_metrics(
    output: QCESOutput, batch: Dict[str, torch.Tensor], threshold: float = 0.5
) -> Dict[str, torch.Tensor]:
    frames = output.composition.evidence_probability.size(1)
    anchor = F.adaptive_max_pool1d(batch["anchor_mask"][:, None], frames)[:, 0]
    answer = F.adaptive_max_pool1d(batch["answer_mask"][:, None], frames)[:, 0]
    target = (anchor + answer) > 0.5
    predicted = output.composition.evidence_probability >= threshold
    intersection = (predicted & target).sum(dim=-1).float()
    union = (predicted | target).sum(dim=-1).float()
    temporal_iou = torch.where(
        union > 0,
        intersection / union.clamp_min(1.0),
        torch.ones_like(union),
    )
    answerable = batch["no_evidence"] < 0.5
    no_evidence = ~answerable
    evidence_si_sdr = scale_invariant_sdr(output.evidence, batch["evidence"])
    evidence_sd_sdr = scale_dependent_sdr(output.evidence, batch["evidence"])
    if answerable.any():
        evidence_si_sdr = evidence_si_sdr[answerable].mean()
        evidence_sd_sdr = evidence_sd_sdr[answerable].mean()
    else:
        evidence_si_sdr = evidence_si_sdr.new_zeros(())
        evidence_sd_sdr = evidence_sd_sdr.new_zeros(())
    no_evidence_prediction = (
        torch.sigmoid(output.composition.no_evidence_logit) >= threshold
    )
    if answerable.any():
        answerable_temporal_iou = temporal_iou[answerable].mean()
    else:
        answerable_temporal_iou = temporal_iou.new_zeros(())
    per_example_retained_ratio = output.evidence.abs().sum(dim=-1) / batch[
        "mixture"
    ].abs().sum(dim=-1).clamp_min(1e-8)
    if no_evidence.any():
        no_evidence_retained_ratio = per_example_retained_ratio[no_evidence].mean()
    else:
        no_evidence_retained_ratio = per_example_retained_ratio.new_zeros(())
    metrics = {
        "evidence_l1": F.l1_loss(output.evidence, batch["evidence"]),
        "residual_l1": F.l1_loss(output.residual, batch["residual"]),
        "evidence_si_sdr": evidence_si_sdr,
        "evidence_sd_sdr": evidence_sd_sdr,
        "residual_si_sdr": scale_invariant_sdr(
            output.residual, batch["residual"]
        ).mean(),
        "mixture_consistency": output.separation.mixture_error.mean(),
        "temporal_iou": temporal_iou.mean(),
        "answerable_temporal_iou": answerable_temporal_iou,
        "no_evidence_accuracy": (no_evidence_prediction == batch["no_evidence"].bool())
        .float()
        .mean(),
        "retained_ratio": output.evidence.abs().sum()
        / batch["mixture"].abs().sum().clamp_min(1e-8),
        "no_evidence_retained_ratio": no_evidence_retained_ratio,
    }
    same_probability = output.composition.same_semantic_probability
    semantic_candidate_weight = output.composition.foundation_semantic_candidate_weight
    if semantic_candidate_weight is not None:
        metrics["foundation_semantic_candidate_weight_descriptive"] = (
            semantic_candidate_weight
        )
    if same_probability is not None and {
        "same_semantic_target",
        "role_semantic_valid",
    }.issubset(batch):
        same_target = batch["same_semantic_target"].to(same_probability.dtype)
        valid = batch["role_semantic_valid"] > 0.5
        if valid.any():
            valid_probability = same_probability[valid]
            valid_target = same_target[valid]
            metrics["same_semantic_brier"] = (
                (valid_probability - valid_target).square().mean()
            )
            metrics["same_semantic_accuracy"] = (
                ((valid_probability >= threshold) == valid_target.bool()).float().mean()
            )
            positives = valid_target > 0.5
            negatives = ~positives
            metrics["same_semantic_probability_on_same"] = (
                valid_probability[positives].mean()
                if positives.any()
                else valid_probability.new_zeros(())
            )
            metrics["same_semantic_probability_on_different"] = (
                valid_probability[negatives].mean()
                if negatives.any()
                else valid_probability.new_zeros(())
            )
        else:
            zero = same_probability.new_zeros(())
            metrics.update(
                same_semantic_brier=zero,
                same_semantic_accuracy=zero,
                same_semantic_probability_on_same=zero,
                same_semantic_probability_on_different=zero,
            )
    return metrics
