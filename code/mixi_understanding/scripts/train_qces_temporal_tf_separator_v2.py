#!/usr/bin/env python3
"""V2 temporal separator with active SI-SDR and energy-calibrated training."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.scripts import train_qces_temporal_tf_separator_v1 as v1


def active_si_sdr(
    prediction: torch.Tensor, target: torch.Tensor, gate: torch.Tensor,
) -> torch.Tensor:
    count = gate.sum(dim=-1, keepdim=True).clamp_min(1)
    prediction = (prediction - (prediction * gate).sum(dim=-1, keepdim=True) / count) * gate
    target = (target - (target * gate).sum(dim=-1, keepdim=True) / count) * gate
    projection = (
        (prediction * target).sum(dim=-1, keepdim=True)
        / target.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
    ) * target
    error = prediction - projection
    ratio = projection.square().sum(dim=-1) / error.square().sum(dim=-1).clamp_min(1e-8)
    return 10.0 * torch.log10(ratio.clamp_min(1e-8))


def v2_loss_function(
    prediction: torch.Tensor, prediction_spec: torch.Tensor,
    target: torch.Tensor, target_spec: torch.Tensor,
    gate: torch.Tensor, frame_gate: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    on_frame = frame_gate[:, None]
    off_frame = 1.0 - on_frame
    complex_error = (prediction_spec - target_spec).abs()
    on_complex = (complex_error * on_frame).sum() / (on_frame.sum() * complex_error.shape[-2]).clamp_min(1)
    off_complex = (prediction_spec.abs() * off_frame).sum() / (off_frame.sum() * complex_error.shape[-2]).clamp_min(1)
    log_error = (torch.log1p(prediction_spec.abs()) - torch.log1p(target_spec.abs())).abs()
    on_log = (log_error * on_frame).sum() / (on_frame.sum() * log_error.shape[-2]).clamp_min(1)
    on_wave = ((prediction - target).abs() * gate).sum() / gate.sum().clamp_min(1)
    off_wave = (prediction.abs() * (1.0 - gate)).sum() / (1.0 - gate).sum().clamp_min(1)
    si_sdr = active_si_sdr(prediction, target, gate)
    si_sdr_loss = -si_sdr.clamp(-30, 30).mean() / 10.0
    prediction_energy = (prediction.square() * gate).sum(dim=-1)
    target_energy = (target.square() * gate).sum(dim=-1).clamp_min(1e-8)
    energy_calibration = torch.log((prediction_energy + 1e-8) / target_energy).abs().mean()
    loss = (
        0.20 * on_complex + 0.05 * off_complex + 0.20 * on_log
        + 0.50 * on_wave + 0.05 * off_wave
        + 0.50 * si_sdr_loss + 0.10 * energy_calibration
    )
    return loss, {
        "on_complex": float(on_complex.detach()), "off_complex": float(off_complex.detach()),
        "on_log": float(on_log.detach()), "on_wave": float(on_wave.detach()),
        "off_wave": float(off_wave.detach()), "active_si_sdr_db": float(si_sdr.mean().detach()),
        "energy_calibration": float(energy_calibration.detach()),
    }


def main() -> None:
    v1.FORMAT = "qces_temporal_tf_separator_training_receipt_v2"
    v1.CHECKPOINT_FORMAT = "qces_temporal_tf_separator_checkpoint_v2"
    v1.loss_function = v2_loss_function
    v1.main()


if __name__ == "__main__":
    main()
