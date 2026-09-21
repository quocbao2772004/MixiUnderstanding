#!/usr/bin/env python3
"""Train a strong-or-identity TF refiner with oracle gate supervision.

This v2 model learns (1) a dense, bounded interpolation mask between crop and
projected AudioSep and (2) an utterance-level edit decision.  The dense target
is the closed-form real mask that minimizes complex target error per TF bin.
At inference the clean target is absent; only crop, AudioSep, and event label
are used.  Edit threshold and checkpoint are selected on an internal
source-disjoint split before the external development cache is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from mixi_understanding.qces.metrics import scale_dependent_sdr
from mixi_understanding.scripts.train_qces_public22_identity_tf_refiner_v1 import (
    DilatedResidualBlock,
    FORMAT as V1_FORMAT,
    HOP,
    N_FFT,
    RATE,
    WIN,
    WaveformCacheDataset,
    _gain_metrics,
    _seed,
    _stft,
    _window,
)


FORMAT = "qces_public22_strong_or_identity_tf_v2"
EDIT_THRESHOLDS = (0.35, 0.50, 0.65)


def parse_args() -> argparse.Namespace:
    root = PROJECT_ROOT / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cache", type=Path,
        default=root / "qces_public22_audiosep_waveforms_v1_train/waveforms.pt",
    )
    parser.add_argument(
        "--dev-cache", type=Path,
        default=root / "qces_public22_audiosep_waveforms_v1_dev/waveforms.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "qces_public22_strong_or_identity_tf_v2",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--internal-dev-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2267)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class StrongOrIdentityTF(nn.Module):
    def __init__(self, num_labels: int, channels: int = 24, semantic_channels: int = 8) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(num_labels, semantic_channels)
        self.input = nn.Conv2d(6 + semantic_channels, channels, 3, padding=1)
        self.blocks = nn.Sequential(
            DilatedResidualBlock(channels, 1),
            DilatedResidualBlock(channels, 2),
            DilatedResidualBlock(channels, 4),
            DilatedResidualBlock(channels, 8),
        )
        self.local_output = nn.Conv2d(channels, 1, 1)
        self.edit_output = nn.Linear(channels, 1)
        nn.init.zeros_(self.local_output.weight)
        nn.init.zeros_(self.local_output.bias)
        nn.init.zeros_(self.edit_output.weight)
        nn.init.constant_(self.edit_output.bias, -0.5)

    def encode(
        self, crop: torch.Tensor, audiosep: torch.Tensor, label_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        crop_spec = _stft(crop)
        audiosep_spec = _stft(audiosep)
        difference = audiosep_spec - crop_spec
        feature = torch.stack((
            crop_spec.real, crop_spec.imag, difference.real, difference.imag,
            torch.log1p(crop_spec.abs()), torch.log1p(audiosep_spec.abs()),
        ), dim=1)
        semantic = self.label_embedding(label_id)[:, :, None, None]
        semantic = semantic.expand(-1, -1, feature.shape[-2], feature.shape[-1])
        hidden = F.silu(self.input(torch.cat((feature, semantic), dim=1)))
        hidden = self.blocks(hidden)
        local_logits = self.local_output(hidden)[:, 0]
        edit_logits = self.edit_output(hidden.mean(dim=(-2, -1)))[:, 0]
        return crop_spec, difference, local_logits, edit_logits

    @staticmethod
    def render(
        crop_spec: torch.Tensor, difference: torch.Tensor,
        local_logits: torch.Tensor, edit_logits: torch.Tensor,
        length: int, threshold: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_gate = torch.sigmoid(local_logits)
        edit_probability = torch.sigmoid(edit_logits)
        if threshold is None:
            edit = edit_probability
        else:
            edit = (edit_probability >= threshold).to(local_gate.dtype)
        gate = local_gate * edit[:, None, None]
        prediction_spec = crop_spec + gate * difference
        prediction = torch.istft(
            prediction_spec, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
            window=_window(prediction_spec.device), length=length, center=True,
        )
        return prediction, gate, edit_probability


def _oracle_gate(
    crop_spec: torch.Tensor, difference: torch.Tensor, target_spec: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    numerator = ((target_spec - crop_spec) * difference.conj()).real
    denominator = difference.abs().square().clamp_min(1e-7)
    gate = (numerator / denominator).clamp(0.0, 1.0)
    weight = difference.abs().sqrt() * (target_spec.abs().sqrt() + 0.1)
    weight = weight / weight.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
    return gate, weight.clamp(0.05, 10.0)


def _loss(
    model: StrongOrIdentityTF,
    crop: torch.Tensor, audiosep: torch.Tensor, target: torch.Tensor,
    label_id: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    crop_spec, difference, local_logits, edit_logits = model.encode(crop, audiosep, label_id)
    target_spec = _stft(target)
    oracle_gate, oracle_weight = _oracle_gate(crop_spec, difference, target_spec)
    with torch.no_grad():
        oracle_wave = torch.istft(
            crop_spec + oracle_gate * difference,
            n_fft=N_FFT, hop_length=HOP, win_length=WIN,
            window=_window(crop.device), length=crop.shape[-1], center=True,
        )
        oracle_gain = (
            scale_dependent_sdr(oracle_wave.float(), target.float())
            - scale_dependent_sdr(crop.float(), target.float())
        )
        edit_target = (oracle_gain >= 0.5).float()
    prediction, gate, edit_probability = model.render(
        crop_spec, difference, local_logits, edit_logits, crop.shape[-1], threshold=None,
    )
    gate_supervision = (
        F.binary_cross_entropy_with_logits(local_logits, oracle_gate, reduction="none")
        * oracle_weight
    ).mean()
    positive_weight = torch.tensor(1.5, device=crop.device)
    edit_supervision = F.binary_cross_entropy_with_logits(
        edit_logits, edit_target, pos_weight=positive_weight,
    )
    target_energy = target.square().sum(-1).clamp_min(1e-6)
    prediction_error = (prediction - target).square().sum(-1) / target_energy
    crop_error = (crop - target).square().sum(-1) / target_energy
    relative_harm = F.relu(prediction_error - crop_error).mean()
    wave_scale = target.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
    waveform = F.smooth_l1_loss(prediction / wave_scale, target / wave_scale, beta=0.05)
    prediction_spec = _stft(prediction)
    spectral_scale = target_spec.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    complex_error = F.smooth_l1_loss(
        torch.view_as_real(prediction_spec / spectral_scale[:, None]),
        torch.view_as_real(target_spec / spectral_scale[:, None]), beta=0.05,
    )
    crop_sdr = scale_dependent_sdr(crop.float(), target.float()).detach()
    identity_weight = torch.sigmoid((crop_sdr - 5.0) / 2.0)
    identity_error = (
        (prediction - crop).square().sum(-1) / crop.square().sum(-1).clamp_min(1e-6)
    )
    identity = (identity_weight * identity_error).mean()
    sdr = scale_dependent_sdr(prediction.float(), target.float()).clamp(-30.0, 30.0)
    total = (
        complex_error + 0.30 * waveform + 0.70 * gate_supervision
        + 0.25 * edit_supervision + 0.50 * relative_harm
        + 0.40 * identity - 0.05 * sdr.mean()
    )
    return total, {
        "loss": float(total.detach()),
        "complex": float(complex_error.detach()),
        "waveform": float(waveform.detach()),
        "gate_supervision": float(gate_supervision.detach()),
        "edit_supervision": float(edit_supervision.detach()),
        "relative_harm": float(relative_harm.detach()),
        "identity": float(identity.detach()),
        "sd_sdr": float(sdr.mean().detach()),
        "mean_gate": float(gate.mean().detach()),
        "mean_edit_probability": float(edit_probability.mean().detach()),
        "edit_target_rate": float(edit_target.mean().detach()),
    }


@torch.inference_mode()
def _evaluate_thresholds(
    model: StrongOrIdentityTF, loader: DataLoader, device: torch.device,
    thresholds: Sequence[float],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, float]]]]:
    model.eval()
    gains = {f"{threshold:.2f}": [] for threshold in thresholds}
    gates = {f"{threshold:.2f}": [] for threshold in thresholds}
    rows = {f"{threshold:.2f}": [] for threshold in thresholds}
    for crop, audiosep, target, label_id, source_index in loader:
        crop, audiosep, target = crop.to(device), audiosep.to(device), target.to(device)
        crop_spec, difference, local_logits, edit_logits = model.encode(
            crop, audiosep, label_id.to(device),
        )
        crop_sdr = scale_dependent_sdr(crop.float(), target.float())
        for threshold in thresholds:
            key = f"{threshold:.2f}"
            prediction, gate, edit_probability = model.render(
                crop_spec, difference, local_logits, edit_logits,
                crop.shape[-1], threshold,
            )
            gain = scale_dependent_sdr(prediction.float(), target.float()) - crop_sdr
            mean_gate = gate.mean(dim=(-2, -1))
            gains[key].extend(gain.cpu().tolist())
            gates[key].extend(mean_gate.cpu().tolist())
            rows[key].extend({
                "source_index": int(source_index[index]),
                "gain_over_crop_db": float(gain[index].cpu()),
                "mean_tf_gate": float(mean_gate[index].cpu()),
                "edit_probability": float(edit_probability[index].cpu()),
            } for index in range(len(crop)))
    metrics = {
        key: _gain_metrics(np.asarray(gains[key]), np.asarray(gates[key]))
        for key in gains
    }
    return metrics, rows


def _objective(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    gain = metrics["gain_over_crop_db_↑"]
    median, mean = float(gain["median"]), float(gain["mean"])
    positive = float(metrics["positive_rate_↑"])
    harmful = float(metrics["harmful_below_minus_1db_rate_↓"])
    safe = float(mean >= 0.0 and harmful <= 0.15)
    passes = float(safe and median >= 1.0 and positive >= 0.60)
    bottleneck = min(median, positive / 0.60) if safe else -harmful
    return passes, safe, bottleneck, mean, -harmful


def main() -> None:
    args = parse_args()
    _seed(args.seed)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_path, dev_path = args.train_cache.resolve(), args.dev_cache.resolve()
    train_payload = torch.load(train_path, map_location="cpu", weights_only=False)
    if train_payload.get("format") != "qces_public22_audiosep_waveform_cache_v1":
        raise ValueError("invalid train cache")
    groups = np.asarray([str(row["source_id"]) for row in train_payload["metadata"]])
    indices = np.arange(len(groups))
    fit_index, internal_index = next(GroupShuffleSplit(
        n_splits=1, test_size=args.internal_dev_fraction, random_state=args.seed,
    ).split(indices, groups=groups))
    if set(groups[fit_index]) & set(groups[internal_index]):
        raise RuntimeError("internal source leakage")
    fit_loader = DataLoader(
        WaveformCacheDataset(train_payload, fit_index), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(args.seed),
        num_workers=args.num_workers, pin_memory=True,
    )
    internal_loader = DataLoader(
        WaveformCacheDataset(train_payload, internal_index), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_labels = int(train_payload["label_id"].max()) + 1
    model = StrongOrIdentityTF(num_labels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_objective: tuple[float, ...] | None = None
    best_epoch, best_threshold, stale = -1, EDIT_THRESHOLDS[0], 0
    checkpoint_path = output / "best.pt"
    for epoch in range(args.epochs):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for crop, audiosep, target, label_id, _source_index in fit_loader:
            crop, audiosep, target = crop.to(device), audiosep.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = _loss(model, crop, audiosep, target, label_id.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        internal_metrics, _ = _evaluate_thresholds(
            model, internal_loader, device, EDIT_THRESHOLDS,
        )
        threshold = max(internal_metrics, key=lambda key: _objective(internal_metrics[key]))
        objective = _objective(internal_metrics[threshold])
        row = {
            "epoch": epoch,
            "train": {key: value / max(batches, 1) for key, value in totals.items()},
            "selected_internal_threshold": threshold,
            "selected_internal_metrics": internal_metrics[threshold],
            "objective": objective,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_epoch = epoch
            best_threshold = float(threshold)
            stale = 0
            torch.save({
                "format": FORMAT,
                "model_state": model.state_dict(),
                "num_labels": num_labels,
                "epoch": epoch,
                "edit_threshold": best_threshold,
                "internal_metrics": internal_metrics[threshold],
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    internal_metrics, _ = _evaluate_thresholds(
        model, internal_loader, device, (best_threshold,),
    )
    # External dev is deliberately loaded only after epoch and threshold lock.
    dev_payload = torch.load(dev_path, map_location="cpu", weights_only=False)
    dev_loader = DataLoader(
        WaveformCacheDataset(dev_payload, np.arange(len(dev_payload["metadata"]))),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    dev_metrics_by_threshold, dev_rows_by_threshold = _evaluate_thresholds(
        model, dev_loader, device, (best_threshold,),
    )
    threshold_key = f"{best_threshold:.2f}"
    dev_metrics = dev_metrics_by_threshold[threshold_key]
    crop = dev_payload["crop"].float()
    audiosep = dev_payload["audiosep_projected"].float()
    target = dev_payload["target"].float()
    audiosep_gain = (
        scale_dependent_sdr(audiosep, target) - scale_dependent_sdr(crop, target)
    ).numpy()
    audiosep_metrics = _gain_metrics(audiosep_gain, np.ones(len(audiosep_gain)))
    gate = {
        "median_gain_db_min": 1.0, "mean_gain_db_min": 0.0,
        "positive_rate_min": 0.60, "harmful_below_minus_1db_rate_max": 0.15,
    }
    gain = dev_metrics["gain_over_crop_db_↑"]
    accepted = (
        float(gain["median"]) >= 1.0 and float(gain["mean"]) >= 0.0
        and float(dev_metrics["positive_rate_↑"]) >= 0.60
        and float(dev_metrics["harmful_below_minus_1db_rate_↓"]) <= 0.15
    )
    decision_rows = []
    for row in dev_rows_by_threshold[threshold_key]:
        metadata = dev_payload["metadata"][int(row["source_index"])]
        decision_rows.append({**metadata, **row})
    decisions_path = output / "dev_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        for row in decision_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "best_epoch": best_epoch,
        "locked_edit_threshold": best_threshold,
        "internal_dev": internal_metrics[threshold_key],
        "external_dev": {
            "audiosep_projected": audiosep_metrics,
            "strong_or_identity_tf": dev_metrics,
        },
        "protocol": {
            "dense_oracle_gate_is_training_supervision_only": True,
            "target_used_at_inference": False,
            "checkpoint_or_threshold_selected_on_external_dev": False,
            "source_overlap_internal": 0,
        },
        "dev_acceptance_gate": {**gate, "passed": accepted},
        "decision": "candidate_for_public22_beta" if accepted else "keep_crop_fallback",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "inputs": {
            "train_cache": {"path": str(train_path), "sha256": _sha256(train_path)},
            "dev_cache": {"path": str(dev_path), "sha256": _sha256(dev_path)},
        },
        "history": history,
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "best_epoch": best_epoch,
        "locked_edit_threshold": best_threshold,
        "internal_dev": receipt["internal_dev"],
        "external_dev": receipt["external_dev"],
        "gate": receipt["dev_acceptance_gate"],
        "decision": receipt["decision"],
        "receipt": str((output / "receipt.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
