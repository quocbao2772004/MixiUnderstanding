#!/usr/bin/env python3
"""Train an identity-preserving local TF refiner for public-22 evidence.

The refiner cannot synthesize an unconstrained source.  It predicts a bounded
time-frequency gate between the detector crop and projected AudioSep output:

    Y[f,t] = C[f,t] + 0.5 * sigmoid(G[f,t]) * (A[f,t] - C[f,t]).

The output head is initialized close to the identity crop.  Training includes
an explicit relative-harm penalty and stronger identity preservation when the
crop already matches the target well.  Model selection uses only a
source-disjoint internal split of the training cache; external dev is scored
once after checkpoint selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
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
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.metrics import scale_dependent_sdr


FORMAT = "qces_public22_identity_tf_refiner_v1"
RATE = 16_000
N_FFT = 512
HOP = 128
WIN = 512
MAX_GATE = 0.50


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3] / "outputs"
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
        default=root / "qces_public22_identity_tf_refiner_v1",
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--internal-dev-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2266)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


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


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _gain_metrics(gain: np.ndarray, gate: np.ndarray) -> dict[str, Any]:
    return {
        "events": int(len(gain)),
        "gain_over_crop_db_↑": _summary(gain.tolist()),
        "positive_rate_↑": float(np.mean(gain > 0.0)),
        "nonnegative_rate_↑": float(np.mean(gain >= 0.0)),
        "harmful_below_minus_1db_rate_↓": float(np.mean(gain < -1.0)),
        "mean_tf_gate": float(np.mean(gate)),
    }


class WaveformCacheDataset(Dataset[tuple[torch.Tensor, ...]]):
    def __init__(self, payload: Mapping[str, Any], indices: Sequence[int]) -> None:
        self.crop = payload["crop"]
        self.audiosep = payload["audiosep_projected"]
        self.target = payload["target"]
        self.label_id = payload["label_id"]
        self.indices = [int(index) for index in indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        source = self.indices[index]
        return (
            self.crop[source].float(),
            self.audiosep[source].float(),
            self.target[source].float(),
            self.label_id[source].long(),
            torch.tensor(source, dtype=torch.long),
        )


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(4, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.silu(value + self.layers(value))


class IdentityTFRefiner(nn.Module):
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
        self.output = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -3.0)

    def forward(
        self, crop: torch.Tensor, audiosep: torch.Tensor, label_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        crop_spec = _stft(crop)
        audiosep_spec = _stft(audiosep)
        difference = audiosep_spec - crop_spec
        feature = torch.stack(
            (
                crop_spec.real,
                crop_spec.imag,
                difference.real,
                difference.imag,
                torch.log1p(crop_spec.abs()),
                torch.log1p(audiosep_spec.abs()),
            ),
            dim=1,
        )
        semantic = self.label_embedding(label_id)[:, :, None, None]
        semantic = semantic.expand(-1, -1, feature.shape[-2], feature.shape[-1])
        hidden = F.silu(self.input(torch.cat((feature, semantic), dim=1)))
        hidden = self.blocks(hidden)
        gate = MAX_GATE * torch.sigmoid(self.output(hidden)[:, 0])
        prediction_spec = crop_spec + gate * difference
        prediction = torch.istft(
            prediction_spec, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
            window=_window(prediction_spec.device), length=crop.shape[-1], center=True,
        )
        return prediction, gate


_WINDOWS: dict[str, torch.Tensor] = {}


def _window(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _WINDOWS:
        _WINDOWS[key] = torch.hann_window(WIN, device=device)
    return _WINDOWS[key]


def _stft(value: torch.Tensor) -> torch.Tensor:
    return torch.stft(
        value, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
        window=_window(value.device), return_complex=True, center=True,
    )


def _loss(
    prediction: torch.Tensor, gate: torch.Tensor,
    crop: torch.Tensor, target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_energy = target.square().sum(-1).clamp_min(1e-6)
    prediction_error = (prediction - target).square().sum(-1) / target_energy
    crop_error = (crop - target).square().sum(-1) / target_energy
    relative_harm = F.relu(prediction_error - crop_error).mean()
    wave_scale = target.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
    waveform = F.smooth_l1_loss(prediction / wave_scale, target / wave_scale, beta=0.05)
    prediction_spec = _stft(prediction)
    target_spec = _stft(target)
    spectral_scale = target_spec.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    complex_error = F.smooth_l1_loss(
        torch.view_as_real(prediction_spec / spectral_scale[:, None]),
        torch.view_as_real(target_spec / spectral_scale[:, None]),
        beta=0.05,
    )
    log_magnitude = F.l1_loss(torch.log1p(prediction_spec.abs()), torch.log1p(target_spec.abs()))
    crop_sdr = scale_dependent_sdr(crop.float(), target.float()).detach()
    identity_weight = torch.sigmoid((crop_sdr - 5.0) / 2.0)
    identity_error = (
        (prediction - crop).square().sum(-1) / crop.square().sum(-1).clamp_min(1e-6)
    )
    identity = (identity_weight * identity_error).mean()
    sdr = scale_dependent_sdr(prediction.float(), target.float()).clamp(-30.0, 30.0)
    total = (
        complex_error + 0.30 * log_magnitude + 0.35 * waveform
        + 0.75 * relative_harm + 0.75 * identity
        + 0.01 * gate.mean() - 0.03 * sdr.mean()
    )
    return total, {
        "loss": float(total.detach()),
        "complex": float(complex_error.detach()),
        "waveform": float(waveform.detach()),
        "relative_harm": float(relative_harm.detach()),
        "identity": float(identity.detach()),
        "sd_sdr": float(sdr.mean().detach()),
        "gate": float(gate.mean().detach()),
    }


@torch.inference_mode()
def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, float]]]:
    model.eval()
    gains: list[float] = []
    gates: list[float] = []
    rows: list[dict[str, float]] = []
    for crop, audiosep, target, label_id, source_index in loader:
        crop, audiosep, target = crop.to(device), audiosep.to(device), target.to(device)
        prediction, gate = model(crop, audiosep, label_id.to(device))
        crop_sdr = scale_dependent_sdr(crop.float(), target.float())
        prediction_sdr = scale_dependent_sdr(prediction.float(), target.float())
        gain = prediction_sdr - crop_sdr
        mean_gate = gate.mean(dim=(-2, -1))
        gains.extend(gain.cpu().tolist())
        gates.extend(mean_gate.cpu().tolist())
        rows.extend(
            {
                "source_index": int(source_index[index]),
                "gain_over_crop_db": float(gain[index].cpu()),
                "mean_tf_gate": float(mean_gate[index].cpu()),
            }
            for index in range(len(crop))
        )
    return _gain_metrics(np.asarray(gains), np.asarray(gates)), rows


def _checkpoint_objective(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    gain = metrics["gain_over_crop_db_↑"]
    median = float(gain["median"])
    mean = float(gain["mean"])
    positive = float(metrics["positive_rate_↑"])
    harmful = float(metrics["harmful_below_minus_1db_rate_↓"])
    safe = float(mean >= 0.0 and harmful <= 0.15)
    passes = float(safe and median >= 1.0 and positive >= 0.60)
    bottleneck = min(median / 1.0, positive / 0.60) if safe else -harmful
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
    dev_payload = torch.load(dev_path, map_location="cpu", weights_only=False)
    if train_payload.get("format") != "qces_public22_audiosep_waveform_cache_v1":
        raise ValueError("invalid train waveform cache")
    if dev_payload.get("format") != "qces_public22_audiosep_waveform_cache_v1":
        raise ValueError("invalid dev waveform cache")
    train_groups = np.asarray([str(row["source_id"]) for row in train_payload["metadata"]])
    all_indices = np.arange(len(train_groups))
    fit_index, internal_index = next(GroupShuffleSplit(
        n_splits=1, test_size=args.internal_dev_fraction, random_state=args.seed,
    ).split(all_indices, groups=train_groups))
    if set(train_groups[fit_index]) & set(train_groups[internal_index]):
        raise RuntimeError("source leakage in internal split")
    fit_dataset = WaveformCacheDataset(train_payload, fit_index)
    internal_dataset = WaveformCacheDataset(train_payload, internal_index)
    dev_dataset = WaveformCacheDataset(dev_payload, np.arange(len(dev_payload["metadata"])))
    generator = torch.Generator().manual_seed(args.seed)
    fit_loader = DataLoader(
        fit_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    internal_loader = DataLoader(
        internal_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_labels = int(max(train_payload["label_id"].max(), dev_payload["label_id"].max())) + 1
    model = IdentityTFRefiner(num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    history: list[dict[str, Any]] = []
    best_objective: tuple[float, ...] | None = None
    best_epoch = -1
    stale = 0
    checkpoint_path = output / "best.pt"
    for epoch in range(args.epochs):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for crop, audiosep, target, label_id, _source_index in fit_loader:
            crop, audiosep, target = crop.to(device), audiosep.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction, gate = model(crop, audiosep, label_id.to(device))
            loss, parts = _loss(prediction, gate, crop, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        internal_metrics, _ = _evaluate(model, internal_loader, device)
        objective = _checkpoint_objective(internal_metrics)
        row = {
            "epoch": epoch,
            "train": {key: value / max(batches, 1) for key, value in totals.items()},
            "internal_dev": internal_metrics,
            "objective": objective,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if best_objective is None or objective > best_objective:
            best_objective = objective
            best_epoch = epoch
            stale = 0
            torch.save({
                "format": FORMAT,
                "model_state": model.state_dict(),
                "num_labels": num_labels,
                "max_gate": MAX_GATE,
                "epoch": epoch,
                "internal_metrics": internal_metrics,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    internal_metrics, _ = _evaluate(model, internal_loader, device)
    dev_metrics, dev_rows = _evaluate(model, dev_loader, device)
    crop = dev_payload["crop"].float()
    audiosep = dev_payload["audiosep_projected"].float()
    target = dev_payload["target"].float()
    audiosep_gain = (
        scale_dependent_sdr(audiosep, target) - scale_dependent_sdr(crop, target)
    ).numpy()
    audiosep_metrics = _gain_metrics(audiosep_gain, np.ones(len(audiosep_gain)))
    gate = {
        "median_gain_db_min": 1.0,
        "mean_gain_db_min": 0.0,
        "positive_rate_min": 0.60,
        "harmful_below_minus_1db_rate_max": 0.15,
    }
    gain = dev_metrics["gain_over_crop_db_↑"]
    accepted = (
        float(gain["median"]) >= gate["median_gain_db_min"]
        and float(gain["mean"]) >= gate["mean_gain_db_min"]
        and float(dev_metrics["positive_rate_↑"]) >= gate["positive_rate_min"]
        and float(dev_metrics["harmful_below_minus_1db_rate_↓"]) <= gate["harmful_below_minus_1db_rate_max"]
    )
    decision_rows = []
    for row in dev_rows:
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
        "equation": "Y = crop_STFT + gate_tf * (AudioSep_projected_STFT - crop_STFT)",
        "max_gate": MAX_GATE,
        "train": {
            "events": len(train_payload["metadata"]),
            "fit_events": len(fit_index),
            "internal_dev_events": len(internal_index),
            "source_overlap": 0,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_internal_dev": internal_metrics,
        },
        "external_dev": {
            "events": len(dev_payload["metadata"]),
            "crop_fallback": _gain_metrics(np.zeros(len(dev_rows)), np.zeros(len(dev_rows))),
            "audiosep_projected": audiosep_metrics,
            "identity_tf_refiner": dev_metrics,
        },
        "protocol": {
            "target_used_at_inference": False,
            "unconstrained_waveform_generation": False,
            "checkpoint_selected_on_external_dev": False,
            "checkpoint_selected_on_source_disjoint_internal_train_split": True,
        },
        "dev_acceptance_gate": {**gate, "passed": accepted},
        "decision": "candidate_for_public22_beta" if accepted else "keep_crop_fallback",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dev_decisions": str(decisions_path.resolve()),
        "dev_decisions_sha256": _sha256(decisions_path),
        "inputs": {
            "train_cache": {"path": str(train_path), "sha256": _sha256(train_path)},
            "dev_cache": {"path": str(dev_path), "sha256": _sha256(dev_path)},
        },
        "history": history,
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "best_epoch": best_epoch,
        "external_dev": receipt["external_dev"],
        "gate": receipt["dev_acceptance_gate"],
        "decision": receipt["decision"],
        "receipt": str((output / "receipt.json").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
