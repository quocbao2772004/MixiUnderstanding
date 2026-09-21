#!/usr/bin/env python3
"""Train an ASR-preserving mixture-phase TF mask for automotive speech."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.metrics import scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve


RATE = 16_000
N_FFT = 512
HOP = 128


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load(path: Path) -> torch.Tensor:
    value, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(value.mean(axis=1)).float()
    return AF.resample(wave, rate, RATE) if rate != RATE else wave


class PairDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], seconds: float, seed: int) -> None:
        self.rows, self.frames, self.seed, self.epoch = rows, int(seconds * RATE), seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        mixture, target = _load(_resolve(row["mixture_path"])), _load(_resolve(row["target_path"]))
        length = min(len(mixture), len(target))
        mixture, target = mixture[:length], target[:length]
        if length < self.frames:
            return F.pad(mixture, (0, self.frames - length)), F.pad(target, (0, self.frames - length))
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        s0, s1 = int(row["speech_onset_seconds"] * RATE), int(row["speech_offset_seconds"] * RATE)
        low = max(0, s0 - self.frames + RATE // 2)
        high = min(length - self.frames, max(s0, s1 - RATE // 2))
        start = rng.randint(low, max(low, high))
        return mixture[start : start + self.frames], target[start : start + self.frames]


class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        groups = min(8, output_channels)
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels), nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(groups, output_channels), nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class TFMaskUNet(nn.Module):
    def __init__(self, base: int = 16, input_channels: int = 1, output_channels: int = 1,
                 activation: str = "sigmoid") -> None:
        super().__init__()
        self.activation = activation
        self.e1, self.e2 = ConvBlock(input_channels, base), ConvBlock(base, base * 2)
        self.e3, self.e4 = ConvBlock(base * 2, base * 4), ConvBlock(base * 4, base * 8)
        self.bottleneck = ConvBlock(base * 8, base * 12)
        self.d4 = ConvBlock(base * 20, base * 8)
        self.d3 = ConvBlock(base * 12, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.output = nn.Conv2d(base, output_channels, 1)

    @staticmethod
    def _down(value: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(value, 2)

    @staticmethod
    def _up(value: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(value)
        e2 = self.e2(self._down(e1))
        e3 = self.e3(self._down(e2))
        e4 = self.e4(self._down(e3))
        hidden = self.bottleneck(self._down(e4))
        hidden = self.d4(torch.cat([self._up(hidden, e4), e4], dim=1))
        hidden = self.d3(torch.cat([self._up(hidden, e3), e3], dim=1))
        hidden = self.d2(torch.cat([self._up(hidden, e2), e2], dim=1))
        hidden = self.d1(torch.cat([self._up(hidden, e1), e1], dim=1))
        output = self.output(hidden)
        if self.activation == "sigmoid":
            return torch.sigmoid(output)
        if self.activation == "tanh2":
            return 2.0 * torch.tanh(output)
        return output


def _stft(wave: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(N_FFT, device=wave.device, dtype=wave.dtype)
    return torch.stft(wave, N_FFT, HOP, N_FFT, window, return_complex=True)


def _istft(spec: torch.Tensor, length: int) -> torch.Tensor:
    window = torch.hann_window(N_FFT, device=spec.device, dtype=spec.real.dtype)
    return torch.istft(spec, N_FFT, HOP, N_FFT, window, length=length)


def enhance(model: nn.Module, mixture: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    spectrum = _stft(mixture)
    feature = torch.log1p(spectrum.abs())[:, None]
    mask = model(feature)[:, 0]
    return _istft(spectrum * mask, mixture.shape[-1]), mask


def _negative_si_sdr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (prediction * target).sum(-1, keepdim=True) * target / (target.square().sum(-1, keepdim=True) + 1e-8)
    noise = prediction - projection
    return -(10 * torch.log10((projection.square().sum(-1) + 1e-8) / (noise.square().sum(-1) + 1e-8))).mean()


@torch.inference_mode()
def _validate(model: nn.Module, scenes: list[dict[str, Any]], device: torch.device) -> dict[str, float]:
    model.eval()
    sisdri, spectral = [], []
    for scene in scenes:
        mixture = _load(_resolve(scene["mixture_path"])).to(device)
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        target = _load(_resolve(speech["stem_path"])).to(device)
        length = min(len(mixture), len(target))
        mixture, target = mixture[:length], target[:length]
        prediction, _ = enhance(model, mixture[None])
        enhanced = float(scale_invariant_sdr(prediction, target[None])[0])
        baseline = float(scale_invariant_sdr(mixture[None], target[None])[0])
        sisdri.append(enhanced - baseline)
        spectral.append(float(F.l1_loss(torch.log1p(_stft(prediction).abs()), torch.log1p(_stft(target[None]).abs()))))
    return {"mean_si_sdri_db_↑": float(np.mean(sisdri)), "mean_log_spectral_l1_↓": float(np.mean(spectral))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_enhancement_train_v1")
    parser.add_argument("--eval-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_tfmask_v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026080525)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(args.train_dir.resolve() / "train.jsonl")
    val = [row for row in _jsonl(args.eval_dir.resolve() / "scenes.jsonl") if row["split"] == "val"]
    dataset = PairDataset(rows, args.chunk_seconds, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    model = TFMaskUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    initial = _validate(model, val, device)
    history = [{"epoch": 0, "train_loss": None, "validation": initial}]
    best_sisdr, best_spectral = initial["mean_si_sdri_db_↑"], initial["mean_log_spectral_l1_↓"]
    print(json.dumps(history[-1]), flush=True)
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch); model.train(); losses = []
        for step, (mixture, target) in enumerate(loader, 1):
            mixture, target = mixture.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                mix_spec, target_spec = _stft(mixture), _stft(target)
                mask = model(torch.log1p(mix_spec.abs())[:, None])[:, 0]
                prediction = _istft(mix_spec * mask, mixture.shape[-1])
                irm = (target_spec.abs() / (mix_spec.abs() + 1e-6)).clamp(0, 1)
                mask_loss = F.l1_loss(mask, irm)
                spectral = F.l1_loss(torch.log1p((mix_spec * mask).abs()), torch.log1p(target_spec.abs()))
                waveform = F.smooth_l1_loss(prediction, target, beta=0.02)
                loss = 2 * mask_loss + 3 * spectral + 8 * waveform + 0.1 * _negative_si_sdr(prediction.float(), target.float())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
            if step % 25 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "steps": len(loader), "loss": float(np.mean(losses[-25:]))}), flush=True)
        metrics = _validate(model, val, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics}
        history.append(record)
        state = {"epoch": epoch, "model": model.state_dict(), "validation": metrics}
        torch.save(state, output / "last.pt")
        if metrics["mean_si_sdri_db_↑"] > best_sisdr:
            best_sisdr = metrics["mean_si_sdri_db_↑"]; torch.save(state, output / "best_sisdr.pt")
        if metrics["mean_log_spectral_l1_↓"] < best_spectral:
            best_spectral = metrics["mean_log_spectral_l1_↓"]; torch.save(state, output / "best_spectral.pt")
        _atomic_text(output / "history.json", json.dumps(history, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    receipt = {"format": "qces_vietnamese_automotive_tfmask_receipt_v1", "complete": True,
               "train_scenes": len(rows), "validation_scenes": len(val), "history": _portable(output / "history.json"),
               "best_si_sdri_db_↑": best_sisdr, "best_log_spectral_l1_↓": best_spectral}
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
