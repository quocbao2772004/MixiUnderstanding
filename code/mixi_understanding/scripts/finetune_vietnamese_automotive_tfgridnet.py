#!/usr/bin/env python3
"""Small, fixed-protocol domain adaptation of TF-GridNet for car speech.

Only the dataset's validation split is used.  At every update, a clean VieNeu
speech stem is remixed with a noise-only signal from another validation scene.
The test split is never loaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import (
    PROJECT_ROOT,
    _atomic_text,
    _portable,
    _resolve,
)
from mixi_understanding.scripts.evaluate_vietnamese_automotive_tfgridnet import (
    MODEL_SAMPLE_RATE,
    _model,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(path: Path) -> torch.Tensor:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1)).float()
    return AF.resample(wave, source_rate, MODEL_SAMPLE_RATE) if source_rate != MODEL_SAMPLE_RATE else wave


def _fit_length(wave: torch.Tensor, length: int, generator: random.Random) -> torch.Tensor:
    if wave.numel() < length:
        repeats = (length + wave.numel() - 1) // wave.numel()
        wave = wave.repeat(repeats)
    if wave.numel() == length:
        return wave
    start = generator.randint(0, wave.numel() - length)
    return wave[start : start + length]


def _remix(speech: torch.Tensor, noise: torch.Tensor, snr_db: float, generator: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    noise = _fit_length(noise, speech.numel(), generator)
    speech_rms = speech.square().mean().sqrt().clamp_min(1e-6)
    noise_rms = noise.square().mean().sqrt().clamp_min(1e-6)
    noise_gain = speech_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
    mixture = speech + noise * noise_gain
    peak = mixture.abs().max().clamp_min(1e-6)
    if peak > 0.95:
        gain = 0.95 / peak
        mixture = mixture * gain
        speech = speech * gain
    return mixture, speech


def _scale_invariant_mr_l1(target: torch.Tensor, estimate: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    eps = 1e-8
    scale = (estimate * target).sum(-1, keepdim=True) / (estimate.square().sum(-1, keepdim=True) + eps)
    aligned = estimate * scale
    time_loss = (aligned - target).abs().mean()
    spectral_loss = target.new_zeros(())
    for n_fft in (256, 512, 768, 1024):
        window = torch.hann_window(n_fft, device=target.device, dtype=target.dtype)
        target_mag = torch.stft(target, n_fft=n_fft, hop_length=n_fft // 2, window=window, return_complex=True).abs()
        estimate_mag = torch.stft(aligned, n_fft=n_fft, hop_length=n_fft // 2, window=window, return_complex=True).abs()
        spectral_loss = spectral_loss + (estimate_mag - target_mag).abs().mean()
    spectral_loss = spectral_loss / 4.0
    # The official checkpoint was trained with the same 0.5 time / 0.5 spectrum balance.
    base = 0.5 * time_loss + 0.5 * spectral_loss
    energy_loss = (torch.log(estimate.square().mean().clamp_min(eps)) - torch.log(target.square().mean().clamp_min(eps))).abs()
    loss = base + 0.01 * energy_loss
    return loss, {
        "loss": float(loss.detach()),
        "time_l1": float(time_loss.detach()),
        "spectral_l1": float(spectral_loss.detach()),
        "energy_log_error": float(energy_loss.detach()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_stress_v2",
    )
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/hub/models--espnet--ms_snsd_tfgridnet/snapshots/"
        "a203c32f0e5a0df3942e52d69d329ece391953ec/valid.loss.best.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_tfgridnet_finetune_v1",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--updates-per-epoch", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    generator = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = [
        scene
        for scene in _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")
        if scene["split"] == "val"
    ]
    if not scenes:
        raise RuntimeError("No validation scenes found")

    examples: list[dict[str, Any]] = []
    for scene in scenes:
        speech_event = next(event for event in scene["events"] if event["label"] == "Speech")
        mixture = _load(_resolve(scene["mixture_path"]))
        speech = _load(_resolve(speech_event["stem_path"]))
        length = min(mixture.numel(), speech.numel())
        mixture, speech = mixture[:length], speech[:length]
        examples.append({"scene_id": scene["scene_id"], "speech": speech, "noise": mixture - speech})

    model = _model(args.initial_checkpoint.resolve(), device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs * args.updates_per_epoch, 1), eta_min=args.learning_rate * 0.1
    )
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_rows = []
        for update in range(1, args.updates_per_epoch + 1):
            speech_example = examples[generator.randrange(len(examples))]
            noise_example = examples[generator.randrange(len(examples))]
            snr_db = generator.uniform(-16.0, -3.0)
            mixture, target = _remix(speech_example["speech"].clone(), noise_example["noise"], snr_db, generator)
            mixture, target = mixture[None].to(device), target[None].to(device)
            lengths = torch.tensor([mixture.shape[-1]], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs, _, _ = model(mixture, lengths)
            loss, parts = _scale_invariant_mr_l1(target, outputs[0])
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            optimizer.step()
            scheduler.step()
            epoch_rows.append({**parts, "grad_norm": grad_norm, "snr_db": snr_db})
        row = {
            "epoch": epoch,
            "updates": args.updates_per_epoch,
            "mean_loss": float(np.mean([item["loss"] for item in epoch_rows])),
            "mean_time_l1": float(np.mean([item["time_l1"] for item in epoch_rows])),
            "mean_spectral_l1": float(np.mean([item["spectral_l1"] for item in epoch_rows])),
            "mean_energy_log_error": float(np.mean([item["energy_log_error"] for item in epoch_rows])),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    checkpoint_path = output_dir / "tfgridnet_vieneu_car_final.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "epochs": args.epochs,
            "updates_per_epoch": args.updates_per_epoch,
            "seed": args.seed,
        },
        checkpoint_path,
    )
    history_path = output_dir / "history.jsonl"
    _atomic_text(history_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in history))
    receipt = {
        "format": "qces_vietnamese_automotive_tfgridnet_finetune_receipt_v1",
        "complete": True,
        "protocol": "fixed 12-epoch adaptation; val scenes only; cross-scene noise remix; test split never loaded",
        "training_scenes": len(examples),
        "epochs": args.epochs,
        "updates_per_epoch": args.updates_per_epoch,
        "snr_sampling_db": [-16.0, -3.0],
        "checkpoint": _portable(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "history": _portable(history_path),
    }
    _atomic_text(output_dir / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
