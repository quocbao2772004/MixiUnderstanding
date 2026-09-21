#!/usr/bin/env python3
"""Train a complex ratio-mask model that also corrects noise-dominated phase."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mixi_understanding.qces.metrics import scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve
from mixi_understanding.scripts.train_vietnamese_automotive_tfmask import (
    PairDataset, TFMaskUNet, _istft, _jsonl, _load, _negative_si_sdr, _stft,
)


def _features(spectrum: torch.Tensor) -> torch.Tensor:
    magnitude = spectrum.abs().clamp_min(1e-6)
    return torch.stack((torch.log1p(magnitude), spectrum.real / magnitude, spectrum.imag / magnitude), dim=1)


def _complex_mask(model: TFMaskUNet, spectrum: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = model(_features(spectrum))
    mask = torch.complex(values[:, 0], values[:, 1])
    return spectrum * mask, values


def enhance_complex(model: TFMaskUNet, mixture: torch.Tensor) -> torch.Tensor:
    spectrum = _stft(mixture)
    prediction, _ = _complex_mask(model, spectrum)
    return _istft(prediction, mixture.shape[-1])


@torch.inference_mode()
def _validate(model: TFMaskUNet, scenes: list[dict], device: torch.device) -> dict[str, float]:
    model.eval(); sisdri, complex_error = [], []
    for scene in scenes:
        mixture = _load(_resolve(scene["mixture_path"])).to(device)
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        target = _load(_resolve(speech["stem_path"])).to(device)
        length = min(len(mixture), len(target)); mixture, target = mixture[:length], target[:length]
        prediction = enhance_complex(model, mixture[None])
        enhanced = float(scale_invariant_sdr(prediction, target[None])[0])
        baseline = float(scale_invariant_sdr(mixture[None], target[None])[0])
        sisdri.append(enhanced - baseline)
        complex_error.append(float((_stft(prediction) - _stft(target[None])).abs().mean()))
    return {"mean_si_sdri_db_↑": float(np.mean(sisdri)), "mean_complex_l1_↓": float(np.mean(complex_error))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_enhancement_train_v1")
    parser.add_argument("--eval-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_complexmask_v1")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026080527)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    rows = _jsonl(args.train_dir.resolve() / "train.jsonl")
    val = [row for row in _jsonl(args.eval_dir.resolve() / "scenes.jsonl") if row["split"] == "val"]
    dataset = PairDataset(rows, args.chunk_seconds, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    model = TFMaskUNet(input_channels=3, output_channels=2, activation="tanh2").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    initial = _validate(model, val, device)
    history = [{"epoch": 0, "train_loss": None, "validation": initial}]
    best_si, best_complex = initial["mean_si_sdri_db_↑"], initial["mean_complex_l1_↓"]
    print(json.dumps(history[-1]), flush=True)
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch); model.train(); losses = []
        for step, (mixture, target) in enumerate(loader, 1):
            mixture, target = mixture.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                mix_spec, target_spec = _stft(mixture), _stft(target)
                predicted_spec, mask_values = _complex_mask(model, mix_spec)
                prediction = _istft(predicted_spec, mixture.shape[-1])
                denominator = mix_spec.abs().square() + 1e-5
                oracle = target_spec * mix_spec.conj() / denominator
                oracle_values = torch.stack((oracle.real.clamp(-2, 2), oracle.imag.clamp(-2, 2)), dim=1)
                mask_loss = F.l1_loss(mask_values, oracle_values)
                complex_loss = (predicted_spec - target_spec).abs().mean()
                spectral = F.l1_loss(torch.log1p(predicted_spec.abs()), torch.log1p(target_spec.abs()))
                waveform = F.smooth_l1_loss(prediction, target, beta=0.02)
                loss = mask_loss + 2 * complex_loss + 2 * spectral + 8 * waveform + 0.1 * _negative_si_sdr(prediction.float(), target.float())
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
            if step % 25 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "steps": len(loader), "loss": float(np.mean(losses[-25:]))}), flush=True)
        metrics = _validate(model, val, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metrics}; history.append(record)
        state = {"epoch": epoch, "model": model.state_dict(), "validation": metrics}
        torch.save(state, output / "last.pt")
        if metrics["mean_si_sdri_db_↑"] > best_si:
            best_si = metrics["mean_si_sdri_db_↑"]; torch.save(state, output / "best_sisdr.pt")
        if metrics["mean_complex_l1_↓"] < best_complex:
            best_complex = metrics["mean_complex_l1_↓"]; torch.save(state, output / "best_complex.pt")
        _atomic_text(output / "history.json", json.dumps(history, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    receipt = {"format": "qces_vietnamese_automotive_complexmask_receipt_v1", "complete": True,
               "train_scenes": len(rows), "validation_scenes": len(val), "best_si_sdri_db_↑": best_si,
               "best_complex_l1_↓": best_complex, "history": _portable(output / "history.json")}
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
