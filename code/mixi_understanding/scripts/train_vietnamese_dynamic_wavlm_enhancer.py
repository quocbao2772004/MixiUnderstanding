#!/usr/bin/env python3
"""Fine-tune the automotive complex-mask enhancer with WavLM preservation.

The dynamic mixer produces source-disjoint dense-noise pairs in memory.  The
waveform/spectral losses teach suppression, while a frozen WavLM backbone
penalizes changes to speech representations that tend to damage ASR.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

from mixi_understanding.qces.metrics import scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve
from mixi_understanding.scripts.train_vietnamese_automotive_complexmask import (
    _complex_mask, _features, enhance_complex,
)
from mixi_understanding.scripts.train_vietnamese_automotive_tfmask import (
    RATE, TFMaskUNet, _istft, _load, _negative_si_sdr, _stft,
)
from mixi_understanding.scripts.train_vietnamese_whisper_noise_adapter import (
    DynamicPairDataset, _dynamic_training_data, _jsonl,
)


class DynamicChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, base: DynamicPairDataset, seconds: float, identity_probability: float, seed: int) -> None:
        self.base = base
        self.frames = int(seconds * RATE)
        self.identity_probability = identity_probability
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.base.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.base[index]
        rng = random.Random(self.seed + self.epoch * 10_000_019 + index)
        noisy = torch.from_numpy(item["noisy"]).float()
        clean = torch.from_numpy(item["clean"]).float()
        length = min(noisy.numel(), clean.numel())
        noisy, clean = noisy[:length], clean[:length]
        # Mixture and target are a paired waveform and must use exactly the
        # same crop. Sampling their starts independently silently turns the
        # reconstruction target into unrelated speech.
        if length >= self.frames:
            start = rng.randint(0, length - self.frames)
            noisy = noisy[start:start + self.frames]
            clean = clean[start:start + self.frames]
        else:
            repeats = int(math.ceil(self.frames / max(length, 1)))
            noisy = noisy.repeat(repeats)[:self.frames]
            clean = clean.repeat(repeats)[:self.frames]
        if rng.random() < self.identity_probability:
            noisy = clean.clone()
        return noisy, clean


def _wavlm_input(value: torch.Tensor) -> torch.Tensor:
    value = value - value.mean(dim=-1, keepdim=True)
    return value / value.var(dim=-1, keepdim=True, unbiased=False).add(1e-7).sqrt()


def _wavlm_loss(wavlm: torch.nn.Module, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Keep this path in FP32: gradients through the frozen WavLM convolutional
    # frontend overflow in FP16 even when the scalar loss is finite.
    predicted_input = _wavlm_input(prediction.float())
    target_input = _wavlm_input(target.float())
    with torch.inference_mode():
        target_hidden = wavlm(target_input).last_hidden_state
    predicted_hidden = wavlm(predicted_input).last_hidden_state
    left = F.layer_norm(predicted_hidden.float(), (predicted_hidden.shape[-1],))
    right = F.layer_norm(target_hidden.float(), (target_hidden.shape[-1],))
    return F.smooth_l1_loss(left, right, beta=0.1)


@torch.inference_mode()
def _validate(
    model: TFMaskUNet,
    wavlm: torch.nn.Module,
    scenes: list[dict[str, Any]],
    device: torch.device,
    representation_seconds: float,
) -> dict[str, float]:
    model.eval(); wavlm.eval()
    sisdri: list[float] = []; complex_errors: list[float] = []; representations: list[float] = []
    maximum = int(representation_seconds * RATE)
    for scene in scenes:
        mixture = _load(_resolve(scene["mixture_path"])).to(device)
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        target = _load(_resolve(speech["stem_path"])).to(device)
        length = min(mixture.numel(), target.numel()); mixture, target = mixture[:length], target[:length]
        prediction = enhance_complex(model, mixture[None])[0]
        enhanced = float(scale_invariant_sdr(prediction[None], target[None])[0])
        baseline = float(scale_invariant_sdr(mixture[None], target[None])[0])
        sisdri.append(enhanced - baseline)
        complex_errors.append(float((_stft(prediction[None]) - _stft(target[None])).abs().mean()))
        start = max(0, int((float(speech["onset_seconds"]) - 0.1) * RATE))
        end = min(length, start + maximum)
        representations.append(float(_wavlm_loss(wavlm, prediction[None, start:end], target[None, start:end])))
    return {
        "mean_si_sdri_db_↑": float(np.mean(sisdri)),
        "mean_complex_l1_↓": float(np.mean(complex_errors)),
        "mean_wavlm_l1_↓": float(np.mean(representations)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleurs", type=Path, default=PROJECT_ROOT / "data/_sources/fleurs_vi_vn/validation.parquet")
    parser.add_argument("--source-bank-root", type=Path, default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1/primary_clean")
    parser.add_argument("--eval-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_chaos_v1")
    parser.add_argument("--initial-checkpoint", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_complexmask_v1/best_sisdr.pt")
    parser.add_argument("--wavlm", default=str(Path.home() / ".cache/huggingface/hub/models--microsoft--wavlm-base-plus-sv/snapshots/feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_dynamic_wavlm_enhancer_v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--scenes-per-epoch", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--identity-probability", type=float, default=0.25)
    parser.add_argument("--wavlm-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026081029)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda")
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(args.eval_dir.resolve() / "scenes.jsonl")
    validation = [scene for scene in scenes if scene["split"] == "val"]
    speech_rows, noise_pool, data_audit = _dynamic_training_data(
        args.fleurs.resolve(), args.source_bank_root.resolve(), scenes,
        maximum_duration=20.0, maximum_words=70,
    )
    base = DynamicPairDataset(speech_rows, noise_pool, args.scenes_per_epoch, args.seed)
    dataset = DynamicChunkDataset(base, args.chunk_seconds, args.identity_probability, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    model = TFMaskUNet(input_channels=3, output_channels=2, activation="tanh2").to(device)
    initial = torch.load(args.initial_checkpoint.resolve(), map_location="cpu", weights_only=False)
    model.load_state_dict(initial["model"], strict=True)
    wavlm = AutoModel.from_pretrained(args.wavlm, local_files_only=True, dtype=torch.float32).to(device).eval()
    for parameter in wavlm.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # Complex masks and the WavLM input-gradient path are numerically unstable
    # in FP16. The enhancer is small, so use FP32 end-to-end here.
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics = _validate(model, wavlm, validation, device, args.chunk_seconds)
    history: list[dict[str, Any]] = [{"epoch": 0, "train_loss": None, "validation": metrics}]
    best_si, best_wavlm = metrics["mean_si_sdri_db_↑"], metrics["mean_wavlm_l1_↓"]
    base_state = {"epoch": 0, "model": model.state_dict(), "validation": metrics}
    torch.save(base_state, output / "best_sisdr.pt"); torch.save(base_state, output / "best_wavlm.pt")
    print(json.dumps(history[-1]), flush=True)

    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch); model.train(); wavlm.eval()
        losses: list[float] = []; rep_losses: list[float] = []
        for step, (mixture, target) in enumerate(loader, 1):
            mixture = mixture.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=False):
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
                representation = _wavlm_loss(wavlm, prediction.float(), target.float())
                loss = mask_loss + 2 * complex_loss + 2 * spectral + 8 * waveform + 0.1 * _negative_si_sdr(prediction.float(), target.float()) + args.wavlm_weight * representation
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu())); rep_losses.append(float(representation.detach().cpu()))
            if step % 25 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "steps": len(loader), "loss": float(np.mean(losses[-25:])), "wavlm_loss": float(np.mean(rep_losses[-25:]))}), flush=True)
        metrics = _validate(model, wavlm, validation, device, args.chunk_seconds)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_wavlm_loss": float(np.mean(rep_losses)), "validation": metrics}
        history.append(record); state = {"epoch": epoch, "model": model.state_dict(), "validation": metrics}
        torch.save(state, output / f"epoch_{epoch}.pt"); torch.save(state, output / "last.pt")
        if metrics["mean_si_sdri_db_↑"] > best_si:
            best_si = metrics["mean_si_sdri_db_↑"]; torch.save(state, output / "best_sisdr.pt")
        if metrics["mean_wavlm_l1_↓"] < best_wavlm:
            best_wavlm = metrics["mean_wavlm_l1_↓"]; torch.save(state, output / "best_wavlm.pt")
        _atomic_text(output / "history.json", json.dumps(history, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(record), flush=True)
    receipt = {
        "format": "qces_vietnamese_dynamic_wavlm_enhancer_receipt_v1", "complete": True,
        "method": "dynamic dense-noise complex-mask fine-tuning with frozen WavLM representation preservation",
        "scenes_per_epoch": len(dataset), "epochs": args.epochs, "data_audit": data_audit,
        "initial_checkpoint": _portable(args.initial_checkpoint.resolve()), "wavlm_weight": args.wavlm_weight,
        "identity_probability": args.identity_probability, "best_si_sdri_db_↑": best_si,
        "best_wavlm_l1_↓": best_wavlm, "history": _portable(output / "history.json"),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
