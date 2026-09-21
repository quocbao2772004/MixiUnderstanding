#!/usr/bin/env python3
"""Domain-adapt SpeechBrain SepFormer to severe Vietnamese automotive noise."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from speechbrain.inference.separation import SepformerSeparation
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.metrics import scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve


SAMPLE_RATE = 16_000


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load(path: Path) -> torch.Tensor:
    value, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(value.mean(axis=1)).float()
    return AF.resample(wave, rate, SAMPLE_RATE) if rate != SAMPLE_RATE else wave


class EnhancementPairs(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, rows: list[dict[str, Any]], chunk_seconds: float, seed: int) -> None:
        self.rows = rows
        self.frames = int(round(chunk_seconds * SAMPLE_RATE))
        self.seed = seed
        self.epoch = 0

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
            pad = self.frames - length
            return F.pad(mixture, (0, pad)), F.pad(target, (0, pad))
        generator = random.Random(self.seed + self.epoch * 1_000_003 + index)
        speech_start = int(float(row["speech_onset_seconds"]) * SAMPLE_RATE)
        speech_end = int(float(row["speech_offset_seconds"]) * SAMPLE_RATE)
        low = max(0, speech_start - self.frames + int(0.4 * SAMPLE_RATE))
        high = min(length - self.frames, max(speech_start, speech_end - int(0.4 * SAMPLE_RATE)))
        start = generator.randint(low, max(low, high))
        return mixture[start : start + self.frames], target[start : start + self.frames]


def _negative_si_sdr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (prediction * target).sum(dim=-1, keepdim=True) * target
    projection = projection / (target.square().sum(dim=-1, keepdim=True) + 1e-8)
    noise = prediction - projection
    score = 10.0 * torch.log10((projection.square().sum(dim=-1) + 1e-8) / (noise.square().sum(dim=-1) + 1e-8))
    return -score.mean()


def _spectral_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    total = prediction.new_tensor(0.0)
    for n_fft, hop in ((256, 64), (512, 128), (1024, 256)):
        window = torch.hann_window(n_fft, device=prediction.device, dtype=prediction.dtype)
        pred = torch.stft(prediction, n_fft, hop, n_fft, window, return_complex=True)
        gold = torch.stft(target, n_fft, hop, n_fft, window, return_complex=True)
        total = total + F.l1_loss(torch.log1p(pred.abs()), torch.log1p(gold.abs()))
    return total / 3.0


@torch.inference_mode()
def _validate(model: SepformerSeparation, scenes: list[dict[str, Any]], device: torch.device) -> dict[str, float]:
    rows = []
    model.mods.eval()
    for scene in scenes:
        mixture = _load(_resolve(scene["mixture_path"]))
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        target = _load(_resolve(speech["stem_path"]))
        prediction = model.separate_batch(mixture[None].to(device))[0, :, 0].cpu()
        length = min(len(prediction), len(mixture), len(target))
        enhanced = float(scale_invariant_sdr(prediction[:length][None], target[:length][None])[0])
        baseline = float(scale_invariant_sdr(mixture[:length][None], target[:length][None])[0])
        rows.append((enhanced, enhanced - baseline))
    return {
        "mean_si_sdr_db_↑": float(np.mean([x[0] for x in rows])),
        "mean_si_sdri_db_↑": float(np.mean([x[1] for x in rows])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_enhancement_train_v1")
    parser.add_argument("--eval-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_sepformer_finetune_v1")
    parser.add_argument("--source", default="speechbrain/sepformer-wham16k-enhancement")
    parser.add_argument("--savedir", type=Path, default=PROJECT_ROOT / "outputs/_models/sepformer-wham16k-enhancement")
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026080523)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_rows = _jsonl(args.train_dir.resolve() / "train.jsonl")
    if args.max_train_scenes:
        train_rows = train_rows[: args.max_train_scenes]
    eval_scenes = [row for row in _jsonl(args.eval_dir.resolve() / "scenes.jsonl") if row["split"] == "val"]
    dataset = EnhancementPairs(train_rows, args.chunk_seconds, args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)

    model = SepformerSeparation.from_hparams(
        source=args.source,
        savedir=str(args.savedir.resolve()),
        run_opts={"device": str(device)},
    )
    if args.initial_checkpoint is not None:
        initial_checkpoint = torch.load(args.initial_checkpoint.resolve(), map_location="cpu", weights_only=False)
        model.mods.load_state_dict(initial_checkpoint["mods"], strict=True)
    # The learned analysis/synthesis filterbank is already general; adapt the
    # separator mask to our severe noise distribution.
    for parameter in model.mods.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.mods.decoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.mods.masknet.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.mods.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    initial = _validate(model, eval_scenes, device)
    history = [{"epoch": 0, "train_loss": None, "validation": initial}]
    best_score, best_epoch = initial["mean_si_sdri_db_↑"], 0
    best_path = output / "best.pt"
    torch.save({"epoch": 0, "mods": model.mods.state_dict(), "validation": initial}, best_path)
    print(json.dumps(history[-1]), flush=True)

    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        model.mods.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for step, (mixture, target) in enumerate(loader, 1):
            mixture, target = mixture.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model.separate_batch(mixture)[:, :, 0]
                sisdr = _negative_si_sdr(prediction.float(), target.float())
                spectral = _spectral_loss(prediction.float(), target.float())
                waveform = F.smooth_l1_loss(prediction.float(), target.float(), beta=0.02)
                loss = sisdr + 2.0 * spectral + 5.0 * waveform
                scaled_loss = loss / args.grad_accum
            scaler.scale(scaled_loss).backward()
            if step % args.grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if step % 25 == 0:
                print(json.dumps({"epoch": epoch, "step": step, "steps": len(loader), "loss": float(np.mean(losses[-25:]))}), flush=True)
        validation = _validate(model, eval_scenes, device)
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": validation}
        history.append(record)
        last_path = output / "last.pt"
        torch.save({"epoch": epoch, "mods": model.mods.state_dict(), "optimizer": optimizer.state_dict(), "validation": validation}, last_path)
        if validation["mean_si_sdri_db_↑"] > best_score:
            best_score, best_epoch = validation["mean_si_sdri_db_↑"], epoch
            torch.save({"epoch": epoch, "mods": model.mods.state_dict(), "validation": validation}, best_path)
        _atomic_text(output / "history.json", json.dumps(history, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(record), flush=True)

    receipt = {
        "format": "qces_vietnamese_automotive_sepformer_finetune_receipt_v1",
        "complete": True,
        "pretrained_source": args.source,
        "train_scenes": len(train_rows),
        "eval_scenes": len(eval_scenes),
        "best_epoch": best_epoch,
        "best_validation": next(row["validation"] for row in history if row["epoch"] == best_epoch),
        "initial_validation": initial,
        "checkpoint": _portable(best_path),
        "history": _portable(output / "history.json"),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
