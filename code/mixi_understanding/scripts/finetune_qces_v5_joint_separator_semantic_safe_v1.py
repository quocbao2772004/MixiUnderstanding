#!/usr/bin/env python3
"""One-shot log-mel-preserving fine-tune of the V5 joint separator.

The experiment is intentionally gated: it may proceed only if waveform
separation is retained and frozen semantic top-1 later beats raw mixture.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader

from mixi_understanding.scripts.train_qces_v5_joint_convtasnet_micro_v1 import (
    SAMPLE_RATE,
    SAMPLES,
    SceneDataset,
    atomic_json,
    atomic_torch,
    evaluate,
    match_active,
    model_forward,
    sha256_file,
    training_loss,
)
from mixi_understanding.scripts.train_qces_v5_joint_convtasnet_pilot_v1 import stable_scenes


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAT = "qces_v5_joint_separator_semantic_safe_receipt_v1"
CHECKPOINT_FORMAT = "qces_v5_joint_separator_semantic_safe_checkpoint_v1"


def parse_args() -> argparse.Namespace:
    data = Path("/var/tmp/qces_full188_tiered_realistic_v5")
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=data / "detector_scene_manifest_tiered_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=data / "detector_scene_manifest_tiered_dev.jsonl")
    parser.add_argument("--initial-checkpoint", type=Path, default=base / "v5_joint_convtasnet_pilot_v1/joint_convtasnet_pilot_v1_best.pt")
    parser.add_argument("--output-dir", type=Path, default=base / "v5_joint_convtasnet_semantic_safe_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8221)
    parser.add_argument("--max-train-scenes", type=int, default=256)
    parser.add_argument("--max-dev-scenes", type=int, default=128)
    parser.add_argument("--num-sources", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--log-mel-weight", type=float, default=2.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "items"}


def make_mel(device: torch.device) -> torch.nn.Module:
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=80,
        power=2.0,
    ).to(device)


def active_log_mel_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    boxes: torch.Tensor,
    num_active: int,
    mel: torch.nn.Module,
) -> torch.Tensor:
    output_indices, target_indices = match_active(prediction, targets[:num_active])
    predicted_mel = torch.log(mel(prediction[output_indices]) + 1e-4)
    target_mel = torch.log(mel(targets[target_indices]) + 1e-4)
    error = (predicted_mel - target_mel).abs()
    frame_count = error.shape[-1]
    mask = torch.zeros(len(target_indices), frame_count, device=prediction.device)
    for row, target_index in enumerate(target_indices.tolist()):
        start, end = boxes[target_index].tolist()
        left = max(0, min(frame_count, int(start / SAMPLES * frame_count)))
        right = max(left + 1, min(frame_count, int(np.ceil(end / SAMPLES * frame_count))))
        mask[row, left:right] = 1.0
    active = (error * mask[:, None]).sum() / (mask.sum() * error.shape[1]).clamp_min(1.0)
    # A small global term suppresses output leakage without letting the long
    # silent region dominate the event-preservation objective.
    return active + 0.10 * error.mean()


@torch.inference_mode()
def evaluate_log_mel(
    model: torch.nn.Module,
    dataset: SceneDataset,
    device: torch.device,
    mel: torch.nn.Module,
    amp: bool,
) -> dict[str, float]:
    values: list[float] = []
    model.eval()
    for sample in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        mixture = sample["mixture"].to(device)
        targets = sample["targets"][0].to(device)
        boxes = sample["boxes"][0].to(device)
        count = int(sample["num_active"].item())
        prediction = model_forward(model, mixture, amp)[0]
        values.append(float(active_log_mel_loss(prediction, targets, boxes, count, mel)))
    array = np.asarray(values, dtype=np.float64)
    return {"scene_mean_\u2193": float(array.mean()), "scene_median_\u2193": float(np.median(array))}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_scenes = stable_scenes(args.train_manifest.resolve(), args.max_train_scenes, args.num_sources, args.seed)
    dev_scenes = stable_scenes(args.dev_manifest.resolve(), args.max_dev_scenes, args.num_sources, args.seed + 1)
    train_sources = {str(event["source_id"]) for scene in train_scenes for event in scene.row["events"]}
    dev_sources = {str(event["source_id"]) for scene in dev_scenes for event in scene.row["events"]}
    if train_sources & dev_sources:
        raise ValueError("train/dev source leakage")
    train_dataset = SceneDataset(train_scenes, args.num_sources)
    dev_dataset = SceneDataset(dev_scenes, args.num_sources)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    initial = torch.load(args.initial_checkpoint.resolve(), map_location="cpu", weights_only=True)
    architecture = initial["architecture"]
    model = torchaudio.models.ConvTasNet(
        num_sources=architecture["num_sources"],
        enc_num_feats=architecture["enc_num_feats"],
        msk_num_feats=architecture["mask_num_feats"],
        msk_num_hidden_feats=architecture["mask_num_hidden_feats"],
        msk_num_layers=architecture["mask_num_layers"],
        msk_num_stacks=architecture["mask_num_stacks"],
    ).to(device)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    mel = make_mel(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.10)
    # The added log-mel gradient overflows the default 65536 initial AMP
    # scale on this shared T4.  A conservative initial scale avoids silently
    # skipping the first fine-tuning epoch.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(args.amp and device.type == "cuda"),
        init_scale=1024.0,
        growth_interval=1000,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda")
    baseline_waveform = evaluate(model, dev_dataset, device, args.amp)
    baseline_mel = evaluate_log_mel(model, dev_dataset, device, mel, args.amp)
    print(json.dumps({"baseline_waveform": compact(baseline_waveform), "baseline_log_mel": baseline_mel}, sort_keys=True), flush=True)
    best_mel = float(baseline_mel["scene_mean_\u2193"])
    best_epoch = 0
    best_waveform = baseline_waveform
    best_mel_metrics = baseline_mel
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "base_loss": 0.0, "active_log_mel": 0.0}
        batches = 0
        for sample in loader:
            mixture = sample["mixture"].to(device)
            targets = sample["targets"][0].to(device)
            boxes = sample["boxes"][0].to(device)
            count = int(sample["num_active"].item())
            prediction = model_forward(model, mixture, args.amp)[0]
            base_loss, _ = training_loss(prediction, mixture[0], targets, count)
            semantic_safe = active_log_mel_loss(prediction, targets, boxes, count, mel)
            loss = base_loss + args.log_mel_weight * semantic_safe
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            totals["base_loss"] += float(base_loss.detach())
            totals["active_log_mel"] += float(semantic_safe.detach())
            batches += 1
        scheduler.step()
        waveform = evaluate(model, dev_dataset, device, args.amp)
        mel_metrics = evaluate_log_mel(model, dev_dataset, device, mel, args.amp)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {key: value / max(1, batches) for key, value in totals.items()},
            "dev_waveform": compact(waveform),
            "dev_log_mel": mel_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        waveform_safe = (
            float(waveform["si_sdri_vs_raw_mixture_db_\u2191"]["median"]) >= 5.0
            and float(waveform["si_sdri_vs_raw_mixture_db_\u2191"]["positive_fraction"]) >= 0.85
        )
        current_mel = float(mel_metrics["scene_mean_\u2193"])
        if waveform_safe and current_mel < best_mel - 1e-3:
            best_mel = current_mel
            best_epoch = epoch
            best_waveform = waveform
            best_mel_metrics = mel_metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint_path = output_dir / "joint_separator_semantic_safe_v1_best.pt"
    atomic_torch({
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": best_state,
        "architecture": architecture,
        "best_epoch": best_epoch,
        "sample_rate": SAMPLE_RATE,
        "samples": SAMPLES,
        "selection": "minimum_dev_active_log_mel_subject_to_waveform_safety",
    }, checkpoint_path)
    mel_reduction = 1.0 - float(best_mel_metrics["scene_mean_\u2193"]) / float(baseline_mel["scene_mean_\u2193"])
    gates = {
        "active_log_mel_relative_reduction_ge_0_05": mel_reduction >= 0.05,
        "dev_median_si_sdri_vs_raw_ge_5dB": float(best_waveform["si_sdri_vs_raw_mixture_db_\u2191"]["median"]) >= 5.0,
        "dev_positive_fraction_ge_0_85": float(best_waveform["si_sdri_vs_raw_mixture_db_\u2191"]["positive_fraction"]) >= 0.85,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one-shot source-disjoint semantic-preservation architecture screen",
        "semantic_audit_gate_pre_registered": "frozen separated top1 must exceed raw-mixture oracle-box top1 0.6026200652",
        "arguments": vars(args) | {
            "train_manifest": str(args.train_manifest.resolve()),
            "dev_manifest": str(args.dev_manifest.resolve()),
            "initial_checkpoint": str(args.initial_checkpoint.resolve()),
            "output_dir": str(output_dir),
        },
        "data": {"train_scenes": len(train_scenes), "dev_scenes": len(dev_scenes), "source_overlap": 0},
        "baseline": {"waveform": baseline_waveform, "log_mel": baseline_mel},
        "best_epoch": best_epoch,
        "best": {"waveform": best_waveform, "log_mel": best_mel_metrics, "relative_log_mel_reduction": mel_reduction},
        "history": history,
        "gates": gates,
        "decision": "run_frozen_semantic_gate" if all(gates.values()) else "semantic_safe_finetune_failed_do_not_audit_or_scale",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    atomic_json(receipt, output_dir / "training_receipt.json")
    print(json.dumps({"best_epoch": best_epoch, "gates": gates, "decision": receipt["decision"], "receipt": str(output_dir / "training_receipt.json")}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
