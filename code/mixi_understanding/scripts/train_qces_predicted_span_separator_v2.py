#!/usr/bin/env python3
"""Adapt the exact-span separator to frozen temporal-v2 proposal boundaries.

This sidecar keeps the original separator code/checkpoint untouched.  It is
initialized from the accepted exact-span model and trains only on proposals
materialized from the source-disjoint training split.  Validation always uses
predicted conditions; a small fraction of exact conditions is retained during
training to reduce catastrophic forgetting.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.scripts.train_qces_span_conditioned_separator_v1 import (
    CHECKPOINT_FORMAT,
    HOP,
    N_FFT,
    RATE,
    SpanMaskNetwork,
    _annotate_overlap,
    _load_mono,
    _negative_si_sdr,
    _separator_forward,
    _stft,
)


FORMAT = "qces_predicted_span_separator_training_receipt_v2"


def parse_args() -> argparse.Namespace:
    conditions = PROJECT_ROOT / "outputs/qces_predicted_span_conditions_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=conditions / "event_components_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=conditions / "event_components_dev.jsonl")
    parser.add_argument(
        "--initial-checkpoint", type=Path,
        default=PROJECT_ROOT / "outputs/qces_semantic_span_separator_full191_v1/best.pt",
    )
    parser.add_argument(
        "--label-embedding-cache", type=Path,
        default=PROJECT_ROOT / "outputs/qces_label_clap191_v1.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_predicted_span_separator_full191_v2",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-seconds", type=float, default=2.56)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--exact-condition-probability", type=float, default=0.25)
    parser.add_argument("--minimum-condition-iou", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2177)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PredictedConditionDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self, rows: Sequence[dict[str, Any]], *, chunk_samples: int, seed: int,
        train: bool, exact_probability: float,
    ) -> None:
        self.rows = list(rows)
        self.chunk_samples = int(chunk_samples)
        self.seed = int(seed)
        self.train = bool(train)
        self.exact_probability = float(exact_probability)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        mixture = _load_mono(row["mixture_path"])
        component = _load_mono(row["component_path"])
        onset, offset = int(row["onset_sample"]), int(row["offset_sample"])
        predicted_onset = int(row["condition_onset_sample"])
        predicted_offset = int(row["condition_offset_sample"])
        if offset - onset != component.numel():
            raise RuntimeError(f"{row['event_id']}: component/event length mismatch")
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        use_exact = self.train and rng.random() < self.exact_probability
        condition_onset = onset if use_exact else predicted_onset
        condition_offset = offset if use_exact else predicted_offset

        union_onset = min(onset, condition_onset)
        union_offset = max(offset, condition_offset)
        low = max(0, union_offset - self.chunk_samples)
        high = min(union_onset, max(0, mixture.numel() - self.chunk_samples))
        if high < low:
            raise RuntimeError(f"{row['event_id']}: target/condition union does not fit chunk")
        if self.train:
            chunk_start = rng.randint(low, high)
            gain_db = rng.uniform(-3.0, 3.0)
        else:
            chunk_start = (low + high) // 2
            gain_db = 0.0
        chunk_end = chunk_start + self.chunk_samples
        mixture_chunk = mixture[chunk_start:chunk_end]
        if mixture_chunk.numel() < self.chunk_samples:
            mixture_chunk = F.pad(mixture_chunk, (0, self.chunk_samples - mixture_chunk.numel()))
        target = torch.zeros(self.chunk_samples, dtype=torch.float32)
        target[onset - chunk_start: offset - chunk_start] = component
        gold_span = torch.zeros(self.chunk_samples, dtype=torch.float32)
        gold_span[onset - chunk_start: offset - chunk_start] = 1.0
        condition_span = torch.zeros(self.chunk_samples, dtype=torch.float32)
        condition_span[
            condition_onset - chunk_start: condition_offset - chunk_start
        ] = 1.0
        gain = 10.0 ** (gain_db / 20.0)
        return (
            mixture_chunk * gain,
            target * gain,
            condition_span,
            gold_span,
            torch.tensor(float(row["condition_iou"]), dtype=torch.float32),
            torch.tensor(int(row["label_id"]), dtype=torch.long),
        )


def _loss(
    model: SpanMaskNetwork, mixture: torch.Tensor, target: torch.Tensor,
    condition_span: torch.Tensor, gold_span: torch.Tensor, label_id: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction, predicted_spec, complex_mask, mixture_spec = _separator_forward(
        model, mixture, condition_span, label_id
    )
    target_spec = _stft(target)
    spectral_scale = target_spec.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    complex_error = F.smooth_l1_loss(
        torch.view_as_real(predicted_spec / spectral_scale),
        torch.view_as_real(target_spec / spectral_scale), beta=0.05,
    )
    log_magnitude = F.l1_loss(torch.log1p(predicted_spec.abs()), torch.log1p(target_spec.abs()))
    wave_scale = target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-3)
    waveform = F.smooth_l1_loss(prediction / wave_scale, target / wave_scale, beta=0.05)
    ideal_mask = target_spec / (mixture_spec + 1e-6)
    mask_weight = target_spec.abs().sqrt()
    mask_weight = mask_weight / mask_weight.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
    mask = (
        (torch.view_as_real(complex_mask) - torch.view_as_real(ideal_mask).clamp(-2.0, 2.0)).abs()
        * mask_weight[..., None]
    ).mean()
    energy = (
        (prediction.square().sum(-1) + 1e-8) / (target.square().sum(-1) + 1e-8)
    ).log().abs().mean()
    # The proposal is only a condition.  It must not redefine the target
    # boundary, so leakage is measured outside the true event support.
    outside = (
        (prediction * (1.0 - gold_span)).abs().sum(-1)
        / (target.abs().sum(-1) + 1e-6)
    ).mean()
    si_sdr = _negative_si_sdr(prediction.float(), target.float())
    total = (
        complex_error + log_magnitude + 0.5 * waveform + 0.20 * mask
        + 0.10 * energy + 0.10 * outside + 0.05 * si_sdr
    )
    return total, {
        "total": float(total.detach().cpu()), "complex": float(complex_error.detach().cpu()),
        "waveform": float(waveform.detach().cpu()), "outside": float(outside.detach().cpu()),
        "negative_si_sdr": float(si_sdr.detach().cpu()),
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()), "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)), "q90": float(np.quantile(array, 0.90)),
    }


def _subset(indices: Sequence[int], pred: Sequence[float], base: Sequence[float]) -> dict[str, Any]:
    predicted = [pred[index] for index in indices]
    baseline = [base[index] for index in indices]
    gain = [left - right for left, right in zip(predicted, baseline, strict=True)]
    return {
        "events": len(indices),
        "predicted_sd_sdr_db_↑": _summary(predicted),
        "predicted_span_window_sd_sdr_db_↑": _summary(baseline),
        "sd_sdri_over_predicted_window_db_↑": _summary(gain),
        "positive_rate_↑": float(np.mean(np.asarray(gain) > 0.0)),
    }


@torch.inference_mode()
def _validate(model: SpanMaskNetwork, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    predicted_sd: list[float] = []
    baseline_sd: list[float] = []
    ious: list[float] = []
    for mixture, target, condition, _gold, condition_iou, label_id in loader:
        mixture, target = mixture.to(device), target.to(device)
        condition, label_id = condition.to(device), label_id.to(device)
        prediction, _, _, _ = _separator_forward(model, mixture, condition, label_id)
        predicted_sd.extend(scale_dependent_sdr(prediction, target).cpu().tolist())
        baseline_sd.extend(scale_dependent_sdr(mixture * condition, target).cpu().tolist())
        ious.extend(condition_iou.tolist())
    all_indices = list(range(len(ious)))
    iou50 = [index for index, value in enumerate(ious) if value >= 0.50]
    iou70 = [index for index, value in enumerate(ious) if value >= 0.70]
    return {
        "events": len(ious),
        "all_matched_iou30": _subset(all_indices, predicted_sd, baseline_sd),
        "matched_iou50": _subset(iou50, predicted_sd, baseline_sd),
        "matched_iou70": _subset(iou70, predicted_sd, baseline_sd),
        "condition_iou": _summary(ious),
    }


def _key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    subset = metrics["matched_iou50"]
    gain = subset["sd_sdri_over_predicted_window_db_↑"]
    return float(gain["median"]), float(subset["positive_rate_↑"]), float(gain["mean"])


def _save(path: Path, model: SpanMaskNetwork, epoch: int, metrics: Mapping[str, Any], config: Mapping[str, Any]):
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "format": CHECKPOINT_FORMAT, "epoch": epoch, "model_state": model.state_dict(),
        "metrics": dict(metrics), "config": dict(config),
    }, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    _seed(args.seed)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_all = _annotate_overlap(_read_jsonl(args.train_manifest.resolve()))
    dev_all = _annotate_overlap(_read_jsonl(args.dev_manifest.resolve()))
    train_rows = [
        row for row in train_all
        if bool(row.get("condition_available")) and float(row["condition_iou"]) >= args.minimum_condition_iou
    ]
    dev_rows = [
        row for row in dev_all
        if bool(row.get("condition_available")) and float(row["condition_iou"]) >= args.minimum_condition_iou
    ]
    chunk_samples = int(round(args.chunk_seconds * RATE))
    train_dataset = PredictedConditionDataset(
        train_rows, chunk_samples=chunk_samples, seed=args.seed, train=True,
        exact_probability=args.exact_condition_probability,
    )
    dev_dataset = PredictedConditionDataset(
        dev_rows, chunk_samples=chunk_samples, seed=args.seed + 1, train=False, exact_probability=0.0,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0,
    )
    cache = torch.load(args.label_embedding_cache.resolve(), map_location="cpu", weights_only=True)
    initial = torch.load(args.initial_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if initial.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("invalid initial separator checkpoint")
    initial_config = initial["config"]
    model = SpanMaskNetwork(
        base_channels=int(initial_config["base_channels"]),
        semantic_channels=int(initial_config["semantic_channels"]),
        label_embeddings=cache["embeddings"].float(),
    ).to(device)
    model.load_state_dict(initial["model_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    config = dict(initial_config)
    config.update({
        "conditioning": "frozen_clap_text_label_plus_temporal_v2_predicted_span",
        "training_condition_mix": {
            "predicted_probability": 1.0 - args.exact_condition_probability,
            "exact_probability": args.exact_condition_probability,
        },
        "initial_checkpoint_sha256": _sha256(args.initial_checkpoint.resolve()),
        "selection": "matched predicted spans IoU>=0.50: median SD-SDRi over matching window",
        "acceptance_gate": {"median_gain_db_min": 1.0, "positive_rate_min": 0.60},
    })
    history: list[dict[str, Any]] = []
    initial_metrics = _validate(model, dev_loader, device)
    best_key = _key(initial_metrics)
    best_epoch = 0
    _save(output / "best.pt", model, 0, initial_metrics, config)
    print(json.dumps({"epoch": 0, "validation": initial_metrics}), flush=True)
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        sums: dict[str, float] = {}
        steps = 0
        for step, (mixture, target, condition, gold, _iou, label_id) in enumerate(train_loader, 1):
            mixture, target = mixture.to(device), target.to(device)
            condition, gold, label_id = condition.to(device), gold.to(device), label_id.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss, terms = _loss(model, mixture, target, condition, gold, label_id)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + value
            if step % 250 == 0:
                print(json.dumps({
                    "epoch": epoch, "step": step, "steps": len(train_loader),
                    "mean_train_loss": sums["total"] / steps,
                }), flush=True)
        validation = _validate(model, dev_loader, device)
        key = _key(validation)
        scheduler.step(key[0])
        record = {
            "epoch": epoch, "train": {name: value / steps for name, value in sums.items()},
            "validation": validation, "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            _save(output / "best.pt", model, epoch, validation, config)
        else:
            stale += 1
        if stale >= args.patience:
            break
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    best_metrics = best["metrics"]
    gate_subset = best_metrics["matched_iou50"]
    accepted = (
        float(gate_subset["sd_sdri_over_predicted_window_db_↑"]["median"]) >= 1.0
        and float(gate_subset["positive_rate_↑"]) >= 0.60
    )
    receipt = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "best_epoch": best_epoch, "accepted_predicted_span_gate": accepted,
        "train_events": len(train_rows), "dev_events": len(dev_rows), "config": config,
        "best_validation": best_metrics, "history": history,
        "best_checkpoint": str((output / "best.pt").resolve()),
        "best_checkpoint_sha256": _sha256(output / "best.pt"),
        "train_manifest_sha256": _sha256(args.train_manifest.resolve()),
        "dev_manifest_sha256": _sha256(args.dev_manifest.resolve()),
        "decision": (
            "candidate_for_full_integration_and_oracle-retention_check" if accepted
            else "rejected_by_predeclared_predicted-span_gate"
        ),
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "complete": True, "best_epoch": best_epoch, "accepted": accepted,
        "best_validation": best_metrics,
    }), flush=True)


if __name__ == "__main__":
    main()
