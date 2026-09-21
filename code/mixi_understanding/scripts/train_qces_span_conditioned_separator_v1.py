#!/usr/bin/env python3
"""Train a span-conditioned complex-mask separator on exact event stems.

This is deliberately an oracle-span falsification stage.  It asks one narrow
question: after the target interval is known, can a learned separator remove
polyphonic interference better than merely windowing the mixture?  Parser,
label selection, and predicted-boundary errors are excluded from this stage.
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
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.scripts.train_vietnamese_automotive_tfmask import TFMaskUNet


FORMAT = "qces_span_conditioned_separator_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_span_conditioned_separator_checkpoint_v1"
RATE = 16_000
N_FFT = 512
HOP = 128


def parse_args() -> argparse.Namespace:
    components = PROJECT_ROOT / "outputs/qces_full191_overlap_components_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=components / "event_components_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=components / "event_components_dev.jsonl")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_span_conditioned_separator_v1",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-seconds", type=float, default=2.56)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument(
        "--label-embedding-cache", type=Path, default=None,
        help="Optional frozen 191x512 CLAP text matrix; enables semantic+span conditioning.",
    )
    parser.add_argument("--semantic-channels", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-train-events", type=int, default=0)
    parser.add_argument("--max-dev-events", type=int, default=640)
    parser.add_argument(
        "--overfit-events", type=int, default=0,
        help="Diagnostic only: use the same stratified train rows for training and validation.",
    )
    parser.add_argument("--seed", type=int, default=2101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_mono(path: str | Path) -> torch.Tensor:
    value, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if int(sample_rate) != RATE:
        raise RuntimeError(f"{path}: expected {RATE} Hz, got {sample_rate}")
    return torch.from_numpy(value.mean(axis=1)).float()


def _stratified_limit(rows: Sequence[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)
    rng = random.Random(seed)
    for values in by_label.values():
        rng.shuffle(values)
    labels = sorted(by_label)
    selected: list[dict[str, Any]] = []
    cursor = 0
    while len(selected) < limit:
        label = labels[cursor % len(labels)]
        bucket = by_label[label]
        index = cursor // len(labels)
        if index < len(bucket):
            selected.append(bucket[index])
        cursor += 1
        if cursor > limit * len(labels) * 2:
            break
    if len(selected) < limit:
        used = {str(row["event_id"]) for row in selected}
        remainder = [row for row in rows if str(row["event_id"]) not in used]
        rng.shuffle(remainder)
        selected.extend(remainder[: limit - len(selected)])
    return selected[:limit]


def _annotate_overlap(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the fraction of each target interval covered by another event."""
    by_scene: dict[str, list[dict[str, Any]]] = {}
    copied = [dict(row) for row in rows]
    for row in copied:
        by_scene.setdefault(str(row["scene_id"]), []).append(row)
    for scene_rows in by_scene.values():
        for row in scene_rows:
            onset, offset = int(row["onset_sample"]), int(row["offset_sample"])
            intersections = sorted(
                (max(onset, int(other["onset_sample"])), min(offset, int(other["offset_sample"])))
                for other in scene_rows
                if other is not row
                and max(onset, int(other["onset_sample"])) < min(offset, int(other["offset_sample"]))
            )
            merged: list[list[int]] = []
            for begin, end in intersections:
                if not merged or begin > merged[-1][1]:
                    merged.append([begin, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            overlap_samples = sum(end - begin for begin, end in merged)
            row["_overlap_fraction"] = overlap_samples / max(offset - onset, 1)
    return copied


class EventComponentDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
):
    def __init__(
        self, rows: Sequence[dict[str, Any]], *, chunk_samples: int, seed: int, train: bool,
    ) -> None:
        self.rows = list(rows)
        self.chunk_samples = int(chunk_samples)
        self.seed = int(seed)
        self.train = bool(train)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self, index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        mixture = _load_mono(row["mixture_path"])
        component = _load_mono(row["component_path"])
        onset = int(row["onset_sample"])
        offset = int(row["offset_sample"])
        if offset - onset != component.numel():
            raise RuntimeError(f"{row['event_id']}: component length does not match event interval")
        if component.numel() > self.chunk_samples:
            raise RuntimeError(f"{row['event_id']}: event is longer than training chunk")

        low = max(0, offset - self.chunk_samples)
        high = min(onset, max(0, mixture.numel() - self.chunk_samples))
        if high < low:
            raise RuntimeError(f"{row['event_id']}: cannot fit event in requested chunk")
        if self.train:
            rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
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
        local_onset = onset - chunk_start
        local_offset = local_onset + component.numel()
        target[local_onset:local_offset] = component
        span = torch.zeros(self.chunk_samples, dtype=torch.float32)
        span[local_onset:local_offset] = 1.0
        gain = 10.0 ** (gain_db / 20.0)
        return (
            mixture_chunk * gain, target * gain, span,
            torch.tensor(float(row.get("_overlap_fraction", 0.0)), dtype=torch.float32),
            torch.tensor(int(row["label_id"]), dtype=torch.long),
        )


class SpanMaskNetwork(nn.Module):
    """TF-mask U-Net with optional frozen CLAP semantic conditioning."""

    def __init__(
        self, *, base_channels: int, semantic_channels: int,
        label_embeddings: torch.Tensor | None,
    ) -> None:
        super().__init__()
        if label_embeddings is None:
            self.register_buffer("label_embeddings", None)
            self.semantic_projector = None
            input_channels = 4
        else:
            if tuple(label_embeddings.shape) != (191, 512):
                raise ValueError(f"expected 191x512 label embeddings, got {tuple(label_embeddings.shape)}")
            self.register_buffer("label_embeddings", label_embeddings.float().contiguous())
            self.semantic_projector = nn.Sequential(
                nn.Linear(512, 64), nn.GELU(), nn.Linear(64, semantic_channels),
            )
            input_channels = 4 + semantic_channels
        self.unet = TFMaskUNet(
            base=base_channels, input_channels=input_channels, output_channels=2,
            activation="identity",
        )

    @property
    def output_layer(self) -> nn.Conv2d:
        return self.unet.output

    def forward(self, feature: torch.Tensor, label_id: torch.Tensor) -> torch.Tensor:
        if self.semantic_projector is not None:
            assert self.label_embeddings is not None
            semantic = self.semantic_projector(self.label_embeddings[label_id])
            semantic = semantic[:, :, None, None].expand(
                -1, -1, feature.shape[-2], feature.shape[-1]
            )
            feature = torch.cat((feature, semantic), dim=1)
        return self.unet(feature)


def _stft(wave: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(N_FFT, device=wave.device, dtype=wave.dtype)
    return torch.stft(wave, N_FFT, HOP, N_FFT, window, return_complex=True)


def _istft(spec: torch.Tensor, length: int) -> torch.Tensor:
    window = torch.hann_window(N_FFT, device=spec.device, dtype=spec.real.dtype)
    return torch.istft(spec, N_FFT, HOP, N_FFT, window, length=length)


def _separator_forward(
    model: SpanMaskNetwork, mixture: torch.Tensor, sample_span: torch.Tensor,
    label_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mixture_spec = _stft(mixture)
    magnitude = mixture_spec.abs().clamp_min(1e-7)
    unit = mixture_spec / magnitude
    frame_span = F.interpolate(
        sample_span[:, None], size=mixture_spec.shape[-1], mode="nearest"
    )[:, 0]
    span_image = frame_span[:, None].expand(-1, mixture_spec.shape[-2], -1)
    feature = torch.stack(
        (torch.log1p(magnitude), unit.real, unit.imag, span_image), dim=1
    )
    # Learn a bounded correction around the safe oracle-span baseline.  With a
    # zero-initialized output layer the initial system is exactly mixture ×
    # span, rather than an arbitrary destructive mask.
    residual_channels = torch.tanh(model(feature, label_id))
    residual_mask = torch.complex(
        residual_channels[:, 0].float(), residual_channels[:, 1].float()
    )
    complex_mask = torch.complex(span_image.float(), torch.zeros_like(span_image).float()) + residual_mask
    residual_spec = mixture_spec * residual_mask
    prediction = mixture * sample_span + _istft(residual_spec, mixture.shape[-1])
    predicted_spec = _stft(prediction)
    return prediction, predicted_spec, complex_mask, mixture_spec


def _negative_si_sdr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (
        (prediction * target).sum(-1, keepdim=True)
        * target / (target.square().sum(-1, keepdim=True) + 1e-8)
    )
    noise = prediction - projection
    return -(
        10.0 * torch.log10(
            (projection.square().sum(-1) + 1e-8) / (noise.square().sum(-1) + 1e-8)
        )
    ).mean()


def _loss(
    model: SpanMaskNetwork, mixture: torch.Tensor, target: torch.Tensor, span: torch.Tensor,
    label_id: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction, predicted_spec, complex_mask, mixture_spec = _separator_forward(
        model, mixture, span, label_id
    )
    target_spec = _stft(target)
    spectral_scale = target_spec.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-3)
    complex_error = F.smooth_l1_loss(
        torch.view_as_real(predicted_spec / spectral_scale),
        torch.view_as_real(target_spec / spectral_scale),
        beta=0.05,
    )
    log_magnitude = F.l1_loss(
        torch.log1p(predicted_spec.abs()), torch.log1p(target_spec.abs())
    )
    wave_scale = target.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-3)
    waveform = F.smooth_l1_loss(prediction / wave_scale, target / wave_scale, beta=0.05)
    ideal_mask = target_spec / (mixture_spec + 1e-6)
    ideal_mask_ri = torch.view_as_real(ideal_mask).clamp(-2.0, 2.0)
    predicted_mask_ri = torch.view_as_real(complex_mask)
    mask_weight = target_spec.abs().sqrt()
    mask_weight = mask_weight / mask_weight.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
    mask = (
        (predicted_mask_ri - ideal_mask_ri).abs() * mask_weight[..., None]
    ).mean()
    energy_ratio = (
        (prediction.square().sum(-1) + 1e-8) / (target.square().sum(-1) + 1e-8)
    )
    energy = energy_ratio.log().abs().mean()
    outside = (
        (prediction * (1.0 - span)).abs().sum(-1)
        / (target.abs().sum(-1) + 1e-6)
    ).mean()
    si_sdr = _negative_si_sdr(prediction.float(), target.float())
    total = (
        1.0 * complex_error
        + 1.0 * log_magnitude
        + 0.5 * waveform
        + 0.20 * mask
        + 0.10 * energy
        + 0.10 * outside
        + 0.05 * si_sdr
    )
    terms = {
        "total": float(total.detach().cpu()),
        "complex": float(complex_error.detach().cpu()),
        "log_magnitude": float(log_magnitude.detach().cpu()),
        "waveform": float(waveform.detach().cpu()),
        "mask": float(mask.detach().cpu()),
        "energy": float(energy.detach().cpu()),
        "outside": float(outside.detach().cpu()),
        "negative_si_sdr": float(si_sdr.detach().cpu()),
    }
    return total, terms


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _metric_subset(
    *, indices: Sequence[int], predicted_sd: Sequence[float], baseline_sd: Sequence[float],
    predicted_si: Sequence[float], baseline_si: Sequence[float], energy_db: Sequence[float],
) -> dict[str, Any]:
    if not indices:
        raise RuntimeError("metric subset is empty")
    p_sd = [predicted_sd[index] for index in indices]
    b_sd = [baseline_sd[index] for index in indices]
    p_si = [predicted_si[index] for index in indices]
    b_si = [baseline_si[index] for index in indices]
    selected_energy = [energy_db[index] for index in indices]
    sd_improvement = [p - b for p, b in zip(p_sd, b_sd, strict=True)]
    si_improvement = [p - b for p, b in zip(p_si, b_si, strict=True)]
    return {
        "events": len(indices),
        "predicted_sd_sdr_db_↑": _summary(p_sd),
        "oracle_span_mixture_sd_sdr_db_↑": _summary(b_sd),
        "sd_sdri_over_oracle_span_db_↑": _summary(sd_improvement),
        "sd_sdri_over_oracle_span_positive_rate_↑": float(np.mean(np.asarray(sd_improvement) > 0.0)),
        "predicted_si_sdr_db_↑": _summary(p_si),
        "si_sdri_over_oracle_span_db_↑": _summary(si_improvement),
        "predicted_to_target_energy_db_abs_↓": float(np.mean(np.abs(np.asarray(selected_energy)))),
    }


@torch.inference_mode()
def _validate(
    model: SpanMaskNetwork, loader: DataLoader, device: torch.device,
) -> dict[str, Any]:
    model.eval()
    predicted_sd: list[float] = []
    baseline_sd: list[float] = []
    predicted_si: list[float] = []
    baseline_si: list[float] = []
    energy_db: list[float] = []
    overlap_fractions: list[float] = []
    for mixture, target, span, overlap_fraction, label_id in loader:
        mixture = mixture.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        span = span.to(device, non_blocking=True)
        label_id = label_id.to(device, non_blocking=True)
        prediction, _, _, _ = _separator_forward(model, mixture, span, label_id)
        baseline = mixture * span
        predicted_sd.extend(scale_dependent_sdr(prediction, target).cpu().tolist())
        baseline_sd.extend(scale_dependent_sdr(baseline, target).cpu().tolist())
        predicted_si.extend(scale_invariant_sdr(prediction, target).cpu().tolist())
        baseline_si.extend(scale_invariant_sdr(baseline, target).cpu().tolist())
        overlap_fractions.extend(overlap_fraction.tolist())
        energy_db.extend(
            (10.0 * torch.log10(
                (prediction.square().sum(-1) + 1e-8) / (target.square().sum(-1) + 1e-8)
            )).cpu().tolist()
        )
    all_indices = list(range(len(predicted_sd)))
    overlapped = [index for index, value in enumerate(overlap_fractions) if value > 0.0]
    high_overlap = [index for index, value in enumerate(overlap_fractions) if value >= 0.50]
    return {
        "events": len(predicted_sd),
        "all_events": _metric_subset(
            indices=all_indices, predicted_sd=predicted_sd, baseline_sd=baseline_sd,
            predicted_si=predicted_si, baseline_si=baseline_si, energy_db=energy_db,
        ),
        "overlapped_events": _metric_subset(
            indices=overlapped, predicted_sd=predicted_sd, baseline_sd=baseline_sd,
            predicted_si=predicted_si, baseline_si=baseline_si, energy_db=energy_db,
        ),
        "high_overlap_events": _metric_subset(
            indices=high_overlap, predicted_sd=predicted_sd, baseline_sd=baseline_sd,
            predicted_si=predicted_si, baseline_si=baseline_si, energy_db=energy_db,
        ),
        "overlap_fraction": _summary(overlap_fractions),
    }


def _selection_key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    overlap_metrics = metrics["overlapped_events"]
    improvement = overlap_metrics["sd_sdri_over_oracle_span_db_↑"]
    return (
        float(improvement["median"]),
        float(overlap_metrics["sd_sdri_over_oracle_span_positive_rate_↑"]),
        float(improvement["mean"]),
    )


def _save_checkpoint(
    path: Path, model: nn.Module, *, epoch: int, metrics: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "metrics": dict(metrics),
            "config": dict(config),
        },
        temporary,
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_rows_all = _annotate_overlap(_read_jsonl(args.train_manifest.resolve()))
    dev_rows_all = _annotate_overlap(_read_jsonl(args.dev_manifest.resolve()))
    train_rows = _stratified_limit(train_rows_all, args.max_train_events, args.seed)
    dev_rows = _stratified_limit(dev_rows_all, args.max_dev_events, args.seed + 1)
    if args.overfit_events > 0:
        train_rows = _stratified_limit(train_rows_all, args.overfit_events, args.seed)
        dev_rows = list(train_rows)
    chunk_samples = int(round(args.chunk_seconds * RATE))
    train_dataset = EventComponentDataset(
        train_rows, chunk_samples=chunk_samples, seed=args.seed, train=True,
    )
    dev_dataset = EventComponentDataset(
        dev_rows, chunk_samples=chunk_samples, seed=args.seed + 1, train=False,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    label_embeddings: torch.Tensor | None = None
    label_cache_sha256: str | None = None
    if args.label_embedding_cache is not None:
        label_cache_path = args.label_embedding_cache.resolve()
        label_cache = torch.load(label_cache_path, map_location="cpu", weights_only=True)
        if label_cache.get("format") != "qces_label_clap191_v1":
            raise ValueError(f"unexpected label embedding cache format: {label_cache.get('format')}")
        label_embeddings = label_cache["embeddings"].float()
        label_cache_sha256 = _sha256_file(label_cache_path)
    model = SpanMaskNetwork(
        base_channels=args.base_channels,
        semantic_channels=args.semantic_channels,
        label_embeddings=label_embeddings,
    ).to(device)
    nn.init.zeros_(model.output_layer.weight)
    if model.output_layer.bias is not None:
        nn.init.zeros_(model.output_layer.bias)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    config = {
        "rate": RATE,
        "n_fft": N_FFT,
        "hop": HOP,
        "chunk_seconds": args.chunk_seconds,
        "base_channels": args.base_channels,
        "conditioning": "frozen_clap_text_label_plus_oracle_span" if label_embeddings is not None else "oracle_span_only",
        "semantic_channels": args.semantic_channels if label_embeddings is not None else 0,
        "label_embedding_cache_sha256": label_cache_sha256,
        "diagnostic_overfit_events": args.overfit_events,
        "input_channels": ["log_magnitude", "unit_real", "unit_imag", "oracle_span"],
        "output": "oracle_span_complex_mask_plus_bounded_tanh_residual",
        "selection": "overlapped events only: median SD-SDRi over hard oracle-span mixture, then positive rate, then mean",
        "acceptance_gate": {
            "median_sd_sdri_over_oracle_span_db_min": 2.0,
            "positive_rate_min": 0.60,
        },
    }
    history: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf, -math.inf)
    best_epoch = 0
    stale = 0

    initial = _validate(model, dev_loader, device)
    best_key = _selection_key(initial)
    _save_checkpoint(
        output_dir / "best.pt", model, epoch=0, metrics=initial, config=config,
    )
    print(json.dumps({"epoch": 0, "validation": initial}, ensure_ascii=False), flush=True)
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        term_sums: dict[str, float] = {}
        steps = 0
        for step, (mixture, target, span, _overlap_fraction, label_id) in enumerate(train_loader, 1):
            mixture = mixture.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            span = span.to(device, non_blocking=True)
            label_id = label_id.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss, terms = _loss(model, mixture, target, span, label_id)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            for name, value in terms.items():
                term_sums[name] = term_sums.get(name, 0.0) + value
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch, "step": step, "steps": len(train_loader),
                            "mean_train_loss": term_sums["total"] / steps,
                        }
                    ),
                    flush=True,
                )
        validation = _validate(model, dev_loader, device)
        key = _selection_key(validation)
        scheduler.step(key[0])
        record = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": {name: value / max(steps, 1) for name, value in term_sums.items()},
            "validation": validation,
            "selection_key": list(key),
        }
        history.append(record)
        _atomic_json(output_dir / "history.json", {"history": history})
        _save_checkpoint(
            output_dir / "last.pt", model, epoch=epoch, metrics=validation, config=config,
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            stale = 0
            _save_checkpoint(
                output_dir / "best.pt", model, epoch=epoch, metrics=validation, config=config,
            )
        else:
            stale += 1
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    best_path = output_dir / "best.pt"
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    best_metrics = best["metrics"]
    overlap_best = best_metrics["overlapped_events"]
    median_gain = float(overlap_best["sd_sdri_over_oracle_span_db_↑"]["median"])
    positive_rate = float(overlap_best["sd_sdri_over_oracle_span_positive_rate_↑"])
    accepted = median_gain >= 2.0 and positive_rate >= 0.60
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "accepted_by_predeclared_gate": accepted,
        "decision": "keep_for_predicted_span_evaluation" if accepted else "reject_before_pipeline_integration",
        "best_epoch": best_epoch,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": _sha256_file(best_path),
        "train_manifest_sha256": _sha256_file(args.train_manifest.resolve()),
        "dev_manifest_sha256": _sha256_file(args.dev_manifest.resolve()),
        "label_embedding_cache_sha256": label_cache_sha256,
        "train_events_available": len(train_rows_all),
        "train_events_used": len(train_rows),
        "dev_events_available": len(dev_rows_all),
        "dev_events_used_for_selection": len(dev_rows),
        "config": config,
        "best_validation": best_metrics,
    }
    _atomic_json(output_dir / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
