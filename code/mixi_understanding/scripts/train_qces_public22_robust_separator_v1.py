#!/usr/bin/env python3
"""Adapt the accepted semantic separator to noisy 22-label temporal conditions.

Unlike the rejected predicted-span v2 run, training sees exact, predicted and
jittered+dilated conditions.  Validation is always prediction-only with a
fixed conservative dilation.  The accepted 191-label separator weights are
kept as initialization while its frozen CLAP label bank is replaced by the
exact public-22 text bank; all source/proxy strata are reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from mixi_understanding.qces.metrics import scale_dependent_sdr
from mixi_understanding.scripts.train_qces_span_conditioned_separator_v1 import (
    CHECKPOINT_FORMAT,
    RATE,
    SpanMaskNetwork,
    _load_mono,
    _negative_si_sdr,
    _separator_forward,
    _stft,
)


FORMAT = "qces_public22_robust_separator_training_v1"


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_public22_separator_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=data / "event_components_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=data / "event_components_dev.jsonl")
    parser.add_argument(
        "--initial-checkpoint", type=Path,
        default=PROJECT_ROOT / "outputs/qces_semantic_span_separator_full191_v1/best.pt",
    )
    parser.add_argument(
        "--initial-label-cache", type=Path,
        default=PROJECT_ROOT / "outputs/qces_label_clap191_v1.pt",
    )
    parser.add_argument(
        "--label-embedding-cache", type=Path,
        default=PROJECT_ROOT / "outputs/qces_label_clap_public22_v1.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_public22_robust_separator_v1",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--chunk-seconds", type=float, default=2.56)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--exact-probability", type=float, default=0.15)
    parser.add_argument("--predicted-probability", type=float, default=0.55)
    parser.add_argument("--jittered-probability", type=float, default=0.30)
    parser.add_argument("--minimum-condition-iou", type=float, default=0.30)
    parser.add_argument("--train-pad-min-seconds", type=float, default=0.08)
    parser.add_argument("--train-pad-max-seconds", type=float, default=0.24)
    parser.add_argument("--validation-pad-seconds", type=float, default=0.15)
    parser.add_argument("--maximum-jitter-seconds", type=float, default=0.16)
    parser.add_argument("--soft-edge-seconds", type=float, default=0.04)
    parser.add_argument("--max-train-events", type=int, default=0)
    parser.add_argument("--max-dev-events", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2247)
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


def _stratified_limit(rows: Sequence[dict[str, Any]], limit: int, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["label"]), []).append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for label in sorted(buckets):
            if buckets[label] and len(selected) < limit:
                selected.append(buckets[label].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _soft_span(length: int, onset: int, offset: int, edge: int) -> torch.Tensor:
    span = torch.zeros(length, dtype=torch.float32)
    onset = max(0, min(length - 1, int(onset)))
    offset = max(onset + 1, min(length, int(offset)))
    span[onset:offset] = 1.0
    edge = min(int(edge), onset, length - offset)
    if edge > 0:
        ramp = 0.5 - 0.5 * torch.cos(torch.linspace(0.0, torch.pi, edge + 2)[1:-1])
        span[onset - edge:onset] = ramp
        span[offset:offset + edge] = torch.flip(ramp, dims=(0,))
    return span


def _iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / max(union, 1)


class RobustConditionDataset(Dataset):
    def __init__(
        self, rows: Sequence[dict[str, Any]], *, chunk_samples: int, seed: int,
        train: bool, exact_probability: float, predicted_probability: float,
        pad_min: int, pad_max: int, validation_pad: int, maximum_jitter: int,
        soft_edge: int,
    ) -> None:
        self.rows = list(rows)
        self.chunk_samples = int(chunk_samples)
        self.seed = int(seed)
        self.train = bool(train)
        self.exact_probability = float(exact_probability)
        self.predicted_probability = float(predicted_probability)
        self.pad_min = int(pad_min)
        self.pad_max = int(pad_max)
        self.validation_pad = int(validation_pad)
        self.maximum_jitter = int(maximum_jitter)
        self.soft_edge = int(soft_edge)
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
        if offset - onset != component.numel():
            raise RuntimeError(f"{row['event_id']}: component/event length mismatch")
        predicted_onset = int(row["condition_onset_sample"])
        predicted_offset = int(row["condition_offset_sample"])
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        if self.train:
            draw = rng.random()
            if draw < self.exact_probability:
                mode = 0  # exact
                condition_onset, condition_offset = onset, offset
            elif draw < self.exact_probability + self.predicted_probability:
                mode = 1  # predicted+dilated
                pad_left = rng.randint(self.pad_min, self.pad_max)
                pad_right = rng.randint(self.pad_min, self.pad_max)
                condition_onset = predicted_onset - pad_left
                condition_offset = predicted_offset + pad_right
            else:
                mode = 2  # predicted+jittered+dilated
                shift = rng.randint(-self.maximum_jitter, self.maximum_jitter)
                pad_left = rng.randint(self.pad_min, self.pad_max)
                pad_right = rng.randint(self.pad_min, self.pad_max)
                condition_onset = predicted_onset + shift - pad_left
                condition_offset = predicted_offset + shift + pad_right
        else:
            mode = 1
            condition_onset = predicted_onset - self.validation_pad
            condition_offset = predicted_offset + self.validation_pad
        condition_onset = max(0, condition_onset)
        condition_offset = min(mixture.numel(), max(condition_onset + 1, condition_offset))
        union_onset = min(onset, condition_onset)
        union_offset = max(offset, condition_offset)
        low = max(0, union_offset - self.chunk_samples)
        high = min(union_onset, max(0, mixture.numel() - self.chunk_samples))
        if high < low:
            # Very wide proposal: retain the entire gold event and symmetrically
            # cap the condition to what one training chunk can represent.
            center = (onset + offset) // 2
            half = self.chunk_samples // 2
            condition_onset = max(0, center - half)
            condition_offset = min(mixture.numel(), condition_onset + self.chunk_samples)
            low = max(0, max(offset, condition_offset) - self.chunk_samples)
            high = min(min(onset, condition_onset), max(0, mixture.numel() - self.chunk_samples))
        if high < low:
            raise RuntimeError(f"{row['event_id']}: target/condition union does not fit chunk")
        chunk_start = rng.randint(low, high) if self.train else (low + high) // 2
        chunk_end = chunk_start + self.chunk_samples
        mixture_chunk = mixture[chunk_start:chunk_end]
        if mixture_chunk.numel() < self.chunk_samples:
            mixture_chunk = F.pad(mixture_chunk, (0, self.chunk_samples - mixture_chunk.numel()))
        target = torch.zeros(self.chunk_samples, dtype=torch.float32)
        target[onset - chunk_start:offset - chunk_start] = component
        gold = torch.zeros(self.chunk_samples, dtype=torch.float32)
        gold[onset - chunk_start:offset - chunk_start] = 1.0
        condition = _soft_span(
            self.chunk_samples,
            condition_onset - chunk_start,
            condition_offset - chunk_start,
            self.soft_edge,
        )
        gain_db = rng.uniform(-3.0, 3.0) if self.train else 0.0
        gain = 10.0 ** (gain_db / 20.0)
        dynamic_iou = _iou((onset, offset), (condition_onset, condition_offset))
        is_proxy = int(str(row.get("semantic_supervision")) != "exact")
        return (
            mixture_chunk * gain,
            target * gain,
            condition,
            gold,
            torch.tensor(dynamic_iou, dtype=torch.float32),
            torch.tensor(int(row["label_id"]), dtype=torch.long),
            torch.tensor(is_proxy, dtype=torch.long),
            torch.tensor(mode, dtype=torch.long),
        )


def _loss(
    model: SpanMaskNetwork, mixture: torch.Tensor, target: torch.Tensor,
    condition: torch.Tensor, gold: torch.Tensor, label_id: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction, predicted_spec, complex_mask, mixture_spec = _separator_forward(
        model, mixture, condition, label_id
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
    active = (
        ((prediction - target).abs() * gold).sum(-1)
        / (target.abs().sum(-1) + 1e-6)
    ).mean()
    leakage = (
        ((prediction * (1.0 - gold)).square().sum(-1) + 1e-8)
        / (target.square().sum(-1) + 1e-8)
    ).mean()
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
    si_sdr = _negative_si_sdr(prediction.float(), target.float())
    total = (
        complex_error + log_magnitude + 0.50 * waveform + 0.20 * mask
        + 0.20 * active + 0.15 * leakage + 0.10 * energy + 0.05 * si_sdr
    )
    return total, {
        "total": float(total.detach().cpu()),
        "complex": float(complex_error.detach().cpu()),
        "waveform": float(waveform.detach().cpu()),
        "active_recall_error": float(active.detach().cpu()),
        "leakage_energy_ratio": float(leakage.detach().cpu()),
        "negative_si_sdr": float(si_sdr.detach().cpu()),
    }


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": float("nan"), "median": float("nan"), "q10": float("nan"), "q90": float("nan")}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.10)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _subset(indices: Sequence[int], predicted: Sequence[float], baseline: Sequence[float]) -> dict[str, Any]:
    gains = [predicted[index] - baseline[index] for index in indices]
    return {
        "events": len(indices),
        "predicted_sd_sdr_db_↑": _summary([predicted[index] for index in indices]),
        "crop_baseline_sd_sdr_db_↑": _summary([baseline[index] for index in indices]),
        "sd_sdri_over_crop_db_↑": _summary(gains),
        "positive_rate_↑": float(np.mean(np.asarray(gains) > 0.0)) if gains else float("nan"),
    }


@torch.inference_mode()
def _validate(
    model: SpanMaskNetwork, loader: DataLoader, device: torch.device, labels: Sequence[str]
) -> dict[str, Any]:
    model.eval()
    predicted: list[float] = []
    baseline: list[float] = []
    ious: list[float] = []
    label_ids: list[int] = []
    proxy_flags: list[int] = []
    for mixture, target, condition, _gold, condition_iou, label_id, proxy, _mode in loader:
        mixture, target = mixture.to(device), target.to(device)
        condition, label_id_gpu = condition.to(device), label_id.to(device)
        prediction, _, _, _ = _separator_forward(model, mixture, condition, label_id_gpu)
        predicted.extend(scale_dependent_sdr(prediction, target).cpu().tolist())
        baseline.extend(scale_dependent_sdr(mixture * condition, target).cpu().tolist())
        ious.extend(condition_iou.tolist())
        label_ids.extend(label_id.tolist())
        proxy_flags.extend(proxy.tolist())
    indices = list(range(len(predicted)))
    exact = [i for i in indices if proxy_flags[i] == 0]
    proxy = [i for i in indices if proxy_flags[i] == 1]
    by_label = {
        labels[label_id]: _subset(
            [i for i in indices if label_ids[i] == label_id], predicted, baseline
        )
        for label_id in sorted(set(label_ids))
    }
    return {
        "events": len(indices),
        "all": _subset(indices, predicted, baseline),
        "exact_semantics": _subset(exact, predicted, baseline),
        "proxy_semantics": _subset(proxy, predicted, baseline),
        "condition_iou": _summary(ious),
        "per_label": by_label,
    }


def _key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    exact = metrics["exact_semantics"]
    gain = exact["sd_sdri_over_crop_db_↑"]
    return float(gain["median"]), float(exact["positive_rate_↑"]), float(gain["mean"])


def _save(
    path: Path, model: SpanMaskNetwork, epoch: int,
    metrics: Mapping[str, Any], config: Mapping[str, Any], labels: Sequence[str],
) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save({
        "format": CHECKPOINT_FORMAT,
        "training_format": FORMAT,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "metrics": dict(metrics),
        "config": dict(config),
        "labels": list(labels),
    }, temporary)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    _seed(args.seed)
    probability_sum = args.exact_probability + args.predicted_probability + args.jittered_probability
    if abs(probability_sum - 1.0) > 1e-6:
        raise ValueError(f"condition probabilities must sum to 1, got {probability_sum}")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train_rows = [
        row for row in _read_jsonl(args.train_manifest.resolve())
        if bool(row.get("condition_available")) and float(row["condition_iou"]) >= args.minimum_condition_iou
    ]
    dev_rows = [
        row for row in _read_jsonl(args.dev_manifest.resolve())
        if bool(row.get("condition_available")) and float(row["condition_iou"]) >= args.minimum_condition_iou
    ]
    train_rows = _stratified_limit(train_rows, args.max_train_events, args.seed)
    dev_rows = _stratified_limit(dev_rows, args.max_dev_events, args.seed + 1)
    cache = torch.load(args.label_embedding_cache.resolve(), map_location="cpu", weights_only=True)
    if cache.get("format") != "qces_label_clap_public22_v1":
        raise ValueError("invalid public-22 CLAP label cache")
    labels = list(cache["labels"])
    if sorted(set(int(row["label_id"]) for row in train_rows + dev_rows)) != list(range(22)):
        raise ValueError("train/dev manifests do not cover all public-22 label IDs")
    initial_cache = torch.load(args.initial_label_cache.resolve(), map_location="cpu", weights_only=True)
    initial = torch.load(args.initial_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if initial.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("invalid initial separator checkpoint")
    initial_config = initial["config"]
    model = SpanMaskNetwork(
        base_channels=int(initial_config["base_channels"]),
        semantic_channels=int(initial_config["semantic_channels"]),
        label_embeddings=initial_cache["embeddings"].float(),
    )
    model.load_state_dict(initial["model_state"], strict=True)
    model.label_embeddings = cache["embeddings"].float().contiguous()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    chunk_samples = int(round(args.chunk_seconds * RATE))
    dataset_kwargs = {
        "chunk_samples": chunk_samples,
        "exact_probability": args.exact_probability,
        "predicted_probability": args.predicted_probability,
        "pad_min": int(round(args.train_pad_min_seconds * RATE)),
        "pad_max": int(round(args.train_pad_max_seconds * RATE)),
        "validation_pad": int(round(args.validation_pad_seconds * RATE)),
        "maximum_jitter": int(round(args.maximum_jitter_seconds * RATE)),
        "soft_edge": int(round(args.soft_edge_seconds * RATE)),
    }
    train_dataset = RobustConditionDataset(
        train_rows, seed=args.seed, train=True, **dataset_kwargs
    )
    dev_dataset = RobustConditionDataset(
        dev_rows, seed=args.seed + 1, train=False, **dataset_kwargs
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    config = dict(initial_config)
    config.update({
        "conditioning": "public22_clap_text_plus_soft_robust_temporal_condition",
        "label_count": len(labels),
        "labels": labels,
        "condition_mix": {
            "exact": args.exact_probability,
            "predicted_dilated": args.predicted_probability,
            "predicted_jittered_dilated": args.jittered_probability,
        },
        "condition_padding_seconds": [args.train_pad_min_seconds, args.train_pad_max_seconds],
        "validation_padding_seconds": args.validation_pad_seconds,
        "maximum_jitter_seconds": args.maximum_jitter_seconds,
        "soft_edge_seconds": args.soft_edge_seconds,
        "initial_checkpoint_sha256": _sha256(args.initial_checkpoint.resolve()),
        "public22_label_cache_sha256": _sha256(args.label_embedding_cache.resolve()),
        "selection": "exact-semantics median SD-SDRi over padded predicted-span crop",
        "acceptance_gate": {"median_gain_db_min": 1.0, "positive_rate_min": 0.60},
    })
    initial_metrics = _validate(model, dev_loader, device, labels)
    best_key = _key(initial_metrics)
    best_epoch = 0
    _save(output / "best.pt", model, 0, initial_metrics, config, labels)
    print(json.dumps({"epoch": 0, "validation": initial_metrics}, ensure_ascii=False), flush=True)
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        sums: dict[str, float] = {}
        steps = 0
        for step, batch in enumerate(train_loader, 1):
            mixture, target, condition, gold, _iou, label_id, _proxy, _mode = batch
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
            if step % 100 == 0:
                print(json.dumps({
                    "epoch": epoch, "step": step, "steps": len(train_loader),
                    "mean_train_loss": sums["total"] / steps,
                }), flush=True)
        validation = _validate(model, dev_loader, device, labels)
        key = _key(validation)
        scheduler.step(key[0])
        record = {
            "epoch": epoch,
            "train": {name: value / steps for name, value in sums.items()},
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            _save(output / "best.pt", model, epoch, validation, config, labels)
        else:
            stale += 1
        if stale >= args.patience:
            break
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    best_metrics = best["metrics"]
    exact = best_metrics["exact_semantics"]
    accepted = (
        float(exact["sd_sdri_over_crop_db_↑"]["median"]) >= 1.0
        and float(exact["positive_rate_↑"]) >= 0.60
    )
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "best_epoch": best_epoch,
        "accepted_predicted_span_gate": accepted,
        "train_events": len(train_rows),
        "dev_events": len(dev_rows),
        "config": config,
        "best_validation": best_metrics,
        "history": history,
        "best_checkpoint": str((output / "best.pt").resolve()),
        "best_checkpoint_sha256": _sha256(output / "best.pt"),
        "train_manifest_sha256": _sha256(args.train_manifest.resolve()),
        "dev_manifest_sha256": _sha256(args.dev_manifest.resolve()),
        "decision": "candidate_for_public22_integration" if accepted else "keep_crop_fallback",
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps({
        "complete": True,
        "best_epoch": best_epoch,
        "accepted": accepted,
        "best_validation": best_metrics,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
