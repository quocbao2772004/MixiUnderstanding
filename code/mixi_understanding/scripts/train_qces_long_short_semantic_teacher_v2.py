#!/usr/bin/env python3
"""Train a long-view semantic teacher and distill it into short components.

This is the gated follow-up to ``audit_qces_full_source_semantic_ceiling_v1``.
The long branch learns from one full clean source per source identity.  The
short branch sees only the exact <=1.6 s rendered component available in the
current overlap-v3 scenes.  Train/dev identities remain disjoint and no QA
answer, mixture label, or oracle interval is used as an input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline" / "PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset, Sampler

from mixi_understanding.scripts.audit_qces_full_source_semantic_ceiling_v1 import (
    FIXED_SAMPLES,
    NUM_FRAMES,
    SAMPLE_RATE,
    SourcePair,
    _energy_dense_window,
    load_pairs,
)
from mixi_understanding.scripts.train_qces_clean_component_teacher_v1 import (
    CACHE_FORMAT,
    CHECKPOINT_FORMAT as SHORT_V1_CHECKPOINT_FORMAT,
    BalancedClassBatchSampler,
    CleanComponentTeacher,
    TeacherConfig,
    evaluate_teacher,
    identity_audit,
    margin_cross_entropy,
    select_unique_sources,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    interpolate_sequence,
    load_model,
    load_ontology,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json,
    _atomic_torch,
    _sha256_file,
)


FORMAT = "qces_long_short_semantic_teacher_receipt_v2"
CACHE_V2_FORMAT = "qces_full_source_fixed_stats_cache_v2"
CHECKPOINT_V2_FORMAT = "qces_long_short_semantic_teacher_checkpoint_v2"


@dataclass(frozen=True)
class LongHeadConfig:
    input_dim: int
    hidden_dim: int
    embedding_dim: int
    num_classes: int
    dropout: float
    logit_scale: float
    cosine_margin: float


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_gold_natural_v3"
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    clean_v1 = base / "clean_component_teacher_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-components", type=Path, default=data / "event_components_train.jsonl")
    parser.add_argument("--dev-components", type=Path, default=data / "event_components_dev.jsonl")
    parser.add_argument("--train-scenes", type=Path, default=data / "detector_scene_manifest_overlap_train.jsonl")
    parser.add_argument("--dev-scenes", type=Path, default=data / "detector_scene_manifest_overlap_dev.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--short-train-cache", type=Path, default=clean_v1 / "clean_component_features_train.pt")
    parser.add_argument("--short-dev-cache", type=Path, default=clean_v1 / "clean_component_features_dev.pt")
    parser.add_argument("--short-initial-checkpoint", type=Path, default=clean_v1 / "clean_component_teacher_v1_best.pt")
    parser.add_argument("--output-dir", type=Path, default=base / "long_short_semantic_teacher_v2")
    parser.add_argument("--pretrained-checkpoint", default="BEATs_strong_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6202)
    parser.add_argument("--feature-batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--classes-per-batch", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--long-epochs", type=int, default=40)
    parser.add_argument("--short-epochs", type=int, default=28)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--long-learning-rate", type=float, default=3e-4)
    parser.add_argument("--short-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--logit-scale", type=float, default=16.0)
    parser.add_argument("--cosine-margin", type=float, default=0.06)
    parser.add_argument("--distill-weight", type=float, default=0.35)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--r1-teacher-mix", type=float, default=0.25)
    parser.add_argument("--max-train-sources", type=int, default=0)
    parser.add_argument("--max-dev-sources", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_waveform(path: Path) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.float().mean(dim=0)
    if int(sample_rate) != SAMPLE_RATE:
        waveform = AF.resample(waveform, int(sample_rate), SAMPLE_RATE)
    waveform = _energy_dense_window(waveform)
    valid = min(int(waveform.numel()), FIXED_SAMPLES)
    if waveform.numel() < FIXED_SAMPLES:
        waveform = F.pad(waveform, (0, FIXED_SAMPLES - waveform.numel()))
    return waveform[:FIXED_SAMPLES], valid


class FullSourceDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[SourcePair]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, valid = _load_waveform(row.source_path)
        return {"waveform": waveform, "valid_samples": valid, "row": row}


def collate_sources(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "waveforms": torch.stack([item["waveform"] for item in items]),
        "valid_samples": [int(item["valid_samples"]) for item in items],
        "rows": [item["row"] for item in items],
    }


def _fixed_stats(features: torch.Tensor, valid_frames: Sequence[int]) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for feature, length in zip(features, valid_frames):
        current = feature[: max(1, int(length))].float()
        rows.append(
            torch.cat(
                (
                    current.mean(dim=0),
                    current.var(dim=0, unbiased=False).clamp_min(1e-6).sqrt(),
                    current.amax(dim=0),
                ),
                dim=-1,
            )
        )
    return torch.stack(rows)


@torch.inference_mode()
def export_long_cache(
    model: nn.Module,
    rows: Sequence[SourcePair],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    split: str,
) -> dict[str, Any]:
    loader = DataLoader(
        FullSourceDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sources,
    )
    stats: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    processed = 0
    model.eval()
    for batch in loader:
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            mel = model.mel_forward(waveforms)
            features = model.model(mel).float()
            dense_features = interpolate_sequence(features, model.seq_len)
            dense_logits = model.strong_head(model.seq_model(dense_features)).float()
        feature_lengths = [
            max(1, min(features.shape[1], int(math.ceil(valid / FIXED_SAMPLES * features.shape[1]))))
            for valid in batch["valid_samples"]
        ]
        logit_lengths = [
            max(1, min(NUM_FRAMES, int(math.ceil(valid / FIXED_SAMPLES * NUM_FRAMES))))
            for valid in batch["valid_samples"]
        ]
        stats.append(_fixed_stats(features, feature_lengths).cpu().half())
        logits.append(
            torch.stack([value[:length].mean(dim=0) for value, length in zip(dense_logits, logit_lengths)])
            .cpu()
            .half()
        )
        processed += len(batch["rows"])
        if processed == len(batch["rows"]) or processed % 512 < len(batch["rows"]):
            print(f"long_cache split={split} {processed}/{len(rows)}", flush=True)
    return {
        "format": CACHE_V2_FORMAT,
        "split": split,
        "fixed_stats": torch.cat(stats),
        "r1_logits": torch.cat(logits),
        "label_id": torch.tensor([row.label_id for row in rows], dtype=torch.int16),
        "source_id": [row.source_id for row in rows],
        "source_sha256": [row.source_sha256 for row in rows],
    }


class PairedCacheDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, short: Mapping[str, Any], long: Mapping[str, Any]) -> None:
        if short.get("format") != CACHE_FORMAT or long.get("format") != CACHE_V2_FORMAT:
            raise ValueError("invalid paired semantic cache format")
        if list(short["source_id"]) != list(long["source_id"]):
            raise ValueError("short/long cache source order mismatch")
        if not torch.equal(short["label_id"].short(), long["label_id"].short()):
            raise ValueError("short/long cache label mismatch")
        self.short_features = short["features"]
        self.short_valid = short["valid_frames"]
        self.long_stats = long["fixed_stats"]
        self.r1_logits = long["r1_logits"]
        self.label_id = long["label_id"].long()

    def __len__(self) -> int:
        return int(self.label_id.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "short_features": self.short_features[index],
            "short_valid": self.short_valid[index],
            "long_stats": self.long_stats[index],
            "r1_logits": self.r1_logits[index],
            "label_id": self.label_id[index],
        }


class LongSemanticHead(nn.Module):
    def __init__(self, config: LongHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.projector = nn.Sequential(
            nn.LayerNorm(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        # Residual calibration preserves the already useful frozen-R1 full
        # source classifier at initialization.  Epoch zero is therefore an
        # exact, auditable baseline rather than a random classifier.
        self.residual_head = nn.Linear(config.embedding_dim, config.num_classes)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self, stats: torch.Tensor, r1_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = F.normalize(self.projector(stats.float()), dim=-1)
        return r1_logits.float() + self.residual_head(embedding), embedding


@torch.inference_mode()
def evaluate_long(model: LongSemanticHead, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    total = correct1 = correct5 = 0
    class_total: Counter[int] = Counter()
    class_correct: Counter[int] = Counter()
    for batch in loader:
        target = batch["label_id"].long().to(device)
        logits, _ = model(
            batch["long_stats"].to(device), batch["r1_logits"].to(device)
        )
        top = logits.topk(5, dim=-1).indices
        correct = top[:, 0] == target
        total += int(target.numel())
        correct1 += int(correct.sum())
        correct5 += int((top == target[:, None]).any(dim=-1).sum())
        for label, ok in zip(target.cpu().tolist(), correct.cpu().tolist()):
            class_total[int(label)] += 1
            class_correct[int(label)] += int(ok)
    per_class = {str(label): class_correct[label] / class_total[label] for label in sorted(class_total)}
    return {
        "sources": total,
        "top1_accuracy_↑": correct1 / total,
        "top5_accuracy_↑": correct5 / total,
        "macro_top1_accuracy_↑": sum(per_class.values()) / len(per_class),
        "per_class_top1": per_class,
    }


@torch.inference_mode()
def evaluate_short(model: CleanComponentTeacher, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    total = correct1 = correct5 = 0
    class_total: Counter[int] = Counter()
    class_correct: Counter[int] = Counter()
    for batch in loader:
        target = batch["label_id"].long().to(device)
        logits, _ = model(
            batch["short_features"].to(device),
            batch["short_valid"].long().to(device),
            augment=False,
        )
        top = logits.topk(5, dim=-1).indices
        correct = top[:, 0] == target
        total += int(target.numel())
        correct1 += int(correct.sum())
        correct5 += int((top == target[:, None]).any(dim=-1).sum())
        for label, ok in zip(target.cpu().tolist(), correct.cpu().tolist()):
            class_total[int(label)] += 1
            class_correct[int(label)] += int(ok)
    per_class = {str(label): class_correct[label] / class_total[label] for label in sorted(class_total)}
    return {
        "sources": total,
        "top1_accuracy_↑": correct1 / total,
        "top5_accuracy_↑": correct5 / total,
        "macro_top1_accuracy_↑": sum(per_class.values()) / len(per_class),
        "per_class_top1": per_class,
    }


def _distill_loss(
    student: torch.Tensor,
    long_teacher: torch.Tensor,
    r1_teacher: torch.Tensor,
    *,
    temperature: float,
    r1_mix: float,
) -> torch.Tensor:
    long_probability = (long_teacher / temperature).softmax(dim=-1)
    r1_probability = (r1_teacher / temperature).softmax(dim=-1)
    target = (1.0 - r1_mix) * long_probability + r1_mix * r1_probability
    return F.kl_div(
        (student / temperature).log_softmax(dim=-1),
        target,
        reduction="batchmean",
    ) * (temperature**2)


def _metric_key(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["top1_accuracy_↑"]),
        float(metrics["macro_top1_accuracy_↑"]),
        float(metrics["top5_accuracy_↑"]),
    )


def _prefix_cache(payload: Mapping[str, Any], size: int) -> dict[str, Any]:
    """Take a stable cache prefix for smoke tests; size=0 preserves all rows."""
    if size <= 0:
        return dict(payload)
    total = len(payload["source_id"])
    if size > total:
        raise ValueError(f"requested cache prefix {size} exceeds {total}")
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == total:
            result[key] = value[:size]
        elif isinstance(value, list) and len(value) == total:
            result[key] = value[:size]
        else:
            result[key] = value
    return result


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_ontology(args.ontology.resolve())

    short_rows = {
        "train": select_unique_sources(
            args.train_components,
            labels,
            split="train",
            max_sources=args.max_train_sources,
        ),
        "dev": select_unique_sources(
            args.dev_components,
            labels,
            split="dev",
            max_sources=args.max_dev_sources,
        ),
    }
    audit = identity_audit(short_rows["train"], short_rows["dev"])
    long_rows = {
        "train": load_pairs(
            args.train_components,
            args.train_scenes,
            labels,
            max_sources=args.max_train_sources,
        ),
        "dev": load_pairs(
            args.dev_components,
            args.dev_scenes,
            labels,
            max_sources=args.max_dev_sources,
        ),
    }
    for split in ("train", "dev"):
        expected = [row.source_id for row in short_rows[split]]
        observed = [row.source_id for row in long_rows[split]]
        if expected != observed:
            raise ValueError(f"{split}: short/full source identity order mismatch")

    short_cache = {
        "train": _prefix_cache(
            torch.load(args.short_train_cache.resolve(), map_location="cpu", weights_only=True),
            args.max_train_sources,
        ),
        "dev": _prefix_cache(
            torch.load(args.short_dev_cache.resolve(), map_location="cpu", weights_only=True),
            args.max_dev_sources,
        ),
    }
    for split in ("train", "dev"):
        if list(short_cache[split].get("source_id") or []) != [row.source_id for row in long_rows[split]]:
            raise ValueError(f"{split}: existing short cache identity mismatch")

    device = make_device(args.device)
    detector_payload = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if list(detector_payload.get("labels") or []) != labels:
        raise ValueError("detector ontology mismatch")
    backbone = load_model(len(labels), args.pretrained_checkpoint, device)
    backbone.load_state_dict(detector_payload["model_state_dict"], strict=True)
    backbone.eval().requires_grad_(False)
    long_cache_paths = {
        "train": output_dir / "full_source_stats_train.pt",
        "dev": output_dir / "full_source_stats_dev.pt",
    }
    long_cache: dict[str, Mapping[str, Any]] = {}
    for split in ("train", "dev"):
        payload = export_long_cache(
            backbone,
            long_rows[split],
            device=device,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            split=split,
        )
        _atomic_torch(payload, long_cache_paths[split])
        long_cache[split] = payload
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    datasets = {
        split: PairedCacheDataset(short_cache[split], long_cache[split])
        for split in ("train", "dev")
    }
    sampler = BalancedClassBatchSampler(
        datasets["train"].label_id,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    train_loader = DataLoader(datasets["train"], batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda")
    dev_loader = DataLoader(datasets["dev"], batch_size=256, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    long_config = LongHeadConfig(
        input_dim=int(long_cache["train"]["fixed_stats"].shape[-1]),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_classes=len(labels),
        dropout=args.dropout,
        logit_scale=args.logit_scale,
        cosine_margin=args.cosine_margin,
    )
    long_model = LongSemanticHead(long_config).to(device)
    optimizer = torch.optim.AdamW(long_model.parameters(), lr=args.long_learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.long_epochs, eta_min=args.long_learning_rate * 0.05)
    baseline_long_metrics = evaluate_long(long_model, dev_loader, device)
    long_history: list[dict[str, Any]] = [
        {
            "stage": "long",
            "epoch": 0,
            "baseline": "frozen_r1_full_source_valid_mean",
            "dev": {
                key: value
                for key, value in baseline_long_metrics.items()
                if key != "per_class_top1"
            },
        }
    ]
    print(json.dumps(long_history[0], sort_keys=True), flush=True)
    long_best_key: tuple[float, float, float] | None = _metric_key(baseline_long_metrics)
    long_best_metrics: dict[str, Any] | None = baseline_long_metrics
    long_best_epoch = 0
    long_best_state: dict[str, torch.Tensor] | None = {
        name: value.detach().cpu().clone()
        for name, value in long_model.state_dict().items()
    }
    stale = 0
    for epoch in range(1, args.long_epochs + 1):
        sampler.set_epoch(epoch)
        long_model.train()
        seen = correct = 0
        loss_sum = 0.0
        for batch in train_loader:
            target = batch["label_id"].long().to(device)
            logits, _ = long_model(
                batch["long_stats"].to(device), batch["r1_logits"].to(device)
            )
            loss = F.cross_entropy(logits, target, label_smoothing=0.03)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(long_model.parameters(), 2.0)
            optimizer.step()
            seen += int(target.numel())
            correct += int((logits.argmax(dim=-1) == target).sum())
            loss_sum += float(loss.detach()) * target.numel()
        scheduler.step()
        metrics = evaluate_long(long_model, dev_loader, device)
        row = {"stage": "long", "epoch": epoch, "train_loss": loss_sum / seen, "train_top1": correct / seen, "dev": {k: v for k, v in metrics.items() if k != "per_class_top1"}}
        long_history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = _metric_key(metrics)
        if long_best_key is None or key > long_best_key:
            long_best_key = key
            long_best_metrics = metrics
            long_best_epoch = epoch
            long_best_state = {name: value.detach().cpu().clone() for name, value in long_model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if long_best_state is None or long_best_metrics is None:
        raise RuntimeError("long teacher produced no checkpoint")
    long_model.load_state_dict(long_best_state, strict=True)
    long_model.eval().requires_grad_(False)

    initial = torch.load(args.short_initial_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if initial.get("format") != SHORT_V1_CHECKPOINT_FORMAT or list(initial.get("labels") or []) != labels:
        raise ValueError("invalid initial short teacher checkpoint")
    short_config = TeacherConfig(**initial["config"])
    short_model = CleanComponentTeacher(short_config).to(device)
    short_model.load_state_dict(initial["model_state_dict"], strict=True)
    initial_short_metrics = evaluate_short(short_model, dev_loader, device)
    optimizer = torch.optim.AdamW(short_model.parameters(), lr=args.short_learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.short_epochs, eta_min=args.short_learning_rate * 0.05)
    short_history: list[dict[str, Any]] = []
    short_best_key = _metric_key(initial_short_metrics)
    short_best_metrics = initial_short_metrics
    short_best_epoch = 0
    short_best_state = {name: value.detach().cpu().clone() for name, value in short_model.state_dict().items()}
    stale = 0
    for epoch in range(1, args.short_epochs + 1):
        sampler.set_epoch(1000 + epoch)
        short_model.train()
        seen = correct = 0
        total_sum = ce_sum = kd_sum = 0.0
        for batch in train_loader:
            target = batch["label_id"].long().to(device)
            short_logits, _ = short_model(
                batch["short_features"].to(device),
                batch["short_valid"].long().to(device),
                augment=True,
            )
            with torch.no_grad():
                long_logits, _ = long_model(
                    batch["long_stats"].to(device), batch["r1_logits"].to(device)
                )
            ce = margin_cross_entropy(short_logits, target, scale=short_config.logit_scale, margin=short_config.cosine_margin)
            kd = _distill_loss(
                short_logits,
                long_logits,
                batch["r1_logits"].to(device).float(),
                temperature=args.distill_temperature,
                r1_mix=args.r1_teacher_mix,
            )
            loss = ce + args.distill_weight * kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(short_model.parameters(), 2.0)
            optimizer.step()
            seen += int(target.numel())
            correct += int((short_logits.argmax(dim=-1) == target).sum())
            total_sum += float(loss.detach()) * target.numel()
            ce_sum += float(ce.detach()) * target.numel()
            kd_sum += float(kd.detach()) * target.numel()
        scheduler.step()
        metrics = evaluate_short(short_model, dev_loader, device)
        row = {
            "stage": "short_distill",
            "epoch": epoch,
            "train_loss": total_sum / seen,
            "train_ce": ce_sum / seen,
            "train_kd": kd_sum / seen,
            "train_top1": correct / seen,
            "dev": {k: v for k, v in metrics.items() if k != "per_class_top1"},
        }
        short_history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = _metric_key(metrics)
        if key > short_best_key:
            short_best_key = key
            short_best_metrics = metrics
            short_best_epoch = epoch
            short_best_state = {name: value.detach().cpu().clone() for name, value in short_model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    checkpoint_path = output_dir / "long_short_semantic_teacher_v2_best.pt"
    _atomic_torch(
        {
            "format": CHECKPOINT_V2_FORMAT,
            "labels": labels,
            "long_config": asdict(long_config),
            "short_config": asdict(short_config),
            "long_model_state_dict": long_best_state,
            "short_model_state_dict": short_best_state,
            "long_best_epoch": long_best_epoch,
            "short_best_epoch": short_best_epoch,
            "long_dev_metrics": long_best_metrics,
            "short_initial_dev_metrics": initial_short_metrics,
            "short_dev_metrics": short_best_metrics,
        },
        checkpoint_path,
    )
    gates = {
        "long_top1_ge_0_80": float(long_best_metrics["top1_accuracy_↑"]) >= 0.80,
        "long_top5_ge_0_97": float(long_best_metrics["top5_accuracy_↑"]) >= 0.97,
        "short_top1_ge_0_69": float(short_best_metrics["top1_accuracy_↑"]) >= 0.69,
        "short_top5_ge_0_93": float(short_best_metrics["top5_accuracy_↑"]) >= 0.93,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_eligible": False,
        "claim_boundary": "development diagnostic; a new identity-locked test is required",
        "method": "full_source_fixed_stats_teacher_then_posterior_distillation_into_short_attentive_component_student",
        "qa_answer_or_mixture_used_as_input": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "data": {
            "train_unique_sources": len(long_rows["train"]),
            "dev_unique_sources": len(long_rows["dev"]),
            "classes": len(labels),
            "identity_audit": audit,
        },
        "long_best_epoch": long_best_epoch,
        "long_best_metrics": long_best_metrics,
        "short_initial_metrics": initial_short_metrics,
        "short_best_epoch": short_best_epoch,
        "short_best_metrics": short_best_metrics,
        "short_top1_delta_↑": float(short_best_metrics["top1_accuracy_↑"]) - float(initial_short_metrics["top1_accuracy_↑"]),
        "success_gates": gates,
        "all_success_gates_pass": all(gates.values()),
        "decision": (
            "proceed_to_mixture_slot_distillation"
            if all(gates.values())
            else "rebuild_semantically_sufficient_longer_components_before_mixture_training"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "long_cache": {
            split: {"path": str(path), "sha256": _sha256_file(path)}
            for split, path in long_cache_paths.items()
        },
        "long_history": long_history,
        "short_history": short_history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "long": {k: v for k, v in long_best_metrics.items() if k != "per_class_top1"}, "short_initial": {k: v for k, v in initial_short_metrics.items() if k != "per_class_top1"}, "short_best": {k: v for k, v in short_best_metrics.items() if k != "per_class_top1"}, "gates": gates, "decision": receipt["decision"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
