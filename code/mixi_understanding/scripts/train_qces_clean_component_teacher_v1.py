#!/usr/bin/env python3
"""Train a clean-component semantic teacher for the 188-class ontology.

The frozen R1 BEATs backbone extracts frame features from one representative
component per original source.  A class-balanced attentive pooling head then
learns exact event identity.  Train/dev are audited by source id and source
hash before feature extraction.  Mixtures and QA answers are never inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
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

from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    load_model,
    load_ontology,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json,
    _atomic_torch,
    _sha256_file,
)


FORMAT = "qces_clean_component_semantic_teacher_receipt_v1"
CHECKPOINT_FORMAT = "qces_clean_component_semantic_teacher_checkpoint_v1"
CACHE_FORMAT = "qces_clean_component_beats_feature_cache_v1"
SAMPLE_RATE = 16_000
FIXED_SECONDS = 2.0
FIXED_SAMPLES = int(SAMPLE_RATE * FIXED_SECONDS)


@dataclass(frozen=True)
class ComponentSource:
    source_id: str
    source_sha256: str
    component_sha256: str
    component_path: Path
    label: str
    label_id: int
    num_component_samples: int
    audible_samples: int
    tail_zero_padding_samples: int
    split: str


@dataclass(frozen=True)
class TeacherConfig:
    feature_dim: int = 768
    hidden_dim: int = 512
    embedding_dim: int = 256
    num_classes: int = 188
    dropout: float = 0.15
    logit_scale: float = 16.0
    cosine_margin: float = 0.08
    frame_dropout: float = 0.08
    feature_noise_std: float = 0.01


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.resolve().open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_unique_sources(
    manifest: Path, labels: Sequence[str], *, split: str, max_sources: int = 0
) -> list[ComponentSource]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_label: dict[str, str] = {}
    for row in read_jsonl(manifest):
        source_id = str(row.get("source_id") or "").strip()
        label = str(row.get("label") or "").strip()
        if not source_id or label not in label_to_id:
            raise ValueError(f"invalid component row in {manifest}: source={source_id!r} label={label!r}")
        if source_id in source_label and source_label[source_id] != label:
            raise ValueError(f"source {source_id} has multiple exact labels")
        source_label[source_id] = label
        groups[source_id].append(row)

    selected: list[ComponentSource] = []
    for source_id, rows in sorted(groups.items()):
        # Prefer the crop with the most non-padding audio.  Stable lexical
        # tie-breaking makes the cache exactly reproducible.
        best = max(
            rows,
            key=lambda row: (
                int(row.get("num_component_samples") or 0)
                - int(row.get("tail_zero_padding_samples") or 0),
                int(row.get("num_component_samples") or 0),
                str(row.get("component_sha256") or ""),
            ),
        )
        path = Path(str(best["component_path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        samples = int(best["num_component_samples"])
        tail = int(best.get("tail_zero_padding_samples") or 0)
        audible = samples - tail
        if not 1 <= audible <= samples <= FIXED_SAMPLES:
            raise ValueError(f"invalid component duration for {source_id}")
        label = source_label[source_id]
        selected.append(
            ComponentSource(
                source_id=source_id,
                source_sha256=str(best.get("source_sha256") or ""),
                component_sha256=str(best.get("component_sha256") or ""),
                component_path=path,
                label=label,
                label_id=label_to_id[label],
                num_component_samples=samples,
                audible_samples=audible,
                tail_zero_padding_samples=tail,
                split=split,
            )
        )
    if max_sources > 0:
        selected = selected[:max_sources]
    if not selected:
        raise ValueError(f"no component sources selected from {manifest}")
    return selected


def identity_audit(
    train: Sequence[ComponentSource], dev: Sequence[ComponentSource]
) -> dict[str, Any]:
    result: dict[str, Any] = {"passes": True, "overlap": {}}
    for field in ("source_id", "source_sha256", "component_sha256"):
        left = {str(getattr(row, field)) for row in train if str(getattr(row, field))}
        right = {str(getattr(row, field)) for row in dev if str(getattr(row, field))}
        overlap = sorted(left & right)
        result["overlap"][field] = {"count": len(overlap), "examples": overlap[:5]}
        result["passes"] = bool(result["passes"] and not overlap)
    if not result["passes"]:
        raise ValueError(f"clean teacher train/dev identity leakage: {result['overlap']}")
    return result


class WaveformComponentDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[ComponentSource]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, sample_rate = torchaudio.load(row.component_path)
        waveform = waveform.float().mean(dim=0)
        if int(sample_rate) != SAMPLE_RATE:
            waveform = AF.resample(waveform, int(sample_rate), SAMPLE_RATE)
        if waveform.numel() < FIXED_SAMPLES:
            waveform = F.pad(waveform, (0, FIXED_SAMPLES - waveform.numel()))
        else:
            waveform = waveform[:FIXED_SAMPLES]
        return {"waveform": waveform, "row": row}


def collate_waveforms(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "waveforms": torch.stack([row["waveform"] for row in rows]),
        "rows": [row["row"] for row in rows],
    }


@torch.inference_mode()
def export_features(
    model: nn.Module,
    rows: Sequence[ComponentSource],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    split: str,
) -> dict[str, Any]:
    loader = DataLoader(
        WaveformComponentDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_waveforms,
    )
    all_features: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []
    processed = 0
    model.eval()
    for batch in loader:
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            mel = model.mel_forward(waveforms)
            features = model.model(mel).float()
        frame_count = int(features.shape[1])
        lengths = torch.tensor(
            [
                max(
                    1,
                    min(
                        frame_count,
                        int(math.ceil(row.audible_samples / FIXED_SAMPLES * frame_count)),
                    ),
                )
                for row in batch["rows"]
            ],
            dtype=torch.int16,
        )
        all_features.append(features.cpu().half())
        all_lengths.append(lengths)
        processed += len(batch["rows"])
        if processed == len(batch["rows"]) or processed % 512 < len(batch["rows"]):
            print(f"feature_export split={split} {processed}/{len(rows)}", flush=True)
    features = torch.cat(all_features)
    lengths = torch.cat(all_lengths)
    if features.shape[0] != len(rows) or lengths.shape[0] != len(rows):
        raise RuntimeError("feature cache row count mismatch")
    return {
        "format": CACHE_FORMAT,
        "split": split,
        "features": features,
        "valid_frames": lengths,
        "label_id": torch.tensor([row.label_id for row in rows], dtype=torch.int16),
        "source_id": [row.source_id for row in rows],
        "source_sha256": [row.source_sha256 for row in rows],
        "component_sha256": [row.component_sha256 for row in rows],
        "feature_shape": list(features.shape),
    }


class CachedComponentDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("format") != CACHE_FORMAT:
            raise ValueError("not a clean-component BEATs cache")
        self.features = payload["features"]
        self.valid_frames = payload["valid_frames"]
        self.label_id = payload["label_id"].long()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "valid_frames": self.valid_frames[index],
            "label_id": self.label_id[index],
        }


class BalancedClassBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: torch.Tensor,
        *,
        classes_per_batch: int,
        samples_per_class: int,
        seed: int,
    ) -> None:
        self.by_class: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(labels.tolist()):
            self.by_class[int(label)].append(index)
        self.classes = sorted(self.by_class)
        self.classes_per_batch = min(int(classes_per_batch), len(self.classes))
        self.samples_per_class = int(samples_per_class)
        self.seed = int(seed)
        self.epoch = 0
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.num_batches = math.ceil(len(labels) / self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[list[int]]:
        generator = random.Random(self.seed + 1009 * self.epoch)
        for _ in range(self.num_batches):
            chosen = generator.sample(self.classes, self.classes_per_batch)
            batch: list[int] = []
            for label in chosen:
                pool = self.by_class[label]
                if len(pool) >= self.samples_per_class:
                    batch.extend(generator.sample(pool, self.samples_per_class))
                else:
                    batch.extend(generator.choices(pool, k=self.samples_per_class))
            generator.shuffle(batch)
            yield batch


class CleanComponentTeacher(nn.Module):
    def __init__(self, config: TeacherConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_norm = nn.LayerNorm(config.feature_dim)
        self.attention = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        pooled_dim = config.feature_dim * 4
        self.projector = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        )
        self.class_prototypes = nn.Parameter(
            F.normalize(torch.randn(config.num_classes, config.embedding_dim), dim=-1)
        )

    def encode(
        self,
        features: torch.Tensor,
        valid_frames: torch.Tensor,
        *,
        augment: bool,
    ) -> torch.Tensor:
        features = self.frame_norm(features.float())
        batch, frames, _ = features.shape
        valid = torch.arange(frames, device=features.device)[None] < valid_frames[:, None]
        if augment and self.config.frame_dropout > 0:
            keep = torch.rand(batch, frames, device=features.device) >= self.config.frame_dropout
            valid = valid & keep
            # Never erase every frame of a sample.
            empty = ~valid.any(dim=1)
            valid[empty, 0] = True
        if augment and self.config.feature_noise_std > 0:
            features = features + torch.randn_like(features) * self.config.feature_noise_std
        mask = valid[..., None].to(features.dtype)
        count = mask.sum(dim=1).clamp_min(1)
        mean = (features * mask).sum(dim=1) / count
        variance = ((features - mean[:, None]).square() * mask).sum(dim=1) / count
        std = variance.clamp_min(1e-6).sqrt()
        maximum = features.masked_fill(~valid[..., None], -1e4).amax(dim=1)
        attention_logits = self.attention(features).squeeze(-1).masked_fill(~valid, -1e4)
        attention = attention_logits.softmax(dim=1)
        attentive = torch.einsum("bt,btd->bd", attention, features)
        pooled = torch.cat((mean, std, maximum, attentive), dim=-1)
        return F.normalize(self.projector(pooled), dim=-1)

    def logits(self, embedding: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(self.class_prototypes, dim=-1)
        return self.config.logit_scale * embedding @ prototypes.T

    def forward(
        self,
        features: torch.Tensor,
        valid_frames: torch.Tensor,
        *,
        augment: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(features, valid_frames, augment=augment)
        return self.logits(embedding), embedding


def margin_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, *, scale: float, margin: float
) -> torch.Tensor:
    adjusted = logits.clone()
    adjusted[torch.arange(target.shape[0], device=target.device), target] -= scale * margin
    return F.cross_entropy(adjusted, target, label_smoothing=0.03)


@torch.inference_mode()
def evaluate_teacher(
    model: CleanComponentTeacher,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    correct1 = correct5 = total = 0
    class_correct: Counter[int] = Counter()
    class_total: Counter[int] = Counter()
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        lengths = batch["valid_frames"].long().to(device, non_blocking=True)
        target = batch["label_id"].long().to(device, non_blocking=True)
        logits, _ = model(features, lengths, augment=False)
        ordering = logits.topk(5, dim=-1).indices
        correct1 += int((ordering[:, 0] == target).sum())
        correct5 += int((ordering == target[:, None]).any(dim=1).sum())
        total += int(target.numel())
        for label, ok in zip(target.cpu().tolist(), (ordering[:, 0] == target).cpu().tolist()):
            class_total[int(label)] += 1
            class_correct[int(label)] += int(ok)
    per_class = {
        str(label): class_correct[label] / max(class_total[label], 1)
        for label in sorted(class_total)
    }
    return {
        "sources": total,
        "top1_accuracy_↑": correct1 / max(total, 1),
        "top5_accuracy_↑": correct5 / max(total, 1),
        "macro_top1_accuracy_↑": sum(per_class.values()) / max(len(per_class), 1),
        "observed_classes": len(per_class),
        "per_class_top1": per_class,
    }


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_gold_natural_v3"
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-components", type=Path, default=data / "event_components_train.jsonl")
    parser.add_argument("--dev-components", type=Path, default=data / "event_components_dev.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--pretrained-checkpoint", default="BEATs_strong_1")
    parser.add_argument("--output-dir", type=Path, default=base / "clean_component_teacher_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6101)
    parser.add_argument("--feature-batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--classes-per-batch", type=int, default=32)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--logit-scale", type=float, default=16.0)
    parser.add_argument("--cosine-margin", type=float, default=0.08)
    parser.add_argument("--frame-dropout", type=float, default=0.08)
    parser.add_argument("--feature-noise-std", type=float, default=0.01)
    parser.add_argument("--max-train-sources", type=int, default=0)
    parser.add_argument("--max-dev-sources", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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
    train_rows = select_unique_sources(
        args.train_components, labels, split="train", max_sources=args.max_train_sources
    )
    dev_rows = select_unique_sources(
        args.dev_components, labels, split="dev", max_sources=args.max_dev_sources
    )
    audit = identity_audit(train_rows, dev_rows)
    train_counts = Counter(row.label for row in train_rows)
    dev_counts = Counter(row.label for row in dev_rows)
    if set(train_counts) != set(labels) or set(dev_counts) != set(labels):
        raise ValueError("both clean-component splits must cover all ontology classes")

    device = make_device(args.device)
    detector_payload = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if list(detector_payload.get("labels") or []) != labels:
        raise ValueError("detector checkpoint ontology mismatch")
    backbone = load_model(len(labels), args.pretrained_checkpoint, device)
    backbone.load_state_dict(detector_payload["model_state_dict"], strict=True)
    backbone.eval().requires_grad_(False)

    cache_paths = {
        "train": output_dir / "clean_component_features_train.pt",
        "dev": output_dir / "clean_component_features_dev.pt",
    }
    caches = {}
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        cache = export_features(
            backbone,
            rows,
            device=device,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            split=split,
        )
        _atomic_torch(cache, cache_paths[split])
        caches[split] = cache
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_dataset = CachedComponentDataset(caches["train"])
    dev_dataset = CachedComponentDataset(caches["dev"])
    sampler = BalancedClassBatchSampler(
        train_dataset.label_id,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    config = TeacherConfig(
        feature_dim=int(caches["train"]["features"].shape[-1]),
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_classes=len(labels),
        dropout=args.dropout,
        logit_scale=args.logit_scale,
        cosine_margin=args.cosine_margin,
        frame_dropout=args.frame_dropout,
        feature_noise_std=args.feature_noise_std,
    )
    model = CleanComponentTeacher(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    history: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    stale = 0
    checkpoint_path = output_dir / "clean_component_teacher_v1_best.pt"

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = correct = seen = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            lengths = batch["valid_frames"].long().to(device, non_blocking=True)
            target = batch["label_id"].long().to(device, non_blocking=True)
            logits, _ = model(features, lengths, augment=True)
            loss = margin_cross_entropy(
                logits,
                target,
                scale=config.logit_scale,
                margin=config.cosine_margin,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * target.numel()
            correct += int((logits.argmax(dim=-1) == target).sum())
            seen += int(target.numel())
        scheduler.step()
        metrics = evaluate_teacher(model, dev_loader, device)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / max(seen, 1),
            "train_top1": correct / max(seen, 1),
            "dev": {key: value for key, value in metrics.items() if key != "per_class_top1"},
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (
            float(metrics["top1_accuracy_↑"]),
            float(metrics["macro_top1_accuracy_↑"]),
            float(metrics["top5_accuracy_↑"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = metrics
            stale = 0
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT,
                    "epoch": epoch,
                    "config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "labels": labels,
                    "detector_checkpoint": str(args.detector_checkpoint.resolve()),
                    "detector_checkpoint_sha256": _sha256_file(args.detector_checkpoint.resolve()),
                    "feature_cache_sha256": {
                        split: _sha256_file(path) for split, path in cache_paths.items()
                    },
                    "dev_metrics": metrics,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    if best_metrics is None:
        raise RuntimeError("teacher training produced no checkpoint")
    gates = {
        "clean_source_top1_ge_0_87": float(best_metrics["top1_accuracy_↑"]) >= 0.87,
        "clean_source_top5_ge_0_97": float(best_metrics["top5_accuracy_↑"]) >= 0.97,
        "clean_source_macro_top1_ge_0_80": float(best_metrics["macro_top1_accuracy_↑"]) >= 0.80,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "one_source_one_sample_frozen_beats_attentive_stats_cosine_teacher",
        "mixture_or_qa_answer_used_as_input": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "config": asdict(config),
        "data": {
            "train_unique_sources": len(train_rows),
            "dev_unique_sources": len(dev_rows),
            "classes": len(labels),
            "train_sources_per_class": {
                "minimum": min(train_counts.values()),
                "maximum": max(train_counts.values()),
                "mean": len(train_rows) / len(labels),
            },
            "dev_sources_per_class": {
                "minimum": min(dev_counts.values()),
                "maximum": max(dev_counts.values()),
                "mean": len(dev_rows) / len(labels),
            },
            "identity_audit": audit,
        },
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "success_gates": gates,
        "all_success_gates_pass": all(gates.values()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "feature_caches": {
            split: {"path": str(path), "sha256": _sha256_file(path)}
            for split, path in cache_paths.items()
        },
        "history": history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(
        json.dumps(
            {
                "complete": True,
                "best_epoch": best_epoch,
                "best_metrics": {
                    key: value for key, value in best_metrics.items() if key != "per_class_top1"
                },
                "success_gates": gates,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
