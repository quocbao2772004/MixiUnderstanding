#!/usr/bin/env python3
"""Train a 191-way event classifier from onset contrast and interval context.

This is a sidecar experiment.  It does not modify the frozen BEATs detector or
the temporal query checkpoint.  At inference the head receives only a proposed
interval and frozen dense features; the gold/answer label is never an input.

The hypothesis is intentionally narrow: when sources overlap, the difference
between the acoustic context immediately before and after a proposed onset is
more class-discriminative than pooling the whole interval.  The run is useful
only if it clears the pre-declared oracle-span top-1 gate in the receipt.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _atomic_torch,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import (
    assert_dense_identity_disjoint_v2,
)


FORMAT = "qces_onset_contrast_semantic_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_onset_contrast_semantic_checkpoint_v1"
NUM_FRAMES = 250
# Historical diagnostic from the easier non-overlap split.  It is retained in
# the receipt for context only and must never be used as the overlap-v2 gate.
EXTERNAL_NONOVERLAP_ORACLE_SPAN_TOP1 = 0.5528
PREDECLARED_MINIMUM_GAIN = 0.05


@dataclass(frozen=True)
class Config:
    feature_dim: int = 768
    num_labels: int = 191
    hidden_dim: int = 256
    context_frames: int = 6
    early_frames: int = 7
    dropout: float = 0.15


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    data = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, default=base / "dense_overlap_query_train_v2/index.json")
    parser.add_argument("--dev-index", type=Path, default=base / "dense_overlap_query_dev_v2/index.json")
    parser.add_argument("--train-scenes", type=Path, default=data / "scene_ids_overlap_train.txt")
    parser.add_argument("--dev-scenes", type=Path, default=data / "scene_ids_overlap_dev.txt")
    parser.add_argument("--output-dir", type=Path, default=base / "onset_contrast_semantic_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2131)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--boundary-jitter", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def semantic_events(metadata: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        event for event in metadata.get("gold_events", metadata.get("events", ()))
        if str(event.get("event_kind", "semantic")) == "semantic"
    ]


class EventDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        store: DenseFeatureStore,
        scene_ids: Sequence[str],
        *,
        preload: bool,
        training: bool,
        boundary_jitter: int,
    ) -> None:
        self.store = store
        self.scene_ids = list(dict.fromkeys(str(value) for value in scene_ids))
        missing = sorted(set(self.scene_ids) - store.scene_ids)
        if missing:
            raise ValueError(f"scene IDs missing from dense store: {missing[:5]}")
        self.preloaded = store.preload(self.scene_ids) if preload else None
        self.training = bool(training)
        self.boundary_jitter = max(0, int(boundary_jitter))
        self.events: list[tuple[str, int, int, int]] = []
        for scene_id in self.scene_ids:
            metadata = store.metadata(scene_id)
            valid = int(metadata["valid_frames"])
            for event in semantic_events(metadata):
                start = max(0, min(valid - 1, int(event.get("onset_frame", math.floor(float(event["onset_seconds"]) / 0.04)))))
                end = max(start + 1, min(valid, int(event.get("offset_frame", math.ceil(float(event["offset_seconds"]) / 0.04)))))
                self.events.append((scene_id, start, end, int(event["label_id"])))
        if not self.events:
            raise ValueError("event dataset is empty")

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_id, start, end, label = self.events[index]
        dense = self.preloaded[scene_id] if self.preloaded is not None else self.store.get(scene_id)
        valid = int(dense["valid_frames"])
        if self.training and self.boundary_jitter:
            start += random.randint(-self.boundary_jitter, self.boundary_jitter)
            end += random.randint(-self.boundary_jitter, self.boundary_jitter)
            start = max(0, min(valid - 1, start))
            end = max(start + 1, min(valid, end))
        return {
            "features": dense["features"].float(),
            "detector_logits": dense["logits"].float(),
            "start": start,
            "end": end,
            "valid": valid,
            "label": label,
        }


def collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "features": torch.stack([row["features"] for row in rows]),
        "detector_logits": torch.stack([row["detector_logits"] for row in rows]),
        "start": torch.tensor([row["start"] for row in rows], dtype=torch.long),
        "end": torch.tensor([row["end"] for row in rows], dtype=torch.long),
        "valid": torch.tensor([row["valid"] for row in rows], dtype=torch.long),
        "label": torch.tensor([row["label"] for row in rows], dtype=torch.long),
    }


def _masked_pool(
    values: torch.Tensor, start: torch.Tensor, end: torch.Tensor, *, maximum: bool = False
) -> torch.Tensor:
    frames = values.shape[1]
    grid = torch.arange(frames, device=values.device)[None, :]
    mask = (grid >= start[:, None]) & (grid < end[:, None])
    if maximum:
        return values.masked_fill(~mask[:, :, None], -1e4).amax(1)
    return (values * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)


def interval_components(
    features: torch.Tensor,
    detector_logits: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    valid: torch.Tensor,
    *,
    context_frames: int,
    early_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pre_start = (start - context_frames).clamp_min(0)
    pre_end = start.maximum(pre_start + 1)
    early_end = torch.minimum(end, start + early_frames).maximum(start + 1)
    end = torch.minimum(end, valid).maximum(start + 1)
    pre = _masked_pool(features, pre_start, pre_end)
    early = _masked_pool(features, start, early_end)
    whole = _masked_pool(features, start, end)
    maximum = _masked_pool(features, start, end, maximum=True)
    acoustic = torch.stack((pre, early, whole, maximum, early - pre), dim=1)
    logit_pre = _masked_pool(detector_logits, pre_start, pre_end)
    logit_early = _masked_pool(detector_logits, start, early_end)
    logit_whole = _masked_pool(detector_logits, start, end)
    logit_maximum = _masked_pool(detector_logits, start, end, maximum=True)
    detector = torch.cat((logit_early, logit_whole, logit_maximum, logit_early - logit_pre), dim=1)
    return acoustic, detector


class OnsetContrastSemanticHead(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.acoustic_norm = nn.LayerNorm(config.feature_dim)
        self.acoustic_projection = nn.Linear(config.feature_dim, config.hidden_dim)
        self.component_attention = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.GELU(), nn.Linear(config.hidden_dim // 2, 1)
        )
        self.detector_projection = nn.Sequential(
            nn.LayerNorm(config.num_labels * 4),
            nn.Linear(config.num_labels * 4, config.hidden_dim),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 3),
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.num_labels),
        )

    def forward(self, acoustic: torch.Tensor, detector: torch.Tensor) -> torch.Tensor:
        projected = F.gelu(self.acoustic_projection(self.acoustic_norm(acoustic)))
        attention = torch.softmax(self.component_attention(projected).squeeze(-1), dim=1)
        attended = (projected * attention[:, :, None]).sum(1)
        # Components 1 and 4 are early-post and onset contrast respectively.
        acoustic_summary = torch.cat((attended, projected[:, 1], projected[:, 4]), dim=1)
        detector_summary = self.detector_projection(detector)
        # Preserve a fixed-size fusion while allowing the detector prior to act
        # as a residual correction, not as the only semantic signal.
        fused = acoustic_summary + torch.cat(
            (detector_summary, detector_summary, detector_summary), dim=1
        )
        return self.classifier(fused)


def metrics(logits: torch.Tensor, target: torch.Tensor, num_labels: int) -> dict[str, Any]:
    ranking = logits.argsort(1, descending=True)
    result: dict[str, Any] = {}
    for k in (1, 5, 10):
        result[f"top{k}_accuracy_↑"] = float((ranking[:, :k] == target[:, None]).any(1).float().mean())
    class_values = []
    per_class = {}
    for label in target.unique().tolist():
        selected = target == int(label)
        value = float((ranking[selected, 0] == target[selected]).float().mean())
        class_values.append(value)
        per_class[str(label)] = value
    result["macro_top1_accuracy_↑"] = float(np.mean(class_values))
    result["observed_classes"] = len(class_values)
    result["per_class_top1"] = per_class
    return result


@torch.inference_mode()
def evaluate(
    model: OnsetContrastSemanticHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    model.eval()
    predictions, targets, raw_predictions = [], [], []
    for raw in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
        acoustic, detector = interval_components(
            batch["features"], batch["detector_logits"], batch["start"], batch["end"], batch["valid"],
            context_frames=model.config.context_frames, early_frames=model.config.early_frames,
        )
        predictions.append(model(acoustic, detector).cpu())
        targets.append(batch["label"].cpu())
        raw_predictions.append(detector[:, model.config.num_labels:2 * model.config.num_labels].cpu())
    score, target = torch.cat(predictions), torch.cat(targets)
    result = metrics(score, target, model.config.num_labels)
    raw = metrics(torch.cat(raw_predictions), target, model.config.num_labels)
    result["frozen_detector_interval_mean_baseline"] = {
        key: value for key, value in raw.items() if key != "per_class_top1"
    }
    return result, score, target


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        raise SystemExit("epochs, patience, and batch-size must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    indexes = [args.train_index.resolve(), args.dev_index.resolve()]
    store = DenseFeatureStore(indexes, cache_size=32 if args.preload else 8)
    train_scenes = load_scene_list(args.train_scenes.resolve())
    dev_scenes = load_scene_list(args.dev_scenes.resolve())
    if args.max_train_scenes:
        train_scenes = train_scenes[: args.max_train_scenes]
    if args.max_dev_scenes:
        dev_scenes = dev_scenes[: args.max_dev_scenes]
    identity = assert_dense_identity_disjoint_v2(store, {"train": train_scenes, "dev": dev_scenes})
    train = EventDataset(
        store, train_scenes, preload=args.preload, training=True, boundary_jitter=args.boundary_jitter
    )
    dev = EventDataset(store, dev_scenes, preload=args.preload, training=False, boundary_jitter=0)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0,
        pin_memory=torch.cuda.is_available(), collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=torch.cuda.is_available(), collate_fn=collate,
    )
    config = Config(
        feature_dim=int(store.feature_dim), num_labels=len(store.labels or []),
        hidden_dim=args.hidden_dim, dropout=args.dropout,
    )
    device = _device(args.device)
    model = OnsetContrastSemanticHead(config).to(device)
    counts = Counter(label for _, _, _, label in train.events)
    class_weight = torch.tensor(
        [1.0 / math.sqrt(max(counts[index], 1)) for index in range(config.num_labels)], device=device
    )
    class_weight /= class_weight.mean()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    checkpoint = output / "onset_contrast_semantic_best.pt"
    history: list[dict[str, Any]] = []
    best_key = (-1.0, -1.0)
    best_epoch = 0
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for raw in train_loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in raw.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                acoustic, detector = interval_components(
                    batch["features"], batch["detector_logits"], batch["start"], batch["end"], batch["valid"],
                    context_frames=config.context_frames, early_frames=config.early_frames,
                )
                score = model(acoustic, detector)
                loss = F.cross_entropy(
                    score, batch["label"], weight=class_weight, label_smoothing=args.label_smoothing
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(batch["label"].numel())
            loss_sum += float(loss.detach()) * batch_size
            seen += batch_size
        dev_metrics, _, _ = evaluate(model, dev_loader, device)
        key = (float(dev_metrics["macro_top1_accuracy_↑"]), float(dev_metrics["top1_accuracy_↑"]))
        row = {
            "epoch": epoch, "train_loss": loss_sum / max(seen, 1),
            "learning_rate": optimizer.param_groups[0]["lr"], "dev": dev_metrics,
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT, "epoch": epoch, "config": asdict(config),
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "labels": list(store.labels or []), "dev_metrics": dev_metrics,
                    "dense_index_sha256": dict(store.index_sha256),
                },
                checkpoint,
            )
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break

    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    final_metrics, _, _ = evaluate(model, dev_loader, device)
    top1 = float(final_metrics["top1_accuracy_↑"])
    same_split_baseline = float(
        final_metrics["frozen_detector_interval_mean_baseline"]["top1_accuracy_↑"]
    )
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "hypothesis": "onset contrast separates a newly entering event from already-active overlapping sources",
        "answer_or_gold_label_used_as_model_input": False,
        "gold_intervals_used_for_training_and_dev_diagnostic": True,
        "deployment_status": "diagnostic semantic head; predicted-interval integration not yet evaluated",
        "predeclared_same_split_gate": {
            "reference_frozen_detector_overlap_v2_top1": same_split_baseline,
            "minimum_absolute_gain": PREDECLARED_MINIMUM_GAIN,
            "required_top1": same_split_baseline + PREDECLARED_MINIMUM_GAIN,
            "observed_top1": top1,
            "passed": top1 >= same_split_baseline + PREDECLARED_MINIMUM_GAIN,
        },
        "external_nonoverlap_context_not_comparable": {
            "oracle_span_top1": EXTERNAL_NONOVERLAP_ORACLE_SPAN_TOP1,
            "reason": "different, easier non-overlap development split",
        },
        "data": {
            "train_scenes": len(train_scenes), "train_events": len(train),
            "dev_scenes": len(dev_scenes), "dev_events": len(dev),
            "classes": config.num_labels, "identity_audit": identity,
        },
        "configuration": jsonable(
            {**vars(args), "train_index": args.train_index.resolve(),
             "dev_index": args.dev_index.resolve(), "output_dir": output}
        ),
        "model_config": asdict(config),
        "best_epoch": best_epoch,
        "best_dev": final_metrics,
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256_file(checkpoint)},
        "history": history,
    }
    _atomic_json(receipt, output / "receipt.json")
    print(json.dumps({"result": receipt}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
