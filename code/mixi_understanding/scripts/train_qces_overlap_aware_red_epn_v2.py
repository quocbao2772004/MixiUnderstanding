#!/usr/bin/env python3
"""Train a semantic-first overlap-aware RED/EPN head on Gold-natural v3.

The historical RED/EPN objective was dominated by onset/offset losses and did
not improve oracle-interval class accuracy.  This sidecar keeps its useful
class-aware boundary decoder, but adds a hard-negative interval-ranking loss
and rebalances the objective toward semantic presence.  The frozen detector
logits remain an additive prior, so epoch zero exactly preserves its frame
classification behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mixi_understanding.qces.class_aware_red_epn_v1 import (
    ClassAwareRedEpnV1,
    ClassAwareRedEpnV1Config,
    ClassAwareRedEpnV1LossWeights,
    class_aware_red_epn_v1_loss,
)
from mixi_understanding.scripts.train_qces_class_aware_red_epn_v1 import (
    ClassAwareSceneDataset,
    _jsonable,
    _seed_all,
    _to_device,
    collate_class_aware,
    evaluate,
)
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


FORMAT = "qces_overlap_aware_red_epn_training_receipt_v2"
CHECKPOINT_FORMAT = "qces_overlap_aware_red_epn_checkpoint_v2"
DEFAULT_BASE = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
)
DEFAULT_DATA = PROJECT_ROOT / "outputs/qces_full188_overlap_gold_natural_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-index",
        type=Path,
        default=DEFAULT_BASE / "dense_overlap_gold_natural_v3_train/index.json",
    )
    parser.add_argument(
        "--dev-index",
        type=Path,
        default=DEFAULT_BASE / "dense_overlap_gold_natural_v3_dev/index.json",
    )
    parser.add_argument(
        "--train-scenes",
        type=Path,
        default=DEFAULT_DATA / "scene_ids_overlap_train.txt",
    )
    parser.add_argument(
        "--dev-scenes",
        type=Path,
        default=DEFAULT_DATA / "scene_ids_overlap_dev.txt",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_BASE / "overlap_aware_red_epn_v2"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=3181)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--epn-hidden-dim", type=int, default=128)
    parser.add_argument("--rank-weight", type=float, default=2.0)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--hard-negatives", type=int, default=8)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def interval_hard_negative_rank_loss(
    logits: torch.Tensor,
    frame_presence: torch.Tensor,
    gold_events: Sequence[Sequence[Mapping[str, Any]]],
    *,
    margin: float,
    hard_negatives: int,
) -> torch.Tensor:
    """Rank each event above classes that are absent from its whole interval.

    Other true events overlapping any part of the interval are masked out of
    the negative set.  This preserves the multi-label nature of polyphonic
    frames instead of incorrectly treating co-active events as negatives.
    """

    if logits.shape != frame_presence.shape or logits.ndim != 3:
        raise ValueError("logits and frame_presence must have shape [B,T,C]")
    _, frames, classes = logits.shape
    if len(gold_events) != logits.shape[0]:
        raise ValueError("gold event batch size mismatch")
    losses: list[torch.Tensor] = []
    for batch_index, events in enumerate(gold_events):
        for event in events:
            start = max(
                0,
                min(frames - 1, int(math.floor(float(event["start"]) * frames))),
            )
            end = max(
                start + 1,
                min(frames, int(math.ceil(float(event["end"]) * frames))),
            )
            label_id = int(event["label_id"])
            if not 0 <= label_id < classes:
                raise ValueError("gold label id outside detector ontology")
            pooled = logits[batch_index, start:end].mean(dim=0)
            coactive = frame_presence[batch_index, start:end].amax(dim=0) > 0.5
            negative = pooled.masked_fill(coactive, -torch.inf)
            count = min(max(int(hard_negatives), 1), int((~coactive).sum().item()))
            if count < 1:
                continue
            hard = negative.topk(count).values
            positive = pooled[label_id]
            losses.append(F.softplus(hard - positive + margin).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def semantic_first_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    *,
    base_weights: ClassAwareRedEpnV1LossWeights,
    rank_weight: float,
    rank_margin: float,
    hard_negatives: int,
) -> dict[str, torch.Tensor]:
    losses = class_aware_red_epn_v1_loss(
        outputs,
        batch,
        batch["valid_mask"],
        weights=base_weights,
    )
    rank = interval_hard_negative_rank_loss(
        outputs["direct_presence_logits"],
        batch["frame_presence"],
        batch["gold_events"],
        margin=rank_margin,
        hard_negatives=hard_negatives,
    )
    return {**losses, "interval_rank": rank, "loss": losses["loss"] + rank_weight * rank}


def selection_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    top20 = float(metrics["top20_label_iou30_recall_↑"])
    top8 = float(metrics["top8_label_iou30_recall_↑"])
    return (
        0.5 * (top8 + top20),
        top8,
        top20,
        float(metrics["direct_oracle_interval_top1_accuracy_↑"]),
        float(metrics["top20_macro_label_iou50_recall_↑"]),
    )


def main() -> None:
    args = parse_args()
    for name in ("epochs", "patience", "batch_size", "hard_negatives"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.rank_weight < 0 or args.rank_margin < 0:
        raise ValueError("rank weight and margin must be non-negative")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)

    indexes = [args.train_index.resolve(), args.dev_index.resolve()]
    store = DenseFeatureStore(indexes, cache_size=32)
    labels = list(store.labels or [])
    scene_ids = {
        "train": load_scene_list(args.train_scenes.resolve()),
        "dev": load_scene_list(args.dev_scenes.resolve()),
    }
    identity = assert_dense_identity_disjoint_v2(store, scene_ids)
    datasets = {
        name: ClassAwareSceneDataset(
            store,
            ids,
            preload=args.preload,
            max_scenes=args.max_train_scenes if name == "train" else args.max_dev_scenes,
        )
        for name, ids in scene_ids.items()
    }
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_class_aware,
    }
    train_loader = DataLoader(
        datasets["train"],
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_args,
    )
    dev_loader = DataLoader(datasets["dev"], shuffle=False, **loader_args)
    device = _device(args.device)
    config = ClassAwareRedEpnV1Config(
        feature_dim=int(store.feature_dim),
        num_classes=len(labels),
        context_dim=args.context_dim,
        epn_hidden_dim=args.epn_hidden_dim,
    )
    model = ClassAwareRedEpnV1(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    # Unlike v1, semantic presence and ranking dominate the objective.
    base_weights = ClassAwareRedEpnV1LossWeights(
        direct_presence=2.0,
        red_presence=0.25,
        # Raw transition focal losses start around 4--5 each.  A 0.25 weight
        # keeps both boundaries together below the semantic ranking/presence
        # mass instead of recreating v1's boundary-dominated objective.
        onset=0.25,
        offset=0.25,
        duration_iou=0.5,
        clip_presence=1.0,
        residual_l2=0.02,
    )

    baseline_metrics = evaluate(model, dev_loader, device, num_classes=len(labels))
    baseline = {"epoch": 0, "dev": baseline_metrics, "selection_key": list(selection_key(baseline_metrics))}
    print(json.dumps({"baseline": baseline}, ensure_ascii=False, sort_keys=True), flush=True)

    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = selection_key(baseline_metrics)
    best_epoch = 0
    best_metrics: Mapping[str, Any] | None = baseline
    stale = 0
    checkpoint_path = output_dir / "overlap_aware_red_epn_v2_best.pt"
    _atomic_torch(
        {
            "format": CHECKPOINT_FORMAT,
            "epoch": 0,
            "config": asdict(config),
            "labels": labels,
            "model_state_dict": model.state_dict(),
            "selection_metrics": baseline,
            "dense_indexes": [str(path) for path in indexes],
            "dense_index_sha256": {
                str(path): _sha256_file(path) for path in indexes
            },
        },
        checkpoint_path,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Counter[str] = Counter()
        batches = 0
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = model(
                    batch["features"], batch["detector_logits"], batch["valid_mask"]
                )
                losses = semantic_first_loss(
                    outputs,
                    batch,
                    base_weights=base_weights,
                    rank_weight=args.rank_weight,
                    rank_margin=args.rank_margin,
                    hard_negatives=args.hard_negatives,
                )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for name, value in losses.items():
                totals[name] += float(value.detach())
            batches += 1
        scheduler.step()
        dev_metrics = evaluate(model, dev_loader, device, num_classes=len(labels))
        key = selection_key(dev_metrics)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": {name: value / max(batches, 1) for name, value in totals.items()},
            "dev": dev_metrics,
            "selection_key": list(key),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_metrics, stale = key, epoch, row, 0
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT,
                    "epoch": epoch,
                    "config": asdict(config),
                    "labels": labels,
                    "model_state_dict": model.state_dict(),
                    "selection_metrics": row,
                    "dense_indexes": [str(path) for path in indexes],
                    "dense_index_sha256": {
                        str(path): _sha256_file(path) for path in indexes
                    },
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    if best_metrics is None or best_key is None:
        raise RuntimeError("training produced no checkpoint")
    selected_dev = best_metrics["dev"]
    gates = {
        "selected_epoch_after_training": best_epoch > 0,
        "top20_label_iou30_recall_ge_0_75": float(
            selected_dev["top20_label_iou30_recall_↑"]
        )
        >= 0.75,
        "top8_label_iou30_recall_ge_0_60": float(
            selected_dev["top8_label_iou30_recall_↑"]
        )
        >= 0.60,
        "direct_oracle_interval_top1_ge_0_40": float(
            selected_dev["direct_oracle_interval_top1_accuracy_↑"]
        )
        >= 0.40,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "frozen BEATs prior + class-aware RED/EPN + interval hard-negative ranking",
        "answer_label_used_as_input": False,
        "arguments": _jsonable(vars(args)),
        "config": asdict(config),
        "loss_weights": asdict(base_weights) | {"interval_rank": args.rank_weight},
        "data": {name: len(dataset) for name, dataset in datasets.items()},
        "identity_audit": identity,
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "success_gates": gates,
        "all_success_gates_pass": all(gates.values()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "history": history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(
        json.dumps(
            {
                "complete": True,
                "best_epoch": best_epoch,
                "selection_key": list(best_key),
                "success_gates": gates,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
