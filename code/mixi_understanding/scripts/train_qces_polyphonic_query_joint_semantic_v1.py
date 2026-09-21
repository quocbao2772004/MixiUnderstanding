#!/usr/bin/env python3
"""Jointly tune DETR queries for temporal boxes and 191-way event identity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from mixi_understanding.qces.relational_event_slots_semantic_v1 import RelationalEventSlotsSemanticV1
from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1Config,
    RelationalEventSlotsV1LossWeights,
    relational_event_slots_v1_loss,
)
from mixi_understanding.scripts.train_qces_polyphonic_query_branch_v1 import (
    POLYPHONIC_CHECKPOINT_FORMAT,
    _state_hash,
)
from mixi_understanding.scripts.train_qces_polyphonic_query_semantic_head_v1 import (
    SemanticSceneDataset,
    _class_counts,
    _matched,
    _seed_all,
    collate_semantic,
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
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import assert_dense_identity_disjoint_v2


FORMAT = "qces_polyphonic_query_joint_semantic_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_polyphonic_query_joint_semantic_checkpoint_v1"
TRAINABLE_PREFIXES = (
    "decoder.", "slot_queries.", "objectness_head.", "box_head.", "semantic_head."
)


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/multievent"
    overlap = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-checkpoint", type=Path,
        default=base / "polyphonic_query_branch_v2/polyphonic_query_branch_best.pt",
    )
    parser.add_argument("--nonoverlap-train-index", type=Path, default=base / "dense_multi_train_v2/index.json")
    parser.add_argument("--nonoverlap-dev-index", type=Path, default=base / "dense_multi_dev_v2/index.json")
    parser.add_argument("--overlap-train-index", type=Path, default=base / "dense_overlap_query_train_v2/index.json")
    parser.add_argument("--overlap-dev-index", type=Path, default=base / "dense_overlap_query_dev_v2/index.json")
    parser.add_argument("--nonoverlap-train-scenes", type=Path, default=data / "scene_ids_train.txt")
    parser.add_argument("--nonoverlap-dev-scenes", type=Path, default=data / "scene_ids_dev.txt")
    parser.add_argument("--overlap-train-scenes", type=Path, default=overlap / "scene_ids_overlap_train.txt")
    parser.add_argument("--overlap-dev-scenes", type=Path, default=overlap / "scene_ids_overlap_dev.txt")
    parser.add_argument(
        "--output-dir", type=Path,
        default=base / "polyphonic_query_joint_semantic_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2113)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--query-learning-rate", type=float, default=1e-4)
    parser.add_argument("--semantic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--minimum-train-match-iou", type=float, default=0.30)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key in (
        "features", "detector_logits", "valid_mask", "frame_event_target", "frame_onset_target"
    ):
        result[key] = batch[key].to(device, non_blocking=True)
    result["target_intervals"] = [value.to(device, non_blocking=True) for value in batch["target_intervals"]]
    result["target_label_ids"] = [value.to(device, non_blocking=True) for value in batch["target_label_ids"]]
    return result


def _selection_key(nonoverlap: Mapping[str, Any], overlap: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        0.5 * (float(nonoverlap["joint_label_and_iou50_recall_↑"]) + float(overlap["joint_label_and_iou50_recall_↑"])),
        0.5 * (float(nonoverlap["query_semantic_top1_accuracy_↑"]) + float(overlap["query_semantic_top1_accuracy_↑"])),
        0.5 * (float(nonoverlap["temporal_recall_iou50_↑"]) + float(overlap["temporal_recall_iou50_↑"])),
        float(overlap["joint_label_and_iou50_recall_↑"]),
        float(nonoverlap["joint_label_and_iou50_recall_↑"]),
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)
    query_path = args.query_checkpoint.resolve()
    query = torch.load(query_path, map_location="cpu", weights_only=True)
    if query.get("format") != POLYPHONIC_CHECKPOINT_FORMAT:
        raise ValueError("query checkpoint has wrong format")
    indexes = [
        args.nonoverlap_train_index.resolve(), args.nonoverlap_dev_index.resolve(),
        args.overlap_train_index.resolve(), args.overlap_dev_index.resolve(),
    ]
    store = DenseFeatureStore(indexes, cache_size=32)
    labels = list(store.labels or [])
    if labels != list(query.get("labels") or []):
        raise ValueError("query checkpoint and dense indexes disagree on labels")
    scene_ids = {
        "nonoverlap_train": load_scene_list(args.nonoverlap_train_scenes),
        "nonoverlap_dev": load_scene_list(args.nonoverlap_dev_scenes),
        "overlap_train": load_scene_list(args.overlap_train_scenes),
        "overlap_dev": load_scene_list(args.overlap_dev_scenes),
    }
    identity = assert_dense_identity_disjoint_v2(
        store,
        {
            "train": scene_ids["nonoverlap_train"] + scene_ids["overlap_train"],
            "dev": scene_ids["nonoverlap_dev"] + scene_ids["overlap_dev"],
        },
    )
    datasets = {
        name: SemanticSceneDataset(store, ids, preload=args.preload)
        for name, ids in scene_ids.items()
    }
    if len(datasets["nonoverlap_train"]) != len(datasets["overlap_train"]):
        raise RuntimeError("joint training requires equal non-overlap/overlap scene exposure")
    loader_args = {
        "batch_size": args.batch_size, "num_workers": 0,
        "pin_memory": torch.cuda.is_available(), "collate_fn": collate_semantic,
    }
    train_loader = DataLoader(
        ConcatDataset((datasets["nonoverlap_train"], datasets["overlap_train"])),
        shuffle=True, generator=torch.Generator().manual_seed(args.seed), **loader_args,
    )
    nonoverlap_dev_loader = DataLoader(datasets["nonoverlap_dev"], shuffle=False, **loader_args)
    overlap_dev_loader = DataLoader(datasets["overlap_dev"], shuffle=False, **loader_args)
    device = _device(args.device)
    model = RelationalEventSlotsSemanticV1(
        RelationalEventSlotsV1Config(**query["config"]), len(labels)
    ).to(device)
    missing, unexpected = model.load_state_dict(query["model_state_dict"], strict=False)
    if unexpected or any(not name.startswith("semantic_head.") for name in missing):
        raise RuntimeError(f"checkpoint incompatibility missing={missing} unexpected={unexpected}")
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    frozen_names = [name for name in query["model_state_dict"] if not name.startswith(TRAINABLE_PREFIXES)]
    frozen_hash = _state_hash(model.state_dict(), frozen_names)
    if frozen_hash != _state_hash(query["model_state_dict"], frozen_names):
        raise RuntimeError("frozen encoder state differs at initialization")

    counts = _class_counts(
        [datasets["nonoverlap_train"], datasets["overlap_train"]], len(labels)
    )
    class_weights = counts.clamp_min(1).rsqrt()
    class_weights = (class_weights / class_weights.mean()).to(device)
    query_parameters = [
        value for name, value in model.named_parameters()
        if value.requires_grad and not name.startswith("semantic_head.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": query_parameters, "lr": args.query_learning_rate},
            {"params": model.semantic_head.parameters(), "lr": args.semantic_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.query_learning_rate * 0.05
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    temporal_weights = RelationalEventSlotsV1LossWeights(
        objectness=1.0, box_l1=5.0, box_iou=2.0,
        frame_bce=0.0, frame_dice=0.0, onset_bce=0.0,
    )
    history = []
    best_key = None
    best_epoch = 0
    best_metrics = None
    stale = 0
    checkpoint_path = output_dir / "polyphonic_query_joint_semantic_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.eval()
        model.decoder.train()
        model.slot_queries.train()
        model.objectness_head.train()
        model.box_head.train()
        model.semantic_head.train()
        totals: Counter[str] = Counter()
        batches = 0
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(batch["features"], batch["valid_mask"])
                temporal = relational_event_slots_v1_loss(
                    outputs, batch["target_intervals"], batch["frame_event_target"],
                    batch["frame_onset_target"], batch["valid_mask"],
                    weights=temporal_weights, no_object_weight=0.2,
                )
                class_losses = []
                for batch_index, query_index, target_labels, iou in _matched(outputs, batch):
                    keep = iou >= args.minimum_train_match_iou
                    if keep.any():
                        element = F.cross_entropy(
                            outputs["slot_class_logits"][batch_index, query_index[keep]],
                            target_labels[keep], weight=class_weights, reduction="none",
                        )
                        class_losses.append((element * (0.25 + 0.75 * iou[keep].detach())).mean())
                if not class_losses:
                    continue
                semantic_loss = torch.stack(class_losses).mean()
                loss = temporal["loss"] + args.semantic_loss_weight * semantic_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite joint query loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [value for value in model.parameters() if value.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.detach())
            totals["temporal_loss"] += float(temporal["loss"].detach())
            totals["semantic_loss"] += float(semantic_loss.detach())
            batches += 1
        scheduler.step()
        nonoverlap = evaluate(model, nonoverlap_dev_loader, device, classes=len(labels))
        overlap = evaluate(model, overlap_dev_loader, device, classes=len(labels))
        key = _selection_key(nonoverlap, overlap)
        row = {
            "epoch": epoch,
            "train": {name: value / max(batches, 1) for name, value in totals.items()},
            "nonoverlap_dev": nonoverlap,
            "overlap_dev": overlap,
            "selection_key": list(key),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_metrics, stale = key, epoch, row, 0
            current_frozen_hash = _state_hash(model.state_dict(), frozen_names)
            if current_frozen_hash != frozen_hash:
                raise RuntimeError("frozen encoder/frame state changed during joint training")
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT,
                    "epoch": epoch,
                    "config": asdict(model.config),
                    "num_classes": len(labels),
                    "labels": labels,
                    "model_state_dict": model.state_dict(),
                    "query_checkpoint": str(query_path),
                    "query_checkpoint_sha256": _sha256_file(query_path),
                    "frozen_encoder_state_sha256": frozen_hash,
                    "selection_metrics": row,
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_metrics is None:
        raise RuntimeError("joint query-semantic training produced no checkpoint")
    raw_macro = 0.5 * (
        float(best_metrics["nonoverlap_dev"]["raw_detector_oracle_interval_top1_accuracy_↑"])
        + float(best_metrics["overlap_dev"]["raw_detector_oracle_interval_top1_accuracy_↑"])
    )
    semantic_macro = 0.5 * (
        float(best_metrics["nonoverlap_dev"]["query_semantic_top1_accuracy_↑"])
        + float(best_metrics["overlap_dev"]["query_semantic_top1_accuracy_↑"])
    )
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "frozen_encoder_plus_joint_detr_interval_and_semantic_queries",
        "answer_label_used_as_input": False,
        "query_checkpoint": str(query_path),
        "query_checkpoint_sha256": _sha256_file(query_path),
        "frozen_encoder_state_sha256": frozen_hash,
        "trainable_parameter_names": trainable_names,
        "identity_audit": identity,
        "loss_weights": {**asdict(temporal_weights), "semantic": args.semantic_loss_weight},
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "dev_raw_oracle_interval_macro_top1": raw_macro,
        "dev_joint_query_semantic_macro_top1": semantic_macro,
        "beats_raw_oracle_interval_baseline": semantic_macro > raw_macro,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "history": history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "raw_macro": raw_macro, "semantic_macro": semantic_macro, "checkpoint_sha256": receipt["checkpoint_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
