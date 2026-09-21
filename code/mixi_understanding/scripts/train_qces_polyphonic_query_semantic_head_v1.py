#!/usr/bin/env python3
"""Train a semantic classifier on frozen polyphonic DETR query embeddings."""

from __future__ import annotations

import argparse
import json
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
from torch.utils.data import ConcatDataset, DataLoader

from mixi_understanding.qces.relational_event_slots_semantic_v1 import RelationalEventSlotsSemanticV1
from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1Config,
    hungarian_match_event_slots,
    interval_iou_matrix,
)
from mixi_understanding.scripts.train_qces_polyphonic_query_branch_v1 import (
    POLYPHONIC_CHECKPOINT_FORMAT,
    _state_hash,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    NUM_FRAMES,
    _atomic_json,
    _atomic_torch,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import assert_dense_identity_disjoint_v2
from mixi_understanding.scripts.train_qces_relational_event_slots_v1 import (
    SceneSlotDataset,
    collate_scene_slots,
)


FORMAT = "qces_polyphonic_query_semantic_head_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_polyphonic_query_semantic_head_checkpoint_v1"


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
        default=base / "polyphonic_query_semantic_head_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2111)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-train-match-iou", type=float, default=0.30)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class SemanticSceneDataset(SceneSlotDataset):
    def __init__(self, store: DenseFeatureStore, scene_ids: Sequence[str], *, preload: bool) -> None:
        super().__init__(store, scene_ids, preload=preload)
        label_to_id = {label: index for index, label in enumerate(store.labels or [])}
        self.target_labels = {
            scene_id: torch.tensor(
                [label_to_id[str(event["label"])] for event in self.gold[scene_id]["events"]],
                dtype=torch.long,
            )
            for scene_id in self.scene_ids
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = super().__getitem__(index)
        row["target_label_ids"] = self.target_labels[str(row["scene_id"])]
        return row


def collate_semantic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch = collate_scene_slots(rows)
    batch["target_label_ids"] = [row["target_label_ids"] for row in rows]
    return batch


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key in ("features", "detector_logits", "valid_mask"):
        result[key] = batch[key].to(device, non_blocking=True)
    result["target_intervals"] = [value.to(device, non_blocking=True) for value in batch["target_intervals"]]
    result["target_label_ids"] = [value.to(device, non_blocking=True) for value in batch["target_label_ids"]]
    return result


def _matched(
    outputs: Mapping[str, torch.Tensor], batch: Mapping[str, Any]
) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    matches = hungarian_match_event_slots(
        outputs["objectness_logits"], outputs["intervals"], batch["target_intervals"]
    )
    rows = []
    for batch_index, (query_index, target_index) in enumerate(matches):
        if not query_index.numel():
            continue
        prediction = outputs["intervals"][batch_index, query_index]
        target = batch["target_intervals"][batch_index][target_index]
        iou = interval_iou_matrix(prediction, target).diagonal()
        labels = batch["target_label_ids"][batch_index][target_index]
        rows.append((batch_index, query_index, labels, iou))
    return rows


def _class_counts(datasets: Sequence[SemanticSceneDataset], classes: int) -> torch.Tensor:
    count = torch.zeros(classes, dtype=torch.float32)
    for dataset in datasets:
        for labels in dataset.target_labels.values():
            count += torch.bincount(labels, minlength=classes).float()
    return count


@torch.inference_mode()
def evaluate(
    model: RelationalEventSlotsSemanticV1,
    loader: DataLoader,
    device: torch.device,
    *,
    classes: int,
) -> dict[str, Any]:
    model.eval()
    counters: Counter[str] = Counter()
    class_hit = torch.zeros(classes, dtype=torch.long)
    class_total = torch.zeros(classes, dtype=torch.long)
    raw_hit = raw_total = 0
    for raw in loader:
        batch = _to_device(raw, device)
        outputs = model(batch["features"], batch["valid_mask"])
        for batch_index, query_index, labels, iou in _matched(outputs, batch):
            score = outputs["slot_class_logits"][batch_index, query_index]
            ranks = score.argsort(dim=-1, descending=True)
            top1 = ranks[:, 0] == labels
            top5 = (ranks[:, :5] == labels[:, None]).any(dim=1)
            counters["events"] += len(labels)
            counters["top1"] += int(top1.sum())
            counters["top5"] += int(top5.sum())
            counters["iou30"] += int((iou >= 0.30).sum())
            counters["iou50"] += int((iou >= 0.50).sum())
            counters["joint30"] += int((top1 & (iou >= 0.30)).sum())
            counters["joint50"] += int((top1 & (iou >= 0.50)).sum())
            class_total += torch.bincount(labels.cpu(), minlength=classes)
            class_hit += torch.bincount(labels[top1].cpu(), minlength=classes)

            # Raw detector oracle-window baseline on exactly the same events.
            for local, target_index in enumerate(
                hungarian_match_event_slots(
                    outputs["objectness_logits"][batch_index:batch_index + 1],
                    outputs["intervals"][batch_index:batch_index + 1],
                    [batch["target_intervals"][batch_index]],
                )[0][1]
            ):
                interval = batch["target_intervals"][batch_index][target_index]
                start = max(0, min(NUM_FRAMES - 1, int(torch.floor(interval[0] * NUM_FRAMES).item())))
                end = max(start + 1, min(NUM_FRAMES, int(torch.ceil(interval[1] * NUM_FRAMES).item())))
                raw_prediction = raw["detector_logits"][batch_index, start:end].mean(dim=0).argmax()
                raw_hit += int(raw_prediction == labels[local].cpu())
                raw_total += 1
    total = max(int(counters["events"]), 1)
    observed = class_total > 0
    macro = float((class_hit[observed].float() / class_total[observed].float()).mean())
    return {
        "events": int(counters["events"]),
        "classes_observed": int(observed.sum()),
        "query_semantic_top1_accuracy_↑": counters["top1"] / total,
        "query_semantic_macro_top1_accuracy_↑": macro,
        "query_semantic_top5_accuracy_↑": counters["top5"] / total,
        "temporal_recall_iou30_↑": counters["iou30"] / total,
        "temporal_recall_iou50_↑": counters["iou50"] / total,
        "joint_label_and_iou30_recall_↑": counters["joint30"] / total,
        "joint_label_and_iou50_recall_↑": counters["joint50"] / total,
        "raw_detector_oracle_interval_top1_accuracy_↑": raw_hit / max(raw_total, 1),
    }


def _selection_key(nonoverlap: Mapping[str, Any], overlap: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        0.5 * (float(nonoverlap["joint_label_and_iou50_recall_↑"]) + float(overlap["joint_label_and_iou50_recall_↑"])),
        0.5 * (float(nonoverlap["query_semantic_top1_accuracy_↑"]) + float(overlap["query_semantic_top1_accuracy_↑"])),
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
    checkpoint_path = args.query_checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != POLYPHONIC_CHECKPOINT_FORMAT:
        raise ValueError("query checkpoint has wrong format")
    indexes = [
        args.nonoverlap_train_index.resolve(), args.nonoverlap_dev_index.resolve(),
        args.overlap_train_index.resolve(), args.overlap_dev_index.resolve(),
    ]
    store = DenseFeatureStore(indexes, cache_size=32)
    labels = list(store.labels or [])
    if labels != list(checkpoint.get("labels") or []):
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
        raise RuntimeError("semantic training requires equal non-overlap/overlap scene exposure")
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
        RelationalEventSlotsV1Config(**checkpoint["config"]), len(labels)
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if unexpected or any(not name.startswith("semantic_head.") for name in missing):
        raise RuntimeError(f"unexpected checkpoint incompatibility missing={missing} unexpected={unexpected}")
    model.requires_grad_(False)
    model.semantic_head.requires_grad_(True)
    frozen_names = [name for name in checkpoint["model_state_dict"]]
    frozen_hash = _state_hash(model.state_dict(), frozen_names)
    if frozen_hash != _state_hash(checkpoint["model_state_dict"], frozen_names):
        raise RuntimeError("base query state changed while attaching semantic head")

    counts = _class_counts(
        [datasets["nonoverlap_train"], datasets["overlap_train"]], len(labels)
    )
    class_weights = counts.clamp_min(1).rsqrt()
    class_weights = (class_weights / class_weights.mean()).to(device)
    optimizer = torch.optim.AdamW(
        model.semantic_head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history = []
    best_key = None
    best_epoch = 0
    best_metrics = None
    stale = 0
    saved = output_dir / "polyphonic_query_semantic_head_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.eval()
        model.semantic_head.train()
        loss_sum = used = 0.0
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(batch["features"], batch["valid_mask"])
                losses = []
                for batch_index, query_index, target_labels, iou in _matched(outputs, batch):
                    keep = iou >= args.minimum_train_match_iou
                    if keep.any():
                        element = F.cross_entropy(
                            outputs["slot_class_logits"][batch_index, query_index[keep]],
                            target_labels[keep], weight=class_weights, reduction="none",
                        )
                        losses.append((element * (0.25 + 0.75 * iou[keep].detach())).mean())
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.semantic_head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            used += 1
        scheduler.step()
        nonoverlap = evaluate(model, nonoverlap_dev_loader, device, classes=len(labels))
        overlap = evaluate(model, overlap_dev_loader, device, classes=len(labels))
        key = _selection_key(nonoverlap, overlap)
        row = {
            "epoch": epoch, "train_loss": loss_sum / max(used, 1),
            "nonoverlap_dev": nonoverlap, "overlap_dev": overlap,
            "selection_key": list(key),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_metrics, stale = key, epoch, row, 0
            if _state_hash(model.state_dict(), frozen_names) != frozen_hash:
                raise RuntimeError("frozen temporal query state changed during semantic training")
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT,
                    "epoch": epoch,
                    "config": asdict(model.config),
                    "num_classes": len(labels),
                    "labels": labels,
                    "model_state_dict": model.state_dict(),
                    "query_checkpoint": str(checkpoint_path),
                    "query_checkpoint_sha256": _sha256_file(checkpoint_path),
                    "frozen_temporal_state_sha256": frozen_hash,
                    "selection_metrics": row,
                },
                saved,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_metrics is None:
        raise RuntimeError("semantic training produced no checkpoint")
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
        "method": "frozen_polyphonic_query_embeddings_plus_hungarian_matched_semantic_head",
        "answer_label_used_as_input": False,
        "query_checkpoint": str(checkpoint_path),
        "query_checkpoint_sha256": _sha256_file(checkpoint_path),
        "frozen_temporal_state_sha256": frozen_hash,
        "identity_audit": identity,
        "training_exposure": {"nonoverlap": 0.5, "duration_balanced_overlap": 0.5},
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "dev_raw_oracle_interval_macro_top1": raw_macro,
        "dev_query_semantic_macro_top1": semantic_macro,
        "beats_raw_oracle_interval_baseline": semantic_macro > raw_macro,
        "checkpoint": str(saved),
        "checkpoint_sha256": _sha256_file(saved),
        "history": history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "raw_macro": raw_macro, "semantic_macro": semantic_macro, "checkpoint_sha256": receipt["checkpoint_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
