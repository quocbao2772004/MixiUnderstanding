#!/usr/bin/env python3
"""Train a residual 188-class semantic head on frozen V4 event slots.

The frozen localizer proposes class-agnostic event instances.  For each slot,
this script concatenates the slot embedding, mask-pooled pre-head BEATs
features, and mask-pooled frozen-R1 class logits.  A residual head predicts the
188 event labels plus NONE.  It starts as the exact pooled-R1 baseline and is
kept only when source-disjoint development metrics improve.

Questions and QA answers are never model inputs.  Gold event labels are used
only as semantic training targets after Hungarian slot assignment.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.overlap_event_slots_v2 import OverlapEventSlotsV2
from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1Config,
    hungarian_match_event_slots,
    interval_iou_matrix,
)
from mixi_understanding.scripts.train_qces_overlap_event_slots_v2 import (
    OverlapSceneSlotDataset,
    collate_overlap_slots,
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


FORMAT = "qces_v4_slot_semantic_training_receipt_v1"
CACHE_FORMAT = "qces_v4_frozen_slot_semantic_cache_v1"
CHECKPOINT_FORMAT = "qces_v4_slot_semantic_checkpoint_v1"
NUM_FRAMES = 250
NONE_LABEL = 188


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_semantic_sufficient_v4"
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", type=Path, default=Path("/var/tmp/qces_v4_dense_train/index.json"))
    parser.add_argument("--dev-index", type=Path, default=Path("/var/tmp/qces_v4_dense_dev/index.json"))
    parser.add_argument("--train-scenes", type=Path, default=data / "scene_ids_overlap_train.txt")
    parser.add_argument("--dev-scenes", type=Path, default=data / "scene_ids_overlap_dev.txt")
    parser.add_argument("--slot-checkpoint", type=Path, default=base / "overlap_event_slots_v2_semantic_v4/overlap_event_slots_v2_best.pt")
    parser.add_argument("--slot-receipt", type=Path, default=base / "overlap_event_slots_v2_semantic_v4/receipt.json")
    parser.add_argument("--output-dir", type=Path, default=base / "v4_slot_semantic_head_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6401)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--minimum-match-weight", type=float, default=0.25)
    parser.add_argument("--none-class-weight", type=float, default=0.25)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class SemanticSceneDataset(OverlapSceneSlotDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = super().__getitem__(index)
        events = self.gold[str(row["scene_id"])]["events"]
        row["target_labels"] = torch.tensor([int(event["label_id"]) for event in events], dtype=torch.long)
        return row


def collate_semantic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch = collate_overlap_slots(rows)
    batch["target_labels"] = [row["target_labels"] for row in rows]
    return batch


def _pool_slots(
    features: torch.Tensor,
    detector_logits: torch.Tensor,
    slot_mask_logits: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = slot_mask_logits.sigmoid() * valid_mask[:, None].to(slot_mask_logits.dtype)
    weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1e-5)
    pooled_features = torch.einsum("bst,btd->bsd", weight, features)
    pooled_logits = torch.einsum("bst,btc->bsc", weight, detector_logits)
    return pooled_features, pooled_logits


@torch.inference_mode()
def export_cache(
    localizer: OverlapEventSlotsV2,
    dataset: SemanticSceneDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp: bool,
    split: str,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_semantic,
    )
    slot_inputs: list[torch.Tensor] = []
    objectness: list[torch.Tensor] = []
    intervals: list[torch.Tensor] = []
    target_slot_labels: list[torch.Tensor] = []
    target_slot_weights: list[torch.Tensor] = []
    target_slot_ious: list[torch.Tensor] = []
    gold_intervals: list[torch.Tensor] = []
    gold_labels: list[torch.Tensor] = []
    oracle_ranks: list[list[int]] = []
    scene_ids: list[str] = []
    processed = 0
    localizer.eval()
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        detector = batch["detector_logits"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            outputs = localizer(features, valid)
            pooled_features, pooled_logits = _pool_slots(
                features, detector, outputs["slot_mask_logits"], valid
            )
            semantic_input = torch.cat(
                (outputs["slot_embeddings"], pooled_features, pooled_logits), dim=-1
            )
        matches = hungarian_match_event_slots(
            outputs["objectness_logits"], outputs["intervals"],
            [value.to(device) for value in batch["target_intervals"]],
        )
        batch_size_actual, slots = outputs["objectness_logits"].shape
        labels_for_slots = torch.full((batch_size_actual, slots), NONE_LABEL, dtype=torch.long)
        weights_for_slots = torch.full((batch_size_actual, slots), 0.25, dtype=torch.float32)
        ious_for_slots = torch.zeros(batch_size_actual, slots, dtype=torch.float32)
        for index, (prediction_index, target_index) in enumerate(matches):
            predicted = outputs["intervals"][index, prediction_index].float().cpu()
            prediction_cpu = prediction_index.cpu()
            target_cpu = target_index.cpu()
            target = batch["target_intervals"][index][target_cpu].float()
            values = interval_iou_matrix(predicted, target).diag()
            labels_for_slots[index, prediction_cpu] = batch["target_labels"][index][target_cpu]
            ious_for_slots[index, prediction_cpu] = values
            weights_for_slots[index, prediction_cpu] = 0.25 + 0.75 * values

            ranks: list[int] = []
            dense = batch["detector_logits"][index]
            for interval, label in zip(batch["target_intervals"][index], batch["target_labels"][index]):
                start = max(0, min(NUM_FRAMES - 1, int(math.floor(float(interval[0]) * NUM_FRAMES))))
                end = max(start + 1, min(NUM_FRAMES, int(math.ceil(float(interval[1]) * NUM_FRAMES))))
                score = dense[start:end].mean(dim=0)
                ordering = score.argsort(descending=True)
                ranks.append(int(torch.where(ordering == int(label))[0].item()) + 1)
            oracle_ranks.append(ranks)
        slot_inputs.append(semantic_input.detach().cpu().half())
        objectness.append(outputs["objectness_logits"].sigmoid().detach().cpu().half())
        intervals.append(outputs["intervals"].detach().cpu().half())
        target_slot_labels.append(labels_for_slots)
        target_slot_weights.append(weights_for_slots.half())
        target_slot_ious.append(ious_for_slots.half())
        gold_intervals.extend(value.float().cpu() for value in batch["target_intervals"])
        gold_labels.extend(value.long().cpu() for value in batch["target_labels"])
        scene_ids.extend(str(value) for value in batch["scene_id"])
        processed += batch_size_actual
        if processed == batch_size_actual or processed % 512 < batch_size_actual:
            print(f"semantic_cache split={split} {processed}/{len(dataset)}", flush=True)
    return {
        "format": CACHE_FORMAT,
        "split": split,
        "scene_id": scene_ids,
        "slot_input": torch.cat(slot_inputs),
        "objectness": torch.cat(objectness),
        "intervals": torch.cat(intervals),
        "target_slot_labels": torch.cat(target_slot_labels),
        "target_slot_weights": torch.cat(target_slot_weights),
        "target_slot_ious": torch.cat(target_slot_ious),
        "gold_intervals": gold_intervals,
        "gold_labels": gold_labels,
        "oracle_gold_span_ranks": oracle_ranks,
    }


class CachedSlotDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.slot_input = payload["slot_input"]
        self.labels = payload["target_slot_labels"]
        self.weights = payload["target_slot_weights"]

    def __len__(self) -> int:
        return int(self.slot_input.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "slot_input": self.slot_input[index],
            "labels": self.labels[index],
            "weights": self.weights[index],
        }


class ResidualSlotSemanticHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.residual = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.none_head = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 1)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        nn.init.zeros_(self.none_head[-1].weight)
        nn.init.constant_(self.none_head[-1].bias, -10.0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        r1 = value[..., -self.num_classes :].float()
        classes = r1 + self.residual(value.float())
        return torch.cat((classes, self.none_head(value.float())), dim=-1)


def _event_metrics(
    cache: Mapping[str, Any],
    class_logits: torch.Tensor,
    *,
    objectness_threshold: float,
    learned: bool,
) -> dict[str, Any]:
    predicted_count = gold_count = localization_tp = joint_tp = 0
    label_correct = label_top5 = matched_iou50 = 0
    none_predictions = 0
    for index in range(int(class_logits.shape[0])):
        keep = cache["objectness"][index].float() >= objectness_threshold
        predicted_intervals = cache["intervals"][index].float()[keep]
        scores = class_logits[index][keep].float()
        if learned:
            predicted_labels = scores.argmax(dim=-1)
            semantic_keep = predicted_labels != NONE_LABEL
            none_predictions += int((~semantic_keep).sum())
            predicted_intervals = predicted_intervals[semantic_keep]
            scores = scores[semantic_keep, :NONE_LABEL]
            predicted_labels = predicted_labels[semantic_keep]
        else:
            scores = scores[:, :NONE_LABEL]
            predicted_labels = scores.argmax(dim=-1)
        gold_interval = cache["gold_intervals"][index].float()
        gold_label = cache["gold_labels"][index].long()
        predicted_count += int(predicted_intervals.shape[0])
        gold_count += int(gold_interval.shape[0])
        if predicted_intervals.numel() == 0:
            continue
        iou = interval_iou_matrix(predicted_intervals, gold_interval)
        rows, columns = linear_sum_assignment((1.0 - iou).numpy())
        for row, column in zip(rows, columns):
            value = float(iou[row, column])
            if value < 0.5:
                continue
            localization_tp += 1
            matched_iou50 += 1
            correct = int(predicted_labels[row]) == int(gold_label[column])
            label_correct += int(correct)
            label_top5 += int(int(gold_label[column]) in scores[row].topk(5).indices.tolist())
            joint_tp += int(correct)

    def prf(tp: int) -> tuple[float, float, float]:
        precision = tp / max(predicted_count, 1)
        recall = tp / max(gold_count, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return precision, recall, f1

    loc_precision, loc_recall, loc_f1 = prf(localization_tp)
    precision, recall, f1 = prf(joint_tp)
    return {
        "scenes": int(class_logits.shape[0]),
        "predicted_events": predicted_count,
        "gold_events": gold_count,
        "none_predictions_after_objectness": none_predictions,
        "localization_precision_iou50": loc_precision,
        "localization_recall_iou50": loc_recall,
        "localization_f1_iou50": loc_f1,
        "joint_label_iou50_precision_↑": precision,
        "joint_label_iou50_recall_↑": recall,
        "joint_label_iou50_f1_↑": f1,
        "label_top1_given_iou50_↑": label_correct / max(matched_iou50, 1),
        "label_top5_given_iou50_↑": label_top5 / max(matched_iou50, 1),
    }


@torch.inference_mode()
def evaluate_head(
    model: ResidualSlotSemanticHead | None,
    cache: Mapping[str, Any],
    device: torch.device,
    *,
    objectness_threshold: float,
) -> dict[str, Any]:
    values = cache["slot_input"]
    if model is None:
        logits = values[..., -NONE_LABEL:].float()
        # Pad NONE only to share the evaluator contract; it is ignored for R1.
        logits = F.pad(logits, (0, 1))
        learned = False
    else:
        model.eval()
        parts = []
        for start in range(0, int(values.shape[0]), 256):
            parts.append(model(values[start : start + 256].to(device)).cpu())
        logits = torch.cat(parts)
        learned = True
    return _event_metrics(
        cache, logits, objectness_threshold=objectness_threshold, learned=learned
    )


def oracle_summary(cache: Mapping[str, Any]) -> dict[str, Any]:
    ranks = [rank for scene in cache["oracle_gold_span_ranks"] for rank in scene]
    return {
        "events": len(ranks),
        "top1_accuracy_↑": sum(rank == 1 for rank in ranks) / len(ranks),
        "top5_accuracy_↑": sum(rank <= 5 for rank in ranks) / len(ranks),
        "top20_accuracy_↑": sum(rank <= 20 for rank in ranks) / len(ranks),
        "mean_rank_↓": sum(ranks) / len(ranks),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    store = DenseFeatureStore([args.train_index.resolve(), args.dev_index.resolve()], cache_size=20)
    train_ids = load_scene_list(args.train_scenes.resolve())
    dev_ids = load_scene_list(args.dev_scenes.resolve())
    if args.max_train_scenes:
        train_ids = train_ids[: args.max_train_scenes]
    if args.max_dev_scenes:
        dev_ids = dev_ids[: args.max_dev_scenes]
    identity_audit = assert_dense_identity_disjoint_v2(store, {"train": train_ids, "dev": dev_ids})
    labels = list(store.labels or [])
    if len(labels) != NONE_LABEL:
        raise ValueError(f"expected {NONE_LABEL} labels, got {len(labels)}")
    datasets = {
        "train": SemanticSceneDataset(store, train_ids, preload=True),
        "dev": SemanticSceneDataset(store, dev_ids, preload=True),
    }
    checkpoint = torch.load(args.slot_checkpoint.resolve(), map_location="cpu", weights_only=True)
    config = RelationalEventSlotsV1Config(**checkpoint["config"])
    device = _device(args.device)
    localizer = OverlapEventSlotsV2(config).to(device)
    localizer.load_state_dict(checkpoint["model_state_dict"], strict=True)
    localizer.eval().requires_grad_(False)
    receipt = json.loads(args.slot_receipt.resolve().read_text(encoding="utf-8"))
    objectness_threshold = float(receipt["best_selection"]["objectness_threshold"])

    cache_paths = {split: output_dir / f"frozen_slots_{split}.pt" for split in ("train", "dev")}
    caches: dict[str, Mapping[str, Any]] = {}
    for split in ("train", "dev"):
        payload = export_cache(
            localizer, datasets[split], device=device,
            batch_size=args.feature_batch_size, num_workers=args.num_workers,
            amp=args.amp, split=split,
        )
        _atomic_torch(payload, cache_paths[split])
        caches[split] = payload
    del localizer, datasets, store
    if device.type == "cuda":
        torch.cuda.empty_cache()

    input_dim = int(caches["train"]["slot_input"].shape[-1])
    train_dataset = CachedSlotDataset(caches["train"])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        generator=generator, num_workers=0, pin_memory=device.type == "cuda",
    )
    model = ResidualSlotSemanticHead(input_dim, args.hidden_dim, NONE_LABEL, args.dropout).to(device)
    initial_metrics = evaluate_head(None, caches["dev"], device, objectness_threshold=objectness_threshold)
    oracle_metrics = oracle_summary(caches["dev"])
    counts = torch.bincount(caches["train"]["target_slot_labels"].reshape(-1), minlength=NONE_LABEL + 1).float()
    event_counts = counts[:NONE_LABEL].clamp_min(1)
    median = event_counts.median()
    class_weights = (median / event_counts).sqrt().clamp(0.5, 3.0)
    class_weights = torch.cat((class_weights, torch.tensor([args.none_class_weight])))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    history: list[dict[str, Any]] = []
    best_metrics = evaluate_head(model, caches["dev"], device, objectness_threshold=objectness_threshold)
    best_key = (
        float(best_metrics["joint_label_iou50_f1_↑"]),
        float(best_metrics["label_top1_given_iou50_↑"]),
    )
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = seen = 0
        for batch in train_loader:
            value = batch["slot_input"].to(device)
            target = batch["labels"].long().to(device)
            match_weight = batch["weights"].float().to(device)
            logits = model(value)
            loss_element = F.cross_entropy(
                logits.reshape(-1, NONE_LABEL + 1), target.reshape(-1),
                weight=class_weights.to(device), reduction="none", label_smoothing=0.02,
            ).reshape_as(target)
            loss = (loss_element * match_weight).sum() / match_weight.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * target.numel()
            correct += int((logits.argmax(dim=-1) == target).sum())
            seen += int(target.numel())
        scheduler.step()
        metrics = evaluate_head(model, caches["dev"], device, objectness_threshold=objectness_threshold)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": total / seen, "train_slot_accuracy": correct / seen, "dev": metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (
            float(metrics["joint_label_iou50_f1_↑"]),
            float(metrics["label_top1_given_iou50_↑"]),
        )
        if key > best_key:
            best_key = key
            best_metrics = metrics
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    checkpoint_path = output_dir / "v4_slot_semantic_head_v1_best.pt"
    _atomic_torch(
        {
            "format": CHECKPOINT_FORMAT,
            "labels": labels,
            "none_label_id": NONE_LABEL,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "model_state_dict": best_state,
            "objectness_threshold": objectness_threshold,
            "best_epoch": best_epoch,
            "dev_metrics": best_metrics,
        }, checkpoint_path,
    )
    gates = {
        "joint_f1_ge_0_45": float(best_metrics["joint_label_iou50_f1_↑"]) >= 0.45,
        "label_top1_given_iou50_ge_0_60": float(best_metrics["label_top1_given_iou50_↑"]) >= 0.60,
        "joint_f1_improves_r1_by_0_03": float(best_metrics["joint_label_iou50_f1_↑"]) >= float(initial_metrics["joint_label_iou50_f1_↑"]) + 0.03,
    }
    output_receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_eligible": False,
        "claim_boundary": "full development diagnostic; final claims require a new locked source-disjoint test",
        "qa_question_or_answer_used_as_input": False,
        "method": "frozen_localizer_plus_slot_embedding_mask_pooled_beats_and_r1_residual_semantic_head",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "identity_audit": identity_audit,
        "data": {"train_scenes": len(train_ids), "dev_scenes": len(dev_ids), "classes": len(labels)},
        "objectness_threshold": objectness_threshold,
        "oracle_gold_span_r1": oracle_metrics,
        "pooled_r1_baseline": initial_metrics,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "success_gates": gates,
        "all_success_gates_pass": all(gates.values()),
        "decision": "proceed_to_event_graph" if all(gates.values()) else "semantic_slot_bottleneck_requires_separator_or_stronger_encoder",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "slot_checkpoint_sha256": _sha256_file(args.slot_checkpoint.resolve()),
        "cache": {split: {"path": str(path), "sha256": _sha256_file(path)} for split, path in cache_paths.items()},
        "history": history,
    }
    _atomic_json(output_receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "oracle": oracle_metrics, "r1_baseline": initial_metrics, "best_epoch": best_epoch, "best": best_metrics, "gates": gates, "decision": output_receipt["decision"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
