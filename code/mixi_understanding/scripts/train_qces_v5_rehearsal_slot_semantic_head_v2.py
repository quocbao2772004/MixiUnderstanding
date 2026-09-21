#!/usr/bin/env python3
"""Train 188-way semantics on frozen V5/V4 rehearsal event slots.

The frozen localizer and its pre-registered mask decoder first produce event
instances.  The semantic head then labels those instances from the slot state,
mask-component pooled BEATs features, and pooled frozen detector logits.  QA
questions and answers are never model inputs.  Historical trainers and
checkpoints remain untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from mixi_understanding.qces.overlap_event_slots_v2 import OverlapEventSlotsV2
from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1Config,
    interval_iou_matrix,
)
from mixi_understanding.scripts.evaluate_qces_slot_mask_interval_decoder_v1 import (
    _peak_component_interval,
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
from mixi_understanding.scripts.train_qces_v4_slot_semantic_head_v1 import (
    NONE_LABEL,
    ResidualSlotSemanticHead,
)


FORMAT = "qces_v5_rehearsal_slot_semantic_training_receipt_v2"
CACHE_FORMAT = "qces_v5_rehearsal_frozen_slot_semantic_cache_v2"
CHECKPOINT_FORMAT = "qces_v5_rehearsal_slot_semantic_checkpoint_v2"
NUM_FRAMES = 250


class SemanticSceneDataset(OverlapSceneSlotDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = super().__getitem__(index)
        events = self.gold[str(row["scene_id"])]["events"]
        row["target_labels"] = torch.tensor(
            [int(event["label_id"]) for event in events], dtype=torch.long
        )
        return row


def collate_semantic(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch = collate_overlap_slots(rows)
    batch["target_labels"] = [row["target_labels"] for row in rows]
    return batch


def _component_intervals_and_weights(
    slot_mask_probability: torch.Tensor, *, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return decoded normalized intervals and binary component weights on CPU."""
    if slot_mask_probability.ndim != 3:
        raise ValueError("expected [batch, slots, frames] mask probabilities")
    batch_size, slots, frames = slot_mask_probability.shape
    intervals = torch.zeros(batch_size, slots, 2, dtype=torch.float32)
    component = torch.zeros(batch_size, slots, frames, dtype=torch.float32)
    probability_cpu = slot_mask_probability.detach().float().cpu()
    for batch_index in range(batch_size):
        for slot_index in range(slots):
            interval = _peak_component_interval(
                probability_cpu[batch_index, slot_index], threshold
            )
            intervals[batch_index, slot_index] = interval
            start = max(0, min(frames - 1, int(math.floor(float(interval[0]) * frames))))
            end = max(start + 1, min(frames, int(math.ceil(float(interval[1]) * frames))))
            component[batch_index, slot_index, start:end] = 1.0
    return intervals, component


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
    mask_threshold: float,
    minimum_positive_iou: float,
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
    decoded_intervals: list[torch.Tensor] = []
    target_slot_labels: list[torch.Tensor] = []
    target_slot_weights: list[torch.Tensor] = []
    target_slot_ious: list[torch.Tensor] = []
    gold_intervals: list[torch.Tensor] = []
    gold_labels: list[torch.Tensor] = []
    oracle_ranks: list[list[int]] = []
    scene_ids: list[str] = []
    positive_matches = rejected_matches = 0
    localizer.eval()
    processed = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        detector = batch["detector_logits"].to(device, non_blocking=True)
        valid = batch["valid_mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(amp and device.type == "cuda"),
        ):
            outputs = localizer(features, valid)
        mask_probability = outputs["slot_mask_logits"].sigmoid()
        intervals_cpu, component_cpu = _component_intervals_and_weights(
            mask_probability, threshold=mask_threshold
        )
        component = component_cpu.to(device, non_blocking=True)
        component = component * valid[:, None].to(component.dtype)
        component = component / component.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled_features = torch.einsum("bst,btd->bsd", component, features.float())
        pooled_logits = torch.einsum("bst,btc->bsc", component, detector.float())
        semantic_input = torch.cat(
            (outputs["slot_embeddings"].float(), pooled_features, pooled_logits), dim=-1
        )

        batch_actual, slots = outputs["objectness_logits"].shape
        labels_for_slots = torch.full((batch_actual, slots), NONE_LABEL, dtype=torch.long)
        weights_for_slots = torch.full((batch_actual, slots), 0.25, dtype=torch.float32)
        ious_for_slots = torch.zeros(batch_actual, slots, dtype=torch.float32)
        for index in range(batch_actual):
            target_interval = batch["target_intervals"][index].float()
            iou = interval_iou_matrix(intervals_cpu[index], target_interval)
            rows, columns = linear_sum_assignment((1.0 - iou).numpy())
            for row, column in zip(rows, columns):
                value = float(iou[row, column])
                ious_for_slots[index, row] = value
                if value >= minimum_positive_iou:
                    labels_for_slots[index, row] = batch["target_labels"][index][column]
                    weights_for_slots[index, row] = 0.25 + 0.75 * value
                    positive_matches += 1
                else:
                    rejected_matches += 1

            ranks: list[int] = []
            dense = batch["detector_logits"][index]
            for interval, label in zip(
                batch["target_intervals"][index], batch["target_labels"][index]
            ):
                start = max(
                    0,
                    min(NUM_FRAMES - 1, int(math.floor(float(interval[0]) * NUM_FRAMES))),
                )
                end = max(
                    start + 1,
                    min(NUM_FRAMES, int(math.ceil(float(interval[1]) * NUM_FRAMES))),
                )
                score = dense[start:end].mean(dim=0)
                ordering = score.argsort(descending=True)
                ranks.append(int(torch.where(ordering == int(label))[0].item()) + 1)
            oracle_ranks.append(ranks)

        slot_inputs.append(semantic_input.detach().cpu().half())
        objectness.append(outputs["objectness_logits"].sigmoid().detach().cpu().half())
        decoded_intervals.append(intervals_cpu.half())
        target_slot_labels.append(labels_for_slots)
        target_slot_weights.append(weights_for_slots.half())
        target_slot_ious.append(ious_for_slots.half())
        gold_intervals.extend(value.float().cpu() for value in batch["target_intervals"])
        gold_labels.extend(value.long().cpu() for value in batch["target_labels"])
        scene_ids.extend(str(value) for value in batch["scene_id"])
        processed += batch_actual
        if processed == batch_actual or processed % 512 < batch_actual:
            print(f"semantic_cache split={split} {processed}/{len(dataset)}", flush=True)
    return {
        "format": CACHE_FORMAT,
        "split": split,
        "scene_id": scene_ids,
        "slot_input": torch.cat(slot_inputs),
        "objectness": torch.cat(objectness),
        "intervals": torch.cat(decoded_intervals),
        "target_slot_labels": torch.cat(target_slot_labels),
        "target_slot_weights": torch.cat(target_slot_weights),
        "target_slot_ious": torch.cat(target_slot_ious),
        "gold_intervals": gold_intervals,
        "gold_labels": gold_labels,
        "oracle_gold_span_ranks": oracle_ranks,
        "assignment_audit": {
            "minimum_positive_iou": minimum_positive_iou,
            "positive_matches": positive_matches,
            "rejected_low_iou_matches": rejected_matches,
        },
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


def _event_metrics(
    cache: Mapping[str, Any],
    class_logits: torch.Tensor,
    *,
    objectness_threshold: float,
    learned: bool,
) -> dict[str, Any]:
    predicted_count = gold_count = localization_tp = joint_tp = 0
    label_correct = label_top5 = label_top20 = matched_iou50 = none_predictions = 0
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
            if float(iou[row, column]) < 0.5:
                continue
            localization_tp += 1
            matched_iou50 += 1
            correct_label = int(gold_label[column])
            label_correct += int(int(predicted_labels[row]) == correct_label)
            ordering = scores[row].argsort(descending=True)
            label_top5 += int(correct_label in ordering[:5].tolist())
            label_top20 += int(correct_label in ordering[:20].tolist())
            joint_tp += int(int(predicted_labels[row]) == correct_label)

    def prf(true_positive: int) -> tuple[float, float, float]:
        precision = true_positive / max(predicted_count, 1)
        recall = true_positive / max(gold_count, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
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
        "label_top20_given_iou50_↑": label_top20 / max(matched_iou50, 1),
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
        logits = F.pad(values[..., -NONE_LABEL:].float(), (0, 1))
        learned = False
    else:
        model.eval()
        parts: list[torch.Tensor] = []
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
        "top1_accuracy_↑": sum(rank == 1 for rank in ranks) / max(len(ranks), 1),
        "top5_accuracy_↑": sum(rank <= 5 for rank in ranks) / max(len(ranks), 1),
        "top20_accuracy_↑": sum(rank <= 20 for rank in ranks) / max(len(ranks), 1),
        "mean_rank_↓": sum(ranks) / max(len(ranks), 1),
    }


def _harmonic(left: float, right: float) -> float:
    return 2.0 * left * right / max(left + right, 1e-12)


def _selection(main: Mapping[str, Any], stress: Mapping[str, Any]) -> dict[str, Any]:
    main_joint = float(main["joint_label_iou50_f1_↑"])
    stress_joint = float(stress["joint_label_iou50_f1_↑"])
    main_top1 = float(main["label_top1_given_iou50_↑"])
    stress_top1 = float(stress["label_top1_given_iou50_↑"])
    return {
        "main": dict(main),
        "stress": dict(stress),
        "harmonic_joint_f1": _harmonic(main_joint, stress_joint),
        "minimum_joint_f1": min(main_joint, stress_joint),
        "harmonic_label_top1": _harmonic(main_top1, stress_top1),
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["harmonic_joint_f1"]),
        float(row["minimum_joint_f1"]),
        float(row["harmonic_label_top1"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-train-index", type=Path, required=True)
    parser.add_argument("--main-dev-index", type=Path, required=True)
    parser.add_argument("--stress-train-index", type=Path, required=True)
    parser.add_argument("--stress-dev-index", type=Path, required=True)
    parser.add_argument("--main-train-scenes", type=Path, required=True)
    parser.add_argument("--main-dev-scenes", type=Path, required=True)
    parser.add_argument("--stress-train-scenes", type=Path, required=True)
    parser.add_argument("--stress-dev-scenes", type=Path, required=True)
    parser.add_argument("--slot-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6402)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=9)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--main-sampling-ratio", type=float, default=0.75)
    parser.add_argument("--minimum-positive-iou", type=float, default=0.30)
    parser.add_argument("--none-class-weight", type=float, default=0.25)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-scenes-per-domain", type=int, default=0)
    parser.add_argument("--max-dev-scenes-per-domain", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.main_sampling_ratio < 1.0:
        raise ValueError("main-sampling-ratio must lie strictly in (0,1)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_lists = {
        "main_train": load_scene_list(args.main_train_scenes),
        "main_dev": load_scene_list(args.main_dev_scenes),
        "stress_train": load_scene_list(args.stress_train_scenes),
        "stress_dev": load_scene_list(args.stress_dev_scenes),
    }
    if args.max_train_scenes_per_domain > 0:
        scene_lists["main_train"] = scene_lists["main_train"][: args.max_train_scenes_per_domain]
        scene_lists["stress_train"] = scene_lists["stress_train"][: args.max_train_scenes_per_domain]
    if args.max_dev_scenes_per_domain > 0:
        scene_lists["main_dev"] = scene_lists["main_dev"][: args.max_dev_scenes_per_domain]
        scene_lists["stress_dev"] = scene_lists["stress_dev"][: args.max_dev_scenes_per_domain]

    store = DenseFeatureStore(
        [
            args.main_train_index.resolve(),
            args.main_dev_index.resolve(),
            args.stress_train_index.resolve(),
            args.stress_dev_index.resolve(),
        ],
        cache_size=24,
    )
    identity_audit = assert_dense_identity_disjoint_v2(
        store,
        {
            "combined_train": scene_lists["main_train"] + scene_lists["stress_train"],
            "combined_dev": scene_lists["main_dev"] + scene_lists["stress_dev"],
        },
    )
    labels = list(store.labels or [])
    if len(labels) != NONE_LABEL:
        raise ValueError(f"expected {NONE_LABEL} labels, got {len(labels)}")
    datasets = {
        split: SemanticSceneDataset(store, ids, preload=True)
        for split, ids in scene_lists.items()
    }

    slot_path = args.slot_checkpoint.resolve()
    checkpoint = torch.load(slot_path, map_location="cpu", weights_only=True)
    config = RelationalEventSlotsV1Config(**checkpoint["config"])
    objectness_threshold = float(checkpoint["objectness_threshold"])
    decoder = checkpoint.get("mask_decoder") or {}
    mask_threshold = float(decoder.get("mask_threshold", 0.6))
    if float(decoder.get("mask_weight", 1.0)) != 1.0:
        raise ValueError("this trainer requires the pre-registered pure-mask decoder")
    device = _device(args.device)
    localizer = OverlapEventSlotsV2(config).to(device)
    localizer.load_state_dict(checkpoint["model_state_dict"], strict=True)
    localizer.eval().requires_grad_(False)

    cache_paths = {split: output_dir / f"frozen_slots_{split}.pt" for split in datasets}
    caches: dict[str, Mapping[str, Any]] = {}
    for split, dataset in datasets.items():
        payload = export_cache(
            localizer,
            dataset,
            device=device,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            amp=args.amp,
            split=split,
            mask_threshold=mask_threshold,
            minimum_positive_iou=args.minimum_positive_iou,
        )
        _atomic_torch(payload, cache_paths[split])
        caches[split] = payload
    del localizer, datasets, store
    if device.type == "cuda":
        torch.cuda.empty_cache()

    input_dim = int(caches["main_train"]["slot_input"].shape[-1])
    train_domains = [
        CachedSlotDataset(caches["main_train"]),
        CachedSlotDataset(caches["stress_train"]),
    ]
    combined_train = ConcatDataset(train_domains)
    main_weight = args.main_sampling_ratio / len(train_domains[0])
    stress_weight = (1.0 - args.main_sampling_ratio) / len(train_domains[1])
    sample_weights = torch.tensor(
        [main_weight] * len(train_domains[0]) + [stress_weight] * len(train_domains[1]),
        dtype=torch.double,
    )
    samples_per_epoch = round(len(train_domains[0]) / args.main_sampling_ratio)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        combined_train,
        batch_size=args.train_batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    baseline_main = evaluate_head(
        None, caches["main_dev"], device, objectness_threshold=objectness_threshold
    )
    baseline_stress = evaluate_head(
        None, caches["stress_dev"], device, objectness_threshold=objectness_threshold
    )
    baseline_selection = _selection(baseline_main, baseline_stress)
    oracle = {
        "main": oracle_summary(caches["main_dev"]),
        "stress": oracle_summary(caches["stress_dev"]),
    }
    print(json.dumps({"baseline": baseline_selection, "oracle_gold_span": oracle}, sort_keys=True), flush=True)

    model = ResidualSlotSemanticHead(
        input_dim, args.hidden_dim, NONE_LABEL, args.dropout
    ).to(device)
    all_targets = torch.cat(
        [
            caches["main_train"]["target_slot_labels"].reshape(-1),
            caches["stress_train"]["target_slot_labels"].reshape(-1),
        ]
    )
    counts = torch.bincount(all_targets, minlength=NONE_LABEL + 1).float()
    event_counts = counts[:NONE_LABEL].clamp_min(1)
    median = event_counts.median()
    class_weights = (median / event_counts).sqrt().clamp(0.5, 3.0)
    class_weights = torch.cat((class_weights, torch.tensor([args.none_class_weight]))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    initial_main = evaluate_head(
        model, caches["main_dev"], device, objectness_threshold=objectness_threshold
    )
    initial_stress = evaluate_head(
        model, caches["stress_dev"], device, objectness_threshold=objectness_threshold
    )
    best_selection = _selection(initial_main, initial_stress)
    best_key = _selection_key(best_selection)
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = correct = seen = 0
        for batch in train_loader:
            value = batch["slot_input"].to(device)
            target = batch["labels"].long().to(device)
            match_weight = batch["weights"].float().to(device)
            logits = model(value)
            element = F.cross_entropy(
                logits.reshape(-1, NONE_LABEL + 1),
                target.reshape(-1),
                weight=class_weights,
                reduction="none",
                label_smoothing=0.02,
            ).reshape_as(target)
            loss = (element * match_weight).sum() / match_weight.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            batch_count = int(target.numel())
            total_loss += float(loss.detach()) * batch_count
            correct += int((logits.argmax(dim=-1) == target).sum())
            seen += batch_count
        scheduler.step()
        main_metrics = evaluate_head(
            model, caches["main_dev"], device, objectness_threshold=objectness_threshold
        )
        stress_metrics = evaluate_head(
            model, caches["stress_dev"], device, objectness_threshold=objectness_threshold
        )
        selection = _selection(main_metrics, stress_metrics)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": total_loss / max(seen, 1),
            "train_slot_accuracy": correct / max(seen, 1),
            "selection": selection,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = _selection_key(selection)
        if key > best_key:
            best_key = key
            best_selection = selection
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    checkpoint_path = output_dir / "slot_semantic_head_v2_best.pt"
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
            "mask_threshold": mask_threshold,
            "best_epoch": best_epoch,
            "best_selection": best_selection,
        },
        checkpoint_path,
    )
    gates = {
        "harmonic_joint_f1_improves_baseline": float(best_selection["harmonic_joint_f1"])
        > float(baseline_selection["harmonic_joint_f1"]),
        "main_label_top1_ge_0_50": float(best_selection["main"]["label_top1_given_iou50_↑"])
        >= 0.50,
        "stress_label_top1_ge_0_40": float(best_selection["stress"]["label_top1_given_iou50_↑"])
        >= 0.40,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_eligible": False,
        "qa_question_or_answer_used_as_input": False,
        "method": "frozen_rehearsal_localizer_plus_mask_component_pooled_residual_semantic_head",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "config": asdict(config),
        "identity_audit": identity_audit,
        "data": {key: len(value) for key, value in scene_lists.items()} | {"classes": len(labels)},
        "localizer": {
            "checkpoint": str(slot_path),
            "sha256": _sha256_file(slot_path),
            "objectness_threshold": objectness_threshold,
            "mask_threshold": mask_threshold,
        },
        "assignment_audit": {
            split: caches[split]["assignment_audit"] for split in caches
        },
        "oracle_gold_span_r1": oracle,
        "pooled_r1_baseline": baseline_selection,
        "best_epoch": best_epoch,
        "best_selection": best_selection,
        "success_gates": gates,
        "all_success_gates_pass": all(gates.values()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "cache": {
            split: {"path": str(path), "sha256": _sha256_file(path)}
            for split, path in cache_paths.items()
        },
        "history": history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "baseline": baseline_selection, "best": best_selection, "gates": gates}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
