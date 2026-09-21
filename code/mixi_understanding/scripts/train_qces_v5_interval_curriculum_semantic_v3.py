#!/usr/bin/env python3
"""Train V5 semantics with oracle-to-jitter-to-predicted interval curriculum.

The frozen localizer, BEATs features, detector logits, and long semantic
teacher are never updated.  QA questions/answers and separated waveforms are
not inputs.  Historical V2 code/checkpoints remain untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from mixi_understanding.scripts.train_qces_long_short_semantic_teacher_v2 import (
    LongHeadConfig,
    LongSemanticHead,
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
from mixi_understanding.scripts.train_qces_v5_rehearsal_slot_semantic_head_v2 import (
    _event_metrics,
    _harmonic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAT = "qces_v5_interval_curriculum_semantic_training_receipt_v3"
CACHE_FORMAT = "qces_v5_interval_curriculum_semantic_cache_v3"
CHECKPOINT_FORMAT = "qces_v5_interval_curriculum_semantic_checkpoint_v3"
NUM_FRAMES = 250


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    v2 = base / "v5_rehearsal_slot_semantic_head_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-train-index", type=Path, default=Path("/var/tmp/qces_v5_dense_train/index.json"))
    parser.add_argument("--main-dev-index", type=Path, default=Path("/var/tmp/qces_v5_dense_dev/index.json"))
    parser.add_argument("--stress-train-index", type=Path, default=Path("/var/tmp/qces_v4_dense_train/index.json"))
    parser.add_argument("--stress-dev-index", type=Path, default=Path("/var/tmp/qces_v4_dense_dev/index.json"))
    parser.add_argument("--main-train-scenes", type=Path, default=Path("/var/tmp/qces_full188_tiered_realistic_v5/scene_ids_tiered_train.txt"))
    parser.add_argument("--main-dev-scenes", type=Path, default=Path("/var/tmp/qces_full188_tiered_realistic_v5/scene_ids_tiered_dev.txt"))
    parser.add_argument("--stress-train-scenes", type=Path, default=PROJECT_ROOT / "outputs/qces_full188_overlap_semantic_sufficient_v4/scene_ids_overlap_train.txt")
    parser.add_argument("--stress-dev-scenes", type=Path, default=PROJECT_ROOT / "outputs/qces_full188_overlap_semantic_sufficient_v4/scene_ids_overlap_dev.txt")
    parser.add_argument("--main-train-predicted-cache", type=Path, default=v2 / "frozen_slots_main_train.pt")
    parser.add_argument("--main-dev-predicted-cache", type=Path, default=v2 / "frozen_slots_main_dev.pt")
    parser.add_argument("--stress-train-predicted-cache", type=Path, default=v2 / "frozen_slots_stress_train.pt")
    parser.add_argument("--stress-dev-predicted-cache", type=Path, default=v2 / "frozen_slots_stress_dev.pt")
    parser.add_argument("--long-teacher-checkpoint", type=Path, default=base / "long_short_semantic_teacher_v2/long_short_semantic_teacher_v2_best.pt")
    parser.add_argument("--v2-receipt", type=Path, default=v2 / "receipt.json")
    parser.add_argument("--output-dir", type=Path, default=base / "v5_interval_curriculum_semantic_v3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6503)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--oracle-epochs", type=int, default=8)
    parser.add_argument("--rehearsal-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--teacher-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--rehearsal-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--main-sampling-ratio", type=float, default=0.75)
    parser.add_argument("--oracle-branch-ratio", type=float, default=0.50)
    parser.add_argument("--none-class-weight", type=float, default=0.25)
    parser.add_argument("--max-train-scenes-per-domain", type=int, default=0)
    parser.add_argument("--max-dev-scenes-per-domain", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def interval_views(interval: torch.Tensor, key: str, training: bool) -> list[torch.Tensor]:
    exact = interval.float().clamp(0.0, 1.0)
    if not training:
        return [exact]
    digest = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    sign = -1.0 if digest % 2 else 1.0
    step = 2.0 / NUM_FRAMES
    expanded = torch.tensor((max(0.0, float(exact[0]) - step), min(1.0, float(exact[1]) + step)))
    shifted = torch.tensor((
        max(0.0, min(1.0 - step, float(exact[0]) + sign * step)),
        max(step, min(1.0, float(exact[1]) + sign * step)),
    ))
    if shifted[1] <= shifted[0]:
        shifted[1] = min(1.0, shifted[0] + step)
    return [exact, expanded, shifted]


def pooled_views(
    features: torch.Tensor, detector: torch.Tensor, intervals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = (intervals[:, 0] * NUM_FRAMES).floor().long().clamp(0, NUM_FRAMES - 1)
    ends = (intervals[:, 1] * NUM_FRAMES).ceil().long().clamp(1, NUM_FRAMES)
    rows_stats: list[torch.Tensor] = []
    rows_r1: list[torch.Tensor] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        end = max(start + 1, end)
        current = features[start:end].float()
        rows_stats.append(torch.cat((
            current.mean(dim=0),
            current.var(dim=0, unbiased=False).clamp_min(1e-6).sqrt(),
            current.amax(dim=0),
        )))
        rows_r1.append(detector[start:end].float().mean(dim=0))
    return torch.stack(rows_stats), torch.stack(rows_r1)


@torch.inference_mode()
def augment_with_teacher(
    stats: torch.Tensor,
    r1: torch.Tensor,
    teacher: LongSemanticHead,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    teacher.eval()
    for start in range(0, stats.shape[0], batch_size):
        current_stats = stats[start : start + batch_size].to(device)
        current_r1 = r1[start : start + batch_size].to(device)
        logits, embedding = teacher(current_stats, current_r1)
        # R1 must stay last because ResidualSlotSemanticHead uses it as the
        # calibrated zero-residual baseline.
        outputs.append(torch.cat((current_stats, embedding, logits, current_r1), dim=-1).cpu().half())
    return torch.cat(outputs)


@torch.inference_mode()
def build_domain_cache(
    store: DenseFeatureStore,
    scene_ids: Sequence[str],
    predicted: Mapping[str, Any],
    teacher: LongSemanticHead,
    device: torch.device,
    *,
    teacher_batch_size: int,
    training: bool,
    split: str,
) -> dict[str, Any]:
    position = {str(scene_id): index for index, scene_id in enumerate(predicted["scene_id"])}
    oracle_stats: list[torch.Tensor] = []
    oracle_r1: list[torch.Tensor] = []
    oracle_labels: list[int] = []
    oracle_exact: list[bool] = []
    predicted_stats: list[torch.Tensor] = []
    predicted_r1: list[torch.Tensor] = []
    selected_indices: list[int] = []
    for processed, scene_id in enumerate(scene_ids, start=1):
        cache_index = position.get(str(scene_id))
        if cache_index is None:
            raise KeyError(f"scene absent from predicted cache: {scene_id}")
        selected_indices.append(cache_index)
        dense = store.get(str(scene_id))
        features = dense["features"].float()
        detector = dense["logits"].float()
        if features.shape[0] != NUM_FRAMES or detector.shape[0] != NUM_FRAMES:
            raise ValueError(f"unexpected frame count for {scene_id}")
        pred_intervals = predicted["intervals"][cache_index].float()
        stats, r1 = pooled_views(features, detector, pred_intervals)
        predicted_stats.append(stats)
        predicted_r1.append(r1)
        intervals: list[torch.Tensor] = []
        labels: list[int] = []
        exact_flags: list[bool] = []
        for event_index, (interval, label) in enumerate(zip(
            predicted["gold_intervals"][cache_index], predicted["gold_labels"][cache_index]
        )):
            views = interval_views(interval, f"{scene_id}:{event_index}", training)
            intervals.extend(views)
            labels.extend([int(label)] * len(views))
            exact_flags.extend([view_index == 0 for view_index in range(len(views))])
        stats, r1 = pooled_views(features, detector, torch.stack(intervals))
        oracle_stats.append(stats)
        oracle_r1.append(r1)
        oracle_labels.extend(labels)
        oracle_exact.extend(exact_flags)
        if processed == 1 or processed % 512 == 0:
            print(f"interval_cache split={split} {processed}/{len(scene_ids)}", flush=True)
    oracle_stats_tensor = torch.cat(oracle_stats)
    oracle_r1_tensor = torch.cat(oracle_r1)
    predicted_stats_tensor = torch.cat(predicted_stats)
    predicted_r1_tensor = torch.cat(predicted_r1)
    oracle_input = augment_with_teacher(
        oracle_stats_tensor, oracle_r1_tensor, teacher, device, teacher_batch_size
    )
    predicted_flat = augment_with_teacher(
        predicted_stats_tensor, predicted_r1_tensor, teacher, device, teacher_batch_size
    )
    scenes = len(scene_ids)
    slots = int(predicted["slot_input"].shape[1])
    selected = torch.tensor(selected_indices, dtype=torch.long)
    return {
        "format": CACHE_FORMAT,
        "split": split,
        "scene_id": list(scene_ids),
        "oracle_input": oracle_input,
        "oracle_labels": torch.tensor(oracle_labels, dtype=torch.long),
        "oracle_exact": torch.tensor(oracle_exact, dtype=torch.bool),
        "predicted_input": predicted_flat.reshape(scenes, slots, -1),
        "objectness": predicted["objectness"][selected].clone(),
        "intervals": predicted["intervals"][selected].clone(),
        "target_slot_labels": predicted["target_slot_labels"][selected].clone(),
        "target_slot_weights": predicted["target_slot_weights"][selected].clone(),
        "target_slot_ious": predicted["target_slot_ious"][selected].clone(),
        "gold_intervals": [predicted["gold_intervals"][index] for index in selected_indices],
        "gold_labels": [predicted["gold_labels"][index] for index in selected_indices],
    }


class FlatDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, value: torch.Tensor, label: torch.Tensor, weight: torch.Tensor) -> None:
        self.value = value
        self.label = label.long()
        self.weight = weight.float()

    def __len__(self) -> int:
        return int(self.label.numel())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"value": self.value[index], "label": self.label[index], "weight": self.weight[index]}


def oracle_dataset(cache: Mapping[str, Any], exact_only: bool = False) -> FlatDataset:
    keep = cache["oracle_exact"] if exact_only else torch.ones_like(cache["oracle_exact"])
    count = int(keep.sum())
    return FlatDataset(cache["oracle_input"][keep], cache["oracle_labels"][keep], torch.ones(count))


def predicted_dataset(cache: Mapping[str, Any]) -> FlatDataset:
    return FlatDataset(
        cache["predicted_input"].reshape(-1, cache["predicted_input"].shape[-1]),
        cache["target_slot_labels"].reshape(-1),
        cache["target_slot_weights"].reshape(-1),
    )


def sampler_for_domains(
    datasets: Sequence[Dataset[Any]],
    group_weights: Sequence[float],
    samples: int,
    seed: int,
) -> WeightedRandomSampler:
    if len(datasets) != len(group_weights):
        raise ValueError("dataset/group weight mismatch")
    values: list[float] = []
    for dataset, group_weight in zip(datasets, group_weights):
        values.extend([float(group_weight) / max(len(dataset), 1)] * len(dataset))
    return WeightedRandomSampler(
        torch.tensor(values, dtype=torch.double),
        num_samples=samples,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def predicted_metrics(
    model: ResidualSlotSemanticHead | None,
    cache: Mapping[str, Any],
    device: torch.device,
    objectness_threshold: float,
) -> dict[str, Any]:
    value = cache["predicted_input"]
    if model is None:
        logits = F.pad(value[..., -NONE_LABEL:].float(), (0, 1), value=-10.0)
        learned = False
    else:
        parts: list[torch.Tensor] = []
        model.eval()
        flat = value.reshape(-1, value.shape[-1])
        for start in range(0, flat.shape[0], 512):
            parts.append(model(flat[start : start + 512].to(device)).cpu())
        logits = torch.cat(parts).reshape(value.shape[0], value.shape[1], NONE_LABEL + 1)
        learned = True
    return _event_metrics(cache, logits, objectness_threshold=objectness_threshold, learned=learned)


@torch.inference_mode()
def oracle_accuracy(model: ResidualSlotSemanticHead, cache: Mapping[str, Any], device: torch.device) -> dict[str, float]:
    keep = cache["oracle_exact"]
    value = cache["oracle_input"][keep]
    target = cache["oracle_labels"][keep]
    rows: list[torch.Tensor] = []
    model.eval()
    for start in range(0, value.shape[0], 512):
        rows.append(model(value[start : start + 512].to(device))[:, :NONE_LABEL].cpu())
    logits = torch.cat(rows)
    top = logits.topk(5, dim=-1).indices
    return {
        "events": int(target.numel()),
        "top1_accuracy_\u2191": float(top[:, 0].eq(target).float().mean()),
        "top5_accuracy_\u2191": float((top == target[:, None]).any(dim=-1).float().mean()),
    }


def selection(main: Mapping[str, Any], stress: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "main": dict(main),
        "stress": dict(stress),
        "harmonic_joint_f1": _harmonic(float(main["joint_label_iou50_f1_\u2191"]), float(stress["joint_label_iou50_f1_\u2191"])),
        "harmonic_label_top1": _harmonic(float(main["label_top1_given_iou50_\u2191"]), float(stress["label_top1_given_iou50_\u2191"])),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_lists = {
        "main_train": load_scene_list(args.main_train_scenes.resolve()),
        "main_dev": load_scene_list(args.main_dev_scenes.resolve()),
        "stress_train": load_scene_list(args.stress_train_scenes.resolve()),
        "stress_dev": load_scene_list(args.stress_dev_scenes.resolve()),
    }
    if args.max_train_scenes_per_domain:
        scene_lists["main_train"] = scene_lists["main_train"][: args.max_train_scenes_per_domain]
        scene_lists["stress_train"] = scene_lists["stress_train"][: args.max_train_scenes_per_domain]
    if args.max_dev_scenes_per_domain:
        scene_lists["main_dev"] = scene_lists["main_dev"][: args.max_dev_scenes_per_domain]
        scene_lists["stress_dev"] = scene_lists["stress_dev"][: args.max_dev_scenes_per_domain]
    index_paths = [
        args.main_train_index.resolve(), args.main_dev_index.resolve(),
        args.stress_train_index.resolve(), args.stress_dev_index.resolve(),
    ]
    store = DenseFeatureStore(index_paths, cache_size=24)
    identity_audit = assert_dense_identity_disjoint_v2(
        store,
        {"combined_train": scene_lists["main_train"] + scene_lists["stress_train"], "combined_dev": scene_lists["main_dev"] + scene_lists["stress_dev"]},
    )
    if len(store.labels or []) != NONE_LABEL:
        raise ValueError("ontology mismatch")
    predicted_paths = {
        "main_train": args.main_train_predicted_cache.resolve(),
        "main_dev": args.main_dev_predicted_cache.resolve(),
        "stress_train": args.stress_train_predicted_cache.resolve(),
        "stress_dev": args.stress_dev_predicted_cache.resolve(),
    }
    predicted = {key: torch.load(path, map_location="cpu", weights_only=True) for key, path in predicted_paths.items()}
    v2_receipt = json.loads(args.v2_receipt.resolve().read_text(encoding="utf-8"))
    objectness_threshold = float(v2_receipt["localizer"]["objectness_threshold"])
    teacher_payload = torch.load(args.long_teacher_checkpoint.resolve(), map_location="cpu", weights_only=True)
    teacher = LongSemanticHead(LongHeadConfig(**teacher_payload["long_config"]))
    teacher.load_state_dict(teacher_payload["long_model_state_dict"], strict=True)
    device = _device(args.device)
    teacher.to(device).eval()
    cache_paths = {key: output_dir / f"interval_curriculum_{key}.pt" for key in scene_lists}
    caches: dict[str, Mapping[str, Any]] = {}
    for key in ("main_train", "main_dev", "stress_train", "stress_dev"):
        cache = build_domain_cache(
            store, scene_lists[key], predicted[key], teacher, device,
            teacher_batch_size=args.teacher_batch_size,
            training=key.endswith("train"), split=key,
        )
        _atomic_torch(cache, cache_paths[key])
        caches[key] = cache
    del teacher, store
    if device.type == "cuda":
        torch.cuda.empty_cache()
    input_dim = int(caches["main_train"]["predicted_input"].shape[-1])
    model = ResidualSlotSemanticHead(input_dim, args.hidden_dim, NONE_LABEL, args.dropout).to(device)
    all_oracle_labels = torch.cat((caches["main_train"]["oracle_labels"], caches["stress_train"]["oracle_labels"]))
    counts = torch.bincount(all_oracle_labels, minlength=NONE_LABEL).float().clamp_min(1)
    class_weights = (counts.median() / counts).sqrt().clamp(0.5, 3.0)
    full_class_weights = torch.cat((class_weights, torch.tensor([args.none_class_weight]))).to(device)

    oracle_domains = [oracle_dataset(caches["main_train"]), oracle_dataset(caches["stress_train"])]
    oracle_combined = ConcatDataset(oracle_domains)
    oracle_sampler = sampler_for_domains(
        oracle_domains, [args.main_sampling_ratio, 1.0 - args.main_sampling_ratio],
        max(len(oracle_domains[0]), len(oracle_domains[1])), args.seed,
    )
    oracle_loader = DataLoader(oracle_combined, batch_size=args.batch_size, sampler=oracle_sampler, num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    oracle_history: list[dict[str, Any]] = []
    for epoch in range(1, args.oracle_epochs + 1):
        model.train(); total = correct = seen = 0
        for batch in oracle_loader:
            value = batch["value"].to(device); target = batch["label"].long().to(device)
            logits = model(value)[:, :NONE_LABEL]
            loss = F.cross_entropy(logits, target, weight=class_weights.to(device), label_smoothing=0.02)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            total += float(loss.detach()) * target.numel(); correct += int(logits.argmax(dim=-1).eq(target).sum()); seen += int(target.numel())
        row = {
            "epoch": epoch,
            "train_loss": total / max(seen, 1),
            "train_top1": correct / max(seen, 1),
            "oracle_dev_main": oracle_accuracy(model, caches["main_dev"], device),
            "oracle_dev_stress": oracle_accuracy(model, caches["stress_dev"], device),
        }
        oracle_history.append(row); print(json.dumps({"stage": "oracle", **row}, sort_keys=True), flush=True)

    stage1_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    predicted_domains = [predicted_dataset(caches["main_train"]), predicted_dataset(caches["stress_train"])]
    oracle_exact_domains = [oracle_dataset(caches["main_train"], exact_only=True), oracle_dataset(caches["stress_train"], exact_only=True)]
    rehearsal_domains = [*predicted_domains, *oracle_exact_domains]
    main = args.main_sampling_ratio; stress = 1.0 - main; oracle_ratio = args.oracle_branch_ratio
    group_weights = [
        (1.0 - oracle_ratio) * main, (1.0 - oracle_ratio) * stress,
        oracle_ratio * main, oracle_ratio * stress,
    ]
    rehearsal_combined = ConcatDataset(rehearsal_domains)
    samples_per_epoch = max(sum(len(value) for value in predicted_domains), sum(len(value) for value in oracle_exact_domains))
    rehearsal_sampler = sampler_for_domains(rehearsal_domains, group_weights, samples_per_epoch, args.seed + 1)
    rehearsal_loader = DataLoader(rehearsal_combined, batch_size=args.batch_size, sampler=rehearsal_sampler, num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.rehearsal_learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.rehearsal_epochs, eta_min=args.rehearsal_learning_rate * 0.05)
    baseline = selection(
        predicted_metrics(None, caches["main_dev"], device, objectness_threshold),
        predicted_metrics(None, caches["stress_dev"], device, objectness_threshold),
    )
    initial = selection(
        predicted_metrics(model, caches["main_dev"], device, objectness_threshold),
        predicted_metrics(model, caches["stress_dev"], device, objectness_threshold),
    )
    best = initial
    best_key = (float(initial["harmonic_joint_f1"]), float(initial["harmonic_label_top1"]))
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, args.rehearsal_epochs + 1):
        model.train(); total = correct = seen = 0
        for batch in rehearsal_loader:
            value = batch["value"].to(device); target = batch["label"].long().to(device); weight = batch["weight"].to(device)
            logits = model(value)
            element = F.cross_entropy(logits, target, weight=full_class_weights, reduction="none", label_smoothing=0.02)
            loss = (element * weight).sum() / weight.sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            total += float(loss.detach()) * target.numel(); correct += int(logits.argmax(dim=-1).eq(target).sum()); seen += int(target.numel())
        scheduler.step()
        current = selection(
            predicted_metrics(model, caches["main_dev"], device, objectness_threshold),
            predicted_metrics(model, caches["stress_dev"], device, objectness_threshold),
        )
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": total / max(seen, 1), "train_accuracy": correct / max(seen, 1), "selection": current}
        history.append(row); print(json.dumps({"stage": "rehearsal", **row}, sort_keys=True), flush=True)
        key = (float(current["harmonic_joint_f1"]), float(current["harmonic_label_top1"]))
        if key > best_key:
            best = current; best_key = key; best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}; stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True); break
    checkpoint_path = output_dir / "interval_curriculum_semantic_v3_best.pt"
    _atomic_torch({
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": best_state,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "none_label_id": NONE_LABEL,
        "objectness_threshold": objectness_threshold,
        "best_epoch": best_epoch,
        "best_selection": best,
        "stage1_state_dict": stage1_state,
    }, checkpoint_path)
    v2_best = v2_receipt["best_selection"]
    gates = {
        "main_top1_improves_v2": float(best["main"]["label_top1_given_iou50_\u2191"]) > float(v2_best["main"]["label_top1_given_iou50_\u2191"]),
        "main_top1_ge_0_65": float(best["main"]["label_top1_given_iou50_\u2191"]) >= 0.65,
        "stress_top1_ge_0_35": float(best["stress"]["label_top1_given_iou50_\u2191"]) >= 0.35,
        "harmonic_joint_improves_v2": float(best["harmonic_joint_f1"]) > float(v2_best["harmonic_joint_f1"]),
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qa_question_or_answer_used_as_input": False,
        "separated_waveform_used_as_input": False,
        "method": "oracle_jitter_predicted_interval_curriculum_with_beats_stats_and_frozen_long_teacher",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "identity_audit": identity_audit,
        "data": {key: len(value) for key, value in scene_lists.items()} | {"classes": NONE_LABEL, "input_dim": input_dim},
        "v2_reference": v2_best,
        "pooled_r1_baseline": baseline,
        "post_oracle_pretrain": initial,
        "oracle_history": oracle_history,
        "best_epoch": best_epoch,
        "best_selection": best,
        "rehearsal_history": history,
        "gates": gates,
        "decision": "run_event_graph_qa" if all(gates.values()) else "interval_curriculum_insufficient",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "long_teacher_checkpoint_sha256": _sha256_file(args.long_teacher_checkpoint.resolve()),
        "cache": {key: {"path": str(path), "sha256": _sha256_file(path)} for key, path in cache_paths.items()},
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "best": best, "gates": gates, "decision": receipt["decision"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
