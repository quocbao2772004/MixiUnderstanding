#!/usr/bin/env python3
"""R1: exact BEATs-Strong head transfer plus quality-aware curriculum.

The 191 QCES labels are an exact subset of the 447 AudioSet-Strong labels in
the public checkpoint.  Unlike historical QCES runs, this trainer copies the
matching pretrained strong-head rows instead of discarding the whole head when
the output dimension changes.  The frozen backbone is cached once, then:

1. Gold single-event stems adapt the transferred head conservatively.
2. Gold+Silver single events and rendered 3--6 event scenes are trained with
   inverse-sqrt class sampling and a lower Silver loss weight.
3. A small clip-presence auxiliary loss suppresses absent-class false alarms.

Timestamp targets retain the corrected fixed 10 s / 250 frame contract.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline" / "PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(value))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data_util.audioset_classes import as_strong_train_classes
from mixi_understanding.qces.fixed_grid_detector import (
    FRAME_HOP_SECONDS,
    NUM_FRAMES,
    _match_events,
    build_fixed_grid_targets,
    decode_fixed_grid_events,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    SceneDataset,
    SceneItem,
    collate,
    load_model,
    load_ontology,
    load_scene_manifest,
    make_device,
    set_seed,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector_fixedgrid_r0 import (
    atomic_json,
    atomic_torch,
    aggregate_class_counts,
    cache_frozen_features,
    collect_predictions,
    compact_summary,
    duration_and_tier_audit,
    fixed_grid_pos_weight,
    head_logits,
    summarize_predictions,
    threshold_key,
)


FORMAT = "qces_pretrainedsed_headtransfer_r1_v1"
DEFAULT_DATA = PROJECT_ROOT / "outputs" / "qces_full191_r1_data_v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "qces_pretrainedsed_detector_full191" / "headtransfer_r1_v1"
)


class IndexedFeatureDataset(Dataset[tuple[torch.Tensor, SceneItem, float]]):
    def __init__(
        self,
        features: torch.Tensor,
        rows: Sequence[SceneItem],
        indices: Sequence[int] | None = None,
        *,
        silver_weight: float,
    ) -> None:
        if features.ndim != 3 or features.shape[0] != len(rows):
            raise ValueError("feature/row mismatch")
        self.features = features
        self.rows = list(rows)
        self.indices = list(range(len(rows))) if indices is None else list(indices)
        self.silver_weight = float(silver_weight)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, SceneItem, float]:
        source_index = self.indices[index]
        row = self.rows[source_index]
        return self.features[source_index], row, row_quality_weight(row, self.silver_weight)


def feature_collate(
    batch: Sequence[tuple[torch.Tensor, SceneItem, float]],
) -> tuple[torch.Tensor, list[SceneItem], torch.Tensor]:
    return (
        torch.stack([item[0] for item in batch]),
        [item[1] for item in batch],
        torch.tensor([item[2] for item in batch], dtype=torch.float32),
    )


def eval_feature_collate(
    batch: Sequence[tuple[torch.Tensor, SceneItem, float]],
) -> tuple[torch.Tensor, list[SceneItem]]:
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-train", type=Path, default=DEFAULT_DATA / "detector_scene_manifest_single_train.jsonl"
    )
    parser.add_argument(
        "--multi-train", type=Path, default=DEFAULT_DATA / "detector_scene_manifest_multi_train.jsonl"
    )
    parser.add_argument(
        "--dev-manifest", type=Path, default=DEFAULT_DATA / "detector_scene_manifest_multi_dev.jsonl"
    )
    parser.add_argument(
        "--test-manifest", type=Path, default=DEFAULT_DATA / "detector_scene_manifest_multi_test.jsonl"
    )
    parser.add_argument("--ontology", type=Path, default=DEFAULT_DATA / "ontology_191.txt")
    parser.add_argument(
        "--label-map", type=Path, default=DEFAULT_DATA / "pretrained_head_label_map.json"
    )
    parser.add_argument("--audio-root", type=Path, default=Path("/"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="BEATs_strong_1")
    parser.add_argument("--gold-epochs", type=int, default=3)
    parser.add_argument("--mixed-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--gold-lr", type=float, default=3e-4)
    parser.add_argument("--mixed-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--silver-weight", type=float, default=0.35)
    parser.add_argument("--clip-loss-weight", type=float, default=0.20)
    parser.add_argument("--clip-temperature", type=float, default=0.50)
    parser.add_argument("--max-pos-weight", type=float, default=80.0)
    parser.add_argument("--max-clip-pos-weight", type=float, default=20.0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop a curriculum stage after this many epochs without dev event-F1 improvement; 0 disables.",
    )
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    )
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument("--min-duration", type=float, default=0.08)
    parser.add_argument("--merge-gap", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--max-single-train", type=int, default=0)
    parser.add_argument("--max-multi-train", type=int, default=0)
    parser.add_argument("--max-dev", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.gold_epochs < 0 or args.mixed_epochs < 0 or args.gold_epochs + args.mixed_epochs < 1:
        parser.error("training requires at least one epoch")
    if not 0 < args.silver_weight <= 1:
        parser.error("--silver-weight must be in (0,1]")
    if args.batch_size < 1 or args.feature_batch_size < 1:
        parser.error("batch sizes must be positive")
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        parser.error("early-stopping parameters must be non-negative")
    return args


def row_quality_weight(row: SceneItem, silver_weight: float) -> float:
    tiers = [str(event.get("cleanliness_tier") or "gold").lower() for event in row.events]
    if not tiers:
        return 1.0
    return float(np.mean([1.0 if tier == "gold" else silver_weight for tier in tiers]))


def is_gold_single(row: SceneItem) -> bool:
    return len(row.events) == 1 and str(row.events[0].get("cleanliness_tier") or "").lower() == "gold"


def transfer_pretrained_head(
    model: torch.nn.Module,
    *,
    labels: Sequence[str],
    label_map_path: Path,
    checkpoint: str,
) -> dict[str, Any]:
    mapping = json.loads(label_map_path.read_text(encoding="utf-8"))
    checkpoint_path = PRETRAINED_SED_ROOT / "resources" / f"{checkpoint}.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weight = payload["strong_head.weight"]
    bias = payload["strong_head.bias"]
    if weight.shape[0] != len(as_strong_train_classes):
        raise ValueError("pretrained class vocabulary/head size mismatch")
    copied: list[dict[str, Any]] = []
    with torch.no_grad():
        for target_index, label in enumerate(labels):
            entry = mapping.get(label)
            if not isinstance(entry, Mapping):
                raise ValueError(f"missing head mapping for {label}")
            display = str(entry["canonical_display_name"])
            source_index = int(entry["pretrained_head_index"])
            if as_strong_train_classes[source_index] != display:
                raise ValueError(f"stale head mapping for {label}")
            model.strong_head.weight[target_index].copy_(weight[source_index])
            model.strong_head.bias[target_index].copy_(bias[source_index])
            copied.append(
                {"target_index": target_index, "label": label, "source_index": source_index, "display": display}
            )
    return {
        "checkpoint": str(checkpoint_path),
        "source_classes": int(weight.shape[0]),
        "target_classes": len(labels),
        "exact_rows_copied": len(copied),
        "all_exact": len(copied) == len(labels),
        "mapping": copied,
    }


def scene_sampling_weights(rows: Sequence[SceneItem], labels: Sequence[str]) -> torch.Tensor:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update({str(event["label"]) for event in row.events})
    maximum = max(counts.values())
    class_weight = {label: math.sqrt(maximum / max(1, counts[label])) for label in labels}
    weights = []
    for row in rows:
        present = {str(event["label"]) for event in row.events}
        weights.append(float(np.mean([class_weight[label] for label in present])) if present else 1.0)
    result = torch.tensor(weights, dtype=torch.double)
    return result / result.mean().clamp_min(1e-12)


def clip_pos_weight(rows: Sequence[SceneItem], labels: Sequence[str], maximum: float) -> torch.Tensor:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update({str(event["label"]) for event in row.events})
    positives = torch.tensor([float(counts[label]) for label in labels], dtype=torch.float32)
    negatives = float(len(rows)) - positives
    return (negatives / positives.clamp_min(1.0)).clamp(1.0, float(maximum))


def weighted_frame_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    sample_weights: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    error = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight.to(logits), reduction="none"
    )
    mask = valid.to(error.dtype).unsqueeze(-1)
    weights = sample_weights.to(error).view(-1, 1, 1)
    denominator = (mask * weights).sum().clamp_min(1.0) * logits.shape[-1]
    return (error * mask * weights).sum() / denominator


def weighted_clip_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    sample_weights: torch.Tensor,
    pos_weight: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    tau = float(temperature)
    if tau <= 0:
        raise ValueError("clip temperature must be positive")
    valid3 = valid.unsqueeze(-1)
    masked = (logits / tau).masked_fill(~valid3, -torch.inf)
    counts = valid.sum(dim=1).clamp_min(1).to(logits.dtype)
    scene_logits = tau * (torch.logsumexp(masked, dim=1) - counts.log().unsqueeze(-1))
    scene_targets = targets.amax(dim=1)
    error = F.binary_cross_entropy_with_logits(
        scene_logits, scene_targets, pos_weight=pos_weight.to(logits), reduction="none"
    )
    weights = sample_weights.to(error).unsqueeze(-1)
    return (error * weights).sum() / (weights.sum().clamp_min(1.0) * logits.shape[-1])


def make_train_loader(
    features: torch.Tensor,
    all_rows: Sequence[SceneItem],
    indices: Sequence[int],
    *,
    labels: Sequence[str],
    batch_size: int,
    silver_weight: float,
    seed: int,
) -> tuple[DataLoader, list[SceneItem]]:
    stage_rows = [all_rows[index] for index in indices]
    weights = scene_sampling_weights(stage_rows, labels)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(stage_rows),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    dataset = IndexedFeatureDataset(
        features, all_rows, indices, silver_weight=silver_weight
    )
    return (
        DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, collate_fn=feature_collate),
        stage_rows,
    )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    labels: Sequence[str],
    thresholds: Sequence[float],
    device: torch.device,
    event_iou_threshold: float,
    min_duration: float,
    merge_gap: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    predictions = collect_predictions(model, loader, device=device)
    summaries = [
        summarize_predictions(
            predictions,
            labels,
            threshold=threshold,
            event_iou_threshold=event_iou_threshold,
            min_duration=min_duration,
            merge_gap=merge_gap,
        )
        for threshold in thresholds
    ]
    selected = {
        objective: max(summaries, key=lambda row, name=objective: threshold_key(row, name))
        for objective in ("event", "frame", "scene")
    }
    return predictions, summaries, selected


def calibrate_class_thresholds(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    thresholds: Sequence[float],
    event_iou_threshold: float,
    min_duration: float,
    merge_gap: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Choose one event threshold per class on dev, with precision tie-breaks."""

    curves: dict[str, list[dict[str, float]]] = {label: [] for label in labels}
    for threshold in thresholds:
        tp: Counter[str] = Counter()
        fp: Counter[str] = Counter()
        fn: Counter[str] = Counter()
        for item in predictions:
            probabilities = torch.as_tensor(item["logits"]).float().sigmoid()
            decoded = decode_fixed_grid_events(
                probabilities,
                labels=labels,
                valid_frames=int(item["valid_frames"]),
                threshold=float(threshold),
                min_duration_seconds=min_duration,
                merge_gap_seconds=merge_gap,
            )
            item_tp, item_fp, item_fn = _match_events(
                decoded,
                list(item.get("gold_events") or ()),
                iou_threshold=event_iou_threshold,
            )
            tp.update(item_tp)
            fp.update(item_fp)
            fn.update(item_fn)
        for label in labels:
            precision = tp[label] / max(tp[label] + fp[label], 1)
            recall = tp[label] / max(tp[label] + fn[label], 1)
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            curves[label].append(
                {
                    "threshold": float(threshold),
                    "f1": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "tp": float(tp[label]),
                    "fp": float(fp[label]),
                    "fn": float(fn[label]),
                }
            )
    selected: dict[str, float] = {}
    selected_rows: dict[str, Any] = {}
    for label in labels:
        best = max(
            curves[label],
            key=lambda row: (
                row["f1"],
                row["precision"],
                row["recall"],
                -row["fp"],
                row["threshold"],
            ),
        )
        selected[label] = float(best["threshold"])
        selected_rows[label] = best
    return selected, {"selected": selected_rows, "curves": curves}


def summarize_class_thresholds(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    class_thresholds: Mapping[str, float],
    *,
    event_iou_threshold: float,
    min_duration: float,
    merge_gap: float,
) -> dict[str, Any]:
    """Evaluate frame, scene, and event metrics with calibrated thresholds."""

    frame_tp: Counter[str] = Counter()
    frame_fp: Counter[str] = Counter()
    frame_fn: Counter[str] = Counter()
    scene_tp: Counter[str] = Counter()
    scene_fp: Counter[str] = Counter()
    scene_fn: Counter[str] = Counter()
    event_tp: Counter[str] = Counter()
    event_fp: Counter[str] = Counter()
    event_fn: Counter[str] = Counter()
    false_labels: list[int] = []
    unique_thresholds = sorted({float(value) for value in class_thresholds.values()})

    for item in predictions:
        probabilities = torch.as_tensor(item["logits"]).float().sigmoid()
        valid_frames = int(item["valid_frames"])
        gold_events = list(item.get("gold_events") or ())
        target, valid = build_fixed_grid_targets(
            [{"duration_seconds": item["duration_seconds"], "events": gold_events}],
            num_labels=len(labels),
        )
        target = target[0].bool()
        valid = valid[0]
        for label_id, label in enumerate(labels):
            predicted_frame = probabilities[:, label_id] >= float(class_thresholds[label])
            gold_frame = target[:, label_id] & valid
            predicted_frame = predicted_frame & valid
            frame_tp[label] += int((predicted_frame & gold_frame).sum())
            frame_fp[label] += int((predicted_frame & ~gold_frame & valid).sum())
            frame_fn[label] += int((~predicted_frame & gold_frame).sum())

        decoded_by_threshold = {
            threshold: decode_fixed_grid_events(
                probabilities,
                labels=labels,
                valid_frames=valid_frames,
                threshold=threshold,
                min_duration_seconds=min_duration,
                merge_gap_seconds=merge_gap,
            )
            for threshold in unique_thresholds
        }
        decoded = [
            event
            for threshold, events in decoded_by_threshold.items()
            for event in events
            if float(class_thresholds[str(event["label"])]) == threshold
        ]
        predicted_labels = {str(event["label"]) for event in decoded}
        gold_labels = {str(event["label"]) for event in gold_events}
        scene_tp.update(predicted_labels & gold_labels)
        scene_fp.update(predicted_labels - gold_labels)
        scene_fn.update(gold_labels - predicted_labels)
        false_labels.append(len(predicted_labels - gold_labels))
        item_tp, item_fp, item_fn = _match_events(
            decoded, gold_events, iou_threshold=event_iou_threshold
        )
        event_tp.update(item_tp)
        event_fp.update(item_fp)
        event_fn.update(item_fn)

    scene = aggregate_class_counts(scene_tp, scene_fp, scene_fn, labels)
    scene["avg_false_positive_labels_↓"] = float(np.mean(false_labels)) if false_labels else 0.0
    return {
        "threshold_policy": "per_class_dev_event_f1",
        "scene_label": scene,
        "frame": aggregate_class_counts(frame_tp, frame_fp, frame_fn, labels),
        "event_iou": {
            **aggregate_class_counts(event_tp, event_fp, event_fn, labels),
            "iou_threshold": float(event_iou_threshold),
        },
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_ontology(args.ontology.resolve())
    label_to_id = {label: index for index, label in enumerate(labels)}
    single_rows = load_scene_manifest(args.single_train.resolve(), label_to_id, args.max_single_train)
    multi_rows = load_scene_manifest(args.multi_train.resolve(), label_to_id, args.max_multi_train)
    train_rows = single_rows + multi_rows
    dev_rows = load_scene_manifest(args.dev_manifest.resolve(), label_to_id, args.max_dev)
    test_rows = load_scene_manifest(args.test_manifest.resolve(), label_to_id, args.max_test)
    if not train_rows or not dev_rows or not test_rows:
        raise RuntimeError("empty train/dev/test curriculum")
    gold_indices = [index for index, row in enumerate(single_rows) if is_gold_single(row)]
    all_indices = list(range(len(train_rows)))
    if not gold_indices:
        raise RuntimeError("Gold-first stage has no rows")

    device = make_device(args.device)
    model = load_model(len(labels), args.checkpoint, device, unfreeze_last_blocks=0)
    transfer_audit = transfer_pretrained_head(
        model,
        labels=labels,
        label_map_path=args.label_map.resolve(),
        checkpoint=args.checkpoint,
    )
    if not transfer_audit["all_exact"]:
        raise RuntimeError("head transfer was not exact")
    model.model.eval()
    model.model.requires_grad_(False)
    model.strong_head.requires_grad_(True)
    model.weak_head.requires_grad_(False)
    if int(model.seq_len) != NUM_FRAMES:
        raise RuntimeError(f"expected {NUM_FRAMES} BEATs frames, got {model.seq_len}")

    def cache(rows: Sequence[SceneItem], split: str) -> torch.Tensor:
        loader = DataLoader(
            SceneDataset(rows, audio_root=args.audio_root.resolve()),
            batch_size=args.feature_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate,
        )
        return cache_frozen_features(
            model, loader, expected_rows=len(rows), device=device, split=split
        )

    print(
        f"labels={len(labels)} exact_head_rows={transfer_audit['exact_rows_copied']} "
        f"single={len(single_rows)} multi={len(multi_rows)} dev={len(dev_rows)} test={len(test_rows)}",
        flush=True,
    )
    train_features = cache(train_rows, "train_curriculum")
    dev_features = cache(dev_rows, "dev_multievent")
    test_features = cache(test_rows, "test_multievent")
    dev_loader = DataLoader(
        IndexedFeatureDataset(dev_features, dev_rows, silver_weight=args.silver_weight),
        batch_size=max(args.batch_size, 32),
        shuffle=False,
        num_workers=0,
        collate_fn=eval_feature_collate,
    )
    test_loader = DataLoader(
        IndexedFeatureDataset(test_features, test_rows, silver_weight=args.silver_weight),
        batch_size=max(args.batch_size, 32),
        shuffle=False,
        num_workers=0,
        collate_fn=eval_feature_collate,
    )

    best_scores = {objective: (-math.inf,) for objective in ("event", "frame", "scene")}
    best_heads: dict[str, dict[str, torch.Tensor]] = {}
    best_meta: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []

    def evaluate_and_record(stage: str, epoch: int, loss: float | None) -> float:
        _, summaries, selected = evaluate(
            model,
            dev_loader,
            labels=labels,
            thresholds=args.thresholds,
            device=device,
            event_iou_threshold=args.event_iou_threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
        )
        for objective, summary in selected.items():
            score = threshold_key(summary, objective)
            if score > best_scores[objective]:
                best_scores[objective] = score
                best_heads[objective] = {
                    key: value.detach().cpu().clone()
                    for key, value in model.strong_head.state_dict().items()
                }
                best_meta[objective] = {
                    "stage": stage,
                    "epoch": epoch,
                    "metrics": compact_summary(summary),
                }
        row = {
            "stage": stage,
            "epoch": epoch,
            "train_loss": loss,
            "selected": {name: compact_summary(value) for name, value in selected.items()},
        }
        history.append(row)
        atomic_json(
            output_dir / "progress.json",
            {"format": FORMAT, "status": "running", "history": history, "best": best_meta},
        )
        atomic_torch(
            output_dir / "best_heads_running.pt",
            {"format": FORMAT, "labels": labels, "strong_head_states": best_heads, "best": best_meta},
        )
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        return float(selected["event"]["event_iou"]["f1_↑"])

    evaluate_and_record("exact_pretrained_transfer", 0, None)

    stages = [
        ("gold_single", args.gold_epochs, gold_indices, args.gold_lr),
        ("mixed_single_multievent", args.mixed_epochs, all_indices, args.mixed_lr),
    ]
    for stage_index, (stage, epochs, indices, learning_rate) in enumerate(stages):
        if epochs <= 0:
            continue
        train_loader, stage_rows = make_train_loader(
            train_features,
            train_rows,
            indices,
            labels=labels,
            batch_size=args.batch_size,
            silver_weight=args.silver_weight,
            seed=args.seed + stage_index,
        )
        pos_weight = fixed_grid_pos_weight(
            stage_rows, len(labels), max_pos_weight=args.max_pos_weight
        ).to(device)
        scene_pos_weight = clip_pos_weight(
            stage_rows, labels, args.max_clip_pos_weight
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.strong_head.parameters(), lr=learning_rate, weight_decay=args.weight_decay
        )
        stage_best_event_f1 = -math.inf
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            model.strong_head.train()
            losses: list[float] = []
            for features, rows, quality_weights in train_loader:
                features = features.to(device, non_blocking=True)
                logits = head_logits(model, features)
                targets, valid = build_fixed_grid_targets(
                    rows, num_labels=len(labels), device=device
                )
                frame_loss = weighted_frame_loss(
                    logits,
                    targets,
                    valid,
                    quality_weights,
                    pos_weight,
                )
                presence_loss = weighted_clip_loss(
                    logits,
                    targets,
                    valid,
                    quality_weights,
                    scene_pos_weight,
                    args.clip_temperature,
                )
                loss = frame_loss + args.clip_loss_weight * presence_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.strong_head.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            event_f1 = evaluate_and_record(stage, epoch, float(np.mean(losses)))
            if event_f1 > stage_best_event_f1 + args.early_stopping_min_delta:
                stage_best_event_f1 = event_f1
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                args.early_stopping_patience > 0
                and stale_epochs >= args.early_stopping_patience
            ):
                print(
                    json.dumps(
                        {
                            "early_stopping": True,
                            "stage": stage,
                            "epoch": epoch,
                            "patience": args.early_stopping_patience,
                            "min_delta": args.early_stopping_min_delta,
                            "stage_best_event_f1": stage_best_event_f1,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                break

    if "event" not in best_heads:
        raise RuntimeError("no event checkpoint selected")
    model.strong_head.load_state_dict(best_heads["event"], strict=True)
    dev_predictions, dev_summaries, dev_selected = evaluate(
        model,
        dev_loader,
        labels=labels,
        thresholds=args.thresholds,
        device=device,
        event_iou_threshold=args.event_iou_threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
    )
    selected_threshold = float(dev_selected["event"]["threshold"])
    test_predictions = collect_predictions(model, test_loader, device=device)
    test_summary = summarize_predictions(
        test_predictions,
        labels,
        threshold=selected_threshold,
        event_iou_threshold=args.event_iou_threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
    )
    class_thresholds, class_calibration = calibrate_class_thresholds(
        dev_predictions,
        labels,
        thresholds=args.thresholds,
        event_iou_threshold=args.event_iou_threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
    )
    dev_class_threshold_summary = summarize_class_thresholds(
        dev_predictions,
        labels,
        class_thresholds,
        event_iou_threshold=args.event_iou_threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
    )
    test_class_threshold_summary = summarize_class_thresholds(
        test_predictions,
        labels,
        class_thresholds,
        event_iou_threshold=args.event_iou_threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
    )
    atomic_json(
        output_dir / "class_thresholds.json",
        {
            "format": FORMAT,
            "policy": "per_class_dev_event_f1_with_precision_tiebreak",
            "thresholds": class_thresholds,
            "calibration": class_calibration,
        },
    )
    diagnostics = duration_and_tier_audit(
        dev_predictions,
        labels,
        threshold=selected_threshold,
        event_iou_threshold=args.event_iou_threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
    )
    report = {
        "format": FORMAT,
        "status": "complete",
        "labels": len(labels),
        "head_transfer": transfer_audit,
        "data": {
            "single_train": len(single_rows),
            "multi_train": len(multi_rows),
            "gold_stage_rows": len(gold_indices),
            "dev_multievent": len(dev_rows),
            "test_multievent": len(test_rows),
        },
        "training": {
            "gold_epochs": args.gold_epochs,
            "mixed_epochs": args.mixed_epochs,
            "gold_lr": args.gold_lr,
            "mixed_lr": args.mixed_lr,
            "silver_weight": args.silver_weight,
            "clip_loss_weight": args.clip_loss_weight,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "backbone_frozen": True,
            "class_balanced_sampler": "inverse_sqrt_scene_support",
            "fixed_grid": f"{NUM_FRAMES}x{FRAME_HOP_SECONDS:.2f}s",
        },
        "history": history,
        "best": best_meta,
        "selected_threshold": selected_threshold,
        "class_thresholds": {
            "policy": "per_class_dev_event_f1_with_precision_tiebreak",
            "path": str(output_dir / "class_thresholds.json"),
            "dev": dev_class_threshold_summary,
            "locked_test": test_class_threshold_summary,
        },
        "dev_event_checkpoint": {
            "selected": dev_selected["event"],
            "threshold_summaries": dev_summaries,
        },
        "locked_test": test_summary,
        "diagnostics": diagnostics,
    }
    atomic_json(output_dir / "training_report.json", report)
    atomic_json(
        output_dir / "progress.json",
        {"format": FORMAT, "status": "complete", "best": best_meta, "locked_test": compact_summary(test_summary)},
    )
    atomic_torch(
        output_dir / "pretrainedsed_beats_qces_detector.pt",
        {
            "format": FORMAT,
            "model_state_dict": model.state_dict(),
            "labels": labels,
            "best_threshold": selected_threshold,
            "class_thresholds": class_thresholds,
            "checkpoint_objective": "event",
            "report": report,
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "best_dev": best_meta["event"],
                "locked_test": compact_summary(test_summary),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    main()
