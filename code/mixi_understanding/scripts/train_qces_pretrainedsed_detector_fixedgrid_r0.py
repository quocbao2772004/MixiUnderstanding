#!/usr/bin/env python3
"""Correctness-only R0 for the c37 PretrainedSED detector.

This sidecar preserves the historical frozen BEATs-Strong + linear-head
experiment while repairing three correctness problems:

1. timestamps are projected onto a fixed 10 s / 250 frame grid;
2. right-padding is excluded from both loss and metrics; and
3. scene, frame, and event checkpoints are tracked independently.

No class balancing, sampler change, auxiliary loss, augmentation, recurrent
decoder, or backbone fine-tuning is introduced in R0.  Frozen BEATs features
are cached in RAM once; this is mathematically equivalent to recomputing the
same eval-mode frozen backbone at every epoch and substantially shortens the
run.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
for value in (CODE_ROOT,):
    if str(value) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(value))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.fixed_grid_detector import (
    FIXED_AUDIO_SECONDS,
    FRAME_HOP_SECONDS,
    NUM_FRAMES,
    _match_events,
    build_fixed_grid_targets,
    decode_fixed_grid_events,
    masked_balanced_bce,
    valid_frames_for_duration,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    SceneDataset,
    SceneItem,
    collate,
    interpolate_sequence,
    load_model,
    load_ontology,
    load_scene_manifest,
    make_device,
    set_seed,
)


FORMAT = "qces_pretrainedsed_fixedgrid_r0_v1"
DEFAULT_DATA = PROJECT_ROOT / "outputs/qces_full200_single_event_pretrain_c37_v1"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full200/c37_single_event_fixedgrid_r0_e8_v1"
)


class CachedFeatureDataset(Dataset[tuple[torch.Tensor, SceneItem]]):
    def __init__(self, features: torch.Tensor, rows: Sequence[SceneItem]) -> None:
        if features.ndim != 3 or features.shape[0] != len(rows):
            raise ValueError("cached feature/row shape mismatch")
        self.features = features
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, SceneItem]:
        return self.features[index], self.rows[index]


def feature_collate(
    batch: Sequence[tuple[torch.Tensor, SceneItem]],
) -> tuple[torch.Tensor, list[SceneItem]]:
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-manifest", type=Path, default=DEFAULT_DATA / "detector_scene_manifest_train.jsonl"
    )
    parser.add_argument(
        "--val-manifest", type=Path, default=DEFAULT_DATA / "detector_scene_manifest_dev.jsonl"
    )
    parser.add_argument("--ontology", type=Path, default=DEFAULT_DATA / "ontology_200.txt")
    parser.add_argument("--audio-root", type=Path, default=Path("/"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="BEATs_strong_1")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-val-scenes", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
    )
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument("--min-duration", type=float, default=0.08)
    parser.add_argument("--merge-gap", type=float, default=0.12)
    parser.add_argument("--max-pos-weight", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=2037)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def fixed_grid_pos_weight(
    rows: Sequence[SceneItem], num_labels: int, *, max_pos_weight: float
) -> torch.Tensor:
    positive = torch.zeros(num_labels, dtype=torch.float64)
    total_valid = 0
    for row in rows:
        targets, valid = build_fixed_grid_targets([row], num_labels=num_labels)
        positive += targets[0, valid[0]].sum(dim=0).double()
        total_valid += int(valid.sum())
    negative = float(total_valid) - positive
    return (negative / positive.clamp_min(1.0)).clamp(1.0, max_pos_weight).float()


@torch.inference_mode()
def cache_frozen_features(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    expected_rows: int,
    device: torch.device,
    split: str,
) -> torch.Tensor:
    model.eval()
    cached: torch.Tensor | None = None
    cursor = 0
    for step, (waveforms, _) in enumerate(loader, start=1):
        waveforms = waveforms.to(device, non_blocking=True)
        mel = model.mel_forward(waveforms)
        features = model.model(mel)
        features = interpolate_sequence(features, model.seq_len)
        features = model.seq_model(features)
        # Keep float32 so feature caching changes runtime only, not the values
        # seen by the historical float32 linear head.
        features = features.detach().cpu().to(torch.float32)
        if cached is None:
            cached = torch.empty(
                (expected_rows, features.shape[1], features.shape[2]), dtype=torch.float32
            )
        end = cursor + features.shape[0]
        cached[cursor:end].copy_(features)
        cursor = end
        if step % 100 == 0 or cursor == expected_rows:
            print(
                f"feature_cache split={split} rows={cursor}/{expected_rows} "
                f"percent={100.0 * cursor / expected_rows:.1f}",
                flush=True,
            )
    if cached is None or cursor != expected_rows:
        raise RuntimeError(f"incomplete {split} feature cache: {cursor}/{expected_rows}")
    return cached


def head_logits(model: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    return model.strong_head(features.to(dtype=model.strong_head.weight.dtype))


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.strong_head.eval()
    output: list[dict[str, Any]] = []
    for features, rows in loader:
        logits = head_logits(model, features.to(device, non_blocking=True)).cpu().to(torch.float16)
        for index, row in enumerate(rows):
            output.append(
                {
                    "scene_id": row.scene_id,
                    "duration_seconds": float(row.duration_seconds),
                    "valid_frames": valid_frames_for_duration(row.duration_seconds),
                    "gold_events": list(row.events),
                    "logits": logits[index],
                }
            )
    return output


def safe_prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {"precision_↑": precision, "recall_↑": recall, "f1_↑": f1}


def aggregate_class_counts(
    tp: Counter[str], fp: Counter[str], fn: Counter[str], labels: Sequence[str]
) -> dict[str, Any]:
    micro = safe_prf(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    per_class = {label: safe_prf(tp[label], fp[label], fn[label]) for label in labels}
    observed = [per_class[label]["f1_↑"] for label in labels if tp[label] + fn[label] > 0]
    return {
        **micro,
        "macro_f1_observed_classes_↑": float(np.mean(observed)) if observed else 0.0,
        "per_class": per_class,
    }


def decode_for_item(
    probs: torch.Tensor,
    item: Mapping[str, Any],
    labels: Sequence[str],
    *,
    threshold: float,
    min_duration: float,
    merge_gap: float,
) -> list[dict[str, Any]]:
    events = decode_fixed_grid_events(
        probs,
        labels=labels,
        valid_frames=int(item["valid_frames"]),
        threshold=threshold,
        min_duration_seconds=min_duration,
        merge_gap_seconds=merge_gap,
    )
    duration = float(item["duration_seconds"])
    clipped: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        row["offset_seconds"] = min(duration, float(row["offset_seconds"]))
        if float(row["offset_seconds"]) - float(row["onset_seconds"]) >= min_duration - 1e-9:
            clipped.append(row)
    return clipped


def summarize_predictions(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    threshold: float,
    event_iou_threshold: float,
    min_duration: float,
    merge_gap: float,
) -> dict[str, Any]:
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
    kept_labels: list[int] = []

    for item in predictions:
        logits = torch.as_tensor(item["logits"]).float()
        probs = logits.sigmoid()
        valid_frames = int(item["valid_frames"])
        gold_events = list(item.get("gold_events") or ())
        target, valid = build_fixed_grid_targets(
            [{"duration_seconds": item["duration_seconds"], "events": gold_events}],
            num_labels=len(labels),
        )
        target = target[0].bool()
        valid = valid[0]
        pred_frame = probs >= threshold
        for label_id, label in enumerate(labels):
            pred = pred_frame[:, label_id] & valid
            gold = target[:, label_id] & valid
            frame_tp[label] += int((pred & gold).sum())
            frame_fp[label] += int((pred & ~gold & valid).sum())
            frame_fn[label] += int((~pred & gold).sum())

        pred_labels = {
            labels[label_id]
            for label_id in torch.where(probs[:valid_frames].amax(dim=0) >= threshold)[0].tolist()
        }
        gold_labels = {str(event["label"]) for event in gold_events}
        for label in pred_labels & gold_labels:
            scene_tp[label] += 1
        for label in pred_labels - gold_labels:
            scene_fp[label] += 1
        for label in gold_labels - pred_labels:
            scene_fn[label] += 1
        false_labels.append(len(pred_labels - gold_labels))
        kept_labels.append(len(pred_labels))

        decoded = decode_for_item(
            probs,
            item,
            labels,
            threshold=threshold,
            min_duration=min_duration,
            merge_gap=merge_gap,
        )
        tp, fp, fn = _match_events(decoded, gold_events, iou_threshold=event_iou_threshold)
        event_tp.update(tp)
        event_fp.update(fp)
        event_fn.update(fn)

    scene = aggregate_class_counts(scene_tp, scene_fp, scene_fn, labels)
    scene["avg_labels_kept_↓"] = float(np.mean(kept_labels)) if kept_labels else 0.0
    scene["avg_false_positive_labels_↓"] = (
        float(np.mean(false_labels)) if false_labels else 0.0
    )
    return {
        "threshold": float(threshold),
        "scene_label": scene,
        "frame": aggregate_class_counts(frame_tp, frame_fp, frame_fn, labels),
        "event_iou": {
            **aggregate_class_counts(event_tp, event_fp, event_fn, labels),
            "iou_threshold": float(event_iou_threshold),
        },
    }


def threshold_key(summary: Mapping[str, Any], objective: str) -> tuple[float, ...]:
    scene = summary["scene_label"]
    frame = summary["frame"]
    event = summary["event_iou"]
    if objective == "event":
        return (
            float(event["f1_↑"]),
            float(event["macro_f1_observed_classes_↑"]),
            float(frame["f1_↑"]),
            float(scene["f1_↑"]),
            -float(scene["avg_false_positive_labels_↓"]),
        )
    if objective == "frame":
        return (
            float(frame["f1_↑"]),
            float(event["f1_↑"]),
            float(scene["f1_↑"]),
            -float(scene["avg_false_positive_labels_↓"]),
        )
    if objective == "scene":
        return (
            float(scene["f1_↑"]),
            float(event["f1_↑"]),
            float(frame["f1_↑"]),
            -float(scene["avg_false_positive_labels_↓"]),
        )
    raise ValueError(f"unknown objective: {objective}")


def compact_summary(summary: Mapping[str, Any]) -> dict[str, float]:
    return {
        "threshold": float(summary["threshold"]),
        "scene_f1_↑": float(summary["scene_label"]["f1_↑"]),
        "scene_precision_↑": float(summary["scene_label"]["precision_↑"]),
        "scene_recall_↑": float(summary["scene_label"]["recall_↑"]),
        "event_f1_↑": float(summary["event_iou"]["f1_↑"]),
        "event_macro_f1_↑": float(summary["event_iou"]["macro_f1_observed_classes_↑"]),
        "frame_f1_↑": float(summary["frame"]["f1_↑"]),
        "avg_false_positive_labels_↓": float(
            summary["scene_label"]["avg_false_positive_labels_↓"]
        ),
    }


def duration_and_tier_audit(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    threshold: float,
    event_iou_threshold: float,
    min_duration: float,
    merge_gap: float,
) -> dict[str, Any]:
    duration_bins = [
        ("<0.5s", 0.0, 0.5),
        ("0.5-1s", 0.5, 1.0),
        ("1-2s", 1.0, 2.0),
        ("2-5s", 2.0, 5.0),
        ("5-10s", 5.0, 10.0),
        ("10s", 10.0, math.inf),
    ]
    output: dict[str, Any] = {"duration": {}, "cleanliness_tier": {}}
    for name, lower, upper in duration_bins:
        subset = [
            item
            for item in predictions
            if lower <= float(item["duration_seconds"]) < upper
        ]
        if subset:
            output["duration"][name] = {
                "items": len(subset),
                **compact_summary(
                    summarize_predictions(
                        subset,
                        labels,
                        threshold=threshold,
                        event_iou_threshold=event_iou_threshold,
                        min_duration=min_duration,
                        merge_gap=merge_gap,
                    )
                ),
            }
    for tier in ("gold", "silver"):
        subset = [
            item
            for item in predictions
            if any(
                str(event.get("cleanliness_tier", "")).lower() == tier
                for event in item.get("gold_events") or ()
            )
        ]
        if subset:
            output["cleanliness_tier"][tier] = {
                "items": len(subset),
                **compact_summary(
                    summarize_predictions(
                        subset,
                        labels,
                        threshold=threshold,
                        event_iou_threshold=event_iou_threshold,
                        min_duration=min_duration,
                        merge_gap=merge_gap,
                    )
                ),
            }
    return output


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.feature_batch_size <= 0:
        raise ValueError("epochs and batch sizes must be positive")
    if not args.thresholds or any(not 0.0 <= value <= 1.0 for value in args.thresholds):
        raise ValueError("thresholds must be a non-empty list in [0,1]")
    if len(set(args.thresholds)) != len(args.thresholds):
        raise ValueError("thresholds must be unique")
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = make_device(args.device)
    labels = load_ontology(args.ontology.resolve())
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_rows = load_scene_manifest(
        args.train_manifest.resolve(), label_to_id, args.max_train_scenes
    )
    val_rows = load_scene_manifest(args.val_manifest.resolve(), label_to_id, args.max_val_scenes)
    if not train_rows or not val_rows:
        raise RuntimeError("empty train or validation manifest")
    print(
        f"device={device} train_scenes={len(train_rows)} val_scenes={len(val_rows)} "
        f"labels={len(labels)} grid={NUM_FRAMES}x{FRAME_HOP_SECONDS:.2f}s",
        flush=True,
    )

    model = load_model(len(labels), args.checkpoint, device, unfreeze_last_blocks=0)
    model.model.eval()
    model.model.requires_grad_(False)
    model.strong_head.requires_grad_(True)
    model.weak_head.requires_grad_(False)
    if int(model.seq_len) != NUM_FRAMES:
        raise RuntimeError(f"expected seq_len={NUM_FRAMES}, got {model.seq_len}")

    train_audio_loader = DataLoader(
        SceneDataset(train_rows, audio_root=args.audio_root.resolve()),
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    val_audio_loader = DataLoader(
        SceneDataset(val_rows, audio_root=args.audio_root.resolve()),
        batch_size=args.feature_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    train_features = cache_frozen_features(
        model, train_audio_loader, expected_rows=len(train_rows), device=device, split="train"
    )
    val_features = cache_frozen_features(
        model, val_audio_loader, expected_rows=len(val_rows), device=device, split="val"
    )
    del train_audio_loader, val_audio_loader

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        CachedFeatureDataset(train_features, train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=False,
        collate_fn=feature_collate,
    )
    val_loader = DataLoader(
        CachedFeatureDataset(val_features, val_rows),
        batch_size=max(args.batch_size, 32),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=feature_collate,
    )

    pos_weight = fixed_grid_pos_weight(
        train_rows, len(labels), max_pos_weight=args.max_pos_weight
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.strong_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_scores = {objective: (-math.inf,) for objective in ("event", "frame", "scene")}
    best_heads: dict[str, dict[str, torch.Tensor]] = {}
    best_meta: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.strong_head.train()
        losses: list[float] = []
        for step, (features, rows) in enumerate(train_loader, start=1):
            features = features.to(device, non_blocking=True)
            logits = head_logits(model, features)
            targets, valid = build_fixed_grid_targets(
                rows, num_labels=len(labels), device=device
            )
            loss = masked_balanced_bce(
                logits, targets, valid, pos_weight=pos_weight, class_weight=None
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.strong_head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step % 250 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={np.mean(losses[-250:]):.4f}",
                    flush=True,
                )

        predictions = collect_predictions(model, val_loader, device=device)
        summaries = [
            summarize_predictions(
                predictions,
                labels,
                threshold=threshold,
                event_iou_threshold=args.event_iou_threshold,
                min_duration=args.min_duration,
                merge_gap=args.merge_gap,
            )
            for threshold in args.thresholds
        ]
        selected = {
            objective: max(summaries, key=lambda row, name=objective: threshold_key(row, name))
            for objective in ("event", "frame", "scene")
        }
        for objective, summary in selected.items():
            score = threshold_key(summary, objective)
            if score > best_scores[objective]:
                best_scores[objective] = score
                best_heads[objective] = {
                    key: value.detach().cpu().clone()
                    for key, value in model.strong_head.state_dict().items()
                }
                best_meta[objective] = {
                    "epoch": epoch,
                    "metrics": compact_summary(summary),
                }
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "selected": {name: compact_summary(row) for name, row in selected.items()},
        }
        history.append(epoch_row)
        atomic_torch(
            output_dir / "best_heads_running.pt",
            {
                "format": FORMAT,
                "labels": labels,
                "strong_head_states": best_heads,
                "best": best_meta,
            },
        )
        atomic_json(
            output_dir / "progress.json",
            {"format": FORMAT, "status": "running", "history": history, "best": best_meta},
        )
        print(json.dumps(epoch_row, ensure_ascii=False, sort_keys=True), flush=True)

    if "event" not in best_heads:
        raise RuntimeError("no event checkpoint was selected")
    model.strong_head.load_state_dict(best_heads["event"], strict=True)
    final_predictions = collect_predictions(model, val_loader, device=device)
    final_summaries = [
        summarize_predictions(
            final_predictions,
            labels,
            threshold=threshold,
            event_iou_threshold=args.event_iou_threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
        )
        for threshold in args.thresholds
    ]
    final_event = max(final_summaries, key=lambda row: threshold_key(row, "event"))
    best_threshold = float(final_event["threshold"])
    diagnostics = duration_and_tier_audit(
        final_predictions,
        labels,
        threshold=best_threshold,
        event_iou_threshold=args.event_iou_threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
    )

    report = {
        "format": FORMAT,
        "status": "complete",
        "change_scope": [
            "fixed_10s_250frame_timestamp_projection",
            "right_padding_masked_from_loss_and_metrics",
            "separate_scene_frame_event_checkpoint_selection",
            "frozen_feature_cache_for_runtime_only",
        ],
        "explicitly_not_used": [
            "class_balanced_sampler",
            "class_weight",
            "auxiliary_clip_loss",
            "onset_offset_loss",
            "augmentation",
            "backbone_finetuning",
            "pair_ranker",
        ],
        "grid": {
            "fixed_audio_seconds": FIXED_AUDIO_SECONDS,
            "frames": NUM_FRAMES,
            "frame_hop_seconds": FRAME_HOP_SECONDS,
            "padding_excluded_from_loss": True,
            "padding_excluded_from_metrics": True,
        },
        "inputs": {
            "train_manifest": str(args.train_manifest.resolve()),
            "val_manifest": str(args.val_manifest.resolve()),
            "ontology": str(args.ontology.resolve()),
            "checkpoint": args.checkpoint,
        },
        "training": {
            "seed": args.seed,
            "train_scenes": len(train_rows),
            "val_scenes": len(val_rows),
            "labels": len(labels),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "backbone_frozen": True,
        },
        "history": history,
        "best": best_meta,
        "main_checkpoint_objective": "event_f1_at_iou_0.30",
        "best_threshold": best_threshold,
        "val_event_checkpoint": {
            "selected": final_event,
            "threshold_summaries": final_summaries,
        },
        "diagnostics": diagnostics,
    }
    atomic_json(output_dir / "training_report.json", report)
    atomic_json(output_dir / "progress.json", {"format": FORMAT, "status": "complete", "best": best_meta})
    atomic_torch(
        output_dir / "best_heads.pt",
        {
            "format": FORMAT,
            "labels": labels,
            "strong_head_states": best_heads,
            "best": best_meta,
        },
    )
    atomic_torch(
        output_dir / "pretrainedsed_beats_qces_detector.pt",
        {
            "format": FORMAT,
            "model_state_dict": model.state_dict(),
            "labels": labels,
            "best_threshold": best_threshold,
            "checkpoint_objective": "event",
            "grid": report["grid"],
            "report": report,
        },
    )

    with (output_dir / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for item in final_predictions:
            events = decode_for_item(
                torch.as_tensor(item["logits"]).float().sigmoid(),
                item,
                labels,
                threshold=best_threshold,
                min_duration=args.min_duration,
                merge_gap=args.merge_gap,
            )
            handle.write(
                json.dumps(
                    {
                        "scene_id": item["scene_id"],
                        "duration_seconds": item["duration_seconds"],
                        "threshold": best_threshold,
                        "predicted_events": events,
                        "gold_events": item["gold_events"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    selected = compact_summary(final_event)
    markdown = [
        "# QCES fixed-grid R0",
        "",
        "Correctness-only rerun of the c37 frozen BEATs-Strong linear head.",
        "",
        "| metric | result |",
        "|---|---:|",
        f"| Scene-label F1 ↑ | {selected['scene_f1_↑']:.4f} |",
        f"| Scene-label precision ↑ | {selected['scene_precision_↑']:.4f} |",
        f"| Scene-label recall ↑ | {selected['scene_recall_↑']:.4f} |",
        f"| Event F1 @ IoU 0.30 ↑ | {selected['event_f1_↑']:.4f} |",
        f"| Macro event F1 ↑ | {selected['event_macro_f1_↑']:.4f} |",
        f"| Frame F1 ↑ | {selected['frame_f1_↑']:.4f} |",
        f"| False-positive labels/clip ↓ | {selected['avg_false_positive_labels_↓']:.4f} |",
        "",
        "## Checkpoint selection",
        "",
        "| objective | epoch | threshold | primary metric |",
        "|---|---:|---:|---:|",
    ]
    for objective in ("event", "frame", "scene"):
        metric_name = f"{objective}_f1_↑"
        markdown.append(
            f"| {objective} | {best_meta[objective]['epoch']} | "
            f"{best_meta[objective]['metrics']['threshold']:.2f} | "
            f"{best_meta[objective]['metrics'][metric_name]:.4f} |"
        )
    (output_dir / "training_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "best": best_meta, "main": selected}, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()
