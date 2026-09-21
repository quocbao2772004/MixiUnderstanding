#!/usr/bin/env python3
"""Train a frozen BEATs-Strong frame head for QCES speech-event v1.

The expensive backbone is evaluated once per scene and cached.  Only its
strong frame head is trained for ten non-speech classes plus ``Speech``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRETRAINED_SED_ROOT = PROJECT_ROOT / "code/baseline/PretrainedSED"
if str(PRETRAINED_SED_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAINED_SED_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF

import models.prediction_wrapper as pretrained_prediction_wrapper
from models.beats.BEATs_wrapper import BEATsWrapper
from models.prediction_wrapper import PredictionsWrapper


pretrained_prediction_wrapper.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
LABELS = (
    "Speech",
    "Bark",
    "Meow",
    "Reversing_beeps",
    "Crumpling_and_crinkling",
    "Single-lens_reflex_camera",
    "Chirp_and_tweet",
    "Engine_starting",
    "Toilet_flush",
    "Finger_snapping",
    "Glass_shatter",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _load_model(device: torch.device) -> PredictionsWrapper:
    model = PredictionsWrapper(
        BEATsWrapper(),
        checkpoint="BEATs_strong_1",
        n_classes_strong=len(LABELS),
        n_classes_weak=len(LABELS),
        seq_model_type=None,
        head_type="linear",
    )
    model.to(device)
    model.model.eval().requires_grad_(False)
    model.seq_model.eval().requires_grad_(False)
    model.weak_head.requires_grad_(False)
    return model


def _interpolate(features: torch.Tensor, length: int) -> torch.Tensor:
    if features.size(-2) > length:
        return torch.nn.functional.adaptive_avg_pool1d(
            features.transpose(1, 2), length
        ).transpose(1, 2)
    if features.size(-2) < length:
        return torch.nn.functional.interpolate(
            features.transpose(1, 2), size=length, mode="linear"
        ).transpose(1, 2)
    return features


@torch.no_grad()
def _extract_feature(
    model: PredictionsWrapper, scene: Mapping[str, Any], device: torch.device
) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(_resolve(str(scene["mixture_path"])))
    waveform = waveform.float().mean(dim=0)
    if sample_rate != 16_000:
        waveform = AF.resample(waveform, sample_rate, 16_000)
    mel = model.mel_forward(waveform.unsqueeze(0).to(device))
    feature = model.model(mel)
    feature = _interpolate(feature, int(model.seq_len))
    feature = model.seq_model(feature)
    return feature[0].detach().cpu().half()


def _target(scene: Mapping[str, Any], steps: int) -> torch.Tensor:
    output = torch.zeros((steps, len(LABELS)), dtype=torch.float32)
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    duration = max(float(scene["duration_seconds"]), 1e-6)
    for event in scene.get("events") or []:
        label = str(event.get("label") or "")
        if label not in label_to_id:
            continue
        onset = max(0.0, float(event["onset_seconds"]))
        offset = min(duration, float(event["offset_seconds"]))
        if offset <= onset:
            continue
        start = max(0, min(steps - 1, int(math.floor(onset / duration * steps))))
        end = max(start + 1, min(steps, int(math.ceil(offset / duration * steps))))
        output[start:end, label_to_id[label]] = 1.0
    return output


def _frame_f1(prob: torch.Tensor, target: torch.Tensor, thresholds: torch.Tensor) -> float:
    prediction = prob >= thresholds.reshape(1, 1, -1)
    gold = target.bool()
    tp = int((prediction & gold).sum())
    fp = int((prediction & ~gold).sum())
    fn = int((~prediction & gold).sum())
    return 2.0 * tp / max(2 * tp + fp + fn, 1)


def _class_thresholds(prob: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(
        [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    )
    selected: list[float] = []
    for class_id in range(len(LABELS)):
        gold = target[:, :, class_id].bool()
        best = (-1.0, -1.0, 0.5)
        for threshold in grid:
            pred = prob[:, :, class_id] >= threshold
            tp = int((pred & gold).sum())
            fp = int((pred & ~gold).sum())
            fn = int((~pred & gold).sum())
            f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
            precision = tp / max(tp + fp, 1)
            score = (f1, precision, float(threshold))
            if score > best:
                best = score
        selected.append(best[2])
    return torch.tensor(selected, dtype=torch.float32)


def _decode(
    prob: torch.Tensor,
    scene: Mapping[str, Any],
    thresholds: torch.Tensor,
) -> list[dict[str, Any]]:
    duration = float(scene["duration_seconds"])
    steps = prob.size(0)
    frame_seconds = duration / steps
    events: list[dict[str, Any]] = []
    for class_id, label in enumerate(LABELS):
        flags = (prob[:, class_id] >= thresholds[class_id]).tolist() + [False]
        segments: list[tuple[int, int]] = []
        start: int | None = None
        for index, flag in enumerate(flags):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                segments.append((start, index))
                start = None
        merge_frames = max(1, int(round((0.30 if label == "Speech" else 0.16) / frame_seconds)))
        merged: list[tuple[int, int]] = []
        for start, end in segments:
            if merged and start - merged[-1][1] <= merge_frames:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        minimum = 0.45 if label == "Speech" else 0.08
        for start, end in merged:
            onset, offset = start * frame_seconds, end * frame_seconds
            if offset - onset < minimum:
                continue
            events.append(
                {
                    "label": label,
                    "onset_seconds": round(onset, 4),
                    "offset_seconds": round(offset, 4),
                    "confidence": float(prob[start:end, class_id].max()),
                    "mean_confidence": float(prob[start:end, class_id].mean()),
                }
            )
    events.sort(key=lambda event: (event["onset_seconds"], event["offset_seconds"], event["label"]))
    return events


def _iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    intersection = max(
        0.0,
        min(float(left["offset_seconds"]), float(right["offset_seconds"]))
        - max(float(left["onset_seconds"]), float(right["onset_seconds"])),
    )
    union = max(float(left["offset_seconds"]), float(right["offset_seconds"])) - min(
        float(left["onset_seconds"]), float(right["onset_seconds"])
    )
    return intersection / union if union > 0 else 0.0


def _event_metrics(
    scenes: Sequence[Mapping[str, Any]], probs: torch.Tensor, thresholds: torch.Tensor
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tp = fp = fn = 0
    matched_ious: list[float] = []
    rows: list[dict[str, Any]] = []
    per_label = {label: collections.Counter() for label in LABELS}
    for scene, prob in zip(scenes, probs):
        predicted = _decode(prob, scene, thresholds)
        gold = [event for event in scene["events"] if event["label"] in LABELS]
        used: set[int] = set()
        for candidate in sorted(predicted, key=lambda event: event["confidence"], reverse=True):
            choices = [
                (index, _iou(candidate, event))
                for index, event in enumerate(gold)
                if index not in used and event["label"] == candidate["label"]
            ]
            best_index, best_iou = max(choices, key=lambda value: value[1], default=(-1, 0.0))
            if best_iou >= 0.30:
                used.add(best_index)
                tp += 1
                per_label[candidate["label"]]["tp"] += 1
                matched_ious.append(best_iou)
            else:
                fp += 1
                per_label[candidate["label"]]["fp"] += 1
        for index, event in enumerate(gold):
            if index not in used:
                fn += 1
                per_label[event["label"]]["fn"] += 1
        rows.append(
            {
                "format": "qces_speech_event_detector_prediction_v1",
                "scene_id": scene["scene_id"],
                "split": scene["split"],
                "mixture_path": scene["mixture_path"],
                "duration_seconds": scene["duration_seconds"],
                "predicted_events": predicted,
            }
        )
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {
        "event_precision_↑": precision,
        "event_recall_↑": recall,
        "event_f1_↑": 2 * precision * recall / max(precision + recall, 1e-12),
        "matched_mean_iou_↑": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "per_label": {
            label: {
                **counts,
                "recall_↑": counts["tp"] / max(counts["tp"] + counts["fn"], 1),
            }
            for label, counts in per_label.items()
        },
    }
    return result, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-manifest", type=Path,
        default=PROJECT_ROOT / "data/qces_speech_event_v1_smoke100/scenes.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_speech_event_v1_detector",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = _read_jsonl(args.scene_manifest.resolve())
    by_split = {
        split: [scene for scene in scenes if scene["split"] == split]
        for split in ("train", "val", "test")
    }
    model = _load_model(device)
    cache_path = output_dir / "beats_strong_feature_cache.pt"
    features: dict[str, torch.Tensor] = {}
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        features = dict(payload["features"])
    for index, scene in enumerate(scenes, start=1):
        scene_id = str(scene["scene_id"])
        if scene_id not in features:
            features[scene_id] = _extract_feature(model, scene, device)
        if index % 10 == 0 or index == len(scenes):
            print(f"features {index}/{len(scenes)}", flush=True)
    torch.save(
        {
            "format": "qces_speech_event_beats_feature_cache_v1",
            "labels": list(LABELS),
            "scene_manifest_sha256": hashlib.sha256(args.scene_manifest.read_bytes()).hexdigest(),
            "features": features,
        },
        cache_path,
    )

    steps = int(model.seq_len)
    feature_dim = int(next(iter(features.values())).shape[-1])
    train_x = torch.stack([features[scene["scene_id"]].float() for scene in by_split["train"]])
    train_y = torch.stack([_target(scene, steps) for scene in by_split["train"]])
    val_x = torch.stack([features[scene["scene_id"]].float() for scene in by_split["val"]])
    val_y = torch.stack([_target(scene, steps) for scene in by_split["val"]])
    test_x = torch.stack([features[scene["scene_id"]].float() for scene in by_split["test"]])
    test_y = torch.stack([_target(scene, steps) for scene in by_split["test"]])
    pos = train_y.sum(dim=(0, 1))
    total = float(train_y.shape[0] * train_y.shape[1])
    pos_weight = ((total - pos) / pos.clamp_min(1.0)).clamp(1.0, 80.0).to(device)

    head = nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, feature_dim // 2),
        nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(feature_dim // 2, len(LABELS)),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    best = (-1.0, -1.0)
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(args.seed)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_x), generator=generator)
        head.train()
        losses: list[float] = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            logits = head(train_x[indices].to(device))
            loss = criterion(logits, train_y[indices].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        head.eval()
        with torch.no_grad():
            val_prob = torch.sigmoid(head(val_x.to(device))).cpu()
        thresholds = _class_thresholds(val_prob, val_y)
        frame_f1 = _frame_f1(val_prob, val_y, thresholds)
        event, _ = _event_metrics(by_split["val"], val_prob, thresholds)
        score = (event["event_f1_↑"], frame_f1)
        if score > best:
            best = score
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "val_frame_f1_↑": frame_f1,
                "val_event_f1_↑": event["event_f1_↑"],
            }
        )
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(history[-1]), flush=True)

    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(head(val_x.to(device))).cpu()
        test_prob = torch.sigmoid(head(test_x.to(device))).cpu()
    thresholds = _class_thresholds(val_prob, val_y)
    val_metrics, val_rows = _event_metrics(by_split["val"], val_prob, thresholds)
    test_metrics, test_rows = _event_metrics(by_split["test"], test_prob, thresholds)
    val_metrics["frame_f1_↑"] = _frame_f1(val_prob, val_y, thresholds)
    test_metrics["frame_f1_↑"] = _frame_f1(test_prob, test_y, thresholds)

    checkpoint_path = output_dir / "speech_event_detector.pt"
    torch.save(
        {
            "format": "qces_speech_event_detector_checkpoint_v1",
            "labels": list(LABELS),
            "head_state_dict": best_state,
            "feature_dim": feature_dim,
            "sequence_steps": steps,
            "class_thresholds": dict(zip(LABELS, thresholds.tolist())),
            "backbone": "PretrainedSED BEATs_strong_1 frozen",
        },
        checkpoint_path,
    )
    for split, rows in (("val", val_rows), ("test", test_rows)):
        _atomic_text(
            output_dir / f"detector_predictions_{split}.jsonl",
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        )
    receipt = {
        "format": "qces_speech_event_detector_receipt_v1",
        "complete": True,
        "labels": list(LABELS),
        "train_scenes": len(by_split["train"]),
        "val_scenes": len(by_split["val"]),
        "test_scenes": len(by_split["test"]),
        "epochs": args.epochs,
        "class_thresholds": dict(zip(LABELS, thresholds.tolist())),
        "val": val_metrics,
        "test": test_metrics,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "history": history,
    }
    _atomic_text(output_dir / "receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

