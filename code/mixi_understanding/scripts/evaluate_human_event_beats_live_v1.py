#!/usr/bin/env python3
"""Calibrate and evaluate five human-event classes for the live BEATs demo."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import median_filter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRETRAINED_SED_ROOT = PROJECT_ROOT / "code/baseline/PretrainedSED"
for value in (PROJECT_ROOT / "code", PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mixi_understanding.scripts.infer_qces_combined_honest_v1 import (  # noqa: E402
    FRAME_SECONDS,
    MEDIAN_FRAMES,
    contiguous_segments,
    load_audio,
)


RAW_TO_LABEL = {
    "Laughter": "Laughter",
    "Giggle": "Giggle",
    "Conversation": "Conversation",
    "Shout": "Shout",
    "Crying, sobbing": "Crying_and_sobbing",
}
THRESHOLD_GRID = (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def batched(rows: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def gold_segments(row: dict[str, Any], label: str) -> list[tuple[float, float]]:
    return [
        (float(event["onset_seconds"]), float(event["offset_seconds"]))
        for event in row["events"]
        if str(event["label"]) == label
    ]


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def match_events(
    gold: list[tuple[float, float]],
    predicted: list[tuple[float, float]],
    minimum_iou: float = 0.30,
) -> tuple[int, int, int, list[float]]:
    candidates = sorted(
        (
            (interval_iou(gold_span, pred_span), gold_index, pred_index)
            for gold_index, gold_span in enumerate(gold)
            for pred_index, pred_span in enumerate(predicted)
        ),
        reverse=True,
    )
    used_gold: set[int] = set()
    used_predicted: set[int] = set()
    matched_ious: list[float] = []
    for iou, gold_index, pred_index in candidates:
        if iou < minimum_iou:
            break
        if gold_index in used_gold or pred_index in used_predicted:
            continue
        used_gold.add(gold_index)
        used_predicted.add(pred_index)
        matched_ious.append(float(iou))
    return len(matched_ious), len(predicted) - len(matched_ious), len(gold) - len(matched_ious), matched_ious


def decode(scores: np.ndarray, duration: float, threshold: float) -> list[tuple[float, float]]:
    valid_frames = min(len(scores), int(math.ceil(duration / FRAME_SECONDS)))
    return [
        (start * FRAME_SECONDS, min(duration, end * FRAME_SECONDS))
        for start, end in contiguous_segments(
            scores,
            threshold=threshold,
            valid_frames=valid_frames,
            merge_gap_seconds=0.16,
            minimum_seconds=0.08,
        )
    ]


def metrics(
    rows: list[dict[str, Any]],
    score_rows: list[np.ndarray],
    label_index: int,
    label: str,
    threshold: float,
) -> dict[str, Any]:
    clip = Counter()
    event = Counter()
    matched_ious: list[float] = []
    frame_tp = frame_fp = frame_fn = 0
    for row, scores in zip(rows, score_rows):
        duration = float(row["duration_seconds"])
        gold = gold_segments(row, label)
        predicted = decode(scores[:, label_index], duration, threshold)
        gold_present = bool(gold)
        predicted_present = bool(predicted)
        clip[(gold_present, predicted_present)] += 1
        tp, fp, fn, ious = match_events(gold, predicted)
        event["tp"] += tp
        event["fp"] += fp
        event["fn"] += fn
        matched_ious.extend(ious)

        valid_frames = min(len(scores), int(math.ceil(duration / FRAME_SECONDS)))
        gold_mask = np.zeros(valid_frames, dtype=bool)
        for start, end in gold:
            left = max(0, int(math.floor(start / FRAME_SECONDS)))
            right = min(valid_frames, int(math.ceil(end / FRAME_SECONDS)))
            gold_mask[left:right] = True
        predicted_mask = scores[:valid_frames, label_index] >= threshold
        frame_tp += int(np.logical_and(gold_mask, predicted_mask).sum())
        frame_fp += int(np.logical_and(~gold_mask, predicted_mask).sum())
        frame_fn += int(np.logical_and(gold_mask, ~predicted_mask).sum())

    tp_clip = clip[(True, True)]
    fn_clip = clip[(True, False)]
    fp_clip = clip[(False, True)]
    tn_clip = clip[(False, False)]
    positive_recall = tp_clip / max(tp_clip + fn_clip, 1)
    negative_recall = tn_clip / max(tn_clip + fp_clip, 1)
    event_precision = event["tp"] / max(event["tp"] + event["fp"], 1)
    event_recall = event["tp"] / max(event["tp"] + event["fn"], 1)
    frame_precision = frame_tp / max(frame_tp + frame_fp, 1)
    frame_recall = frame_tp / max(frame_tp + frame_fn, 1)
    return {
        "threshold": threshold,
        "clip_tp": tp_clip,
        "clip_fp": fp_clip,
        "clip_fn": fn_clip,
        "clip_tn": tn_clip,
        "clip_accuracy_↑": (tp_clip + tn_clip) / max(len(rows), 1),
        "clip_balanced_accuracy_↑": 0.5 * (positive_recall + negative_recall),
        "clip_positive_recall_↑": positive_recall,
        "clip_negative_recall_↑": negative_recall,
        "event_tp": event["tp"],
        "event_fp": event["fp"],
        "event_fn": event["fn"],
        "event_precision_↑": event_precision,
        "event_recall_↑": event_recall,
        "event_f1_↑": 2 * event_precision * event_recall / max(event_precision + event_recall, 1e-12),
        "matched_mean_iou_↑": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "frame_precision_↑": frame_precision,
        "frame_recall_↑": frame_recall,
        "frame_f1_↑": 2 * frame_precision * frame_recall / max(frame_precision + frame_recall, 1e-12),
    }


def infer_scores(
    rows: list[dict[str, Any]],
    model: Any,
    label_to_id: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> list[np.ndarray]:
    score_rows: list[np.ndarray] = []
    raw_labels = list(RAW_TO_LABEL)
    selected_ids = [label_to_id[label] for label in raw_labels]
    for batch_index, batch in enumerate(batched(rows, batch_size), 1):
        waveforms: list[torch.Tensor] = []
        for row in batch:
            waveform, sample_rate = load_audio(resolve(row["mixture_path"]))
            target = int(10.0 * sample_rate)
            tensor = torch.from_numpy(waveform[:target])
            if len(tensor) < target:
                tensor = F.pad(tensor, (0, target - len(tensor)))
            waveforms.append(tensor)
        audio = torch.stack(waveforms).to(device)
        with torch.inference_mode():
            mel = model.mel_forward(audio)
            logits, _ = model(mel)
            probabilities = logits.sigmoid().transpose(1, 2).float().cpu().numpy()
        for sample in probabilities:
            selected = sample[:, selected_ids]
            selected = median_filter(selected, size=(MEDIAN_FRAMES, 1), mode="nearest")
            score_rows.append(np.asarray(selected, dtype=np.float32))
        if batch_index % 25 == 0 or len(score_rows) == len(rows):
            print(f"[INFER] {len(score_rows)}/{len(rows)}", flush=True)
    return score_rows


def data_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in RAW_TO_LABEL.values():
        scenes = 0
        occurrences = 0
        duration = 0.0
        for row in rows:
            events = [event for event in row["events"] if event["label"] == label]
            scenes += bool(events)
            occurrences += len(events)
            duration += sum(float(event["offset_seconds"]) - float(event["onset_seconds"]) for event in events)
        result[label] = {
            "positive_scenes": scenes,
            "occurrences": occurrences,
            "active_seconds": duration,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_clean_detector_protocol_v1_current",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_live_human_events_beats_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    import models.prediction_wrapper as prediction_wrapper
    from data_util.audioset_classes import as_strong_train_classes
    from models.beats.BEATs_wrapper import BEATsWrapper
    from models.prediction_wrapper import PredictionsWrapper

    prediction_wrapper.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = args.protocol_dir.resolve()
    splits = {
        "train": read_jsonl(protocol / "detector_manifest_train.jsonl"),
        "dev": read_jsonl(protocol / "detector_manifest_dev.jsonl"),
        "test": read_jsonl(protocol / "detector_manifest_test.jsonl"),
    }
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
    model.eval().to(device)
    label_to_id = {label: index for index, label in enumerate(as_strong_train_classes)}
    missing = sorted(set(RAW_TO_LABEL) - set(label_to_id))
    if missing:
        raise RuntimeError(f"BEATs-Strong is missing classes: {missing}")

    score_cache: dict[str, list[np.ndarray]] = {}
    for split in ("dev", "test"):
        print(f"[SPLIT] {split} scenes={len(splits[split])}", flush=True)
        score_cache[split] = infer_scores(
            splits[split], model, label_to_id, device, args.batch_size
        )
        torch.save(
            {
                "scene_ids": [row["scene_id"] for row in splits[split]],
                "scores": torch.from_numpy(np.stack(score_cache[split])).half(),
                "labels": list(RAW_TO_LABEL.values()),
                "raw_labels": list(RAW_TO_LABEL),
                "frame_seconds": FRAME_SECONDS,
            },
            output / f"{split}_frame_scores.pt",
        )

    selected_thresholds: dict[str, float] = {}
    dev_grid: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}
    for label_index, label in enumerate(RAW_TO_LABEL.values()):
        values = [
            metrics(splits["dev"], score_cache["dev"], label_index, label, threshold)
            for threshold in THRESHOLD_GRID
        ]
        # Presence-balanced accuracy is the primary user-facing objective;
        # temporal event F1 breaks ties without touching the test split.
        best = max(
            values,
            key=lambda row: (
                row["clip_balanced_accuracy_↑"],
                row["event_f1_↑"],
                row["matched_mean_iou_↑"],
                -row["threshold"],
            ),
        )
        selected_thresholds[label] = float(best["threshold"])
        dev_grid[label] = {"selected": best, "grid": values}
        test_metrics[label] = metrics(
            splits["test"],
            score_cache["test"],
            label_index,
            label,
            float(best["threshold"]),
        )

    macro = {
        metric: float(np.mean([row[metric] for row in test_metrics.values()]))
        for metric in (
            "clip_accuracy_↑",
            "clip_balanced_accuracy_↑",
            "event_f1_↑",
            "matched_mean_iou_↑",
            "frame_f1_↑",
        )
    }
    receipt = {
        "format": "qces_live_human_events_beats_eval_v1",
        "complete": True,
        "model": "public PretrainedSED BEATs-Strong (unmodified)",
        "qces_finetuning": False,
        "classes": list(RAW_TO_LABEL.values()),
        "raw_class_mapping": RAW_TO_LABEL,
        "data": {split: data_counts(rows) for split, rows in splits.items()},
        "data_scene_counts": {split: len(rows) for split, rows in splits.items()},
        "threshold_selection": "per-class maximum dev clip balanced accuracy; event F1 and IoU tie-breakers",
        "selected_thresholds": selected_thresholds,
        "dev": dev_grid,
        "test": test_metrics,
        "test_macro": macro,
        "metric_definitions": {
            "clip_accuracy": "presence/absence accuracy; inflated by class imbalance",
            "clip_balanced_accuracy": "mean of positive recall and negative recall",
            "event_f1": "greedy same-class event matching at temporal IoU >= 0.30",
            "matched_mean_iou": "mean temporal IoU over matched true-positive events only",
            "frame_f1": "40 ms frame-level activity F1",
        },
        "limitations": [
            "AudioSet-Strong-derived evaluation is not guaranteed disjoint from public BEATs pretraining.",
            "Matched mean IoU excludes missed events and must be read together with event recall/F1.",
        ],
        "test_annotations_used_for_threshold_selection": False,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"thresholds": selected_thresholds, "test": test_metrics, "macro": macro}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
