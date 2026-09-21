#!/usr/bin/env python3
"""Detector-first QCES baseline with PretrainedSED BEATs-Strong.

This is a sidecar implementation; it does not modify the previous temporary
BEATs frame detector.  The intended pipeline is:

    mixture -> BEATs-Strong frame detector -> event inventory -> QCES executor

The detector is trained on unique scenes, not duplicated QA rows, because the
audio detector should learn from each acoustic scene once.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRETRAINED_SED_ROOT = PROJECT_ROOT / "code/baseline/PretrainedSED"
if str(PRETRAINED_SED_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAINED_SED_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset

import models.prediction_wrapper as pretrained_prediction_wrapper
from models.beats.BEATs_wrapper import BEATsWrapper
from models.prediction_wrapper import PredictionsWrapper


DEFAULT_AUDIT_DIR = PROJECT_ROOT / "outputs/qces_detector_first_data_audit/v1"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data/qces_v6_full_cropbank_v2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_30class/beats_strong_head_v1"

# Keep downloaded checkpoints inside the cloned PretrainedSED repo regardless
# of where this script is launched from.
pretrained_prediction_wrapper.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")


@dataclass(frozen=True)
class SceneItem:
    scene_id: str
    split: str
    mixture_path: str
    duration_seconds: float
    sample_rate: int
    events: tuple[dict[str, Any], ...]


class SceneDataset(Dataset[tuple[torch.Tensor, SceneItem]]):
    def __init__(
        self,
        rows: Sequence[SceneItem],
        *,
        audio_root: Path,
        target_sample_rate: int = 16_000,
        fixed_seconds: float = 10.0,
    ) -> None:
        self.rows = list(rows)
        self.audio_root = audio_root
        self.target_sample_rate = int(target_sample_rate)
        self.fixed_samples = int(round(float(fixed_seconds) * self.target_sample_rate))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, SceneItem]:
        row = self.rows[index]
        path = self.audio_root / row.mixture_path
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float()
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.reshape(-1)
        if sample_rate != self.target_sample_rate:
            waveform = AF.resample(waveform, sample_rate, self.target_sample_rate)
        if waveform.numel() < self.fixed_samples:
            waveform = torch.nn.functional.pad(waveform, (0, self.fixed_samples - waveform.numel()))
        elif waveform.numel() > self.fixed_samples:
            waveform = waveform[: self.fixed_samples]
        return waveform, row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_AUDIT_DIR / "detector_scene_manifest_train.jsonl")
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_AUDIT_DIR / "detector_scene_manifest_val.jsonl")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_AUDIT_DIR / "ontology_ready_train.txt")
    parser.add_argument("--qa-val-manifest", type=Path, default=DEFAULT_DATASET_ROOT / "qces_val.jsonl")
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default="BEATs_strong_1")
    parser.add_argument(
        "--resume-detector-checkpoint",
        type=Path,
        help="Optional previously trained QCES detector checkpoint to continue from.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--unfreeze-last-blocks",
        type=int,
        default=0,
        help="Unfreeze the last N BEATs transformer blocks; 0 keeps the whole backbone frozen.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-val-scenes", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    )
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument("--min-duration", type=float, default=0.08)
    parser.add_argument("--merge-gap", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=2034)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_ontology(path: Path) -> list[str]:
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label = line.strip()
        if label and label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError(f"empty ontology: {path}")
    return labels


def load_scene_manifest(path: Path, label_to_id: Mapping[str, int], max_scenes: int = 0) -> list[SceneItem]:
    rows: list[SceneItem] = []
    for raw in read_jsonl(path):
        events: list[dict[str, Any]] = []
        for event in raw.get("events") or []:
            label = str(event.get("label") or "")
            if label not in label_to_id:
                continue
            events.append(
                {
                    **event,
                    "label": label,
                    "label_id": int(label_to_id[label]),
                    "onset_seconds": float(event.get("onset_seconds", 0.0)),
                    "offset_seconds": float(event.get("offset_seconds", 0.0)),
                }
            )
        rows.append(
            SceneItem(
                scene_id=str(raw["scene_id"]),
                split=str(raw.get("split") or ""),
                mixture_path=str(raw["mixture_path"]),
                duration_seconds=float(raw.get("duration_seconds") or 10.0),
                sample_rate=int(raw.get("sample_rate") or 32_000),
                events=tuple(events),
            )
        )
        if max_scenes and len(rows) >= max_scenes:
            break
    return rows


def collate(batch: Sequence[tuple[torch.Tensor, SceneItem]]) -> tuple[torch.Tensor, list[SceneItem]]:
    return torch.stack([item[0] for item in batch], dim=0), [item[1] for item in batch]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model(
    num_labels: int,
    checkpoint: str,
    device: torch.device,
    *,
    unfreeze_last_blocks: int = 0,
) -> PredictionsWrapper:
    model = PredictionsWrapper(
        BEATsWrapper(),
        checkpoint=checkpoint,
        n_classes_strong=num_labels,
        n_classes_weak=num_labels,
        seq_model_type=None,
        head_type="linear",
    )
    model.to(device)
    # PretrainedSED checkpoint gives a strong BEATs transformer.  Default keeps
    # it frozen; optionally unfreeze the last transformer blocks for light
    # domain adaptation.
    model.model.eval()
    model.model.requires_grad_(False)
    if unfreeze_last_blocks > 0:
        if unfreeze_last_blocks > 12:
            raise ValueError("--unfreeze-last-blocks must be <= 12 for BEATs")
        first_unfrozen = 12 - int(unfreeze_last_blocks)
        for name, parameter in model.model.named_parameters():
            for layer_index in range(first_unfrozen, 12):
                if f"beats.encoder.layers.{layer_index}." in name:
                    parameter.requires_grad = True
                    break
            if "beats.encoder.layer_norm." in name:
                parameter.requires_grad = True
    return model


def interpolate_sequence(features: torch.Tensor, seq_len: int) -> torch.Tensor:
    if features.size(-2) > seq_len:
        return torch.nn.functional.adaptive_avg_pool1d(features.transpose(1, 2), seq_len).transpose(1, 2)
    if features.size(-2) < seq_len:
        return torch.nn.functional.interpolate(features.transpose(1, 2), size=seq_len, mode="linear").transpose(1, 2)
    return features


def strong_logits_from_waveform(model: PredictionsWrapper, waveforms: torch.Tensor) -> torch.Tensor:
    backbone_trainable = any(parameter.requires_grad for parameter in model.model.parameters())
    context = contextlib.nullcontext() if backbone_trainable else torch.no_grad()
    with context:
        mel = model.mel_forward(waveforms)
        features = model.model(mel)
        features = interpolate_sequence(features, model.seq_len)
        features = model.seq_model(features)
    return model.strong_head(features)


def build_targets(rows: Sequence[SceneItem], time_steps: int, num_labels: int, device: torch.device) -> torch.Tensor:
    targets = torch.zeros((len(rows), time_steps, num_labels), dtype=torch.float32, device=device)
    for batch_index, row in enumerate(rows):
        duration = max(float(row.duration_seconds), 1e-6)
        for event in row.events:
            label_id = int(event["label_id"])
            onset = max(0.0, float(event["onset_seconds"]))
            offset = min(duration, float(event["offset_seconds"]))
            if label_id < 0 or label_id >= num_labels or offset <= onset:
                continue
            start = max(0, min(time_steps - 1, int(math.floor(onset / duration * time_steps))))
            end = max(start + 1, min(time_steps, int(math.ceil(offset / duration * time_steps))))
            targets[batch_index, start:end, label_id] = 1.0
    return targets


def compute_pos_weight(rows: Sequence[SceneItem], num_labels: int, time_steps: int = 250) -> torch.Tensor:
    pos = torch.zeros(num_labels, dtype=torch.float64)
    total = float(len(rows) * time_steps)
    for row in rows:
        pos += build_targets([row], time_steps, num_labels, torch.device("cpu"))[0].sum(dim=0).double()
    neg = torch.full_like(pos, total) - pos
    return (neg / pos.clamp_min(1.0)).clamp(min=1.0, max=80.0).float()


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    inter = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return inter / union if union > 0 else 0.0


def probs_to_events(
    probs: torch.Tensor,
    row: SceneItem,
    labels: Sequence[str],
    *,
    threshold: float,
    min_duration: float,
    merge_gap: float,
) -> list[dict[str, Any]]:
    time_steps, num_labels = probs.shape
    duration = max(float(row.duration_seconds), 1e-6)
    frame_seconds = duration / max(time_steps, 1)
    events: list[dict[str, Any]] = []
    active = (probs >= threshold).cpu().numpy().astype(bool)
    for label_id in range(num_labels):
        segments: list[tuple[int, int]] = []
        start: int | None = None
        for index, flag in enumerate(active[:, label_id].tolist() + [False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                segments.append((start, index))
                start = None
        merged: list[tuple[float, float, float]] = []
        for start_idx, end_idx in segments:
            onset = start_idx * frame_seconds
            offset = end_idx * frame_seconds
            confidence = float(probs[start_idx:end_idx, label_id].max().item())
            if merged and onset <= merged[-1][1] + merge_gap:
                prev_onset, prev_offset, prev_confidence = merged[-1]
                merged[-1] = (prev_onset, max(prev_offset, offset), max(prev_confidence, confidence))
            else:
                merged.append((onset, offset, confidence))
        for onset, offset, confidence in merged:
            if offset - onset < min_duration:
                continue
            events.append(
                {
                    "label": labels[label_id],
                    "label_id": label_id,
                    "onset_seconds": round(onset, 4),
                    "offset_seconds": round(offset, 4),
                    "confidence": confidence,
                }
            )
    events.sort(key=lambda item: (float(item["onset_seconds"]), float(item["offset_seconds"]), str(item["label"])))
    return events


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision_↑": precision, "recall_↑": recall, "f1_↑": f1}


def match_event_f1(
    predicted: Sequence[Mapping[str, Any]],
    gold: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[int, int, int]:
    used_gold: set[int] = set()
    tp = 0
    for pred in sorted(predicted, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        best_index: int | None = None
        best_iou = 0.0
        for index, gold_event in enumerate(gold):
            if index in used_gold or str(pred["label"]) != str(gold_event["label"]):
                continue
            score = interval_iou(
                (float(pred["onset_seconds"]), float(pred["offset_seconds"])),
                (float(gold_event["onset_seconds"]), float(gold_event["offset_seconds"])),
            )
            if score > best_iou:
                best_index = index
                best_iou = score
        if best_index is not None and best_iou >= iou_threshold:
            used_gold.add(best_index)
            tp += 1
    return tp, len(predicted) - tp, len(gold) - tp


@torch.no_grad()
def run_model(
    model: PredictionsWrapper,
    loader: DataLoader,
    labels: Sequence[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    output: list[dict[str, Any]] = []
    for waveforms, rows in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        logits = strong_logits_from_waveform(model, waveforms)
        probs = torch.sigmoid(logits).cpu()
        targets = build_targets(rows, probs.size(1), len(labels), torch.device("cpu"))
        for index, row in enumerate(rows):
            output.append(
                {
                    "scene_id": row.scene_id,
                    "split": row.split,
                    "duration_seconds": row.duration_seconds,
                    "probs": probs[index],
                    "targets": targets[index],
                    "gold_events": list(row.events),
                }
            )
    return output


def summarize_detector(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    threshold: float,
    event_iou_threshold: float,
    min_duration: float,
    merge_gap: float,
) -> dict[str, Any]:
    frame_tp = frame_fp = frame_fn = 0
    scene_tp = scene_fp = scene_fn = 0
    event_tp = event_fp = event_fn = 0
    kept_labels: list[float] = []
    false_labels: list[float] = []
    label_recalls: list[float] = []
    onset_errors: list[float] = []
    offset_errors: list[float] = []

    for row in predictions:
        probs: torch.Tensor = row["probs"]
        targets: torch.Tensor = row["targets"]
        pred_frame = probs >= threshold
        gold_frame = targets >= 0.5
        frame_tp += int((pred_frame & gold_frame.bool()).sum().item())
        frame_fp += int((pred_frame & ~gold_frame.bool()).sum().item())
        frame_fn += int((~pred_frame & gold_frame.bool()).sum().item())

        pred_labels = {labels[index] for index in torch.where(probs.max(dim=0).values >= threshold)[0].tolist()}
        gold_labels = {str(event["label"]) for event in row["gold_events"]}
        scene_tp += len(pred_labels & gold_labels)
        scene_fp += len(pred_labels - gold_labels)
        scene_fn += len(gold_labels - pred_labels)
        kept_labels.append(float(len(pred_labels)))
        false_labels.append(float(len(pred_labels - gold_labels)))
        label_recalls.append(len(pred_labels & gold_labels) / max(len(gold_labels), 1))

        pseudo_scene = SceneItem(
            scene_id=str(row["scene_id"]),
            split=str(row["split"]),
            mixture_path="",
            duration_seconds=float(row["duration_seconds"]),
            sample_rate=16_000,
            events=tuple(row["gold_events"]),
        )
        pred_events = probs_to_events(
            probs,
            pseudo_scene,
            labels,
            threshold=threshold,
            min_duration=min_duration,
            merge_gap=merge_gap,
        )
        tp, fp, fn = match_event_f1(pred_events, row["gold_events"], iou_threshold=event_iou_threshold)
        event_tp += tp
        event_fp += fp
        event_fn += fn

        used_gold: set[int] = set()
        for pred in sorted(pred_events, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
            best_index: int | None = None
            best_iou = 0.0
            for index, gold_event in enumerate(row["gold_events"]):
                if index in used_gold or pred["label"] != gold_event["label"]:
                    continue
                score = interval_iou(
                    (float(pred["onset_seconds"]), float(pred["offset_seconds"])),
                    (float(gold_event["onset_seconds"]), float(gold_event["offset_seconds"])),
                )
                if score > best_iou:
                    best_index = index
                    best_iou = score
            if best_index is not None and best_iou >= event_iou_threshold:
                gold_event = row["gold_events"][best_index]
                used_gold.add(best_index)
                onset_errors.append(abs(float(pred["onset_seconds"]) - float(gold_event["onset_seconds"])))
                offset_errors.append(abs(float(pred["offset_seconds"]) - float(gold_event["offset_seconds"])))

    return {
        "threshold": threshold,
        "frame": prf(frame_tp, frame_fp, frame_fn),
        "scene_label": {
            **prf(scene_tp, scene_fp, scene_fn),
            "avg_labels_kept_↓": float(np.mean(kept_labels)) if kept_labels else 0.0,
            "avg_false_positive_labels_↓": float(np.mean(false_labels)) if false_labels else 0.0,
            "avg_label_recall_↑": float(np.mean(label_recalls)) if label_recalls else 0.0,
        },
        "event_iou": {
            **prf(event_tp, event_fp, event_fn),
            "iou_threshold": event_iou_threshold,
            "mean_onset_abs_error_s_↓": float(np.mean(onset_errors)) if onset_errors else None,
            "mean_offset_abs_error_s_↓": float(np.mean(offset_errors)) if offset_errors else None,
        },
    }


def occurrence(events: Sequence[Mapping[str, Any]], label: str, ordinal: int) -> Mapping[str, Any] | None:
    candidates = [event for event in events if str(event.get("label")) == label]
    candidates.sort(key=lambda item: (float(item["onset_seconds"]), float(item["offset_seconds"])))
    if ordinal <= 0 or len(candidates) < ordinal:
        return None
    return candidates[ordinal - 1]


def execute_qa(row: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    relation = str(row.get("relation") or "")
    # Historical QA manifests used query_label/query_instance_ordinal, while
    # the clean-evidence manifests use anchor_label/anchor_ordinal.  Treating
    # the latter as empty silently turns every positive question into NONE and
    # produces a misleading 0% answerable-accuracy report.
    query_label = str(row.get("query_label") or row.get("anchor_label") or "")
    ordinal = int(
        row.get("query_instance_ordinal") or row.get("anchor_ordinal") or 1
    )
    if relation in {"after", "before"}:
        anchor = occurrence(events, query_label, ordinal)
        if anchor is None:
            return "no_evidence", True
        anchor_onset = float(anchor["onset_seconds"])
        if relation == "after":
            candidates = [event for event in events if float(event["onset_seconds"]) > anchor_onset + 1e-6]
            if not candidates:
                return "no_evidence", True
            answer = min(candidates, key=lambda item: (float(item["onset_seconds"]), -float(item.get("confidence", 0.0))))
        else:
            candidates = [event for event in events if float(event["onset_seconds"]) < anchor_onset - 1e-6]
            if not candidates:
                return "no_evidence", True
            answer = max(candidates, key=lambda item: (float(item["onset_seconds"]), float(item.get("confidence", 0.0))))
        return str(answer["label"]), False
    if relation == "first":
        candidate_labels = [str(label) for label in row.get("query_candidate_labels") or []]
        candidates = [event for event in events if event.get("label") in set(candidate_labels)]
        if not candidates:
            return "no_evidence", True
        answer = min(candidates, key=lambda item: (float(item["onset_seconds"]), -float(item.get("confidence", 0.0))))
        return str(answer["label"]), False
    return "unsupported", True


def summarize_qa(
    qa_rows: Sequence[Mapping[str, Any]],
    pred_events_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in qa_rows:
        scene_id = str(row.get("scene_id") or "")
        pred_answer, pred_noev = execute_qa(row, pred_events_by_scene.get(scene_id, ()))
        ok = pred_answer == str(row.get("answer")) and pred_noev == bool(row.get("no_evidence"))
        rows.append(
            {
                "scene_id": scene_id,
                "question": row.get("question"),
                "relation": row.get("relation"),
                "answer": row.get("answer"),
                "no_evidence": row.get("no_evidence"),
                "pred_answer": pred_answer,
                "pred_no_evidence": pred_noev,
                "ok": ok,
            }
        )
    by_relation: dict[str, dict[str, Any]] = {}
    for relation in sorted({str(row["relation"]) for row in rows}):
        subset = [row for row in rows if str(row["relation"]) == relation]
        by_relation[relation] = {
            "items": len(subset),
            "accuracy_↑": sum(bool(row["ok"]) for row in subset) / max(len(subset), 1),
        }
    answerable = [row for row in rows if not bool(row["no_evidence"])]
    no_evidence = [row for row in rows if bool(row["no_evidence"])]
    return {
        "items": len(rows),
        "accuracy_↑": sum(bool(row["ok"]) for row in rows) / max(len(rows), 1),
        "answerable_accuracy_↑": sum(bool(row["ok"]) for row in answerable) / max(len(answerable), 1),
        "no_evidence_accuracy_↑": sum(bool(row["ok"]) for row in no_evidence) / max(len(no_evidence), 1),
        "answerable_items": len(answerable),
        "no_evidence_items": len(no_evidence),
        "by_relation": by_relation,
        "rows": rows,
    }


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def choose_threshold(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    scene = summary["scene_label"]
    event = summary["event_iou"]
    # Main detector role is building a clean inventory; choose label F1 first,
    # then temporal event F1, then fewer false positives.
    return (
        float(scene["f1_↑"]),
        float(event["f1_↑"]),
        -float(scene["avg_false_positive_labels_↓"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = make_device(args.device)
    labels = load_ontology(args.ontology.resolve())
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_rows = load_scene_manifest(args.train_manifest.resolve(), label_to_id, args.max_train_scenes)
    val_rows = load_scene_manifest(args.val_manifest.resolve(), label_to_id, args.max_val_scenes)
    if not train_rows or not val_rows:
        raise SystemExit("empty train/val detector manifest")
    print(
        f"device={device} checkpoint={args.checkpoint} train_scenes={len(train_rows)} "
        f"val_scenes={len(val_rows)} labels={len(labels)}",
        flush=True,
    )

    model = load_model(
        len(labels),
        args.checkpoint,
        device,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
    )
    if args.resume_detector_checkpoint is not None:
        checkpoint_path = args.resume_detector_checkpoint.resolve()
        payload = torch.load(checkpoint_path, map_location="cpu")
        checkpoint_labels = payload.get("labels")
        if checkpoint_labels != labels:
            raise ValueError(
                "resume checkpoint labels do not match ontology: "
                f"checkpoint={len(checkpoint_labels or [])}, current={len(labels)}"
            )
        model.load_state_dict(payload["model_state_dict"], strict=True)
        print(f"resumed_detector_checkpoint={checkpoint_path}", flush=True)
    train_loader = DataLoader(
        SceneDataset(train_rows, audio_root=args.audio_root.resolve()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    val_loader = DataLoader(
        SceneDataset(val_rows, audio_root=args.audio_root.resolve()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    pos_weight = compute_pos_weight(train_rows, len(labels), time_steps=model.seq_len).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    head_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("model.")
    ]
    backbone_params = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("model.")
    ]
    trainable_params = [*head_params, *backbone_params]
    optimizer_groups: list[dict[str, Any]] = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        optimizer_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    print(
        f"trainable_head_params={sum(p.numel() for p in head_params)} "
        f"trainable_backbone_params={sum(p.numel() for p in backbone_params)}",
        flush=True,
    )

    best_score = (-1.0, -1.0, -1e9)
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if any(parameter.requires_grad for parameter in model.model.parameters()):
            model.model.train()
        else:
            model.model.eval()
        losses: list[float] = []
        for step, (waveforms, rows) in enumerate(train_loader, start=1):
            waveforms = waveforms.to(device, non_blocking=True)
            logits = strong_logits_from_waveform(model, waveforms)
            targets = build_targets(rows, logits.size(1), len(labels), device)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step % 50 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={np.mean(losses[-50:]):.4f}", flush=True)

        val_predictions = run_model(model, val_loader, labels, device)
        threshold_summaries = [
            summarize_detector(
                val_predictions,
                labels,
                threshold=threshold,
                event_iou_threshold=args.event_iou_threshold,
                min_duration=args.min_duration,
                merge_gap=args.merge_gap,
            )
            for threshold in args.thresholds
        ]
        selected = max(threshold_summaries, key=choose_threshold)
        selected_score = choose_threshold(selected)
        if selected_score > best_score:
            best_score = selected_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "selected_threshold": selected["threshold"],
            "scene_label_f1_↑": selected["scene_label"]["f1_↑"],
            "scene_label_precision_↑": selected["scene_label"]["precision_↑"],
            "scene_label_recall_↑": selected["scene_label"]["recall_↑"],
            "event_f1_↑": selected["event_iou"]["f1_↑"],
            "frame_f1_↑": selected["frame"]["f1_↑"],
            "avg_false_positive_labels_↓": selected["scene_label"]["avg_false_positive_labels_↓"],
            "threshold_summaries": threshold_summaries,
        }
        history.append(epoch_row)
        print(json.dumps({k: v for k, v in epoch_row.items() if k != "threshold_summaries"}, sort_keys=True), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    val_predictions = run_model(model, val_loader, labels, device)
    threshold_summaries = [
        summarize_detector(
            val_predictions,
            labels,
            threshold=threshold,
            event_iou_threshold=args.event_iou_threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
        )
        for threshold in args.thresholds
    ]
    selected = max(threshold_summaries, key=choose_threshold)
    best_threshold = float(selected["threshold"])

    pred_events_by_scene: dict[str, list[dict[str, Any]]] = {}
    prediction_rows: list[dict[str, Any]] = []
    for row in val_predictions:
        pseudo_scene = SceneItem(
            scene_id=str(row["scene_id"]),
            split=str(row["split"]),
            mixture_path="",
            duration_seconds=float(row["duration_seconds"]),
            sample_rate=16_000,
            events=tuple(row["gold_events"]),
        )
        events = probs_to_events(
            row["probs"],
            pseudo_scene,
            labels,
            threshold=best_threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
        )
        pred_events_by_scene[pseudo_scene.scene_id] = events
        prediction_rows.append(
            {
                "scene_id": pseudo_scene.scene_id,
                "split": pseudo_scene.split,
                "threshold": best_threshold,
                "predicted_events": events,
                "gold_events": row["gold_events"],
            }
        )

    qa_summary: dict[str, Any] | None = None
    if args.qa_val_manifest and args.qa_val_manifest.exists():
        qa_rows = [
            row for row in read_jsonl(args.qa_val_manifest.resolve())
            if str(row.get("scene_id") or "") in pred_events_by_scene
        ]
        qa_summary = summarize_qa(qa_rows, pred_events_by_scene)

    report = {
        "format": "qces_pretrainedsed_beats_strong_detector_v1",
        "note": "Frozen PretrainedSED BEATs-Strong backbone with trainable 30-class QCES frame heads.",
        "pretrained_sed_root": str(PRETRAINED_SED_ROOT),
        "checkpoint": args.checkpoint,
        "resume_detector_checkpoint": (
            str(args.resume_detector_checkpoint.resolve())
            if args.resume_detector_checkpoint is not None
            else None
        ),
        "train_manifest": str(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "ontology": str(args.ontology.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "device": str(device),
        "train_scenes": len(train_rows),
        "val_scenes": len(val_rows),
        "labels": labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "backbone_lr": args.backbone_lr,
        "unfreeze_last_blocks": args.unfreeze_last_blocks,
        "min_duration": args.min_duration,
        "merge_gap": args.merge_gap,
        "history": history,
        "best_threshold": best_threshold,
        "val_detector": {
            "threshold_summaries": threshold_summaries,
            "selected": selected,
        },
        "val_qa": {key: value for key, value in (qa_summary or {}).items() if key != "rows"} if qa_summary else None,
    }

    save_json(output_dir / "training_report.json", report)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": labels,
            "best_threshold": best_threshold,
            "report": report,
        },
        output_dir / "pretrainedsed_beats_qces_detector.pt",
    )
    with (output_dir / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if qa_summary is not None:
        with (output_dir / "val_qa_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in qa_summary["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    md = [
        "# QCES PretrainedSED BEATs-Strong detector",
        "",
        "Sidecar detector-first experiment: frozen PretrainedSED BEATs-Strong backbone + trainable QCES frame head.",
        "",
        f"- train scenes: {len(train_rows)}",
        f"- val scenes: {len(val_rows)}",
        f"- labels: {len(labels)}",
        f"- best threshold: {best_threshold:.2f}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| scene-label F1 ↑ | {selected['scene_label']['f1_↑']:.3f} |",
        f"| scene-label precision ↑ | {selected['scene_label']['precision_↑']:.3f} |",
        f"| scene-label recall ↑ | {selected['scene_label']['recall_↑']:.3f} |",
        f"| avg labels kept ↓ | {selected['scene_label']['avg_labels_kept_↓']:.2f} |",
        f"| avg false-positive labels ↓ | {selected['scene_label']['avg_false_positive_labels_↓']:.2f} |",
        f"| event F1 @ IoU {args.event_iou_threshold:.2f} ↑ | {selected['event_iou']['f1_↑']:.3f} |",
        f"| frame F1 ↑ | {selected['frame']['f1_↑']:.3f} |",
    ]
    if qa_summary is not None:
        md.extend(
            [
                "",
                "## Downstream QA with predicted event inventory",
                "",
                "| split | items | accuracy ↑ | answerable ↑ | no-evidence ↑ |",
                "|---|---:|---:|---:|---:|",
                (
                    f"| val | {qa_summary['items']} | {qa_summary['accuracy_↑']:.3f} | "
                    f"{qa_summary['answerable_accuracy_↑']:.3f} | {qa_summary['no_evidence_accuracy_↑']:.3f} |"
                ),
                "",
                "| relation | items | accuracy ↑ |",
                "|---|---:|---:|",
            ]
        )
        for relation, row in qa_summary["by_relation"].items():
            md.append(f"| {relation} | {row['items']} | {row['accuracy_↑']:.3f} |")
    (output_dir / "training_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
