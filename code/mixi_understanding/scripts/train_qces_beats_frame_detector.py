#!/usr/bin/env python3
"""Temporary detector-first baseline: frozen BEATs + trainable frame head.

This is not the final PretrainedSED/BEATs-Strong implementation.  It uses the
BEATs AS2M checkpoint already present in this repository as a pragmatic smoke
test for the detector-first direction:

    mixture audio -> frozen BEATs patch tokens -> 30-class frame head

The script trains on unique-scene detector manifests produced by
``audit_qces_detector_first_data.py`` and can optionally evaluate downstream
QCES after/before/first QA using the predicted event inventory.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BEATS_ROOT = PROJECT_ROOT / "code/baseline/beats-unilm"
BEATS_SOURCE = BEATS_ROOT / "beats"
if str(BEATS_SOURCE) not in sys.path:
    sys.path.insert(0, str(BEATS_SOURCE))

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as audio_functional
from torch.utils.data import DataLoader, Dataset


DEFAULT_AUDIT_DIR = PROJECT_ROOT / "outputs/qces_detector_first_data_audit/v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_beats_frame_detector_30class/smoke_v1"
DEFAULT_BEATS_CHECKPOINT = (
    BEATS_ROOT / "checkpoint/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
)
DEFAULT_AUDIO_ROOT = PROJECT_ROOT / "data/qces_v6_full_cropbank_v2"


@dataclass(frozen=True)
class SceneItem:
    scene_id: str
    split: str
    mixture_path: str
    duration_seconds: float
    sample_rate: int
    events: tuple[dict[str, Any], ...]


class SceneDataset(Dataset[SceneItem]):
    def __init__(
        self,
        rows: Sequence[SceneItem],
        *,
        project_root: Path,
        target_sample_rate: int = 16_000,
        fixed_seconds: float = 10.0,
    ) -> None:
        self.rows = list(rows)
        self.project_root = project_root
        self.target_sample_rate = target_sample_rate
        self.fixed_samples = int(round(fixed_seconds * target_sample_rate))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, SceneItem]:
        row = self.rows[index]
        path = self.project_root / row.mixture_path
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.float()
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.reshape(-1)
        if sample_rate != self.target_sample_rate:
            waveform = audio_functional.resample(waveform, sample_rate, self.target_sample_rate)
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
    parser.add_argument("--qa-val-manifest", type=Path, default=PROJECT_ROOT / "data/qces_v6_full_cropbank_v2/qces_val.jsonl")
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--beats-checkpoint", type=Path, default=DEFAULT_BEATS_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-val-scenes", type=int, default=0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_scene_manifest(path: Path, label_to_id: Mapping[str, int], max_scenes: int = 0) -> list[SceneItem]:
    rows: list[SceneItem] = []
    for raw in read_jsonl(path):
        events = []
        for event in raw.get("events") or []:
            label = str(event.get("label") or "")
            if label not in label_to_id:
                continue
            events.append(
                {
                    **event,
                    "label": label,
                    "label_id": label_to_id[label],
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


def load_ontology(path: Path) -> list[str]:
    labels = []
    for line in path.read_text(encoding="utf-8").splitlines():
        label = line.strip()
        if label and label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError(f"empty ontology: {path}")
    return labels


def collate(batch: Sequence[tuple[torch.Tensor, SceneItem]]) -> tuple[torch.Tensor, list[SceneItem]]:
    waveforms = torch.stack([item[0] for item in batch], dim=0)
    rows = [item[1] for item in batch]
    return waveforms, rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_beats_backbone(checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    beats_module = importlib.import_module("BEATs")
    config = beats_module.BEATsConfig(checkpoint["cfg"])
    model = beats_module.BEATs(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    # Disable the AudioSet clip-level predictor to expose patch tokens.
    model.predictor = None
    model.eval().to(device)
    model.requires_grad_(False)
    return model


@torch.no_grad()
def beats_time_features(model: nn.Module, waveforms: torch.Tensor) -> torch.Tensor:
    features, _padding_mask = model.extract_features(waveforms)
    if features.ndim != 3:
        raise RuntimeError(f"unexpected BEATs feature shape: {tuple(features.shape)}")
    # BEATs flattens Conv2d patches over time x mel-frequency.  With 128 mel bins
    # and patch size 16, there are 8 frequency patches per time step.
    freq_patches = max(1, 128 // int(model.input_patch_size))
    if features.size(1) % freq_patches == 0:
        batch, tokens, dim = features.shape
        features = features.reshape(batch, tokens // freq_patches, freq_patches, dim).mean(dim=2)
    return features


class FrameHead(nn.Module):
    def __init__(self, input_dim: int, num_labels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(input_dim // 2, num_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def build_targets(rows: Sequence[SceneItem], time_steps: int, num_labels: int, device: torch.device) -> torch.Tensor:
    targets = torch.zeros((len(rows), time_steps, num_labels), dtype=torch.float32, device=device)
    for batch_index, row in enumerate(rows):
        duration = max(float(row.duration_seconds), 1e-6)
        for event in row.events:
            label_id = int(event["label_id"])
            onset = max(0.0, float(event["onset_seconds"]))
            offset = min(duration, float(event["offset_seconds"]))
            if offset <= onset or label_id < 0 or label_id >= num_labels:
                continue
            start = max(0, min(time_steps - 1, int(math.floor(onset / duration * time_steps))))
            end = max(start + 1, min(time_steps, int(math.ceil(offset / duration * time_steps))))
            targets[batch_index, start:end, label_id] = 1.0
    return targets


def compute_pos_weight(rows: Sequence[SceneItem], num_labels: int, time_steps: int = 64) -> torch.Tensor:
    pos = torch.zeros(num_labels, dtype=torch.float64)
    total = float(len(rows) * time_steps)
    for row in rows:
        target = build_targets([row], time_steps, num_labels, torch.device("cpu"))[0]
        pos += target.sum(dim=0).double()
    neg = torch.full_like(pos, total) - pos
    weight = neg / pos.clamp_min(1.0)
    return weight.clamp(min=1.0, max=40.0).float()


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return float(inter / union) if union > 0 else 0.0


def probs_to_events(
    probs: torch.Tensor,
    row: SceneItem,
    labels: Sequence[str],
    *,
    threshold: float,
    min_duration: float = 0.08,
    merge_gap: float = 0.12,
) -> list[dict[str, Any]]:
    time_steps, num_labels = probs.shape
    duration = max(float(row.duration_seconds), 1e-6)
    frame_seconds = duration / max(time_steps, 1)
    events: list[dict[str, Any]] = []
    active = probs >= threshold
    for label_id in range(num_labels):
        mask = active[:, label_id].cpu().numpy().astype(bool)
        segments: list[tuple[int, int]] = []
        start: int | None = None
        for index, flag in enumerate(mask.tolist() + [False]):
            if flag and start is None:
                start = index
            elif not flag and start is not None:
                segments.append((start, index))
                start = None
        merged: list[tuple[float, float, float]] = []
        for start_idx, end_idx in segments:
            onset = start_idx * frame_seconds
            offset = end_idx * frame_seconds
            conf = float(probs[start_idx:end_idx, label_id].max().item())
            if merged and onset <= merged[-1][1] + merge_gap:
                prev_on, prev_off, prev_conf = merged[-1]
                merged[-1] = (prev_on, max(prev_off, offset), max(prev_conf, conf))
            else:
                merged.append((onset, offset, conf))
        for onset, offset, conf in merged:
            if offset - onset < min_duration:
                continue
            events.append(
                {
                    "label": labels[label_id],
                    "label_id": label_id,
                    "onset_seconds": round(onset, 4),
                    "offset_seconds": round(offset, 4),
                    "confidence": conf,
                }
            )
    events.sort(key=lambda item: (item["onset_seconds"], item["offset_seconds"], item["label"]))
    return events


def match_event_f1(
    predicted: Sequence[dict[str, Any]],
    gold: Sequence[dict[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[int, int, int]:
    used_gold: set[int] = set()
    tp = 0
    pred_sorted = sorted(predicted, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
    for pred in pred_sorted:
        best_index = None
        best_iou = 0.0
        for index, gold_event in enumerate(gold):
            if index in used_gold or pred["label"] != gold_event["label"]:
                continue
            score = interval_iou(
                (float(pred["onset_seconds"]), float(pred["offset_seconds"])),
                (float(gold_event["onset_seconds"]), float(gold_event["offset_seconds"])),
            )
            if score > best_iou:
                best_iou = score
                best_index = index
        if best_index is not None and best_iou >= iou_threshold:
            used_gold.add(best_index)
            tp += 1
    fp = len(predicted) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision_↑": precision, "recall_↑": recall, "f1_↑": f1}


@torch.no_grad()
def run_model(
    beats: nn.Module,
    head: nn.Module,
    loader: DataLoader,
    labels: Sequence[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    head.eval()
    rows_out: list[dict[str, Any]] = []
    for waveforms, rows in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        features = beats_time_features(beats, waveforms)
        logits = head(features)
        probs = torch.sigmoid(logits).cpu()
        targets = build_targets(rows, probs.size(1), len(labels), torch.device("cpu"))
        for index, row in enumerate(rows):
            rows_out.append(
                {
                    "scene_id": row.scene_id,
                    "split": row.split,
                    "duration_seconds": row.duration_seconds,
                    "probs": probs[index],
                    "targets": targets[index],
                    "gold_events": list(row.events),
                }
            )
    return rows_out


def summarize_detector(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    threshold: float,
    event_iou_threshold: float,
) -> dict[str, Any]:
    frame_tp = frame_fp = frame_fn = 0
    scene_tp = scene_fp = scene_fn = 0
    event_tp = event_fp = event_fn = 0
    kept_labels = []
    false_labels = []
    label_recalls = []
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

        pred_events = probs_to_events(probs, SceneItem(
            scene_id=str(row["scene_id"]),
            split=str(row["split"]),
            mixture_path="",
            duration_seconds=float(row["duration_seconds"]),
            sample_rate=16_000,
            events=tuple(row["gold_events"]),
        ), labels, threshold=threshold)
        tp, fp, fn = match_event_f1(pred_events, row["gold_events"], iou_threshold=event_iou_threshold)
        event_tp += tp
        event_fp += fp
        event_fn += fn

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
        },
    }


def load_qa_rows(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def occurrence(events: Sequence[dict[str, Any]], label: str, ordinal: int) -> dict[str, Any] | None:
    candidates = [event for event in events if event["label"] == label]
    candidates.sort(key=lambda item: (float(item["onset_seconds"]), float(item["offset_seconds"])))
    if ordinal <= 0 or len(candidates) < ordinal:
        return None
    return candidates[ordinal - 1]


def execute_qa(row: Mapping[str, Any], events: Sequence[dict[str, Any]]) -> tuple[str, bool]:
    relation = str(row.get("relation") or "")
    query_label = str(row.get("query_label") or "")
    ordinal = int(row.get("query_instance_ordinal") or 1)
    if relation in {"after", "before"}:
        anchor = occurrence(events, query_label, ordinal)
        if anchor is None:
            return "no_evidence", True
        anchor_onset = float(anchor["onset_seconds"])
        if relation == "after":
            candidates = [
                event for event in events
                if float(event["onset_seconds"]) > anchor_onset + 1e-6
            ]
            if not candidates:
                return "no_evidence", True
            answer = min(candidates, key=lambda item: (float(item["onset_seconds"]), -float(item.get("confidence", 0.0))))
        else:
            candidates = [
                event for event in events
                if float(event["onset_seconds"]) < anchor_onset - 1e-6
            ]
            if not candidates:
                return "no_evidence", True
            answer = max(candidates, key=lambda item: (float(item["onset_seconds"]), float(item.get("confidence", 0.0))))
        return str(answer["label"]), False
    if relation == "first":
        candidate_labels = [str(label) for label in row.get("query_candidate_labels") or []]
        candidates = [event for event in events if event["label"] in set(candidate_labels)]
        if not candidates:
            return "no_evidence", True
        answer = min(candidates, key=lambda item: (float(item["onset_seconds"]), -float(item.get("confidence", 0.0))))
        return str(answer["label"]), False
    return "unsupported", True


def summarize_qa(
    qa_rows: Sequence[Mapping[str, Any]],
    pred_events_by_scene: Mapping[str, Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    rows = []
    for row in qa_rows:
        scene_id = str(row.get("scene_id") or "")
        pred_answer, pred_noev = execute_qa(row, pred_events_by_scene.get(scene_id, ()))
        ok = pred_answer == str(row.get("answer")) and pred_noev == bool(row.get("no_evidence"))
        rows.append(
            {
                "scene_id": scene_id,
                "question_type": row.get("question_type"),
                "relation": row.get("relation"),
                "answer": row.get("answer"),
                "no_evidence": row.get("no_evidence"),
                "pred_answer": pred_answer,
                "pred_no_evidence": pred_noev,
                "ok": ok,
            }
        )
    by_relation = {}
    for relation in sorted({str(row["relation"]) for row in rows}):
        subset = [row for row in rows if str(row["relation"]) == relation]
        by_relation[relation] = {
            "items": len(subset),
            "accuracy_↑": sum(bool(row["ok"]) for row in subset) / max(len(subset), 1),
        }
    return {
        "items": len(rows),
        "accuracy_↑": sum(bool(row["ok"]) for row in rows) / max(len(rows), 1),
        "by_relation": by_relation,
        "rows": rows,
    }


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    labels = load_ontology(args.ontology.resolve())
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_rows = load_scene_manifest(args.train_manifest.resolve(), label_to_id, args.max_train_scenes)
    val_rows = load_scene_manifest(args.val_manifest.resolve(), label_to_id, args.max_val_scenes)
    if not train_rows or not val_rows:
        raise SystemExit("empty train/val detector manifest")

    print(
        f"device={device} train_scenes={len(train_rows)} val_scenes={len(val_rows)} labels={len(labels)}",
        flush=True,
    )
    beats = load_beats_backbone(args.beats_checkpoint.resolve(), device)
    train_loader = DataLoader(
        SceneDataset(train_rows, project_root=args.audio_root.resolve()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    val_loader = DataLoader(
        SceneDataset(val_rows, project_root=args.audio_root.resolve()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    # Probe feature dim.
    waveforms, _rows = next(iter(train_loader))
    with torch.no_grad():
        features = beats_time_features(beats, waveforms.to(device))
    head = FrameHead(features.size(-1), len(labels)).to(device)
    pos_weight = compute_pos_weight(train_rows, len(labels)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for step, (waveforms, rows) in enumerate(train_loader, start=1):
            waveforms = waveforms.to(device, non_blocking=True)
            with torch.no_grad():
                features = beats_time_features(beats, waveforms)
            targets = build_targets(rows, features.size(1), len(labels), device)
            logits = head(features)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step % 50 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={np.mean(losses[-50:]):.4f}", flush=True)

        val_predictions = run_model(beats, head, val_loader, labels, device)
        threshold_summaries = [
            summarize_detector(
                val_predictions,
                labels,
                threshold=threshold,
                event_iou_threshold=args.event_iou_threshold,
            )
            for threshold in args.thresholds
        ]
        # Prefer scene-label F1, then event F1. This is the right order for the
        # detector-first inventory problem.
        selected = max(
            threshold_summaries,
            key=lambda item: (
                item["scene_label"]["f1_↑"],
                item["event_iou"]["f1_↑"],
                -item["scene_label"]["avg_false_positive_labels_↓"],
            ),
        )
        score = float(selected["scene_label"]["f1_↑"])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "selected_threshold": selected["threshold"],
            "selected_scene_label_f1_↑": selected["scene_label"]["f1_↑"],
            "selected_event_f1_↑": selected["event_iou"]["f1_↑"],
            "threshold_summaries": threshold_summaries,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False, sort_keys=True), flush=True)

    if best_state is not None:
        head.load_state_dict(best_state)
    val_predictions = run_model(beats, head, val_loader, labels, device)
    threshold_summaries = [
        summarize_detector(
            val_predictions,
            labels,
            threshold=threshold,
            event_iou_threshold=args.event_iou_threshold,
        )
        for threshold in args.thresholds
    ]
    best_threshold_summary = max(
        threshold_summaries,
        key=lambda item: (
            item["scene_label"]["f1_↑"],
            item["event_iou"]["f1_↑"],
            -item["scene_label"]["avg_false_positive_labels_↓"],
        ),
    )
    best_threshold = float(best_threshold_summary["threshold"])
    pred_events_by_scene: dict[str, list[dict[str, Any]]] = {}
    prediction_rows = []
    for row in val_predictions:
        scene = SceneItem(
            scene_id=str(row["scene_id"]),
            split=str(row["split"]),
            mixture_path="",
            duration_seconds=float(row["duration_seconds"]),
            sample_rate=16_000,
            events=tuple(row["gold_events"]),
        )
        events = probs_to_events(row["probs"], scene, labels, threshold=best_threshold)
        pred_events_by_scene[scene.scene_id] = events
        prediction_rows.append(
            {
                "scene_id": scene.scene_id,
                "split": scene.split,
                "threshold": best_threshold,
                "predicted_events": events,
                "gold_events": row["gold_events"],
            }
        )

    qa_summary: dict[str, Any] | None = None
    if args.qa_val_manifest and args.qa_val_manifest.exists():
        qa_rows = [
            row for row in load_qa_rows(args.qa_val_manifest.resolve())
            if str(row.get("scene_id")) in pred_events_by_scene
        ]
        qa_summary = summarize_qa(qa_rows, pred_events_by_scene)

    report = {
        "format": "qces_beats_frame_detector_tmp_v1",
        "note": "Temporary frozen-BEATs frame-head detector; not BEATs-Strong/PretrainedSED.",
        "train_manifest": str(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "ontology": str(args.ontology.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "beats_checkpoint": str(args.beats_checkpoint.resolve()),
        "device": str(device),
        "train_scenes": len(train_rows),
        "val_scenes": len(val_rows),
        "labels": labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "history": history,
        "best_threshold": best_threshold,
        "val_detector": {
            "threshold_summaries": threshold_summaries,
            "selected": best_threshold_summary,
        },
        "val_qa": {key: value for key, value in (qa_summary or {}).items() if key != "rows"} if qa_summary else None,
    }
    save_json(output_dir / "training_report.json", report)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "labels": labels,
            "best_threshold": best_threshold,
            "report": report,
        },
        output_dir / "beats_frame_head.pt",
    )
    with (output_dir / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in prediction_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if qa_summary is not None:
        with (output_dir / "val_qa_predictions.jsonl").open("w", encoding="utf-8") as handle:
            for row in qa_summary["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    md = [
        "# Temporary QCES BEATs frame detector",
        "",
        "This is a detector-first smoke baseline using frozen BEATs AS2M tokens plus a trainable frame head.",
        "",
        f"- train scenes: {len(train_rows)}",
        f"- val scenes: {len(val_rows)}",
        f"- labels: {len(labels)}",
        f"- best threshold: {best_threshold:.2f}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| scene-label F1 ↑ | {best_threshold_summary['scene_label']['f1_↑']:.3f} |",
        f"| scene-label precision ↑ | {best_threshold_summary['scene_label']['precision_↑']:.3f} |",
        f"| scene-label recall ↑ | {best_threshold_summary['scene_label']['recall_↑']:.3f} |",
        f"| avg labels kept ↓ | {best_threshold_summary['scene_label']['avg_labels_kept_↓']:.2f} |",
        f"| avg false-positive labels ↓ | {best_threshold_summary['scene_label']['avg_false_positive_labels_↓']:.2f} |",
        f"| event F1 @ IoU {args.event_iou_threshold:.2f} ↑ | {best_threshold_summary['event_iou']['f1_↑']:.3f} |",
        f"| frame F1 ↑ | {best_threshold_summary['frame']['f1_↑']:.3f} |",
    ]
    if qa_summary is not None:
        md.extend(
            [
                "",
                "## Downstream QA with predicted inventory",
                "",
                "| relation | items | accuracy ↑ |",
                "|---|---:|---:|",
            ]
        )
        for relation, row in qa_summary["by_relation"].items():
            md.append(f"| {relation} | {row['items']} | {row['accuracy_↑']:.3f} |")
        md.append(f"| overall | {qa_summary['items']} | {qa_summary['accuracy_↑']:.3f} |")
    (output_dir / "training_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
