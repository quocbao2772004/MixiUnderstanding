#!/usr/bin/env python3
"""Train a QCES detector with explicit onset/offset heads.

This sidecar starts from the current PretrainedSED BEATs-Strong 30-class frame
detector checkpoint and adds onset/offset heads. The goal is to improve event
instance decoding for ordinal and before/after QA, not to change the original
Claude pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from score_qces_option_aware_executor import option_aware_execute, summarize as summarize_option_rows
from sweep_qces_event_decoder import (
    contiguous_true_segments,
    local_peak_indices,
    summarize_detector_from_decoded,
)
from train_qces_pretrainedsed_detector import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_DATASET_ROOT,
    SceneDataset,
    SceneItem,
    build_targets,
    collate,
    compute_pos_weight,
    interpolate_sequence,
    load_model,
    load_ontology,
    load_scene_manifest,
    make_device,
    read_jsonl,
    save_json,
    summarize_qa,
)


DEFAULT_INIT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_30class/full801_val102_e12/pretrainedsed_beats_qces_detector.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_pretrainedsed_onset_detector_30class/onset_head_v1"


@dataclass(frozen=True)
class OnsetDecoderConfig:
    scene_threshold: float
    frame_low_threshold: float
    onset_threshold: float
    offset_threshold: float
    min_duration: float
    min_onset_gap: float
    fallback_to_frame: bool


class FrameOnsetOffsetDetector(nn.Module):
    def __init__(self, base: nn.Module, num_labels: int) -> None:
        super().__init__()
        self.base = base
        hidden_dim = int(base.strong_head.in_features)
        self.onset_head = nn.Linear(hidden_dim, num_labels)
        self.offset_head = nn.Linear(hidden_dim, num_labels)

    @property
    def seq_len(self) -> int:
        return int(self.base.seq_len)

    def extract_features(self, waveforms: torch.Tensor) -> torch.Tensor:
        backbone_trainable = any(parameter.requires_grad for parameter in self.base.model.parameters())
        context = contextlib.nullcontext() if backbone_trainable else torch.no_grad()
        with context:
            mel = self.base.mel_forward(waveforms)
            features = self.base.model(mel)
            features = interpolate_sequence(features, self.base.seq_len)
            features = self.base.seq_model(features)
        return features

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(waveforms)
        return self.base.strong_head(features), self.onset_head(features), self.offset_head(features)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_AUDIT_DIR / "detector_scene_manifest_train.jsonl")
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_AUDIT_DIR / "detector_scene_manifest_val.jsonl")
    parser.add_argument("--qa-val-manifest", type=Path, default=DEFAULT_DATASET_ROOT / "qces_val.jsonl")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_AUDIT_DIR / "ontology_ready_train.txt")
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-val-scenes", type=int, default=0)
    parser.add_argument("--frame-lr", type=float, default=2e-4)
    parser.add_argument("--onset-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--frame-loss-weight", type=float, default=0.5)
    parser.add_argument("--onset-loss-weight", type=float, default=1.0)
    parser.add_argument("--offset-loss-weight", type=float, default=0.5)
    parser.add_argument("--onset-dilation-frames", type=int, default=1)
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument(
        "--decoder-grid",
        choices=["tiny", "quick", "full"],
        default="tiny",
        help="Tiny/quick grids focus around the best frame-hysteresis operating region.",
    )
    parser.add_argument("--seed", type=int, default=2041)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_boundary_targets(
    rows: Sequence[SceneItem],
    time_steps: int,
    num_labels: int,
    device: torch.device,
    *,
    dilation_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    onset_targets = torch.zeros((len(rows), time_steps, num_labels), dtype=torch.float32, device=device)
    offset_targets = torch.zeros_like(onset_targets)
    dilation = max(0, int(dilation_frames))
    for batch_index, row in enumerate(rows):
        duration = max(float(row.duration_seconds), 1e-6)
        for event in row.events:
            label_id = int(event["label_id"])
            if label_id < 0 or label_id >= num_labels:
                continue
            onset = max(0.0, min(duration, float(event["onset_seconds"])))
            offset = max(0.0, min(duration, float(event["offset_seconds"])))
            onset_index = max(0, min(time_steps - 1, int(round(onset / duration * time_steps))))
            offset_index = max(0, min(time_steps - 1, int(round(offset / duration * time_steps))))
            onset_start = max(0, onset_index - dilation)
            onset_end = min(time_steps, onset_index + dilation + 1)
            offset_start = max(0, offset_index - dilation)
            offset_end = min(time_steps, offset_index + dilation + 1)
            onset_targets[batch_index, onset_start:onset_end, label_id] = 1.0
            offset_targets[batch_index, offset_start:offset_end, label_id] = 1.0
    return onset_targets, offset_targets


def compute_boundary_pos_weight(
    rows: Sequence[SceneItem],
    num_labels: int,
    time_steps: int,
    *,
    dilation_frames: int,
) -> torch.Tensor:
    pos = torch.zeros(num_labels, dtype=torch.float64)
    total = float(len(rows) * time_steps)
    for row in rows:
        onset, _ = build_boundary_targets([row], time_steps, num_labels, torch.device("cpu"), dilation_frames=dilation_frames)
        pos += onset[0].sum(dim=0).double()
    neg = torch.full_like(pos, total) - pos
    return (neg / pos.clamp_min(1.0)).clamp(min=1.0, max=200.0).float()


def load_initialized_model(labels: Sequence[str], init_checkpoint: Path, device: torch.device) -> FrameOnsetOffsetDetector:
    checkpoint = torch.load(init_checkpoint.resolve(), map_location="cpu", weights_only=False)
    checkpoint_labels = list(checkpoint.get("labels") or [])
    if checkpoint_labels and list(labels) != checkpoint_labels:
        raise ValueError("init checkpoint labels do not match ontology labels")
    report = checkpoint.get("report") or {}
    model_checkpoint_name = str(report.get("checkpoint") or "BEATs_strong_1")
    base = load_model(len(labels), model_checkpoint_name, device)
    base.load_state_dict(checkpoint["model_state_dict"], strict=True)
    base.model.eval()
    base.model.requires_grad_(False)
    return FrameOnsetOffsetDetector(base, len(labels)).to(device)


@torch.no_grad()
def run_model(
    model: FrameOnsetOffsetDetector,
    loader: DataLoader,
    labels: Sequence[str],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    out: list[dict[str, Any]] = []
    for waveforms, rows in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        frame_logits, onset_logits, offset_logits = model(waveforms)
        frame_probs = torch.sigmoid(frame_logits).cpu()
        onset_probs = torch.sigmoid(onset_logits).cpu()
        offset_probs = torch.sigmoid(offset_logits).cpu()
        targets = build_targets(rows, frame_probs.size(1), len(labels), torch.device("cpu"))
        for index, row in enumerate(rows):
            out.append(
                {
                    "scene_id": row.scene_id,
                    "split": row.split,
                    "duration_seconds": row.duration_seconds,
                    "frame_probs": frame_probs[index],
                    "onset_probs": onset_probs[index],
                    "offset_probs": offset_probs[index],
                    "targets": targets[index],
                    "gold_events": list(row.events),
                }
            )
    return out


def frame_segments(values: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    return contiguous_true_segments((values >= threshold).tolist())


def decode_onset_events(
    row: Mapping[str, Any],
    labels: Sequence[str],
    config: OnsetDecoderConfig,
) -> list[dict[str, Any]]:
    frame_probs: torch.Tensor = row["frame_probs"]
    onset_probs: torch.Tensor = row["onset_probs"]
    offset_probs: torch.Tensor = row["offset_probs"]
    time_steps, num_labels = frame_probs.shape
    duration = max(float(row["duration_seconds"]), 1e-6)
    frame_seconds = duration / max(time_steps, 1)
    min_gap_frames = max(1, int(round(config.min_onset_gap / frame_seconds)))
    events: list[dict[str, Any]] = []

    for label_id in range(num_labels):
        frame_values = frame_probs[:, label_id].detach().cpu().numpy()
        onset_values = onset_probs[:, label_id].detach().cpu().numpy()
        offset_values = offset_probs[:, label_id].detach().cpu().numpy()
        if float(frame_values.max()) < config.scene_threshold and float(onset_values.max()) < config.onset_threshold:
            continue

        peaks = local_peak_indices(onset_values, threshold=config.onset_threshold, min_gap_frames=min_gap_frames)
        peaks = [peak for peak in peaks if frame_values[max(0, peak - 1): min(time_steps, peak + 2)].max() >= config.frame_low_threshold]

        if not peaks and config.fallback_to_frame:
            for start, end in frame_segments(frame_values, config.scene_threshold):
                if (end - start) * frame_seconds >= config.min_duration:
                    peak = int(np.argmax(frame_values[start:end]) + start)
                    peaks.append(peak)
        peaks = sorted(set(peaks))
        if not peaks:
            continue

        for peak_index, peak in enumerate(peaks):
            left_limit = 0
            right_limit = time_steps
            if peak_index > 0:
                left_limit = (peaks[peak_index - 1] + peak) // 2
            if peak_index + 1 < len(peaks):
                right_limit = max(left_limit + 1, (peak + peaks[peak_index + 1]) // 2)

            left = peak
            while left > left_limit and frame_values[left - 1] >= config.frame_low_threshold:
                left -= 1

            offset_candidates = [
                index
                for index in local_peak_indices(offset_values[peak:right_limit], threshold=config.offset_threshold, min_gap_frames=1)
            ]
            if offset_candidates:
                right = peak + offset_candidates[0] + 1
            else:
                right = peak + 1
                while right < right_limit and frame_values[right] >= config.frame_low_threshold:
                    right += 1

            onset = left * frame_seconds
            offset = right * frame_seconds
            if offset - onset < config.min_duration:
                continue
            confidence = float(max(frame_values[left:right].max(), onset_values[peak]))
            events.append(
                {
                    "label": labels[label_id],
                    "label_id": label_id,
                    "onset_seconds": round(onset, 4),
                    "offset_seconds": round(offset, 4),
                    "confidence": confidence,
                    "onset_confidence": float(onset_values[peak]),
                }
            )
    events.sort(key=lambda item: (float(item["onset_seconds"]), float(item["offset_seconds"]), str(item["label"])))
    return events


def materialize_decoded_rows(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    config: OnsetDecoderConfig,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        events = decode_onset_events(row, labels, config)
        scene_id = str(row["scene_id"])
        by_scene[scene_id] = events
        rows.append(
            {
                "scene_id": scene_id,
                "split": row.get("split"),
                "duration_seconds": float(row["duration_seconds"]),
                "decoder_config": asdict(config),
                "predicted_events": events,
                "gold_events": list(row["gold_events"]),
                "targets": row["targets"],
            }
        )
    return rows, by_scene


def summarize_option_qa(
    qa_rows: Sequence[Mapping[str, Any]],
    pred_events_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for row in qa_rows:
        scene_id = str(row.get("scene_id") or "")
        if scene_id not in pred_events_by_scene:
            continue
        pred_answer, pred_noev = option_aware_execute(row, pred_events_by_scene[scene_id])
        ok = pred_answer == str(row.get("answer")) and pred_noev == bool(row.get("no_evidence"))
        rows.append(
            {
                "scene_id": scene_id,
                "question": row.get("question"),
                "relation": row.get("relation"),
                "answer_options": row.get("answer_options"),
                "answer": row.get("answer"),
                "no_evidence": row.get("no_evidence"),
                "no_evidence_reason": row.get("no_evidence_reason"),
                "pred_answer": pred_answer,
                "pred_no_evidence": pred_noev,
                "ok": ok,
            }
        )
    return summarize_option_rows(rows), rows


def score_config(
    predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    qa_rows: Sequence[Mapping[str, Any]],
    config: OnsetDecoderConfig,
    *,
    event_iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    decoded_rows, by_scene = materialize_decoded_rows(predictions, labels, config)
    detector = summarize_detector_from_decoded(decoded_rows, labels, event_iou_threshold=event_iou_threshold)
    open_qa = summarize_qa(qa_rows, by_scene)
    option_summary, option_rows = summarize_option_qa(qa_rows, by_scene)
    mixed_score = (
        0.35 * float(option_summary["answerable_accuracy_↑"])
        + 0.25 * float(option_summary["no_evidence_accuracy_↑"])
        + 0.25 * float(detector["event_iou"]["f1_↑"])
        + 0.15 * float(detector["frame"]["f1_↑"])
    )
    return (
        {
            "config": asdict(config),
            "mixed_score_↑": mixed_score,
            "detector": detector,
            "open_qa": {key: value for key, value in open_qa.items() if key != "rows"},
            "option_qa": option_summary,
        },
        decoded_rows,
        open_qa["rows"],
        option_rows,
    )


def decoder_configs(grid: str) -> list[OnsetDecoderConfig]:
    configs: list[OnsetDecoderConfig] = []
    if grid == "tiny":
        scene_thresholds = [0.90, 0.925]
        low_ratios = [0.85]
        onset_thresholds = [0.60, 0.70]
        offset_thresholds = [0.65]
        min_durations = [0.04, 0.08, 0.12]
    elif grid == "quick":
        scene_thresholds = [0.875, 0.90, 0.925]
        low_ratios = [0.70, 0.85]
        onset_thresholds = [0.45, 0.60, 0.70]
        offset_thresholds = [0.50, 0.65]
        min_durations = [0.04, 0.08, 0.12]
    else:
        scene_thresholds = [0.70, 0.80, 0.90, 0.925]
        low_ratios = [0.55, 0.70, 0.85]
        onset_thresholds = [0.25, 0.40, 0.55, 0.70]
        offset_thresholds = [0.35, 0.50, 0.65]
        min_durations = [0.04, 0.08, 0.12]
    for scene_threshold in scene_thresholds:
        for low_ratio in low_ratios:
            low = max(0.05, scene_threshold * low_ratio)
            for onset_threshold in onset_thresholds:
                for offset_threshold in offset_thresholds:
                    for min_duration in min_durations:
                        configs.append(
                            OnsetDecoderConfig(
                                scene_threshold=scene_threshold,
                                frame_low_threshold=low,
                                onset_threshold=onset_threshold,
                                offset_threshold=offset_threshold,
                                min_duration=min_duration,
                                min_onset_gap=0.20,
                                fallback_to_frame=True,
                            )
                        )
    return configs


def compact_decoded_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": row["scene_id"],
            "split": row.get("split"),
            "duration_seconds": row.get("duration_seconds"),
            "decoder_config": row.get("decoder_config"),
            "predicted_events": row.get("predicted_events"),
            "gold_events": row.get("gold_events"),
        }
        for row in rows
    ]


def choose_score(summary: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(summary["mixed_score_↑"]),
        float(summary["detector"]["event_iou"]["f1_↑"]),
        float(summary["option_qa"]["accuracy_↑"]),
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
    qa_rows = [row for row in read_jsonl(args.qa_val_manifest.resolve())]
    qa_scene_ids = {row.scene_id for row in val_rows}
    qa_rows = [row for row in qa_rows if str(row.get("scene_id") or "") in qa_scene_ids]

    print(
        f"device={device} train_scenes={len(train_rows)} val_scenes={len(val_rows)} labels={len(labels)}",
        flush=True,
    )
    model = load_initialized_model(labels, args.init_checkpoint, device)

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

    frame_pos_weight = compute_pos_weight(train_rows, len(labels), time_steps=model.seq_len).to(device)
    boundary_pos_weight = compute_boundary_pos_weight(
        train_rows,
        len(labels),
        model.seq_len,
        dilation_frames=args.onset_dilation_frames,
    ).to(device)
    frame_loss_fn = nn.BCEWithLogitsLoss(pos_weight=frame_pos_weight)
    boundary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=boundary_pos_weight)

    frame_params = list(model.base.strong_head.parameters())
    boundary_params = [*model.onset_head.parameters(), *model.offset_head.parameters()]
    optimizer = torch.optim.AdamW(
        [
            {"params": frame_params, "lr": args.frame_lr},
            {"params": boundary_params, "lr": args.onset_lr},
        ],
        weight_decay=args.weight_decay,
    )

    best_score = (-1.0, -1.0, -1.0)
    best_state: dict[str, torch.Tensor] | None = None
    best_summary: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    configs = decoder_configs(args.decoder_grid)
    print(f"decoder_grid={args.decoder_grid} decoder_configs={len(configs)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.base.model.eval()
        losses: list[float] = []
        frame_losses: list[float] = []
        onset_losses: list[float] = []
        offset_losses: list[float] = []
        for step, (waveforms, rows) in enumerate(train_loader, start=1):
            waveforms = waveforms.to(device, non_blocking=True)
            frame_logits, onset_logits, offset_logits = model(waveforms)
            frame_targets = build_targets(rows, frame_logits.size(1), len(labels), device)
            onset_targets, offset_targets = build_boundary_targets(
                rows,
                frame_logits.size(1),
                len(labels),
                device,
                dilation_frames=args.onset_dilation_frames,
            )
            frame_loss = frame_loss_fn(frame_logits, frame_targets)
            onset_loss = boundary_loss_fn(onset_logits, onset_targets)
            offset_loss = boundary_loss_fn(offset_logits, offset_targets)
            loss = (
                args.frame_loss_weight * frame_loss
                + args.onset_loss_weight * onset_loss
                + args.offset_loss_weight * offset_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*frame_params, *boundary_params], 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            frame_losses.append(float(frame_loss.detach().cpu()))
            onset_losses.append(float(onset_loss.detach().cpu()))
            offset_losses.append(float(offset_loss.detach().cpu()))
            if step % 50 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={np.mean(losses[-50:]):.4f}", flush=True)

        val_predictions = run_model(model, val_loader, labels, device)
        summaries: list[dict[str, Any]] = []
        for config in configs:
            summary, _, _, _ = score_config(
                val_predictions,
                labels,
                qa_rows,
                config,
                event_iou_threshold=args.event_iou_threshold,
            )
            summaries.append(summary)
        selected = max(summaries, key=choose_score)
        selected_score = choose_score(selected)
        if selected_score > best_score:
            best_score = selected_score
            best_summary = selected
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        epoch_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "frame_loss": float(np.mean(frame_losses)),
            "onset_loss": float(np.mean(onset_losses)),
            "offset_loss": float(np.mean(offset_losses)),
            "selected": {
                "config": selected["config"],
                "mixed_score_↑": selected["mixed_score_↑"],
                "event_f1_↑": selected["detector"]["event_iou"]["f1_↑"],
                "frame_f1_↑": selected["detector"]["frame"]["f1_↑"],
                "option_qa_↑": selected["option_qa"]["accuracy_↑"],
                "option_answerable_↑": selected["option_qa"]["answerable_accuracy_↑"],
                "option_no_evidence_↑": selected["option_qa"]["no_evidence_accuracy_↑"],
                "open_qa_↑": selected["open_qa"]["accuracy_↑"],
            },
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, ensure_ascii=False, sort_keys=True), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    val_predictions = run_model(model, val_loader, labels, device)
    assert best_summary is not None
    best_config = OnsetDecoderConfig(**best_summary["config"])
    final_summary, decoded_rows, open_rows, option_rows = score_config(
        val_predictions,
        labels,
        qa_rows,
        best_config,
        event_iou_threshold=args.event_iou_threshold,
    )
    report = {
        "format": "qces_pretrainedsed_onset_detector_v1",
        "note": "Frozen BEATs backbone, initialized from frame detector, with trainable frame/onset/offset heads.",
        "init_checkpoint": str(args.init_checkpoint.resolve()),
        "train_manifest": str(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "qa_val_manifest": str(args.qa_val_manifest.resolve()),
        "ontology": str(args.ontology.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "device": str(device),
        "train_scenes": len(train_rows),
        "val_scenes": len(val_rows),
        "labels": labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "frame_lr": args.frame_lr,
        "onset_lr": args.onset_lr,
        "loss_weights": {
            "frame": args.frame_loss_weight,
            "onset": args.onset_loss_weight,
            "offset": args.offset_loss_weight,
        },
        "decoder_grid": args.decoder_grid,
        "decoder_configs": len(configs),
        "history": history,
        "best_config": asdict(best_config),
        "val_summary": final_summary,
    }
    save_json(output_dir / "training_report.json", report)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": labels,
            "best_config": asdict(best_config),
            "report": report,
        },
        output_dir / "pretrainedsed_beats_qces_onset_detector.pt",
    )
    write_jsonl(output_dir / "val_predictions.jsonl", compact_decoded_rows(decoded_rows))
    write_jsonl(output_dir / "val_qa_predictions.jsonl", open_rows)
    write_jsonl(output_dir / "val_option_aware_predictions.jsonl", option_rows)

    md = [
        "# QCES PretrainedSED onset detector",
        "",
        "Frozen BEATs backbone initialized from the frame detector, plus onset/offset heads.",
        "",
        f"- train scenes: {len(train_rows)}",
        f"- val scenes: {len(val_rows)}",
        f"- labels: {len(labels)}",
        f"- best config: `{json.dumps(asdict(best_config), sort_keys=True)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| mixed score ↑ | {final_summary['mixed_score_↑']:.3f} |",
        f"| event F1 @ {args.event_iou_threshold:.2f} ↑ | {final_summary['detector']['event_iou']['f1_↑']:.3f} |",
        f"| frame F1 ↑ | {final_summary['detector']['frame']['f1_↑']:.3f} |",
        f"| scene-label F1 ↑ | {final_summary['detector']['scene_label']['f1_↑']:.3f} |",
        f"| open QA accuracy ↑ | {final_summary['open_qa']['accuracy_↑']:.3f} |",
        f"| option QA accuracy ↑ | {final_summary['option_qa']['accuracy_↑']:.3f} |",
        f"| option answerable accuracy ↑ | {final_summary['option_qa']['answerable_accuracy_↑']:.3f} |",
        f"| option no-evidence accuracy ↑ | {final_summary['option_qa']['no_evidence_accuracy_↑']:.3f} |",
    ]
    (output_dir / "training_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
