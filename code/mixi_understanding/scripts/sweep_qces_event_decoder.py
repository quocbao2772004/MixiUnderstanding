#!/usr/bin/env python3
"""Sweep instance-aware event decoders for the QCES PretrainedSED detector.

This is a decoder-only sidecar. It does not retrain the detector and does not
modify the earlier Claude/current pipeline. The purpose is to test whether the
current BEATs-Strong frame probabilities can be converted into better event
instances for temporal QA.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from score_qces_option_aware_executor import option_aware_execute, summarize as summarize_option_rows
from train_qces_pretrainedsed_detector import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_DATASET_ROOT,
    SceneDataset,
    SceneItem,
    collate,
    interval_iou,
    load_model,
    load_ontology,
    load_scene_manifest,
    make_device,
    match_event_f1,
    prf,
    read_jsonl,
    run_model,
    save_json,
    summarize_qa,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_30class/full801_val102_e12/pretrainedsed_beats_qces_detector.pt"
)


@dataclass(frozen=True)
class DecoderConfig:
    decoder: str
    high_threshold: float
    low_threshold: float
    min_duration: float
    merge_gap: float
    min_peak_gap: float
    valley_ratio: float
    smooth_frames: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--detector-manifest", type=Path, default=DEFAULT_AUDIT_DIR / "detector_scene_manifest_val.jsonl")
    parser.add_argument("--qa-manifest", type=Path, default=DEFAULT_DATASET_ROOT / "qces_val.jsonl")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_AUDIT_DIR / "ontology_ready_train.txt")
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument("--decoder-config", type=Path, help="Evaluate one JSON decoder config instead of sweeping.")
    parser.add_argument("--max-configs", type=int, default=0, help="Optional cap for smoke tests.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_decoder_config(path: Path) -> DecoderConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "config" in payload:
        payload = payload["config"]
    return DecoderConfig(**payload)


def smooth_1d(probs: torch.Tensor, frames: int) -> torch.Tensor:
    if frames <= 1:
        return probs
    if frames % 2 == 0:
        frames += 1
    x = probs.transpose(0, 1).unsqueeze(0)
    x = torch.nn.functional.avg_pool1d(x, kernel_size=frames, stride=1, padding=frames // 2)
    return x.squeeze(0).transpose(0, 1)


def contiguous_true_segments(mask: Sequence[bool]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            segments.append((start, index))
            start = None
    return segments


def local_peak_indices(values: np.ndarray, threshold: float, min_gap_frames: int) -> list[int]:
    candidates: list[int] = []
    for index, value in enumerate(values):
        if value < threshold:
            continue
        left = values[index - 1] if index > 0 else -np.inf
        right = values[index + 1] if index + 1 < len(values) else -np.inf
        if value >= left and value >= right:
            candidates.append(index)
    if not candidates:
        return []

    # Non-maximum suppression by peak height.
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: float(values[item]), reverse=True):
        if all(abs(index - kept) >= min_gap_frames for kept in selected):
            selected.append(index)
    return sorted(selected)


def split_high_segment_by_valleys(
    values: np.ndarray,
    start: int,
    end: int,
    *,
    min_peak_gap_frames: int,
    valley_ratio: float,
) -> list[tuple[int, int]]:
    if end <= start + 1:
        return [(start, end)]
    local_values = values[start:end]
    peaks = local_peak_indices(local_values, threshold=float(local_values.max()) * 0.70, min_gap_frames=min_peak_gap_frames)
    peaks = [start + peak for peak in peaks]
    if len(peaks) <= 1:
        return [(start, end)]

    cuts: list[int] = []
    for left_peak, right_peak in zip(peaks, peaks[1:]):
        if right_peak <= left_peak + 1:
            continue
        valley_region = values[left_peak:right_peak + 1]
        valley_local_index = int(np.argmin(valley_region))
        valley_index = left_peak + valley_local_index
        peak_floor = min(float(values[left_peak]), float(values[right_peak]))
        if float(values[valley_index]) <= peak_floor * valley_ratio:
            cuts.append(max(start + 1, min(end - 1, valley_index)))
    if not cuts:
        return [(start, end)]

    out: list[tuple[int, int]] = []
    cursor = start
    for cut in cuts:
        if cut > cursor:
            out.append((cursor, cut))
        cursor = cut
    if cursor < end:
        out.append((cursor, end))
    return out or [(start, end)]


def decode_label_segments(values: np.ndarray, config: DecoderConfig, frame_seconds: float) -> list[tuple[int, int, float]]:
    high_mask = values >= config.high_threshold
    high_segments = contiguous_true_segments(high_mask.tolist())
    if not high_segments:
        return []

    min_peak_gap_frames = max(1, int(round(config.min_peak_gap / frame_seconds)))
    low_mask = values >= config.low_threshold
    expanded: list[tuple[int, int, float]] = []

    if config.decoder == "threshold":
        seed_segments = high_segments
    elif config.decoder == "hysteresis":
        seed_segments = high_segments
    elif config.decoder == "hysteresis_valley":
        seed_segments = []
        for start, end in high_segments:
            seed_segments.extend(
                split_high_segment_by_valleys(
                    values,
                    start,
                    end,
                    min_peak_gap_frames=min_peak_gap_frames,
                    valley_ratio=config.valley_ratio,
                )
            )
    elif config.decoder == "peak":
        peaks = local_peak_indices(values, threshold=config.high_threshold, min_gap_frames=min_peak_gap_frames)
        seed_segments = [(peak, peak + 1) for peak in peaks]
    else:
        raise ValueError(f"unknown decoder: {config.decoder}")

    peak_centers = [int(np.argmax(values[start:end]) + start) for start, end in seed_segments if end > start]
    for seed_index, (start, end) in enumerate(seed_segments):
        if end <= start:
            continue
        peak = int(np.argmax(values[start:end]) + start)
        left_limit = 0
        right_limit = len(values)
        if seed_index > 0 and seed_index - 1 < len(peak_centers):
            left_limit = (peak_centers[seed_index - 1] + peak) // 2
        if seed_index + 1 < len(peak_centers):
            right_limit = max(left_limit + 1, (peak + peak_centers[seed_index + 1]) // 2)

        if config.decoder == "threshold":
            left = start
            right = end
        else:
            left = peak
            while left > left_limit and low_mask[left - 1]:
                left -= 1
            right = peak + 1
            while right < right_limit and low_mask[right]:
                right += 1
        confidence = float(values[left:right].max()) if right > left else float(values[peak])
        expanded.append((left, right, confidence))

    # Optional conservative merge after expansion.
    expanded.sort(key=lambda item: (item[0], item[1]))
    merge_gap_frames = int(round(config.merge_gap / frame_seconds))
    merged: list[tuple[int, int, float]] = []
    for start, end, confidence in expanded:
        if merged and start <= merged[-1][1] + merge_gap_frames:
            prev_start, prev_end, prev_conf = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end), max(prev_conf, confidence))
        else:
            merged.append((start, end, confidence))
    return merged


def decode_events(
    probs: torch.Tensor,
    row: Mapping[str, Any],
    labels: Sequence[str],
    config: DecoderConfig,
) -> list[dict[str, Any]]:
    probs = smooth_1d(probs.float(), config.smooth_frames)
    time_steps, num_labels = probs.shape
    duration = max(float(row["duration_seconds"]), 1e-6)
    frame_seconds = duration / max(time_steps, 1)
    events: list[dict[str, Any]] = []
    for label_id in range(num_labels):
        values = probs[:, label_id].detach().cpu().numpy()
        segments = decode_label_segments(values, config, frame_seconds)
        for start, end, confidence in segments:
            onset = start * frame_seconds
            offset = end * frame_seconds
            if offset - onset < config.min_duration:
                continue
            events.append(
                {
                    "label": labels[label_id],
                    "label_id": label_id,
                    "onset_seconds": round(onset, 4),
                    "offset_seconds": round(offset, 4),
                    "confidence": float(confidence),
                }
            )
    events.sort(key=lambda item: (float(item["onset_seconds"]), float(item["offset_seconds"]), str(item["label"])))
    return events


def events_to_mask(events: Sequence[Mapping[str, Any]], labels: Sequence[str], duration: float, time_steps: int) -> torch.Tensor:
    label_to_id = {label: index for index, label in enumerate(labels)}
    mask = torch.zeros((time_steps, len(labels)), dtype=torch.bool)
    for event in events:
        label = str(event.get("label"))
        if label not in label_to_id:
            continue
        onset = max(0.0, float(event.get("onset_seconds", 0.0)))
        offset = min(duration, float(event.get("offset_seconds", 0.0)))
        if offset <= onset:
            continue
        start = max(0, min(time_steps - 1, int(np.floor(onset / duration * time_steps))))
        end = max(start + 1, min(time_steps, int(np.ceil(offset / duration * time_steps))))
        mask[start:end, label_to_id[label]] = True
    return mask


def summarize_detector_from_decoded(
    decoded_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    event_iou_threshold: float,
) -> dict[str, Any]:
    frame_tp = frame_fp = frame_fn = 0
    scene_tp = scene_fp = scene_fn = 0
    event_tp = event_fp = event_fn = 0
    kept_labels: list[float] = []
    false_labels: list[float] = []
    label_recalls: list[float] = []
    onset_errors: list[float] = []
    offset_errors: list[float] = []

    for row in decoded_rows:
        pred_events = list(row["predicted_events"])
        gold_events = list(row["gold_events"])
        targets: torch.Tensor = row["targets"]
        duration = max(float(row["duration_seconds"]), 1e-6)
        pred_frame = events_to_mask(pred_events, labels, duration, int(targets.size(0)))
        gold_frame = targets.bool()
        frame_tp += int((pred_frame & gold_frame).sum().item())
        frame_fp += int((pred_frame & ~gold_frame).sum().item())
        frame_fn += int((~pred_frame & gold_frame).sum().item())

        pred_labels = {str(event["label"]) for event in pred_events}
        gold_labels = {str(event["label"]) for event in gold_events}
        scene_tp += len(pred_labels & gold_labels)
        scene_fp += len(pred_labels - gold_labels)
        scene_fn += len(gold_labels - pred_labels)
        kept_labels.append(float(len(pred_labels)))
        false_labels.append(float(len(pred_labels - gold_labels)))
        label_recalls.append(len(pred_labels & gold_labels) / max(len(gold_labels), 1))

        tp, fp, fn = match_event_f1(pred_events, gold_events, iou_threshold=event_iou_threshold)
        event_tp += tp
        event_fp += fp
        event_fn += fn

        used_gold: set[int] = set()
        for pred in sorted(pred_events, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
            best_index: int | None = None
            best_iou = 0.0
            for index, gold_event in enumerate(gold_events):
                if index in used_gold or str(pred["label"]) != str(gold_event["label"]):
                    continue
                score = interval_iou(
                    (float(pred["onset_seconds"]), float(pred["offset_seconds"])),
                    (float(gold_event["onset_seconds"]), float(gold_event["offset_seconds"])),
                )
                if score > best_iou:
                    best_index = index
                    best_iou = score
            if best_index is not None and best_iou >= event_iou_threshold:
                gold_event = gold_events[best_index]
                used_gold.add(best_index)
                onset_errors.append(abs(float(pred["onset_seconds"]) - float(gold_event["onset_seconds"])))
                offset_errors.append(abs(float(pred["offset_seconds"]) - float(gold_event["offset_seconds"])))

    return {
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


def materialize_decoded_rows(
    model_predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    config: DecoderConfig,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in model_predictions:
        events = decode_events(row["probs"], row, labels, config)
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


def compact_decoded_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for row in rows:
        compact.append(
            {
                "scene_id": row["scene_id"],
                "split": row.get("split"),
                "duration_seconds": row.get("duration_seconds"),
                "decoder_config": row.get("decoder_config"),
                "predicted_events": row.get("predicted_events"),
                "gold_events": row.get("gold_events"),
            }
        )
    return compact


def score_config(
    model_predictions: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    qa_rows: Sequence[Mapping[str, Any]],
    config: DecoderConfig,
    *,
    event_iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    decoded_rows, by_scene = materialize_decoded_rows(model_predictions, labels, config)
    detector = summarize_detector_from_decoded(decoded_rows, labels, event_iou_threshold=event_iou_threshold)
    open_qa = summarize_qa(qa_rows, by_scene)
    option_summary, option_rows = summarize_option_qa(qa_rows, by_scene)
    mixed_score = (
        0.35 * float(option_summary["answerable_accuracy_↑"])
        + 0.25 * float(option_summary["no_evidence_accuracy_↑"])
        + 0.25 * float(detector["event_iou"]["f1_↑"])
        + 0.15 * float(detector["frame"]["f1_↑"])
    )
    summary = {
        "config": asdict(config),
        "mixed_score_↑": mixed_score,
        "detector": detector,
        "open_qa": {key: value for key, value in open_qa.items() if key != "rows"},
        "option_qa": option_summary,
    }
    return summary, decoded_rows, open_qa["rows"], option_rows


def candidate_configs(max_configs: int = 0) -> list[DecoderConfig]:
    configs: list[DecoderConfig] = []
    high_thresholds = [0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.925, 0.95]
    min_durations = [0.04, 0.08, 0.12]
    merge_gaps = [0.00, 0.04, 0.08, 0.12]
    smooth_frames_values = [1, 3]
    for high in high_thresholds:
        for min_duration in min_durations:
            for merge_gap in merge_gaps:
                configs.append(
                    DecoderConfig(
                        decoder="threshold",
                        high_threshold=high,
                        low_threshold=high,
                        min_duration=min_duration,
                        merge_gap=merge_gap,
                        min_peak_gap=0.32,
                        valley_ratio=0.60,
                        smooth_frames=1,
                    )
                )
        for low_ratio in [0.55, 0.70, 0.85]:
            low = max(0.05, high * low_ratio)
            for min_duration in min_durations:
                for merge_gap in [0.00, 0.04, 0.08]:
                    for smooth_frames in smooth_frames_values:
                        configs.append(
                            DecoderConfig(
                                decoder="hysteresis",
                                high_threshold=high,
                                low_threshold=low,
                                min_duration=min_duration,
                                merge_gap=merge_gap,
                                min_peak_gap=0.32,
                                valley_ratio=0.60,
                                smooth_frames=smooth_frames,
                            )
                        )
        for low_ratio in [0.70, 0.85]:
            low = max(0.05, high * low_ratio)
            for min_duration in [0.04, 0.08]:
                for merge_gap in [0.00, 0.04]:
                    for min_peak_gap in [0.20, 0.32, 0.48]:
                        for valley_ratio in [0.45, 0.60, 0.75]:
                            configs.append(
                                DecoderConfig(
                                    decoder="hysteresis_valley",
                                    high_threshold=high,
                                    low_threshold=low,
                                    min_duration=min_duration,
                                    merge_gap=merge_gap,
                                    min_peak_gap=min_peak_gap,
                                    valley_ratio=valley_ratio,
                                    smooth_frames=1,
                                )
                            )
        for min_peak_gap in [0.20, 0.32, 0.48]:
            for low_ratio in [0.55, 0.70]:
                low = max(0.05, high * low_ratio)
                configs.append(
                    DecoderConfig(
                        decoder="peak",
                        high_threshold=high,
                        low_threshold=low,
                        min_duration=0.04,
                        merge_gap=0.00,
                        min_peak_gap=min_peak_gap,
                        valley_ratio=0.60,
                        smooth_frames=1,
                    )
                )
    if max_configs > 0:
        return configs[:max_configs]
    return configs


def run_prob_model(args: argparse.Namespace, labels: Sequence[str]) -> list[dict[str, Any]]:
    checkpoint = torch.load(args.checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    checkpoint_labels = list(checkpoint.get("labels") or labels)
    if list(labels) != checkpoint_labels:
        raise ValueError("ontology labels do not match checkpoint labels")
    report = checkpoint.get("report") or {}
    model_checkpoint_name = str(report.get("checkpoint") or "BEATs_strong_1")
    device = make_device(args.device)
    label_to_id = {label: index for index, label in enumerate(labels)}
    scene_rows = load_scene_manifest(args.detector_manifest.resolve(), label_to_id)
    model = load_model(len(labels), model_checkpoint_name, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    loader = DataLoader(
        SceneDataset(scene_rows, audio_root=args.audio_root.resolve()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    print(f"running detector: scenes={len(scene_rows)} labels={len(labels)} device={device}", flush=True)
    return run_model(model, loader, labels, device)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_ontology(args.ontology.resolve())
    qa_rows = list(read_jsonl(args.qa_manifest.resolve()))
    model_predictions = run_prob_model(args, labels)
    qa_scene_ids = {str(row.get("scene_id") or "") for row in qa_rows}
    qa_rows = [row for row in qa_rows if str(row.get("scene_id") or "") in qa_scene_ids]

    if args.decoder_config:
        config = load_decoder_config(args.decoder_config.resolve())
        summary, decoded_rows, open_rows, option_rows = score_config(
            model_predictions,
            labels,
            qa_rows,
            config,
            event_iou_threshold=args.event_iou_threshold,
        )
        report = {
            "format": "qces_event_decoder_eval_v1",
            "checkpoint_path": str(args.checkpoint_path.resolve()),
            "detector_manifest": str(args.detector_manifest.resolve()),
            "qa_manifest": str(args.qa_manifest.resolve()),
            "ontology": str(args.ontology.resolve()),
            "summary": summary,
        }
        save_json(output_dir / "eval_report.json", report)
        write_jsonl(output_dir / "predicted_events.jsonl", compact_decoded_rows(decoded_rows))
        write_jsonl(output_dir / "qa_predictions.jsonl", open_rows)
        write_jsonl(output_dir / "option_aware_predictions.jsonl", option_rows)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
        return

    configs = candidate_configs(args.max_configs)
    summaries: list[dict[str, Any]] = []
    best_by_mixed: tuple[float, dict[str, Any], DecoderConfig] | None = None
    best_by_event: tuple[float, dict[str, Any], DecoderConfig] | None = None
    best_by_option: tuple[float, dict[str, Any], DecoderConfig] | None = None
    for index, config in enumerate(configs, start=1):
        summary, _, _, _ = score_config(
            model_predictions,
            labels,
            qa_rows,
            config,
            event_iou_threshold=args.event_iou_threshold,
        )
        summaries.append(summary)
        event_score = float(summary["detector"]["event_iou"]["f1_↑"])
        option_score = float(summary["option_qa"]["accuracy_↑"])
        mixed_score = float(summary["mixed_score_↑"])
        if best_by_mixed is None or mixed_score > best_by_mixed[0]:
            best_by_mixed = (mixed_score, summary, config)
        if best_by_event is None or event_score > best_by_event[0]:
            best_by_event = (event_score, summary, config)
        if best_by_option is None or option_score > best_by_option[0]:
            best_by_option = (option_score, summary, config)
        if index % 100 == 0:
            print(f"swept {index}/{len(configs)} configs best_mixed={best_by_mixed[0]:.4f}", flush=True)

    summaries.sort(
        key=lambda item: (
            float(item["mixed_score_↑"]),
            float(item["option_qa"]["accuracy_↑"]),
            float(item["detector"]["event_iou"]["f1_↑"]),
        ),
        reverse=True,
    )
    report = {
        "format": "qces_event_decoder_sweep_v1",
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "detector_manifest": str(args.detector_manifest.resolve()),
        "qa_manifest": str(args.qa_manifest.resolve()),
        "ontology": str(args.ontology.resolve()),
        "configs_tested": len(configs),
        "best_mixed": best_by_mixed[1],
        "best_event_f1": best_by_event[1],
        "best_option_qa": best_by_option[1],
        "top_25": summaries[:25],
    }
    save_json(output_dir / "sweep_report.json", report)
    save_json(output_dir / "best_mixed_config.json", {"config": report["best_mixed"]["config"]})
    save_json(output_dir / "best_event_f1_config.json", {"config": report["best_event_f1"]["config"]})
    save_json(output_dir / "best_option_qa_config.json", {"config": report["best_option_qa"]["config"]})

    rows = []
    for rank, summary in enumerate(summaries[:100], start=1):
        rows.append({"rank": rank, **summary})
    write_jsonl(output_dir / "top_100_configs.jsonl", rows)

    top = report["best_mixed"]
    print(json.dumps({
        "configs_tested": len(configs),
        "best_mixed_config": top["config"],
        "best_mixed": {
            "mixed_score_↑": top["mixed_score_↑"],
            "event_f1_↑": top["detector"]["event_iou"]["f1_↑"],
            "frame_f1_↑": top["detector"]["frame"]["f1_↑"],
            "option_qa_↑": top["option_qa"]["accuracy_↑"],
            "option_answerable_↑": top["option_qa"]["answerable_accuracy_↑"],
            "option_no_evidence_↑": top["option_qa"]["no_evidence_accuracy_↑"],
            "open_qa_↑": top["open_qa"]["accuracy_↑"],
        },
    }, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
