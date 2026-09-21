#!/usr/bin/env python3
"""Evaluate hybrid QCES evidence: onset QA + hysteresis boundary refinement.

The previous ablations show a split:

- onset-head inventory improves QA but hurts event IoU;
- hysteresis decoding improves event IoU but hurts QA.

This script evaluates a hybrid without retraining:

1. Run the onset-head detector to select answer/anchor instances.
2. Run the frame detector with hysteresis to get cleaner boundaries.
3. Refine selected onset instances with nearest same-label hysteresis spans.

It reports both scene-level inventory metrics and question-conditioned evidence
IoU, because the hybrid is fundamentally question-conditioned evidence rather
than just a generic scene inventory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_qces_pretrainedsed_detector as frame_mod
import train_qces_pretrainedsed_onset_detector as onset_mod
from score_qces_option_aware_executor import option_aware_execute, summarize as summarize_option_rows
from sweep_qces_event_decoder import (
    DecoderConfig,
    decode_events as decode_hysteresis_events,
    summarize_detector_from_decoded,
)


DEFAULT_FRAME_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_30class/full801_val102_e12/pretrainedsed_beats_qces_detector.pt"
)
DEFAULT_ONSET_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_onset_detector_30class/full801_val102_e6_tiny_v1/pretrainedsed_beats_qces_onset_detector.pt"
)
DEFAULT_FRAME_DECODER_CONFIG = (
    PROJECT_ROOT / "outputs/qces_event_decoder_sweep/head_only_val_full/best_event_f1_config.json"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-checkpoint-path", type=Path, default=DEFAULT_FRAME_CHECKPOINT)
    parser.add_argument("--onset-checkpoint-path", type=Path, default=DEFAULT_ONSET_CHECKPOINT)
    parser.add_argument("--frame-decoder-config", type=Path, default=DEFAULT_FRAME_DECODER_CONFIG)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=frame_mod.DEFAULT_AUDIT_DIR / "ontology_ready_train.txt")
    parser.add_argument("--audio-root", type=Path, default=frame_mod.DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument("--match-tolerance", type=float, default=0.75)
    parser.add_argument("--nms-iou", type=float, default=0.65)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_decoder_config(path: Path) -> DecoderConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "config" in payload:
        payload = payload["config"]
    return DecoderConfig(**payload)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def event_interval(event: Mapping[str, Any]) -> tuple[float, float]:
    return float(event["onset_seconds"]), float(event["offset_seconds"])


def event_midpoint(event: Mapping[str, Any]) -> float:
    onset, offset = event_interval(event)
    return 0.5 * (onset + offset)


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    inter = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return inter / union if union > 0 else 0.0


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = sorted((float(a), float(b)) for a, b in intervals if float(b) > float(a))
    out: list[tuple[float, float]] = []
    for onset, offset in clean:
        if out and onset <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], offset))
        else:
            out.append((onset, offset))
    return out


def interval_list_duration(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(max(0.0, b - a) for a, b in merge_intervals(intervals))


def interval_list_iou(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> float:
    left_m = merge_intervals(left)
    right_m = merge_intervals(right)
    if not left_m and not right_m:
        return 1.0
    if not left_m or not right_m:
        return 0.0
    inter = 0.0
    for a0, a1 in left_m:
        for b0, b1 in right_m:
            inter += max(0.0, min(a1, b1) - max(a0, b0))
    union = interval_list_duration(left_m) + interval_list_duration(right_m) - inter
    return inter / union if union > 0 else 0.0


def occurrence(events: Sequence[Mapping[str, Any]], label: str, ordinal: int) -> Mapping[str, Any] | None:
    candidates = [event for event in events if str(event.get("label")) == label]
    candidates.sort(key=lambda item: (float(item["onset_seconds"]), float(item["offset_seconds"])))
    if ordinal <= 0 or len(candidates) < ordinal:
        return None
    return candidates[ordinal - 1]


def answer_options(row: Mapping[str, Any]) -> set[str]:
    return {str(option) for option in row.get("answer_options") or [] if str(option) != "no_evidence"}


def execute_with_event(
    row: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    option_aware: bool,
) -> tuple[str, bool, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    relation = str(row.get("relation") or "")
    query_label = str(row.get("query_label") or "")
    ordinal = int(row.get("query_instance_ordinal") or 1)
    options = answer_options(row) if option_aware else None

    if relation in {"after", "before"}:
        anchor = occurrence(events, query_label, ordinal)
        if anchor is None:
            return "no_evidence", True, None, None
        anchor_onset = float(anchor["onset_seconds"])
        if relation == "after":
            candidates = [event for event in events if float(event["onset_seconds"]) > anchor_onset + 1e-6]
            if options is not None:
                candidates = [event for event in candidates if str(event.get("label")) in options]
            if not candidates:
                return "no_evidence", True, anchor, None
            answer = min(candidates, key=lambda item: (float(item["onset_seconds"]), -float(item.get("confidence", 0.0))))
        else:
            candidates = [event for event in events if float(event["onset_seconds"]) < anchor_onset - 1e-6]
            if options is not None:
                candidates = [event for event in candidates if str(event.get("label")) in options]
            if not candidates:
                return "no_evidence", True, anchor, None
            answer = max(candidates, key=lambda item: (float(item["onset_seconds"]), float(item.get("confidence", 0.0))))
        return str(answer["label"]), False, anchor, answer

    if relation == "first":
        candidate_labels = [str(label) for label in row.get("query_candidate_labels") or []]
        if not candidate_labels and options is not None:
            candidate_labels = sorted(options)
        candidates = [event for event in events if str(event.get("label")) in set(candidate_labels)]
        if not candidates:
            return "no_evidence", True, None, None
        answer = min(candidates, key=lambda item: (float(item["onset_seconds"]), -float(item.get("confidence", 0.0))))
        return str(answer["label"]), False, None, answer

    return "unsupported", True, None, None


def nearest_same_label_event(
    source: Mapping[str, Any] | None,
    candidates: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
    used: set[int] | None = None,
) -> tuple[int | None, Mapping[str, Any] | None]:
    if source is None:
        return None, None
    label = str(source.get("label"))
    source_interval = event_interval(source)
    source_mid = event_midpoint(source)
    best_index: int | None = None
    best_score: tuple[float, float] | None = None
    for index, candidate in enumerate(candidates):
        if used is not None and index in used:
            continue
        if str(candidate.get("label")) != label:
            continue
        mid_distance = abs(event_midpoint(candidate) - source_mid)
        overlap_bonus = interval_iou(source_interval, event_interval(candidate))
        if mid_distance > tolerance and overlap_bonus <= 0.0:
            continue
        score = (mid_distance, -overlap_bonus)
        if best_score is None or score < best_score:
            best_index = index
            best_score = score
    return best_index, (candidates[best_index] if best_index is not None else None)


def refined_event(
    source: Mapping[str, Any] | None,
    hysteresis_events: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
) -> Mapping[str, Any] | None:
    _, match = nearest_same_label_event(source, hysteresis_events, tolerance=tolerance)
    if match is None:
        return source
    output = dict(match)
    output["source"] = "hysteresis_refined"
    output["source_onset_seconds"] = source.get("onset_seconds") if source else None
    output["source_offset_seconds"] = source.get("offset_seconds") if source else None
    output["source_confidence"] = source.get("confidence") if source else None
    output["confidence"] = max(float(match.get("confidence", 0.0)), float(source.get("confidence", 0.0) if source else 0.0))
    return output


def nms_events(events: Sequence[Mapping[str, Any]], *, nms_iou: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        duplicate = False
        for kept in selected:
            if str(kept.get("label")) != str(event.get("label")):
                continue
            if interval_iou(event_interval(kept), event_interval(event)) >= nms_iou:
                duplicate = True
                break
        if not duplicate:
            selected.append(dict(event))
    selected.sort(key=lambda item: (float(item["onset_seconds"]), float(item["offset_seconds"]), str(item["label"])))
    return selected


def refine_onset_inventory(
    onset_events: Sequence[Mapping[str, Any]],
    hysteresis_events: Sequence[Mapping[str, Any]],
    *,
    tolerance: float,
    nms_iou: float,
) -> list[dict[str, Any]]:
    used: set[int] = set()
    output: list[dict[str, Any]] = []
    for event in sorted(onset_events, key=lambda item: float(item.get("confidence", 0.0)), reverse=True):
        match_index, match = nearest_same_label_event(event, hysteresis_events, tolerance=tolerance, used=used)
        if match_index is not None and match is not None:
            used.add(match_index)
            refined = dict(match)
            refined["source"] = "hysteresis_refined"
            refined["source_onset_seconds"] = event.get("onset_seconds")
            refined["source_offset_seconds"] = event.get("offset_seconds")
            refined["source_confidence"] = event.get("confidence")
            refined["confidence"] = max(float(match.get("confidence", 0.0)), float(event.get("confidence", 0.0)))
            output.append(refined)
        else:
            fallback = dict(event)
            fallback["source"] = "onset_fallback"
            output.append(fallback)
    return nms_events(output, nms_iou=nms_iou)


def summarize_open_qa_with_rows(
    qa_rows: Sequence[Mapping[str, Any]],
    pred_events_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for row in qa_rows:
        scene_id = str(row.get("scene_id") or "")
        pred_answer, pred_noev, _, _ = execute_with_event(row, pred_events_by_scene.get(scene_id, ()), option_aware=False)
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
    summary = frame_mod.summarize_qa(qa_rows, pred_events_by_scene)
    return {key: value for key, value in summary.items() if key != "rows"}, rows


def summarize_option_qa_with_rows(
    qa_rows: Sequence[Mapping[str, Any]],
    pred_events_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for row in qa_rows:
        scene_id = str(row.get("scene_id") or "")
        events = pred_events_by_scene.get(scene_id, ())
        pred_answer, pred_noev = option_aware_execute(row, events)
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


def gold_answer_intervals(row: Mapping[str, Any]) -> list[tuple[float, float]]:
    return [(float(a), float(b)) for a, b in row.get("answer_intervals") or [] if float(b) > float(a)]


def gold_evidence_intervals(row: Mapping[str, Any]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for a, b in row.get("anchor_intervals") or []:
        if float(b) > float(a):
            intervals.append((float(a), float(b)))
    intervals.extend(gold_answer_intervals(row))
    return intervals


def summarize_question_evidence(
    qa_rows: Sequence[Mapping[str, Any]],
    onset_events_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
    hysteresis_events_by_scene: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    match_tolerance: float,
    option_aware: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for row in qa_rows:
        scene_id = str(row.get("scene_id") or "")
        onset_events = onset_events_by_scene.get(scene_id, ())
        hyst_events = hysteresis_events_by_scene.get(scene_id, ())
        pred_answer, pred_noev, onset_anchor, onset_answer = execute_with_event(
            row,
            onset_events,
            option_aware=option_aware,
        )
        raw_anchor = onset_anchor
        raw_answer = onset_answer
        refined_anchor = refined_event(onset_anchor, hyst_events, tolerance=match_tolerance)
        refined_answer = refined_event(onset_answer, hyst_events, tolerance=match_tolerance)

        answer_ok = pred_answer == str(row.get("answer")) and pred_noev == bool(row.get("no_evidence"))
        no_evidence_gold = bool(row.get("no_evidence"))
        answer_intervals = gold_answer_intervals(row)
        evidence_intervals = gold_evidence_intervals(row)

        raw_answer_pred_intervals = [event_interval(raw_answer)] if raw_answer is not None else []
        refined_answer_pred_intervals = [event_interval(refined_answer)] if refined_answer is not None else []
        raw_evidence_pred_intervals = []
        refined_evidence_pred_intervals = []
        if raw_anchor is not None:
            raw_evidence_pred_intervals.append(event_interval(raw_anchor))
        if raw_answer is not None:
            raw_evidence_pred_intervals.append(event_interval(raw_answer))
        if refined_anchor is not None:
            refined_evidence_pred_intervals.append(event_interval(refined_anchor))
        if refined_answer is not None:
            refined_evidence_pred_intervals.append(event_interval(refined_answer))

        raw_answer_iou = interval_list_iou(raw_answer_pred_intervals, answer_intervals) if not no_evidence_gold else None
        refined_answer_iou = interval_list_iou(refined_answer_pred_intervals, answer_intervals) if not no_evidence_gold else None
        raw_evidence_iou = interval_list_iou(raw_evidence_pred_intervals, evidence_intervals) if not no_evidence_gold else None
        refined_evidence_iou = interval_list_iou(refined_evidence_pred_intervals, evidence_intervals) if not no_evidence_gold else None

        rows.append(
            {
                "scene_id": scene_id,
                "question": row.get("question"),
                "relation": row.get("relation"),
                "answer": row.get("answer"),
                "answer_options": row.get("answer_options"),
                "no_evidence": no_evidence_gold,
                "no_evidence_reason": row.get("no_evidence_reason"),
                "pred_answer": pred_answer,
                "pred_no_evidence": pred_noev,
                "qa_ok": answer_ok,
                "raw_answer_iou": raw_answer_iou,
                "refined_answer_iou": refined_answer_iou,
                "raw_evidence_iou": raw_evidence_iou,
                "refined_evidence_iou": refined_evidence_iou,
                "raw_answer_event": raw_answer,
                "refined_answer_event": refined_answer,
                "raw_anchor_event": raw_anchor,
                "refined_anchor_event": refined_anchor,
            }
        )

    answerable = [row for row in rows if not bool(row["no_evidence"])]
    no_evidence = [row for row in rows if bool(row["no_evidence"])]
    raw_answer_ious = [float(row["raw_answer_iou"]) for row in answerable if row["raw_answer_iou"] is not None]
    refined_answer_ious = [float(row["refined_answer_iou"]) for row in answerable if row["refined_answer_iou"] is not None]
    raw_evidence_ious = [float(row["raw_evidence_iou"]) for row in answerable if row["raw_evidence_iou"] is not None]
    refined_evidence_ious = [float(row["refined_evidence_iou"]) for row in answerable if row["refined_evidence_iou"] is not None]

    def mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def rate_at(values: Sequence[float], threshold: float) -> float:
        return sum(value >= threshold for value in values) / max(len(values), 1)

    summary = {
        "items": len(rows),
        "answerable_items": len(answerable),
        "no_evidence_items": len(no_evidence),
        "qa_accuracy_↑": sum(bool(row["qa_ok"]) for row in rows) / max(len(rows), 1),
        "answerable_accuracy_↑": sum(bool(row["qa_ok"]) for row in answerable) / max(len(answerable), 1),
        "no_evidence_accuracy_↑": sum(bool(row["qa_ok"]) for row in no_evidence) / max(len(no_evidence), 1),
        "raw_answer_iou_mean_↑": mean(raw_answer_ious),
        "refined_answer_iou_mean_↑": mean(refined_answer_ious),
        "raw_answer_iou@0.30_↑": rate_at(raw_answer_ious, 0.30),
        "refined_answer_iou@0.30_↑": rate_at(refined_answer_ious, 0.30),
        "raw_evidence_iou_mean_↑": mean(raw_evidence_ious),
        "refined_evidence_iou_mean_↑": mean(refined_evidence_ious),
        "raw_evidence_iou@0.30_↑": rate_at(raw_evidence_ious, 0.30),
        "refined_evidence_iou@0.30_↑": rate_at(refined_evidence_ious, 0.30),
    }
    return summary, rows


def compact_inventory_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scene_id": row["scene_id"],
            "split": row.get("split"),
            "duration_seconds": row.get("duration_seconds"),
            "predicted_events": row.get("predicted_events"),
            "gold_events": row.get("gold_events"),
        }
        for row in rows
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = frame_mod.load_ontology(args.ontology.resolve())
    device = frame_mod.make_device(args.device)
    label_to_id = {label: index for index, label in enumerate(labels)}
    scene_rows = frame_mod.load_scene_manifest(args.detector_manifest.resolve(), label_to_id)
    qa_rows = [
        row
        for row in frame_mod.read_jsonl(args.qa_manifest.resolve())
        if str(row.get("scene_id") or "") in {scene.scene_id for scene in scene_rows}
    ]
    loader = DataLoader(
        frame_mod.SceneDataset(scene_rows, audio_root=args.audio_root.resolve()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=frame_mod.collate,
    )

    frame_checkpoint = torch.load(args.frame_checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    frame_report = frame_checkpoint.get("report") or {}
    frame_model_name = str(frame_report.get("checkpoint") or "BEATs_strong_1")
    frame_model = frame_mod.load_model(len(labels), frame_model_name, device)
    frame_model.load_state_dict(frame_checkpoint["model_state_dict"], strict=True)
    frame_model.to(device).eval()
    frame_predictions = frame_mod.run_model(frame_model, loader, labels, device)

    onset_checkpoint = torch.load(args.onset_checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    onset_report = onset_checkpoint.get("report") or {}
    init_checkpoint = Path(onset_report.get("init_checkpoint") or "")
    if not init_checkpoint.exists():
        raise SystemExit(f"missing onset init checkpoint: {init_checkpoint}")
    onset_model = onset_mod.load_initialized_model(labels, init_checkpoint, device)
    onset_model.load_state_dict(onset_checkpoint["model_state_dict"], strict=True)
    onset_model.to(device).eval()
    onset_predictions = onset_mod.run_model(onset_model, loader, labels, device)

    frame_decoder_config = read_decoder_config(args.frame_decoder_config.resolve())
    onset_decoder_config = onset_mod.OnsetDecoderConfig(**onset_checkpoint["best_config"])

    hysteresis_by_scene: dict[str, list[dict[str, Any]]] = {}
    onset_by_scene: dict[str, list[dict[str, Any]]] = {}
    hybrid_by_scene: dict[str, list[dict[str, Any]]] = {}
    hysteresis_rows: list[dict[str, Any]] = []
    onset_rows: list[dict[str, Any]] = []
    hybrid_rows: list[dict[str, Any]] = []

    onset_pred_by_scene = {str(row["scene_id"]): row for row in onset_predictions}
    for frame_row in frame_predictions:
        scene_id = str(frame_row["scene_id"])
        hyst_events = decode_hysteresis_events(frame_row["probs"], frame_row, labels, frame_decoder_config)
        onset_events = onset_mod.decode_onset_events(onset_pred_by_scene[scene_id], labels, onset_decoder_config)
        hybrid_events = refine_onset_inventory(
            onset_events,
            hyst_events,
            tolerance=args.match_tolerance,
            nms_iou=args.nms_iou,
        )
        hysteresis_by_scene[scene_id] = hyst_events
        onset_by_scene[scene_id] = onset_events
        hybrid_by_scene[scene_id] = hybrid_events

        common = {
            "scene_id": scene_id,
            "split": frame_row.get("split"),
            "duration_seconds": float(frame_row["duration_seconds"]),
            "gold_events": list(frame_row["gold_events"]),
            "targets": frame_row["targets"],
        }
        hysteresis_rows.append({**common, "predicted_events": hyst_events})
        onset_rows.append({**common, "predicted_events": onset_events})
        hybrid_rows.append({**common, "predicted_events": hybrid_events})

    def summarize_inventory(name: str, rows: Sequence[Mapping[str, Any]], by_scene: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
        detector = summarize_detector_from_decoded(rows, labels, event_iou_threshold=args.event_iou_threshold)
        open_summary, open_rows = summarize_open_qa_with_rows(qa_rows, by_scene)
        option_summary, option_rows = summarize_option_qa_with_rows(qa_rows, by_scene)
        write_jsonl(output_dir / f"{name}_qa_predictions.jsonl", open_rows)
        write_jsonl(output_dir / f"{name}_option_aware_predictions.jsonl", option_rows)
        write_jsonl(output_dir / f"{name}_predicted_events.jsonl", compact_inventory_rows(rows))
        return {
            "detector": detector,
            "open_qa": open_summary,
            "option_qa": option_summary,
        }

    inventory = {
        "hysteresis": summarize_inventory("hysteresis", hysteresis_rows, hysteresis_by_scene),
        "onset": summarize_inventory("onset", onset_rows, onset_by_scene),
        "hybrid_refine_onset": summarize_inventory("hybrid_refine_onset", hybrid_rows, hybrid_by_scene),
    }
    qce_summary, qce_rows = summarize_question_evidence(
        qa_rows,
        onset_by_scene,
        hysteresis_by_scene,
        match_tolerance=args.match_tolerance,
        option_aware=True,
    )
    write_jsonl(output_dir / "question_conditioned_hybrid_evidence.jsonl", qce_rows)

    report = {
        "format": "qces_hybrid_onset_hysteresis_eval_v1",
        "frame_checkpoint_path": str(args.frame_checkpoint_path.resolve()),
        "onset_checkpoint_path": str(args.onset_checkpoint_path.resolve()),
        "frame_decoder_config_path": str(args.frame_decoder_config.resolve()),
        "detector_manifest": str(args.detector_manifest.resolve()),
        "qa_manifest": str(args.qa_manifest.resolve()),
        "ontology": str(args.ontology.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "device": str(device),
        "scenes": len(scene_rows),
        "labels": labels,
        "match_tolerance": args.match_tolerance,
        "nms_iou": args.nms_iou,
        "frame_decoder_config": frame_decoder_config.__dict__,
        "onset_decoder_config": onset_decoder_config.__dict__,
        "inventory": inventory,
        "question_conditioned_hybrid": qce_summary,
    }
    frame_mod.save_json(output_dir / "eval_report.json", report)
    md = [
        "# QCES hybrid onset+hysteresis evaluation",
        "",
        f"- scenes: {len(scene_rows)}",
        f"- QA rows: {len(qa_rows)}",
        f"- match tolerance: {args.match_tolerance:.2f}s",
        "",
        "## Scene-level inventory",
        "",
        "| method | option QA ↑ | answerable ↑ | no-evidence ↑ | event F1 ↑ | frame F1 ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in inventory.items():
        md.append(
            f"| {name} | {item['option_qa']['accuracy_↑']:.3f} | "
            f"{item['option_qa']['answerable_accuracy_↑']:.3f} | "
            f"{item['option_qa']['no_evidence_accuracy_↑']:.3f} | "
            f"{item['detector']['event_iou']['f1_↑']:.3f} | "
            f"{item['detector']['frame']['f1_↑']:.3f} |"
        )
    md.extend(
        [
            "",
            "## Question-conditioned hybrid evidence",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| QA accuracy ↑ | {qce_summary['qa_accuracy_↑']:.3f} |",
            f"| answerable accuracy ↑ | {qce_summary['answerable_accuracy_↑']:.3f} |",
            f"| no-evidence accuracy ↑ | {qce_summary['no_evidence_accuracy_↑']:.3f} |",
            f"| raw answer IoU mean ↑ | {qce_summary['raw_answer_iou_mean_↑']:.3f} |",
            f"| refined answer IoU mean ↑ | {qce_summary['refined_answer_iou_mean_↑']:.3f} |",
            f"| raw answer IoU@0.30 ↑ | {qce_summary['raw_answer_iou@0.30_↑']:.3f} |",
            f"| refined answer IoU@0.30 ↑ | {qce_summary['refined_answer_iou@0.30_↑']:.3f} |",
            f"| raw evidence IoU mean ↑ | {qce_summary['raw_evidence_iou_mean_↑']:.3f} |",
            f"| refined evidence IoU mean ↑ | {qce_summary['refined_evidence_iou_mean_↑']:.3f} |",
            f"| raw evidence IoU@0.30 ↑ | {qce_summary['raw_evidence_iou@0.30_↑']:.3f} |",
            f"| refined evidence IoU@0.30 ↑ | {qce_summary['refined_evidence_iou@0.30_↑']:.3f} |",
        ]
    )
    (output_dir / "eval_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
