#!/usr/bin/env python3
"""Question-conditioned anchor retrieval + pair reranking for detector QA."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.evaluate_qces_detector_inventory_qa import (
    QAItem,
    build_qa_items,
    interval_iou,
    intervals_iou,
    read_jsonl,
    summarize_rows,
    write_json,
)


DEFAULT_FRAME_PROBS = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200/partial_as_fuss_e4_unfreeze1_highthr_resume/val_frame_probs.pt"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/detector_manifest_val.jsonl"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/qces_detector_pair_retrieval_qa/multisource200_unfreeze1_val_v1"
)


@dataclass(frozen=True)
class Segment:
    label: str
    label_id: int
    onset_seconds: float
    offset_seconds: float
    confidence: float
    source: str = "predicted"

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.offset_seconds - self.onset_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "label_id": self.label_id,
            "onset_seconds": round(self.onset_seconds, 4),
            "offset_seconds": round(self.offset_seconds, 4),
            "confidence": float(self.confidence),
            "source": self.source,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-probs", type=Path, default=DEFAULT_FRAME_PROBS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-answerable-per-scene", type=int, default=4)
    parser.add_argument("--max-no-evidence-per-scene", type=int, default=2)
    parser.add_argument("--min-gap-seconds", type=float, default=0.0)
    parser.add_argument("--evidence-iou-threshold", type=float, default=0.30)
    parser.add_argument("--anchor-top-k", type=int, default=5)
    parser.add_argument("--answer-top-k-per-gold-label", type=int, default=5)
    parser.add_argument("--anchor-peak-ratio", type=float, default=0.35)
    parser.add_argument("--anchor-min-peak-score", type=float, default=0.02)
    parser.add_argument("--answer-threshold", type=float, default=0.70)
    parser.add_argument("--answer-min-duration", type=float, default=0.08)
    parser.add_argument("--merge-gap", type=float, default=0.12)
    parser.add_argument("--max-answer-proposals", type=int, default=160)
    parser.add_argument("--distance-penalty", type=float, default=0.10)
    parser.add_argument("--intervening-penalty", type=float, default=0.05)
    parser.add_argument("--pair-min-score", type=float, default=0.0)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "predicted_pair",
            "oracle_anchor",
            "oracle_answer",
            "oracle_labels_pred_ts",
            "pred_labels_oracle_ts",
            "oracle_both",
        ],
        choices=[
            "predicted_pair",
            "oracle_anchor",
            "oracle_answer",
            "oracle_labels_pred_ts",
            "pred_labels_oracle_ts",
            "oracle_both",
        ],
    )
    parser.add_argument("--sweep", action="store_true", help="Run a small threshold/score sweep for predicted_pair.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def frame_bounds(index: int, duration: float, time_steps: int) -> tuple[float, float]:
    frame_seconds = duration / max(time_steps, 1)
    return index * frame_seconds, (index + 1) * frame_seconds


def segment_iou(left: Segment, right: Segment) -> float:
    return interval_iou(
        (left.onset_seconds, left.offset_seconds),
        (right.onset_seconds, right.offset_seconds),
    )


def smooth_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() < 3:
        return scores.float()
    x = scores.float().view(1, 1, -1)
    return torch.nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1).view(-1)


def arg_cache(args: argparse.Namespace, name: str) -> dict[Any, Any]:
    cache = getattr(args, name, None)
    if cache is None:
        cache = {}
        setattr(args, name, cache)
    return cache


def topk_label_segments(
    scores: torch.Tensor,
    *,
    label: str,
    label_id: int,
    duration: float,
    top_k: int,
    peak_ratio: float,
    min_peak_score: float,
    min_duration: float,
    nms_iou: float = 0.50,
    source: str = "anchor_topk",
) -> list[Segment]:
    values = smooth_scores(scores).clone()
    time_steps = int(values.numel())
    segments: list[Segment] = []
    if time_steps == 0 or top_k <= 0:
        return segments
    for _ in range(top_k * 3):
        if len(segments) >= top_k:
            break
        peak_score, peak_idx_t = torch.max(values, dim=0)
        peak = float(peak_score.item())
        peak_idx = int(peak_idx_t.item())
        if not math.isfinite(peak) or peak < min_peak_score:
            break
        cutoff = max(min_peak_score, peak * peak_ratio)
        left = peak_idx
        while left > 0 and float(values[left - 1].item()) >= cutoff:
            left -= 1
        right = peak_idx + 1
        while right < time_steps and float(values[right].item()) >= cutoff:
            right += 1
        onset, _ = frame_bounds(left, duration, time_steps)
        _, offset = frame_bounds(right - 1, duration, time_steps)
        if offset - onset < min_duration:
            center = (onset + offset) / 2.0
            onset = max(0.0, center - min_duration / 2.0)
            offset = min(duration, center + min_duration / 2.0)
        candidate = Segment(
            label=label,
            label_id=label_id,
            onset_seconds=onset,
            offset_seconds=offset,
            confidence=peak,
            source=source,
        )
        if all(segment_iou(candidate, old) < nms_iou for old in segments):
            segments.append(candidate)
        values[left:right] = -1.0
        if torch.max(values).item() < min_peak_score:
            break
    segments.sort(key=lambda item: (item.onset_seconds, -item.confidence))
    return segments[:top_k]


def threshold_label_segments(
    scores: torch.Tensor,
    *,
    label: str,
    label_id: int,
    duration: float,
    threshold: float,
    min_duration: float,
    merge_gap: float,
    source: str = "answer_threshold",
) -> list[Segment]:
    values = smooth_scores(scores)
    active = (values >= threshold).cpu().numpy().astype(bool).tolist()
    time_steps = int(values.numel())
    raw: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(active + [False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            raw.append((start, index))
            start = None
    merged: list[Segment] = []
    for start_idx, end_idx in raw:
        onset, _ = frame_bounds(start_idx, duration, time_steps)
        _, offset = frame_bounds(end_idx - 1, duration, time_steps)
        if merged and onset <= merged[-1].offset_seconds + merge_gap:
            prev = merged[-1]
            conf = max(prev.confidence, float(values[start_idx:end_idx].max().item()))
            merged[-1] = Segment(
                label=label,
                label_id=label_id,
                onset_seconds=prev.onset_seconds,
                offset_seconds=max(prev.offset_seconds, offset),
                confidence=conf,
                source=source,
            )
        else:
            merged.append(
                Segment(
                    label=label,
                    label_id=label_id,
                    onset_seconds=onset,
                    offset_seconds=offset,
                    confidence=float(values[start_idx:end_idx].max().item()),
                    source=source,
                )
            )
    return [seg for seg in merged if seg.duration_seconds >= min_duration]


def all_threshold_segments(
    probs: torch.Tensor,
    labels: Sequence[str],
    *,
    duration: float,
    threshold: float,
    min_duration: float,
    merge_gap: float,
    max_segments: int,
) -> list[Segment]:
    segments: list[Segment] = []
    for label_id, label in enumerate(labels):
        segments.extend(
            threshold_label_segments(
                probs[:, label_id],
                label=label,
                label_id=label_id,
                duration=duration,
                threshold=threshold,
                min_duration=min_duration,
                merge_gap=merge_gap,
            )
        )
    segments.sort(key=lambda item: item.confidence, reverse=True)
    keep = segments[:max_segments] if max_segments > 0 else segments
    return sorted(keep, key=lambda item: (item.onset_seconds, item.offset_seconds, item.label))


def cached_all_threshold_segments(
    entry: Mapping[str, Any],
    labels: Sequence[str],
    args: argparse.Namespace,
    *,
    threshold: float,
) -> list[Segment]:
    cache = arg_cache(args, "_all_threshold_cache")
    key = (
        id(entry),
        round(float(threshold), 4),
        round(float(args.answer_min_duration), 4),
        round(float(args.merge_gap), 4),
        int(args.max_answer_proposals),
    )
    if key not in cache:
        cache[key] = all_threshold_segments(
            entry["probs"].float(),
            labels,
            duration=float(entry["duration_seconds"]),
            threshold=threshold,
            min_duration=args.answer_min_duration,
            merge_gap=args.merge_gap,
            max_segments=args.max_answer_proposals,
        )
    return cache[key]


def oracle_segment(event: tuple[float, float] | None, label: str | None, labels: Sequence[str], *, source: str) -> Segment | None:
    if event is None or label is None:
        return None
    label_id = labels.index(label) if label in labels else -1
    return Segment(label, label_id, float(event[0]), float(event[1]), 1.0, source=source)


def frame_slice_for_interval(interval: tuple[float, float], duration: float, time_steps: int) -> slice:
    start = max(0, min(time_steps - 1, int(math.floor(interval[0] / max(duration, 1e-6) * time_steps))))
    end = max(start + 1, min(time_steps, int(math.ceil(interval[1] / max(duration, 1e-6) * time_steps))))
    return slice(start, end)


def predicted_label_at_oracle_time(
    probs: torch.Tensor,
    labels: Sequence[str],
    *,
    gold_interval: tuple[float, float] | None,
    duration: float,
    source: str,
) -> Segment | None:
    if gold_interval is None:
        return None
    sl = frame_slice_for_interval(gold_interval, duration, probs.size(0))
    mean_scores = probs[sl].float().mean(dim=0)
    score, label_id_t = torch.max(mean_scores, dim=0)
    label_id = int(label_id_t.item())
    return Segment(
        label=labels[label_id],
        label_id=label_id,
        onset_seconds=float(gold_interval[0]),
        offset_seconds=float(gold_interval[1]),
        confidence=float(score.item()),
        source=source,
    )


def temporal_gap(relation: str, anchor: Segment, answer: Segment) -> float | None:
    if relation == "after":
        if answer.onset_seconds <= anchor.onset_seconds + 1e-6:
            return None
        return max(0.0, answer.onset_seconds - anchor.offset_seconds)
    if relation == "before":
        if answer.onset_seconds >= anchor.onset_seconds - 1e-6:
            return None
        return max(0.0, anchor.onset_seconds - answer.offset_seconds)
    return None


def intervening_count(relation: str, anchor: Segment, answer: Segment, events: Sequence[Segment]) -> int:
    lo = min(anchor.onset_seconds, answer.onset_seconds)
    hi = max(anchor.onset_seconds, answer.onset_seconds)
    count = 0
    for event in events:
        if event.label in {anchor.label, answer.label} and (
            segment_iou(event, anchor) > 0.5 or segment_iou(event, answer) > 0.5
        ):
            continue
        if lo + 1e-6 < event.onset_seconds < hi - 1e-6 and event.confidence >= 0.80:
            count += 1
    return count


def choose_pair(
    item: QAItem,
    anchors: Sequence[Segment],
    answers: Sequence[Segment],
    *,
    context_events: Sequence[Segment],
    distance_penalty: float,
    intervening_penalty: float,
    pair_min_score: float,
) -> tuple[str, bool, tuple[Segment, ...], str, float]:
    if not anchors:
        return "no_evidence", True, (), "anchor_missing", float("-inf")
    best: tuple[float, Segment, Segment] | None = None
    for anchor in anchors:
        for answer in answers:
            if answer.label == anchor.label and segment_iou(anchor, answer) > 0.80:
                continue
            gap = temporal_gap(item.relation, anchor, answer)
            if gap is None:
                continue
            score = (
                float(anchor.confidence)
                + float(answer.confidence)
                - distance_penalty * gap
                - intervening_penalty * intervening_count(item.relation, anchor, answer, context_events)
            )
            if best is None or score > best[0]:
                best = (score, anchor, answer)
    if best is None:
        reason = "no_event_after_anchor" if item.relation == "after" else "no_event_before_anchor"
        return "no_evidence", True, (), reason, float("-inf")
    score, anchor, answer = best
    if score < pair_min_score:
        return "no_evidence", True, (), "pair_score_below_min", score
    return answer.label, False, (anchor, answer), "ok", score


def select_anchor_ordinal(item: QAItem, anchors: Sequence[Segment]) -> list[Segment]:
    ordered = sorted(anchors, key=lambda segment: (segment.onset_seconds, segment.offset_seconds, -segment.confidence))
    index = int(item.anchor_ordinal) - 1
    if index < 0 or index >= len(ordered):
        return []
    return [ordered[index]]


def anchor_segments_for_item(
    item: QAItem,
    entry: Mapping[str, Any],
    labels: Sequence[str],
    args: argparse.Namespace,
) -> list[Segment]:
    cache = arg_cache(args, "_anchor_cache")
    key = (
        id(entry),
        item.anchor_label,
        max(args.anchor_top_k, item.anchor_ordinal),
        round(float(args.anchor_peak_ratio), 4),
        round(float(args.anchor_min_peak_score), 4),
        round(float(args.answer_min_duration), 4),
    )
    if key in cache:
        return cache[key]
    probs = entry["probs"].float()
    if item.anchor_label not in labels:
        return []
    label_id = labels.index(item.anchor_label)
    segments = topk_label_segments(
        probs[:, label_id],
        label=item.anchor_label,
        label_id=label_id,
        duration=float(entry["duration_seconds"]),
        top_k=max(args.anchor_top_k, item.anchor_ordinal),
        peak_ratio=args.anchor_peak_ratio,
        min_peak_score=args.anchor_min_peak_score,
        min_duration=args.answer_min_duration,
        source="anchor_topk",
    )
    cache[key] = segments
    return segments


def answer_segments_for_item(
    item: QAItem,
    entry: Mapping[str, Any],
    labels: Sequence[str],
    args: argparse.Namespace,
    *,
    restrict_to_gold_label: bool = False,
) -> list[Segment]:
    probs = entry["probs"].float()
    duration = float(entry["duration_seconds"])
    if restrict_to_gold_label:
        if item.answer_label is None or item.answer_label not in labels:
            return []
        cache = arg_cache(args, "_gold_label_cache")
        key = (
            id(entry),
            item.answer_label,
            int(args.answer_top_k_per_gold_label),
            round(float(args.anchor_peak_ratio), 4),
            round(float(args.anchor_min_peak_score), 4),
            round(float(args.answer_min_duration), 4),
        )
        if key in cache:
            return cache[key]
        label_id = labels.index(item.answer_label)
        segments = topk_label_segments(
            probs[:, label_id],
            label=item.answer_label,
            label_id=label_id,
            duration=duration,
            top_k=args.answer_top_k_per_gold_label,
            peak_ratio=args.anchor_peak_ratio,
            min_peak_score=args.anchor_min_peak_score,
            min_duration=args.answer_min_duration,
            source="answer_gold_label_topk",
        )
        cache[key] = segments
        return segments
    return cached_all_threshold_segments(entry, labels, args, threshold=args.answer_threshold)


def execute_mode(
    item: QAItem,
    entry: Mapping[str, Any],
    labels: Sequence[str],
    args: argparse.Namespace,
    *,
    mode: str,
) -> tuple[str, bool, tuple[Segment, ...], str, float]:
    probs = entry["probs"].float()
    duration = float(entry["duration_seconds"])
    context_events = cached_all_threshold_segments(entry, labels, args, threshold=max(args.answer_threshold, 0.80))

    if mode == "oracle_both":
        anchor = oracle_segment(item.gold_anchor_interval, item.anchor_label, labels, source="oracle_anchor")
        if item.no_evidence:
            return "no_evidence", True, (anchor,) if anchor else (), "oracle_no_answer", 1.0
        answer = oracle_segment(item.gold_answer_interval, item.answer_label, labels, source="oracle_answer")
        evidence = tuple(seg for seg in (anchor, answer) if seg is not None)
        return item.answer, False, evidence, "oracle_both", 1.0

    if mode == "pred_labels_oracle_ts":
        anchor = predicted_label_at_oracle_time(
            probs,
            labels,
            gold_interval=item.gold_anchor_interval,
            duration=duration,
            source="pred_label_oracle_anchor_time",
        )
        if anchor is None or anchor.label != item.anchor_label:
            return "no_evidence", True, (), "anchor_wrong_class_at_oracle_time", float(anchor.confidence if anchor else 0.0)
        if item.no_evidence:
            return "no_evidence", True, (anchor,), "oracle_no_answer", anchor.confidence
        answer = predicted_label_at_oracle_time(
            probs,
            labels,
            gold_interval=item.gold_answer_interval,
            duration=duration,
            source="pred_label_oracle_answer_time",
        )
        if answer is None:
            return "no_evidence", True, (anchor,), "answer_missing_at_oracle_time", anchor.confidence
        return answer.label, False, (anchor, answer), "pred_labels_oracle_ts", anchor.confidence + answer.confidence

    if mode == "oracle_anchor":
        anchor = oracle_segment(item.gold_anchor_interval, item.anchor_label, labels, source="oracle_anchor")
        anchors = [anchor] if anchor else []
        answers = answer_segments_for_item(item, entry, labels, args)
    elif mode == "oracle_answer":
        anchors = select_anchor_ordinal(item, anchor_segments_for_item(item, entry, labels, args))
        answer = oracle_segment(item.gold_answer_interval, item.answer_label, labels, source="oracle_answer")
        answers = [answer] if answer else []
    elif mode == "oracle_labels_pred_ts":
        anchors = select_anchor_ordinal(item, anchor_segments_for_item(item, entry, labels, args))
        answers = answer_segments_for_item(item, entry, labels, args, restrict_to_gold_label=True)
    elif mode == "predicted_pair":
        anchors = select_anchor_ordinal(item, anchor_segments_for_item(item, entry, labels, args))
        answers = answer_segments_for_item(item, entry, labels, args)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    if item.no_evidence and mode in {"oracle_answer", "oracle_labels_pred_ts"}:
        if anchors:
            return "no_evidence", True, (anchors[min(item.anchor_ordinal - 1, len(anchors) - 1)],), "oracle_no_answer", 1.0
        return "no_evidence", True, (), "anchor_missing", float("-inf")

    return choose_pair(
        item,
        anchors,
        answers,
        context_events=context_events,
        distance_penalty=args.distance_penalty,
        intervening_penalty=args.intervening_penalty,
        pair_min_score=args.pair_min_score,
    )


def run_eval(
    qa_items: Sequence[QAItem],
    frame_payload: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = list(frame_payload["labels"])
    scenes = frame_payload["scenes"]
    rows: list[dict[str, Any]] = []
    for item in qa_items:
        entry = scenes.get(item.scene_id)
        if entry is None:
            continue
        answer, pred_noev, evidence_segments, reason, pair_score = execute_mode(
            item,
            entry,
            labels,
            args,
            mode=mode,
        )
        pred_intervals = tuple((seg.onset_seconds, seg.offset_seconds) for seg in evidence_segments if not item.no_evidence)
        # For no-evidence, this metric is not comparable to positive evidence IoU.
        evidence_iou = 1.0 if item.no_evidence and pred_noev else intervals_iou(item.gold_evidence_intervals, pred_intervals)
        answer_correct = answer == item.answer and pred_noev == item.no_evidence
        rows.append(
            {
                "mode": mode,
                "item_id": item.item_id,
                "scene_id": item.scene_id,
                "source_route": item.source_route,
                "relation": item.relation,
                "question": item.question,
                "gold_answer": item.answer,
                "gold_no_evidence": item.no_evidence,
                "gold_anchor_label": item.anchor_label,
                "gold_anchor_ordinal": item.anchor_ordinal,
                "gold_answer_label": item.answer_label,
                "gold_evidence_intervals": [list(x) for x in item.gold_evidence_intervals],
                "pred_answer": answer,
                "pred_no_evidence": pred_noev,
                "pred_evidence_intervals": [list(x) for x in pred_intervals],
                "pred_evidence_events": [seg.to_dict() for seg in evidence_segments],
                "pair_score": pair_score,
                "answer_correct": answer_correct,
                "evidence_iou_↑": evidence_iou,
                "answer_and_evidence_iou030_correct": (
                    answer_correct and (item.no_evidence or evidence_iou >= args.evidence_iou_threshold)
                ),
                "reason": reason,
            }
        )
    return summarize_rows(rows, args.evidence_iou_threshold), rows


def compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "items": metrics["items"],
        "accuracy_↑": metrics["accuracy_↑"],
        "answerable_accuracy_↑": metrics["answerable_accuracy_↑"],
        "no_evidence_accuracy_↑": metrics["no_evidence_accuracy_↑"],
        "mean_evidence_iou_answerable_↑": metrics["mean_evidence_iou_answerable_↑"],
        "median_evidence_iou_answerable_↑": metrics["median_evidence_iou_answerable_↑"],
        "answer_and_evidence_iou030_accuracy_↑": metrics["answer_and_evidence_iou030_accuracy_↑"],
        "predicted_no_evidence_rate": metrics["predicted_no_evidence_rate"],
        "wrong_reasons": metrics["wrong_reasons"],
    }


def run_sweep(
    qa_items: Sequence[QAItem],
    frame_payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    original_answer_threshold = args.answer_threshold
    original_pair_min = args.pair_min_score
    rows: list[dict[str, Any]] = []
    for answer_threshold in [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        for pair_min in [-0.20, 0.0, 0.20, 0.40, 0.60]:
            args.answer_threshold = answer_threshold
            args.pair_min_score = pair_min
            metrics, _ = run_eval(qa_items, frame_payload, args, mode="predicted_pair")
            rows.append(
                {
                    "answer_threshold": answer_threshold,
                    "pair_min_score": pair_min,
                    **{k: v for k, v in compact_metrics(metrics).items() if k != "wrong_reasons"},
                }
            )
    args.answer_threshold = original_answer_threshold
    args.pair_min_score = original_pair_min
    rows.sort(
        key=lambda row: (
            float(row["answerable_accuracy_↑"]),
            float(row["answer_and_evidence_iou030_accuracy_↑"]),
            float(row["no_evidence_accuracy_↑"]),
        ),
        reverse=True,
    )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_jsonl(args.manifest.resolve())
    qa_items = build_qa_items(
        manifest_rows,
        max_scenes=args.max_scenes,
        max_answerable_per_scene=args.max_answerable_per_scene,
        max_no_evidence_per_scene=args.max_no_evidence_per_scene,
        min_gap_seconds=args.min_gap_seconds,
    )
    frame_payload = torch.load(args.frame_probs.resolve(), map_location="cpu", weights_only=False)

    mode_metrics: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for mode in args.modes:
        metrics, rows = run_eval(qa_items, frame_payload, args, mode=mode)
        mode_metrics[mode] = metrics
        all_rows.extend(rows)

    sweep_rows = run_sweep(qa_items, frame_payload, args) if args.sweep else []
    summary = {
        "format": "qces_detector_pair_retrieval_qa_v1",
        "frame_probs": str(args.frame_probs.resolve()),
        "manifest": str(args.manifest.resolve()),
        "settings": {
            "max_scenes": args.max_scenes,
            "max_answerable_per_scene": args.max_answerable_per_scene,
            "max_no_evidence_per_scene": args.max_no_evidence_per_scene,
            "anchor_top_k": args.anchor_top_k,
            "anchor_peak_ratio": args.anchor_peak_ratio,
            "anchor_min_peak_score": args.anchor_min_peak_score,
            "answer_threshold": args.answer_threshold,
            "pair_min_score": args.pair_min_score,
            "distance_penalty": args.distance_penalty,
            "intervening_penalty": args.intervening_penalty,
        },
        "metrics": mode_metrics,
        "compact_metrics": {mode: compact_metrics(metrics) for mode, metrics in mode_metrics.items()},
        "sweep": sweep_rows,
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "qa_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if sweep_rows:
        with (output_dir / "sweep.jsonl").open("w", encoding="utf-8") as handle:
            for row in sweep_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    md = [
        "# QCES detector pair-retrieval QA",
        "",
        "Question-conditioned anchor top-K retrieval + anchor-answer pair reranking.",
        "",
        "| mode | items | acc ↑ | answerable ↑ | no-evidence ↑ | evidence IoU ↑ | answer+IoU≥0.30 ↑ | pred noev rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, metrics in mode_metrics.items():
        md.append(
            f"| {mode} | {metrics['items']} | {metrics['accuracy_↑']:.4f} | "
            f"{metrics['answerable_accuracy_↑']:.4f} | {metrics['no_evidence_accuracy_↑']:.4f} | "
            f"{metrics['mean_evidence_iou_answerable_↑']:.4f} | "
            f"{metrics['answer_and_evidence_iou030_accuracy_↑']:.4f} | "
            f"{metrics['predicted_no_evidence_rate']:.4f} |"
        )
    if sweep_rows:
        md.extend(
            [
                "",
                "## Top predicted_pair sweep rows",
                "",
                "| ans thr | pair min | acc ↑ | answerable ↑ | no-evidence ↑ | evidence IoU ↑ | answer+IoU ↑ |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sweep_rows[:10]:
            md.append(
                f"| {row['answer_threshold']:.2f} | {row['pair_min_score']:.2f} | "
                f"{row['accuracy_↑']:.4f} | {row['answerable_accuracy_↑']:.4f} | "
                f"{row['no_evidence_accuracy_↑']:.4f} | {row['mean_evidence_iou_answerable_↑']:.4f} | "
                f"{row['answer_and_evidence_iou030_accuracy_↑']:.4f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary["compact_metrics"], ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
