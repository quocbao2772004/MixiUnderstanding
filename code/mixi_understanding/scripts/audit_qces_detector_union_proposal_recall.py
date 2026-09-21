#!/usr/bin/env python3
"""Audit recall of a union proposal pool for QCES detector QA."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.evaluate_qces_detector_inventory_qa import (
    build_qa_items,
    interval_iou,
    read_jsonl,
    write_json,
)
from mixi_understanding.scripts.evaluate_qces_detector_pair_retrieval_qa import (
    Segment,
    smooth_scores,
    topk_label_segments,
)


DEFAULT_FRAME_PROBS = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200/partial_as_fuss_e4_unfreeze1_highthr_resume/val_frame_probs.pt"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/detector_manifest_val.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_detector_union_proposal_recall/multisource200_unfreeze1_val_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-probs", type=Path, default=DEFAULT_FRAME_PROBS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-answerable-per-scene", type=int, default=4)
    parser.add_argument("--max-no-evidence-per-scene", type=int, default=2)
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100, 160, 240, 320])
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--merge-gaps", type=float, nargs="+", default=[0.04, 0.12, 0.24])
    parser.add_argument("--min-duration", type=float, default=0.06)
    parser.add_argument("--hysteresis-high", type=float, nargs="+", default=[0.50, 0.70, 0.85])
    parser.add_argument("--hysteresis-low-ratio", type=float, nargs="+", default=[0.35, 0.50, 0.65])
    parser.add_argument("--peak-top-k-per-label", type=int, default=3)
    parser.add_argument("--peak-ratios", type=float, nargs="+", default=[0.25, 0.35, 0.50])
    parser.add_argument("--peak-min-score", type=float, default=0.02)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-proposals", type=int, default=320)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def frame_bounds(index: int, duration: float, time_steps: int) -> tuple[float, float]:
    frame_seconds = duration / max(time_steps, 1)
    return index * frame_seconds, (index + 1) * frame_seconds


def threshold_segments_for_label(
    scores: torch.Tensor,
    *,
    label: str,
    label_id: int,
    duration: float,
    threshold: float,
    merge_gap: float,
    min_duration: float,
    source: str,
) -> list[Segment]:
    values = smooth_scores(scores)
    flags = (values >= threshold).cpu().numpy().astype(bool).tolist()
    time_steps = int(values.numel())
    spans: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx))
            start = None
    out: list[Segment] = []
    for start_idx, end_idx in spans:
        onset, _ = frame_bounds(start_idx, duration, time_steps)
        _, offset = frame_bounds(end_idx - 1, duration, time_steps)
        conf = float(values[start_idx:end_idx].max().item())
        if out and onset <= out[-1].offset_seconds + merge_gap:
            prev = out[-1]
            out[-1] = Segment(
                label=label,
                label_id=label_id,
                onset_seconds=prev.onset_seconds,
                offset_seconds=max(prev.offset_seconds, offset),
                confidence=max(prev.confidence, conf),
                source=source,
            )
        else:
            out.append(
                Segment(
                    label=label,
                    label_id=label_id,
                    onset_seconds=onset,
                    offset_seconds=offset,
                    confidence=conf,
                    source=source,
                )
            )
    return [seg for seg in out if seg.duration_seconds >= min_duration]


def hysteresis_segments_for_label(
    scores: torch.Tensor,
    *,
    label: str,
    label_id: int,
    duration: float,
    high: float,
    low: float,
    merge_gap: float,
    min_duration: float,
) -> list[Segment]:
    values = smooth_scores(scores)
    time_steps = int(values.numel())
    high_flags = (values >= high).cpu().numpy().astype(bool).tolist()
    low_flags = (values >= low).cpu().numpy().astype(bool).tolist()
    peaks = [idx for idx, flag in enumerate(high_flags) if flag]
    spans: list[tuple[int, int]] = []
    used = [False] * time_steps
    for peak in peaks:
        if used[peak]:
            continue
        left = peak
        while left > 0 and low_flags[left - 1]:
            left -= 1
        right = peak + 1
        while right < time_steps and low_flags[right]:
            right += 1
        for idx in range(left, right):
            used[idx] = True
        spans.append((left, right))
    spans.sort()
    out: list[Segment] = []
    for start_idx, end_idx in spans:
        onset, _ = frame_bounds(start_idx, duration, time_steps)
        _, offset = frame_bounds(end_idx - 1, duration, time_steps)
        conf = float(values[start_idx:end_idx].max().item())
        if out and onset <= out[-1].offset_seconds + merge_gap:
            prev = out[-1]
            out[-1] = Segment(
                label=label,
                label_id=label_id,
                onset_seconds=prev.onset_seconds,
                offset_seconds=max(prev.offset_seconds, offset),
                confidence=max(prev.confidence, conf),
                source=f"hyst_{high:.2f}_{low:.2f}",
            )
        else:
            out.append(
                Segment(
                    label=label,
                    label_id=label_id,
                    onset_seconds=onset,
                    offset_seconds=offset,
                    confidence=conf,
                    source=f"hyst_{high:.2f}_{low:.2f}",
                )
            )
    return [seg for seg in out if seg.duration_seconds >= min_duration]


def segment_iou(left: Segment, right: Segment) -> float:
    return interval_iou((left.onset_seconds, left.offset_seconds), (right.onset_seconds, right.offset_seconds))


def dedup_segments(segments: Sequence[Segment], *, nms_iou: float, max_proposals: int) -> list[Segment]:
    ordered = sorted(
        segments,
        key=lambda seg: (
            seg.confidence,
            seg.duration_seconds,
            -seg.onset_seconds,
        ),
        reverse=True,
    )
    kept: list[Segment] = []
    for seg in ordered:
        duplicate = False
        for old in kept:
            if seg.label == old.label and segment_iou(seg, old) >= nms_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(seg)
        if max_proposals > 0 and len(kept) >= max_proposals:
            break
    return kept


def build_union_pool(
    probs: torch.Tensor,
    labels: Sequence[str],
    *,
    duration: float,
    args: argparse.Namespace,
) -> list[Segment]:
    segments: list[Segment] = []
    probs = probs.float()
    for label_id, label in enumerate(labels):
        scores = probs[:, label_id]
        for threshold in args.thresholds:
            for merge_gap in args.merge_gaps:
                segments.extend(
                    threshold_segments_for_label(
                        scores,
                        label=label,
                        label_id=label_id,
                        duration=duration,
                        threshold=threshold,
                        merge_gap=merge_gap,
                        min_duration=args.min_duration,
                        source=f"thr_{threshold:.2f}_mg_{merge_gap:.2f}",
                    )
                )
        for high in args.hysteresis_high:
            for low_ratio in args.hysteresis_low_ratio:
                low = high * low_ratio
                for merge_gap in args.merge_gaps:
                    segments.extend(
                        hysteresis_segments_for_label(
                            scores,
                            label=label,
                            label_id=label_id,
                            duration=duration,
                            high=high,
                            low=low,
                            merge_gap=merge_gap,
                            min_duration=args.min_duration,
                        )
                    )
        for peak_ratio in args.peak_ratios:
            segments.extend(
                topk_label_segments(
                    scores,
                    label=label,
                    label_id=label_id,
                    duration=duration,
                    top_k=args.peak_top_k_per_label,
                    peak_ratio=peak_ratio,
                    min_peak_score=args.peak_min_score,
                    min_duration=args.min_duration,
                    source=f"peak_ratio_{peak_ratio:.2f}",
                )
            )
    return dedup_segments(segments, nms_iou=args.nms_iou, max_proposals=args.max_proposals)


def gold_interval(pair: Sequence[float] | tuple[float, float] | None) -> tuple[float, float] | None:
    if pair is None or len(pair) != 2:
        return None
    return (float(pair[0]), float(pair[1]))


def rank_recall(
    pool: Sequence[Segment],
    *,
    label: str,
    interval: tuple[float, float],
    iou_threshold: float,
) -> tuple[int | None, int | None, float]:
    label_rank = None
    segment_rank = None
    best_iou = 0.0
    for idx, seg in enumerate(pool, start=1):
        if seg.label == label and label_rank is None:
            label_rank = idx
        if seg.label == label:
            iou = interval_iou((seg.onset_seconds, seg.offset_seconds), interval)
            best_iou = max(best_iou, iou)
            if segment_rank is None and iou >= iou_threshold:
                segment_rank = idx
    return label_rank, segment_rank, best_iou


def summarize(rows: list[dict[str, Any]], top_k_values: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "answerable_items": len(rows),
        "mean_pool_size": float(np.mean([row["pool_size"] for row in rows])) if rows else 0.0,
        "median_pool_size": float(np.median([row["pool_size"] for row in rows])) if rows else 0.0,
        "mean_anchor_best_iou": float(np.mean([row["anchor_best_iou"] for row in rows])) if rows else 0.0,
        "mean_answer_best_iou": float(np.mean([row["answer_best_iou"] for row in rows])) if rows else 0.0,
    }
    for k in top_k_values:
        out[f"anchor_label_recall@{k}_↑"] = sum(
            row["anchor_label_rank"] is not None and row["anchor_label_rank"] <= k for row in rows
        ) / max(len(rows), 1)
        out[f"anchor_segment_recall@{k}_iou030_↑"] = sum(
            row["anchor_segment_rank"] is not None and row["anchor_segment_rank"] <= k for row in rows
        ) / max(len(rows), 1)
        out[f"answer_label_recall@{k}_↑"] = sum(
            row["answer_label_rank"] is not None and row["answer_label_rank"] <= k for row in rows
        ) / max(len(rows), 1)
        out[f"answer_segment_recall@{k}_iou030_↑"] = sum(
            row["answer_segment_rank"] is not None and row["answer_segment_rank"] <= k for row in rows
        ) / max(len(rows), 1)
        out[f"joint_segment_recall@{k}_iou030_↑"] = sum(
            row["anchor_segment_rank"] is not None
            and row["anchor_segment_rank"] <= k
            and row["answer_segment_rank"] is not None
            and row["answer_segment_rank"] <= k
            for row in rows
        ) / max(len(rows), 1)
    return out


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
        min_gap_seconds=0.0,
    )
    answerable = [item for item in qa_items if not item.no_evidence and item.answer_label and item.gold_answer_interval]
    frame_payload = torch.load(args.frame_probs.resolve(), map_location="cpu", weights_only=False)
    labels = list(frame_payload["labels"])
    scenes: Mapping[str, Any] = frame_payload["scenes"]
    pool_cache: dict[str, list[Segment]] = {}

    rows: list[dict[str, Any]] = []
    for item in answerable:
        entry = scenes.get(item.scene_id)
        if entry is None:
            continue
        if item.scene_id not in pool_cache:
            pool_cache[item.scene_id] = build_union_pool(
                entry["probs"],
                labels,
                duration=float(entry["duration_seconds"]),
                args=args,
            )
        pool = pool_cache[item.scene_id]
        anchor_interval = gold_interval(item.gold_anchor_interval)
        answer_interval = gold_interval(item.gold_answer_interval)
        if anchor_interval is None or answer_interval is None:
            continue
        anchor_label_rank, anchor_segment_rank, anchor_best_iou = rank_recall(
            pool,
            label=item.anchor_label,
            interval=anchor_interval,
            iou_threshold=args.iou_threshold,
        )
        answer_label_rank, answer_segment_rank, answer_best_iou = rank_recall(
            pool,
            label=str(item.answer_label),
            interval=answer_interval,
            iou_threshold=args.iou_threshold,
        )
        rows.append(
            {
                "item_id": item.item_id,
                "scene_id": item.scene_id,
                "relation": item.relation,
                "question": item.question,
                "anchor_label": item.anchor_label,
                "answer_label": item.answer_label,
                "anchor_interval": list(anchor_interval),
                "answer_interval": list(answer_interval),
                "pool_size": len(pool),
                "anchor_label_rank": anchor_label_rank,
                "anchor_segment_rank": anchor_segment_rank,
                "anchor_best_iou": anchor_best_iou,
                "answer_label_rank": answer_label_rank,
                "answer_segment_rank": answer_segment_rank,
                "answer_best_iou": answer_best_iou,
                "top_candidates": [seg.to_dict() for seg in pool[:12]],
            }
        )

    summary = {
        "format": "qces_detector_union_proposal_recall_v1",
        "frame_probs": str(args.frame_probs.resolve()),
        "manifest": str(args.manifest.resolve()),
        "settings": {
            "thresholds": args.thresholds,
            "merge_gaps": args.merge_gaps,
            "hysteresis_high": args.hysteresis_high,
            "hysteresis_low_ratio": args.hysteresis_low_ratio,
            "peak_top_k_per_label": args.peak_top_k_per_label,
            "peak_ratios": args.peak_ratios,
            "nms_iou": args.nms_iou,
            "max_proposals": args.max_proposals,
            "top_k": args.top_k,
        },
        "metrics": summarize(rows, args.top_k),
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = summary["metrics"]
    md = [
        "# QCES union proposal recall audit",
        "",
        "| K | anchor label ↑ | anchor seg ↑ | answer label ↑ | answer seg ↑ | joint seg ↑ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in args.top_k:
        md.append(
            f"| {k} | {metrics.get(f'anchor_label_recall@{k}_↑', 0.0):.4f} | "
            f"{metrics.get(f'anchor_segment_recall@{k}_iou030_↑', 0.0):.4f} | "
            f"{metrics.get(f'answer_label_recall@{k}_↑', 0.0):.4f} | "
            f"{metrics.get(f'answer_segment_recall@{k}_iou030_↑', 0.0):.4f} | "
            f"{metrics.get(f'joint_segment_recall@{k}_iou030_↑', 0.0):.4f} |"
        )
    md.extend(
        [
            "",
            f"- answerable items: {metrics['answerable_items']}",
            f"- mean pool size: {metrics['mean_pool_size']:.2f}",
            f"- median pool size: {metrics['median_pool_size']:.2f}",
            f"- mean anchor best IoU: {metrics['mean_anchor_best_iou']:.4f}",
            f"- mean answer best IoU: {metrics['mean_answer_best_iou']:.4f}",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
