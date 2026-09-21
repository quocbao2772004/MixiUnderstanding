#!/usr/bin/env python3
"""Audit whether the gold answer event exists in detector candidate pools."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
    all_threshold_segments,
    anchor_segments_for_item,
    answer_segments_for_item,
    select_anchor_ordinal,
)


DEFAULT_FRAME_PROBS = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200/partial_as_fuss_e4_unfreeze1_highthr_resume/val_frame_probs.pt"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/detector_manifest_val.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_detector_candidate_recall/multisource200_unfreeze1_val_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-probs", type=Path, default=DEFAULT_FRAME_PROBS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-answerable-per-scene", type=int, default=4)
    parser.add_argument("--max-no-evidence-per-scene", type=int, default=2)
    parser.add_argument("--answer-thresholds", type=float, nargs="+", default=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50, 100, 160])
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--anchor-top-k", type=int, default=5)
    parser.add_argument("--anchor-peak-ratio", type=float, default=0.35)
    parser.add_argument("--anchor-min-peak-score", type=float, default=0.02)
    parser.add_argument("--answer-top-k-per-gold-label", type=int, default=5)
    parser.add_argument("--answer-min-duration", type=float, default=0.08)
    parser.add_argument("--merge-gap", type=float, default=0.12)
    parser.add_argument("--max-answer-proposals", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def interval_from_pair(pair: Sequence[float] | tuple[float, float] | None) -> tuple[float, float] | None:
    if pair is None:
        return None
    if len(pair) != 2:
        return None
    return (float(pair[0]), float(pair[1]))


def segment_matches_gold(seg: Segment, gold_label: str, gold_interval: tuple[float, float], iou_threshold: float) -> bool:
    return seg.label == gold_label and interval_iou(
        (seg.onset_seconds, seg.offset_seconds),
        gold_interval,
    ) >= iou_threshold


def rank_gold_answer(
    candidates: Sequence[Segment],
    *,
    gold_label: str,
    gold_interval: tuple[float, float],
    iou_threshold: float,
) -> tuple[int | None, int | None, float]:
    label_rank: int | None = None
    segment_rank: int | None = None
    best_iou = 0.0
    for index, candidate in enumerate(candidates, start=1):
        if candidate.label == gold_label and label_rank is None:
            label_rank = index
        if candidate.label == gold_label:
            best_iou = max(
                best_iou,
                interval_iou((candidate.onset_seconds, candidate.offset_seconds), gold_interval),
            )
        if segment_matches_gold(candidate, gold_label, gold_interval, iou_threshold) and segment_rank is None:
            segment_rank = index
    return label_rank, segment_rank, best_iou


def anchor_found(
    item: Any,
    entry: Mapping[str, Any],
    labels: Sequence[str],
    args: argparse.Namespace,
) -> bool:
    anchors = select_anchor_ordinal(item, anchor_segments_for_item(item, entry, labels, args))
    if not anchors or item.gold_anchor_interval is None:
        return False
    anchor = anchors[0]
    return anchor.label == item.anchor_label and interval_iou(
        (anchor.onset_seconds, anchor.offset_seconds),
        item.gold_anchor_interval,
    ) >= args.iou_threshold


def summarize(rows: list[dict[str, Any]], top_k_values: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "answerable_items": len(rows),
        "anchor_recall_iou030_↑": sum(bool(r["anchor_found_iou030"]) for r in rows) / max(len(rows), 1),
        "mean_best_answer_iou_same_label_↑": float(np.mean([r["best_answer_iou_same_label"] for r in rows])) if rows else 0.0,
    }
    for k in top_k_values:
        out[f"answer_label_recall@{k}_↑"] = sum(
            r["answer_label_rank"] is not None and int(r["answer_label_rank"]) <= k for r in rows
        ) / max(len(rows), 1)
        out[f"answer_segment_recall@{k}_iou030_↑"] = sum(
            r["answer_segment_rank_iou030"] is not None and int(r["answer_segment_rank_iou030"]) <= k for r in rows
        ) / max(len(rows), 1)
        out[f"joint_anchor_answer_segment_recall@{k}_iou030_↑"] = sum(
            bool(r["anchor_found_iou030"])
            and r["answer_segment_rank_iou030"] is not None
            and int(r["answer_segment_rank_iou030"]) <= k
            for r in rows
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
    scenes = frame_payload["scenes"]

    threshold_summaries: dict[str, Any] = {}
    threshold_rows: dict[str, list[dict[str, Any]]] = {}
    candidate_cache: dict[tuple[str, float], list[Segment]] = {}
    anchor_cache: dict[str, bool] = {}
    for threshold in args.answer_thresholds:
        rows: list[dict[str, Any]] = []
        for item in answerable:
            entry = scenes.get(item.scene_id)
            if entry is None:
                continue
            candidate_key = (item.scene_id, round(float(threshold), 4))
            if candidate_key not in candidate_cache:
                candidate_cache[candidate_key] = all_threshold_segments(
                    entry["probs"].float(),
                    labels,
                    duration=float(entry["duration_seconds"]),
                    threshold=threshold,
                    min_duration=args.answer_min_duration,
                    merge_gap=args.merge_gap,
                    max_segments=max(args.max_answer_proposals, max(args.top_k)),
                )
            candidates = candidate_cache[candidate_key]
            candidates_by_conf = sorted(candidates, key=lambda seg: seg.confidence, reverse=True)
            gold_interval = interval_from_pair(item.gold_answer_interval)
            if gold_interval is None:
                continue
            label_rank, segment_rank, best_iou = rank_gold_answer(
                candidates_by_conf,
                gold_label=str(item.answer_label),
                gold_interval=gold_interval,
                iou_threshold=args.iou_threshold,
            )
            rows.append(
                {
                    "item_id": item.item_id,
                    "scene_id": item.scene_id,
                    "source_route": item.source_route,
                    "relation": item.relation,
                    "question": item.question,
                    "answer_label": item.answer_label,
                    "answer_interval": list(gold_interval),
                    "answer_threshold": threshold,
                    "num_candidates": len(candidates_by_conf),
                    "answer_label_rank": label_rank,
                    "answer_segment_rank_iou030": segment_rank,
                    "best_answer_iou_same_label": best_iou,
                    "anchor_found_iou030": anchor_cache.setdefault(
                        item.item_id,
                        anchor_found(item, entry, labels, args),
                    ),
                    "top_candidates": [seg.to_dict() for seg in candidates_by_conf[:10]],
                }
            )
        key = f"{threshold:.2f}"
        threshold_rows[key] = rows
        threshold_summaries[key] = summarize(rows, args.top_k)

    best_threshold = max(
        threshold_summaries,
        key=lambda key: (
            threshold_summaries[key].get("answer_segment_recall@20_iou030_↑", 0.0),
            threshold_summaries[key].get("answer_label_recall@20_↑", 0.0),
            threshold_summaries[key].get("answer_segment_recall@10_iou030_↑", 0.0),
        ),
    )
    summary = {
        "format": "qces_detector_candidate_recall_v1",
        "frame_probs": str(args.frame_probs.resolve()),
        "manifest": str(args.manifest.resolve()),
        "settings": {
            "top_k": args.top_k,
            "answer_thresholds": args.answer_thresholds,
            "iou_threshold": args.iou_threshold,
            "answer_min_duration": args.answer_min_duration,
            "merge_gap": args.merge_gap,
            "max_answer_proposals": args.max_answer_proposals,
        },
        "best_threshold_by_segment_recall@20": best_threshold,
        "threshold_summaries": threshold_summaries,
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "candidate_recall_rows.jsonl").open("w", encoding="utf-8") as handle:
        for threshold, rows in threshold_rows.items():
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    md = [
        "# QCES detector candidate recall audit",
        "",
        "Anchor recall is independent of answer threshold in this audit because anchor proposals are decoded by top-K over the anchor label, not by the answer-candidate threshold.",
        "",
        "| threshold | ans items | anchor recall ↑ | label@5 ↑ | label@20 ↑ | seg@5 ↑ | seg@20 ↑ | joint@20 ↑ | seg@50 ↑ | joint@50 ↑ | mean best IoU ↑ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for threshold in sorted(threshold_summaries, key=lambda x: float(x)):
        row = threshold_summaries[threshold]
        md.append(
            f"| {threshold} | {row['answerable_items']} | {row['anchor_recall_iou030_↑']:.4f} | "
            f"{row.get('answer_label_recall@5_↑', 0.0):.4f} | {row.get('answer_label_recall@20_↑', 0.0):.4f} | "
            f"{row.get('answer_segment_recall@5_iou030_↑', 0.0):.4f} | "
            f"{row.get('answer_segment_recall@20_iou030_↑', 0.0):.4f} | "
            f"{row.get('joint_anchor_answer_segment_recall@20_iou030_↑', 0.0):.4f} | "
            f"{row.get('answer_segment_recall@50_iou030_↑', 0.0):.4f} | "
            f"{row.get('joint_anchor_answer_segment_recall@50_iou030_↑', 0.0):.4f} | "
            f"{row['mean_best_answer_iou_same_label_↑']:.4f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
