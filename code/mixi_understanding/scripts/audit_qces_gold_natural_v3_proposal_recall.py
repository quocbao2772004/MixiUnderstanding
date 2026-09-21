#!/usr/bin/env python3
"""Audit deployable event-proposal recall on Gold-natural v3 mixtures.

The audit deliberately runs before any QA ranker is trained.  It answers three
separate questions using only frozen detector scores:

1. is the gold event label present near the top of the scene inventory?
2. can a label-preserving multi-decoder union recover its temporal interval?
3. can the same finite candidate pool recover both members of an adjacent
   before/after pair?

Gold labels and intervals are used only after proposal generation for scoring.
They never select a threshold, a class, or a proposal.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch

from mixi_understanding.qces.benchmark_integrity import semantic_events
from mixi_understanding.scripts.audit_qces_detector_union_proposal_recall import (
    build_union_pool,
)
from mixi_understanding.scripts.evaluate_qces_detector_inventory_qa import interval_iou
from mixi_understanding.scripts.evaluate_qces_detector_pair_retrieval_qa import Segment
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    load_scene_list,
)


FORMAT = "qces_gold_natural_v3_proposal_recall_v1"
DEFAULT_BASE = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
)
DEFAULT_DATA = PROJECT_ROOT / "outputs/qces_full188_overlap_gold_natural_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_BASE / "dense_overlap_gold_natural_v3_dev/index.json",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=DEFAULT_DATA / "scene_ids_overlap_dev.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BASE / "proposal_recall_gold_natural_v3",
    )
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[5, 8, 20, 40, 80, 160, 320]
    )
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.95],
    )
    parser.add_argument(
        "--merge-gaps", type=float, nargs="+", default=[0.04, 0.12, 0.24]
    )
    parser.add_argument(
        "--hysteresis-high",
        type=float,
        nargs="+",
        default=[0.10, 0.30, 0.50, 0.70, 0.90],
    )
    parser.add_argument(
        "--hysteresis-low-ratio",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75],
    )
    parser.add_argument("--peak-top-k-per-label", type=int, default=3)
    parser.add_argument(
        "--peak-ratios", type=float, nargs="+", default=[0.25, 0.50, 0.75]
    )
    parser.add_argument("--peak-min-score", type=float, default=0.001)
    parser.add_argument("--min-duration", type=float, default=0.04)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-events-per-label", type=int, default=3)
    parser.add_argument("--max-proposals", type=int, default=320)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _proposal_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        thresholds=args.thresholds,
        merge_gaps=args.merge_gaps,
        min_duration=args.min_duration,
        hysteresis_high=args.hysteresis_high,
        hysteresis_low_ratio=args.hysteresis_low_ratio,
        peak_top_k_per_label=args.peak_top_k_per_label,
        peak_ratios=args.peak_ratios,
        peak_min_score=args.peak_min_score,
        nms_iou=args.nms_iou,
        max_proposals=args.max_proposals,
    )


def _event_interval(event: Mapping[str, Any]) -> tuple[float, float]:
    return float(event["onset_seconds"]), float(event["offset_seconds"])


def cap_events_per_label(
    pool: Sequence[Segment], max_events_per_label: int
) -> list[Segment]:
    """Preserve score order while preventing one false class from filling K."""

    if max_events_per_label <= 0:
        return list(pool)
    counts: Counter[str] = Counter()
    kept: list[Segment] = []
    for proposal in pool:
        if counts[proposal.label] >= max_events_per_label:
            continue
        counts[proposal.label] += 1
        kept.append(proposal)
    return kept


def build_balanced_union_pool(
    probs: torch.Tensor,
    labels: Sequence[str],
    *,
    duration: float,
    args: argparse.Namespace,
) -> list[Segment]:
    """Decode each class independently, then apply one global candidate budget."""

    merged: list[Segment] = []
    for label_id, label in enumerate(labels):
        proposal_args = _proposal_args(args)
        proposal_args.max_proposals = args.max_events_per_label
        local = build_union_pool(
            probs[:, label_id : label_id + 1],
            [label],
            duration=duration,
            args=proposal_args,
        )
        merged.extend(
            Segment(
                label=proposal.label,
                label_id=label_id,
                onset_seconds=proposal.onset_seconds,
                offset_seconds=proposal.offset_seconds,
                confidence=proposal.confidence,
                source=proposal.source,
            )
            for proposal in local
        )
    merged.sort(
        key=lambda proposal: (
            proposal.confidence,
            proposal.duration_seconds,
            -proposal.onset_seconds,
        ),
        reverse=True,
    )
    balanced = cap_events_per_label(merged, args.max_events_per_label)
    if args.max_proposals > 0:
        return balanced[: args.max_proposals]
    return balanced


def proposal_rank(
    pool: Sequence[Segment],
    *,
    label: str,
    interval: Sequence[float],
    iou_threshold: float,
) -> tuple[int | None, int | None, float]:
    """Return first same-label rank, first IoU-hit rank, and best same-label IoU."""

    label_rank: int | None = None
    segment_rank: int | None = None
    best_iou = 0.0
    for rank, proposal in enumerate(pool, start=1):
        if proposal.label != label:
            continue
        if label_rank is None:
            label_rank = rank
        iou = interval_iou(
            (proposal.onset_seconds, proposal.offset_seconds),
            (float(interval[0]), float(interval[1])),
        )
        best_iou = max(best_iou, iou)
        if segment_rank is None and iou >= iou_threshold:
            segment_rank = rank
    return label_rank, segment_rank, best_iou


def adjacent_pairs(events: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    """Return unambiguous chronological neighbours; tied onsets are excluded."""

    ordered = sorted(
        enumerate(events),
        key=lambda row: (
            float(row[1]["onset_seconds"]),
            float(row[1]["offset_seconds"]),
            str(row[1]["event_id"]),
        ),
    )
    pairs: list[tuple[int, int]] = []
    for left, right in zip(ordered, ordered[1:]):
        if math.isclose(
            float(left[1]["onset_seconds"]),
            float(right[1]["onset_seconds"]),
            abs_tol=1e-9,
        ):
            continue
        pairs.append((left[0], right[0]))
    return pairs


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def summarize(
    event_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    scene_rows: Sequence[Mapping[str, Any]],
    *,
    top_k: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "events": len(event_rows),
        "adjacent_pairs": len(pair_rows),
        "scenes": len(scene_rows),
        "mean_best_same_label_iou_↑": float(
            np.mean([float(row["best_same_label_iou"]) for row in event_rows])
        ),
        "mean_pool_size": float(
            np.mean([int(row["pool_size"]) for row in scene_rows])
        ),
    }
    for k in top_k:
        label_hits = sum(
            row["label_rank"] is not None and int(row["label_rank"]) <= k
            for row in event_rows
        )
        segment_hits = sum(
            row["segment_rank"] is not None and int(row["segment_rank"]) <= k
            for row in event_rows
        )
        pair_hits = sum(
            row["left_segment_rank"] is not None
            and int(row["left_segment_rank"]) <= k
            and row["right_segment_rank"] is not None
            and int(row["right_segment_rank"]) <= k
            for row in pair_rows
        )
        all_scene_hits = sum(
            all(rank is not None and int(rank) <= k for rank in row["segment_ranks"])
            for row in scene_rows
        )
        result[f"event_label_recall@{k}_↑"] = _fraction(label_hits, len(event_rows))
        result[f"event_label_iou030_recall@{k}_↑"] = _fraction(
            segment_hits, len(event_rows)
        )
        result[f"adjacent_pair_joint_iou030_recall@{k}_↑"] = _fraction(
            pair_hits, len(pair_rows)
        )
        result[f"all_events_scene_iou030_recall@{k}_↑"] = _fraction(
            all_scene_hits, len(scene_rows)
        )
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 < args.iou_threshold <= 1.0:
        raise ValueError("--iou-threshold must be in (0,1]")

    store = DenseFeatureStore([args.dense_index.resolve()], cache_size=8)
    labels = list(store.labels or [])
    scene_ids = load_scene_list(args.scene_list.resolve())
    if args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]
    missing = sorted(set(scene_ids) - store.scene_ids)
    if missing:
        raise ValueError(f"dense index misses scenes: {missing[:5]}")

    event_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    class_totals: Counter[str] = Counter()
    class_hits: dict[int, Counter[str]] = defaultdict(Counter)

    for scene_number, scene_id in enumerate(scene_ids, start=1):
        dense = store.get(scene_id)
        metadata = store.metadata(scene_id)
        valid_frames = int(dense["valid_frames"])
        logits = dense["logits"][:valid_frames].float()
        probs = logits.sigmoid()
        duration = float(metadata["duration_seconds"])
        events = semantic_events(metadata)
        pool = build_balanced_union_pool(
            probs,
            labels,
            duration=duration,
            args=args,
        )
        scene_event_rows: list[dict[str, Any]] = []
        for event in events:
            label = str(event["label"])
            label_rank, segment_rank, best_iou = proposal_rank(
                pool,
                label=label,
                interval=_event_interval(event),
                iou_threshold=args.iou_threshold,
            )
            row = {
                "event_id": str(event["event_id"]),
                "scene_id": scene_id,
                "label": label,
                "interval": list(_event_interval(event)),
                "pool_size": len(pool),
                "label_rank": label_rank,
                "segment_rank": segment_rank,
                "best_same_label_iou": best_iou,
            }
            scene_event_rows.append(row)
            event_rows.append(row)
            class_totals[label] += 1
            for k in args.top_k:
                if segment_rank is not None and segment_rank <= k:
                    class_hits[k][label] += 1

        for left_index, right_index in adjacent_pairs(events):
            left = scene_event_rows[left_index]
            right = scene_event_rows[right_index]
            pair_rows.append(
                {
                    "scene_id": scene_id,
                    "left_event_id": left["event_id"],
                    "right_event_id": right["event_id"],
                    "left_segment_rank": left["segment_rank"],
                    "right_segment_rank": right["segment_rank"],
                }
            )
        scene_rows.append(
            {
                "scene_id": scene_id,
                "pool_size": len(pool),
                "segment_ranks": [row["segment_rank"] for row in scene_event_rows],
            }
        )
        if scene_number % 50 == 0 or scene_number == len(scene_ids):
            print(f"proposal audit: {scene_number}/{len(scene_ids)} scenes", flush=True)

    metrics = summarize(
        event_rows,
        pair_rows,
        scene_rows,
        top_k=args.top_k,
    )
    per_class = {
        label: {
            "events": int(total),
            **{
                f"event_label_iou030_recall@{k}_↑": _fraction(
                    class_hits[k][label], total
                )
                for k in args.top_k
            },
        }
        for label, total in sorted(class_totals.items())
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "answer_label_used_as_model_input": False,
        "gold_used_only_for_post_inference_scoring": True,
        "inputs": {
            "dense_index": str(args.dense_index.resolve()),
            "scene_list": str(args.scene_list.resolve()),
        },
        "settings": {
            "top_k": args.top_k,
            "iou_threshold": args.iou_threshold,
            "thresholds": args.thresholds,
            "merge_gaps": args.merge_gaps,
            "hysteresis_high": args.hysteresis_high,
            "hysteresis_low_ratio": args.hysteresis_low_ratio,
            "peak_top_k_per_label": args.peak_top_k_per_label,
            "peak_ratios": args.peak_ratios,
            "max_events_per_label": args.max_events_per_label,
            "max_proposals": args.max_proposals,
        },
        "metrics": metrics,
        "per_class": per_class,
    }
    _atomic_json(receipt, output_dir / "report.json")
    with (output_dir / "event_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in event_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    lines = [
        "# Gold-natural v3 proposal recall",
        "",
        "Gold labels/timestamps are used only for post-inference scoring.",
        "",
        "| K | label recall ↑ | label+IoU≥0.30 ↑ | adjacent joint ↑ | all scene events ↑ |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in args.top_k:
        lines.append(
            f"| {k} | {metrics[f'event_label_recall@{k}_↑']:.4f} | "
            f"{metrics[f'event_label_iou030_recall@{k}_↑']:.4f} | "
            f"{metrics[f'adjacent_pair_joint_iou030_recall@{k}_↑']:.4f} | "
            f"{metrics[f'all_events_scene_iou030_recall@{k}_↑']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"- scenes: {metrics['scenes']}",
            f"- events: {metrics['events']}",
            f"- adjacent pairs: {metrics['adjacent_pairs']}",
            f"- mean pool size: {metrics['mean_pool_size']:.2f}",
            f"- mean best same-label IoU: {metrics['mean_best_same_label_iou_↑']:.4f}",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
