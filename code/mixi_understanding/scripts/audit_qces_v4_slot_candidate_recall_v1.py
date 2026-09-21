#!/usr/bin/env python3
"""Audit semantic shortlist recall on frozen V4 event slots.

This is a decision audit before any expensive separator reranking.  It measures
whether the correct event label is present in the top-K semantic candidates for
predicted slots that already localize a gold event at IoU >= 0.50.  Questions,
answers, and QA annotations are never used as model inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from scipy.optimize import linear_sum_assignment

from mixi_understanding.qces.relational_event_slots_v1 import interval_iou_matrix
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _sha256_file
from mixi_understanding.scripts.train_qces_v4_slot_semantic_head_v1 import (
    NONE_LABEL,
    ResidualSlotSemanticHead,
)


FORMAT = "qces_v4_slot_candidate_recall_audit_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1/v4_slot_semantic_head_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=base / "frozen_slots_dev.pt")
    parser.add_argument("--semantic-checkpoint", type=Path, default=base / "v4_slot_semantic_head_v1_best.pt")
    parser.add_argument("--output", type=Path, default=base / "candidate_recall_audit_v1.json")
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 40])
    return parser.parse_args()


@torch.inference_mode()
def semantic_logits(cache: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> torch.Tensor:
    model = ResidualSlotSemanticHead(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
        NONE_LABEL,
        float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    values = cache["slot_input"]
    parts = [model(values[start : start + 256]).cpu() for start in range(0, len(values), 256)]
    return torch.cat(parts, dim=0)


def audit_method(
    cache: Mapping[str, Any],
    logits: torch.Tensor,
    *,
    objectness_threshold: float,
    iou_threshold: float,
    ks: list[int],
    learned: bool,
) -> dict[str, Any]:
    ranks: list[int] = []
    gold_count = 0
    localized_count = 0
    predicted_count = 0
    for scene_index in range(int(logits.shape[0])):
        keep = cache["objectness"][scene_index].float() >= objectness_threshold
        intervals = cache["intervals"][scene_index].float()[keep]
        scores = logits[scene_index][keep].float()
        if learned:
            # NONE may suppress a slot, but is never a semantic candidate.
            semantic_keep = scores.argmax(dim=-1) != NONE_LABEL
            intervals = intervals[semantic_keep]
            scores = scores[semantic_keep, :NONE_LABEL]
        else:
            scores = scores[:, :NONE_LABEL]
        gold_intervals = cache["gold_intervals"][scene_index].float()
        gold_labels = cache["gold_labels"][scene_index].long()
        gold_count += int(gold_intervals.shape[0])
        predicted_count += int(intervals.shape[0])
        if intervals.numel() == 0:
            continue
        iou = interval_iou_matrix(intervals, gold_intervals)
        rows, columns = linear_sum_assignment((1.0 - iou).numpy())
        for row, column in zip(rows, columns):
            if float(iou[row, column]) < iou_threshold:
                continue
            localized_count += 1
            order = scores[row].argsort(descending=True)
            position = torch.where(order == int(gold_labels[column]))[0]
            ranks.append(int(position.item()) + 1)
    return {
        "gold_events": gold_count,
        "predicted_events": predicted_count,
        "localized_events_iou50": localized_count,
        "localization_recall_iou50": localized_count / max(gold_count, 1),
        "conditional_candidate_recall": {
            f"recall_at_{k}_given_iou50_\u2191": sum(rank <= k for rank in ranks) / max(len(ranks), 1)
            for k in ks
        },
        "end_to_end_candidate_ceiling": {
            f"joint_recall_at_{k}_iou50_\u2191": sum(rank <= k for rank in ranks) / max(gold_count, 1)
            for k in ks
        },
        "rank": {
            "mean_\u2193": sum(ranks) / max(len(ranks), 1),
            "median_\u2193": float(torch.tensor(ranks).median()) if ranks else None,
        },
    }


def main() -> None:
    args = parse_args()
    cache_path = args.cache.resolve()
    checkpoint_path = args.semantic_checkpoint.resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    learned_logits = semantic_logits(cache, checkpoint)
    r1_logits = torch.nn.functional.pad(cache["slot_input"][..., -NONE_LABEL:].float(), (0, 1))
    threshold = float(checkpoint["objectness_threshold"])
    methods = {
        "pooled_r1": audit_method(
            cache, r1_logits, objectness_threshold=threshold,
            iou_threshold=args.iou_threshold, ks=args.ks, learned=False,
        ),
        "learned_residual_head": audit_method(
            cache, learned_logits, objectness_threshold=threshold,
            iou_threshold=args.iou_threshold, ks=args.ks, learned=True,
        ),
    }
    learned20 = methods["learned_residual_head"]["conditional_candidate_recall"].get(
        "recall_at_20_given_iou50_\u2191", 0.0
    )
    decision = (
        "proceed_to_top20_separator_rerank_pilot"
        if float(learned20) >= 0.70
        else "shortlist_recall_too_low_for_separator_rerank"
    )
    payload = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "paper_eligible": False,
        "purpose": "pre-registered feasibility decision before separator reranking",
        "qa_question_or_answer_used_as_input": False,
        "iou_threshold": args.iou_threshold,
        "objectness_threshold": threshold,
        "ks": args.ks,
        "methods": methods,
        "success_gate": {"learned_recall_at_20_given_iou50_ge_0_70": float(learned20) >= 0.70},
        "decision": decision,
        "inputs": {
            "cache": str(cache_path),
            "cache_sha256": _sha256_file(cache_path),
            "semantic_checkpoint": str(checkpoint_path),
            "semantic_checkpoint_sha256": _sha256_file(checkpoint_path),
        },
    }
    _atomic_json(payload, args.output.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
