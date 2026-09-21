#!/usr/bin/env python3
"""Evaluate a fixed frequency-aware aggregation of frozen BEATs grid tokens."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mixi_understanding.qces.relational_event_slots_v1 import interval_iou_matrix
from mixi_understanding.scripts.export_qces_beats_tf_grid_v1 import (
    FEATURE_DIM, FORMAT as GRID_FORMAT, FREQUENCY_PATCHES, TIME_PATCHES,
)
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _sha256_file


FORMAT = "qces_beats_tf_semantic_audit_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-index", type=Path, default=Path("/var/tmp/qces_v4_beats_tf_dev/index.json"))
    parser.add_argument("--slot-cache", type=Path, default=base / "v4_slot_semantic_head_v1/frozen_slots_dev.pt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--slot-objectness-threshold", type=float, default=0.90)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--output", type=Path, default=base / "v4_slot_semantic_head_v1/beats_tf_semantic_audit_v1.json")
    return parser.parse_args()


def load_grid(index_path: Path) -> tuple[list[str], torch.Tensor, Mapping[str, Any]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("format") != GRID_FORMAT or not index.get("complete"):
        raise ValueError("incomplete/incompatible TF-grid index")
    scene_ids: list[str] = []
    tensors = []
    for descriptor in index["shards"]:
        path = index_path.parent / descriptor["path"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scene_ids.extend(str(value) for value in payload["scene_ids"])
        tensors.append(payload["grid_features"])
    grid = torch.cat(tensors)
    expected = (len(scene_ids), TIME_PATCHES, FREQUENCY_PATCHES, FEATURE_DIM)
    if tuple(grid.shape) != expected:
        raise ValueError(f"grid shape mismatch: {tuple(grid.shape)} != {expected}")
    return scene_ids, grid, index


def temporal_overlap_weights(interval: torch.Tensor) -> torch.Tensor:
    start, end = float(interval[0]), float(interval[1])
    left = torch.arange(TIME_PATCHES, dtype=torch.float32) / TIME_PATCHES
    right = left + 1.0 / TIME_PATCHES
    weight = (torch.minimum(right, torch.tensor(end)) - torch.maximum(left, torch.tensor(start))).clamp_min(0)
    if float(weight.sum()) <= 0:
        center = max(0, min(TIME_PATCHES - 1, int(round(((start + end) / 2) * TIME_PATCHES - 0.5))))
        weight[center] = 1.0
    return weight / weight.sum()


def interval_scores(tf_logits: torch.Tensor, interval: torch.Tensor) -> torch.Tensor:
    # Fixed source-sensitive pooling: preserve the strongest compatible
    # frequency bands without a hard max, then average by exact patch overlap.
    frequency_log_mean_exp = torch.logsumexp(tf_logits.float(), dim=1) - math.log(FREQUENCY_PATCHES)
    return (frequency_log_mean_exp * temporal_overlap_weights(interval)[:, None]).sum(dim=0)


def rank(score: torch.Tensor, label: int) -> int:
    order = score.argsort(descending=True)
    return int(torch.where(order == int(label))[0].item()) + 1


def summarize(ranks: list[int]) -> dict[str, Any]:
    return {
        "events": len(ranks),
        "top1_accuracy_\u2191": sum(value <= 1 for value in ranks) / max(len(ranks), 1),
        "top5_accuracy_\u2191": sum(value <= 5 for value in ranks) / max(len(ranks), 1),
        "top20_accuracy_\u2191": sum(value <= 20 for value in ranks) / max(len(ranks), 1),
        "mean_rank_\u2193": sum(ranks) / max(len(ranks), 1),
    }


def main() -> None:
    args = parse_args()
    index_path = args.grid_index.resolve()
    cache_path = args.slot_cache.resolve()
    checkpoint_path = args.detector_checkpoint.resolve()
    scene_ids, grid, index = load_grid(index_path)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if scene_ids != [str(value) for value in cache["scene_id"]]:
        raise ValueError("TF-grid and frozen-slot scene order differ")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]
    weight = checkpoint["model_state_dict"]["strong_head.weight"].float()
    bias = checkpoint["model_state_dict"]["strong_head.bias"].float()
    if tuple(weight.shape) != (len(labels), FEATURE_DIM):
        raise ValueError("unexpected strong-head shape")

    oracle_ranks: list[int] = []
    slot_ranks: list[int] = []
    gold_count = localized_count = 0
    for scene_index in range(len(scene_ids)):
        tf_logits = F.linear(grid[scene_index].float(), weight, bias)
        gold_intervals = cache["gold_intervals"][scene_index].float()
        gold_labels = cache["gold_labels"][scene_index].long()
        gold_count += len(gold_labels)
        for interval, label in zip(gold_intervals, gold_labels):
            oracle_ranks.append(rank(interval_scores(tf_logits, interval), int(label)))

        keep = cache["objectness"][scene_index].float() >= args.slot_objectness_threshold
        predicted_intervals = cache["intervals"][scene_index].float()[keep]
        if predicted_intervals.numel() == 0:
            continue
        iou = interval_iou_matrix(predicted_intervals, gold_intervals)
        rows, columns = linear_sum_assignment((1.0 - iou).numpy())
        for pred_index, gold_index in zip(rows, columns):
            if float(iou[pred_index, gold_index]) < args.iou_threshold:
                continue
            localized_count += 1
            score = interval_scores(tf_logits, predicted_intervals[pred_index])
            slot_ranks.append(rank(score, int(gold_labels[gold_index])))
        if (scene_index + 1) % 100 == 0:
            print(f"audit {scene_index + 1}/{len(scene_ids)}", flush=True)

    oracle = summarize(oracle_ranks)
    slot = summarize(slot_ranks)
    slot.update({
        "gold_events": gold_count,
        "localization_recall_iou50_\u2191": localized_count / max(gold_count, 1),
        "joint_top1_recall_iou50_\u2191": sum(value <= 1 for value in slot_ranks) / max(gold_count, 1),
    })
    baselines = {
        "oracle_gold_span_top1": 0.2490211433046202,
        "predicted_slot_top1_given_iou50": 0.32616487455197135,
    }
    gates = {
        "oracle_top1_improves_by_ge_0_10": oracle["top1_accuracy_\u2191"] >= baselines["oracle_gold_span_top1"] + 0.10,
        "slot_top1_given_iou50_improves_by_ge_0_08": slot["top1_accuracy_\u2191"] >= baselines["predicted_slot_top1_given_iou50"] + 0.08,
    }
    payload = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "paper_eligible": False,
        "method": "frozen_BEATs_grid_then_fixed_frequency_log_mean_exp_and_exact_temporal_overlap_pooling",
        "learned_parameters_added": 0,
        "qa_question_or_answer_used_as_input": False,
        "baselines": baselines, "oracle_gold_span": oracle, "predicted_slots": slot,
        "success_gates": gates,
        "decision": "use_fixed_tf_semantic_scorer" if all(gates.values()) else "train_tf_source_slot_attention",
        "inputs": {
            "grid_index_sha256": _sha256_file(index_path),
            "grid_manifest_sha256": index["manifest_sha256"],
            "slot_cache_sha256": _sha256_file(cache_path),
            "detector_checkpoint_sha256": _sha256_file(checkpoint_path),
        },
    }
    _atomic_json(payload, args.output.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
