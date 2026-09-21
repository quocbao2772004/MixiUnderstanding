#!/usr/bin/env python3
"""Stratify V4 semantic accuracy by acoustic overlap and target/interference ratio."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf
import torch
from scipy.optimize import linear_sum_assignment

from mixi_understanding.qces.relational_event_slots_v1 import interval_iou_matrix
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _sha256_file
from mixi_understanding.scripts.train_qces_v4_slot_semantic_head_v1 import (
    NONE_LABEL,
    ResidualSlotSemanticHead,
)


FORMAT = "qces_v4_overlap_semantic_bottleneck_audit_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    semantic = base / "v4_slot_semantic_head_v1"
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_semantic_sufficient_v4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=semantic / "frozen_slots_dev.pt")
    parser.add_argument("--semantic-checkpoint", type=Path, default=semantic / "v4_slot_semantic_head_v1_best.pt")
    parser.add_argument("--manifest", type=Path, default=data / "detector_scene_manifest_overlap_dev.jsonl")
    parser.add_argument("--output", type=Path, default=semantic / "overlap_semantic_bottleneck_audit_v1.json")
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[str(row["scene_id"])] = row
    return result


@torch.inference_mode()
def learned_logits(cache: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> torch.Tensor:
    model = ResidualSlotSemanticHead(
        int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]),
        NONE_LABEL, float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    values = cache["slot_input"]
    return torch.cat([model(values[i : i + 256]).cpu() for i in range(0, len(values), 256)])


def overlap_ratio(target: Mapping[str, Any], events: list[Mapping[str, Any]]) -> tuple[float, int]:
    start, end = float(target["onset_seconds"]), float(target["offset_seconds"])
    duration = max(end - start, 1e-8)
    intervals = []
    boundaries = {start, end}
    for other in events:
        if other is target:
            continue
        left = max(start, float(other["onset_seconds"]))
        right = min(end, float(other["offset_seconds"]))
        if right > left:
            intervals.append((left, right))
            boundaries.update((left, right))
    intervals.sort()
    covered = 0.0
    if intervals:
        left, right = intervals[0]
        for next_left, next_right in intervals[1:]:
            if next_left <= right:
                right = max(right, next_right)
            else:
                covered += right - left
                left, right = next_left, next_right
        covered += right - left
    ordered = sorted(boundaries)
    maximum_concurrency = 1
    for left, right in zip(ordered, ordered[1:]):
        if right <= left:
            continue
        middle = (left + right) / 2
        concurrency = sum(
            float(event["onset_seconds"]) <= middle < float(event["offset_seconds"])
            for event in events
        )
        maximum_concurrency = max(maximum_concurrency, concurrency)
    return min(1.0, covered / duration), maximum_concurrency


def scene_sirs(events: list[Mapping[str, Any]]) -> list[float]:
    clips = []
    sample_rate = None
    for event in events:
        waveform, rate = sf.read(Path(str(event["component_path"])), dtype="float32", always_2d=False)
        value = torch.as_tensor(waveform).float()
        if value.ndim > 1:
            value = value.mean(dim=-1)
        sample_rate = int(rate) if sample_rate is None else sample_rate
        if int(rate) != sample_rate:
            raise ValueError("component sample rates differ inside scene")
        clips.append(value)
    starts = [int(round(float(event["onset_seconds"]) * sample_rate)) for event in events]
    length = max(start + clip.numel() for start, clip in zip(starts, clips))
    aligned = torch.zeros(len(events), length)
    for index, (start, clip) in enumerate(zip(starts, clips)):
        end = min(length, start + clip.numel())
        aligned[index, start:end] = clip[: end - start]
    result = []
    for index, event in enumerate(events):
        start = max(0, min(length - 1, int(round(float(event["onset_seconds"]) * sample_rate))))
        end = max(start + 1, min(length, int(round(float(event["offset_seconds"]) * sample_rate))))
        target = aligned[index, start:end]
        interference = aligned[:, start:end].sum(dim=0) - target
        target_power = target.square().mean()
        interference_power = interference.square().mean()
        sir = 10.0 * torch.log10((target_power + 1e-12) / (interference_power + 1e-12))
        result.append(float(sir.clamp(-80, 80)))
    return result


def overlap_bin(value: float) -> str:
    if value <= 1e-8:
        return "0_no_overlap"
    if value <= 0.25:
        return "1_(0,0.25]"
    if value <= 0.50:
        return "2_(0.25,0.50]"
    if value <= 0.75:
        return "3_(0.50,0.75]"
    return "4_(0.75,1.00]"


def sir_bin(value: float) -> str:
    if value < -10:
        return "0_<-10dB"
    if value < 0:
        return "1_[-10,0)dB"
    if value < 10:
        return "2_[0,10)dB"
    return "3_>=10dB"


def duration_bin(value: float) -> str:
    if value < 0.5:
        return "0_<0.5s"
    if value < 1.0:
        return "1_[0.5,1)s"
    if value < 2.0:
        return "2_[1,2)s"
    return "3_>=2s"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    localized = [row for row in rows if row["slot_rank"] is not None]
    return {
        "events": count,
        "oracle_gold_span_r1": {
            "top1_accuracy_\u2191": sum(row["oracle_rank"] <= 1 for row in rows) / max(count, 1),
            "top5_accuracy_\u2191": sum(row["oracle_rank"] <= 5 for row in rows) / max(count, 1),
            "top20_accuracy_\u2191": sum(row["oracle_rank"] <= 20 for row in rows) / max(count, 1),
        },
        "predicted_slot": {
            "localization_recall_iou50_\u2191": len(localized) / max(count, 1),
            "top1_given_iou50_\u2191": sum(row["slot_rank"] <= 1 for row in localized) / max(len(localized), 1),
            "top5_given_iou50_\u2191": sum(row["slot_rank"] <= 5 for row in localized) / max(len(localized), 1),
            "top20_given_iou50_\u2191": sum(row["slot_rank"] <= 20 for row in localized) / max(len(localized), 1),
            "joint_top1_recall_iou50_\u2191": sum(row["slot_rank"] <= 1 for row in localized) / max(count, 1),
        },
    }


def main() -> None:
    args = parse_args()
    cache_path = args.cache.resolve()
    checkpoint_path = args.semantic_checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    logits = learned_logits(cache, checkpoint)
    manifest = read_manifest(manifest_path)
    threshold = float(checkpoint["objectness_threshold"])
    rows: list[dict[str, Any]] = []
    for scene_index, scene_id_raw in enumerate(cache["scene_id"]):
        scene_id = str(scene_id_raw)
        events = manifest[scene_id]["events"]
        gold_labels = cache["gold_labels"][scene_index].long()
        gold_intervals = cache["gold_intervals"][scene_index].float()
        if len(events) != len(gold_labels):
            raise ValueError(f"gold event count mismatch: {scene_id}")
        for event, label, interval in zip(events, gold_labels, gold_intervals):
            if int(event["label_id"]) != int(label):
                raise ValueError(f"gold label order mismatch: {scene_id}")
            expected = torch.tensor([event["onset_seconds"] / 10.0, event["offset_seconds"] / 10.0])
            if not torch.allclose(interval, expected, atol=1e-4):
                raise ValueError(f"gold interval order mismatch: {scene_id}")
        sirs = scene_sirs(events)
        keep = cache["objectness"][scene_index].float() >= threshold
        predicted_intervals = cache["intervals"][scene_index].float()[keep]
        predicted_scores = logits[scene_index][keep].float()
        semantic_keep = predicted_scores.argmax(dim=-1) != NONE_LABEL
        predicted_intervals = predicted_intervals[semantic_keep]
        predicted_scores = predicted_scores[semantic_keep, :NONE_LABEL]
        ranks_by_gold: dict[int, int] = {}
        if predicted_intervals.numel():
            iou = interval_iou_matrix(predicted_intervals, gold_intervals)
            prediction_indices, gold_indices = linear_sum_assignment((1.0 - iou).numpy())
            for pred_index, gold_index in zip(prediction_indices, gold_indices):
                if float(iou[pred_index, gold_index]) < args.iou_threshold:
                    continue
                order = predicted_scores[pred_index].argsort(descending=True)
                rank = int(torch.where(order == int(gold_labels[gold_index]))[0].item()) + 1
                ranks_by_gold[int(gold_index)] = rank
        oracle_ranks = cache["oracle_gold_span_ranks"][scene_index]
        for gold_index, event in enumerate(events):
            ratio, concurrency = overlap_ratio(event, events)
            duration = float(event["offset_seconds"]) - float(event["onset_seconds"])
            rows.append({
                "scene_id": scene_id,
                "event_id": event["event_id"],
                "label_id": int(event["label_id"]),
                "overlap_ratio": ratio,
                "max_concurrency": concurrency,
                "sir_db": sirs[gold_index],
                "duration_seconds": duration,
                "oracle_rank": int(oracle_ranks[gold_index]),
                "slot_rank": ranks_by_gold.get(gold_index),
            })
        if (scene_index + 1) % 100 == 0:
            print(f"audit {scene_index + 1}/{len(cache['scene_id'])}", flush=True)

    strata: dict[str, Any] = {}
    definitions = {
        "overlap_ratio": lambda row: overlap_bin(row["overlap_ratio"]),
        "target_to_interference_ratio": lambda row: sir_bin(row["sir_db"]),
        "maximum_concurrency": lambda row: str(row["max_concurrency"]),
        "duration": lambda row: duration_bin(row["duration_seconds"]),
    }
    for name, key_fn in definitions.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)
        strata[name] = {key: summarize(groups[key]) for key in sorted(groups)}

    # V4 deliberately gives every event some overlap, so an exactly-zero
    # reference stratum is empty by construction.  Use the predeclared light
    # overlap range (<=25%) as the intelligible reference condition.
    light_overlap = [row for row in rows if row["overlap_ratio"] <= 0.25]
    heavy_overlap = [row for row in rows if row["overlap_ratio"] > 0.75]
    payload = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "paper_eligible": False,
        "purpose": "architecture decision audit on source-disjoint development data",
        "qa_question_or_answer_used_as_input": False,
        "all": summarize(rows),
        "strata": strata,
        "diagnostic_gate": {
            "light_overlap_le_0_25_oracle_top1_ge_0_60": summarize(light_overlap)["oracle_gold_span_r1"]["top1_accuracy_\u2191"] >= 0.60 if light_overlap else False,
            "heavy_overlap_oracle_top1_is_0_15_lower_than_light_overlap": (
                summarize(heavy_overlap)["oracle_gold_span_r1"]["top1_accuracy_\u2191"]
                <= summarize(light_overlap)["oracle_gold_span_r1"]["top1_accuracy_\u2191"] - 0.15
            ) if light_overlap and heavy_overlap else False,
        },
        "decision_rule": "frequency_aware_demixing_if_both_diagnostic_gates_pass_else_stronger_general_semantic_encoder",
        "rows": rows,
        "inputs": {
            "cache_sha256": _sha256_file(cache_path),
            "semantic_checkpoint_sha256": _sha256_file(checkpoint_path),
            "manifest_sha256": _sha256_file(manifest_path),
        },
    }
    gates = payload["diagnostic_gate"]
    payload["decision"] = (
        "build_frequency_aware_source_slot_encoder"
        if all(gates.values())
        else "semantic_backbone_is_weak_beyond_overlap_use_stronger_encoder"
    )
    _atomic_json(payload, args.output.resolve())
    print(json.dumps({
        "complete": True, "all": payload["all"], "strata": payload["strata"],
        "diagnostic_gate": gates, "decision": payload["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
