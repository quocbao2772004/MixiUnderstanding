#!/usr/bin/env python3
"""Analyze a frozen overlap-onset evaluation by controlled stress stratum."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from torch.utils.data import DataLoader

from mixi_understanding.qces.clean_evidence_scenes import _atomic_write_text
from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
)
from mixi_understanding.scripts.eval_qces_relational_event_slots_locked_test_v1 import (
    _case_analysis,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import load_bound_qa_manifest_v2
from mixi_understanding.scripts.train_qces_relational_event_slots_v1 import (
    CHECKPOINT_FORMAT,
    SceneSlotDataset,
    collate_scene_slots,
    collect_predictions,
    decode_hysteresis_event_slots,
)


FORMAT = "qces_overlap_onset_stress_analysis_v1"
FIXED_AUDIO_SECONDS = 10.0


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    data = PROJECT_ROOT / "outputs/qces_full191_overlap_onset_stress_test_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen-eval-receipt", type=Path,
        default=base / "relational_event_slots_v1_overlap_onset_stress_eval/receipt.json",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=base / "relational_event_slots_v1/relational_event_slots_v1_best.pt",
    )
    parser.add_argument(
        "--dense-index", type=Path,
        default=base / "dense_overlap_onset_stress_test_v1/index.json",
    )
    parser.add_argument("--scene-list", type=Path, default=data / "scene_ids_test.txt")
    parser.add_argument("--qa-manifest", type=Path, default=data / "qa_manifest_test.jsonl")
    parser.add_argument("--scene-manifest", type=Path, default=data / "scene_manifest_test.jsonl")
    parser.add_argument(
        "--output-dir", type=Path,
        default=base / "relational_event_slots_v1_overlap_onset_stress_analysis",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.resolve().open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0])))
    union = (
        float(left[1]) - float(left[0])
        + float(right[1]) - float(right[0])
        - intersection
    )
    return intersection / max(union, 1e-8)


def _target_pair(scene: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    events = list(scene["events"])
    protocol = scene["overlap_protocol"]
    anchor = next(event for event in events if event["stress_role"] == "anchor")
    if protocol["answerable"]:
        other = next(event for event in events if event["stress_role"] == "answer")
        return (
            [float(anchor["onset_seconds"]), float(anchor["offset_seconds"])],
            [float(other["onset_seconds"]), float(other["offset_seconds"])],
        )
    role = "overlap_partner" if int(protocol["target_concurrency"]) == 2 else "overlap_partner_2"
    other = next(event for event in events if event["stress_role"] == role)
    return (
        [float(other["onset_seconds"]), float(other["offset_seconds"])],
        [float(anchor["onset_seconds"]), float(anchor["offset_seconds"])],
    )


def _pair_resolution(
    decoded: Sequence[Mapping[str, float]],
    left_seconds: Sequence[float],
    right_seconds: Sequence[float],
) -> dict[str, Any]:
    slots = [
        [float(slot["start"]) * FIXED_AUDIO_SECONDS, float(slot["end"]) * FIXED_AUDIO_SECONDS]
        for slot in decoded
    ]
    if not slots:
        return {
            "pair_resolved_distinct_iou30": False,
            "pair_resolved_distinct_iou50": False,
            "same_best_slot_for_both": False,
            "same_best_slot_for_both_iou30": False,
        }
    left_ious = [_iou(slot, left_seconds) for slot in slots]
    right_ious = [_iou(slot, right_seconds) for slot in slots]
    best_left = max(range(len(slots)), key=left_ious.__getitem__)
    best_right = max(range(len(slots)), key=right_ious.__getitem__)

    def resolved(threshold: float) -> bool:
        return any(
            left_index != right_index
            and left_ious[left_index] >= threshold
            and right_ious[right_index] >= threshold
            for left_index in range(len(slots))
            for right_index in range(len(slots))
        )

    return {
        "pair_resolved_distinct_iou30": resolved(0.30),
        "pair_resolved_distinct_iou50": resolved(0.50),
        "same_best_slot_for_both": best_left == best_right,
        "same_best_slot_for_both_iou30": (
            best_left == best_right
            and left_ious[best_left] >= 0.30
            and right_ious[best_right] >= 0.30
        ),
    }


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if not row["no_evidence"]]
    negatives = [row for row in rows if row["no_evidence"]]

    def mean(name: str, values: Sequence[Mapping[str, Any]] = rows) -> float:
        return sum(float(row[name]) for row in values) / max(len(values), 1)

    positive_accuracy = mean("strict_evidence_correct_iou50", positives)
    negative_accuracy = mean("strict_evidence_correct_iou50", negatives)
    return {
        "count": len(rows),
        "answerable": len(positives),
        "no_evidence": len(negatives),
        "anchor_accuracy_iou50_↑": mean("anchor_iou50_correct"),
        "answerable_evidence_accuracy_iou50_↑": positive_accuracy,
        "no_evidence_evidence_accuracy_iou50_↑": negative_accuracy,
        "balanced_evidence_accuracy_iou50_↑": 0.5 * (positive_accuracy + negative_accuracy),
        "predicted_none_rate": mean("predicted_none"),
        "mean_gold_slot_count": mean("gold_slot_count"),
        "mean_predicted_slot_count": mean("predicted_slot_count"),
        "slot_count_underprediction_rate_↓": mean("slot_count_underpredicted"),
        "mean_slot_count_deficit_↓": mean("slot_count_deficit"),
        "target_pair_resolved_distinct_iou30_↑": mean("pair_resolved_distinct_iou30"),
        "target_pair_resolved_distinct_iou50_↑": mean("pair_resolved_distinct_iou50"),
        "target_pair_same_best_slot_iou30_rate_↓": mean("same_best_slot_for_both_iou30"),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_receipt_path = args.frozen_eval_receipt.resolve()
    frozen_receipt = _load_json(frozen_receipt_path)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha = _sha256_file(checkpoint_path)
    if str(frozen_receipt["checkpoint_sha256"]) != checkpoint_sha:
        raise ValueError("analysis checkpoint differs from frozen evaluation")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not a relational-event-slots v1 checkpoint")
    low = float(frozen_receipt["frozen_decoder"]["boundary_low_threshold"])
    high = float(frozen_receipt["frozen_decoder"]["presence_high_threshold"])
    if low != float(checkpoint["boundary_low_threshold"]) or high != float(checkpoint["presence_high_threshold"]):
        raise ValueError("frozen evaluation thresholds differ from checkpoint")

    store = DenseFeatureStore([args.dense_index.resolve()], cache_size=8)
    if list(checkpoint.get("labels") or []) != list(store.labels or []):
        raise ValueError("checkpoint and overlap dense label order differ")
    scene_ids = load_scene_list(args.scene_list.resolve())
    bound_qa = load_bound_qa_manifest_v2(
        args.qa_manifest.resolve(), allowed_scene_ids=scene_ids, store=store, max_ordinal=10
    )
    dataset = SceneSlotDataset(store, scene_ids, preload=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_scene_slots,
    )
    device = _device(args.device)
    model = RelationalEventSlotsV1(RelationalEventSlotsV1Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    predictions = collect_predictions(model, loader, device)
    label_to_id = {label: index for index, label in enumerate(store.labels or [])}
    case_rows, original_breakdown = _case_analysis(
        predictions, bound_qa, label_to_id, low=low, high=high
    )
    cases = {row["scene_id"]: row for row in case_rows}
    scenes = {str(row["scene_id"]): row for row in _load_jsonl(args.scene_manifest)}
    if set(cases) != set(scenes):
        raise ValueError("case/scene ids differ")

    augmented = []
    for scene_id in scene_ids:
        row = dict(cases[scene_id])
        scene = scenes[scene_id]
        protocol = scene["overlap_protocol"]
        decoded = decode_hysteresis_event_slots(
            predictions[scene_id]["frame_event_probability"],
            boundary_low_threshold=low,
            presence_high_threshold=high,
        )
        left, right = _target_pair(scene)
        pair = _pair_resolution(decoded, left, right)
        gold_count = len(scene["events"])
        predicted_count = len(decoded)
        row.update(
            {
                "requested_overlap_fraction": float(protocol["requested_overlap_fraction"]),
                "realized_target_pair_overlap_fraction": float(protocol["realized_target_pair_overlap_fraction"]),
                "target_concurrency": int(protocol["target_concurrency"]),
                "requested_gain_delta_db": float(protocol["requested_answer_or_partner_gain_delta_db"]),
                "measured_other_over_anchor_snr_db": float(protocol["measured_other_over_anchor_snr_db"]),
                "gold_slot_count": gold_count,
                "predicted_slot_count": predicted_count,
                "slot_count_underpredicted": predicted_count < gold_count,
                "slot_count_deficit": max(0, gold_count - predicted_count),
                "anchor_iou50_correct": float(row["anchor_iou"]) >= 0.50,
                **pair,
            }
        )
        augmented.append(row)

    groups: dict[str, list[dict[str, Any]]] = {"overall": augmented}
    for answerability in (False, True):
        groups["no_evidence" if answerability else "answerable"] = [
            row for row in augmented if bool(row["no_evidence"]) == answerability
        ]
    for tier in sorted({row["requested_overlap_fraction"] for row in augmented}):
        groups[f"overlap_{tier:.2f}"] = [
            row for row in augmented if row["requested_overlap_fraction"] == tier
        ]
    for concurrency in sorted({row["target_concurrency"] for row in augmented}):
        groups[f"concurrency_{concurrency}"] = [
            row for row in augmented if row["target_concurrency"] == concurrency
        ]
    for gain in sorted({row["requested_gain_delta_db"] for row in augmented}):
        groups[f"gain_delta_{gain:+.0f}db"] = [
            row for row in augmented if row["requested_gain_delta_db"] == gain
        ]
    breakdown = {name: _summarize(rows) for name, rows in groups.items()}

    cases_path = output_dir / "cases.jsonl"
    _atomic_write_text(
        cases_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in augmented),
    )
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_hoc_diagnostic": True,
        "test_threshold_tuning": False,
        "answer_label_used_for_prediction_or_selection": False,
        "frozen_evaluation_receipt": str(frozen_receipt_path),
        "frozen_evaluation_receipt_sha256": _sha256_file(frozen_receipt_path),
        "checkpoint_sha256": checkpoint_sha,
        "frozen_decoder": {"boundary_low_threshold": low, "presence_high_threshold": high},
        "original_metrics": frozen_receipt["metrics"],
        "original_case_breakdown": original_breakdown,
        "stress_breakdown": breakdown,
        "cases": {"path": str(cases_path), "sha256": _sha256_file(cases_path)},
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
