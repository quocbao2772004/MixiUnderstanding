#!/usr/bin/env python3
"""Calibrate and lock a deployable decoder for polyphonic temporal slots.

The class-agnostic query branch proposes event intervals.  The question's
anchor label scores those proposals with the frozen detector; onset order then
selects the immediate before/after neighbour.  A single objectness threshold
controls both event retention and the NONE decision.  It is selected on the
complete overlap-v2 development set and then locked for all source-disjoint
diagnostic test sets.

This evaluator reports *answer-event localization*, not answer-label accuracy.
No answer label is read by proposal generation, ranking, or threshold fitting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
)
from mixi_understanding.scripts.train_qces_polyphonic_query_branch_v1 import (
    POLYPHONIC_CHECKPOINT_FORMAT,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_relational_event_slots_v1 import (
    SceneSlotDataset,
    _pool_anchor_score,
    collate_scene_slots,
    collect_predictions,
)


FORMAT = "qces_polyphonic_deployable_decoder_evaluation_v1"
FIXED_AUDIO_SECONDS = 10.0


@dataclass(frozen=True)
class QA:
    item_id: str
    scene_id: str
    anchor_label: str
    anchor_ordinal: int
    relation: str
    no_evidence: bool
    gold_anchor: tuple[float, float]
    gold_answer: tuple[float, float] | None
    answer_label_diagnostic_only: str | None
    question: str


@dataclass(frozen=True)
class TestSpec:
    name: str
    dense_index: Path
    scene_list: Path
    qa_manifest: Path


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    overlap_dev = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2"
    nonoverlap = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/multievent"
    repeated = PROJECT_ROOT / "outputs/qces_full191_repeated_ordinal_stress_test_v1"
    overlap = PROJECT_ROOT / "outputs/qces_full191_overlap_onset_stress_test_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=base / "polyphonic_query_branch_v2/polyphonic_query_branch_best.pt",
    )
    parser.add_argument("--dev-index", type=Path, default=base / "dense_overlap_query_dev_v2/index.json")
    parser.add_argument("--dev-scenes", type=Path, default=overlap_dev / "scene_ids_overlap_dev.txt")
    parser.add_argument(
        "--output-dir", type=Path,
        default=base / "polyphonic_deployable_decoder_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(tests=(
        TestSpec(
            "nonoverlap_locked_test", base / "dense_multi_test_v2/index.json",
            nonoverlap / "scene_ids_test.txt", nonoverlap / "qa_manifest_test.jsonl",
        ),
        TestSpec(
            "repeated_ordinal_stress", base / "dense_repeated_ordinal_stress_test_v2/index.json",
            repeated / "scene_ids_test.txt", repeated / "qa_manifest_test.jsonl",
        ),
        TestSpec(
            "overlap_onset_stress", base / "dense_overlap_onset_stress_test_v1/index.json",
            overlap / "scene_ids_test.txt", overlap / "qa_manifest_test.jsonl",
        ),
    ))
    return parser.parse_args()


def interval_iou(left: Sequence[float] | None, right: Sequence[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    intersection = max(0.0, min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0])))
    union = max(0.0, float(left[1]) - float(left[0])) + max(0.0, float(right[1]) - float(right[0])) - intersection
    return intersection / max(union, 1e-8)


def normalize_interval(value: Sequence[float] | None) -> tuple[float, float] | None:
    if value is None:
        return None
    return float(value[0]) / FIXED_AUDIO_SECONDS, float(value[1]) / FIXED_AUDIO_SECONDS


def dev_qa(store: DenseFeatureStore, scene_ids: Sequence[str]) -> list[QA]:
    rows: list[QA] = []
    for scene_id in scene_ids:
        events = sorted(
            store.metadata(scene_id)["gold_events"],
            key=lambda event: (float(event["onset_seconds"]), float(event["offset_seconds"]), str(event["event_id"])),
        )
        for position, anchor in enumerate(events):
            for relation, delta in (("before", -1), ("after", 1)):
                answer_position = position + delta
                answer = events[answer_position] if 0 <= answer_position < len(events) else None
                rows.append(QA(
                    item_id=f"{scene_id}:{relation}:{position}", scene_id=scene_id,
                    anchor_label=str(anchor["label"]), anchor_ordinal=1, relation=relation,
                    no_evidence=answer is None,
                    gold_anchor=normalize_interval((anchor["onset_seconds"], anchor["offset_seconds"])),
                    gold_answer=(None if answer is None else normalize_interval((answer["onset_seconds"], answer["offset_seconds"]))),
                    answer_label_diagnostic_only=None if answer is None else str(answer["label"]),
                    question=f"What sound has the immediate {relation} onset relative to {anchor['label']}?",
                ))
    return rows


def manifest_qa(path: Path, allowed_scenes: set[str]) -> list[QA]:
    rows = []
    with path.resolve().open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if str(raw["scene_id"]) not in allowed_scenes:
                continue
            rows.append(QA(
                item_id=str(raw["item_id"]), scene_id=str(raw["scene_id"]),
                anchor_label=str(raw["anchor_label"]), anchor_ordinal=int(raw.get("anchor_ordinal", 1)),
                relation=str(raw["relation"]), no_evidence=bool(raw["no_evidence"]),
                gold_anchor=normalize_interval(raw["gold_anchor_interval"]),
                gold_answer=normalize_interval(raw.get("gold_answer_interval")),
                answer_label_diagnostic_only=raw.get("answer_label"), question=str(raw["question"]),
            ))
    if not rows:
        raise ValueError(f"no QA records selected from {path}")
    return rows


def decode(
    record: Mapping[str, Any], qa: QA, label_to_id: Mapping[str, int], threshold: float
) -> dict[str, Any]:
    slots = [dict(slot) for slot in record["slots"] if float(slot["score"]) >= threshold]
    label_id = label_to_id[qa.anchor_label]
    for slot in slots:
        slot["anchor_score"] = _pool_anchor_score(
            record["detector_logits"], (float(slot["start"]), float(slot["end"])), label_id
        )
    if len(slots) < qa.anchor_ordinal:
        return {"anchor": None, "answer": None, "predicted_none": True, "retained_slots": len(slots)}
    likely = sorted(slots, key=lambda slot: float(slot["anchor_score"]), reverse=True)[:qa.anchor_ordinal]
    likely.sort(key=lambda slot: (float(slot["start"]), float(slot["end"])))
    anchor = likely[qa.anchor_ordinal - 1]
    ordered = sorted(slots, key=lambda slot: (float(slot["start"]), float(slot["end"])))
    anchor_position = next(i for i, slot in enumerate(ordered) if int(slot["slot_index"]) == int(anchor["slot_index"]))
    answer_position = anchor_position - 1 if qa.relation == "before" else anchor_position + 1
    answer = ordered[answer_position] if 0 <= answer_position < len(ordered) else None
    return {"anchor": anchor, "answer": answer, "predicted_none": answer is None, "retained_slots": len(slots)}


def evaluate(
    predictions: Mapping[str, Mapping[str, Any]], qas: Sequence[QA],
    label_to_id: Mapping[str, int], threshold: float, *, return_cases: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counts = defaultdict(int)
    cases = []
    retained = []
    for qa in qas:
        result = decode(predictions[qa.scene_id], qa, label_to_id, threshold)
        anchor = result["anchor"]
        answer = result["answer"]
        anchor_interval = None if anchor is None else (anchor["start"], anchor["end"])
        answer_interval = None if answer is None else (answer["start"], answer["end"])
        anchor_iou = interval_iou(anchor_interval, qa.gold_anchor)
        answer_iou = interval_iou(answer_interval, qa.gold_answer)
        predicted_none = bool(result["predicted_none"])
        retained.append(int(result["retained_slots"]))
        counts["total"] += 1
        counts["anchor30"] += anchor_iou >= 0.30
        counts["anchor50"] += anchor_iou >= 0.50
        counts["answerability"] += predicted_none == qa.no_evidence
        if qa.no_evidence:
            counts["negative"] += 1
            counts["negative_none"] += predicted_none
            counts["negative_strict30"] += predicted_none and anchor_iou >= 0.30
            counts["negative_strict50"] += predicted_none and anchor_iou >= 0.50
        else:
            counts["positive"] += 1
            counts["positive_has_answer"] += not predicted_none
            counts["answer30"] += answer_iou >= 0.30
            counts["answer50"] += answer_iou >= 0.50
            counts["joint30"] += anchor_iou >= 0.30 and answer_iou >= 0.30
            counts["joint50"] += anchor_iou >= 0.50 and answer_iou >= 0.50
        if return_cases:
            cases.append({
                "item_id": qa.item_id, "scene_id": qa.scene_id, "question": qa.question,
                "anchor_label": qa.anchor_label, "answer_label_diagnostic_only": qa.answer_label_diagnostic_only,
                "no_evidence": qa.no_evidence, "predicted_none": predicted_none,
                "anchor_iou": anchor_iou, "answer_iou": answer_iou,
                "retained_slots": int(result["retained_slots"]),
                "predicted_anchor": anchor_interval, "predicted_answer": answer_interval,
            })
    positive = max(counts["positive"], 1)
    negative = max(counts["negative"], 1)
    total = max(counts["total"], 1)
    pos50 = counts["answer50"] / positive
    neg50 = counts["negative_strict50"] / negative
    result = {
        "questions": counts["total"], "answerable_questions": counts["positive"],
        "no_evidence_questions": counts["negative"], "objectness_threshold": threshold,
        "mean_retained_slots": float(np.mean(retained)),
        "anchor_localization_iou30_↑": counts["anchor30"] / total,
        "anchor_localization_iou50_↑": counts["anchor50"] / total,
        "answerability_accuracy_↑": counts["answerability"] / total,
        "answerable_has_neighbor_rate": counts["positive_has_answer"] / positive,
        "answer_event_localization_iou30_↑": counts["answer30"] / positive,
        "answer_event_localization_iou50_↑": pos50,
        "joint_evidence_localization_iou30_↑": counts["joint30"] / positive,
        "joint_evidence_localization_iou50_↑": counts["joint50"] / positive,
        "no_evidence_none_accuracy_↑": counts["negative_none"] / negative,
        "no_evidence_strict_anchor_iou30_↑": counts["negative_strict30"] / negative,
        "no_evidence_strict_anchor_iou50_↑": neg50,
        "strict_balanced_accuracy_iou50_↑": 0.5 * (pos50 + neg50),
        "minimum_positive_negative_strict_iou50_↑": min(pos50, neg50),
        "answer_label_accuracy": None,
    }
    return result, cases


def collect(
    model: RelationalEventSlotsV1, store: DenseFeatureStore, scene_ids: Sequence[str],
    device: torch.device, batch_size: int,
) -> dict[str, dict[str, Any]]:
    dataset = SceneSlotDataset(store, scene_ids, preload=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=device.type == "cuda", collate_fn=collate_scene_slots,
    )
    return collect_predictions(model, loader, device)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("invalid batch-size")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != POLYPHONIC_CHECKPOINT_FORMAT:
        raise ValueError("checkpoint is not a polyphonic query branch")
    all_indexes = [args.dev_index.resolve()] + [spec.dense_index.resolve() for spec in args.tests]
    store = DenseFeatureStore(all_indexes, cache_size=32)
    if list(checkpoint["labels"]) != list(store.labels or []):
        raise ValueError("checkpoint/dense label order mismatch")
    label_to_id = {label: index for index, label in enumerate(store.labels or [])}
    device = _device(args.device)
    model = RelationalEventSlotsV1(RelationalEventSlotsV1Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    dev_scenes = load_scene_list(args.dev_scenes.resolve())
    dev_predictions = collect(model, store, dev_scenes, device, args.batch_size)
    calibration_qa = dev_qa(store, dev_scenes)
    thresholds = [round(value, 3) for value in np.linspace(0.50, 0.995, 100)]
    curve = []
    best_threshold = thresholds[0]
    best_key = (-1.0, -1.0, -1.0)
    for threshold in thresholds:
        values, _ = evaluate(dev_predictions, calibration_qa, label_to_id, threshold)
        key = (
            float(values["strict_balanced_accuracy_iou50_↑"]),
            float(values["minimum_positive_negative_strict_iou50_↑"]),
            float(values["answerability_accuracy_↑"]),
        )
        curve.append(values)
        if key > best_key:
            best_key, best_threshold = key, threshold
    calibration_metrics, _ = evaluate(dev_predictions, calibration_qa, label_to_id, best_threshold)

    test_results = {}
    all_cases = []
    for spec in args.tests:
        scene_ids = load_scene_list(spec.scene_list.resolve())
        predictions = collect(model, store, scene_ids, device, args.batch_size)
        qas = manifest_qa(spec.qa_manifest.resolve(), set(scene_ids))
        metrics, cases = evaluate(predictions, qas, label_to_id, best_threshold, return_cases=True)
        test_results[spec.name] = {
            "metrics": metrics, "scenes": len(scene_ids),
            "dense_index_sha256": _sha256_file(spec.dense_index.resolve()),
            "qa_manifest_sha256": _sha256_file(spec.qa_manifest.resolve()),
        }
        all_cases.extend({"dataset": spec.name, **row} for row in cases)
        print(json.dumps({spec.name: metrics}, ensure_ascii=False, sort_keys=True), flush=True)

    cases_path = output / "cases.jsonl"
    cases_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in all_cases),
        encoding="utf-8",
    )
    receipt = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "metric_scope": {
            "answer_event_localization": True, "answer_label_accuracy": False,
            "warning": "Do not rename answer-event localization as answer accuracy.",
        },
        "protocol": {
            "answer_label_used": False, "anchor_label_from_question_used": True,
            "threshold_selected_on": "complete overlap-v2 development set",
            "development_source_groups_cannot_form_balanced_halves": True,
            "test_sources_are_disjoint_from_development": True,
            "test_threshold_tuning": False,
            "threshold_grid": thresholds,
            "selection_key": ["strict_balanced_accuracy_iou50", "minimum_positive_negative_strict_iou50", "answerability_accuracy"],
        },
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256_file(checkpoint_path), "epoch": checkpoint["epoch"]},
        "dev_split": {
            "total_scenes": len(dev_scenes), "calibration_scenes": len(dev_scenes),
            "locked_threshold": best_threshold, "calibration_metrics": calibration_metrics,
            "calibration_curve": curve,
        },
        "tests": test_results,
        "cases": {"path": str(cases_path), "sha256": _sha256_file(cases_path)},
    }
    _atomic_json(receipt, output / "receipt.json")
    print(json.dumps({"locked_threshold": best_threshold, "development": calibration_metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
