#!/usr/bin/env python3
"""Train class-agnostic slots and evaluate question-conditioned evidence.

This is a sidecar experiment.  It does not modify Q-DOR, R1, or Claude's
historical training code.  Gold labels supervise neither the slot predictor
nor slot selection.  During evaluation the question supplies only an anchor
label, ordinal, and relation; frozen R1 logits score that anchor inside each
predicted slot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.benchmark_integrity import semantic_events
from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
    RelationalEventSlotsV1LossWeights,
    interval_iou_matrix,
    relational_event_slots_v1_loss,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _atomic_torch,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import (
    BoundQAItemV2,
    assert_dense_identity_disjoint_v2,
    load_bound_qa_manifest_v2,
)


FORMAT = "qces_relational_event_slots_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_relational_event_slots_checkpoint_v1"
NUM_FRAMES = 250
FRAME_HOP_SECONDS = 0.04
FIXED_AUDIO_SECONDS = 10.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def split_source_disjoint_scenes(
    store: DenseFeatureStore,
    scene_ids: Sequence[str],
    *,
    seed: int,
    calibration_fraction: float,
) -> tuple[list[str], list[str]]:
    """Split whole connected source-identity groups, never individual scenes."""

    unique = list(dict.fromkeys(str(value) for value in scene_ids))
    if len(unique) < 2 or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("need >=2 scenes and calibration_fraction in (0,1)")
    parent = {scene_id: scene_id for scene_id in unique}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners: dict[tuple[str, str], str] = {}
    identity_fields = ("source_sha256", "source_path", "source_id", "source_video_id")
    for scene_id in unique:
        for event in semantic_events(store.metadata(scene_id)):
            for field in identity_fields:
                value = str(event.get(field) or "").strip()
                if not value:
                    continue
                identity = (field, value)
                if identity in owners:
                    union(scene_id, owners[identity])
                else:
                    owners[identity] = scene_id
    groups: dict[str, list[str]] = {}
    for scene_id in unique:
        groups.setdefault(find(scene_id), []).append(scene_id)
    ranked = sorted(
        groups.values(),
        key=lambda group: hashlib.sha256(
            f"{seed}:event-slots-source-group:{'|'.join(sorted(group))}".encode()
        ).hexdigest(),
    )
    target = len(unique) * calibration_fraction
    calibration: list[str] = []
    selection: list[str] = []
    # Greedy assignment preserves the seeded random order but keeps sizes near
    # the requested fraction even when source-connected groups have >1 scene.
    for group in ranked:
        if len(calibration) < target:
            calibration.extend(group)
        else:
            selection.extend(group)
    if not calibration or not selection:
        raise ValueError("source-connected groups cannot form two non-empty dev subsets")
    return calibration, selection


def _frame_targets(events: Sequence[Mapping[str, Any]], valid_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    activity = torch.zeros(NUM_FRAMES, dtype=torch.float32)
    onset = torch.zeros(NUM_FRAMES, dtype=torch.float32)
    for event in events:
        start = float(event["onset_seconds"])
        end = float(event["offset_seconds"])
        start_frame = max(0, min(valid_frames - 1, int(math.floor(start / FRAME_HOP_SECONDS))))
        end_frame = max(start_frame + 1, min(valid_frames, int(math.ceil(end / FRAME_HOP_SECONDS))))
        activity[start_frame:end_frame] = 1.0
        # Three-frame triangular supervision tolerates the 40 ms feature grid
        # while retaining a unique onset peak.
        onset[start_frame] = 1.0
        if start_frame > 0:
            onset[start_frame - 1] = max(float(onset[start_frame - 1]), 0.35)
        if start_frame + 1 < valid_frames:
            onset[start_frame + 1] = max(float(onset[start_frame + 1]), 0.35)
    return activity, onset


class SceneSlotDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        store: DenseFeatureStore,
        scene_ids: Sequence[str],
        *,
        preload: bool,
        max_scenes: int = 0,
    ) -> None:
        selected = list(dict.fromkeys(str(value) for value in scene_ids))
        if max_scenes > 0:
            selected = selected[:max_scenes]
        if not selected:
            raise ValueError("scene dataset cannot be empty")
        missing = sorted(set(selected) - store.scene_ids)
        if missing:
            raise ValueError(f"scenes missing from dense store: {missing[:5]}")
        self.store = store
        self.scene_ids = selected
        self.preloaded = store.preload(selected) if preload else None
        self.gold: dict[str, dict[str, Any]] = {}
        for scene_id in selected:
            metadata = store.metadata(scene_id)
            events = semantic_events(metadata)
            if not events or len(events) > 8:
                raise ValueError(f"{scene_id}: expected 1..8 semantic events, got {len(events)}")
            valid_frames = int(metadata["valid_frames"])
            activity, onset = _frame_targets(events, valid_frames)
            intervals = torch.tensor(
                [
                    [
                        float(event["onset_seconds"]) / FIXED_AUDIO_SECONDS,
                        float(event["offset_seconds"]) / FIXED_AUDIO_SECONDS,
                    ]
                    for event in events
                ],
                dtype=torch.float32,
            )
            self.gold[scene_id] = {
                "events": events,
                "valid_frames": valid_frames,
                "intervals": intervals,
                "frame_event_target": activity,
                "frame_onset_target": onset,
            }

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_id = self.scene_ids[index]
        dense = self.preloaded[scene_id] if self.preloaded is not None else self.store.get(scene_id)
        gold = self.gold[scene_id]
        valid_frames = int(gold["valid_frames"])
        return {
            "scene_id": scene_id,
            "features": dense["features"].float(),
            "detector_logits": dense["logits"].float(),
            "valid_mask": torch.arange(NUM_FRAMES) < valid_frames,
            "target_intervals": gold["intervals"],
            "frame_event_target": gold["frame_event_target"],
            "frame_onset_target": gold["frame_onset_target"],
        }


def collate_scene_slots(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "scene_id": [str(row["scene_id"]) for row in rows],
        "features": torch.stack([row["features"] for row in rows]),
        "detector_logits": torch.stack([row["detector_logits"] for row in rows]),
        "valid_mask": torch.stack([row["valid_mask"] for row in rows]),
        "target_intervals": [row["target_intervals"] for row in rows],
        "frame_event_target": torch.stack([row["frame_event_target"] for row in rows]),
        "frame_onset_target": torch.stack([row["frame_onset_target"] for row in rows]),
    }


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key in ("features", "detector_logits", "valid_mask", "frame_event_target", "frame_onset_target"):
        result[key] = result[key].to(device, non_blocking=True)
    result["target_intervals"] = [value.to(device, non_blocking=True) for value in batch["target_intervals"]]
    return result


def _interval_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0])))
    union = max(0.0, float(left[1]) - float(left[0])) + max(0.0, float(right[1]) - float(right[0])) - intersection
    return intersection / max(union, 1e-8)


def _one_to_one_ious(predicted: Sequence[Sequence[float]], gold: Sequence[Sequence[float]]) -> list[float]:
    if not predicted or not gold:
        return []
    left = torch.tensor(predicted, dtype=torch.float32)
    right = torch.tensor(gold, dtype=torch.float32)
    iou = interval_iou_matrix(left, right).numpy()
    rows, columns = linear_sum_assignment(1.0 - iou)
    return [float(iou[row, column]) for row, column in zip(rows, columns)]


def _pool_anchor_score(logits: torch.Tensor, interval: Sequence[float], label_id: int) -> float:
    start = max(0, min(NUM_FRAMES - 1, int(math.floor(float(interval[0]) * NUM_FRAMES))))
    end = max(start + 1, min(NUM_FRAMES, int(math.ceil(float(interval[1]) * NUM_FRAMES))))
    return float(logits[start:end, label_id].mean())


def decode_hysteresis_event_slots(
    frame_probability: torch.Tensor,
    *,
    boundary_low_threshold: float,
    presence_high_threshold: float,
    minimum_frames: int = 2,
) -> list[dict[str, float | int]]:
    """Decode temporal slots while separating presence from boundary extent."""

    if frame_probability.ndim != 1:
        raise ValueError("frame_probability must have shape [T]")
    if not 0.0 <= boundary_low_threshold <= presence_high_threshold <= 1.0:
        raise ValueError("hysteresis thresholds must satisfy 0 <= low <= high <= 1")
    low_active = (frame_probability >= boundary_low_threshold).tolist()
    rows: list[dict[str, float | int]] = []
    start: int | None = None
    for frame, active in enumerate(low_active + [False]):
        if active and start is None:
            start = frame
            continue
        if active or start is None:
            continue
        end = frame
        region = frame_probability[start:end]
        if end - start >= minimum_frames and float(region.max()) >= presence_high_threshold:
            rows.append(
                {
                    "slot_index": len(rows),
                    "score": float(region.max()),
                    "start": start / NUM_FRAMES,
                    "end": end / NUM_FRAMES,
                }
            )
        start = None
    return rows


def execute_relational_query(
    predicted_slots: Sequence[Mapping[str, float]],
    detector_logits: torch.Tensor,
    qa: BoundQAItemV2,
    label_to_id: Mapping[str, int],
) -> dict[str, Any]:
    """Select anchor occurrence then its immediate temporal neighbour.

    The answer label is intentionally never read.  The only semantic score is
    for the anchor label explicitly present in the question.
    """

    ordinal = int(qa.qa.anchor_ordinal)
    label_id = label_to_id[qa.qa.anchor_label]
    slots = [dict(value) for value in predicted_slots]
    for slot in slots:
        slot["anchor_score"] = _pool_anchor_score(
            detector_logits, (float(slot["start"]), float(slot["end"])), label_id
        )
    if len(slots) < ordinal:
        return {"anchor": None, "answer": None, "predicted_none": True}
    top = sorted(slots, key=lambda row: float(row["anchor_score"]), reverse=True)[:ordinal]
    top.sort(key=lambda row: (float(row["start"]), float(row["end"])))
    anchor = top[ordinal - 1]
    ordered = sorted(slots, key=lambda row: (float(row["start"]), float(row["end"])))
    anchor_position = next(index for index, row in enumerate(ordered) if int(row["slot_index"]) == int(anchor["slot_index"]))
    answer_position = anchor_position - 1 if qa.qa.relation == "before" else anchor_position + 1
    answer = ordered[answer_position] if 0 <= answer_position < len(ordered) else None
    return {"anchor": anchor, "answer": answer, "predicted_none": answer is None}


@torch.inference_mode()
def collect_predictions(
    model: RelationalEventSlotsV1,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    model.eval()
    collected: dict[str, dict[str, Any]] = {}
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        outputs = model(batch["features"], batch["valid_mask"])
        confidence = outputs["objectness_logits"].sigmoid().cpu()
        intervals = outputs["intervals"].cpu()
        frame_probability = outputs["frame_event_logits"].sigmoid().cpu()
        logits = raw_batch["detector_logits"].cpu()
        for index, scene_id in enumerate(raw_batch["scene_id"]):
            slots = [
                {
                    "slot_index": slot,
                    "score": float(confidence[index, slot]),
                    "start": float(intervals[index, slot, 0]),
                    "end": float(intervals[index, slot, 1]),
                }
                for slot in range(confidence.shape[1])
            ]
            collected[scene_id] = {
                "slots": slots,
                "frame_event_probability": frame_probability[index],
                "detector_logits": logits[index],
                "gold_intervals": raw_batch["target_intervals"][index].tolist(),
            }
    return collected


def evaluate_threshold(
    predictions: Mapping[str, Mapping[str, Any]],
    qa_items: Sequence[BoundQAItemV2],
    label_to_id: Mapping[str, int],
    *,
    boundary_low_threshold: float,
    presence_high_threshold: float,
) -> dict[str, float | int]:
    true_positive_30 = true_positive_50 = predicted_count = gold_count = exact_count = 0
    matched_iou_sum = 0.0
    matched_count = 0
    decoded: dict[str, list[dict[str, float]]] = {}
    for scene_id, record in predictions.items():
        slots = decode_hysteresis_event_slots(
            record["frame_event_probability"],
            boundary_low_threshold=boundary_low_threshold,
            presence_high_threshold=presence_high_threshold,
        )
        decoded[scene_id] = slots
        predicted_intervals = [[float(row["start"]), float(row["end"])] for row in slots]
        gold = record["gold_intervals"]
        ious = _one_to_one_ious(predicted_intervals, gold)
        true_positive_30 += sum(value >= 0.30 for value in ious)
        true_positive_50 += sum(value >= 0.50 for value in ious)
        predicted_count += len(predicted_intervals)
        gold_count += len(gold)
        exact_count += int(len(predicted_intervals) == len(gold))
        matched_iou_sum += sum(ious)
        matched_count += len(ious)

    counters: Counter[str] = Counter()
    for bound in qa_items:
        qa = bound.qa
        record = predictions[qa.scene_id]
        result = execute_relational_query(
            decoded[qa.scene_id], record["detector_logits"], bound, label_to_id
        )
        predicted_anchor = result["anchor"]
        predicted_answer = result["answer"]
        anchor_interval = None if predicted_anchor is None else [predicted_anchor["start"], predicted_anchor["end"]]
        answer_interval = None if predicted_answer is None else [predicted_answer["start"], predicted_answer["end"]]
        anchor_iou = 0.0 if anchor_interval is None else _interval_iou(anchor_interval, [value / FIXED_AUDIO_SECONDS for value in qa.gold_anchor_interval])
        counters["total"] += 1
        counters["anchor_30"] += int(anchor_iou >= 0.30)
        counters["anchor_50"] += int(anchor_iou >= 0.50)
        predicted_none = bool(result["predicted_none"])
        counters["answerability_correct"] += int(predicted_none == qa.no_evidence)
        if qa.no_evidence:
            counters["negative"] += 1
            counters["negative_answer_correct"] += int(predicted_none)
            counters["negative_evidence_30"] += int(predicted_none and anchor_iou >= 0.30)
            counters["negative_evidence_50"] += int(predicted_none and anchor_iou >= 0.50)
        else:
            counters["positive"] += 1
            assert qa.gold_answer_interval is not None
            answer_iou = 0.0 if answer_interval is None else _interval_iou(answer_interval, [value / FIXED_AUDIO_SECONDS for value in qa.gold_answer_interval])
            counters["positive_answer_30"] += int(answer_iou >= 0.30)
            counters["positive_answer_50"] += int(answer_iou >= 0.50)
            counters["positive_evidence_30"] += int(anchor_iou >= 0.30 and answer_iou >= 0.30)
            counters["positive_evidence_50"] += int(anchor_iou >= 0.50 and answer_iou >= 0.50)

    def ratio(numerator: str, denominator: str) -> float:
        return float(counters[numerator]) / max(int(counters[denominator]), 1)

    slot_precision_30 = true_positive_30 / max(predicted_count, 1)
    slot_recall_30 = true_positive_30 / max(gold_count, 1)
    slot_precision_50 = true_positive_50 / max(predicted_count, 1)
    slot_recall_50 = true_positive_50 / max(gold_count, 1)
    positive_evidence_30 = ratio("positive_evidence_30", "positive")
    positive_evidence_50 = ratio("positive_evidence_50", "positive")
    negative_evidence_30 = ratio("negative_evidence_30", "negative")
    negative_evidence_50 = ratio("negative_evidence_50", "negative")
    return {
        "decoder": "hysteresis_frame_refined_slots",
        "boundary_low_threshold": boundary_low_threshold,
        "presence_high_threshold": presence_high_threshold,
        "num_scenes": len(predictions),
        "num_qa": int(counters["total"]),
        "slot_precision_iou30": slot_precision_30,
        "slot_recall_iou30": slot_recall_30,
        "slot_f1_iou30": 2 * slot_precision_30 * slot_recall_30 / max(slot_precision_30 + slot_recall_30, 1e-12),
        "slot_precision_iou50": slot_precision_50,
        "slot_recall_iou50": slot_recall_50,
        "slot_f1_iou50": 2 * slot_precision_50 * slot_recall_50 / max(slot_precision_50 + slot_recall_50, 1e-12),
        "slot_exact_count_accuracy": exact_count / max(len(predictions), 1),
        "slot_mean_matched_iou": matched_iou_sum / max(matched_count, 1),
        "anchor_ordinal_accuracy_iou30": ratio("anchor_30", "total"),
        "anchor_ordinal_accuracy_iou50": ratio("anchor_50", "total"),
        "answerability_accuracy": ratio("answerability_correct", "total"),
        "answerable_answer_accuracy_iou30": ratio("positive_answer_30", "positive"),
        "answerable_answer_accuracy_iou50": ratio("positive_answer_50", "positive"),
        "answerable_evidence_accuracy_iou30": positive_evidence_30,
        "answerable_evidence_accuracy_iou50": positive_evidence_50,
        "no_evidence_decision_accuracy": ratio("negative_answer_correct", "negative"),
        "no_evidence_evidence_accuracy_iou30": negative_evidence_30,
        "no_evidence_evidence_accuracy_iou50": negative_evidence_50,
        "balanced_evidence_accuracy_iou30": 0.5 * (positive_evidence_30 + negative_evidence_30),
        "balanced_evidence_accuracy_iou50": 0.5 * (positive_evidence_50 + negative_evidence_50),
    }


def _threshold_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    positive = float(metrics["answerable_evidence_accuracy_iou50"])
    negative = float(metrics["no_evidence_evidence_accuracy_iou50"])
    return (
        min(positive, negative),
        float(metrics["balanced_evidence_accuracy_iou50"]),
        float(metrics["slot_f1_iou50"]),
        -abs(float(metrics["boundary_low_threshold"]) - 0.2),
        -abs(float(metrics["presence_high_threshold"]) - 0.5),
    )


def _checkpoint_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    positive = float(metrics["answerable_evidence_accuracy_iou50"])
    negative = float(metrics["no_evidence_evidence_accuracy_iou50"])
    return (
        min(positive, negative),
        float(metrics["balanced_evidence_accuracy_iou50"]),
        float(metrics["anchor_ordinal_accuracy_iou50"]),
        float(metrics["slot_recall_iou50"]),
        float(metrics["slot_f1_iou50"]),
    )


def _parse_threshold_grid(value: str) -> list[float]:
    thresholds = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if not thresholds or any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise argparse.ArgumentTypeError("threshold grid must contain values in [0,1]")
    return thresholds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    data_root = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
    parser.add_argument("--dense-index", type=Path, action="append", default=None)
    parser.add_argument("--train-scene-list", type=Path, default=data_root / "multievent/scene_ids_train.txt")
    parser.add_argument("--dev-scene-list", type=Path, default=data_root / "multievent/scene_ids_dev.txt")
    parser.add_argument("--dev-qa-manifest", type=Path, default=data_root / "multievent/qa_manifest_dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=default_root / "relational_event_slots_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--boundary-low-grid", type=_parse_threshold_grid, default=_parse_threshold_grid("0.05,0.10,0.20,0.30,0.40"))
    parser.add_argument("--presence-high-grid", type=_parse_threshold_grid, default=_parse_threshold_grid("0.40,0.50,0.60,0.70,0.80"))
    parser.add_argument("--shard-cache-size", type=int, default=20)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dense_index is None:
        base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
        args.dense_index = [base / "dense_multi_train_v2/index.json", base / "dense_multi_dev_v2/index.json"]
    for name in ("epochs", "patience", "batch_size", "num_slots"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"{name} must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)
    train_scene_ids = load_scene_list(args.train_scene_list)
    dev_scene_ids = load_scene_list(args.dev_scene_list)
    if args.max_train_scenes > 0:
        train_scene_ids = train_scene_ids[: args.max_train_scenes]
    if args.max_dev_scenes > 0:
        dev_scene_ids = dev_scene_ids[: args.max_dev_scenes]
    store = DenseFeatureStore(args.dense_index, cache_size=args.shard_cache_size)
    calibration_ids, selection_ids = split_source_disjoint_scenes(
        store,
        dev_scene_ids,
        seed=args.seed,
        calibration_fraction=args.calibration_fraction,
    )
    identity_audit = assert_dense_identity_disjoint_v2(
        store, {"train": train_scene_ids, "calibration": calibration_ids, "selection": selection_ids}
    )
    dev_bound = load_bound_qa_manifest_v2(
        args.dev_qa_manifest,
        allowed_scene_ids=dev_scene_ids,
        store=store,
        max_ordinal=10,
    )
    calibration_set = set(calibration_ids)
    selection_set = set(selection_ids)
    calibration_qa = [item for item in dev_bound if item.qa.scene_id in calibration_set]
    selection_qa = [item for item in dev_bound if item.qa.scene_id in selection_set]
    train_dataset = SceneSlotDataset(store, train_scene_ids, preload=args.preload)
    calibration_dataset = SceneSlotDataset(store, calibration_ids, preload=args.preload)
    selection_dataset = SceneSlotDataset(store, selection_ids, preload=args.preload)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_scene_slots,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_options)
    selection_loader = DataLoader(selection_dataset, shuffle=False, **loader_options)
    config = RelationalEventSlotsV1Config(
        feature_dim=int(store.feature_dim or 0),
        hidden_dim=args.hidden_dim,
        num_slots=args.num_slots,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward_dim=args.hidden_dim * 3,
        dropout=args.dropout,
    )
    device = _device(args.device)
    model = RelationalEventSlotsV1(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    label_to_id = {label: index for index, label in enumerate(store.labels or [])}
    weights = RelationalEventSlotsV1LossWeights()
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_selection: dict[str, Any] | None = None
    stale_epochs = 0
    checkpoint_path = output_dir / "relational_event_slots_v1_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Counter[str] = Counter()
        examples = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(batch["features"], batch["valid_mask"])
                losses = relational_event_slots_v1_loss(
                    outputs,
                    batch["target_intervals"],
                    batch["frame_event_target"],
                    batch["frame_onset_target"],
                    batch["valid_mask"],
                    weights=weights,
                )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            count = int(batch["features"].shape[0])
            examples += count
            for name, value in losses.items():
                totals[name] += float(value.detach()) * count
        scheduler.step()
        train_metrics = {name: value / max(examples, 1) for name, value in totals.items()}
        calibration_predictions = collect_predictions(model, calibration_loader, device)
        calibration_grid = [
            evaluate_threshold(
                calibration_predictions,
                calibration_qa,
                label_to_id,
                boundary_low_threshold=low,
                presence_high_threshold=high,
            )
            for low in args.boundary_low_grid
            for high in args.presence_high_grid
            if low <= high
        ]
        calibration_metrics = max(calibration_grid, key=_threshold_key)
        selected_low_threshold = float(calibration_metrics["boundary_low_threshold"])
        selected_high_threshold = float(calibration_metrics["presence_high_threshold"])
        selection_predictions = collect_predictions(model, selection_loader, device)
        selection_metrics = evaluate_threshold(
            selection_predictions,
            selection_qa,
            label_to_id,
            boundary_low_threshold=selected_low_threshold,
            presence_high_threshold=selected_high_threshold,
        )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "calibration": calibration_metrics,
            "selection": selection_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        current_key = _checkpoint_key(selection_metrics)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_selection = dict(selection_metrics)
            stale_epochs = 0
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT,
                    "epoch": epoch,
                    "config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "decoder": "hysteresis_frame_refined_slots",
                    "boundary_low_threshold": selected_low_threshold,
                    "presence_high_threshold": selected_high_threshold,
                    "calibration_metrics": calibration_metrics,
                    "selection_metrics": selection_metrics,
                    "labels": list(store.labels or []),
                    "dense_index_sha256": store.index_sha256,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    if best_selection is None:
        raise RuntimeError("training produced no checkpoint")
    success_gates = {
        "slot_recall_iou50_ge_0_90": float(best_selection["slot_recall_iou50"]) >= 0.90,
        "anchor_ordinal_accuracy_iou50_ge_0_85": float(best_selection["anchor_ordinal_accuracy_iou50"]) >= 0.85,
        "answerable_evidence_accuracy_iou50_ge_0_80": float(best_selection["answerable_evidence_accuracy_iou50"]) >= 0.80,
        "no_evidence_evidence_accuracy_iou50_ge_0_85": float(best_selection["no_evidence_evidence_accuracy_iou50"]) >= 0.85,
        "balanced_evidence_accuracy_iou50_ge_0_82": float(best_selection["balanced_evidence_accuracy_iou50"]) >= 0.82,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "class_agnostic_temporal_slots_then_question_anchor_ordinal_then_adjacent_pointer",
        "answer_label_used_for_slot_prediction_or_selection": False,
        "config": asdict(config),
        "loss_weights": asdict(weights),
        "arguments": _jsonable(vars(args)),
        "splits": {
            "train_scenes": len(train_scene_ids),
            "calibration_scenes": len(calibration_ids),
            "selection_scenes": len(selection_ids),
            "calibration_qa": len(calibration_qa),
            "selection_qa": len(selection_qa),
        },
        "identity_audit": identity_audit,
        "best_epoch": best_epoch,
        "best_selection": best_selection,
        "success_gates": success_gates,
        "all_success_gates_pass": all(success_gates.values()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "history": history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "best_selection": best_selection, "success_gates": success_gates}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
