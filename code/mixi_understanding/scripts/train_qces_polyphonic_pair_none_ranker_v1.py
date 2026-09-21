#!/usr/bin/env python3
"""Train a listwise (anchor, answer) / NONE ranker over frozen temporal slots.

The query branch and BEATs detector stay frozen.  For each relational question,
all temporally valid slot pairs plus one NONE candidate per anchor slot are
scored jointly.  Soft targets are formed from anchor/answer IoU.  If an
answerable record has no proposal pair with both IoU >= 0.30, it is excluded
from the ranking loss instead of receiving a fabricated target.

The answer label is diagnostic metadata only and is never a model input.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.relational_event_slots_v1 import RelationalEventSlotsV1, RelationalEventSlotsV1Config
from mixi_understanding.scripts.evaluate_qces_polyphonic_deployable_decoder_v1 import (
    QA,
    TestSpec,
    collect,
    dev_qa,
    interval_iou,
    manifest_qa,
)
from mixi_understanding.scripts.train_qces_polyphonic_query_branch_v1 import POLYPHONIC_CHECKPOINT_FORMAT
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _atomic_torch,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import assert_dense_identity_disjoint_v2
from mixi_understanding.scripts.train_qces_relational_event_slots_v1 import _pool_anchor_score


FORMAT = "qces_polyphonic_pair_none_ranker_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_polyphonic_pair_none_ranker_checkpoint_v1"
FEATURE_NAMES = (
    "anchor_objectness", "answer_objectness", "anchor_label_probability_at_anchor",
    "anchor_label_probability_at_answer", "anchor_label_rank", "answer_anchor_label_rank",
    "anchor_objectness_rank", "answer_objectness_rank", "anchor_start", "anchor_end",
    "anchor_width", "answer_start", "answer_end", "answer_width", "relation_onset_distance",
    "nonoverlap_gap", "interval_iou", "relation_after", "candidate_is_none", "slot_count_fraction",
)


@dataclass(frozen=True)
class Config:
    feature_dim: int = len(FEATURE_NAMES)
    hidden_dim: int = 96
    dropout: float = 0.15


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    overlap_data = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2"
    nonoverlap = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/multievent"
    repeated = PROJECT_ROOT / "outputs/qces_full191_repeated_ordinal_stress_test_v1"
    overlap = PROJECT_ROOT / "outputs/qces_full191_overlap_onset_stress_test_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-checkpoint", type=Path, default=base / "polyphonic_query_branch_v2/polyphonic_query_branch_best.pt")
    parser.add_argument("--train-index", type=Path, default=base / "dense_overlap_query_train_v2/index.json")
    parser.add_argument("--dev-index", type=Path, default=base / "dense_overlap_query_dev_v2/index.json")
    parser.add_argument("--train-scenes", type=Path, default=overlap_data / "scene_ids_overlap_train.txt")
    parser.add_argument("--dev-scenes", type=Path, default=overlap_data / "scene_ids_overlap_dev.txt")
    parser.add_argument("--output-dir", type=Path, default=base / "polyphonic_pair_none_ranker_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2179)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--soft-target-temperature", type=float, default=0.10)
    parser.add_argument("--positive-iou-threshold", type=float, default=0.30)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(tests=(
        TestSpec("nonoverlap_locked_test", base / "dense_multi_test_v2/index.json", nonoverlap / "scene_ids_test.txt", nonoverlap / "qa_manifest_test.jsonl"),
        TestSpec("repeated_ordinal_stress", base / "dense_repeated_ordinal_stress_test_v2/index.json", repeated / "scene_ids_test.txt", repeated / "qa_manifest_test.jsonl"),
        TestSpec("overlap_onset_stress", base / "dense_overlap_onset_stress_test_v1/index.json", overlap / "scene_ids_test.txt", overlap / "qa_manifest_test.jsonl"),
    ))
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    result = [0.0] * len(values)
    denominator = max(len(values) - 1, 1)
    for rank, index in enumerate(order):
        result[index] = 1.0 - rank / denominator
    return result


def candidate_examples(
    predictions: Mapping[str, Mapping[str, Any]], qas: Sequence[QA],
    label_to_id: Mapping[str, int], *, positive_iou_threshold: float,
    soft_target_temperature: float,
) -> list[dict[str, Any]]:
    examples = []
    for qa in qas:
        record = predictions[qa.scene_id]
        slots = sorted(record["slots"], key=lambda slot: int(slot["slot_index"]))
        label_id = label_to_id[qa.anchor_label]
        anchor_raw = [
            _pool_anchor_score(record["detector_logits"], (float(slot["start"]), float(slot["end"])), label_id)
            for slot in slots
        ]
        anchor_probability = [float(torch.sigmoid(torch.tensor(value))) for value in anchor_raw]
        anchor_rank = ranks(anchor_raw)
        objectness = [float(slot["score"]) for slot in slots]
        objectness_rank = ranks(objectness)
        # Repeated-ordinal evaluation keeps the historical deterministic anchor
        # resolver.  Ranker training data contains unique labels (ordinal 1).
        if qa.anchor_ordinal > 1:
            likely = sorted(range(len(slots)), key=lambda index: anchor_raw[index], reverse=True)[:qa.anchor_ordinal]
            likely.sort(key=lambda index: (float(slots[index]["start"]), float(slots[index]["end"])))
            anchor_indices = [likely[qa.anchor_ordinal - 1]] if len(likely) >= qa.anchor_ordinal else []
        else:
            anchor_indices = list(range(len(slots)))
        features = []
        metadata = []
        quality = []
        for anchor_index in anchor_indices:
            anchor = slots[anchor_index]
            a_start, a_end = float(anchor["start"]), float(anchor["end"])
            answer_indices: list[int | None] = [None]
            answer_indices += [
                index for index, answer in enumerate(slots)
                if index != anchor_index and (
                    float(answer["start"]) < a_start if qa.relation == "before" else float(answer["start"]) > a_start
                )
            ]
            for answer_index in answer_indices:
                is_none = answer_index is None
                answer = None if is_none else slots[int(answer_index)]
                if answer is None:
                    b_start = b_end = b_width = b_obj = b_label = b_label_rank = b_obj_rank = 0.0
                    onset_distance = gap = pair_iou = 0.0
                else:
                    b_start, b_end = float(answer["start"]), float(answer["end"])
                    b_width = b_end - b_start
                    b_obj = objectness[int(answer_index)]
                    b_label = anchor_probability[int(answer_index)]
                    b_label_rank = anchor_rank[int(answer_index)]
                    b_obj_rank = objectness_rank[int(answer_index)]
                    onset_distance = (b_start - a_start) if qa.relation == "after" else (a_start - b_start)
                    gap = max(0.0, max(a_start, b_start) - min(a_end, b_end))
                    pair_iou = interval_iou((a_start, a_end), (b_start, b_end))
                features.append([
                    objectness[anchor_index], b_obj, anchor_probability[anchor_index], b_label,
                    anchor_rank[anchor_index], b_label_rank, objectness_rank[anchor_index], b_obj_rank,
                    a_start, a_end, a_end - a_start, b_start, b_end, b_width, onset_distance, gap,
                    pair_iou, float(qa.relation == "after"), float(is_none), len(slots) / 8.0,
                ])
                anchor_iou = interval_iou((a_start, a_end), qa.gold_anchor)
                answer_iou = 0.0 if answer is None else interval_iou((b_start, b_end), qa.gold_answer)
                if qa.no_evidence:
                    candidate_quality = anchor_iou if is_none and anchor_iou >= positive_iou_threshold else 0.0
                else:
                    candidate_quality = min(anchor_iou, answer_iou) if (
                        not is_none and anchor_iou >= positive_iou_threshold and answer_iou >= positive_iou_threshold
                    ) else 0.0
                quality.append(candidate_quality)
                metadata.append({
                    "anchor": (a_start, a_end), "answer": None if answer is None else (b_start, b_end),
                    "is_none": is_none, "anchor_iou": anchor_iou, "answer_iou": answer_iou,
                })
        if not features:
            raise RuntimeError(f"no ranker candidates for {qa.item_id}")
        quality_tensor = torch.tensor(quality, dtype=torch.float32)
        positive = quality_tensor > 0
        target = torch.zeros_like(quality_tensor)
        if bool(positive.any()):
            target[positive] = torch.softmax(
                quality_tensor[positive] / soft_target_temperature, dim=0
            )
        examples.append({
            "qa": qa, "features": torch.tensor(features, dtype=torch.float32), "target": target,
            "trainable": bool(positive.any()), "metadata": metadata,
        })
    return examples


class ExampleDataset(Dataset):
    def __init__(self, examples: Sequence[Mapping[str, Any]], *, trainable_only: bool) -> None:
        self.examples = [row for row in examples if bool(row["trainable"]) or not trainable_only]
    def __len__(self) -> int:
        return len(self.examples)
    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return self.examples[index]


def collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    maximum = max(int(row["features"].shape[0]) for row in rows)
    dimension = int(rows[0]["features"].shape[1])
    features = torch.zeros(len(rows), maximum, dimension)
    mask = torch.zeros(len(rows), maximum, dtype=torch.bool)
    target = torch.zeros(len(rows), maximum)
    no_evidence = torch.zeros(len(rows), dtype=torch.bool)
    for index, row in enumerate(rows):
        count = int(row["features"].shape[0])
        features[index, :count] = row["features"]
        mask[index, :count] = True
        target[index, :count] = row["target"]
        no_evidence[index] = bool(row["qa"].no_evidence)
    return {"features": features, "mask": mask, "target": target, "no_evidence": no_evidence, "rows": rows}


class PairNoneRanker(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.LayerNorm(config.feature_dim), nn.Linear(config.feature_dim, config.hidden_dim),
            nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(), nn.Dropout(config.dropout), nn.Linear(config.hidden_dim, 1),
        )
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1).masked_fill(~mask, -1e4)


def metric_from_choices(examples: Sequence[Mapping[str, Any]], choices: Sequence[int]) -> dict[str, Any]:
    counts = defaultdict(int)
    for example, choice in zip(examples, choices, strict=True):
        qa: QA = example["qa"]
        selected = example["metadata"][int(choice)]
        anchor_iou, answer_iou = float(selected["anchor_iou"]), float(selected["answer_iou"])
        predicted_none = bool(selected["is_none"])
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
    positive, negative, total = max(counts["positive"], 1), max(counts["negative"], 1), max(counts["total"], 1)
    pos50 = counts["answer50"] / positive
    neg50 = counts["negative_strict50"] / negative
    return {
        "questions": counts["total"], "answerable_questions": counts["positive"], "no_evidence_questions": counts["negative"],
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


@torch.inference_mode()
def evaluate_model(model: PairNoneRanker, examples: Sequence[Mapping[str, Any]], device: torch.device, batch_size: int) -> tuple[dict[str, Any], list[int]]:
    loader = DataLoader(ExampleDataset(examples, trainable_only=False), batch_size=batch_size, shuffle=False, collate_fn=collate)
    model.eval()
    choices = []
    ordered = []
    for raw in loader:
        score = model(raw["features"].to(device), raw["mask"].to(device)).cpu()
        choices.extend(score.argmax(1).tolist())
        ordered.extend(raw["rows"])
    return metric_from_choices(ordered, choices), choices


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.query_batch_size) < 1:
        raise SystemExit("epochs/patience/batch sizes must be positive")
    if not 0.0 < args.positive_iou_threshold <= 1.0 or args.soft_target_temperature <= 0:
        raise SystemExit("invalid IoU threshold/temperature")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    indexes = [args.train_index.resolve(), args.dev_index.resolve()] + [spec.dense_index.resolve() for spec in args.tests]
    store = DenseFeatureStore(indexes, cache_size=32)
    train_scenes = load_scene_list(args.train_scenes.resolve())
    dev_scenes = load_scene_list(args.dev_scenes.resolve())
    if args.max_train_scenes: train_scenes = train_scenes[:args.max_train_scenes]
    if args.max_dev_scenes: dev_scenes = dev_scenes[:args.max_dev_scenes]
    identity = assert_dense_identity_disjoint_v2(store, {"train": train_scenes, "dev": dev_scenes})
    query_checkpoint_path = args.query_checkpoint.resolve()
    query_checkpoint = torch.load(query_checkpoint_path, map_location="cpu", weights_only=True)
    if query_checkpoint.get("format") != POLYPHONIC_CHECKPOINT_FORMAT:
        raise ValueError("bad query checkpoint format")
    if list(query_checkpoint["labels"]) != list(store.labels or []):
        raise ValueError("query checkpoint/dense labels differ")
    label_to_id = {label: index for index, label in enumerate(store.labels or [])}
    device = _device(args.device)
    query_model = RelationalEventSlotsV1(RelationalEventSlotsV1Config(**query_checkpoint["config"])).to(device)
    query_model.load_state_dict(query_checkpoint["model_state_dict"], strict=True)
    query_model.eval().requires_grad_(False)

    print("collecting train query slots", flush=True)
    train_predictions = collect(query_model, store, train_scenes, device, args.query_batch_size)
    train_qa = dev_qa(store, train_scenes)
    train_examples = candidate_examples(
        train_predictions, train_qa, label_to_id,
        positive_iou_threshold=args.positive_iou_threshold,
        soft_target_temperature=args.soft_target_temperature,
    )
    del train_predictions
    gc.collect()
    print("collecting dev query slots", flush=True)
    dev_predictions = collect(query_model, store, dev_scenes, device, args.query_batch_size)
    development_qa = dev_qa(store, dev_scenes)
    dev_examples = candidate_examples(
        dev_predictions, development_qa, label_to_id,
        positive_iou_threshold=args.positive_iou_threshold,
        soft_target_temperature=args.soft_target_temperature,
    )
    del dev_predictions
    gc.collect()
    train_coverage = float(np.mean([row["trainable"] for row in train_examples]))
    dev_coverage = float(np.mean([row["trainable"] for row in dev_examples]))
    print(json.dumps({"train_candidate_coverage": train_coverage, "dev_candidate_coverage": dev_coverage}), flush=True)

    train_dataset = ExampleDataset(train_examples, trainable_only=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate)
    config = Config(hidden_dim=args.hidden_dim, dropout=args.dropout)
    model = PairNoneRanker(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    # Temper class balancing: full inverse-frequency weighting made the tiny
    # smoke model collapse to NONE.  Square-root balancing still compensates
    # the 4:1 positive/negative frequency without letting either side dominate.
    negative_weight = math.sqrt(
        sum(not row["qa"].no_evidence for row in train_dataset.examples)
        / max(sum(row["qa"].no_evidence for row in train_dataset.examples), 1)
    )
    checkpoint_path = output / "pair_none_ranker_best.pt"
    best_key = (-1.0, -1.0, -1.0)
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for raw in train_loader:
            features, mask, target = raw["features"].to(device), raw["mask"].to(device), raw["target"].to(device)
            weights = torch.where(raw["no_evidence"].to(device), negative_weight, 1.0)
            optimizer.zero_grad(set_to_none=True)
            score = model(features, mask)
            per_example = -(target * F.log_softmax(score, dim=1)).sum(1)
            loss = (per_example * weights).sum() / weights.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(raw["rows"])
            seen += len(raw["rows"])
        dev_metrics, _ = evaluate_model(model, dev_examples, device, args.batch_size)
        key = (
            float(dev_metrics["minimum_positive_negative_strict_iou50_↑"]),
            float(dev_metrics["strict_balanced_accuracy_iou50_↑"]),
            float(dev_metrics["joint_evidence_localization_iou50_↑"]),
        )
        row = {"epoch": epoch, "train_loss": loss_sum / max(seen, 1), "learning_rate": optimizer.param_groups[0]["lr"], "dev": dev_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            _atomic_torch({
                "format": CHECKPOINT_FORMAT, "epoch": epoch, "config": asdict(config),
                "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "feature_names": FEATURE_NAMES, "labels": list(store.labels or []), "dev_metrics": dev_metrics,
                "query_checkpoint_sha256": _sha256_file(query_checkpoint_path),
            }, checkpoint_path)
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience: break

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    best_dev, _ = evaluate_model(model, dev_examples, device, args.batch_size)
    test_results = {}
    cases = []
    for spec in args.tests:
        scene_ids = load_scene_list(spec.scene_list.resolve())
        predictions = collect(query_model, store, scene_ids, device, args.query_batch_size)
        qas = manifest_qa(spec.qa_manifest.resolve(), set(scene_ids))
        examples = candidate_examples(
            predictions, qas, label_to_id,
            positive_iou_threshold=args.positive_iou_threshold,
            soft_target_temperature=args.soft_target_temperature,
        )
        metrics, choices = evaluate_model(model, examples, device, args.batch_size)
        coverage = float(np.mean([row["trainable"] for row in examples]))
        test_results[spec.name] = {"metrics": metrics, "candidate_coverage_iou30": coverage, "scenes": len(scene_ids)}
        for example, choice in zip(examples, choices, strict=True):
            qa = example["qa"]; selected = example["metadata"][choice]
            cases.append({
                "dataset": spec.name, "item_id": qa.item_id, "scene_id": qa.scene_id,
                "question": qa.question, "no_evidence": qa.no_evidence,
                "answer_label_diagnostic_only": qa.answer_label_diagnostic_only,
                "predicted_none": selected["is_none"], "anchor_iou": selected["anchor_iou"],
                "answer_iou": selected["answer_iou"], "candidate_pool_has_gold_iou30": example["trainable"],
            })
        print(json.dumps({spec.name: test_results[spec.name]}, ensure_ascii=False, sort_keys=True), flush=True)
        del predictions, examples
        gc.collect()
    cases_path = output / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in cases), encoding="utf-8")
    receipt = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
        "answer_label_used_as_input": False,
        "metric_scope": {"answer_event_localization": True, "answer_label_accuracy": False},
        "method": "frozen temporal-v2 slots + listwise pair/NONE ranker",
        "data": {"train_scenes": len(train_scenes), "train_questions": len(train_examples), "train_candidate_coverage_iou30": train_coverage,
                 "dev_scenes": len(dev_scenes), "dev_questions": len(dev_examples), "dev_candidate_coverage_iou30": dev_coverage,
                 "identity_audit": identity},
        "loss_policy": {"soft_iou_targets": True, "positive_iou_threshold": args.positive_iou_threshold,
                        "uncovered_answerable_records_excluded_from_loss": True, "none_is_listwise_candidate": True,
                        "negative_example_weight": negative_weight},
        "feature_names": FEATURE_NAMES, "best_epoch": best_epoch, "best_dev": best_dev,
        "tests": test_results, "history": history,
        "query_checkpoint": {"path": str(query_checkpoint_path), "sha256": _sha256_file(query_checkpoint_path)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": _sha256_file(checkpoint_path)},
        "cases": {"path": str(cases_path), "sha256": _sha256_file(cases_path)},
    }
    _atomic_json(receipt, output / "receipt.json")
    print(json.dumps({"best_epoch": best_epoch, "best_dev": best_dev}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
