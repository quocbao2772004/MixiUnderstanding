#!/usr/bin/env python3
"""Run one locked test of relational event slots without test-time tuning."""

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
from torch.utils.data import DataLoader

from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
)
from mixi_understanding.qces.clean_evidence_scenes import _atomic_write_text
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import (
    assert_dense_identity_disjoint_v2,
    load_bound_qa_manifest_v2,
)
from mixi_understanding.scripts.train_qces_relational_event_slots_v1 import (
    CHECKPOINT_FORMAT,
    SceneSlotDataset,
    collate_scene_slots,
    collect_predictions,
    decode_hysteresis_event_slots,
    evaluate_threshold,
    execute_relational_query,
)


FORMAT = "qces_relational_event_slots_locked_test_receipt_v1"
FIXED_AUDIO_SECONDS = 10.0


def _interval_iou(left: list[float] | None, right: tuple[float, float] | None) -> float:
    if left is None or right is None:
        return 0.0
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(0.0, left[1] - left[0]) + max(0.0, right[1] - right[0]) - intersection
    return intersection / max(union, 1e-8)


def _case_analysis(
    predictions: Mapping[str, Mapping[str, Any]],
    test_qa: list[Any],
    label_to_id: Mapping[str, int],
    *,
    low: float,
    high: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decoded = {
        scene_id: decode_hysteresis_event_slots(
            record["frame_event_probability"],
            boundary_low_threshold=low,
            presence_high_threshold=high,
        )
        for scene_id, record in predictions.items()
    }
    rows: list[dict[str, Any]] = []
    for bound in test_qa:
        qa = bound.qa
        record = predictions[qa.scene_id]
        result = execute_relational_query(
            decoded[qa.scene_id], record["detector_logits"], bound, label_to_id
        )
        anchor = result["anchor"]
        answer = result["answer"]
        anchor_interval_normalized = None if anchor is None else [
            float(anchor["start"]), float(anchor["end"])
        ]
        answer_interval_normalized = None if answer is None else [
            float(answer["start"]), float(answer["end"])
        ]
        anchor_interval = None if anchor is None else [
            float(anchor["start"]) * FIXED_AUDIO_SECONDS,
            float(anchor["end"]) * FIXED_AUDIO_SECONDS,
        ]
        answer_interval = None if answer is None else [
            float(answer["start"]) * FIXED_AUDIO_SECONDS,
            float(answer["end"]) * FIXED_AUDIO_SECONDS,
        ]
        anchor_iou = _interval_iou(
            anchor_interval_normalized,
            tuple(value / FIXED_AUDIO_SECONDS for value in qa.gold_anchor_interval),
        )
        answer_iou = _interval_iou(
            answer_interval_normalized,
            None
            if qa.gold_answer_interval is None
            else tuple(value / FIXED_AUDIO_SECONDS for value in qa.gold_answer_interval),
        )
        predicted_none = bool(result["predicted_none"])
        if qa.no_evidence:
            strict_correct = predicted_none and anchor_iou >= 0.50
        else:
            strict_correct = anchor_iou >= 0.50 and answer_iou >= 0.50
        rows.append(
            {
                "item_id": qa.item_id,
                "scene_id": qa.scene_id,
                "question": qa.question,
                "relation": qa.relation,
                "anchor_label": qa.anchor_label,
                "anchor_ordinal": qa.anchor_ordinal,
                "gold_answer_label": qa.answer_label,
                "no_evidence": qa.no_evidence,
                "predicted_none": predicted_none,
                "predicted_slot_count": len(decoded[qa.scene_id]),
                "anchor_iou": anchor_iou,
                "answer_iou": None if qa.no_evidence else answer_iou,
                "gold_anchor_interval": list(qa.gold_anchor_interval),
                "predicted_anchor_interval": anchor_interval,
                "gold_answer_interval": (
                    None if qa.gold_answer_interval is None else list(qa.gold_answer_interval)
                ),
                "predicted_answer_interval": answer_interval,
                "strict_evidence_correct_iou50": strict_correct,
            }
        )

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        positives = [row for row in subset if not row["no_evidence"]]
        negatives = [row for row in subset if row["no_evidence"]]
        return {
            "count": len(subset),
            "answerable": len(positives),
            "no_evidence": len(negatives),
            "anchor_accuracy_iou50": sum(row["anchor_iou"] >= 0.50 for row in subset) / max(len(subset), 1),
            "answerable_evidence_accuracy_iou50": sum(row["strict_evidence_correct_iou50"] for row in positives) / max(len(positives), 1),
            "no_evidence_evidence_accuracy_iou50": sum(row["strict_evidence_correct_iou50"] for row in negatives) / max(len(negatives), 1),
            "balanced_evidence_accuracy_iou50": 0.5 * (
                sum(row["strict_evidence_correct_iou50"] for row in positives) / max(len(positives), 1)
                + sum(row["strict_evidence_correct_iou50"] for row in negatives) / max(len(negatives), 1)
            ),
        }

    breakdown: dict[str, Any] = {"overall": summarize(rows)}
    for ordinal in sorted({int(row["anchor_ordinal"]) for row in rows}):
        breakdown[f"ordinal_{ordinal}"] = summarize(
            [row for row in rows if int(row["anchor_ordinal"]) == ordinal]
        )
    for relation in sorted({str(row["relation"]) for row in rows}):
        breakdown[f"relation_{relation}"] = summarize(
            [row for row in rows if str(row["relation"]) == relation]
        )
    return rows, breakdown


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-receipt",
        type=Path,
        default=base / "relational_event_slots_v1/receipt.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base / "relational_event_slots_v1/relational_event_slots_v1_best.pt",
    )
    parser.add_argument(
        "--dense-index", type=Path, action="append", default=None,
        help="Repeat for train, dev, and test indexes so source leakage can be audited.",
    )
    parser.add_argument("--train-scene-list", type=Path, default=data / "multievent/scene_ids_train.txt")
    parser.add_argument("--dev-scene-list", type=Path, default=data / "multievent/scene_ids_dev.txt")
    parser.add_argument("--test-scene-list", type=Path, default=data / "multievent/scene_ids_test.txt")
    parser.add_argument("--test-qa-manifest", type=Path, default=data / "multievent/qa_manifest_test.jsonl")
    parser.add_argument(
        "--output-dir", type=Path, default=base / "relational_event_slots_v1_locked_test"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-cache-size", type=int, default=20)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _index_provenance(index_paths: list[Path]) -> dict[str, Any]:
    payloads = [_load_json(path) for path in index_paths]
    checkpoints = {str(payload.get("detector_checkpoint") or "") for payload in payloads}
    ontologies = {str(payload.get("ontology") or "") for payload in payloads}
    labels = [payload.get("labels") for payload in payloads]
    if len(checkpoints) != 1 or "" in checkpoints:
        raise ValueError("dense indexes were not exported by one detector checkpoint")
    if len(ontologies) != 1 or "" in ontologies:
        raise ValueError("dense indexes do not share one ontology")
    if not labels or any(value != labels[0] for value in labels[1:]):
        raise ValueError("dense indexes do not share identical label order")
    return {
        "detector_checkpoint": next(iter(checkpoints)),
        "detector_checkpoint_sha256": _sha256_file(Path(next(iter(checkpoints)))),
        "ontology": next(iter(ontologies)),
        "ontology_sha256": _sha256_file(Path(next(iter(ontologies)))),
        "indexes": {str(path.resolve()): _sha256_file(path.resolve()) for path in index_paths},
        "scene_counts": {str(path.resolve()): int(payload["scene_count"]) for path, payload in zip(index_paths, payloads)},
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.dense_index is None:
        base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
        args.dense_index = [
            base / "dense_multi_train_v2/index.json",
            base / "dense_multi_dev_v2/index.json",
            base / "dense_multi_test_v2/index.json",
        ]
    index_paths = [path.resolve() for path in args.dense_index]
    if len(index_paths) < 3:
        raise ValueError("locked evaluation requires train/dev/test dense indexes")
    output_dir = args.output_dir.resolve()
    receipt_path = output_dir / "receipt.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"locked-test output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    training_receipt_path = args.training_receipt.resolve()
    checkpoint_path = args.checkpoint.resolve()
    training_receipt = _load_json(training_receipt_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if str(training_receipt.get("checkpoint_sha256")) != checkpoint_sha256:
        raise ValueError("checkpoint SHA256 disagrees with the frozen training receipt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not a relational-event-slots v1 checkpoint")
    if int(checkpoint.get("epoch", -1)) != int(training_receipt.get("best_epoch", -2)):
        raise ValueError("checkpoint epoch disagrees with training receipt")
    low = float(checkpoint["boundary_low_threshold"])
    high = float(checkpoint["presence_high_threshold"])
    # The locked evaluator reads thresholds only from the frozen checkpoint.
    # There is deliberately no CLI argument that can tune them on test.

    provenance = _index_provenance(index_paths)
    store = DenseFeatureStore(index_paths, cache_size=args.shard_cache_size)
    if list(checkpoint.get("labels") or []) != list(store.labels or []):
        raise ValueError("checkpoint labels disagree with dense test label order")
    train_ids = load_scene_list(args.train_scene_list)
    dev_ids = load_scene_list(args.dev_scene_list)
    test_ids = load_scene_list(args.test_scene_list)
    if set(test_ids) & (set(train_ids) | set(dev_ids)):
        raise ValueError("test scene ids overlap train/dev")
    identity_audit = assert_dense_identity_disjoint_v2(
        store, {"train": train_ids, "dev": dev_ids, "test": test_ids}
    )
    test_qa = load_bound_qa_manifest_v2(
        args.test_qa_manifest,
        allowed_scene_ids=test_ids,
        store=store,
        max_ordinal=10,
    )
    test_dataset = SceneSlotDataset(store, test_ids, preload=args.preload)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_scene_slots,
    )
    device = _device(args.device)
    model = RelationalEventSlotsV1(
        RelationalEventSlotsV1Config(**checkpoint["config"])
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    predictions = collect_predictions(model, test_loader, device)
    label_to_id = {label: index for index, label in enumerate(store.labels or [])}
    metrics = evaluate_threshold(
        predictions,
        test_qa,
        label_to_id,
        boundary_low_threshold=low,
        presence_high_threshold=high,
    )
    case_rows, breakdown = _case_analysis(
        predictions, test_qa, label_to_id, low=low, high=high
    )
    failures = [row for row in case_rows if not row["strict_evidence_correct_iou50"]]
    failure_path = output_dir / "failures_iou50.jsonl"
    _atomic_write_text(
        failure_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in failures),
    )
    gates = {
        "slot_recall_iou50_ge_0_90": float(metrics["slot_recall_iou50"]) >= 0.90,
        "anchor_ordinal_accuracy_iou50_ge_0_85": float(metrics["anchor_ordinal_accuracy_iou50"]) >= 0.85,
        "answerable_evidence_accuracy_iou50_ge_0_80": float(metrics["answerable_evidence_accuracy_iou50"]) >= 0.80,
        "no_evidence_evidence_accuracy_iou50_ge_0_85": float(metrics["no_evidence_evidence_accuracy_iou50"]) >= 0.85,
        "balanced_evidence_accuracy_iou50_ge_0_82": float(metrics["balanced_evidence_accuracy_iou50"]) >= 0.82,
    }
    receipt: dict[str, Any] = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "locked_test": True,
        "threshold_calibration_on_test": False,
        "answer_label_used_for_slot_prediction_or_selection": False,
        "training_receipt": str(training_receipt_path),
        "training_receipt_sha256": _sha256_file(training_receipt_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "frozen_decoder": {
            "name": str(checkpoint["decoder"]),
            "boundary_low_threshold": low,
            "presence_high_threshold": high,
            "source": "training_checkpoint_selected_on_dev_calibration",
        },
        "test_scene_count": len(test_ids),
        "test_qa_count": len(test_qa),
        "dense_provenance": provenance,
        "identity_audit": identity_audit,
        "metrics": metrics,
        "qa_breakdown": breakdown,
        "failure_analysis": {
            "strict_iou50_failures": len(failures),
            "path": str(failure_path),
            "sha256": _sha256_file(failure_path),
        },
        "success_gates": gates,
        "all_success_gates_pass": all(gates.values()),
    }
    _atomic_json(receipt, receipt_path)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
