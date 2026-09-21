#!/usr/bin/env python3
"""Audit frozen DETR interval-query capacity before changing QCES architecture.

No threshold or model parameter is selected on test.  For each fixed K in
``1,2,3,4,6,8`` the script keeps the K highest-objectness interval queries and
reports proposal recall.  K=8 is explicitly an oracle proposal ceiling, not a
deployable selected decoder.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    _interval_iou,
    _one_to_one_ious,
    collate_scene_slots,
    collect_predictions,
)


FORMAT = "qces_relational_raw_interval_query_audit_v1"
TOP_K_VALUES = (1, 2, 3, 4, 6, 8)
FIXED_AUDIO_SECONDS = 10.0


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dense_index: Path
    scene_list: Path
    qa_manifest: Path
    frozen_eval_receipt: Path


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/multievent"
    overlap = PROJECT_ROOT / "outputs/qces_full191_overlap_onset_stress_test_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=base / "relational_event_slots_v1/relational_event_slots_v1_best.pt",
    )
    parser.add_argument("--nonoverlap-dense-index", type=Path, default=base / "dense_multi_test_v2/index.json")
    parser.add_argument("--nonoverlap-scene-list", type=Path, default=data / "scene_ids_test.txt")
    parser.add_argument("--nonoverlap-qa-manifest", type=Path, default=data / "qa_manifest_test.jsonl")
    parser.add_argument(
        "--nonoverlap-eval-receipt", type=Path,
        default=base / "relational_event_slots_v1_locked_test/receipt.json",
    )
    parser.add_argument(
        "--overlap-dense-index", type=Path,
        default=base / "dense_overlap_onset_stress_test_v1/index.json",
    )
    parser.add_argument("--overlap-scene-list", type=Path, default=overlap / "scene_ids_test.txt")
    parser.add_argument("--overlap-qa-manifest", type=Path, default=overlap / "qa_manifest_test.jsonl")
    parser.add_argument(
        "--overlap-eval-receipt", type=Path,
        default=base / "relational_event_slots_v1_overlap_onset_stress_eval/receipt.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=base / "relational_event_slots_v1_raw_query_audit/receipt.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _top_queries(record: Mapping[str, Any], k: int) -> list[list[float]]:
    ranked = sorted(record["slots"], key=lambda row: float(row["score"]), reverse=True)
    return [
        [float(row["start"]), float(row["end"])] for row in ranked[:k]
    ]


def _exists(intervals: Sequence[Sequence[float]], gold: Sequence[float], threshold: float) -> bool:
    return any(_interval_iou(interval, gold) >= threshold for interval in intervals)


def _distinct_pair(
    intervals: Sequence[Sequence[float]],
    anchor: Sequence[float],
    answer: Sequence[float],
    threshold: float,
) -> bool:
    return any(
        left_index != right_index
        and _interval_iou(intervals[left_index], anchor) >= threshold
        and _interval_iou(intervals[right_index], answer) >= threshold
        for left_index in range(len(intervals))
        for right_index in range(len(intervals))
    )


def _audit_k(
    predictions: Mapping[str, Mapping[str, Any]],
    bound_qa: Sequence[Any],
    *,
    k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scene_true_positive_30 = scene_true_positive_50 = gold_count = 0
    scene_all_resolved_30 = scene_all_resolved_50 = 0
    scene_rows = []
    for scene_id, record in predictions.items():
        intervals = _top_queries(record, k)
        gold = list(record["gold_intervals"])
        matched = _one_to_one_ious(intervals, gold)
        tp30 = sum(value >= 0.30 for value in matched)
        tp50 = sum(value >= 0.50 for value in matched)
        scene_true_positive_30 += tp30
        scene_true_positive_50 += tp50
        gold_count += len(gold)
        scene_all_resolved_30 += int(tp30 == len(gold))
        scene_all_resolved_50 += int(tp50 == len(gold))
        scene_rows.append(
            {
                "scene_id": scene_id,
                "gold_count": len(gold),
                "matched_iou30": tp30,
                "matched_iou50": tp50,
            }
        )

    positives = [bound for bound in bound_qa if not bound.qa.no_evidence]
    negatives = [bound for bound in bound_qa if bound.qa.no_evidence]
    positive_anchor_30 = positive_anchor_50 = 0
    answer_30 = answer_50 = pair_30 = pair_50 = 0
    negative_anchor_30 = negative_anchor_50 = 0
    qa_rows: list[dict[str, Any]] = []
    for bound in bound_qa:
        qa = bound.qa
        intervals = _top_queries(predictions[qa.scene_id], k)
        anchor = [value / FIXED_AUDIO_SECONDS for value in qa.gold_anchor_interval]
        anchor30 = _exists(intervals, anchor, 0.30)
        anchor50 = _exists(intervals, anchor, 0.50)
        if qa.no_evidence:
            negative_anchor_30 += int(anchor30)
            negative_anchor_50 += int(anchor50)
            qa_rows.append(
                {
                    "item_id": qa.item_id,
                    "scene_id": qa.scene_id,
                    "no_evidence": True,
                    "anchor_recall_iou30": anchor30,
                    "anchor_recall_iou50": anchor50,
                }
            )
            continue
        positive_anchor_30 += int(anchor30)
        positive_anchor_50 += int(anchor50)
        if qa.gold_answer_interval is None:
            raise RuntimeError("positive QA has no gold answer interval")
        answer = [value / FIXED_AUDIO_SECONDS for value in qa.gold_answer_interval]
        answer30 = _exists(intervals, answer, 0.30)
        answer50 = _exists(intervals, answer, 0.50)
        distinct30 = _distinct_pair(intervals, anchor, answer, 0.30)
        distinct50 = _distinct_pair(intervals, anchor, answer, 0.50)
        answer_30 += int(answer30)
        answer_50 += int(answer50)
        pair_30 += int(distinct30)
        pair_50 += int(distinct50)
        qa_rows.append(
            {
                "item_id": qa.item_id,
                "scene_id": qa.scene_id,
                "no_evidence": False,
                "anchor_recall_iou30": anchor30,
                "anchor_recall_iou50": anchor50,
                "answer_recall_iou30": answer30,
                "answer_recall_iou50": answer50,
                "distinct_pair_recall_iou30": distinct30,
                "distinct_pair_recall_iou50": distinct50,
            }
        )

    p = max(len(positives), 1)
    n = max(len(negatives), 1)
    metrics = {
        "top_k": k,
        "scene_count": len(predictions),
        "positive_qa_count": len(positives),
        "negative_qa_count": len(negatives),
        "all_gold_event_recall_iou30_↑": scene_true_positive_30 / max(gold_count, 1),
        "all_gold_event_recall_iou50_↑": scene_true_positive_50 / max(gold_count, 1),
        "all_events_resolved_per_scene_iou30_↑": scene_all_resolved_30 / max(len(predictions), 1),
        "all_events_resolved_per_scene_iou50_↑": scene_all_resolved_50 / max(len(predictions), 1),
        "positive_anchor_proposal_recall_iou30_↑": positive_anchor_30 / p,
        "positive_anchor_proposal_recall_iou50_↑": positive_anchor_50 / p,
        "positive_answer_proposal_recall_iou30_↑": answer_30 / p,
        "positive_answer_proposal_recall_iou50_↑": answer_50 / p,
        "positive_distinct_anchor_answer_pair_recall_iou30_↑": pair_30 / p,
        "positive_distinct_anchor_answer_pair_recall_iou50_↑": pair_50 / p,
        "negative_anchor_proposal_recall_iou30_↑": negative_anchor_30 / n,
        "negative_anchor_proposal_recall_iou50_↑": negative_anchor_50 / n,
    }
    return metrics, scene_rows + qa_rows


def _run_dataset(
    spec: DatasetSpec,
    *,
    model: RelationalEventSlotsV1,
    checkpoint_sha: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    frozen = _json(spec.frozen_eval_receipt)
    if str(frozen["checkpoint_sha256"]) != checkpoint_sha:
        raise ValueError(f"{spec.name}: frozen evaluation used another checkpoint")
    store = DenseFeatureStore([spec.dense_index.resolve()], cache_size=8)
    scene_ids = load_scene_list(spec.scene_list.resolve())
    bound_qa = load_bound_qa_manifest_v2(
        spec.qa_manifest.resolve(), allowed_scene_ids=scene_ids, store=store, max_ordinal=10
    )
    dataset = SceneSlotDataset(store, scene_ids, preload=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_scene_slots,
    )
    predictions = collect_predictions(model, loader, device)
    curves = []
    rows_by_k: dict[str, list[dict[str, Any]]] = {}
    for k in TOP_K_VALUES:
        metrics, rows = _audit_k(predictions, bound_qa, k=k)
        curves.append(metrics)
        rows_by_k[str(k)] = rows
    return {
        "name": spec.name,
        "scene_count": len(scene_ids),
        "qa_count": len(bound_qa),
        "frozen_hysteresis_metrics": frozen["metrics"],
        "fixed_top_k_curve": curves,
        "oracle_all_8_query_ceiling": curves[-1],
        "dense_index": str(spec.dense_index.resolve()),
        "dense_index_sha256": _sha256_file(spec.dense_index.resolve()),
        "frozen_eval_receipt": str(spec.frozen_eval_receipt.resolve()),
        "frozen_eval_receipt_sha256": _sha256_file(spec.frozen_eval_receipt.resolve()),
        "diagnostic_rows_by_k": rows_by_k,
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha = _sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not a relational-event-slots v1 checkpoint")
    device = _device(args.device)
    model = RelationalEventSlotsV1(RelationalEventSlotsV1Config(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    specs = (
        DatasetSpec(
            "nonoverlap_locked_test",
            args.nonoverlap_dense_index,
            args.nonoverlap_scene_list,
            args.nonoverlap_qa_manifest,
            args.nonoverlap_eval_receipt,
        ),
        DatasetSpec(
            "overlap_onset_stress",
            args.overlap_dense_index,
            args.overlap_scene_list,
            args.overlap_qa_manifest,
            args.overlap_eval_receipt,
        ),
    )
    datasets = [
        _run_dataset(
            spec,
            model=model,
            checkpoint_sha=checkpoint_sha,
            device=device,
            batch_size=args.batch_size,
        )
        for spec in specs
    ]
    # Keep the receipt compact enough to inspect: per-example rows are only
    # needed for K=8 failure analysis and are written to a sidecar.
    cases_path = output.parent / "oracle_k8_cases.jsonl"
    with cases_path.open("w", encoding="utf-8") as handle:
        for dataset in datasets:
            for row in dataset["diagnostic_rows_by_k"]["8"]:
                handle.write(json.dumps({"dataset": dataset["name"], **row}, ensure_ascii=False, sort_keys=True) + "\n")
            del dataset["diagnostic_rows_by_k"]
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "num_interval_queries": int(checkpoint["config"]["num_slots"]),
        "test_threshold_tuning": False,
        "answer_label_used_for_query_generation_or_ranking": False,
        "protocol": {
            "top_k_values_fixed_before_audit": list(TOP_K_VALUES),
            "ranker": "frozen DETR objectness only",
            "k8_interpretation": "oracle proposal capacity ceiling, not selected deployment decoder",
        },
        "datasets": {dataset["name"]: dataset for dataset in datasets},
        "oracle_k8_cases": {
            "path": str(cases_path),
            "sha256": _sha256_file(cases_path),
        },
    }
    _atomic_json(receipt, output)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
