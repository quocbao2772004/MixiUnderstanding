#!/usr/bin/env python3
"""Train a span-aware QCES-v6 pair reranker.

The original pair reranker labels a candidate positive when its answer label
matches the gold answer.  That is sufficient for answer accuracy but weak for
evidence grounding: the model is not penalised for choosing the wrong
occurrence of the same label.

This variant labels a candidate positive only when it both predicts the gold
answer label and overlaps the gold evidence span.  If no candidate for an
answerable record reaches the IoU threshold, the best overlapping candidate is
used as a fallback positive so the record still gives a learning signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import joblib
import numpy as np
import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.event_proposals import EventProposal
from mixi_understanding.scripts.evaluate_qces_v6_pipeline import interval_iou
from mixi_understanding.scripts.train_qces_v6_pair_reranker import (
    Candidate,
    baseline_summary,
    build_models,
    candidate_features,
    evaluate_candidates,
    family_calibration_mask,
    prepare_split,
    relation_thresholds,
    scores_for,
)


FORMAT_VERSION = "qces_v6_spanaware_pair_reranker_v1"
FEATURE_NAMES = (
    "rel_after",
    "rel_before",
    "rel_first",
    "anchor_or_candidate_conf",
    "neighbor_conf",
    "confidence_product",
    "anchor_or_candidate_onset_norm",
    "neighbor_onset_norm",
    "anchor_or_candidate_duration",
    "neighbor_duration",
    "anchor_or_candidate_occurrence_rank",
    "anchor_rank_abs_error",
    "anchor_rank_exact",
    "num_anchor_or_label_proposals",
    "num_total_proposals",
    "neighbor_distance_rank",
    "anchor_neighbor_distance",
    "log1p_anchor_neighbor_distance",
    "neighbor_same_as_anchor",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--proposal-head", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calib-family-frac", type=float, default=0.2)
    parser.add_argument("--relation-threshold", action="append", default=[])
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--positive-iou-threshold", type=float, default=0.30)
    parser.add_argument("--models", nargs="+", default=["hgb", "extra", "rf"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gold_spans(record: QCESV5Record) -> list[tuple[float, float]]:
    return [
        (
            float(record.event_by_id(event_id).onset_seconds),
            float(record.event_by_id(event_id).offset_seconds),
        )
        for event_id in record.evidence_event_ids
    ]


def candidate_span_iou(record: QCESV5Record, events: Sequence[EventProposal]) -> float:
    predicted = [(float(event.onset_seconds), float(event.offset_seconds)) for event in events]
    return interval_iou(predicted, gold_spans(record))


def split_candidates_spanaware(
    prepared: tuple[
        list[QCESV5Record],
        dict[str, Any],
        dict[str, tuple[str, ...]],
        dict[str, dict[tuple[str, tuple[str, ...]], list[EventProposal]]],
    ],
    *,
    max_neighbors: int,
    positive_iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, list[Candidate], dict[str, QCESV5Record], np.ndarray]:
    records, parsed, labels_by_id, decoded_by_relation = prepared
    features: list[tuple[float, ...]] = []
    targets: list[int] = []
    candidate_ious: list[float] = []
    candidates: list[Candidate] = []
    records_by_id = {record.sample_id: record for record in records}
    grouped_indices: dict[str, list[int]] = {}

    for record in records:
        question = parsed[record.sample_id]
        if question.relation not in decoded_by_relation:
            continue
        proposals = decoded_by_relation[question.relation][
            (record.scene_id, labels_by_id[record.sample_id])
        ]
        for answer_label, events, feature_row in candidate_features(
            record, question, proposals, max_neighbors=max_neighbors
        ):
            iou = 0.0 if record.no_evidence else candidate_span_iou(record, events)
            label_ok = (not record.no_evidence) and answer_label == record.answer
            target = int(label_ok and iou >= positive_iou_threshold)
            candidate = Candidate(
                sample_id=record.sample_id,
                answer_label=answer_label,
                events=events,
                features=feature_row,
                target=target,
            )
            grouped_indices.setdefault(record.sample_id, []).append(len(candidates))
            candidates.append(candidate)
            features.append(feature_row)
            targets.append(target)
            candidate_ious.append(iou)

    # Fallback positive: if an answerable record has no threshold-positive
    # candidate but has a gold-label candidate with non-zero overlap, mark the
    # best-overlap one positive. This avoids dropping hard records entirely.
    targets_arr = np.asarray(targets, dtype=np.int8)
    ious_arr = np.asarray(candidate_ious, dtype=np.float32)
    for record in records:
        if record.no_evidence:
            continue
        indices = grouped_indices.get(record.sample_id, [])
        if not indices or bool(targets_arr[indices].any()):
            continue
        label_indices = [
            idx for idx in indices if candidates[idx].answer_label == record.answer
        ]
        if not label_indices:
            continue
        best = max(label_indices, key=lambda idx: float(ious_arr[idx]))
        if float(ious_arr[best]) > 0.0:
            targets_arr[best] = 1
            candidates[best] = Candidate(
                sample_id=candidates[best].sample_id,
                answer_label=candidates[best].answer_label,
                events=candidates[best].events,
                features=candidates[best].features,
                target=1,
            )

    return (
        np.asarray(features, dtype=np.float32),
        targets_arr,
        candidates,
        records_by_id,
        ious_arr,
    )


def choose_threshold_for_iou(
    *,
    records: Sequence[QCESV5Record],
    candidates: Sequence[Candidate],
    scores: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    thresholds = np.unique(
        np.r_[
            [-1e9],
            np.linspace(0.0, 0.9, 91),
            np.quantile(scores, np.linspace(0.0, 0.95, 101)),
        ]
    )
    best: tuple[tuple[float, float, float, float], float, dict[str, Any]] | None = None
    for threshold in thresholds:
        summary, _ = evaluate_candidates(
            records=records,
            candidates=candidates,
            scores=scores,
            threshold=float(threshold),
        )
        key = (
            float(summary["span_iou_mean_↑"]),
            float(summary["answer_accuracy_↑"]),
            float(summary["no_evidence_decision_accuracy_↑"]),
            -float(summary["abstain_rate_on_answerable_↓"]),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), summary)
    assert best is not None
    return best[1], best[2]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    thresholds = relation_thresholds(args.relation_threshold)

    train_prepared = prepare_split(
        manifest=args.train_manifest.resolve(),
        cache_path=args.train_cache.resolve(),
        dataset_config=args.dataset_config.resolve(),
        proposal_head=args.proposal_head.resolve(),
        thresholds=thresholds,
        device=device,
    )
    val_prepared = prepare_split(
        manifest=args.val_manifest.resolve(),
        cache_path=args.val_cache.resolve(),
        dataset_config=args.dataset_config.resolve(),
        proposal_head=args.proposal_head.resolve(),
        thresholds=thresholds,
        device=device,
    )
    train_features, train_targets, train_candidates, train_records_by_id, train_ious = (
        split_candidates_spanaware(
            train_prepared,
            max_neighbors=args.max_neighbors,
            positive_iou_threshold=args.positive_iou_threshold,
        )
    )
    val_features, val_targets, val_candidates, _val_records_by_id, val_ious = (
        split_candidates_spanaware(
            val_prepared,
            max_neighbors=args.max_neighbors,
            positive_iou_threshold=args.positive_iou_threshold,
        )
    )
    calib_mask = family_calibration_mask(
        train_candidates, train_records_by_id, frac=args.calib_family_frac
    )
    fit_mask = ~calib_mask
    if fit_mask.sum() == 0 or calib_mask.sum() == 0:
        raise SystemExit("empty fit/calibration split")
    calib_record_ids = {
        candidate.sample_id
        for candidate, is_calib in zip(train_candidates, calib_mask)
        if is_calib
    }
    calib_records = [record for record in train_prepared[0] if record.sample_id in calib_record_ids]
    calib_candidates = [
        candidate
        for candidate, is_calib in zip(train_candidates, calib_mask)
        if is_calib
    ]

    model_reports: dict[str, Any] = {}
    best_choice: tuple[tuple[float, float, float, float], str, float, Any] | None = None
    for model_name, model in build_models(args.models).items():
        model.fit(train_features[fit_mask], train_targets[fit_mask])
        calib_scores = scores_for(model, train_features[calib_mask])
        selected_threshold, calib_summary = choose_threshold_for_iou(
            records=calib_records,
            candidates=calib_candidates,
            scores=calib_scores,
        )
        final_model = build_models([model_name])[model_name]
        final_model.fit(train_features, train_targets)
        train_scores = scores_for(final_model, train_features)
        val_scores = scores_for(final_model, val_features)
        train_summary, _ = evaluate_candidates(
            records=train_prepared[0],
            candidates=train_candidates,
            scores=train_scores,
            threshold=selected_threshold,
        )
        val_summary, _ = evaluate_candidates(
            records=val_prepared[0],
            candidates=val_candidates,
            scores=val_scores,
            threshold=selected_threshold,
        )
        model_reports[model_name] = {
            "selected_threshold": selected_threshold,
            "calibration_summary": calib_summary,
            "train_summary_after_refit": train_summary,
            "val_summary": val_summary,
        }
        key = (
            float(calib_summary["span_iou_mean_↑"]),
            float(calib_summary["answer_accuracy_↑"]),
            float(calib_summary["no_evidence_decision_accuracy_↑"]),
            -float(calib_summary["abstain_rate_on_answerable_↓"]),
        )
        if best_choice is None or key > best_choice[0]:
            best_choice = (key, model_name, selected_threshold, final_model)

    assert best_choice is not None
    _key, best_name, best_threshold, best_model = best_choice
    val_scores = scores_for(best_model, val_features)
    best_val_summary, best_val_items = evaluate_candidates(
        records=val_prepared[0],
        candidates=val_candidates,
        scores=val_scores,
        threshold=best_threshold,
    )

    model_path = output_dir / "pair_reranker.joblib"
    joblib.dump(
        {
            "format": FORMAT_VERSION,
            "model_name": best_name,
            "model": best_model,
            "feature_names": FEATURE_NAMES,
            "threshold": best_threshold,
            "relation_thresholds": thresholds,
            "max_neighbors": args.max_neighbors,
            "positive_iou_threshold": args.positive_iou_threshold,
        },
        model_path,
    )
    (output_dir / "val_predictions.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in best_val_items
        ),
        encoding="utf-8",
    )
    report = {
        "format": FORMAT_VERSION,
        "device": str(device),
        "train_manifest": str(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "train_cache": str(args.train_cache.resolve()),
        "val_cache": str(args.val_cache.resolve()),
        "dataset_config": str(args.dataset_config.resolve()),
        "proposal_head": str(args.proposal_head.resolve()),
        "proposal_head_sha256": sha256_file(args.proposal_head.resolve()),
        "relation_thresholds": thresholds,
        "feature_names": FEATURE_NAMES,
        "max_neighbors": args.max_neighbors,
        "positive_iou_threshold": args.positive_iou_threshold,
        "calib_family_frac": args.calib_family_frac,
        "train_candidate_count": int(train_features.shape[0]),
        "val_candidate_count": int(val_features.shape[0]),
        "train_positive_rate": float(train_targets.mean()) if train_targets.size else None,
        "val_positive_rate": float(val_targets.mean()) if val_targets.size else None,
        "train_candidate_iou_mean": float(train_ious.mean()) if train_ious.size else None,
        "val_candidate_iou_mean": float(val_ious.mean()) if val_ious.size else None,
        "baseline_relation_threshold_train": baseline_summary(train_prepared),
        "baseline_relation_threshold_val": baseline_summary(val_prepared),
        "models": model_reports,
        "selected_model": best_name,
        "selected_threshold": best_threshold,
        "selected_val_summary": best_val_summary,
        "model_path": str(model_path),
    }
    (output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "selected_model": best_name,
                "threshold": best_threshold,
                "baseline_val": report["baseline_relation_threshold_val"],
                "selected_val": best_val_summary,
                "output": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
