#!/usr/bin/env python3
"""Calibrate pair-reranker abstention / no-evidence decisions.

This script evaluates two policies on top of a trained QCES-v6 pair reranker:

1. ``balanced_score_threshold``: abstain when top pair score is below a
   threshold chosen on train-calibration families, with a minimum no-evidence
   recall constraint.
2. ``no_evidence_head``: a separate classifier predicts no-evidence from the
   top pair score, score margin, candidate count and relation/proposal features.

No validation annotation is used to choose thresholds or models.
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.scripts.train_qces_v6_pair_reranker import (
    Candidate,
    baseline_summary,
    evaluate_candidates,
    family_calibration_mask,
    prepare_split,
    split_candidates,
)

FORMAT_VERSION = "qces_v6_pair_reranker_calibration_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--proposal-head", type=Path, required=True)
    parser.add_argument("--eval", action="append", nargs=3, metavar=("NAME", "MANIFEST", "CACHE"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calib-family-frac", type=float, default=0.2)
    parser.add_argument("--min-noev-source", choices=["baseline", "fixed"], default="baseline")
    parser.add_argument("--fixed-min-noev", type=float, default=0.58)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_scores(model: Any, features: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float32)


def top_candidate_by_record(
    candidates: Sequence[Candidate], scores: np.ndarray
) -> dict[str, tuple[float, Candidate, float | None, int]]:
    grouped: dict[str, list[tuple[float, Candidate]]] = {}
    for score, candidate in zip(scores, candidates):
        grouped.setdefault(candidate.sample_id, []).append((float(score), candidate))
    result: dict[str, tuple[float, Candidate, float | None, int]] = {}
    for sample_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: item[0], reverse=True)
        top_score, top_candidate = ordered[0]
        second = ordered[1][0] if len(ordered) > 1 else None
        result[sample_id] = (top_score, top_candidate, second, len(ordered))
    return result


def record_feature_table(
    *,
    records: Sequence[QCESV5Record],
    candidates: Sequence[Candidate],
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[QCESV5Record], dict[str, tuple[float, Candidate, float | None, int]]]:
    top = top_candidate_by_record(candidates, scores)
    features: list[list[float]] = []
    targets: list[int] = []
    kept_records: list[QCESV5Record] = []
    for record in records:
        row = top.get(record.sample_id)
        rel_after = float(record.relation == "after")
        rel_before = float(record.relation == "before")
        rel_first = float(record.relation == "first")
        if row is None:
            feature = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, rel_after, rel_before, rel_first] + [0.0] * 19
        else:
            top_score, candidate, second, count = row
            second_score = 0.0 if second is None else float(second)
            margin = top_score - second_score
            feature = [
                top_score,
                second_score,
                margin,
                float(count),
                float(second is None),
                0.0,
                rel_after,
                rel_before,
                rel_first,
                *candidate.features,
            ]
        features.append(feature)
        targets.append(int(record.no_evidence))
        kept_records.append(record)
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.int8), kept_records, top


def evaluate_noev_head(
    *,
    records: Sequence[QCESV5Record],
    top: Mapping[str, tuple[float, Candidate, float | None, int]],
    noev_prob: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    answer_correct = answer_total = 0
    pure_noev_correct = pure_noev_total = 0
    decision_correct = 0
    abstain = 0
    by_relation: dict[str, list[int]] = {}
    for record, prob in zip(records, noev_prob):
        abstain_record = float(prob) >= threshold
        selection = top.get(record.sample_id)
        pred_label = None if abstain_record or selection is None else selection[1].answer_label
        decision_correct += int(abstain_record == bool(record.no_evidence))
        if record.no_evidence:
            pure_noev_total += 1
            pure_noev_correct += int(abstain_record)
        else:
            answer_total += 1
            abstain += int(abstain_record)
            ok = pred_label == record.answer
            answer_correct += int(ok)
            by_relation.setdefault(record.relation, [0, 0])
            by_relation[record.relation][0] += int(ok)
            by_relation[record.relation][1] += 1
    return {
        "answer_accuracy_↑": answer_correct / answer_total if answer_total else None,
        "no_evidence_pure_accuracy_↑": pure_noev_correct / pure_noev_total if pure_noev_total else None,
        "no_evidence_decision_accuracy_↑": decision_correct / len(records) if records else None,
        "abstain_rate_on_answerable_↓": abstain / answer_total if answer_total else None,
        "answer_accuracy_by_relation_↑": {
            relation: value[0] / value[1] for relation, value in sorted(by_relation.items())
        },
    }


def choose_threshold(
    *,
    records: Sequence[QCESV5Record],
    candidates: Sequence[Candidate],
    scores: np.ndarray,
    min_noev: float,
) -> tuple[float, dict[str, Any]]:
    thresholds = np.unique(
        np.r_[
            [-1e9],
            np.linspace(0.0, 0.9, 91),
            np.quantile(scores, np.linspace(0.0, 0.95, 101)),
        ]
    )
    best_feasible: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    best_any: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for threshold in thresholds:
        summary, _ = evaluate_candidates(
            records=records,
            candidates=candidates,
            scores=scores,
            threshold=float(threshold),
        )
        key = (
            float(summary["answer_accuracy_↑"]),
            float(summary["no_evidence_pure_accuracy_↑"]),
            -float(summary["abstain_rate_on_answerable_↓"]),
        )
        if best_any is None or key > best_any[0]:
            best_any = (key, float(threshold), summary)
        if float(summary["no_evidence_pure_accuracy_↑"]) >= min_noev:
            if best_feasible is None or key > best_feasible[0]:
                best_feasible = (key, float(threshold), summary)
    chosen = best_feasible or best_any
    assert chosen is not None
    return chosen[1], chosen[2]


def choose_noev_policy(
    *,
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    calib_features: np.ndarray,
    calib_records: Sequence[QCESV5Record],
    calib_top: Mapping[str, tuple[float, Candidate, float | None, int]],
    min_noev: float,
) -> tuple[str, float, Any, dict[str, Any]]:
    models = {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0),
        ),
        "rf": RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=7,
        ),
        "extra": ExtraTreesClassifier(
            n_estimators=220,
            max_depth=10,
            min_samples_leaf=8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=8,
        ),
    }
    best: tuple[tuple[float, float, float], str, float, Any, dict[str, Any]] | None = None
    for name, model in models.items():
        model.fit(fit_features, fit_targets)
        probs = np.asarray(model.predict_proba(calib_features)[:, 1], dtype=np.float32)
        thresholds = np.unique(
            np.r_[np.linspace(0.0, 1.0, 101), np.quantile(probs, np.linspace(0.0, 1.0, 81))]
        )
        for threshold in thresholds:
            summary = evaluate_noev_head(
                records=calib_records,
                top=calib_top,
                noev_prob=probs,
                threshold=float(threshold),
            )
            pure = float(summary["no_evidence_pure_accuracy_↑"])
            if pure < min_noev:
                continue
            key = (
                float(summary["answer_accuracy_↑"]),
                float(summary["no_evidence_decision_accuracy_↑"]),
                -float(summary["abstain_rate_on_answerable_↓"]),
            )
            if best is None or key > best[0]:
                best = (key, name, float(threshold), model, summary)
    if best is None:
        # Fall back to answer-first if no feasible point exists.
        for name, model in models.items():
            model.fit(fit_features, fit_targets)
            probs = np.asarray(model.predict_proba(calib_features)[:, 1], dtype=np.float32)
            for threshold in np.unique(
                np.r_[np.linspace(0.0, 1.0, 101), np.quantile(probs, np.linspace(0.0, 1.0, 81))]
            ):
                summary = evaluate_noev_head(
                    records=calib_records,
                    top=calib_top,
                    noev_prob=probs,
                    threshold=float(threshold),
                )
                key = (
                    float(summary["answer_accuracy_↑"]),
                    float(summary["no_evidence_pure_accuracy_↑"]),
                    -float(summary["abstain_rate_on_answerable_↓"]),
                )
                if best is None or key > best[0]:
                    best = (key, name, float(threshold), model, summary)
    assert best is not None
    return best[1], best[2], best[3], best[4]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    payload = joblib.load(args.model.resolve())
    pair_model = payload["model"]
    relation_thresholds = payload["relation_thresholds"]
    max_neighbors = int(payload["max_neighbors"])

    train_prepared = prepare_split(
        manifest=args.train_manifest.resolve(),
        cache_path=args.train_cache.resolve(),
        dataset_config=args.dataset_config.resolve(),
        proposal_head=args.proposal_head.resolve(),
        thresholds=relation_thresholds,
        device=device,
    )
    train_features, _, train_candidates, records_by_id = split_candidates(
        train_prepared, max_neighbors=max_neighbors
    )
    train_scores = pair_scores(pair_model, train_features)
    calib_mask_candidates = family_calibration_mask(
        train_candidates, records_by_id, frac=args.calib_family_frac
    )
    calib_ids = {
        candidate.sample_id
        for candidate, is_calib in zip(train_candidates, calib_mask_candidates)
        if is_calib
    }
    fit_ids = set(records_by_id) - calib_ids
    calib_records = [record for record in train_prepared[0] if record.sample_id in calib_ids]
    fit_records = [record for record in train_prepared[0] if record.sample_id in fit_ids]
    calib_candidates = [
        candidate
        for candidate, is_calib in zip(train_candidates, calib_mask_candidates)
        if is_calib
    ]
    calib_scores = train_scores[calib_mask_candidates]
    fit_candidates = [
        candidate
        for candidate, is_calib in zip(train_candidates, calib_mask_candidates)
        if not is_calib
    ]
    fit_scores = train_scores[~calib_mask_candidates]

    baseline_train = baseline_summary(train_prepared)
    min_noev = (
        float(baseline_train["no_evidence_pure_accuracy_↑"])
        if args.min_noev_source == "baseline"
        else args.fixed_min_noev
    )
    balanced_threshold, balanced_calib = choose_threshold(
        records=calib_records,
        candidates=calib_candidates,
        scores=calib_scores,
        min_noev=min_noev,
    )

    fit_record_features, fit_noev_targets, fit_record_list, fit_top = record_feature_table(
        records=fit_records, candidates=fit_candidates, scores=fit_scores
    )
    calib_record_features, calib_noev_targets, calib_record_list, calib_top = record_feature_table(
        records=calib_records, candidates=calib_candidates, scores=calib_scores
    )
    noev_model_name, noev_threshold, noev_model, noev_calib = choose_noev_policy(
        fit_features=fit_record_features,
        fit_targets=fit_noev_targets,
        calib_features=calib_record_features,
        calib_records=calib_record_list,
        calib_top=calib_top,
        min_noev=min_noev,
    )
    # Refit selected no-evidence model on all train records after calibration.
    all_record_features, all_noev_targets, all_record_list, all_top = record_feature_table(
        records=train_prepared[0], candidates=train_candidates, scores=train_scores
    )
    noev_model.fit(all_record_features, all_noev_targets)

    eval_results: dict[str, Any] = {}
    for name, manifest_text, cache_text in args.eval:
        prepared = prepare_split(
            manifest=Path(manifest_text).resolve(),
            cache_path=Path(cache_text).resolve(),
            dataset_config=args.dataset_config.resolve(),
            proposal_head=args.proposal_head.resolve(),
            thresholds=relation_thresholds,
            device=device,
        )
        features, _, candidates, _ = split_candidates(prepared, max_neighbors=max_neighbors)
        scores = pair_scores(pair_model, features)
        baseline = baseline_summary(prepared)
        answer_first, _ = evaluate_candidates(
            records=prepared[0],
            candidates=candidates,
            scores=scores,
            threshold=float(payload["threshold"]),
        )
        balanced, _ = evaluate_candidates(
            records=prepared[0],
            candidates=candidates,
            scores=scores,
            threshold=balanced_threshold,
        )
        rec_features, noev_targets, rec_list, top = record_feature_table(
            records=prepared[0], candidates=candidates, scores=scores
        )
        probs = np.asarray(noev_model.predict_proba(rec_features)[:, 1], dtype=np.float32)
        noev_head = evaluate_noev_head(
            records=rec_list,
            top=top,
            noev_prob=probs,
            threshold=noev_threshold,
        )
        eval_results[name] = {
            "baseline_relation_threshold": baseline,
            "pair_answer_first": answer_first,
            "pair_balanced_threshold": balanced,
            "pair_no_evidence_head": noev_head,
        }

    report = {
        "format": FORMAT_VERSION,
        "device": str(device),
        "pair_model": str(args.model.resolve()),
        "pair_model_sha256": sha256_file(args.model.resolve()),
        "relation_thresholds": relation_thresholds,
        "max_neighbors": max_neighbors,
        "min_noev_source": args.min_noev_source,
        "min_noev_target": min_noev,
        "balanced_threshold": balanced_threshold,
        "balanced_calibration_summary": balanced_calib,
        "noev_model_name": noev_model_name,
        "noev_threshold": noev_threshold,
        "noev_calibration_summary": noev_calib,
        "baseline_train": baseline_train,
        "eval_results": eval_results,
    }
    joblib.dump(
        {
            "format": FORMAT_VERSION,
            "pair_model_payload": payload,
            "balanced_threshold": balanced_threshold,
            "noev_model_name": noev_model_name,
            "noev_model": noev_model,
            "noev_threshold": noev_threshold,
        },
        output_dir / "calibrated_policy.joblib",
    )
    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
