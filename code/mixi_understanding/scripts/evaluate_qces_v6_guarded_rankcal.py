#!/usr/bin/env python3
"""Evaluate guarded QCES-RankCal policies.

The previous calibration exposed a real trade-off:

* pair_answer_first: high answer accuracy, poor no-evidence recall;
* pair_no_evidence_head: safer no-evidence, lower answer accuracy.

This script searches train-only calibration families for a guarded policy that
uses both signals:

    high no-evidence probability -> abstain
    high pair score and top1-top2 margin -> answer-first
    otherwise -> fallback

It reads existing frozen proposal caches and the saved pair/no-evidence models.
It does not re-run AudioSep and does not modify Claude's original pipeline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
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
from mixi_understanding.qces.relational_planner import plan_from_proposals
from mixi_understanding.scripts.calibrate_qces_v6_pair_reranker import (
    pair_scores,
    record_feature_table,
    top_candidate_by_record,
)
from mixi_understanding.scripts.evaluate_qces_v6_pipeline import interval_iou
from mixi_understanding.scripts.train_qces_v6_pair_reranker import (
    Candidate,
    evaluate_candidates,
    family_calibration_mask,
    prepare_split,
    split_candidates,
)

FORMAT_VERSION = "qces_v6_guarded_rankcal_v1"


@dataclass(frozen=True)
class GuardPolicy:
    name: str
    objective: str
    noev_block_threshold: float
    pair_score_threshold: float
    margin_threshold: float
    fallback: str
    fallback_noev_threshold: float
    calibration_summary: dict[str, Any]


@dataclass(frozen=True)
class PreparedEval:
    records: list[QCESV5Record]
    candidates: list[Candidate]
    pair_scores: np.ndarray
    top: dict[str, tuple[float, Candidate, float | None, int]]
    noev_probs: dict[str, float]
    baseline: dict[str, tuple[bool, str | None, tuple[EventProposal, ...]]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrated-policy",
        type=Path,
        required=True,
        help="calibrated_policy.joblib from calibrate_qces_v6_pair_reranker.py",
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--proposal-head", type=Path, required=True)
    parser.add_argument("--eval", action="append", nargs=3, metavar=("NAME", "MANIFEST", "CACHE"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calib-family-frac", type=float, default=0.2)
    parser.add_argument("--strict-noev", type=float, default=0.65)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def harmonic(answer: float | None, noev: float | None) -> float:
    if answer is None or noev is None or answer <= 0.0 or noev <= 0.0:
        return 0.0
    return 2.0 * answer * noev / (answer + noev)


def baseline_predictions(
    prepared: tuple[
        list[QCESV5Record],
        dict[str, Any],
        dict[str, tuple[str, ...]],
        dict[str, dict[tuple[str, tuple[str, ...]], list[EventProposal]]],
    ],
) -> dict[str, tuple[bool, str | None, tuple[EventProposal, ...]]]:
    records, parsed, labels_by_id, decoded_by_relation = prepared
    result: dict[str, tuple[bool, str | None, tuple[EventProposal, ...]]] = {}
    for record in records:
        question = parsed[record.sample_id]
        proposals = decoded_by_relation.get(question.relation or "", {}).get(
            (record.scene_id, labels_by_id[record.sample_id]), []
        )
        plan = plan_from_proposals(question, proposals)
        result[record.sample_id] = (
            bool(plan.no_evidence),
            plan.answer_label,
            tuple(plan.evidence),
        )
    return result


def prepare_eval(
    *,
    manifest: Path,
    cache_path: Path,
    dataset_config: Path,
    proposal_head: Path,
    relation_thresholds: Mapping[str, float],
    pair_model: Any,
    noev_model: Any,
    max_neighbors: int,
    device: torch.device,
) -> PreparedEval:
    prepared = prepare_split(
        manifest=manifest.resolve(),
        cache_path=cache_path.resolve(),
        dataset_config=dataset_config.resolve(),
        proposal_head=proposal_head.resolve(),
        thresholds=relation_thresholds,
        device=device,
    )
    features, _, candidates, _ = split_candidates(
        prepared, max_neighbors=max_neighbors
    )
    scores = pair_scores(pair_model, features)
    rec_features, _, rec_list, top = record_feature_table(
        records=prepared[0], candidates=candidates, scores=scores
    )
    probs = np.asarray(noev_model.predict_proba(rec_features)[:, 1], dtype=np.float32)
    noev_probs = {
        record.sample_id: float(prob) for record, prob in zip(rec_list, probs)
    }
    return PreparedEval(
        records=list(prepared[0]),
        candidates=candidates,
        pair_scores=scores,
        top=top_candidate_by_record(candidates, scores),
        noev_probs=noev_probs,
        baseline=baseline_predictions(prepared),
    )


def decide_record(
    *,
    record: QCESV5Record,
    state: PreparedEval,
    policy: GuardPolicy,
    answer_first_threshold: float,
) -> tuple[bool, str | None, tuple[EventProposal, ...], float | None, float | None]:
    noev_prob = state.noev_probs.get(record.sample_id, 1.0)
    selection = state.top.get(record.sample_id)
    if selection is None:
        return True, None, (), None, noev_prob
    top_score, candidate, second, _count = selection
    second_score = 0.0 if second is None else float(second)
    margin = float(top_score) - second_score

    if noev_prob >= policy.noev_block_threshold:
        return True, None, (), float(top_score), noev_prob
    if top_score >= policy.pair_score_threshold and margin >= policy.margin_threshold:
        return False, candidate.answer_label, candidate.events, float(top_score), noev_prob

    if policy.fallback == "no_evidence":
        return True, None, (), float(top_score), noev_prob
    if policy.fallback == "baseline":
        pred_noev, answer, events = state.baseline[record.sample_id]
        return pred_noev, answer, events, float(top_score), noev_prob
    if policy.fallback == "noev_head":
        if noev_prob >= policy.fallback_noev_threshold or top_score < answer_first_threshold:
            return True, None, (), float(top_score), noev_prob
        return False, candidate.answer_label, candidate.events, float(top_score), noev_prob
    raise ValueError(f"unknown fallback: {policy.fallback}")


def evaluate_policy(
    *,
    state: PreparedEval,
    policy: GuardPolicy,
    answer_first_threshold: float,
    collect_items: bool = True,
    compute_iou: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answer_correct = answer_total = 0
    pure_noev_correct = pure_noev_total = 0
    decision_correct = 0
    abstain = 0
    span_iou_sum = span_iou_count = 0
    by_relation: dict[str, list[int]] = {}
    items: list[dict[str, Any]] = []
    for record in state.records:
        pred_noev, pred_answer, events, score, noev_prob = decide_record(
            record=record,
            state=state,
            policy=policy,
            answer_first_threshold=answer_first_threshold,
        )
        decision_correct += int(pred_noev == bool(record.no_evidence))
        if record.no_evidence:
            pure_noev_total += 1
            pure_noev_correct += int(pred_noev)
            answer_ok = None
            span_iou = None
        else:
            answer_total += 1
            answer_ok = pred_answer == record.answer
            answer_correct += int(answer_ok)
            abstain += int(pred_noev)
            if compute_iou:
                gold_spans = [
                    (
                        float(record.event_by_id(event_id).onset_seconds),
                        float(record.event_by_id(event_id).offset_seconds),
                    )
                    for event_id in record.evidence_event_ids
                ]
                predicted_spans = [
                    (item.onset_seconds, item.offset_seconds) for item in events
                ]
                span_iou = interval_iou(predicted_spans, gold_spans)
                span_iou_sum += span_iou
                span_iou_count += 1
            else:
                span_iou = None
            by_relation.setdefault(record.relation, [0, 0])
            by_relation[record.relation][0] += int(answer_ok)
            by_relation[record.relation][1] += 1
        if collect_items:
            items.append(
                {
                    "id": record.sample_id,
                    "scene_id": record.scene_id,
                    "scene_family_id": record.scene_family_id,
                    "relation": record.relation,
                    "question": record.question,
                    "answer": record.answer,
                    "no_evidence": bool(record.no_evidence),
                    "predicted_no_evidence": pred_noev,
                    "predicted_answer": pred_answer,
                    "answer_correct": answer_ok,
                    "score": score,
                    "noev_probability": noev_prob,
                    "span_iou_↑": span_iou,
                    "selected_events": [
                        {
                            "label": item.label,
                            "onset_seconds": item.onset_seconds,
                            "offset_seconds": item.offset_seconds,
                            "confidence": item.confidence,
                        }
                        for item in events
                    ],
                }
            )
    summary = {
        "record_count": len(state.records),
        "answerable_count": answer_total,
        "no_evidence_count": pure_noev_total,
        "answer_accuracy_↑": answer_correct / answer_total if answer_total else None,
        "no_evidence_pure_accuracy_↑": (
            pure_noev_correct / pure_noev_total if pure_noev_total else None
        ),
        "no_evidence_decision_accuracy_↑": (
            decision_correct / len(state.records) if state.records else None
        ),
        "abstain_rate_on_answerable_↓": abstain / answer_total if answer_total else None,
        "span_iou_mean_↑": span_iou_sum / span_iou_count if span_iou_count else None,
        "answer_accuracy_by_relation_↑": {
            relation: value[0] / value[1] for relation, value in sorted(by_relation.items())
        },
    }
    return summary, items


def baseline_summary_from_state(state: PreparedEval) -> dict[str, Any]:
    answer_correct = answer_total = 0
    noev_correct = noev_total = 0
    abstain = 0
    by_relation: dict[str, list[int]] = {}
    for record in state.records:
        pred_noev, pred_answer, _events = state.baseline[record.sample_id]
        if record.no_evidence:
            noev_total += 1
            noev_correct += int(pred_noev)
        else:
            answer_total += 1
            ok = pred_answer == record.answer
            answer_correct += int(ok)
            abstain += int(pred_noev)
            by_relation.setdefault(record.relation, [0, 0])
            by_relation[record.relation][0] += int(ok)
            by_relation[record.relation][1] += 1
    return {
        "answer_accuracy_↑": answer_correct / answer_total if answer_total else None,
        "no_evidence_pure_accuracy_↑": noev_correct / noev_total if noev_total else None,
        "abstain_rate_on_answerable_↓": abstain / answer_total if answer_total else None,
        "answer_accuracy_by_relation_↑": {
            relation: value[0] / value[1] for relation, value in sorted(by_relation.items())
        },
    }


def threshold_grids(
    *,
    state: PreparedEval,
    base_answer_threshold: float,
    base_noev_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    top_scores = []
    margins = []
    probs = []
    for record in state.records:
        probs.append(state.noev_probs.get(record.sample_id, 1.0))
        selection = state.top.get(record.sample_id)
        if selection is None:
            continue
        top_score, _candidate, second, _count = selection
        second_score = 0.0 if second is None else float(second)
        top_scores.append(float(top_score))
        margins.append(float(top_score) - second_score)
    score_grid = np.unique(
        np.r_[
            [base_answer_threshold, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70],
            np.quantile(top_scores, [0.20, 0.40, 0.60, 0.80, 0.90]) if top_scores else [],
        ]
    )
    margin_grid = np.unique(
        np.r_[
            [0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20],
            np.quantile(margins, [0.25, 0.50, 0.75, 0.90]) if margins else [],
        ]
    )
    noev_grid = np.unique(
        np.r_[
            [base_noev_threshold, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
            np.quantile(probs, [0.10, 0.25, 0.50, 0.75, 0.90]) if probs else [],
        ]
    )
    return score_grid.astype(float), margin_grid.astype(float), noev_grid.astype(float)


def choose_policy(
    *,
    state: PreparedEval,
    answer_first_threshold: float,
    base_noev_threshold: float,
    objective: str,
    min_noev: float | None = None,
) -> GuardPolicy:
    score_grid, margin_grid, noev_grid = threshold_grids(
        state=state,
        base_answer_threshold=answer_first_threshold,
        base_noev_threshold=base_noev_threshold,
    )
    fallbacks = ("no_evidence", "baseline", "noev_head")
    best_any: tuple[tuple[float, ...], GuardPolicy] | None = None
    best_feasible: tuple[tuple[float, ...], GuardPolicy] | None = None
    for noev_t in noev_grid:
        for score_t in score_grid:
            for margin_t in margin_grid:
                for fallback in fallbacks:
                    provisional = GuardPolicy(
                        name="candidate",
                        objective=objective,
                        noev_block_threshold=float(noev_t),
                        pair_score_threshold=float(score_t),
                        margin_threshold=float(margin_t),
                        fallback=fallback,
                        fallback_noev_threshold=float(base_noev_threshold),
                        calibration_summary={},
                    )
                    summary, _ = evaluate_policy(
                        state=state,
                        policy=provisional,
                        answer_first_threshold=answer_first_threshold,
                        collect_items=False,
                        compute_iou=False,
                    )
                    answer = float(summary["answer_accuracy_↑"])
                    noev = float(summary["no_evidence_pure_accuracy_↑"])
                    decision = float(summary["no_evidence_decision_accuracy_↑"])
                    abstain = float(summary["abstain_rate_on_answerable_↓"])
                    h = harmonic(answer, noev)
                    if objective == "hmean":
                        key = (h, answer + noev, decision, -abstain)
                    elif objective == "max_answer_min_noev":
                        key = (answer, decision, h, -abstain)
                    else:
                        raise ValueError(objective)
                    policy = GuardPolicy(
                        name=objective if min_noev is None else f"{objective}_{min_noev:.2f}",
                        objective=objective,
                        noev_block_threshold=float(noev_t),
                        pair_score_threshold=float(score_t),
                        margin_threshold=float(margin_t),
                        fallback=fallback,
                        fallback_noev_threshold=float(base_noev_threshold),
                        calibration_summary=summary,
                    )
                    if best_any is None or key > best_any[0]:
                        best_any = (key, policy)
                    if min_noev is None or noev >= min_noev:
                        if best_feasible is None or key > best_feasible[0]:
                            best_feasible = (key, policy)
    chosen = best_feasible or best_any
    assert chosen is not None
    return chosen[1]


def markdown_table(report: Mapping[str, Any]) -> str:
    rows = []
    for split, split_rows in report["eval_results"].items():
        for mode, summary in split_rows.items():
            rows.append(
                [
                    split,
                    mode,
                    f"{summary['answer_accuracy_↑']:.3f}",
                    f"{summary['no_evidence_pure_accuracy_↑']:.3f}",
                    f"{summary['no_evidence_decision_accuracy_↑']:.3f}",
                    f"{summary['abstain_rate_on_answerable_↓']:.3f}",
                    (
                        "--"
                        if summary.get("span_iou_mean_↑") is None
                        else f"{summary['span_iou_mean_↑']:.3f}"
                    ),
                ]
            )
    header = [
        "| Split | Policy | answer ↑ | no-evid ↑ | decision ↑ | abstain ↓ | IoU ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(
        [
            "# QCES guarded RankCal results",
            "",
            "Policies are selected only on train-calibration families.",
            "",
            *header,
            *body,
            "",
            "## Selected policy parameters",
            "",
            "```json",
            json.dumps(report["policies"], indent=2, ensure_ascii=False),
            "```",
        ]
    )


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
    policy_payload = joblib.load(args.calibrated_policy.resolve())
    pair_payload = policy_payload["pair_model_payload"]
    pair_model = pair_payload["model"]
    noev_model = policy_payload["noev_model"]
    relation_thresholds = pair_payload["relation_thresholds"]
    max_neighbors = int(pair_payload["max_neighbors"])
    answer_first_threshold = float(pair_payload["threshold"])
    base_noev_threshold = float(policy_payload["noev_threshold"])

    train_state_full = prepare_eval(
        manifest=args.train_manifest,
        cache_path=args.train_cache,
        dataset_config=args.dataset_config,
        proposal_head=args.proposal_head,
        relation_thresholds=relation_thresholds,
        pair_model=pair_model,
        noev_model=noev_model,
        max_neighbors=max_neighbors,
        device=device,
    )
    calib_mask_candidates = family_calibration_mask(
        train_state_full.candidates,
        {record.sample_id: record for record in train_state_full.records},
        frac=args.calib_family_frac,
    )
    calib_ids = {
        candidate.sample_id
        for candidate, is_calib in zip(train_state_full.candidates, calib_mask_candidates)
        if is_calib
    }
    calib_records = [record for record in train_state_full.records if record.sample_id in calib_ids]
    calib_candidates = [
        candidate
        for candidate, is_calib in zip(train_state_full.candidates, calib_mask_candidates)
        if is_calib
    ]
    calib_scores = train_state_full.pair_scores[calib_mask_candidates]
    # Slice state to calibration records/candidates but keep predictions reusable.
    calib_top = top_candidate_by_record(calib_candidates, calib_scores)
    calib_state = PreparedEval(
        records=calib_records,
        candidates=calib_candidates,
        pair_scores=calib_scores,
        top=calib_top,
        noev_probs={rid: train_state_full.noev_probs[rid] for rid in calib_ids},
        baseline={rid: train_state_full.baseline[rid] for rid in calib_ids},
    )
    train_baseline = baseline_summary_from_state(train_state_full)
    min_noev_baseline = float(train_baseline["no_evidence_pure_accuracy_↑"])
    policies = [
        choose_policy(
            state=calib_state,
            answer_first_threshold=answer_first_threshold,
            base_noev_threshold=base_noev_threshold,
            objective="hmean",
            min_noev=None,
        ),
        choose_policy(
            state=calib_state,
            answer_first_threshold=answer_first_threshold,
            base_noev_threshold=base_noev_threshold,
            objective="max_answer_min_noev",
            min_noev=min_noev_baseline,
        ),
        choose_policy(
            state=calib_state,
            answer_first_threshold=answer_first_threshold,
            base_noev_threshold=base_noev_threshold,
            objective="max_answer_min_noev",
            min_noev=float(args.strict_noev),
        ),
    ]

    eval_results: dict[str, Any] = {}
    eval_items: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, manifest_text, cache_text in args.eval:
        state = prepare_eval(
            manifest=Path(manifest_text),
            cache_path=Path(cache_text),
            dataset_config=args.dataset_config,
            proposal_head=args.proposal_head,
            relation_thresholds=relation_thresholds,
            pair_model=pair_model,
            noev_model=noev_model,
            max_neighbors=max_neighbors,
            device=device,
        )
        split_results: dict[str, Any] = {}
        split_items: dict[str, list[dict[str, Any]]] = {}
        for policy in policies:
            summary, items = evaluate_policy(
                state=state,
                policy=policy,
                answer_first_threshold=answer_first_threshold,
            )
            split_results[policy.name] = summary
            split_items[policy.name] = items
        eval_results[name] = split_results
        eval_items[name] = split_items

    report = {
        "format": FORMAT_VERSION,
        "device": str(device),
        "calibrated_policy": str(args.calibrated_policy.resolve()),
        "answer_first_threshold": answer_first_threshold,
        "base_noev_threshold": base_noev_threshold,
        "strict_noev_target": args.strict_noev,
        "train_baseline_noev_target": min_noev_baseline,
        "policies": [asdict(policy) for policy in policies],
        "eval_results": eval_results,
    }
    (output_dir / "guarded_rankcal_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "guarded_rankcal_results.md").write_text(
        markdown_table(report) + "\n",
        encoding="utf-8",
    )
    for split, modes in eval_items.items():
        for mode, items in modes.items():
            path = output_dir / f"{split}__{mode}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for item in items:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(output_dir / "guarded_rankcal_results.md")


if __name__ == "__main__":
    main()
