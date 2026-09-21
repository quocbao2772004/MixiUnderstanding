#!/usr/bin/env python3
"""Calibrate a train-only temporal span refiner for QCES-v6 evidence IoU.

This script is intentionally separate from Claude's original QCES-v6 pipeline.
It keeps the existing frozen proposal head and pair reranker, then learns a
small post-processing policy over predicted event spans:

* per-relation pair-score threshold;
* per-relation left/right temporal padding.

The policy is selected on train-calibration scene families only and then applied
unchanged to val/test/fast384 manifests.  It never uses validation/test oracle
spans to choose hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import joblib
import numpy as np
import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.event_proposals import EventProposal
from mixi_understanding.scripts.calibrate_qces_v6_pair_reranker import (
    top_candidate_by_record,
)
from mixi_understanding.scripts.train_qces_v6_pair_reranker import (
    Candidate,
    family_calibration_mask,
    prepare_split,
    scores_for,
    split_candidates,
)


FORMAT_VERSION = "qces_v6_span_iou_refiner_v1"
RELATIONS = ("after", "before", "first")


@dataclass(frozen=True)
class RelationSpanPolicy:
    relation: str
    score_threshold: float
    left_padding_seconds: float
    right_padding_seconds: float
    calibration_mean_iou: float
    calibration_answer_accuracy: float
    calibration_no_evidence_accuracy: float
    calibration_mean_span_seconds: float


@dataclass(frozen=True)
class SpanPolicy:
    name: str
    objective: str
    relations: dict[str, RelationSpanPolicy]


@dataclass(frozen=True)
class PreparedSplit:
    name: str
    records: list[QCESV5Record]
    candidates: list[Candidate]
    scores: np.ndarray
    top: dict[str, tuple[float, Candidate, float | None, int]]
    records_by_id: dict[str, QCESV5Record]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--proposal-head", type=Path, required=True)
    parser.add_argument(
        "--eval",
        action="append",
        nargs=3,
        metavar=("NAME", "MANIFEST", "CACHE"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calib-family-frac", type=float, default=0.2)
    parser.add_argument("--padding-max-seconds", type=float, default=0.60)
    parser.add_argument("--padding-step-seconds", type=float, default=0.04)
    parser.add_argument("--min-noev-for-balanced", type=float, default=0.60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def merge_intervals(spans: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    clean = sorted((max(0.0, float(a)), min(10.0, float(b))) for a, b in spans if b > a)
    merged: list[list[float]] = []
    for start, end in clean:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(a, b) for a, b in merged]


def interval_length(spans: Sequence[tuple[float, float]]) -> float:
    return sum(max(0.0, b - a) for a, b in merge_intervals(spans))


def intersection_length(
    left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]
) -> float:
    a = merge_intervals(left)
    b = merge_intervals(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def temporal_iou(
    predicted: Sequence[tuple[float, float]], gold: Sequence[tuple[float, float]]
) -> float:
    pred = merge_intervals(predicted)
    target = merge_intervals(gold)
    union = interval_length(pred) + interval_length(target) - intersection_length(pred, target)
    if union <= 0.0:
        return 1.0
    return intersection_length(pred, target) / union


def gold_spans(record: QCESV5Record) -> list[tuple[float, float]]:
    return [
        (
            float(record.event_by_id(event_id).onset_seconds),
            float(record.event_by_id(event_id).offset_seconds),
        )
        for event_id in record.evidence_event_ids
    ]


def event_to_span(event: EventProposal) -> tuple[float, float]:
    return (float(event.onset_seconds), float(event.offset_seconds))


def refine_events(
    events: Sequence[EventProposal],
    *,
    left_padding: float,
    right_padding: float,
) -> list[EventProposal]:
    refined: list[EventProposal] = []
    for event in events:
        start = max(0.0, float(event.onset_seconds) - left_padding)
        end = min(10.0, float(event.offset_seconds) + right_padding)
        if end <= start:
            continue
        refined.append(
            EventProposal(
                label=event.label,
                onset_seconds=start,
                offset_seconds=end,
                confidence=event.confidence,
            )
        )
    return refined


def prepare_named_split(
    *,
    name: str,
    manifest: Path,
    cache: Path,
    dataset_config: Path,
    proposal_head: Path,
    relation_thresholds: Mapping[str, float],
    model: Any,
    max_neighbors: int,
    device: torch.device,
) -> PreparedSplit:
    prepared = prepare_split(
        manifest=manifest.resolve(),
        cache_path=cache.resolve(),
        dataset_config=dataset_config.resolve(),
        proposal_head=proposal_head.resolve(),
        thresholds=relation_thresholds,
        device=device,
    )
    features, _targets, candidates, records_by_id = split_candidates(
        prepared, max_neighbors=max_neighbors
    )
    scores = scores_for(model, features)
    return PreparedSplit(
        name=name,
        records=list(prepared[0]),
        candidates=candidates,
        scores=scores,
        top=top_candidate_by_record(candidates, scores),
        records_by_id=records_by_id,
    )


def score_grid(
    records: Sequence[QCESV5Record],
    top: Mapping[str, tuple[float, Candidate, float | None, int]],
    *,
    base_threshold: float,
) -> np.ndarray:
    values = [
        float(top[record.sample_id][0])
        for record in records
        if record.sample_id in top
    ]
    if not values:
        return np.asarray([base_threshold], dtype=np.float32)
    return np.unique(
        np.r_[
            [-1e9, base_threshold],
            np.linspace(0.0, 0.90, 46),
            np.quantile(values, np.linspace(0.0, 0.95, 40)),
        ].astype(np.float32)
    )


def padding_grid(max_seconds: float, step_seconds: float) -> np.ndarray:
    count = int(math.floor(max_seconds / step_seconds + 1e-9))
    return np.asarray([round(index * step_seconds, 6) for index in range(count + 1)])


def evaluate_relation_policy(
    records: Sequence[QCESV5Record],
    top: Mapping[str, tuple[float, Candidate, float | None, int]],
    *,
    threshold: float,
    left_padding: float,
    right_padding: float,
) -> dict[str, float]:
    answer_total = answer_correct = noev_total = noev_correct = 0
    iou_sum = iou_count = 0
    span_seconds_sum = active_count = 0
    for record in records:
        selection = top.get(record.sample_id)
        if selection is None or float(selection[0]) < threshold:
            pred_noev = True
            pred_answer = None
            events: list[EventProposal] = []
        else:
            pred_noev = False
            pred_answer = selection[1].answer_label
            events = refine_events(
                selection[1].events,
                left_padding=left_padding,
                right_padding=right_padding,
            )
        if record.no_evidence:
            noev_total += 1
            noev_correct += int(pred_noev)
            continue
        answer_total += 1
        answer_correct += int(pred_answer == record.answer and not pred_noev)
        spans = [event_to_span(event) for event in events]
        iou_sum += temporal_iou(spans, gold_spans(record))
        iou_count += 1
        seconds = interval_length(spans)
        if seconds > 0:
            active_count += 1
            span_seconds_sum += seconds
    return {
        "mean_iou": iou_sum / iou_count if iou_count else 0.0,
        "answer_accuracy": answer_correct / answer_total if answer_total else 0.0,
        "no_evidence_accuracy": noev_correct / noev_total if noev_total else 0.0,
        "mean_span_seconds": span_seconds_sum / active_count if active_count else 0.0,
    }


def calibrate_relation(
    *,
    relation: str,
    records: Sequence[QCESV5Record],
    top: Mapping[str, tuple[float, Candidate, float | None, int]],
    base_threshold: float,
    pads: Sequence[float],
    objective: str,
    min_noev: float,
) -> RelationSpanPolicy:
    rel_records = [record for record in records if record.relation == relation]
    thresholds = score_grid(rel_records, top, base_threshold=base_threshold)
    answerable = np.asarray([not bool(record.no_evidence) for record in rel_records], dtype=bool)
    no_evidence = ~answerable
    answer_total = max(1, int(answerable.sum()))
    noev_total = max(1, int(no_evidence.sum()))
    scores = np.full(len(rel_records), -np.inf, dtype=np.float32)
    answer_ok = np.zeros(len(rel_records), dtype=bool)
    raw_events_by_record: list[tuple[EventProposal, ...]] = []
    gold_by_record: list[list[tuple[float, float]]] = []
    for index, record in enumerate(rel_records):
        selection = top.get(record.sample_id)
        if selection is None:
            raw_events_by_record.append(())
        else:
            score, candidate, _second, _count = selection
            scores[index] = float(score)
            answer_ok[index] = (not bool(record.no_evidence)) and candidate.answer_label == record.answer
            raw_events_by_record.append(tuple(candidate.events))
        gold_by_record.append(gold_spans(record) if not bool(record.no_evidence) else [])

    active_by_threshold = scores[None, :] >= thresholds[:, None]
    answer_acc_by_threshold = (
        (active_by_threshold & answer_ok[None, :] & answerable[None, :]).sum(axis=1)
        / answer_total
    )
    noev_acc_by_threshold = (
        ((~active_by_threshold) & no_evidence[None, :]).sum(axis=1) / noev_total
    )

    best_any: tuple[tuple[float, ...], RelationSpanPolicy] | None = None
    best_feasible: tuple[tuple[float, ...], RelationSpanPolicy] | None = None
    for left in pads:
        for right in pads:
            ious = np.zeros(len(rel_records), dtype=np.float32)
            span_seconds = np.zeros(len(rel_records), dtype=np.float32)
            for index, record in enumerate(rel_records):
                if not answerable[index] or not raw_events_by_record[index]:
                    continue
                refined = refine_events(
                    raw_events_by_record[index],
                    left_padding=float(left),
                    right_padding=float(right),
                )
                spans = [event_to_span(event) for event in refined]
                ious[index] = temporal_iou(spans, gold_by_record[index])
                span_seconds[index] = interval_length(spans)
            refined_iou_by_threshold = (
                (active_by_threshold * ious[None, :] * answerable[None, :]).sum(axis=1)
                / answer_total
            )
            active_answerable_with_span = (
                active_by_threshold & answerable[None, :] & (span_seconds[None, :] > 0.0)
            )
            span_sum = (active_by_threshold * span_seconds[None, :] * answerable[None, :]).sum(axis=1)
            span_count = active_answerable_with_span.sum(axis=1)
            mean_span_seconds_by_threshold = np.divide(
                span_sum,
                np.maximum(span_count, 1),
                out=np.zeros_like(span_sum, dtype=np.float64),
                where=span_count > 0,
            )
            for idx, threshold in enumerate(thresholds):
                mean_iou = float(refined_iou_by_threshold[idx])
                answer_acc = float(answer_acc_by_threshold[idx])
                noev_acc = float(noev_acc_by_threshold[idx])
                mean_span = float(mean_span_seconds_by_threshold[idx])
                key = (mean_iou, answer_acc, noev_acc, -mean_span)
                policy = RelationSpanPolicy(
                    relation=relation,
                    score_threshold=float(threshold),
                    left_padding_seconds=float(left),
                    right_padding_seconds=float(right),
                    calibration_mean_iou=mean_iou,
                    calibration_answer_accuracy=answer_acc,
                    calibration_no_evidence_accuracy=noev_acc,
                    calibration_mean_span_seconds=mean_span,
                )
                if best_any is None or key > best_any[0]:
                    best_any = (key, policy)
                if objective == "iou_max" or noev_acc >= min_noev:
                    if best_feasible is None or key > best_feasible[0]:
                        best_feasible = (key, policy)
    chosen = best_feasible or best_any
    assert chosen is not None
    return chosen[1]


def calibrate_policy(
    *,
    name: str,
    objective: str,
    records: Sequence[QCESV5Record],
    top: Mapping[str, tuple[float, Candidate, float | None, int]],
    base_threshold: float,
    pads: Sequence[float],
    min_noev: float,
) -> SpanPolicy:
    relations = {
        relation: calibrate_relation(
            relation=relation,
            records=records,
            top=top,
            base_threshold=base_threshold,
            pads=pads,
            objective=objective,
            min_noev=min_noev,
        )
        for relation in RELATIONS
    }
    return SpanPolicy(name=name, objective=objective, relations=relations)


def predict_record(
    record: QCESV5Record,
    top: Mapping[str, tuple[float, Candidate, float | None, int]],
    policy: SpanPolicy,
) -> tuple[bool, str | None, float | None, list[EventProposal], list[EventProposal]]:
    relation_policy = policy.relations.get(record.relation)
    selection = top.get(record.sample_id)
    if relation_policy is None or selection is None:
        return True, None, None, [], []
    score, candidate, _second, _count = selection
    if float(score) < relation_policy.score_threshold:
        return True, None, float(score), [], []
    raw_events = list(candidate.events)
    refined = refine_events(
        raw_events,
        left_padding=relation_policy.left_padding_seconds,
        right_padding=relation_policy.right_padding_seconds,
    )
    return False, candidate.answer_label, float(score), raw_events, refined


def event_rows(events: Sequence[EventProposal]) -> list[dict[str, Any]]:
    return [
        {
            "label": event.label,
            "onset_seconds": round(float(event.onset_seconds), 6),
            "offset_seconds": round(float(event.offset_seconds), 6),
            "confidence": float(event.confidence),
        }
        for event in events
    ]


def evaluate_split(split: PreparedSplit, policy: SpanPolicy) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    answer_total = answer_correct = noev_total = noev_correct = decision_correct = 0
    raw_iou_sum = refined_iou_sum = iou_count = 0
    by_relation: dict[str, dict[str, float]] = {
        relation: {"sum": 0.0, "count": 0.0} for relation in RELATIONS
    }
    for record in split.records:
        pred_noev, pred_answer, score, raw_events, refined_events = predict_record(
            record, split.top, policy
        )
        decision_correct += int(pred_noev == bool(record.no_evidence))
        if record.no_evidence:
            noev_total += 1
            noev_correct += int(pred_noev)
            raw_iou = None
            refined_iou = None
            answer_ok = None
        else:
            answer_total += 1
            answer_ok = (not pred_noev) and pred_answer == record.answer
            answer_correct += int(answer_ok)
            target = gold_spans(record)
            raw_iou = temporal_iou([event_to_span(event) for event in raw_events], target)
            refined_iou = temporal_iou(
                [event_to_span(event) for event in refined_events], target
            )
            raw_iou_sum += raw_iou
            refined_iou_sum += refined_iou
            iou_count += 1
            by_relation[record.relation]["sum"] += refined_iou
            by_relation[record.relation]["count"] += 1.0
        items.append(
            {
                "id": record.sample_id,
                "scene_id": record.scene_id,
                "scene_family_id": record.scene_family_id,
                "relation": record.relation,
                "question": record.question,
                "answer": record.answer,
                "no_evidence": bool(record.no_evidence),
                "predicted_no_evidence": bool(pred_noev),
                "predicted_answer": pred_answer,
                "answer_correct": answer_ok,
                "score": score,
                "span_iou_raw_↑": raw_iou,
                "span_iou_↑": refined_iou,
                "span_iou_delta_↑": (
                    None if raw_iou is None or refined_iou is None else refined_iou - raw_iou
                ),
                "selected_events_raw": event_rows(raw_events),
                "selected_events": event_rows(refined_events),
            }
        )
    summary = {
        "record_count": len(split.records),
        "answerable_count": answer_total,
        "no_evidence_count": noev_total,
        "answer_accuracy_↑": answer_correct / answer_total if answer_total else None,
        "no_evidence_accuracy_↑": noev_correct / noev_total if noev_total else None,
        "decision_accuracy_↑": decision_correct / len(split.records) if split.records else None,
        "span_iou_raw_mean_↑": raw_iou_sum / iou_count if iou_count else None,
        "span_iou_refined_mean_↑": refined_iou_sum / iou_count if iou_count else None,
        "span_iou_gain_↑": (
            (refined_iou_sum - raw_iou_sum) / iou_count if iou_count else None
        ),
        "span_iou_refined_by_relation_↑": {
            relation: (
                values["sum"] / values["count"] if values["count"] else None
            )
            for relation, values in by_relation.items()
        },
    }
    return summary, items


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# QCES-v6 span IoU refiner benchmark",
        "",
        "Policy is calibrated on train-calibration scene families only.",
        "",
        "| Split | Policy | answer ↑ | no-evid ↑ | decision ↑ | IoU raw ↑ | IoU refined ↑ | gain ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, policies in report["eval_results"].items():
        for policy, summary in policies.items():
            def f(value: Any) -> str:
                return "..." if value is None else f"{float(value):.3f}"

            lines.append(
                "| "
                + " | ".join(
                    [
                        split,
                        policy,
                        f(summary["answer_accuracy_↑"]),
                        f(summary["no_evidence_accuracy_↑"]),
                        f(summary["decision_accuracy_↑"]),
                        f(summary["span_iou_raw_mean_↑"]),
                        f(summary["span_iou_refined_mean_↑"]),
                        f(summary["span_iou_gain_↑"]),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Learned policy",
            "",
            "```json",
            json.dumps(report["policies"], ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    payload = joblib.load(args.model.resolve())
    model = payload["model"]
    base_threshold = float(payload["threshold"])
    relation_thresholds = payload["relation_thresholds"]
    max_neighbors = int(payload["max_neighbors"])

    train = prepare_named_split(
        name="train",
        manifest=args.train_manifest,
        cache=args.train_cache,
        dataset_config=args.dataset_config,
        proposal_head=args.proposal_head,
        relation_thresholds=relation_thresholds,
        model=model,
        max_neighbors=max_neighbors,
        device=device,
    )
    calib_mask = family_calibration_mask(
        train.candidates, train.records_by_id, frac=args.calib_family_frac
    )
    calib_ids = {
        candidate.sample_id
        for candidate, keep in zip(train.candidates, calib_mask)
        if keep
    }
    calib_records = [record for record in train.records if record.sample_id in calib_ids]
    calib_candidates = [
        candidate for candidate, keep in zip(train.candidates, calib_mask) if keep
    ]
    calib_scores = train.scores[calib_mask]
    calib_top = top_candidate_by_record(calib_candidates, calib_scores)
    pads = padding_grid(args.padding_max_seconds, args.padding_step_seconds)

    policies = [
        calibrate_policy(
            name="iou_max",
            objective="iou_max",
            records=calib_records,
            top=calib_top,
            base_threshold=base_threshold,
            pads=pads,
            min_noev=args.min_noev_for_balanced,
        ),
        calibrate_policy(
            name=f"iou_noev{args.min_noev_for_balanced:.2f}",
            objective="iou_noev_guarded",
            records=calib_records,
            top=calib_top,
            base_threshold=base_threshold,
            pads=pads,
            min_noev=args.min_noev_for_balanced,
        ),
    ]

    evals = [
        prepare_named_split(
            name=name,
            manifest=Path(manifest),
            cache=Path(cache),
            dataset_config=args.dataset_config,
            proposal_head=args.proposal_head,
            relation_thresholds=relation_thresholds,
            model=model,
            max_neighbors=max_neighbors,
            device=device,
        )
        for name, manifest, cache in args.eval
    ]

    eval_results: dict[str, Any] = {}
    for split in evals:
        eval_results[split.name] = {}
        for policy in policies:
            summary, items = evaluate_split(split, policy)
            eval_results[split.name][policy.name] = summary
            write_jsonl(out / f"{split.name}__{policy.name}.jsonl", items)

    policy_payload = {
        policy.name: {
            "objective": policy.objective,
            "relations": {
                relation: asdict(relation_policy)
                for relation, relation_policy in policy.relations.items()
            },
        }
        for policy in policies
    }
    report = {
        "format": FORMAT_VERSION,
        "model": str(args.model.resolve()),
        "dataset_config": str(args.dataset_config.resolve()),
        "proposal_head": str(args.proposal_head.resolve()),
        "device": str(device),
        "base_pair_threshold": base_threshold,
        "pair_relation_thresholds": relation_thresholds,
        "max_neighbors": max_neighbors,
        "calib_family_frac": args.calib_family_frac,
        "padding_grid_seconds": [float(value) for value in pads],
        "min_noev_for_balanced": args.min_noev_for_balanced,
        "policies": policy_payload,
        "eval_results": eval_results,
    }
    (out / "span_iou_refiner_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out / "span_iou_refiner_results.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main(sys.argv[1:])
