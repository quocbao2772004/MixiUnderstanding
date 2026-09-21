#!/usr/bin/env python3
"""Calibrate frame-activity temporal evidence spans for QCES-v6.

The previous span refiner only learned per-relation padding.  This variant uses
the proposal head's frame activity curve to re-cut each selected event inside a
local window, then calibrates the activity threshold and small padding on
train-calibration families only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch

from mixi_understanding.qces.event_proposals import EventProposal, ProposalHead
from mixi_understanding.qces.stem_features import FrameGrid
from mixi_understanding.scripts.calibrate_qces_v6_span_iou_refiner import (
    PreparedSplit,
    SpanPolicy,
    event_rows,
    event_to_span,
    gold_spans,
    interval_length,
    merge_intervals,
    padding_grid,
    prepare_named_split,
    score_grid,
    temporal_iou,
)
from mixi_understanding.scripts.evaluate_qces_v6_pipeline import scene_activity
from mixi_understanding.scripts.train_qces_v6_pair_reranker import family_calibration_mask


FORMAT_VERSION = "qces_v6_activity_span_refiner_v1"
RELATIONS = ("after", "before", "first")


@dataclass(frozen=True)
class ActivityRelationPolicy:
    relation: str
    score_threshold: float
    activity_threshold: float
    window_padding_seconds: float
    left_padding_seconds: float
    right_padding_seconds: float
    calibration_mean_iou: float
    calibration_answer_accuracy: float
    calibration_no_evidence_accuracy: float
    calibration_mean_span_seconds: float


@dataclass(frozen=True)
class ActivityPolicy:
    name: str
    objective: str
    relations: dict[str, ActivityRelationPolicy]


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
    parser.add_argument("--min-noev-for-balanced", type=float, default=0.60)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_activity(cache_path: Path, proposal_head: Path, device: torch.device) -> dict[str, Any]:
    cache = torch.load(cache_path.resolve(), map_location="cpu", weights_only=False)
    payload = torch.load(proposal_head.resolve(), map_location="cpu", weights_only=False)
    head = ProposalHead(
        channels=int(payload.get("channels", 96)),
        dropout=float(payload.get("dropout", 0.1)),
    )
    head.load_state_dict(payload["state_dict"])
    head = head.to(device).eval()
    zeroed = tuple(payload.get("zeroed_feature_groups", ()))
    return scene_activity(cache, head, device, zeroed)


def active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    if not bool(active.any()):
        return []
    padded = np.concatenate([[False], active.astype(bool), [False]])
    edges = padded[1:].astype(np.int8) - padded[:-1].astype(np.int8)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def frame_span(grid: FrameGrid, start: int, stop: int) -> tuple[float, float]:
    return grid.frame_to_seconds(start), grid.frame_to_seconds(stop)


def refine_event_from_activity(
    event: EventProposal,
    activity_curve: torch.Tensor | np.ndarray | None,
    *,
    grid: FrameGrid,
    activity_threshold: float,
    window_padding_seconds: float,
    left_padding_seconds: float,
    right_padding_seconds: float,
) -> EventProposal:
    if activity_curve is None:
        start = max(0.0, event.onset_seconds - left_padding_seconds)
        end = min(10.0, event.offset_seconds + right_padding_seconds)
        return EventProposal(event.label, start, end, event.confidence)

    curve = (
        activity_curve.detach().float().cpu().numpy()
        if isinstance(activity_curve, torch.Tensor)
        else np.asarray(activity_curve, dtype=np.float32)
    )
    num_frames = len(curve)
    raw_start = int(max(0, min(num_frames, round(event.onset_seconds / grid.frame_seconds))))
    raw_stop = int(max(raw_start + 1, min(num_frames, round(event.offset_seconds / grid.frame_seconds))))
    win_start = int(
        max(0, min(num_frames, round((event.onset_seconds - window_padding_seconds) / grid.frame_seconds)))
    )
    win_stop = int(
        max(win_start + 1, min(num_frames, round((event.offset_seconds + window_padding_seconds) / grid.frame_seconds)))
    )
    local_runs = active_runs(curve[win_start:win_stop] >= activity_threshold)
    if not local_runs:
        start = max(0.0, event.onset_seconds - left_padding_seconds)
        end = min(10.0, event.offset_seconds + right_padding_seconds)
        return EventProposal(event.label, start, end, event.confidence)

    runs = [(a + win_start, b + win_start) for a, b in local_runs]

    def run_key(run: tuple[int, int]) -> tuple[float, float, float]:
        start, stop = run
        overlap = max(0, min(stop, raw_stop) - max(start, raw_start))
        center = 0.5 * (start + stop)
        raw_center = 0.5 * (raw_start + raw_stop)
        energy = float(curve[start:stop].mean()) if stop > start else 0.0
        return (float(overlap), -abs(center - raw_center), energy)

    best_start, best_stop = max(runs, key=run_key)
    start_s, stop_s = frame_span(grid, best_start, best_stop)
    start_s = max(0.0, start_s - left_padding_seconds)
    stop_s = min(10.0, stop_s + right_padding_seconds)
    if stop_s <= start_s:
        start_s = max(0.0, event.onset_seconds - left_padding_seconds)
        stop_s = min(10.0, event.offset_seconds + right_padding_seconds)
    return EventProposal(event.label, start_s, stop_s, event.confidence)


def refine_events_activity(
    record_id: str,
    scene_id: str,
    events: Sequence[EventProposal],
    activity_by_scene: Mapping[str, Mapping[str, Any]],
    *,
    grid: FrameGrid,
    policy: ActivityRelationPolicy,
) -> list[EventProposal]:
    scene = activity_by_scene.get(scene_id, {})
    refined: list[EventProposal] = []
    for event in events:
        row = scene.get(event.label)
        curve = row[0] if row is not None else None
        refined.append(
            refine_event_from_activity(
                event,
                curve,
                grid=grid,
                activity_threshold=policy.activity_threshold,
                window_padding_seconds=policy.window_padding_seconds,
                left_padding_seconds=policy.left_padding_seconds,
                right_padding_seconds=policy.right_padding_seconds,
            )
        )
    return refined


def calibrate_relation(
    *,
    relation: str,
    split: PreparedSplit,
    activity_by_scene: Mapping[str, Mapping[str, Any]],
    base_threshold: float,
    objective: str,
    min_noev: float,
) -> ActivityRelationPolicy:
    rel_records = [record for record in split.records if record.relation == relation]
    thresholds = score_grid(rel_records, split.top, base_threshold=base_threshold)
    answerable = np.asarray([not bool(record.no_evidence) for record in rel_records], dtype=bool)
    no_evidence = ~answerable
    answer_total = max(1, int(answerable.sum()))
    noev_total = max(1, int(no_evidence.sum()))
    scores = np.full(len(rel_records), -np.inf, dtype=np.float32)
    answer_ok = np.zeros(len(rel_records), dtype=bool)
    raw_events: list[tuple[EventProposal, ...]] = []
    for idx, record in enumerate(rel_records):
        selection = split.top.get(record.sample_id)
        if selection is None:
            raw_events.append(())
            continue
        score, candidate, _second, _count = selection
        scores[idx] = float(score)
        answer_ok[idx] = (not record.no_evidence) and candidate.answer_label == record.answer
        raw_events.append(tuple(candidate.events))

    active_by_threshold = scores[None, :] >= thresholds[:, None]
    answer_acc_by_threshold = (
        (active_by_threshold & answer_ok[None, :] & answerable[None, :]).sum(axis=1)
        / answer_total
    )
    noev_acc_by_threshold = (
        ((~active_by_threshold) & no_evidence[None, :]).sum(axis=1) / noev_total
    )

    grid = FrameGrid(sample_rate=32_000)
    activity_thresholds = np.asarray(
        [0.05, 0.10, 0.15, 0.25, 0.40],
        dtype=np.float32,
    )
    window_pads = np.asarray([0.16, 0.64, 1.00], dtype=np.float32)
    final_pads = np.asarray([0.0, 0.16, 0.32], dtype=np.float32)

    best_any: tuple[tuple[float, ...], ActivityRelationPolicy] | None = None
    best_feasible: tuple[tuple[float, ...], ActivityRelationPolicy] | None = None
    for activity_threshold in activity_thresholds:
        for window_padding in window_pads:
            for left_padding in final_pads:
                for right_padding in final_pads:
                    proto = ActivityRelationPolicy(
                        relation=relation,
                        score_threshold=float(base_threshold),
                        activity_threshold=float(activity_threshold),
                        window_padding_seconds=float(window_padding),
                        left_padding_seconds=float(left_padding),
                        right_padding_seconds=float(right_padding),
                        calibration_mean_iou=0.0,
                        calibration_answer_accuracy=0.0,
                        calibration_no_evidence_accuracy=0.0,
                        calibration_mean_span_seconds=0.0,
                    )
                    ious = np.zeros(len(rel_records), dtype=np.float32)
                    span_seconds = np.zeros(len(rel_records), dtype=np.float32)
                    for idx, record in enumerate(rel_records):
                        if not answerable[idx] or not raw_events[idx]:
                            continue
                        refined = refine_events_activity(
                            record.sample_id,
                            record.scene_id,
                            raw_events[idx],
                            activity_by_scene,
                            grid=grid,
                            policy=proto,
                        )
                        spans = [event_to_span(event) for event in refined]
                        ious[idx] = temporal_iou(spans, gold_spans(record))
                        span_seconds[idx] = interval_length(spans)
                    iou_by_threshold = (
                        (active_by_threshold * ious[None, :] * answerable[None, :]).sum(axis=1)
                        / answer_total
                    )
                    span_sum = (
                        active_by_threshold * span_seconds[None, :] * answerable[None, :]
                    ).sum(axis=1)
                    span_count = (
                        active_by_threshold & answerable[None, :] & (span_seconds[None, :] > 0)
                    ).sum(axis=1)
                    mean_span = np.divide(
                        span_sum,
                        np.maximum(span_count, 1),
                        out=np.zeros_like(span_sum, dtype=np.float64),
                        where=span_count > 0,
                    )
                    for th_idx, threshold in enumerate(thresholds):
                        mean_iou = float(iou_by_threshold[th_idx])
                        answer_acc = float(answer_acc_by_threshold[th_idx])
                        noev_acc = float(noev_acc_by_threshold[th_idx])
                        mean_span_seconds = float(mean_span[th_idx])
                        key = (mean_iou, answer_acc, noev_acc, -mean_span_seconds)
                        policy = ActivityRelationPolicy(
                            relation=relation,
                            score_threshold=float(threshold),
                            activity_threshold=float(activity_threshold),
                            window_padding_seconds=float(window_padding),
                            left_padding_seconds=float(left_padding),
                            right_padding_seconds=float(right_padding),
                            calibration_mean_iou=mean_iou,
                            calibration_answer_accuracy=answer_acc,
                            calibration_no_evidence_accuracy=noev_acc,
                            calibration_mean_span_seconds=mean_span_seconds,
                        )
                        if best_any is None or key > best_any[0]:
                            best_any = (key, policy)
                        if objective == "activity_iou_max" or noev_acc >= min_noev:
                            if best_feasible is None or key > best_feasible[0]:
                                best_feasible = (key, policy)
    chosen = best_feasible or best_any
    assert chosen is not None
    return chosen[1]


def calibrate_policy(
    *,
    name: str,
    objective: str,
    split: PreparedSplit,
    activity_by_scene: Mapping[str, Mapping[str, Any]],
    base_threshold: float,
    min_noev: float,
) -> ActivityPolicy:
    return ActivityPolicy(
        name=name,
        objective=objective,
        relations={
            relation: calibrate_relation(
                relation=relation,
                split=split,
                activity_by_scene=activity_by_scene,
                base_threshold=base_threshold,
                objective=objective,
                min_noev=min_noev,
            )
            for relation in RELATIONS
        },
    )


def predict_record(
    record: Any,
    split: PreparedSplit,
    activity_by_scene: Mapping[str, Mapping[str, Any]],
    policy: ActivityPolicy,
) -> tuple[bool, str | None, float | None, list[EventProposal], list[EventProposal]]:
    relation_policy = policy.relations.get(record.relation)
    selection = split.top.get(record.sample_id)
    if relation_policy is None or selection is None:
        return True, None, None, [], []
    score, candidate, _second, _count = selection
    if float(score) < relation_policy.score_threshold:
        return True, None, float(score), [], []
    raw = list(candidate.events)
    refined = refine_events_activity(
        record.sample_id,
        record.scene_id,
        raw,
        activity_by_scene,
        grid=FrameGrid(sample_rate=32_000),
        policy=relation_policy,
    )
    return False, candidate.answer_label, float(score), raw, refined


def evaluate_split(
    split: PreparedSplit,
    activity_by_scene: Mapping[str, Mapping[str, Any]],
    policy: ActivityPolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    items: list[dict[str, Any]] = []
    answer_total = answer_correct = noev_total = noev_correct = decision_correct = 0
    raw_iou_sum = refined_iou_sum = iou_count = 0
    by_relation = {relation: {"sum": 0.0, "count": 0.0} for relation in RELATIONS}
    for record in split.records:
        pred_noev, pred_answer, score, raw_events, refined_events = predict_record(
            record, split, activity_by_scene, policy
        )
        decision_correct += int(pred_noev == bool(record.no_evidence))
        if record.no_evidence:
            noev_total += 1
            noev_correct += int(pred_noev)
            raw_iou = refined_iou = answer_ok = None
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
    def f(value: Any) -> str:
        return "..." if value is None else f"{float(value):.3f}"

    lines = [
        "# QCES-v6 activity span refiner benchmark",
        "",
        "Policy is calibrated on train-calibration scene families only.",
        "",
        "| Split | Policy | answer ↑ | no-evid ↑ | decision ↑ | IoU raw ↑ | IoU refined ↑ | gain ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, policies in report["eval_results"].items():
        for policy, summary in policies.items():
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
    import joblib

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
    train_activity = load_activity(args.train_cache, args.proposal_head, device)
    calib_mask = family_calibration_mask(
        train.candidates, train.records_by_id, frac=args.calib_family_frac
    )
    calib_ids = {
        candidate.sample_id
        for candidate, keep in zip(train.candidates, calib_mask)
        if keep
    }
    calib_candidates = [
        candidate for candidate, keep in zip(train.candidates, calib_mask) if keep
    ]
    calib_scores = train.scores[calib_mask]
    from mixi_understanding.scripts.calibrate_qces_v6_pair_reranker import top_candidate_by_record

    calib = PreparedSplit(
        name="train_calib",
        records=[record for record in train.records if record.sample_id in calib_ids],
        candidates=calib_candidates,
        scores=calib_scores,
        top=top_candidate_by_record(calib_candidates, calib_scores),
        records_by_id={rid: train.records_by_id[rid] for rid in calib_ids},
    )
    policies = [
        calibrate_policy(
            name="activity_iou_max",
            objective="activity_iou_max",
            split=calib,
            activity_by_scene=train_activity,
            base_threshold=base_threshold,
            min_noev=args.min_noev_for_balanced,
        ),
        calibrate_policy(
            name=f"activity_iou_noev{args.min_noev_for_balanced:.2f}",
            objective="activity_iou_noev_guarded",
            split=calib,
            activity_by_scene=train_activity,
            base_threshold=base_threshold,
            min_noev=args.min_noev_for_balanced,
        ),
    ]

    eval_results: dict[str, Any] = {}
    for name, manifest, cache in args.eval:
        split = prepare_named_split(
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
        activity = load_activity(Path(cache), args.proposal_head, device)
        eval_results[name] = {}
        for policy in policies:
            summary, items = evaluate_split(split, activity, policy)
            eval_results[name][policy.name] = summary
            write_jsonl(out / f"{name}__{policy.name}.jsonl", items)

    report = {
        "format": FORMAT_VERSION,
        "model": str(args.model.resolve()),
        "dataset_config": str(args.dataset_config.resolve()),
        "proposal_head": str(args.proposal_head.resolve()),
        "base_pair_threshold": base_threshold,
        "pair_relation_thresholds": relation_thresholds,
        "max_neighbors": max_neighbors,
        "calib_family_frac": args.calib_family_frac,
        "min_noev_for_balanced": args.min_noev_for_balanced,
        "policies": {
            policy.name: {
                "objective": policy.objective,
                "relations": {
                    relation: asdict(relation_policy)
                    for relation, relation_policy in policy.relations.items()
                },
            }
            for policy in policies
        },
        "eval_results": eval_results,
    }
    (out / "activity_span_refiner_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rendered = markdown(report)
    (out / "activity_span_refiner_results.md").write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main(sys.argv[1:])
