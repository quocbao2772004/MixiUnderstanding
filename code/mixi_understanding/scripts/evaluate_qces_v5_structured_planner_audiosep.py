#!/usr/bin/env python3
"""Evaluate a structured event-planner AudioSep baseline for QCES-v5.

This diagnostic removes the free soft prompt and the target-metric prompt
selection loop.  It uses only the question semantics and the scene event
inventory to plan which events are relevant, renders one canonical text prompt,
and gates the frozen AudioSep output with the planned event spans.

It is still not a fully deployable system because the event inventory is
annotation-provided.  The point is to isolate whether a discrete planner +
canonical text prompt + temporal grounding can close the gap to the oracle
upper bound.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Event, QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    _v5_item_metrics,
    describe,
    encode_prompts,
    summarize_v5_items,
    write_item_artifacts,
)


FORMAT_VERSION = "qces_v5_structured_planner_audiosep_v1"
MODE_PLANNER_TEXT_GATE = "structured_planner_text__planned_event_gate"
MODE_PLANNER_TEXT_NO_GATE = "structured_planner_text__no_gate"


@dataclass(frozen=True)
class Plan:
    event_ids: tuple[str, ...]
    labels: tuple[str, ...]
    prompt: str
    no_evidence: bool
    reason: str
    relation: str
    planned_answer_label: str | None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--no-render-audio", action="store_true")
    parser.add_argument("--render-item-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--gate-dilation-seconds",
        type=float,
        default=0.0,
        help="Optional symmetric dilation applied to planned event spans.",
    )
    args = parser.parse_args(argv)
    if args.text_batch_size <= 0:
        parser.error("--text-batch-size must be positive")
    if args.max_records < 0:
        parser.error("--max-records must be non-negative")
    if args.gate_dilation_seconds < 0:
        parser.error("--gate-dilation-seconds must be non-negative")
    if len(args.render_item_id) != len(set(args.render_item_id)):
        parser.error("--render-item-id values must be unique")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def prompt_for_labels(labels: Sequence[str]) -> str:
    phrases = [describe(label).replace("_", " ") for label in labels]
    if not phrases:
        return ""
    if len(phrases) == 1:
        joined = phrases[0]
    else:
        joined = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return f"the sounds of {joined}"


def chronological_events(record: QCESV5Record) -> list[QCESV5Event]:
    return sorted(
        record.events,
        key=lambda event: (event.onset_seconds, event.offset_seconds, event.event_id),
    )


def event_lookup(record: QCESV5Record) -> dict[str, QCESV5Event]:
    return {event.event_id: event for event in record.events}


def find_event_by_label_ordinal(
    record: QCESV5Record, label: str | None, ordinal: int | None
) -> QCESV5Event | None:
    if not label or ordinal is None:
        return None
    candidates = [
        event
        for event in chronological_events(record)
        if event.label == label
    ]
    if ordinal < 1 or ordinal > len(candidates):
        return None
    return candidates[ordinal - 1]


def first_occurrence_by_label(record: QCESV5Record, label: str) -> QCESV5Event | None:
    candidates = [
        event for event in chronological_events(record) if event.label == label
    ]
    return candidates[0] if candidates else None


def next_event_after(record: QCESV5Record, anchor: QCESV5Event) -> QCESV5Event | None:
    candidates = [
        event
        for event in chronological_events(record)
        if event.event_id != anchor.event_id and event.onset_seconds > anchor.onset_seconds
    ]
    return candidates[0] if candidates else None


def previous_event_before(
    record: QCESV5Record, anchor: QCESV5Event
) -> QCESV5Event | None:
    candidates = [
        event
        for event in chronological_events(record)
        if event.event_id != anchor.event_id and event.onset_seconds < anchor.onset_seconds
    ]
    return candidates[-1] if candidates else None


def plan_record(record: QCESV5Record) -> Plan:
    """Plan evidence events without using answer/evidence annotations."""

    relation = record.relation
    if relation in {"after", "before"}:
        anchor = find_event_by_label_ordinal(
            record, record.query_label, record.query_instance_ordinal
        )
        if anchor is None:
            return Plan(
                event_ids=(),
                labels=(),
                prompt="",
                no_evidence=True,
                reason="missing_query_anchor_in_event_inventory",
                relation=relation,
                planned_answer_label=None,
            )
        answer = next_event_after(record, anchor) if relation == "after" else previous_event_before(record, anchor)
        if answer is None:
            return Plan(
                event_ids=(),
                labels=(),
                prompt="",
                no_evidence=True,
                reason=f"no_{relation}_neighbor_in_event_inventory",
                relation=relation,
                planned_answer_label=None,
            )
        ordered = sorted(
            (anchor, answer),
            key=lambda event: (event.onset_seconds, event.event_id),
        )
        labels = unique_in_order(event.label for event in ordered)
        return Plan(
            event_ids=tuple(event.event_id for event in ordered),
            labels=labels,
            prompt=prompt_for_labels(labels),
            no_evidence=False,
            reason="planned_from_query_anchor_and_temporal_neighbor",
            relation=relation,
            planned_answer_label=answer.label,
        )

    if relation == "first":
        candidate_labels = tuple(record.query_candidate_labels)
        if len(candidate_labels) < 2:
            return Plan(
                event_ids=(),
                labels=(),
                prompt="",
                no_evidence=True,
                reason="missing_first_candidate_labels",
                relation=relation,
                planned_answer_label=None,
            )
        events: list[QCESV5Event] = []
        missing_labels: list[str] = []
        for label in candidate_labels:
            event = first_occurrence_by_label(record, label)
            if event is None:
                missing_labels.append(label)
            else:
                events.append(event)
        if missing_labels or len(events) < 2:
            return Plan(
                event_ids=(),
                labels=(),
                prompt="",
                no_evidence=True,
                reason="missing_first_candidate_in_event_inventory:"
                + ",".join(missing_labels),
                relation=relation,
                planned_answer_label=None,
            )
        ordered = sorted(events, key=lambda event: (event.onset_seconds, event.event_id))
        labels = unique_in_order(event.label for event in ordered)
        return Plan(
            event_ids=unique_in_order(event.event_id for event in ordered),
            labels=labels,
            prompt=prompt_for_labels(labels),
            no_evidence=False,
            reason="planned_from_first_candidate_labels",
            relation=relation,
            planned_answer_label=ordered[0].label,
        )

    return Plan(
        event_ids=(),
        labels=(),
        prompt="",
        no_evidence=True,
        reason=f"unsupported_relation:{relation}",
        relation=relation,
        planned_answer_label=None,
    )


def planned_gate(
    record: QCESV5Record,
    plan: Plan,
    num_samples: int,
    device: torch.device,
    dilation_seconds: float,
) -> torch.Tensor:
    gate = torch.zeros(num_samples, device=device)
    events = event_lookup(record)
    dilation_samples = int(round(dilation_seconds * record.sample_rate))
    for event_id in plan.event_ids:
        event = events[event_id]
        start = int(round(event.onset_seconds * record.sample_rate)) - dilation_samples
        stop = int(round(event.offset_seconds * record.sample_rate)) + dilation_samples
        start = max(0, min(start, num_samples))
        stop = max(start, min(stop, num_samples))
        gate[start:stop] = 1.0
    return gate


def metadata(record: QCESV5Record, plan: Plan) -> Dict[str, Any]:
    gold_event_ids = tuple(record.evidence_event_ids)
    planned_event_set = set(plan.event_ids)
    gold_event_set = set(gold_event_ids)
    intersection = planned_event_set & gold_event_set
    union = planned_event_set | gold_event_set
    event_jaccard = float(len(intersection) / len(union)) if union else 1.0
    return {
        "id": record.sample_id,
        "split": record.split,
        "evaluation_axis": record.evaluation_axis,
        "scene_id": record.scene_id,
        "scene_family_id": record.scene_family_id,
        "variant_id": record.variant_id,
        "counterfactual_group_id": record.counterfactual_group_id,
        "question_semantics_id": record.question_semantics_id,
        "paraphrase_family_id": record.paraphrase_family_id,
        "question_index": record.question_index,
        "question_type": record.question_type,
        "relation": record.relation,
        "question": record.question,
        "answer": record.answer,
        "no_evidence": record.no_evidence,
        "gold_answer_label": None if record.no_evidence else record.answer,
        "planned_answer_label": plan.planned_answer_label,
        "target_labels": tuple(
            dict.fromkeys(record.event_by_id(event_id).label for event_id in gold_event_ids)
        ),
        "planned_labels": plan.labels,
        "gold_evidence_event_ids": gold_event_ids,
        "planned_evidence_event_ids": plan.event_ids,
        "planned_no_evidence": plan.no_evidence,
        "planner_reason": plan.reason,
        "planner_event_exact_match": tuple(plan.event_ids) == gold_event_ids
        or planned_event_set == gold_event_set,
        "planner_event_jaccard_↑": event_jaccard,
        "planner_answer_correct": (
            bool(record.no_evidence and plan.no_evidence)
            if record.no_evidence
            else plan.planned_answer_label == record.answer
        ),
    }


def summarize_by_mode(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for mode in sorted({str(item["mode"]) for item in items}):
        subset = [item for item in items if str(item["mode"]) == mode]
        summary = summarize_v5_items(subset)
        summary.update(planner_summary(subset))
        result[mode] = summary
    return result


def mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def planner_summary(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    unique_by_id: dict[str, Mapping[str, Any]] = {}
    for item in items:
        unique_by_id[str(item["id"])] = item
    rows = list(unique_by_id.values())
    positives = [row for row in rows if not bool(row["no_evidence"])]
    negatives = [row for row in rows if bool(row["no_evidence"])]
    return {
        "planner_answer_accuracy_↑": mean(
            [float(bool(row["planner_answer_correct"])) for row in rows]
        ),
        "planner_event_exact_match_rate_↑": mean(
            [
                float(bool(row["planner_event_exact_match"]))
                for row in positives
            ]
        ),
        "planner_event_jaccard_mean_↑": mean(
            [float(row["planner_event_jaccard_↑"]) for row in positives]
        ),
        "planner_no_evidence_accuracy_↑": mean(
            [
                float(bool(row["planned_no_evidence"]) == bool(row["no_evidence"]))
                for row in rows
            ]
        ),
        "planner_false_no_evidence_rate_on_positive_↓": mean(
            [float(bool(row["planned_no_evidence"])) for row in positives]
        ),
        "planner_false_evidence_rate_on_negative_↓": mean(
            [float(not bool(row["planned_no_evidence"])) for row in negatives]
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = QCESManifestDataset(args.manifest.resolve(), crop_samples=None)
    records = [
        record for record in dataset.records if isinstance(record, QCESV5Record)
    ]
    if len(records) != len(dataset.records):
        raise SystemExit("this baseline requires a pure QCES-v5 manifest")
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise SystemExit("selected record set is empty")
    selected_ids = {record.sample_id for record in records}
    unknown_render = sorted(set(args.render_item_id) - selected_ids)
    if unknown_render:
        raise SystemExit("render IDs are absent from selection: " + ", ".join(unknown_render))

    plans = {record.sample_id: plan_record(record) for record in records}
    prompts = sorted(
        {plan.prompt for plan in plans.values() if not plan.no_evidence and plan.prompt}
    )
    prompt_embeddings = (
        encode_prompts(
            args.audiosep_root.resolve(),
            args.audiosep_checkpoint.resolve(),
            prompts,
            batch_size=args.text_batch_size,
        )
        if prompts
        else {}
    )
    separator = _load_separator(args, device) if prompts else None

    render_ids = set(args.render_item_id)
    items: list[dict[str, Any]] = []
    separator_calls = 0
    with torch.inference_mode():
        for index, record in enumerate(records):
            example = dataset[index]
            mixture = example.mixture.to(device)
            target = example.evidence.to(device)
            target_residual = example.residual.to(device)
            plan = plans[record.sample_id]
            if plan.no_evidence:
                raw = torch.zeros_like(mixture)
            else:
                assert separator is not None
                condition = prompt_embeddings[plan.prompt][None].to(device)
                raw = separator(
                    {"mixture": mixture[None, None], "condition": condition}
                )["waveform"][0, 0]
                separator_calls += 1
            gate = planned_gate(
                record,
                plan,
                mixture.numel(),
                device,
                args.gate_dilation_seconds,
            )
            mode_to_evidence = {
                MODE_PLANNER_TEXT_GATE: raw * gate,
                MODE_PLANNER_TEXT_NO_GATE: raw,
            }
            for mode, evidence in mode_to_evidence.items():
                if plan.no_evidence:
                    evidence = torch.zeros_like(mixture)
                residual = mixture - evidence
                metrics, descriptives = _v5_item_metrics(
                    no_evidence=record.no_evidence,
                    evidence=evidence,
                    mixture=mixture,
                    target=target,
                    target_residual=target_residual,
                )
                item = {
                    **metadata(record, plan),
                    "mode": mode,
                    "baseline_access": "structured_question_fields_plus_oracle_event_inventory",
                    "prompt": plan.prompt,
                    "metrics": metrics,
                    "descriptives": descriptives,
                }
                should_render = not args.no_render_audio and (
                    not render_ids or record.sample_id in render_ids
                )
                write_item_artifacts(
                    question_dir=(
                        output_dir
                        / mode
                        / record.scene_id
                        / f"q{record.question_index}_{record.question_type}"
                    ),
                    item=item,
                    evidence=evidence,
                    residual=residual,
                    sample_rate=record.sample_rate,
                    render_audio=should_render,
                )
                items.append(item)

    report = {
        "format": FORMAT_VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "selected_record_count": len(records),
        "audiosep_frozen": True,
        "audiosep_checkpoint": str(args.audiosep_checkpoint.resolve()),
        "audiosep_checkpoint_sha256": sha256_file(args.audiosep_checkpoint.resolve()),
        "device": str(device),
        "separator_calls": separator_calls,
        "gate_dilation_seconds": args.gate_dilation_seconds,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "mode_registry": {
            MODE_PLANNER_TEXT_GATE: {
                "uses_answer_or_target_to_plan": False,
                "uses_oracle_event_inventory": True,
                "uses_planned_time": True,
                "description": (
                    "Structured planner picks events from question fields and scene inventory, "
                    "renders one canonical text prompt, then gates AudioSep by planned event spans."
                ),
            },
            MODE_PLANNER_TEXT_NO_GATE: {
                "uses_answer_or_target_to_plan": False,
                "uses_oracle_event_inventory": True,
                "uses_planned_time": False,
                "description": "Same planner text prompt without planned temporal gate.",
            },
        },
        "protocol_limitations": [
            "Uses annotation-provided event inventory and structured question fields; this is not yet raw audio + natural question inference.",
            "Planner never uses gold answer, gold evidence_event_ids, or target waveform to choose prompt/events.",
            "Gold labels and stems are used only for evaluation metrics.",
        ],
        "summaries_by_mode": summarize_by_mode(items),
        "items": items,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summaries_by_mode"], indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
