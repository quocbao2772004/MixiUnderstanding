#!/usr/bin/env python3
"""Evaluate QA + temporal evidence from a predicted detector event inventory.

This is a detector-first sidecar benchmark:

    scene manifest with gold events
        + detector val_predictions.jsonl
        -> generated temporal QA rows
        -> symbolic executor over predicted inventory
        -> answer accuracy + no-evidence accuracy + evidence IoU

The script deliberately evaluates the detector output as an event inventory.
It does not run AF3/Qwen/Phi and it does not use oracle annotations at
inference time.  Gold events are used only to generate/scoring the QA rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/detector_manifest_val.jsonl"
)
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200/partial_as_fuss_e4_unfreeze1_highthr_resume/val_predictions.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs/qces_detector_inventory_qa/multisource200_unfreeze1_val_v1"
)


@dataclass(frozen=True)
class QAItem:
    item_id: str
    scene_id: str
    relation: str
    question: str
    answer: str
    no_evidence: bool
    anchor_label: str
    anchor_ordinal: int
    answer_label: str | None
    gold_evidence_intervals: tuple[tuple[float, float], ...]
    gold_anchor_interval: tuple[float, float] | None
    gold_answer_interval: tuple[float, float] | None
    source_route: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--max-answerable-per-scene", type=int, default=4)
    parser.add_argument("--max-no-evidence-per-scene", type=int, default=2)
    parser.add_argument("--min-gap-seconds", type=float, default=0.0)
    parser.add_argument("--evidence-iou-threshold", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def label_text(label: str) -> str:
    return label.replace("_and_", " / ").replace("_", " ")


def ordinal_word(index: int) -> str:
    words = {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
    }
    return words.get(index, str(index))


def event_sort_key(event: Mapping[str, Any]) -> tuple[float, float, str]:
    return (
        float(event.get("onset_seconds", 0.0)),
        float(event.get("offset_seconds", 0.0)),
        str(event.get("label", "")),
    )


def semantic_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in row.get("events") or row.get("gold_events") or []:
        if event.get("event_kind", "semantic") != "semantic":
            continue
        onset = float(event.get("onset_seconds", 0.0))
        offset = float(event.get("offset_seconds", onset))
        label = str(event.get("label", ""))
        if not label or offset <= onset:
            continue
        events.append({**event, "label": label, "onset_seconds": onset, "offset_seconds": offset})
    return sorted(events, key=event_sort_key)


def occurrence_ordinal(events: Sequence[Mapping[str, Any]], index: int) -> int:
    label = str(events[index]["label"])
    return 1 + sum(1 for event in events[:index] if str(event.get("label")) == label)


def find_occurrence(events: Sequence[Mapping[str, Any]], label: str, ordinal: int) -> Mapping[str, Any] | None:
    matches = [event for event in events if str(event.get("label")) == label]
    matches.sort(key=event_sort_key)
    if ordinal <= 0 or len(matches) < ordinal:
        return None
    return matches[ordinal - 1]


def interval(event: Mapping[str, Any]) -> tuple[float, float]:
    return (float(event["onset_seconds"]), float(event["offset_seconds"]))


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    inter = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return inter / union if union > 0 else 0.0


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    valid = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not valid:
        return []
    merged = [valid[0]]
    for onset, offset in valid[1:]:
        prev_onset, prev_offset = merged[-1]
        if onset <= prev_offset:
            merged[-1] = (prev_onset, max(prev_offset, offset))
        else:
            merged.append((onset, offset))
    return merged


def intervals_duration(intervals: Sequence[tuple[float, float]]) -> float:
    return sum(max(0.0, b - a) for a, b in merge_intervals(intervals))


def intervals_intersection(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> float:
    total = 0.0
    left_m = merge_intervals(left)
    right_m = merge_intervals(right)
    i = j = 0
    while i < len(left_m) and j < len(right_m):
        a0, a1 = left_m[i]
        b0, b1 = right_m[j]
        total += max(0.0, min(a1, b1) - max(a0, b0))
        if a1 < b1:
            i += 1
        else:
            j += 1
    return total


def intervals_iou(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> float:
    inter = intervals_intersection(left, right)
    union = intervals_duration(left) + intervals_duration(right) - inter
    return inter / union if union > 0 else 1.0 if not left and not right else 0.0


def event_sentence(event: Mapping[str, Any]) -> str:
    return f"{label_text(str(event['label']))} ({float(event['onset_seconds']):.2f}–{float(event['offset_seconds']):.2f}s)"


def build_qa_items(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    max_scenes: int,
    max_answerable_per_scene: int,
    max_no_evidence_per_scene: int,
    min_gap_seconds: float,
) -> list[QAItem]:
    items: list[QAItem] = []
    scene_count = 0
    for row in manifest_rows:
        scene_id = str(row.get("scene_id") or "")
        route = str(row.get("source_route") or row.get("split") or "")
        events = semantic_events(row)
        if len(events) < 2:
            continue
        scene_count += 1
        if max_scenes and scene_count > max_scenes:
            break

        local: list[QAItem] = []
        for i in range(len(events) - 1):
            anchor = events[i]
            answer = events[i + 1]
            if float(answer["onset_seconds"]) - float(anchor["onset_seconds"]) < min_gap_seconds:
                continue
            ordinal = occurrence_ordinal(events, i)
            anchor_label = str(anchor["label"])
            answer_label = str(answer["label"])
            q = (
                f"What sound occurs immediately after the "
                f"{ordinal_word(ordinal)} {label_text(anchor_label)}?"
            )
            local.append(
                QAItem(
                    item_id=f"{scene_id}:after:{i}",
                    scene_id=scene_id,
                    relation="after",
                    question=q,
                    answer=answer_label,
                    no_evidence=False,
                    anchor_label=anchor_label,
                    anchor_ordinal=ordinal,
                    answer_label=answer_label,
                    gold_evidence_intervals=(interval(anchor), interval(answer)),
                    gold_anchor_interval=interval(anchor),
                    gold_answer_interval=interval(answer),
                    source_route=route,
                )
            )
        for i in range(1, len(events)):
            answer = events[i - 1]
            anchor = events[i]
            if float(anchor["onset_seconds"]) - float(answer["onset_seconds"]) < min_gap_seconds:
                continue
            ordinal = occurrence_ordinal(events, i)
            anchor_label = str(anchor["label"])
            answer_label = str(answer["label"])
            q = (
                f"What sound occurs immediately before the "
                f"{ordinal_word(ordinal)} {label_text(anchor_label)}?"
            )
            local.append(
                QAItem(
                    item_id=f"{scene_id}:before:{i}",
                    scene_id=scene_id,
                    relation="before",
                    question=q,
                    answer=answer_label,
                    no_evidence=False,
                    anchor_label=anchor_label,
                    anchor_ordinal=ordinal,
                    answer_label=answer_label,
                    gold_evidence_intervals=(interval(anchor), interval(answer)),
                    gold_anchor_interval=interval(anchor),
                    gold_answer_interval=interval(answer),
                    source_route=route,
                )
            )

        answerable = local[: max(0, max_answerable_per_scene)]
        noev: list[QAItem] = []
        first = events[0]
        first_ordinal = occurrence_ordinal(events, 0)
        first_label = str(first["label"])
        noev.append(
            QAItem(
                item_id=f"{scene_id}:before_first",
                scene_id=scene_id,
                relation="before",
                question=(
                    f"What sound occurs immediately before the "
                    f"{ordinal_word(first_ordinal)} {label_text(first_label)}?"
                ),
                answer="no_evidence",
                no_evidence=True,
                anchor_label=first_label,
                anchor_ordinal=first_ordinal,
                answer_label=None,
                gold_evidence_intervals=(),
                gold_anchor_interval=interval(first),
                gold_answer_interval=None,
                source_route=route,
            )
        )
        last = events[-1]
        last_ordinal = occurrence_ordinal(events, len(events) - 1)
        last_label = str(last["label"])
        noev.append(
            QAItem(
                item_id=f"{scene_id}:after_last",
                scene_id=scene_id,
                relation="after",
                question=(
                    f"What sound occurs immediately after the "
                    f"{ordinal_word(last_ordinal)} {label_text(last_label)}?"
                ),
                answer="no_evidence",
                no_evidence=True,
                anchor_label=last_label,
                anchor_ordinal=last_ordinal,
                answer_label=None,
                gold_evidence_intervals=(),
                gold_anchor_interval=interval(last),
                gold_answer_interval=None,
                source_route=route,
            )
        )
        items.extend(answerable)
        items.extend(noev[: max(0, max_no_evidence_per_scene)])
    return items


def execute_item(
    item: QAItem,
    pred_events: Sequence[Mapping[str, Any]],
) -> tuple[str, bool, tuple[Mapping[str, Any], ...], str]:
    events = sorted(pred_events, key=event_sort_key)
    anchor = find_occurrence(events, item.anchor_label, item.anchor_ordinal)
    if anchor is None:
        return "no_evidence", True, (), "anchor_missing"
    anchor_onset = float(anchor["onset_seconds"])
    if item.relation == "after":
        candidates = [event for event in events if float(event["onset_seconds"]) > anchor_onset + 1e-6]
        if not candidates:
            return "no_evidence", True, (), "no_event_after_anchor"
        answer = min(
            candidates,
            key=lambda event: (
                float(event["onset_seconds"]),
                -float(event.get("confidence", 0.0)),
                str(event.get("label", "")),
            ),
        )
    elif item.relation == "before":
        candidates = [event for event in events if float(event["onset_seconds"]) < anchor_onset - 1e-6]
        if not candidates:
            return "no_evidence", True, (), "no_event_before_anchor"
        answer = max(
            candidates,
            key=lambda event: (
                float(event["onset_seconds"]),
                float(event.get("confidence", 0.0)),
                str(event.get("label", "")),
            ),
        )
    else:
        return "unsupported", True, (), "unsupported_relation"
    return str(answer["label"]), False, (anchor, answer), "ok"


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision_↑": precision, "recall_↑": recall, "f1_↑": f1}


def summarize_rows(rows: Sequence[Mapping[str, Any]], evidence_iou_threshold: float) -> dict[str, Any]:
    total = len(rows)
    answerable = [row for row in rows if not bool(row["gold_no_evidence"])]
    noev = [row for row in rows if bool(row["gold_no_evidence"])]
    correct = [row for row in rows if bool(row["answer_correct"])]
    ev_scored = [row for row in answerable if not bool(row["pred_no_evidence"])]
    both = [
        row
        for row in answerable
        if bool(row["answer_correct"]) and float(row["evidence_iou_↑"]) >= evidence_iou_threshold
    ]

    pred_noev = [row for row in rows if bool(row["pred_no_evidence"])]
    tp_noev = sum(bool(row["gold_no_evidence"]) and bool(row["pred_no_evidence"]) for row in rows)
    fp_noev = sum((not bool(row["gold_no_evidence"])) and bool(row["pred_no_evidence"]) for row in rows)
    fn_noev = sum(bool(row["gold_no_evidence"]) and (not bool(row["pred_no_evidence"])) for row in rows)

    by_relation: dict[str, dict[str, Any]] = {}
    for relation in sorted({str(row["relation"]) for row in rows}):
        subset = [row for row in rows if str(row["relation"]) == relation]
        ans_subset = [row for row in subset if not bool(row["gold_no_evidence"])]
        noev_subset = [row for row in subset if bool(row["gold_no_evidence"])]
        by_relation[relation] = {
            "items": len(subset),
            "accuracy_↑": sum(bool(row["answer_correct"]) for row in subset) / max(len(subset), 1),
            "answerable_accuracy_↑": sum(bool(row["answer_correct"]) for row in ans_subset) / max(len(ans_subset), 1),
            "no_evidence_accuracy_↑": sum(bool(row["answer_correct"]) for row in noev_subset) / max(len(noev_subset), 1),
            "mean_evidence_iou_answerable_↑": float(np.mean([float(row["evidence_iou_↑"]) for row in ans_subset]))
            if ans_subset
            else 0.0,
        }

    by_route: dict[str, dict[str, Any]] = {}
    for route in sorted({str(row["source_route"]) for row in rows}):
        subset = [row for row in rows if str(row["source_route"]) == route]
        by_route[route] = {
            "items": len(subset),
            "accuracy_↑": sum(bool(row["answer_correct"]) for row in subset) / max(len(subset), 1),
            "mean_evidence_iou_answerable_↑": float(
                np.mean([float(row["evidence_iou_↑"]) for row in subset if not bool(row["gold_no_evidence"])])
            )
            if any(not bool(row["gold_no_evidence"]) for row in subset)
            else 0.0,
        }

    reasons = Counter(str(row["reason"]) for row in rows)
    wrong_reasons = Counter(str(row["reason"]) for row in rows if not bool(row["answer_correct"]))
    return {
        "items": total,
        "answerable_items": len(answerable),
        "no_evidence_items": len(noev),
        "accuracy_↑": len(correct) / max(total, 1),
        "answerable_accuracy_↑": sum(bool(row["answer_correct"]) for row in answerable) / max(len(answerable), 1),
        "no_evidence_accuracy_↑": sum(bool(row["answer_correct"]) for row in noev) / max(len(noev), 1),
        "mean_evidence_iou_answerable_↑": float(np.mean([float(row["evidence_iou_↑"]) for row in answerable]))
        if answerable
        else 0.0,
        "median_evidence_iou_answerable_↑": float(np.median([float(row["evidence_iou_↑"]) for row in answerable]))
        if answerable
        else 0.0,
        "answer_and_evidence_iou030_accuracy_↑": len(both) / max(len(answerable), 1),
        "predicted_no_evidence_rate": len(pred_noev) / max(total, 1),
        "no_evidence_detection": prf(tp_noev, fp_noev, fn_noev),
        "by_relation": by_relation,
        "by_route": by_route,
        "reasons": dict(reasons.most_common()),
        "wrong_reasons": dict(wrong_reasons.most_common()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_jsonl(args.manifest.resolve())
    prediction_rows = read_jsonl(args.predictions.resolve())
    pred_by_scene = {str(row["scene_id"]): list(row.get("predicted_events") or []) for row in prediction_rows}

    qa_items = build_qa_items(
        manifest_rows,
        max_scenes=args.max_scenes,
        max_answerable_per_scene=args.max_answerable_per_scene,
        max_no_evidence_per_scene=args.max_no_evidence_per_scene,
        min_gap_seconds=args.min_gap_seconds,
    )

    rows: list[dict[str, Any]] = []
    for item in qa_items:
        answer, pred_noev, evidence_events, reason = execute_item(item, pred_by_scene.get(item.scene_id, ()))
        pred_intervals = tuple(interval(event) for event in evidence_events)
        evidence_iou = intervals_iou(item.gold_evidence_intervals, pred_intervals)
        answer_correct = answer == item.answer and pred_noev == item.no_evidence
        rows.append(
            {
                "item_id": item.item_id,
                "scene_id": item.scene_id,
                "source_route": item.source_route,
                "relation": item.relation,
                "question": item.question,
                "gold_answer": item.answer,
                "gold_no_evidence": item.no_evidence,
                "gold_anchor_label": item.anchor_label,
                "gold_anchor_ordinal": item.anchor_ordinal,
                "gold_answer_label": item.answer_label,
                "gold_evidence_intervals": [list(x) for x in item.gold_evidence_intervals],
                "pred_answer": answer,
                "pred_no_evidence": pred_noev,
                "pred_evidence_intervals": [list(x) for x in pred_intervals],
                "pred_evidence_events": [dict(event) for event in evidence_events],
                "answer_correct": answer_correct,
                "evidence_iou_↑": evidence_iou,
                "answer_and_evidence_iou030_correct": (
                    answer_correct
                    and (item.no_evidence or evidence_iou >= args.evidence_iou_threshold)
                ),
                "reason": reason,
            }
        )

    summary = {
        "format": "qces_detector_inventory_qa_v1",
        "manifest": str(args.manifest.resolve()),
        "predictions": str(args.predictions.resolve()),
        "max_scenes": args.max_scenes,
        "max_answerable_per_scene": args.max_answerable_per_scene,
        "max_no_evidence_per_scene": args.max_no_evidence_per_scene,
        "min_gap_seconds": args.min_gap_seconds,
        "evidence_iou_threshold": args.evidence_iou_threshold,
        "metrics": summarize_rows(rows, args.evidence_iou_threshold),
    }
    write_json(output_dir / "summary.json", summary)
    with (output_dir / "qa_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    metrics = summary["metrics"]
    md = [
        "# QCES detector inventory QA",
        "",
        "Generated temporal QA over detector validation scenes. Inference uses only the predicted event inventory.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| items | {metrics['items']} |",
        f"| answer accuracy ↑ | {metrics['accuracy_↑']:.4f} |",
        f"| answerable accuracy ↑ | {metrics['answerable_accuracy_↑']:.4f} |",
        f"| no-evidence accuracy ↑ | {metrics['no_evidence_accuracy_↑']:.4f} |",
        f"| mean evidence IoU answerable ↑ | {metrics['mean_evidence_iou_answerable_↑']:.4f} |",
        f"| median evidence IoU answerable ↑ | {metrics['median_evidence_iou_answerable_↑']:.4f} |",
        (
            f"| answer+evidence IoU≥{args.evidence_iou_threshold:.2f} accuracy ↑ | "
            f"{metrics['answer_and_evidence_iou030_accuracy_↑']:.4f} |"
        ),
        f"| predicted no-evidence rate | {metrics['predicted_no_evidence_rate']:.4f} |",
        "",
        "## By relation",
        "",
        "| relation | items | acc ↑ | answerable ↑ | no-evidence ↑ | evidence IoU ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for relation, row in metrics["by_relation"].items():
        md.append(
            f"| {relation} | {row['items']} | {row['accuracy_↑']:.4f} | "
            f"{row['answerable_accuracy_↑']:.4f} | {row['no_evidence_accuracy_↑']:.4f} | "
            f"{row['mean_evidence_iou_answerable_↑']:.4f} |"
        )
    md.extend(
        [
            "",
            "## By source route",
            "",
            "| route | items | acc ↑ | evidence IoU ↑ |",
            "|---|---:|---:|---:|",
        ]
    )
    for route, row in metrics["by_route"].items():
        md.append(f"| {route} | {row['items']} | {row['accuracy_↑']:.4f} | {row['mean_evidence_iou_answerable_↑']:.4f} |")
    md.extend(
        [
            "",
            "## Wrong reasons",
            "",
            "| reason | count |",
            "|---|---:|",
        ]
    )
    for reason, count in metrics["wrong_reasons"].items():
        md.append(f"| {reason} | {count} |")
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
