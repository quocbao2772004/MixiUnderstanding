#!/usr/bin/env python3
"""Score QCES predicted event inventories with an option-aware executor.

The detector outputs an event inventory.  The previous executor answered
after/before questions by selecting from every detected label.  QCES records
also contain explicit multiple-choice ``answer_options``; AF3/Qwen/Phi runs in
this project are option-scored.  This scorer restricts candidate answer labels
to those options, which is the fair comparison setting for multiple-choice QA.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted-events", type=Path, required=True)
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--baseline-qa-predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def occurrence(events: Sequence[Mapping[str, Any]], label: str, ordinal: int) -> Mapping[str, Any] | None:
    candidates = [event for event in events if str(event.get("label")) == label]
    candidates.sort(key=lambda item: (float(item["onset_seconds"]), float(item["offset_seconds"])))
    if ordinal <= 0 or len(candidates) < ordinal:
        return None
    return candidates[ordinal - 1]


def answer_options(row: Mapping[str, Any]) -> set[str]:
    return {str(option) for option in row.get("answer_options") or [] if str(option) != "no_evidence"}


def option_aware_execute(row: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> tuple[str, bool]:
    relation = str(row.get("relation") or "")
    query_label = str(row.get("query_label") or "")
    ordinal = int(row.get("query_instance_ordinal") or 1)
    options = answer_options(row)

    if relation in {"after", "before"}:
        anchor = occurrence(events, query_label, ordinal)
        if anchor is None:
            return "no_evidence", True
        anchor_onset = float(anchor["onset_seconds"])
        if relation == "after":
            candidates = [
                event
                for event in events
                if str(event.get("label")) in options and float(event["onset_seconds"]) > anchor_onset + 1e-6
            ]
            if not candidates:
                return "no_evidence", True
            answer = min(candidates, key=lambda item: (float(item["onset_seconds"]), -float(item.get("confidence", 0.0))))
        else:
            candidates = [
                event
                for event in events
                if str(event.get("label")) in options and float(event["onset_seconds"]) < anchor_onset - 1e-6
            ]
            if not candidates:
                return "no_evidence", True
            answer = max(candidates, key=lambda item: (float(item["onset_seconds"]), float(item.get("confidence", 0.0))))
        return str(answer["label"]), False

    if relation == "first":
        candidate_labels = [str(label) for label in row.get("query_candidate_labels") or []]
        if not candidate_labels:
            candidate_labels = sorted(options)
        candidates = [event for event in events if str(event.get("label")) in set(candidate_labels)]
        if not candidates:
            return "no_evidence", True
        answer = min(candidates, key=lambda item: (float(item["onset_seconds"]), -float(item.get("confidence", 0.0))))
        return str(answer["label"]), False

    return "unsupported", True


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_relation: dict[str, dict[str, Any]] = {}
    by_noev_reason: dict[str, dict[str, Any]] = {}
    for relation in sorted({str(row.get("relation")) for row in rows}):
        subset = [row for row in rows if str(row.get("relation")) == relation]
        by_relation[relation] = {
            "items": len(subset),
            "accuracy_↑": sum(bool(row["ok"]) for row in subset) / max(len(subset), 1),
        }
    for reason in sorted({str(row.get("no_evidence_reason")) for row in rows}):
        subset = [row for row in rows if str(row.get("no_evidence_reason")) == reason]
        by_noev_reason[reason] = {
            "items": len(subset),
            "accuracy_↑": sum(bool(row["ok"]) for row in subset) / max(len(subset), 1),
        }

    answerable = [row for row in rows if not bool(row.get("no_evidence"))]
    no_evidence = [row for row in rows if bool(row.get("no_evidence"))]
    return {
        "items": len(rows),
        "accuracy_↑": sum(bool(row["ok"]) for row in rows) / max(len(rows), 1),
        "answerable_items": len(answerable),
        "answerable_accuracy_↑": sum(bool(row["ok"]) for row in answerable) / max(len(answerable), 1),
        "no_evidence_items": len(no_evidence),
        "no_evidence_accuracy_↑": sum(bool(row["ok"]) for row in no_evidence) / max(len(no_evidence), 1),
        "by_relation": by_relation,
        "by_no_evidence_reason": by_noev_reason,
    }


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    predicted_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(args.predicted_events.resolve()):
        predicted_by_scene[str(row["scene_id"])] = list(row.get("predicted_events") or [])

    baseline_by_scene_question: dict[tuple[str, str], bool] = {}
    if args.baseline_qa_predictions and args.baseline_qa_predictions.exists():
        for row in read_jsonl(args.baseline_qa_predictions.resolve()):
            baseline_by_scene_question[(str(row.get("scene_id")), str(row.get("question")))] = bool(row.get("ok"))

    rows: list[dict[str, Any]] = []
    transitions = Counter()
    for row in read_jsonl(args.qa_manifest.resolve()):
        scene_id = str(row.get("scene_id") or "")
        if scene_id not in predicted_by_scene:
            continue
        pred_answer, pred_noev = option_aware_execute(row, predicted_by_scene[scene_id])
        ok = pred_answer == str(row.get("answer")) and pred_noev == bool(row.get("no_evidence"))
        baseline_ok = baseline_by_scene_question.get((scene_id, str(row.get("question"))))
        if baseline_ok is not None:
            if baseline_ok and ok:
                transitions["baseline_correct_option_correct"] += 1
            elif baseline_ok and not ok:
                transitions["baseline_correct_option_wrong"] += 1
            elif not baseline_ok and ok:
                transitions["baseline_wrong_option_correct"] += 1
            else:
                transitions["baseline_wrong_option_wrong"] += 1
        rows.append(
            {
                "scene_id": scene_id,
                "question": row.get("question"),
                "relation": row.get("relation"),
                "answer_options": row.get("answer_options"),
                "answer": row.get("answer"),
                "no_evidence": row.get("no_evidence"),
                "no_evidence_reason": row.get("no_evidence_reason"),
                "pred_answer": pred_answer,
                "pred_no_evidence": pred_noev,
                "ok": ok,
                "baseline_ok": baseline_ok,
            }
        )

    summary = {
        "format": "qces_option_aware_executor_score_v1",
        "predicted_events": str(args.predicted_events.resolve()),
        "qa_manifest": str(args.qa_manifest.resolve()),
        "baseline_qa_predictions": str(args.baseline_qa_predictions.resolve()) if args.baseline_qa_predictions else None,
        "summary": summarize(rows),
        "baseline_to_option_transitions": dict(sorted(transitions.items())),
    }
    save_json(output_dir / "option_aware_report.json", summary)
    with (output_dir / "option_aware_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    s = summary["summary"]
    md = [
        "# Option-aware QCES executor score",
        "",
        "Candidate answer labels are restricted to `answer_options` except `no_evidence`.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| QA accuracy ↑ | {s['accuracy_↑']:.3f} |",
        f"| answerable accuracy ↑ | {s['answerable_accuracy_↑']:.3f} |",
        f"| no-evidence accuracy ↑ | {s['no_evidence_accuracy_↑']:.3f} |",
        f"| items | {s['items']} |",
        "",
        "| relation | items | accuracy ↑ |",
        "|---|---:|---:|",
    ]
    for relation, item in s["by_relation"].items():
        md.append(f"| {relation} | {item['items']} | {item['accuracy_↑']:.3f} |")
    if transitions:
        md.extend(["", "| transition | count |", "|---|---:|"])
        for key, value in sorted(transitions.items()):
            md.append(f"| {key} | {value} |")
    (output_dir / "option_aware_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
