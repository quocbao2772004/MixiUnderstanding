#!/usr/bin/env python3
"""Audit the QCES-v6 question parser against the withheld annotation.

The parser is what lets the deployable system read a question instead of the
annotation's structured query fields, so the paper needs a fingerprinted receipt
that it recovers those fields rather than a claim that it does.  The annotation
is read here only to score the parser; the parser itself never sees it.

The same pass also scores the symbolic planner under a *given* event inventory,
which isolates how much of the remaining error belongs to proposal quality
rather than to question understanding or relational reasoning.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.event_proposals import EventProposal
from mixi_understanding.qces.question_parsing import parse_question
from mixi_understanding.qces.relational_planner import plan_from_proposals

FORMAT_VERSION = "qces_v6_question_parser_audit_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test_iid", "test_label_ood", "test_compositional_ood"],
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_taxonomy(path: Path) -> tuple[str, ...]:
    composition = json.loads(path.read_text(encoding="utf-8"))["composition"]
    labels: list[str] = []
    for key in ("seen_labels", "nuisance_labels", "heldout_labels"):
        labels.extend(composition.get(key, []))
    return tuple(dict.fromkeys(labels))


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.dataset_root.resolve()
    taxonomy = load_taxonomy(root / "dataset_config.json")

    splits: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for split in args.splits:
        manifest = root / f"qces_{split}.jsonl"
        counts: dict[str, int] = defaultdict(int)
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                counts["records"] += 1
                parsed = parse_question(
                    row["question"], row["answer_options"], taxonomy
                )
                counts["parsed"] += int(parsed.ok)
                relation_ok = parsed.relation == row["relation"]
                counts["relation"] += int(relation_ok)
                if row["relation"] == "first":
                    query_ok = set(parsed.candidate_labels) == set(
                        row["query_candidate_labels"]
                    )
                else:
                    query_ok = (
                        parsed.anchor_label == row["query_label"]
                        and parsed.anchor_ordinal == row["query_instance_ordinal"]
                    )
                counts["query"] += int(query_ok)

                # Planner under a given inventory: the annotated semantic events
                # restricted to the labels the parser could name.
                named = set(parsed.query_labels)
                inventory = [
                    EventProposal(
                        label=event["label"],
                        onset_seconds=float(event["onset_seconds"]),
                        offset_seconds=float(event["offset_seconds"]),
                        confidence=1.0,
                    )
                    for event in row["events"]
                    if event.get("event_kind") == "semantic"
                    and event["label"] in named
                ]
                plan = plan_from_proposals(parsed, inventory)
                gold_no_evidence = bool(row["no_evidence"])
                counts["planner_no_evidence"] += int(
                    plan.no_evidence == gold_no_evidence
                )
                if not gold_no_evidence:
                    counts["answerable"] += 1
                    counts["planner_answer"] += int(plan.answer_label == row["answer"])
                    events = {event["event_id"]: event for event in row["events"]}
                    gold_spans = sorted(
                        (
                            round(float(events[event_id]["onset_seconds"]), 6),
                            round(float(events[event_id]["offset_seconds"]), 6),
                        )
                        for event_id in row["evidence_event_ids"]
                    )
                    predicted = sorted(
                        (round(onset, 6), round(offset, 6))
                        for onset, offset in plan.spans
                    )
                    counts["planner_span_exact"] += int(predicted == gold_spans)
                if not (relation_ok and query_ok) and len(failures) < 25:
                    failures.append(
                        {
                            "split": split,
                            "id": row["id"],
                            "question": row["question"],
                            "gold_relation": row["relation"],
                            "parsed_relation": parsed.relation,
                            "gold_query_label": row.get("query_label"),
                            "parsed_anchor_label": parsed.anchor_label,
                            "gold_ordinal": row.get("query_instance_ordinal"),
                            "parsed_ordinal": parsed.anchor_ordinal,
                            "reason": parsed.reason,
                        }
                    )

        records = counts["records"]
        answerable = max(counts["answerable"], 1)
        splits[split] = {
            "manifest_sha256": sha256_file(manifest),
            "records": records,
            "answerable_records": counts["answerable"],
            "parse_success_rate_↑": counts["parsed"] / records,
            "relation_accuracy_↑": counts["relation"] / records,
            "query_field_accuracy_↑": counts["query"] / records,
            "planner_given_inventory_no_evidence_accuracy_↑": (
                counts["planner_no_evidence"] / records
            ),
            "planner_given_inventory_answer_accuracy_↑": (
                counts["planner_answer"] / answerable
            ),
            "planner_given_inventory_span_exact_rate_↑": (
                counts["planner_span_exact"] / answerable
            ),
        }

    report = {
        "format": FORMAT_VERSION,
        "dataset_root": str(root),
        "taxonomy_size": len(taxonomy),
        "protocol": (
            "The parser reads the question string, the answer options and the "
            "published label taxonomy. The annotation is read only to score it."
        ),
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "splits": splits,
        "failure_examples": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for split, values in splits.items():
        print(
            f"{split:26s} n={values['records']:6d} "
            f"relation={values['relation_accuracy_↑']:.4f} "
            f"query={values['query_field_accuracy_↑']:.4f} "
            f"planner_ans={values['planner_given_inventory_answer_accuracy_↑']:.4f} "
            f"planner_span={values['planner_given_inventory_span_exact_rate_↑']:.4f}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
