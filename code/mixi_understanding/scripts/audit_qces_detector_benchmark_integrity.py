#!/usr/bin/env python3
"""Fail-fast integrity audit for detector-first QCES experiments.

The command is read-only with respect to input manifests.  It reports exact
cross-split source overlap, 200-class training support, and the answerability
shortcut introduced by the historical capped QA generator.  Historical
findings remain visible for comparison but do not fail the current balanced,
unambiguous protocol.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mixi_understanding.qces.benchmark_integrity import (
    audit_class_support,
    audit_qa_event_ambiguity,
    audit_qa_shortcuts,
    audit_split_overlaps,
    build_balanced_qa_items,
)
from mixi_understanding.scripts.evaluate_qces_detector_inventory_qa import (
    build_qa_items as build_historical_qa_items,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_ROOT / "detector_manifest_train.jsonl")
    parser.add_argument("--val", type=Path, default=DEFAULT_ROOT / "detector_manifest_val.jsonl")
    parser.add_argument("--test", type=Path, default=DEFAULT_ROOT / "detector_manifest_test.jsonl")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ROOT / "ontology_200_trainable.txt")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_detector_benchmark_integrity/current/audit.json",
    )
    parser.add_argument("--qa-max-scenes", type=int, default=600)
    parser.add_argument("--max-answerable-per-scene", type=int, default=4)
    parser.add_argument("--max-no-evidence-per-scene", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--minimum-train-scenes-per-class", type=int, default=20)
    parser.add_argument("--minimum-train-active-seconds", type=float, default=60.0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_ontology(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _markdown(report: Mapping[str, Any]) -> str:
    overlap = report["split_overlap"]
    support = report["train_class_support"]
    historical = report["qa_shortcuts"]["historical"]
    balanced = report["qa_shortcuts"]["balanced"]
    ambiguity = report["balanced_qa_event_ambiguity"]
    lines = [
        "# QCES detector benchmark integrity audit",
        "",
        f"Overall status: **{'PASS' if report['passes'] else 'FAIL'}**",
        "",
        "| check | value | status |",
        "|---|---:|---|",
        f"| hard cross-split identity overlaps ↓ | {overlap['hard_overlap_count']} | {'pass' if overlap['passes'] else 'FAIL'} |",
        f"| train-positive ontology labels ↑ | {support['positive_labels']}/{support['labels']} | {'pass' if support['positive_labels'] == support['labels'] else 'FAIL'} |",
        f"| train-ready labels ↑ | {support['ready_labels']}/{support['labels']} | {'pass' if support['ready_labels'] == support['labels'] else 'FAIL'} |",
        f"| historical relation-only balanced accuracy ↓ | {historical['text_only_majority_baselines']['relation']['balanced_accuracy']:.4f} | diagnostic |",
        f"| balanced relation-only balanced accuracy ↓ | {balanced['text_only_majority_baselines']['relation']['balanced_accuracy']:.4f} | {'FAIL' if balanced['relation_shortcut'] else 'pass'} |",
        f"| balanced ambiguous headline items ↓ | {ambiguity['ambiguous_items']} | {'pass' if ambiguity['passes'] else 'FAIL'} |",
        "",
        "## Critical failures",
        "",
    ]
    if report["critical_failures"]:
        lines.extend(f"- {failure}" for failure in report["critical_failures"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Historical findings (non-gating)", ""])
    if report["historical_findings"]:
        lines.extend(f"- {finding}" for finding in report["historical_findings"])
    else:
        lines.append("- None.")
    lines.extend(["", "## Cross-split overlap", ""])
    for pair, pair_report in overlap["pairs"].items():
        lines.append(f"- `{pair}`: {pair_report['hard_overlap_count']} hard overlaps")
        for field, field_report in pair_report["fields"].items():
            lines.append(
                f"  - `{field}`: {field_report['count']} ({field_report['severity']})"
            )
    lines.extend(
        [
            "",
            "## Class support",
            "",
            f"- Scene support min/median: {support['scene_support_min']}/{support['scene_support_median']}",
            f"- Labels meeting requested readiness: {support['ready_labels']}/{support['labels']}",
            f"- Labels by minimum scene count: `{json.dumps(support['labels_by_minimum_scene_support'], sort_keys=True)}`",
            "",
            "## QA shortcut",
            "",
            f"- Historical relation × target: `{json.dumps(historical['relation_target_counts'], sort_keys=True)}`",
            f"- Balanced relation × target: `{json.dumps(balanced['relation_target_counts'], sort_keys=True)}`",
            f"- Historical invalid temporal pairs: {historical['temporal_pair_audit']['relation_invalid']}/{historical['temporal_pair_audit']['answerable_items']}",
            f"- Balanced invalid temporal pairs: {balanced['temporal_pair_audit']['relation_invalid']}/{balanced['temporal_pair_audit']['answerable_items']}",
            f"- Balanced unambiguous-event policy: `{report['qa_generation']['balanced_require_unambiguous_events']}`",
            f"- Balanced ambiguous/missing-gold items: {ambiguity['ambiguous_items']}/{ambiguity['missing_gold_items']}",
            f"- Historical text-only exact-answer accuracy (relation + anchor + ordinal): {historical['text_only_exact_answer_baselines']['relation_plus_anchor_label_plus_ordinal']['accuracy']:.4f}",
            f"- Balanced text-only exact-answer accuracy (relation + anchor + ordinal): {balanced['text_only_exact_answer_baselines']['relation_plus_anchor_label_plus_ordinal']['accuracy']:.4f}",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = {"train": args.train.resolve(), "val": args.val.resolve(), "test": args.test.resolve()}
    split_rows = {split: read_jsonl(path) for split, path in paths.items()}
    ontology = read_ontology(args.ontology.resolve())

    overlap = audit_split_overlaps(split_rows)
    support = audit_class_support(
        split_rows["train"],
        ontology=ontology,
        minimum_scenes=args.minimum_train_scenes_per_class,
        minimum_active_seconds=args.minimum_train_active_seconds,
    )
    historical_items = build_historical_qa_items(
        split_rows["val"],
        max_scenes=args.qa_max_scenes,
        max_answerable_per_scene=args.max_answerable_per_scene,
        max_no_evidence_per_scene=args.max_no_evidence_per_scene,
        min_gap_seconds=0.0,
    )
    balanced_items = build_balanced_qa_items(
        split_rows["val"],
        max_scenes=args.qa_max_scenes,
        max_answerable_per_scene=args.max_answerable_per_scene,
        max_no_evidence_per_scene=args.max_no_evidence_per_scene,
        require_unambiguous_events=True,
        seed=args.seed,
    )
    historical_shortcuts = audit_qa_shortcuts(historical_items)
    balanced_shortcuts = audit_qa_shortcuts(balanced_items)
    balanced_ambiguity = audit_qa_event_ambiguity(
        balanced_items,
        split_rows["val"],
    )

    failures: list[str] = []
    if not overlap["passes"]:
        failures.append("Exact audio/source identities overlap across train/val/test.")
    if support["positive_labels"] != support["labels"]:
        failures.append("At least one ontology label has no positive training scene.")
    if support["ready_labels"] != support["labels"]:
        failures.append(
            "The 200-class ontology is under-supported at the requested independent-scene/active-seconds gate."
        )
    historical_findings: list[str] = []
    if historical_shortcuts["relation_shortcut"]:
        historical_findings.append(
            "Historical capped QA generation leaks answerability through before/after relation."
        )
    if historical_shortcuts["temporal_pair_audit"]["relation_invalid"]:
        historical_findings.append(
            "Historical QA contains simultaneous-onset pairs that contradict strict before/after execution."
        )
    if balanced_shortcuts["relation_shortcut"]:
        failures.append("Balanced QA generation still leaks answerability through relation.")
    if balanced_shortcuts["temporal_pair_audit"]["relation_invalid"]:
        failures.append(
            "Balanced QA contains a pair that contradicts strict before/after execution."
        )
    if not balanced_ambiguity["passes"]:
        failures.append(
            "Balanced QA generation emitted an ambiguous or annotation-missing headline item."
        )

    report = {
        "format": "qces_detector_benchmark_integrity_v2",
        "input_manifests": {split: str(path) for split, path in paths.items()},
        "ontology": str(args.ontology.resolve()),
        "qa_generation": {
            "max_scenes": args.qa_max_scenes,
            "max_answerable_per_scene": args.max_answerable_per_scene,
            "max_no_evidence_per_scene": args.max_no_evidence_per_scene,
            "balanced_seed": args.seed,
            "balanced_require_unambiguous_events": True,
        },
        "split_overlap": overlap,
        "train_class_support": support,
        "qa_shortcuts": {
            "historical": historical_shortcuts,
            "balanced": balanced_shortcuts,
        },
        "balanced_qa_event_ambiguity": balanced_ambiguity,
        "historical_findings": historical_findings,
        "critical_failures": failures,
        "passes": not failures,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({
        "passes": report["passes"],
        "critical_failures": failures,
        "historical_findings": historical_findings,
        "hard_overlap_count": overlap["hard_overlap_count"],
        "ready_labels": support["ready_labels"],
        "labels": support["labels"],
        "historical_relation_balanced_accuracy": historical_shortcuts["text_only_majority_baselines"]["relation"]["balanced_accuracy"],
        "balanced_relation_balanced_accuracy": balanced_shortcuts["text_only_majority_baselines"]["relation"]["balanced_accuracy"],
        "balanced_ambiguous_items": balanced_ambiguity["ambiguous_items"],
        "balanced_missing_gold_items": balanced_ambiguity["missing_gold_items"],
        "output": str(output),
    }, ensure_ascii=False, indent=2), flush=True)
    return 1 if args.strict and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
