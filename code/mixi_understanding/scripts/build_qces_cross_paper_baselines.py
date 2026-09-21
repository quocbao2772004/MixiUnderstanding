#!/usr/bin/env python3
"""Build cross-paper baseline tables for QCES.

The QCES output space is new: prior AudioQA systems do not output
question-conditioned evidence/residual waveforms.  This script therefore does
not copy raw numbers from prior papers.  Instead it aggregates the QCES runs
that adapt the closest prior paradigms onto the QCES benchmark:

* DCASE-ADQA-style audio-dependency controls: silence/chance/mixture auditor.
* Spatial-AQA-style separation-before-QA: mixture vs predicted/oracle evidence.
* LASS/AudioSep/WavCraft-style text-query tool baselines: raw question and
  oracle text/span prompts to a frozen separator.
* DAQA/CLEAR2-style temporal reasoning: symbolic planner with oracle vs
  predicted acoustic event inventory.
* QCES-RankCal: calibrated event-pair reranking and no-evidence decision.

All input files are local receipts produced by the existing QCES pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-manifest",
        type=Path,
        default=Path("data/qces_v6_full_cropbank_v2/qces_all.jsonl"),
    )
    parser.add_argument(
        "--pipeline-val",
        type=Path,
        default=Path("outputs/qces_v6_results/pipeline_val/evaluation_report.json"),
    )
    parser.add_argument(
        "--pipeline-test-iid",
        type=Path,
        default=Path("outputs/qces_v6_results/pipeline_test_iid/evaluation_report.json"),
    )
    parser.add_argument(
        "--pipeline-test-compositional-ood",
        type=Path,
        default=Path(
            "outputs/qces_v6_results/pipeline_test_compositional_ood/evaluation_report.json"
        ),
    )
    parser.add_argument(
        "--pipeline-test-label-ood",
        type=Path,
        default=Path(
            "outputs/qces_v6_results/pipeline_test_label_ood/evaluation_report.json"
        ),
    )
    parser.add_argument(
        "--rankcal",
        type=Path,
        default=Path(
            "outputs/qces_v6_pair_reranker_results/calibration_full_v1/"
            "calibration_report.json"
        ),
    )
    parser.add_argument(
        "--guarded-rankcal",
        type=Path,
        default=Path(
            "outputs/qces_v6_guarded_rankcal_results/guarded_v1/"
            "guarded_rankcal_report.json"
        ),
    )
    parser.add_argument(
        "--auditor-val",
        type=Path,
        default=Path("outputs/qces_v6_results/auditor_val/evaluation_report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qces_v6_cross_paper_baselines"),
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper/icassp2027"),
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metric(row: Mapping[str, Any], name: str) -> float | None:
    value = row.get(name)
    if value is None:
        return None
    return float(value)


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def fmt_db(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:+.2f}"


def normalize_question(question: str) -> str:
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def table_md(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def tex_escape(value: str) -> str:
    return (
        value.replace("→", "to")
        .replace("↑", "up")
        .replace("↓", "down")
        .replace("–", "--")
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def table_tex(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[str]],
    align: str | None = None,
    wide: bool = False,
) -> str:
    align = align or ("l" * len(headers))
    environment = "table*" if wide else "table"
    width = "\\textwidth" if wide else "\\columnwidth"
    body = [f"\\begin{{{environment}}}[t]", "\\centering", "\\small"]
    body.append(f"\\caption{{{tex_escape(caption)}}}")
    body.append(f"\\label{{{label}}}")
    body.append(f"\\resizebox{{{width}}}{{!}}{{%")
    body.append(f"\\begin{{tabular}}{{{align}}}")
    body.append("\\toprule")
    body.append(" & ".join(tex_escape(h) for h in headers) + " \\\\")
    body.append("\\midrule")
    for row in rows:
        body.append(" & ".join(tex_escape(cell) for cell in row) + " \\\\")
    body.append("\\bottomrule")
    body.append("\\end{tabular}")
    body.append("}%")
    body.append(f"\\end{{{environment}}}")
    return "\n".join(body)


@dataclass(frozen=True)
class EvidenceSummary:
    split: str
    method: str
    prior_paradigm: str
    annotation: str
    answer: float | None
    no_evidence: float | None
    iou: float | None
    sd_sdri: float | None
    positive_rate: float | None
    abstain: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "method": self.method,
            "prior_paradigm": self.prior_paradigm,
            "annotation": self.annotation,
            "answer_accuracy_up": self.answer,
            "no_evidence_accuracy_up": self.no_evidence,
            "temporal_iou_up": self.iou,
            "sd_sdri_db_up": self.sd_sdri,
            "positive_rate_up": self.positive_rate,
            "abstain_rate_down": self.abstain,
        }


def pipeline_row(
    split: str,
    method: str,
    paradigm: str,
    annotation: str,
    row: Mapping[str, Any],
) -> EvidenceSummary:
    return EvidenceSummary(
        split=split,
        method=method,
        prior_paradigm=paradigm,
        annotation=annotation,
        answer=metric(row, "planner_answer_accuracy_↑"),
        no_evidence=metric(row, "planner_no_evidence_accuracy_↑"),
        iou=metric(row, "planner_span_iou_mean_↑"),
        sd_sdri=metric(row, "evidence_sd_sdri_answerable_mean_db_↑"),
        positive_rate=metric(row, "evidence_sd_sdri_answerable_positive_rate_↑"),
        abstain=metric(row, "planner_abstain_rate_on_answerable_↓"),
    )


def separation_only_row(
    split: str,
    method: str,
    paradigm: str,
    annotation: str,
    row: Mapping[str, Any],
) -> EvidenceSummary:
    return EvidenceSummary(
        split=split,
        method=method,
        prior_paradigm=paradigm,
        annotation=annotation,
        answer=None,
        no_evidence=None,
        iou=None,
        sd_sdri=metric(row, "evidence_sd_sdri_answerable_mean_db_↑"),
        positive_rate=metric(row, "evidence_sd_sdri_answerable_positive_rate_↑"),
        abstain=None,
    )


def rankcal_row(
    split: str,
    method: str,
    paradigm: str,
    row: Mapping[str, Any],
) -> EvidenceSummary:
    return EvidenceSummary(
        split=split,
        method=method,
        prior_paradigm=paradigm,
        annotation="none",
        answer=metric(row, "answer_accuracy_↑"),
        no_evidence=metric(row, "no_evidence_pure_accuracy_↑"),
        iou=metric(row, "span_iou_mean_↑"),
        sd_sdri=None,
        positive_rate=None,
        abstain=metric(row, "abstain_rate_on_answerable_↓"),
    )


def build_evidence_summaries(args: argparse.Namespace) -> list[EvidenceSummary]:
    pipeline_paths = {
        "val": args.pipeline_val,
        "test-IID": args.pipeline_test_iid,
        "test-comp-OOD": args.pipeline_test_compositional_ood,
        "test-label-OOD": args.pipeline_test_label_ood,
    }
    summaries: list[EvidenceSummary] = []
    for split, path in pipeline_paths.items():
        if not path.exists():
            continue
        modes = load_json(path)["summaries_by_mode"]
        if "question_prompt__no_gate" in modes:
            summaries.append(
                separation_only_row(
                    split,
                    "Raw question → AudioSep",
                    "LASS/AudioSep direct text query",
                    "none",
                    modes["question_prompt__no_gate"],
                )
            )
        if "energy_predicted__predicted_gate" in modes:
            summaries.append(
                pipeline_row(
                    split,
                    "Energy reader",
                    "SED-style acoustic proposal",
                    "none",
                    modes["energy_predicted__predicted_gate"],
                )
            )
        if "predicted__predicted_gate" in modes:
            summaries.append(
                pipeline_row(
                    split,
                    "QCES proposal head + gate",
                    "Structured QCES baseline",
                    "none",
                    modes["predicted__predicted_gate"],
                )
            )
        if "predicted__no_gate" in modes:
            summaries.append(
                pipeline_row(
                    split,
                    "QCES proposal head, no gate",
                    "Structured QCES ablation",
                    "none",
                    modes["predicted__no_gate"],
                )
            )
        if "oracle_inventory__planned_gate" in modes:
            summaries.append(
                pipeline_row(
                    split,
                    "Symbolic planner + oracle inventory",
                    "DAQA/CLEAR2 temporal reasoning upper bound",
                    "inventory",
                    modes["oracle_inventory__planned_gate"],
                )
            )
        if "oracle_text__oracle_gate" in modes:
            summaries.append(
                separation_only_row(
                    split,
                    "Oracle text + oracle span",
                    "Spatial-AQA oracle mask analogue",
                    "labels+spans",
                    modes["oracle_text__oracle_gate"],
                )
            )

    if args.rankcal.exists():
        rankcal = load_json(args.rankcal)["eval_results"]
        split_names = {
            "val": "val",
            "test_iid": "test-IID",
            "test_compositional_ood": "test-comp-OOD",
            "test_label_ood": "test-label-OOD",
        }
        method_names = {
            "baseline_relation_threshold": (
                "Relation-threshold planner",
                "DAQA/CLEAR2-style symbolic temporal decoding",
            ),
            "pair_answer_first": (
                "QCES-RankCal answer-first",
                "QCES event-pair reranking",
            ),
            "pair_balanced_threshold": (
                "QCES-RankCal balanced threshold",
                "QCES event-pair reranking + calibration",
            ),
            "pair_no_evidence_head": (
                "QCES-RankCal no-evidence head",
                "QCES guarded reranking",
            ),
        }
        for raw_split, rows in rankcal.items():
            split = split_names.get(raw_split, raw_split)
            for mode, row in rows.items():
                method, paradigm = method_names[mode]
                summaries.append(rankcal_row(split, method, paradigm, row))
    if args.guarded_rankcal.exists():
        guarded = load_json(args.guarded_rankcal)["eval_results"]
        split_names = {
            "val": "val",
            "test_iid": "test-IID",
            "test_compositional_ood": "test-comp-OOD",
            "test_label_ood": "test-label-OOD",
        }
        for raw_split, rows in guarded.items():
            split = split_names.get(raw_split, raw_split)
            if "max_answer_min_noev_0.57" in rows:
                summaries.append(
                    rankcal_row(
                        split,
                        "QCES-RankCal guarded",
                        "QCES guarded answer-first reranking",
                        rows["max_answer_min_noev_0.57"],
                    )
                )
    return summaries


def build_auditor_table(args: argparse.Namespace) -> dict[str, Any]:
    if not args.auditor_val.exists():
        return {"available": False, "rows": [], "paired_metrics": {}}
    report = load_json(args.auditor_val)
    rows = []
    paradigm_by_condition = {
        "silence": "DCASE-ADQA silence control",
        "mixture": "AudioQA-only on mixture",
        "predicted_evidence": "Spatial-AQA predicted mask analogue",
        "oracle_evidence": "Spatial-AQA oracle mask analogue",
        "predicted_residual": "MUSA-style source-confusion residual check",
    }
    for condition, metrics in report["condition_metrics"].items():
        rows.append(
            {
                "condition": condition,
                "prior_paradigm": paradigm_by_condition.get(condition, "auditor"),
                "all_accuracy_up": metric(metrics, "multiple_choice_accuracy_all_↑"),
                "answerable_accuracy_up": metric(metrics, "answerable_accuracy_↑"),
                "no_evidence_accuracy_up": metric(metrics, "no_evidence_accuracy_↑"),
                "balanced_accuracy_up": metric(
                    metrics, "answerability_balanced_accuracy_↑"
                ),
                "chance_accuracy_up": metric(
                    metrics, "candidate_aware_chance_accuracy_↑"
                ),
                "false_answer_rate_on_no_evidence_down": metric(
                    metrics, "false_answer_rate_on_no_evidence_↓"
                ),
            }
        )
    paired = report.get("paired_metrics", {})
    return {
        "available": True,
        "counts": report.get("counts", {}),
        "rows": rows,
        "paired_metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in paired.items()
        },
    }


def _answerable(row: Mapping[str, Any]) -> bool:
    return not bool(row.get("no_evidence", False))


def _row_key(row: Mapping[str, Any], kind: str) -> str:
    if kind == "position":
        return "__position__"
    if kind == "relation_query":
        query = row.get("query_label")
        candidates = row.get("query_candidate_labels") or []
        if query:
            query_key = str(query)
        else:
            query_key = "|".join(sorted(str(item) for item in candidates))
        return f"{row.get('relation')}|{query_key}"
    if kind == "exact_question":
        return normalize_question(str(row.get("question", "")))
    raise ValueError(kind)


def _mode_answer(counts: Mapping[str, int], options: Sequence[str]) -> str | None:
    compatible = {label: int(counts.get(label, 0)) for label in options}
    if not compatible or max(compatible.values()) <= 0:
        return None
    return min(options, key=lambda label: (-compatible[label], options.index(label)))


def _position_prior_answer(train: Sequence[Mapping[str, Any]], row: Mapping[str, Any]) -> str:
    counts = Counter(int(item["answer_option_index"]) for item in train)
    index = min(range(len(row["answer_options"])), key=lambda idx: (-counts[idx], idx))
    return str(row["answer_options"][index])


def _fit_answer_lookup(
    train: Sequence[Mapping[str, Any]], kind: str
) -> dict[str, Counter[str]]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for row in train:
        table[_row_key(row, kind)][str(row["answer"])] += 1
    return table


def _predict_no_audio(
    train: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    mode: str,
) -> list[str]:
    if mode == "answer_option_position_prior":
        return [_position_prior_answer(train, row) for row in evaluation]
    if mode == "always_no_evidence":
        return ["no_evidence" for _ in evaluation]
    kind = {
        "relation_plus_query_lookup": "relation_query",
        "normalized_exact_question_lookup": "exact_question",
    }[mode]
    lookup = _fit_answer_lookup(train, kind)
    global_counts = Counter(str(row["answer"]) for row in train)
    predictions = []
    for row in evaluation:
        options = [str(item) for item in row["answer_options"]]
        answer = _mode_answer(lookup.get(_row_key(row, kind), {}), options)
        if answer is None:
            answer = _mode_answer(global_counts, options)
        if answer is None:
            answer = _position_prior_answer(train, row)
        predictions.append(answer)
    return predictions


def _candidate_aware_chance(rows: Sequence[Mapping[str, Any]]) -> float:
    # Keep the same conservative reference used by the historical QCES shortcut
    # audit: first(A, B) exposes two named candidates, while before/after stay
    # five-way multiple choice.
    return sum(0.5 if row.get("relation") == "first" else 0.2 for row in rows) / len(rows)


def _summarize_no_audio(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[str]
) -> dict[str, Any]:
    correct = [pred == str(row["answer"]) for row, pred in zip(rows, predictions)]
    answerable_pairs = [
        (row, pred)
        for row, pred in zip(rows, predictions)
        if _answerable(row)
    ]
    no_evidence_pairs = [
        (row, pred)
        for row, pred in zip(rows, predictions)
        if not _answerable(row)
    ]

    def mean(values: Sequence[bool]) -> float | None:
        return None if not values else sum(float(v) for v in values) / len(values)

    answerable_acc = mean(
        [pred == str(row["answer"]) for row, pred in answerable_pairs]
    )
    no_evidence_recall = mean([pred == "no_evidence" for _, pred in no_evidence_pairs])
    answerable_binary_recall = mean([pred != "no_evidence" for _, pred in answerable_pairs])
    balanced = (
        None
        if no_evidence_recall is None or answerable_binary_recall is None
        else 0.5 * (no_evidence_recall + answerable_binary_recall)
    )
    chance = _candidate_aware_chance(rows)
    accuracy = sum(float(value) for value in correct) / len(correct)
    return {
        "multiple_choice_accuracy_up": accuracy,
        "answerable_accuracy_up": answerable_acc,
        "no_evidence_recall_up": no_evidence_recall,
        "answerable_binary_recall_up": answerable_binary_recall,
        "no_evidence_balanced_accuracy_up": balanced,
        "candidate_aware_chance_up": chance,
        "excess_over_candidate_chance_down": accuracy - chance,
    }


def build_no_audio_controls(args: argparse.Namespace) -> dict[str, Any]:
    if not args.all_manifest.exists():
        return {"available": False, "splits": {}}
    rows = load_jsonl(args.all_manifest)
    train = [row for row in rows if row.get("split") == "train"]
    modes = [
        "answer_option_position_prior",
        "relation_plus_query_lookup",
        "normalized_exact_question_lookup",
        "always_no_evidence",
    ]
    split_names = ["val", "test_iid", "test_compositional_ood", "test_label_ood"]
    output: dict[str, Any] = {}
    for split in split_names:
        evaluation = [row for row in rows if row.get("split") == split]
        if not evaluation:
            continue
        split_output = {}
        for mode in modes:
            predictions = _predict_no_audio(train, evaluation, mode)
            split_output[mode] = _summarize_no_audio(evaluation, predictions)
        output[split] = split_output
    return {
        "available": True,
        "counts": {
            "train": len(train),
            **{
                split: sum(1 for row in rows if row.get("split") == split)
                for split in split_names
            },
        },
        "splits": output,
    }


def md_for_evidence(rows: list[EvidenceSummary]) -> str:
    selected_methods = {
        "Raw question → AudioSep",
        "Energy reader",
        "QCES proposal head + gate",
        "Relation-threshold planner",
        "QCES-RankCal answer-first",
        "QCES-RankCal guarded",
        "QCES-RankCal no-evidence head",
        "Symbolic planner + oracle inventory",
        "Oracle text + oracle span",
    }
    table_rows = []
    order = {
        "Raw question → AudioSep": 0,
        "Energy reader": 1,
        "QCES proposal head + gate": 2,
        "Relation-threshold planner": 3,
        "QCES-RankCal answer-first": 4,
        "QCES-RankCal guarded": 5,
        "QCES-RankCal no-evidence head": 6,
        "Symbolic planner + oracle inventory": 7,
        "Oracle text + oracle span": 8,
    }
    for row in sorted(
        (r for r in rows if r.method in selected_methods),
        key=lambda r: (r.split, order.get(r.method, 99)),
    ):
        table_rows.append(
            [
                row.split,
                row.method,
                row.prior_paradigm,
                row.annotation,
                fmt(row.answer),
                fmt(row.no_evidence),
                fmt(row.iou),
                fmt_db(row.sd_sdri),
                fmt(row.abstain),
            ]
        )
    return table_md(
        [
            "Split",
            "Adapted baseline",
            "Prior paradigm represented",
            "Ann.",
            "answer ↑",
            "no-evid ↑",
            "IoU ↑",
            "SD-SDRi ↑",
            "abstain ↓",
        ],
        table_rows,
    )


def md_for_no_audio(no_audio: Mapping[str, Any]) -> str:
    if not no_audio.get("available"):
        return "_No no-audio manifest found._"
    rows = []
    split_map = {
        "val": "val",
        "test_iid": "test-IID",
        "test_compositional_ood": "test-comp-OOD",
        "test_label_ood": "test-label-OOD",
    }
    method_map = {
        "answer_option_position_prior": "Option-position prior",
        "relation_plus_query_lookup": "Relation+query lookup",
        "normalized_exact_question_lookup": "Exact-question lookup",
        "always_no_evidence": "Always no-evidence",
    }
    for split, systems in no_audio["splits"].items():
        for mode, metrics in systems.items():
            rows.append(
                [
                    split_map.get(split, split),
                    method_map.get(mode, mode),
                    fmt(metrics["multiple_choice_accuracy_up"]),
                    fmt(metrics["answerable_accuracy_up"]),
                    fmt(metrics["no_evidence_recall_up"]),
                    fmt(metrics["no_evidence_balanced_accuracy_up"]),
                    fmt(metrics["candidate_aware_chance_up"]),
                    fmt(metrics["excess_over_candidate_chance_down"]),
                ]
            )
    return table_md(
        [
            "Split",
            "No-audio baseline",
            "MC acc ↑",
            "answerable acc ↑",
            "no-evid recall ↑",
            "NE balanced ↑",
            "cand. chance ↑",
            "excess over chance ↓",
        ],
        rows,
    )


def md_for_auditor(auditor: Mapping[str, Any]) -> str:
    if not auditor.get("available"):
        return "_No auditor receipt found._"
    rows = []
    for row in auditor["rows"]:
        rows.append(
            [
                row["condition"],
                row["prior_paradigm"],
                fmt(row["all_accuracy_up"]),
                fmt(row["answerable_accuracy_up"]),
                fmt(row["no_evidence_accuracy_up"]),
                fmt(row["balanced_accuracy_up"]),
                fmt(row["false_answer_rate_on_no_evidence_down"]),
            ]
        )
    return table_md(
        [
            "Auditor input",
            "Prior paradigm represented",
            "all acc ↑",
            "answerable acc ↑",
            "no-evid acc ↑",
            "balanced ↑",
            "false-answer no-evid ↓",
        ],
        rows,
    )


def build_markdown(
    evidence_rows: list[EvidenceSummary],
    auditor: Mapping[str, Any],
    no_audio: Mapping[str, Any],
) -> str:
    paired = auditor.get("paired_metrics", {})
    counts = auditor.get("counts", {})
    return "\n\n".join(
        [
            "# QCES cross-paper baseline suite",
            (
                "This table does not copy numbers from prior papers with different "
                "datasets.  Instead, it adapts the closest prior paradigms to the "
                "QCES benchmark so the comparison is on the same records and output "
                "space."
            ),
            "## 1. Evidence separation / planning baselines",
            md_for_evidence(evidence_rows),
            "## 2. No-audio shortcut controls",
            (
                "These are DCASE-ADQA-style controls: they never read waveform "
                "samples and should not substantially exceed candidate-aware chance."
            ),
            md_for_no_audio(no_audio),
            "## 3. Frozen audio-language auditor baselines",
            (
                f"Auditor subset: {counts.get('records', '--')} records; "
                f"{counts.get('answerable', '--')} answerable; "
                f"{counts.get('no_evidence', '--')} no-evidence."
            ),
            md_for_auditor(auditor),
            "## 4. Residual/evidence diagnostic metrics",
            table_md(
                ["Metric", "Value", "Desired direction"],
                [
                    [
                        "conditional sufficiency given mixture correct",
                        fmt(
                            paired.get(
                                "conditional_sufficiency_given_mixture_correct_↑"
                            )
                        ),
                        "↑",
                    ],
                    [
                        "conditional residual leakage given mixture correct",
                        fmt(
                            paired.get(
                                "conditional_residual_leakage_given_mixture_correct_↓"
                            )
                        ),
                        "↓",
                    ],
                    [
                        "predicted evidence gain over mixture",
                        fmt(
                            paired.get(
                                "predicted_evidence_accuracy_gain_over_mixture_↑"
                            )
                        ),
                        "↑",
                    ],
                    [
                        "predicted residual answer leakage",
                        fmt(
                            paired.get(
                                "predicted_residual_answer_leakage_accuracy_↓"
                            )
                        ),
                        "↓",
                    ],
                    [
                        "silence answerable accuracy",
                        fmt(paired.get("silence_control_answerable_accuracy_↓")),
                        "↓",
                    ],
                ],
            ),
            "## 5. How these rows map to prior work",
            table_md(
                ["Prior line", "QCES-adapted experiment", "What it tests"],
                [
                    [
                        "DAQA / CLEAR2 / NAAQA",
                        "symbolic temporal planner with oracle vs predicted inventory",
                        "whether temporal reasoning or acoustic proposal is bottleneck",
                    ],
                    [
                        "DCASE ADQA",
                        "silence/chance/audio-dependency controls",
                        "whether answers come from audio instead of text priors",
                    ],
                    [
                        "LASS / AudioSep",
                        "raw question or canonical text as separator query",
                        "whether text-query separation alone solves QCES",
                    ],
                    [
                        "WavCraft-style tool planning",
                        "question-to-prompt-to-separator baselines",
                        "whether a tool pipeline is enough without evidence selection",
                    ],
                    [
                        "Spatial AudioQA",
                        "mixture vs predicted evidence vs oracle evidence",
                        "separation/masking before QA under controlled evidence",
                    ],
                    [
                        "MUSA",
                        "QA on residual after evidence extraction",
                        "whether separation leaves source-confusion/leakage",
                    ],
                ],
            ),
        ]
    )


def build_tex(
    evidence_rows: list[EvidenceSummary],
    auditor: Mapping[str, Any],
    no_audio: Mapping[str, Any],
) -> str:
    compact_rows = []
    keep = [
        ("val", "Raw question → AudioSep"),
        ("val", "Energy reader"),
        ("val", "QCES proposal head + gate"),
        ("val", "Relation-threshold planner"),
        ("val", "QCES-RankCal answer-first"),
        ("val", "QCES-RankCal guarded"),
        ("val", "QCES-RankCal no-evidence head"),
        ("val", "Symbolic planner + oracle inventory"),
    ]
    by_key = {(row.split, row.method): row for row in evidence_rows}
    for key in keep:
        row = by_key.get(key)
        if row is None:
            continue
        compact_rows.append(
            [
                row.method,
                row.prior_paradigm,
                row.annotation,
                fmt(row.answer),
                fmt(row.no_evidence),
                fmt(row.iou),
                fmt_db(row.sd_sdri),
            ]
        )
    auditor_rows = []
    if auditor.get("available"):
        for row in auditor["rows"]:
            if row["condition"] in {
                "silence",
                "mixture",
                "predicted_evidence",
                "predicted_residual",
                "oracle_evidence",
            }:
                auditor_rows.append(
                    [
                        row["condition"],
                        row["prior_paradigm"],
                        fmt(row["all_accuracy_up"]),
                        fmt(row["answerable_accuracy_up"]),
                        fmt(row["no_evidence_accuracy_up"]),
                        fmt(row["false_answer_rate_on_no_evidence_down"]),
                    ]
                )
    no_audio_rows = []
    if no_audio.get("available"):
        split_map = {
            "val": "val",
            "test_iid": "test-IID",
            "test_compositional_ood": "test-comp-OOD",
            "test_label_ood": "test-label-OOD",
        }
        for split, systems in no_audio["splits"].items():
            position = systems["answer_option_position_prior"]
            lookup = systems["relation_plus_query_lookup"]
            no_audio_rows.append(
                [
                    split_map.get(split, split),
                    fmt(position["multiple_choice_accuracy_up"]),
                    fmt(lookup["multiple_choice_accuracy_up"]),
                    fmt(lookup["answerable_accuracy_up"]),
                    fmt(lookup["no_evidence_recall_up"]),
                    fmt(lookup["candidate_aware_chance_up"]),
                    fmt(lookup["excess_over_candidate_chance_down"]),
                ]
            )
    return "\n\n".join(
        [
            table_tex(
                caption=(
                    "QCES-adapted baselines representing prior AudioQA, "
                    "text-query separation, and separation-before-QA paradigms "
                    "on validation."
                ),
                label="tab:qces_cross_paper_evidence",
                headers=[
                    "Adapted baseline",
                    "Prior paradigm",
                    "Ann.",
                    "Ans.",
                    "No-ev.",
                    "IoU",
                    "SD-SDRi",
                ],
                rows=compact_rows,
                align="lllrrrr",
                wide=True,
            ),
            table_tex(
                caption=(
                    "No-audio shortcut controls adapted from audio-dependent QA "
                    "evaluation.  Systems see only question/options, never waveforms."
                ),
                label="tab:qces_no_audio_controls",
                headers=[
                    "Split",
                    "Pos.",
                    "Lookup",
                    "Ans.",
                    "No-ev.",
                    "Chance",
                    "Excess",
                ],
                rows=no_audio_rows,
                align="lrrrrrr",
            ),
            table_tex(
                caption=(
                    "Frozen auditor checks for audio-dependency, sufficiency, "
                    "and residual leakage."
                ),
                label="tab:qces_cross_paper_auditor",
                headers=[
                    "Input",
                    "Prior paradigm",
                    "All",
                    "Ans.",
                    "No-ev.",
                    "False no-ev",
                ],
                rows=auditor_rows,
                align="llrrrr",
            ),
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_dir.mkdir(parents=True, exist_ok=True)

    evidence = build_evidence_summaries(args)
    auditor = build_auditor_table(args)
    no_audio = build_no_audio_controls(args)
    payload = {
        "format": "qces_cross_paper_baseline_suite_v1",
        "interpretation": (
            "Prior papers are represented by matched QCES baselines, not by "
            "raw off-dataset scores."
        ),
        "inputs": {
            "all_manifest": str(args.all_manifest),
            "pipeline_val": str(args.pipeline_val),
            "pipeline_test_iid": str(args.pipeline_test_iid),
            "pipeline_test_compositional_ood": str(
                args.pipeline_test_compositional_ood
            ),
            "pipeline_test_label_ood": str(args.pipeline_test_label_ood),
            "rankcal": str(args.rankcal),
            "guarded_rankcal": str(args.guarded_rankcal),
            "auditor_val": str(args.auditor_val),
        },
        "evidence_rows": [row.as_dict() for row in evidence],
        "no_audio_controls": no_audio,
        "auditor": auditor,
    }
    (args.output_dir / "cross_paper_baselines.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown = build_markdown(evidence, auditor, no_audio)
    (args.output_dir / "cross_paper_baselines.md").write_text(
        markdown + "\n", encoding="utf-8"
    )
    (args.paper_dir / "qces_cross_paper_baselines.md").write_text(
        markdown + "\n", encoding="utf-8"
    )
    (args.paper_dir / "qces_cross_paper_baselines.tex").write_text(
        build_tex(evidence, auditor, no_audio) + "\n", encoding="utf-8"
    )
    print(args.output_dir / "cross_paper_baselines.md")
    print(args.paper_dir / "qces_cross_paper_baselines.tex")


if __name__ == "__main__":
    main()
