#!/usr/bin/env python3
"""Export paper-ready external AudioQA comparison tables for QCES."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--af3", type=Path, required=True)
    parser.add_argument("--phi4", type=Path, required=True)
    parser.add_argument("--rankcal", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(report: Mapping[str, Any], condition: str, key: str) -> float | None:
    value = report["condition_metrics"][condition][key]
    return None if value is None else float(value)


def fmt(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "--"
    if signed:
        return f"{value:+.3f}"
    return f"{value:.3f}"


def best_indices(values: Sequence[float | None], *, higher: bool = True) -> set[int]:
    finite = [(idx, value) for idx, value in enumerate(values) if value is not None]
    if not finite:
        return set()
    best = max(value for _idx, value in finite) if higher else min(value for _idx, value in finite)
    return {idx for idx, value in finite if abs(value - best) < 1e-12}


def maybe_bold(text: str, idx: int, best: set[int]) -> str:
    return f"**{text}**" if idx in best else text


def maybe_bold_tex(text: str, idx: int, best: set[int]) -> str:
    return f"\\textbf{{{text}}}" if idx in best else text


def rankcal_val(report: Mapping[str, Any]) -> tuple[float, float]:
    val = report["eval_results"]["val"]
    if "max_answer_min_noev_0.57" in val:
        row = val["max_answer_min_noev_0.57"]
    elif "max_answer_min_noev_0.65" in val:
        row = val["max_answer_min_noev_0.65"]
    else:
        # Fall back to the first non-hmean policy if the policy name changes.
        names = [name for name in val if name != "hmean"]
        row = val[names[0] if names else next(iter(val))]
    return (
        float(row["answer_accuracy_↑"]),
        float(row["no_evidence_pure_accuracy_↑"]),
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = {
        "Qwen2-Audio": load_json(args.qwen.resolve()),
        "Audio Flamingo 3": load_json(args.af3.resolve()),
        "Phi-4 Multimodal": load_json(args.phi4.resolve()),
    }
    rankcal_report = load_json(args.rankcal.resolve())
    ours_answer, ours_noev = rankcal_val(rankcal_report)

    direct_rows = []
    for name, report in reports.items():
        direct_rows.append(
            {
                "method": name,
                "input": "mixture",
                "answer": metric(report, "mixture", "answerable_accuracy_↑"),
                "noev": metric(report, "mixture", "no_evidence_accuracy_↑"),
            }
        )
    direct_rows.append(
        {
            "method": "Ours: QCES-RankCal",
            "input": "mixture + evidence policy",
            "answer": ours_answer,
            "noev": ours_noev,
        }
    )

    evidence_rows = []
    for name, report in reports.items():
        mixture = metric(report, "mixture", "answerable_accuracy_↑")
        pred = metric(report, "predicted_evidence", "answerable_accuracy_↑")
        oracle = metric(report, "oracle_evidence", "answerable_accuracy_↑")
        residual = metric(report, "predicted_residual", "answerable_accuracy_↑")
        qonly = metric(report, "question_only", "answerable_accuracy_↑")
        evidence_rows.append(
            {
                "auditor": name,
                "mixture": mixture,
                "predicted_evidence": pred,
                "gain": pred - mixture,
                "oracle_evidence": oracle,
                "predicted_residual": residual,
                "question_only": qonly,
            }
        )

    direct_answer_best = best_indices([row["answer"] for row in direct_rows])
    direct_noev_best = best_indices([row["noev"] for row in direct_rows])
    evidence_gain_best = best_indices([row["gain"] for row in evidence_rows])
    evidence_pred_best = best_indices([row["predicted_evidence"] for row in evidence_rows])
    evidence_oracle_best = best_indices([row["oracle_evidence"] for row in evidence_rows])
    evidence_resid_best = best_indices([row["predicted_residual"] for row in evidence_rows], higher=False)

    md = [
        "# External AudioQA comparison on QCES validation",
        "",
        "Metric directions: answer/no-evidence/gain/oracle are higher-is-better; residual leakage is lower-is-better.",
        "",
        "## Direct AudioQA vs Ours",
        "",
        "| Method | Input | Ans. ↑ | No-ev. ↑ |",
        "|---|---|---:|---:|",
    ]
    for idx, row in enumerate(direct_rows):
        md.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    row["input"],
                    maybe_bold(fmt(row["answer"]), idx, direct_answer_best),
                    maybe_bold(fmt(row["noev"]), idx, direct_noev_best),
                ]
            )
            + " |"
        )
    md.extend(
        [
            "",
            "## Does QCES evidence help frozen AudioQA?",
            "",
            "| Frozen AudioQA model | mixture Ans. ↑ | QCES evidence Ans. ↑ | gain ↑ | oracle evidence Ans. ↑ | residual Ans. ↓ | question-only Ans. ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for idx, row in enumerate(evidence_rows):
        md.append(
            "| "
            + " | ".join(
                [
                    row["auditor"],
                    fmt(row["mixture"]),
                    maybe_bold(fmt(row["predicted_evidence"]), idx, evidence_pred_best),
                    maybe_bold(fmt(row["gain"], signed=True), idx, evidence_gain_best),
                    maybe_bold(fmt(row["oracle_evidence"]), idx, evidence_oracle_best),
                    maybe_bold(fmt(row["predicted_residual"]), idx, evidence_resid_best),
                    fmt(row["question_only"]),
                ]
            )
            + " |"
        )
    md.extend(
        [
            "",
            "## Source receipts",
            "",
            f"- Qwen2-Audio: `{args.qwen}`",
            f"- Audio Flamingo 3: `{args.af3}`",
            f"- Phi-4 Multimodal: `{args.phi4}`",
            f"- QCES-RankCal: `{args.rankcal}`",
            "",
        ]
    )

    tex = [
        "% Generated by export_qces_external_audioqa_tables.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Direct frozen AudioQA baselines versus QCES-RankCal on validation.}",
        "\\label{tab:external_direct_audioqa}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "Method & Input & Ans. $\\uparrow$ & No-ev. $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for idx, row in enumerate(direct_rows):
        tex.append(
            f"{row['method']} & {row['input']} & "
            f"{maybe_bold_tex(fmt(row['answer']), idx, direct_answer_best)} & "
            f"{maybe_bold_tex(fmt(row['noev']), idx, direct_noev_best)} \\\\"
        )
    tex.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table}",
            "",
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\caption{Frozen AudioQA auditors on mixture, predicted QCES evidence, oracle evidence and residual.}",
            "\\label{tab:external_evidence_help}",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{lrrrrrr}",
            "\\toprule",
            "Auditor & Mix. Ans. $\\uparrow$ & QCES-E Ans. $\\uparrow$ & Gain $\\uparrow$ & Oracle-E Ans. $\\uparrow$ & Residual Ans. $\\downarrow$ & Q-only Ans. $\\downarrow$ \\\\",
            "\\midrule",
        ]
    )
    for idx, row in enumerate(evidence_rows):
        tex.append(
            f"{row['auditor']} & {fmt(row['mixture'])} & "
            f"{maybe_bold_tex(fmt(row['predicted_evidence']), idx, evidence_pred_best)} & "
            f"{maybe_bold_tex(fmt(row['gain'], signed=True), idx, evidence_gain_best)} & "
            f"{maybe_bold_tex(fmt(row['oracle_evidence']), idx, evidence_oracle_best)} & "
            f"{maybe_bold_tex(fmt(row['predicted_residual']), idx, evidence_resid_best)} & "
            f"{fmt(row['question_only'])} \\\\"
        )
    tex.extend(
        [
            "\\bottomrule",
            "\\end{tabular}}",
            "\\end{table*}",
            "",
        ]
    )

    payload = {
        "direct_audioqa_vs_ours": direct_rows,
        "evidence_help": evidence_rows,
        "sources": {
            "qwen": str(args.qwen),
            "af3": str(args.af3),
            "phi4": str(args.phi4),
            "rankcal": str(args.rankcal),
        },
    }
    (output_dir / "external_audioqa_tables.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "external_audioqa_tables.md").write_text(
        "\n".join(md), encoding="utf-8"
    )
    (output_dir / "external_audioqa_tables.tex").write_text(
        "\n".join(tex), encoding="utf-8"
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
