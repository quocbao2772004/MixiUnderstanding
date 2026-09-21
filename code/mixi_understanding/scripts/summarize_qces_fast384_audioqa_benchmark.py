#!/usr/bin/env python3
"""Summarize fast384 AudioQA benchmark rows across backends and evidence modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]


BACKENDS = {
    "AF3": {
        "base": "af3",
        "oracle": "af3",
    },
    "Qwen2-Audio": {
        "base": "qwen2_audio",
        "oracle": "qwen2_audio",
    },
    "Phi-4MM": {
        "base": "phi4mm",
        "oracle": "phi4mm",
    },
}


def rows_from(path: Path, condition: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if condition is None or row.get("condition") == condition:
                rows.append(row)
    return rows


def acc(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(bool(row.get("correct")) for row in rows) / len(rows)


def metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    answerable = [row for row in rows if not bool(row.get("no_evidence"))]
    noev = [row for row in rows if bool(row.get("no_evidence"))]
    return {
        "n": len(rows),
        "all_accuracy_↑": acc(rows),
        "answerable_accuracy_↑": acc(answerable),
        "no_evidence_accuracy_↑": acc(noev),
    }


def fmt(value: float | int | None) -> str:
    if value is None:
        return "..."
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}"


def source_table() -> list[dict[str, Any]]:
    root = PROJECT_ROOT
    out: list[dict[str, Any]] = []
    for backend_name, spec in BACKENDS.items():
        base = spec["base"]
        oracle = spec["oracle"]
        sources = [
            (
                "Mixture + original question",
                root
                / f"outputs/qces_v6_external_audioqa/fast384/{base}_context_span_text/"
                "mixture_original_question/items.jsonl",
                "mixture",
            ),
            (
                "Mixture + span-text prompt",
                root
                / f"outputs/qces_v6_external_audioqa/fast384/{base}_context_span_text/"
                "span_text_mixture_and_predicted_evidence/items.jsonl",
                "mixture",
            ),
            (
                "Predicted context evidence (ours) + span-text prompt",
                root
                / f"outputs/qces_v6_external_audioqa/fast384/{base}_context_span_text/"
                "span_text_mixture_and_predicted_evidence/items.jsonl",
                "predicted_evidence",
            ),
            (
                "Predicted answer-only evidence (ours) + answer-span prompt",
                root
                / "outputs/qces_v6_external_audioqa/fast384_predicted_answer_only_span_text/"
                f"{base}_predicted_answer_only_span_text/items.jsonl",
                "predicted_evidence",
            ),
            (
                "Oracle answer-only clean stem + answer-span prompt",
                root
                / "outputs/qces_v6_external_audioqa/fast384_oracle_answer_clean_stem/"
                f"{oracle}_oracle_answer_clean_stem/items.jsonl",
                "predicted_evidence",
            ),
        ]
        for input_name, path, condition in sources:
            row_metrics = metrics(rows_from(path, condition))
            out.append(
                {
                    "backend": backend_name,
                    "input": input_name,
                    "path": str(path.relative_to(root)),
                    **row_metrics,
                }
            )
    return out


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# QCES fast384 AudioQA benchmark",
        "",
        "| Backend | Input | N | All ↑ | Answerable ↑ | No-evidence ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["backend"]),
                    str(row["input"]),
                    fmt(row["n"]),
                    fmt(row["all_accuracy_↑"]),
                    fmt(row["answerable_accuracy_↑"]),
                    fmt(row["no_evidence_accuracy_↑"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `↑` means higher is better.",
            "- `Predicted ... (ours)` uses only the current QCES predictor outputs.",
            "- `Oracle answer-only clean stem` is an upper-bound diagnostic, not a deployable method.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = source_table()
    out_root = PROJECT_ROOT / "outputs/qces_v6_external_audioqa/fast384_predicted_answer_only_span_text"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "benchmark_summary.json").write_text(
        json.dumps({"format": "qces_fast384_audioqa_benchmark_summary_v1", "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    report = render_markdown(rows)
    (out_root / "benchmark_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
