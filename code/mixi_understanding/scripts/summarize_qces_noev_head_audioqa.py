#!/usr/bin/env python3
"""Summarize external AudioQA runs for the noev-head QCES evidence."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


BACKENDS = (
    ("AF3", "af3"),
    ("Qwen2-Audio", "qwen2_audio"),
    ("Phi-4MM", "phi4mm"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def load_items(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_id_condition(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    table: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        table[str(row["id"])][str(row["condition"])] = row
    return table


def acc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    if not rows:
        return None
    return sum(bool(row.get("correct")) for row in rows) / len(rows)


def fmt(value: float | int | None, *, signed: bool = False) -> str:
    if value is None:
        return "..."
    if isinstance(value, int):
        return str(value)
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    answerable = [row for row in rows if not bool(row.get("no_evidence"))]
    noev = [row for row in rows if bool(row.get("no_evidence"))]
    return {
        "n": len(rows),
        "all_accuracy_↑": acc(rows),
        "answerable_accuracy_↑": acc(answerable),
        "no_evidence_accuracy_↑": acc(noev),
    }


def paired_delta(
    base: Mapping[str, Mapping[str, Mapping[str, Any]]],
    base_cond: str,
    new: Mapping[str, Mapping[str, Mapping[str, Any]]],
    new_cond: str,
) -> dict[str, Any]:
    pairs = [
        (base[sid][base_cond], new[sid][new_cond])
        for sid in sorted(set(base) & set(new))
        if base_cond in base[sid] and new_cond in new[sid]
    ]
    base_rows = [left for left, _right in pairs]
    new_rows = [right for _left, right in pairs]
    wc = sum((not bool(left["correct"])) and bool(right["correct"]) for left, right in pairs)
    cw = sum(bool(left["correct"]) and (not bool(right["correct"])) for left, right in pairs)
    cc = sum(bool(left["correct"]) and bool(right["correct"]) for left, right in pairs)
    ww = sum((not bool(left["correct"])) and (not bool(right["correct"])) for left, right in pairs)
    ans = [(left, right) for left, right in pairs if not bool(left.get("no_evidence"))]
    noev = [(left, right) for left, right in pairs if bool(left.get("no_evidence"))]
    return {
        "n": len(pairs),
        "base_accuracy_↑": acc(base_rows),
        "new_accuracy_↑": acc(new_rows),
        "delta_↑": None if acc(base_rows) is None or acc(new_rows) is None else acc(new_rows) - acc(base_rows),
        "wrong_to_correct_↑": wc,
        "correct_to_wrong_↓": cw,
        "same_correct": cc,
        "same_wrong": ww,
        "answerable_base_accuracy_↑": acc([left for left, _right in ans]),
        "answerable_new_accuracy_↑": acc([right for _left, right in ans]),
        "noev_base_accuracy_↑": acc([left for left, _right in noev]),
        "noev_new_accuracy_↑": acc([right for _left, right in noev]),
        "wrong_to_correct_by_relation": dict(Counter(left["relation"] for left, right in pairs if (not bool(left["correct"])) and bool(right["correct"]))),
        "correct_to_wrong_by_relation": dict(Counter(left["relation"] for left, right in pairs if bool(left["correct"]) and (not bool(right["correct"])))),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = args.root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for backend_name, backend_dir in BACKENDS:
        orig_items = load_items(root / backend_dir / "mixture_original_question" / "items.jsonl")
        span_items = load_items(root / backend_dir / "span_text_mixture_and_predicted_evidence" / "items.jsonl")
        orig = by_id_condition(orig_items)
        span = by_id_condition(span_items)
        sources = [
            ("mixture + original question", orig_items, "mixture"),
            ("mixture + span-text prompt (from ours)", span_items, "mixture"),
            ("predicted evidence (from ours) + span-text prompt (from ours)", span_items, "predicted_evidence"),
        ]
        for input_name, source_rows, condition in sources:
            cond_rows = [row for row in source_rows if row.get("condition") == condition]
            rows.append(
                {
                    "backend": backend_name,
                    "input": input_name,
                    **metrics(cond_rows),
                }
            )
        for title, base, base_cond in [
            ("mixture original → predicted evidence", orig, "mixture"),
            ("mixture span-text → predicted evidence", span, "mixture"),
        ]:
            pair_rows.append(
                {
                    "backend": backend_name,
                    "comparison": title,
                    **paired_delta(base, base_cond, span, "predicted_evidence"),
                }
            )

    md = [
        "# QCES fast384 external AudioQA benchmark: noev-head min0.60 evidence",
        "",
        "| Backend | Input | N | All ↑ | Answerable ↑ | No-evidence ↑ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
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
    md.extend(
        [
            "",
            "## Paired changes",
            "",
            "| Backend | Comparison | N | base ↑ | new ↑ | Δ ↑ | wrong→correct ↑ | correct→wrong ↓ | ans base ↑ | ans new ↑ | noev base ↑ | noev new ↑ |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        md.append(
            "| "
            + " | ".join(
                [
                    str(row["backend"]),
                    str(row["comparison"]),
                    fmt(row["n"]),
                    fmt(row["base_accuracy_↑"]),
                    fmt(row["new_accuracy_↑"]),
                    fmt(row["delta_↑"], signed=True),
                    fmt(row["wrong_to_correct_↑"]),
                    fmt(row["correct_to_wrong_↓"]),
                    fmt(row["answerable_base_accuracy_↑"]),
                    fmt(row["answerable_new_accuracy_↑"]),
                    fmt(row["noev_base_accuracy_↑"]),
                    fmt(row["noev_new_accuracy_↑"]),
                ]
            )
            + " |"
        )
    md.extend(
        [
            "",
            "Notes:",
            "",
            "- All metrics marked `↑` are higher-is-better; `correct→wrong ↓` is lower-is-better.",
            "- Predicted evidence is generated by QCES first-pair + proposal-head IoU-v2 + decoupled no-evidence head min0.60.",
        ]
    )
    payload = {
        "format": "qces_fast384_noev_head_audioqa_summary_v1",
        "rows": rows,
        "paired_changes": pair_rows,
    }
    (out / "benchmark_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = "\n".join(md) + "\n"
    (out / "benchmark_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
