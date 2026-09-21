#!/usr/bin/env python3
"""Evaluate an LLM question parser for QCES-style event programs.

This is intentionally a sidecar script: it does not modify the current QCES
planner/evidence code.  It tests whether a local instruction LLM can translate
natural-language questions into the structured program consumed by QCES:

* after/before: anchor label + occurrence ordinal
* first: two candidate labels

The parser is evaluated against the record metadata only for measurement.  The
prompt gives the model the public answer options and scene label inventory, not
the gold relation/query fields.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.question_parsing import parse_question


DEFAULT_MODEL = (
    "/home/cuongpv/.cache/huggingface/hub/models--Qwen--Qwen2-Audio-7B-Instruct/"
    "snapshots/0a095220c30b7b31434169c3086508ef3ea5bf0a"
)


@dataclass(frozen=True)
class ParsedProgram:
    relation: str | None
    anchor_label: str | None
    anchor_ordinal: int | None
    candidate_labels: tuple[str, ...]
    ok: bool
    reason: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="Evaluate deterministic wider-surface paraphrases of each question.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Only run the rule parser baseline; useful for checking metrics.",
    )
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_taxonomy(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    composition = payload.get("composition", {})
    labels: list[str] = []
    for key in ("seen_labels", "nuisance_labels", "heldout_labels"):
        labels.extend(composition.get(key, []))
    return tuple(dict.fromkeys(labels))


def scene_labels(row: Mapping[str, Any]) -> tuple[str, ...]:
    labels = [
        str(event["label"])
        for event in row.get("events", [])
        if event.get("event_kind") == "semantic"
    ]
    for label in row.get("answer_options", []):
        if label != "no_evidence":
            labels.append(str(label))
    for label in row.get("query_candidate_labels", []):
        labels.append(str(label))
    label = row.get("query_label")
    if label:
        labels.append(str(label))
    return tuple(dict.fromkeys(labels))


def ordinal_word(value: int | None) -> str:
    return {
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
    }.get(int(value or 1), "first")


ORDINAL_VALUES: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}


def deterministic_paraphrase(row: Mapping[str, Any]) -> str:
    """Generate a wider surface form from metadata for parser stress testing."""

    relation = str(row.get("relation"))
    label = row.get("query_label")
    ordinal = ordinal_word(row.get("query_instance_ordinal"))
    candidates = [str(x) for x in row.get("query_candidate_labels", [])]
    if relation == "after" and label:
        return f"In this clip, listen for the {ordinal} {label}. What sound starts right after it?"
    if relation == "before" and label:
        return f"In this clip, identify the event immediately preceding the {ordinal} {label} onset."
    if relation == "first" and len(candidates) >= 2:
        return f"Between {candidates[0]} and {candidates[1]}, which one starts earlier?"
    return str(row.get("question", ""))


def prompt_for(row: Mapping[str, Any], question: str) -> str:
    labels = list(scene_labels(row))
    options = [str(x) for x in row.get("answer_options", [])]
    return (
        "You are a strict parser for audio event questions. "
        "Return JSON only, no markdown.\n\n"
        "Allowed relations:\n"
        "- after: question asks for the event whose onset occurs after a named anchor occurrence.\n"
        "- before: question asks for the event whose onset occurs before a named anchor occurrence.\n"
        "- first: question compares two named candidate events and asks which starts earlier.\n\n"
        "JSON schema:\n"
        "{"
        '"relation":"after|before|first|unsupported",'
        '"anchor_label":"label or null",'
        '"anchor_ordinal":1,'
        '"candidate_labels":["label1","label2"],'
        '"confidence":0.0'
        "}\n\n"
        "Rules:\n"
        "- Use labels exactly as written from Available labels.\n"
        "- For after/before, set anchor_label and anchor_ordinal; candidate_labels may be empty.\n"
        "- For first, set candidate_labels to the two compared labels; anchor_label must be null and anchor_ordinal 1.\n"
        "- If unsupported, set relation unsupported.\n\n"
        "Examples:\n"
        "Question: The third Camera begins; what begins next?\n"
        '{"relation":"after","anchor_label":"Camera","anchor_ordinal":3,"candidate_labels":[],"confidence":1.0}\n'
        "Question: What begins just before the first Keys_jangling begins?\n"
        '{"relation":"before","anchor_label":"Keys_jangling","anchor_ordinal":1,"candidate_labels":[],"confidence":1.0}\n'
        "Question: Of Microwave_oven and Camera, which has the initial onset?\n"
        '{"relation":"first","anchor_label":null,"anchor_ordinal":1,"candidate_labels":["Microwave_oven","Camera"],"confidence":1.0}\n\n'
        f"Available labels: {labels}\n"
        f"Answer options: {options}\n"
        f"Question: {question}\n"
    )


def extract_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    starts = [i for i, ch in enumerate(stripped) if ch == "{"]
    for start in starts:
        depth = 0
        for end in range(start, len(stripped)):
            ch = stripped[end]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : end + 1]
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        try:
                            obj = ast.literal_eval(candidate)
                        except (SyntaxError, ValueError):
                            continue
                    return obj if isinstance(obj, dict) else None
    return None


def normalize_label(value: Any, labels: Sequence[str]) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().strip('"')
    if not raw or raw.lower() in {"none", "null", "n/a"}:
        return None
    if raw in labels:
        return raw
    lowered = {label.lower(): label for label in labels}
    if raw.lower() in lowered:
        return lowered[raw.lower()]
    compact = re.sub(r"[^a-z0-9]+", "", raw.lower())
    for label in labels:
        if re.sub(r"[^a-z0-9]+", "", label.lower()) == compact:
            return label
    # LLMs sometimes include the ordinal in the label field, e.g.
    # "third Camera".  Accept a unique available label contained in the field.
    contained = [
        label
        for label in labels
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])",
            raw,
            flags=re.IGNORECASE,
        )
    ]
    if len(contained) == 1:
        return contained[0]
    return None


def relation_cue(question: str) -> str | None:
    lowered = question.lower()
    if any(
        cue in lowered
        for cue in (
            "begins next",
            "starts next",
            "what begins next",
            "what starts next",
            "right after",
            "after it",
            "subsequent onset after",
            "following onset after",
        )
    ):
        return "after"
    if any(
        cue in lowered
        for cue in (
            "just before",
            "right before",
            "prior onset before",
            "preceding",
            "immediately preceding",
        )
    ):
        return "before"
    if any(
        cue in lowered
        for cue in (
            "which begins sooner",
            "which starts sooner",
            "which one starts earlier",
            "which starts earlier",
            "initial onset",
            "starts earlier",
            "begins earlier",
            "between ",
        )
    ):
        return "first"
    return None


def infer_anchor_from_question(
    question: str, labels: Sequence[str]
) -> tuple[str | None, int | None]:
    positions: list[tuple[int, str]] = []
    for label in labels:
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])",
            question,
            flags=re.IGNORECASE,
        )
        if match is not None:
            positions.append((match.start(), label))
    if not positions:
        return None, None
    positions.sort()
    label_pos, label = positions[0]
    lowered = question.lower()
    ordinal: int | None = None
    for word, value in ORDINAL_VALUES.items():
        for match in re.finditer(rf"\b{word}\b", lowered):
            if match.start() <= label_pos:
                ordinal = value
    return label, ordinal or 1


def normalize_program(
    obj: Mapping[str, Any] | None,
    labels: Sequence[str],
    question: str = "",
) -> ParsedProgram:
    if not obj:
        return ParsedProgram(None, None, None, (), False, "no_json")
    rel_raw = str(obj.get("relation", "")).strip().lower()
    relation_aliases = {
        "subsequent": "after",
        "following": "after",
        "next": "after",
        "prior": "before",
        "preceding": "before",
        "earlier": "first",
        "earliest": "first",
        "initial": "first",
    }
    relation = relation_aliases.get(rel_raw, rel_raw)
    if relation not in {"after", "before", "first"}:
        return ParsedProgram(None, None, None, (), False, f"unsupported_relation:{rel_raw}")
    try:
        ordinal = int(obj.get("anchor_ordinal") or 1)
    except (TypeError, ValueError):
        ordinal = 1
    anchor = normalize_label(obj.get("anchor_label"), labels)
    candidates_raw = obj.get("candidate_labels") or []
    if isinstance(candidates_raw, str):
        candidates_raw = [candidates_raw]
    candidates = tuple(
        label
        for label in (normalize_label(item, labels) for item in candidates_raw)
        if label is not None
    )
    cue = relation_cue(question)
    if cue in {"after", "before"}:
        inferred_anchor, inferred_ordinal = infer_anchor_from_question(question, labels)
        if anchor is None:
            anchor = inferred_anchor
        if inferred_ordinal is not None:
            ordinal = inferred_ordinal
        if anchor is not None:
            relation = cue
    elif cue == "first":
        relation = "first"
    if relation in {"after", "before"}:
        if anchor is None:
            return ParsedProgram(relation, None, ordinal, (), False, "missing_anchor")
        return ParsedProgram(relation, anchor, ordinal, (), True, "ok")
    if relation == "first" and anchor is not None and anchor not in candidates:
        candidates = (anchor, *candidates)
    if len(candidates) < 2:
        return ParsedProgram(relation, None, 1, candidates, False, "missing_candidates")
    return ParsedProgram("first", None, 1, candidates[:2], True, "ok")


def gold_program(row: Mapping[str, Any]) -> ParsedProgram:
    relation = str(row.get("relation"))
    if relation == "first":
        return ParsedProgram(
            relation="first",
            anchor_label=None,
            anchor_ordinal=1,
            candidate_labels=tuple(str(x) for x in row.get("query_candidate_labels", [])[:2]),
            ok=True,
            reason="gold",
        )
    return ParsedProgram(
        relation=relation,
        anchor_label=str(row.get("query_label")) if row.get("query_label") else None,
        anchor_ordinal=int(row.get("query_instance_ordinal") or 1),
        candidate_labels=(),
        ok=True,
        reason="gold",
    )


def rule_program(row: Mapping[str, Any], taxonomy: Sequence[str], question: str) -> ParsedProgram:
    parsed = parse_question(question, row.get("answer_options", []), taxonomy)
    return ParsedProgram(
        relation=parsed.relation,
        anchor_label=parsed.anchor_label,
        anchor_ordinal=parsed.anchor_ordinal,
        candidate_labels=parsed.candidate_labels,
        ok=parsed.ok,
        reason=parsed.reason,
    )


def program_equal(left: ParsedProgram, right: ParsedProgram) -> bool:
    if not left.ok:
        return False
    if left.relation != right.relation:
        return False
    if left.relation == "first":
        return set(left.candidate_labels[:2]) == set(right.candidate_labels[:2])
    return left.anchor_label == right.anchor_label and left.anchor_ordinal == right.anchor_ordinal


class QwenTextParser:
    def __init__(self, model_path: str, max_new_tokens: int) -> None:
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2AudioForConditionalGeneration,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            quantization_config=quant,
            trust_remote_code=True,
        )
        self.max_new_tokens = max_new_tokens

    def parse(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        generated = output[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0]


def evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    taxonomy: Sequence[str],
    llm: QwenTextParser | None,
    paraphrase: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    outputs: list[dict[str, Any]] = []
    counts = {
        "n": 0,
        "rule_exact": 0,
        "llm_exact": 0,
        "llm_ok": 0,
        "llm_json_ok": 0,
    }
    by_relation: dict[str, dict[str, int]] = {}
    for index, row in enumerate(rows, start=1):
        question = deterministic_paraphrase(row) if paraphrase else str(row["question"])
        labels = scene_labels(row)
        gold = gold_program(row)
        rule = rule_program(row, taxonomy, question)
        llm_text = ""
        llm_program = ParsedProgram(None, None, None, (), False, "no_llm")
        obj = None
        if llm is not None:
            llm_text = llm.parse(prompt_for(row, question))
            obj = extract_json(llm_text)
            llm_program = normalize_program(obj, labels, question)
        rule_exact = program_equal(rule, gold)
        llm_exact = program_equal(llm_program, gold)
        counts["n"] += 1
        counts["rule_exact"] += int(rule_exact)
        counts["llm_exact"] += int(llm_exact)
        counts["llm_ok"] += int(llm_program.ok)
        counts["llm_json_ok"] += int(obj is not None)
        rel = str(row.get("relation"))
        bucket = by_relation.setdefault(rel, {"n": 0, "rule": 0, "llm": 0})
        bucket["n"] += 1
        bucket["rule"] += int(rule_exact)
        bucket["llm"] += int(llm_exact)
        outputs.append(
            {
                "index": index,
                "id": row.get("id"),
                "question": question,
                "original_question": row.get("question"),
                "relation": row.get("relation"),
                "answer": row.get("answer"),
                "no_evidence": row.get("no_evidence"),
                "available_labels": list(labels),
                "gold_program": asdict(gold),
                "rule_program": asdict(rule),
                "rule_exact": rule_exact,
                "llm_raw": llm_text,
                "llm_json": obj,
                "llm_program": asdict(llm_program),
                "llm_exact": llm_exact,
            }
        )
        print(
            f"[{index}/{len(rows)}] {row.get('id')} "
            f"rule={'ok' if rule_exact else 'bad'} "
            f"llm={'ok' if llm_exact else 'bad'} "
            f"{llm_program.relation}/{llm_program.reason}",
            flush=True,
        )
    summary = {
        "record_count": counts["n"],
        "surface": "paraphrase" if paraphrase else "original",
        "rule_parse_exact_accuracy_↑": counts["rule_exact"] / max(counts["n"], 1),
        "llm_parse_exact_accuracy_↑": counts["llm_exact"] / max(counts["n"], 1),
        "llm_program_ok_rate_↑": counts["llm_ok"] / max(counts["n"], 1),
        "llm_json_ok_rate_↑": counts["llm_json_ok"] / max(counts["n"], 1),
        "by_relation": {
            rel: {
                "n": bucket["n"],
                "rule_parse_exact_accuracy_↑": bucket["rule"] / bucket["n"],
                "llm_parse_exact_accuracy_↑": bucket["llm"] / bucket["n"],
            }
            for rel, bucket in sorted(by_relation.items())
        },
    }
    return summary, outputs


def markdown(summary: Mapping[str, Any]) -> str:
    def f(value: Any) -> str:
        return "..." if value is None else f"{float(value):.3f}"

    lines = [
        "# QCES LLM question parser probe",
        "",
        f"- Surface: `{summary['surface']}`",
        f"- N: {summary['record_count']}",
        "",
        "| Parser | Exact program parse ↑ | JSON/program ok ↑ |",
        "|---|---:|---:|",
        f"| Rule baseline | {f(summary['rule_parse_exact_accuracy_↑'])} | ... |",
        f"| Qwen2-Audio text-only | {f(summary['llm_parse_exact_accuracy_↑'])} | {f(summary['llm_program_ok_rate_↑'])} |",
        "",
        "## By relation",
        "",
        "| Relation | N | Rule exact ↑ | Qwen exact ↑ |",
        "|---|---:|---:|---:|",
    ]
    for rel, bucket in summary["by_relation"].items():
        lines.append(
            f"| {rel} | {bucket['n']} | "
            f"{f(bucket['rule_parse_exact_accuracy_↑'])} | "
            f"{f(bucket['llm_parse_exact_accuracy_↑'])} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir is not empty: {out}; pass --overwrite")
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)
    rows = rows[args.offset : args.offset + args.limit if args.limit else None]
    taxonomy = load_taxonomy(args.dataset_config)
    llm = None if args.no_llm else QwenTextParser(args.model, args.max_new_tokens)
    summary, outputs = evaluate_rows(rows, taxonomy, llm, args.paraphrase)
    write_jsonl(out / "items.jsonl", outputs)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = markdown(summary)
    (out / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
