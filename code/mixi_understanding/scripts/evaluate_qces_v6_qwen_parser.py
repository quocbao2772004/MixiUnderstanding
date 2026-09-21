#!/usr/bin/env python3
"""Evaluate a local Qwen parser for QCES structured question understanding.

This is a side-car experiment.  It does not change the registered rule parser
or QCES-RankCal code.  It asks a local Qwen-family model to emit a strict JSON
query from question text, answer options and the public label taxonomy, then
scores the output against withheld structured fields.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.event_proposals import EventProposal
from mixi_understanding.qces.question_parsing import parse_question
from mixi_understanding.qces.qwen_question_parser import (
    qwen_parser_prompt,
    result_from_generation,
)
from mixi_understanding.qces.relational_planner import plan_from_proposals


FORMAT_VERSION = "qces_v6_qwen_question_parser_eval_v1"
NO_EVIDENCE_OPTION = "no_evidence"


@dataclass(frozen=True)
class QuestionExample:
    record_index: int
    record_id: str
    scene_id: str
    scene_family_id: str
    split: str
    variant: str
    question: str
    row: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("qwen2_audio", "qwen2_audio_hybrid", "rule"),
        default="qwen2_audio",
        help=(
            "qwen2_audio uses only Qwen JSON. qwen2_audio_hybrid falls back to "
            "the rule parser when Qwen emits invalid JSON/schema. rule is a "
            "cheap baseline/sanity check."
        ),
    )
    parser.add_argument("--model", help="Local HF model directory for Qwen backends.")
    parser.add_argument(
        "--mode",
        choices=("original", "paraphrase", "both"),
        default="both",
        help="Question surface set to evaluate.",
    )
    parser.add_argument("--paraphrases-per-record", type=int, default=1)
    parser.add_argument("--limit-records", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--quantization", choices=("none", "4bit", "8bit"), default="4bit"
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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


def ordinal_word(value: int) -> str:
    table = {
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
    return table.get(value, str(value))


def render_paraphrase(row: Mapping[str, Any], variant_index: int) -> str:
    """Deterministic paraphrases built from gold semantics for stress-testing."""

    relation = row["relation"]
    variant = variant_index % 3
    if relation in {"after", "before"}:
        label = row["query_label"]
        ordinal = int(row["query_instance_ordinal"])
        word = ordinal_word(ordinal)
        if relation == "after":
            templates = (
                f"Find occurrence #{ordinal} of {label}; which sound has the next onset?",
                f"In the timeline, what starts immediately later than {label} occurrence number {ordinal}?",
                f"Once {label} appears for the {word} time, which event begins right after it?",
            )
        else:
            templates = (
                f"Find occurrence #{ordinal} of {label}; which sound has the immediately previous onset?",
                f"In the timeline, what starts just earlier than {label} occurrence number {ordinal}?",
                f"What event begins right before occurrence #{ordinal} of {label}?",
            )
        return templates[variant]

    candidates = list(row["query_candidate_labels"])
    if len(candidates) != 2:
        return row["question"]
    first, second = candidates
    templates = (
        f"Between {first} and {second}, which sound starts earlier?",
        f"Choose the earlier onset: {first} versus {second}.",
        f"Of the two sounds {first} and {second}, which begins first?",
    )
    return templates[variant]


def iter_examples(
    manifest: Path,
    *,
    mode: str,
    paraphrases_per_record: int,
    limit_records: int | None,
) -> list[QuestionExample]:
    examples: list[QuestionExample] = []
    with manifest.open(encoding="utf-8") as handle:
        for record_index, line in enumerate(handle):
            if limit_records is not None and record_index >= limit_records:
                break
            row = json.loads(line)
            record_id = row["id"]
            if mode in {"original", "both"}:
                examples.append(
                    QuestionExample(
                        record_index=record_index,
                        record_id=record_id,
                        scene_id=row["scene_id"],
                        scene_family_id=row["scene_family_id"],
                        split=row["split"],
                        variant="original",
                        question=row["question"],
                        row=row,
                    )
                )
            if mode in {"paraphrase", "both"}:
                for variant_index in range(paraphrases_per_record):
                    examples.append(
                        QuestionExample(
                            record_index=record_index,
                            record_id=record_id,
                            scene_id=row["scene_id"],
                            scene_family_id=row["scene_family_id"],
                            split=row["split"],
                            variant=f"paraphrase_{variant_index}",
                            question=render_paraphrase(row, variant_index),
                            row=row,
                        )
                    )
    return examples


class Qwen2AudioJsonGenerator:
    """Small text-only generation wrapper around cached Qwen2-Audio."""

    def __init__(
        self,
        *,
        model_name: str,
        quantization: str,
        dtype: str,
        device: str,
        device_map: str,
        attention_implementation: str,
        local_files_only: bool,
        seed: int,
        max_new_tokens: int,
    ) -> None:
        import numpy as np
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        dtype_value = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]
        load_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
            "dtype": dtype_value,
            "attn_implementation": attention_implementation,
            "low_cpu_mem_usage": True,
        }
        if quantization != "none":
            if device not in {"auto", "cuda"}:
                raise ValueError("quantized Qwen loading requires auto/cuda device")
            from transformers import BitsAndBytesConfig

            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=dtype_value,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["device_map"] = device_map

        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name, **load_kwargs
        ).eval()
        if quantization == "none":
            resolved_device = (
                "cuda"
                if device == "auto" and torch.cuda.is_available()
                else "cpu" if device == "auto" else device
            )
            self.model.to(resolved_device)
        self.input_device = next(self.model.parameters()).device
        self.model_name = model_name
        self.quantization = quantization
        self.dtype = dtype
        self.attention_implementation = attention_implementation

    def provenance(self) -> dict[str, Any]:
        config = self.model.config
        return {
            "parser_family": "qwen2_audio_text_only",
            "model_argument": self.model_name,
            "resolved_commit_hash": getattr(config, "_commit_hash", None),
            "model_class": type(self.model).__name__,
            "processor_class": type(self.processor).__name__,
            "transformers_version": package_version("transformers"),
            "torch_version": package_version("torch"),
            "bitsandbytes_version": package_version("bitsandbytes"),
            "accelerate_version": package_version("accelerate"),
            "quantization": self.quantization,
            "dtype": self.dtype,
            "input_device": str(self.input_device),
            "attention_implementation": self.attention_implementation,
            "max_new_tokens": self.max_new_tokens,
        }

    def generate(self, prompt: str) -> str:
        conversation = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]
        text_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        prepared = self.processor(
            text=text_prompt, return_tensors="pt", padding=True
        )
        inputs = {
            key: value.to(self.input_device)
            for key, value in prepared.items()
            if key in {"input_ids", "attention_mask"}
        }
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        prefix_width = int(inputs["input_ids"].shape[1])
        continuation = generated[0, prefix_width:].detach().cpu().tolist()
        return self.processor.tokenizer.decode(
            continuation, skip_special_tokens=True
        ).strip()


def gold_query_ok(row: Mapping[str, Any], parsed) -> tuple[bool, bool, bool, bool]:
    relation_ok = parsed.relation == row["relation"]
    if row["relation"] == "first":
        anchor_ok = parsed.anchor_label is None
        ordinal_ok = True
        query_ok = set(parsed.candidate_labels) == set(row["query_candidate_labels"])
    else:
        anchor_ok = parsed.anchor_label == row["query_label"]
        ordinal_ok = parsed.anchor_ordinal == row["query_instance_ordinal"]
        query_ok = anchor_ok and ordinal_ok
    return relation_ok, anchor_ok, ordinal_ok, query_ok


def planner_scores(row: Mapping[str, Any], parsed) -> tuple[bool, bool | None, bool | None]:
    named = set(parsed.query_labels)
    inventory = [
        EventProposal(
            label=event["label"],
            onset_seconds=float(event["onset_seconds"]),
            offset_seconds=float(event["offset_seconds"]),
            confidence=1.0,
        )
        for event in row["events"]
        if event.get("event_kind") == "semantic" and event["label"] in named
    ]
    plan = plan_from_proposals(parsed, inventory)
    noev_ok = bool(plan.no_evidence) == bool(row["no_evidence"])
    if row["no_evidence"]:
        return noev_ok, None, None
    answer_ok = plan.answer_label == row["answer"]
    events = {event["event_id"]: event for event in row["events"]}
    gold_spans = sorted(
        (
            round(float(events[event_id]["onset_seconds"]), 6),
            round(float(events[event_id]["offset_seconds"]), 6),
        )
        for event_id in row["evidence_event_ids"]
    )
    predicted = sorted(
        (round(onset, 6), round(offset, 6)) for onset, offset in plan.spans
    )
    return noev_ok, answer_ok, predicted == gold_spans


def summarize(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        groups["all"].append(item)
        groups[item["variant_group"]].append(item)
        groups[f"relation:{item['gold_relation']}"].append(item)

    def rate(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
        values = [row[key] for row in rows if row[key] is not None]
        if not values:
            return None
        return sum(bool(value) for value in values) / len(values)

    summary: dict[str, Any] = {}
    for name, rows in sorted(groups.items()):
        summary[name] = {
            "examples": len(rows),
            "valid_json_rate_↑": rate(rows, "valid_json"),
            "schema_valid_rate_↑": rate(rows, "schema_valid"),
            "parse_ok_rate_↑": rate(rows, "parse_ok"),
            "relation_accuracy_↑": rate(rows, "relation_ok"),
            "anchor_accuracy_↑": rate(rows, "anchor_ok"),
            "ordinal_accuracy_↑": rate(rows, "ordinal_ok"),
            "query_field_accuracy_↑": rate(rows, "query_ok"),
            "structured_query_all_↑": rate(rows, "structured_query_all"),
            "planner_no_evidence_given_inventory_↑": rate(rows, "planner_noev_ok"),
            "planner_answer_given_inventory_↑": rate(rows, "planner_answer_ok"),
            "planner_span_exact_given_inventory_↑": rate(rows, "planner_span_ok"),
        }
    return summary


def markdown_table(report: Mapping[str, Any]) -> str:
    rows = []
    for group in ("all", "original", "paraphrase"):
        if group not in report["summary"]:
            continue
        value = report["summary"][group]
        rows.append(
            [
                group,
                str(value["examples"]),
                f"{value['valid_json_rate_↑']:.3f}" if value["valid_json_rate_↑"] is not None else "--",
                f"{value['schema_valid_rate_↑']:.3f}" if value["schema_valid_rate_↑"] is not None else "--",
                f"{value['structured_query_all_↑']:.3f}" if value["structured_query_all_↑"] is not None else "--",
                f"{value['planner_answer_given_inventory_↑']:.3f}" if value["planner_answer_given_inventory_↑"] is not None else "--",
                f"{value['planner_no_evidence_given_inventory_↑']:.3f}" if value["planner_no_evidence_given_inventory_↑"] is not None else "--",
            ]
        )
    header = [
        "| Question set | n | JSON ↑ | schema ↑ | query all ↑ | oracle-inv answer ↑ | oracle-inv no-ev ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(
        [
            "# QCES Qwen parser evaluation",
            "",
            "Higher is better for all columns.",
            "",
            *header,
            *body,
            "",
            "```json",
            json.dumps(report["provenance"], indent=2, sort_keys=True),
            "```",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = load_taxonomy(args.dataset_config.resolve())
    examples = iter_examples(
        args.manifest.resolve(),
        mode=args.mode,
        paraphrases_per_record=args.paraphrases_per_record,
        limit_records=args.limit_records,
    )
    if not examples:
        raise SystemExit("no examples to evaluate")

    generator = None
    if args.backend.startswith("qwen2_audio"):
        if not args.model:
            raise SystemExit("--model is required for Qwen backends")
        generator = Qwen2AudioJsonGenerator(
            model_name=args.model,
            quantization=args.quantization,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
            attention_implementation=args.attn_implementation,
            local_files_only=args.local_files_only,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )

    items: list[dict[str, Any]] = []
    items_path = output_dir / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as handle:
        for index, example in enumerate(examples, start=1):
            if args.backend == "rule":
                parsed = parse_question(
                    example.question, example.row["answer_options"], taxonomy
                )
                result_payload = {
                    "parsed": parsed,
                    "valid_json": None,
                    "schema_valid": None,
                    "raw_text": "",
                    "normalized_payload": {},
                    "reason": parsed.reason,
                }
            else:
                assert generator is not None
                prompt = qwen_parser_prompt(
                    question=example.question,
                    answer_options=example.row["answer_options"],
                    taxonomy=taxonomy,
                )
                raw_text = generator.generate(prompt)
                result = result_from_generation(
                    raw_text,
                    question=example.question,
                    answer_options=example.row["answer_options"],
                    taxonomy=taxonomy,
                    fallback_to_rule=args.backend.endswith("_hybrid"),
                )
                result_payload = {
                    "parsed": result.parsed,
                    "valid_json": result.valid_json,
                    "schema_valid": result.schema_valid,
                    "raw_text": result.raw_text,
                    "normalized_payload": result.normalized_payload,
                    "reason": result.reason,
                }

            parsed = result_payload["parsed"]
            relation_ok, anchor_ok, ordinal_ok, query_ok = gold_query_ok(
                example.row, parsed
            )
            planner_noev_ok, planner_answer_ok, planner_span_ok = planner_scores(
                example.row, parsed
            )
            variant_group = (
                "original" if example.variant == "original" else "paraphrase"
            )
            item = {
                "example_index": index - 1,
                "record_index": example.record_index,
                "record_id": example.record_id,
                "scene_id": example.scene_id,
                "scene_family_id": example.scene_family_id,
                "split": example.split,
                "variant": example.variant,
                "variant_group": variant_group,
                "question": example.question,
                "original_question": example.row["question"],
                "answer_options": example.row["answer_options"],
                "gold_relation": example.row["relation"],
                "gold_query_label": example.row.get("query_label"),
                "gold_query_instance_ordinal": example.row.get(
                    "query_instance_ordinal"
                ),
                "gold_query_candidate_labels": example.row.get(
                    "query_candidate_labels"
                ),
                "gold_answer": example.row["answer"],
                "gold_no_evidence": bool(example.row["no_evidence"]),
                "parsed": asdict(parsed),
                "valid_json": result_payload["valid_json"],
                "schema_valid": result_payload["schema_valid"],
                "parse_ok": bool(parsed.ok),
                "relation_ok": relation_ok,
                "anchor_ok": anchor_ok,
                "ordinal_ok": ordinal_ok,
                "query_ok": query_ok,
                "structured_query_all": relation_ok and query_ok,
                "planner_noev_ok": planner_noev_ok,
                "planner_answer_ok": planner_answer_ok,
                "planner_span_ok": planner_span_ok,
                "reason": result_payload["reason"],
                "normalized_payload": result_payload["normalized_payload"],
                "raw_text": result_payload["raw_text"],
            }
            items.append(item)
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            if index % 25 == 0 or index == len(examples):
                print(f"processed {index}/{len(examples)}", flush=True)

    provenance = {
        "format": FORMAT_VERSION,
        "backend": args.backend,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "dataset_config": str(args.dataset_config.resolve()),
        "dataset_config_sha256": sha256_file(args.dataset_config.resolve()),
        "taxonomy_size": len(taxonomy),
        "mode": args.mode,
        "paraphrases_per_record": args.paraphrases_per_record,
        "limit_records": args.limit_records,
        "parser_prompt_contract": (
            "Qwen reads question text, answer options and public taxonomy only; "
            "gold structured fields are used only for deterministic paraphrase "
            "construction and scoring."
        ),
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }
    if generator is not None:
        provenance["qwen"] = generator.provenance()
    report = {
        "format": FORMAT_VERSION,
        "provenance": provenance,
        "summary": summarize(items),
        "failure_examples": [
            item
            for item in items
            if not item["structured_query_all"]
        ][:25],
    }
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "evaluation_report.md").write_text(
        markdown_table(report), encoding="utf-8"
    )
    print(markdown_table(report))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
