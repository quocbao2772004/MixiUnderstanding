#!/usr/bin/env python3
"""Generate a non-oracle caption/planner prompt baseline for QCES-v5.

Stage 1 captions each unique mixture once without seeing a question.  Stage 2
receives only that frozen caption and one relational question, then emits an
explicit acoustic source phrase (or ``no_evidence``) for a downstream frozen
text-conditioned separator.  Gold answers, options, event labels, stems, and
timestamps are never passed to either generation stage; annotations are used
only after generation for diagnostic planner scoring.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.scripts.evaluate_qces_audioqa import (
    AudioFlamingo3OptionScorer,
    _git_provenance,
    _model_source_provenance,
    _package_version,
    _sha256_file,
    _sha256_json,
)


FORMAT_VERSION = "qces_v5_caption_planner_prompts_v1"
CAPTION_PROMPT_VERSION = "qces_scene_event_inventory_json_v1"
PLANNER_PROMPT_VERSION = "qces_relational_evidence_phrase_json_v1"
CAPTION_PROMPT = """You are an acoustic scene transcriber.
Listen to the attached mixture and inventory every distinct audible sound-event
occurrence, including repeated occurrences of the same class. Preserve their
chronological onset order. Do not answer a question and do not infer hidden
events. Return ONLY one JSON object with this exact schema:
{"events":[{"order":1,"sound":"short acoustic description"}]}
Use concise source descriptions suitable for open-vocabulary source separation.
Do not include prose, Markdown, timestamps, confidence, or a final answer."""


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument(
        "--quantization", choices=("none", "4bit", "8bit"), default="4bit"
    )
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"), default="float16"
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default="sdpa",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--hash-model-weights", action="store_true")
    parser.add_argument("--caption-max-new-tokens", type=int, default=192)
    parser.add_argument("--planner-max-new-tokens", type=int, default=96)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--allow-test-split",
        action="store_true",
        help="Required acknowledgement before opening any QCES-v5 test split.",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_records is not None and args.max_records <= 0:
        parser.error("--max-records must be positive")
    if args.caption_max_new_tokens <= 0 or args.planner_max_new_tokens <= 0:
        parser.error("generation token limits must be positive")
    local_model = Path(args.model).expanduser().exists()
    if not local_model and args.revision is None:
        parser.error("remote --model requires an explicit pinned --revision")
    if local_model and not args.hash_model_weights and not args.validate_only:
        parser.error("a full local-model run requires --hash-model-weights")
    return args


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl_by_key(
    path: Path, *, key: str, run_fingerprint: str
) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("run_fingerprint") != run_fingerprint:
            raise ValueError(f"{path}:{line_number} has another run fingerprint")
        value = item.get(key)
        if not isinstance(value, str) or value in result:
            raise ValueError(f"{path}:{line_number} has an invalid/duplicate {key}")
        result[value] = item
    return result


def extract_first_json_object(text: str) -> Optional[Mapping[str, Any]]:
    """Extract one balanced JSON object without accepting trailing prose as data."""

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, Mapping) else None
    return None


def _clean_phrase(value: str, *, maximum: int = 240) -> str:
    result = " ".join(value.strip().split())
    result = result.strip("`\"'")
    if not result or len(result) > maximum or any(c in result for c in "{}[]"):
        raise ValueError("invalid source phrase")
    return result


def parse_caption_generation(raw: str) -> Tuple[str, List[Dict[str, Any]], str]:
    payload = extract_first_json_object(raw)
    if payload is None or set(payload) != {"events"}:
        return "invalid_json", [], raw.strip()
    events = payload.get("events")
    if not isinstance(events, list) or not events or len(events) > 32:
        return "invalid_schema", [], raw.strip()
    normalized: List[Dict[str, Any]] = []
    try:
        for expected_order, event in enumerate(events, 1):
            if not isinstance(event, Mapping) or set(event) != {"order", "sound"}:
                raise ValueError
            if event["order"] != expected_order or not isinstance(event["sound"], str):
                raise ValueError
            normalized.append(
                {"order": expected_order, "sound": _clean_phrase(event["sound"])}
            )
    except ValueError:
        return "invalid_schema", [], raw.strip()
    canonical = json.dumps(
        {"events": normalized}, ensure_ascii=False, separators=(",", ":")
    )
    return "valid", normalized, canonical


def build_planner_prompt(caption: str, question: str) -> str:
    """Build the text-only stage without any answer/option/annotation argument."""

    return f"""You plan acoustic evidence extraction from an UNTRUSTED predicted event inventory.
The inventory may omit, duplicate, or misname sounds. Use only the inventory and
the relational question below; never invent an event that is not in the inventory.

For an AFTER or BEFORE question, evidence must include both the named anchor
occurrence and the immediately adjacent answer occurrence. For a FIRST question,
evidence must include both candidate occurrences named by the question. If the
required anchor/candidates are absent, choose no_evidence. Do NOT answer the
question. Return ONLY one JSON object in exactly one of these forms:
{{"decision":"extract","source_phrase":"concise description of the required evidence sounds"}}
{{"decision":"no_evidence","source_phrase":""}}

PREDICTED_EVENT_INVENTORY:
{caption}

RELATIONAL_QUESTION:
{question}"""


def parse_planner_generation(raw: str, question: str) -> Dict[str, Any]:
    payload = extract_first_json_object(raw)
    if payload is not None and set(payload) == {"decision", "source_phrase"}:
        decision = payload.get("decision")
        phrase = payload.get("source_phrase")
        if decision == "no_evidence" and phrase == "":
            return {
                "parse_status": "valid",
                "decision": "no_evidence",
                "source_phrase": "",
                "fallback_used": False,
            }
        if decision == "extract" and isinstance(phrase, str):
            try:
                cleaned = _clean_phrase(phrase)
            except ValueError:
                pass
            else:
                return {
                    "parse_status": "valid",
                    "decision": "extract",
                    "source_phrase": cleaned,
                    "fallback_used": False,
                }
    # A malformed planner response falls back to the raw relational question.
    # This is deployable, annotation-free, and exactly the weak AudioSep control;
    # it avoids rewarding parser failures with an oracle no-evidence decision.
    return {
        "parse_status": "invalid_fallback_raw_question",
        "decision": "extract",
        "source_phrase": _clean_phrase(question),
        "fallback_used": True,
    }


def _normalize_label(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def expected_label_set(record: QCESV5Record) -> Tuple[str, ...]:
    if record.no_evidence:
        return ()
    return tuple(
        sorted(
            {
                _normalize_label(record.event_by_id(event_id).label)
                for event_id in record.evidence_event_ids
            }
        )
    )


def detected_label_set(
    source_phrase: str, vocabulary: Iterable[str]
) -> Tuple[str, ...]:
    normalized_phrase = f" {_normalize_label(source_phrase)} "
    return tuple(
        sorted(
            label
            for label in {_normalize_label(value) for value in vocabulary}
            if label and f" {label} " in normalized_phrase
        )
    )


def planner_diagnostics(
    record: QCESV5Record,
    *,
    decision: str,
    source_phrase: str,
    vocabulary: Iterable[str],
) -> Dict[str, Any]:
    expected = expected_label_set(record)
    detected = detected_label_set(source_phrase, vocabulary)
    expected_decision = "no_evidence" if record.no_evidence else "extract"
    return {
        "expected_decision_post_hoc": expected_decision,
        "expected_label_set_post_hoc": list(expected),
        "detected_label_set_post_hoc": list(detected),
        "decision_correct_↑": decision == expected_decision,
        "normalized_label_set_exact_match_↑": (
            decision == expected_decision and detected == expected
        ),
        "normalized_label_recall_↑": (
            1.0 if not expected else len(set(expected) & set(detected)) / len(expected)
        ),
    }


def _selected_dataset(
    manifest: Path, max_records: Optional[int]
) -> Tuple[QCESManifestDataset, List[Tuple[int, QCESV5Record]]]:
    dataset = QCESManifestDataset(manifest, crop_samples=None)
    selected = [
        (index, record)
        for index, record in enumerate(dataset.records)
        if isinstance(record, QCESV5Record)
    ]
    if len(selected) != len(dataset.records):
        raise ValueError("caption/planner requires one pure QCES-v5 manifest")
    if max_records is not None:
        selected = selected[:max_records]
    if not selected:
        raise ValueError("selected QCES-v5 record set is empty")
    return dataset, selected


def _model_input_contract() -> Dict[str, Any]:
    return {
        "caption_stage": {
            "mixture_audio": True,
            "fixed_caption_instruction": True,
            "question": False,
            "gold_answer": False,
            "answer_options": False,
            "event_labels_or_stems": False,
            "timestamps": False,
        },
        "planner_stage": {
            "predicted_caption": True,
            "question": True,
            "mixture_audio": False,
            "gold_answer": False,
            "answer_options": False,
            "event_labels_or_stems": False,
            "timestamps": False,
        },
        "annotations_used_only_for_post_hoc_diagnostics": True,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = args.manifest.resolve()
    dataset, selected = _selected_dataset(manifest, args.max_records)
    splits = sorted({record.split for _, record in selected})
    if any(split.startswith("test") for split in splits) and not args.allow_test_split:
        raise SystemExit("test split is sealed; pass --allow-test-split explicitly")

    scene_to_index: Dict[str, int] = {}
    for index, record in selected:
        scene_to_index.setdefault(record.scene_id, index)
    selected_ids = [record.sample_id for _, record in selected]
    model_source = _model_source_provenance(args.model, args.hash_model_weights)
    script_path = Path(__file__).resolve()
    run_config = {
        "format": FORMAT_VERSION,
        "caption_prompt_version": CAPTION_PROMPT_VERSION,
        "planner_prompt_version": PLANNER_PROMPT_VERSION,
        "script_sha256": _sha256_file(script_path),
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "selected_record_ids_sha256": _sha256_json(selected_ids),
        "selected_scene_ids_sha256": _sha256_json(sorted(scene_to_index)),
        "max_records": args.max_records,
        "splits": splits,
        "model": args.model,
        "revision": args.revision,
        "model_source_inventory_sha256": _sha256_json(model_source),
        "hash_model_weights": args.hash_model_weights,
        "quantization": args.quantization,
        "dtype": args.dtype,
        "device": args.device,
        "device_map": args.device_map,
        "attention_implementation": args.attention_implementation,
        "local_files_only": args.local_files_only,
        "caption_max_new_tokens": args.caption_max_new_tokens,
        "planner_max_new_tokens": args.planner_max_new_tokens,
        "seed": args.seed,
        "deterministic_generation": {"do_sample": False, "num_beams": 1},
        "runtime_package_versions": {
            name: _package_version(name)
            for name in ("transformers", "torch", "numpy", "scipy", "soundfile")
        },
    }
    run_fingerprint = _sha256_json(run_config)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "run_fingerprint": run_fingerprint,
                    "records_↑": len(selected),
                    "unique_scenes_↑": len(scene_to_index),
                    "test_records_accessed_↓": sum(
                        record.split.startswith("test") for _, record in selected
                    ),
                    "model_input_contract": _model_input_contract(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    output_dir = args.output_dir.resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    captions_path = output_dir / "scene_captions.jsonl"
    items_path = output_dir / "planner_items.jsonl"
    report_path = output_dir / "planner_report.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("run_fingerprint") != run_fingerprint:
            raise SystemExit("existing output has another fingerprint; use --overwrite")
    elif any(output_dir.iterdir()):
        raise SystemExit(
            "output is non-empty without run_metadata.json; use --overwrite"
        )
    else:
        metadata = {
            "run_fingerprint": run_fingerprint,
            "run_config": run_config,
            "script": str(script_path),
            "git": _git_provenance(),
            "model_source": model_source,
            "model_input_contract": _model_input_contract(),
            "runtime_model": None,
        }
        _atomic_json(metadata_path, metadata)

    captions = _load_jsonl_by_key(
        captions_path, key="scene_id", run_fingerprint=run_fingerprint
    )
    planner_items = _load_jsonl_by_key(
        items_path, key="id", run_fingerprint=run_fingerprint
    )
    unexpected_scenes = set(captions) - set(scene_to_index)
    unexpected_ids = set(planner_items) - set(selected_ids)
    if unexpected_scenes or unexpected_ids:
        raise ValueError("resume files contain unexpected scene/record IDs")

    pending_captions = sorted(set(scene_to_index) - set(captions))
    pending_records = [
        (index, record)
        for index, record in selected
        if record.sample_id not in planner_items
    ]
    generator: Optional[AudioFlamingo3OptionScorer] = None
    if pending_captions or pending_records:
        generator = AudioFlamingo3OptionScorer(
            model_name=args.model,
            revision=args.revision,
            quantization=args.quantization,
            dtype=args.dtype,
            device=args.device,
            device_map=args.device_map,
            attention_implementation=args.attention_implementation,
            local_files_only=args.local_files_only,
            seed=args.seed,
        )
        metadata["runtime_model"] = dict(generator.provenance())
        _atomic_json(metadata_path, metadata)

    assert generator is not None or (not pending_captions and not pending_records)
    for scene_id in pending_captions:
        index = scene_to_index[scene_id]
        record = dataset.records[index]
        assert isinstance(record, QCESV5Record)
        waveform = np.ascontiguousarray(
            dataset[index].mixture.numpy(), dtype=np.float32
        )
        assert generator is not None
        raw = generator.generate_text(
            CAPTION_PROMPT,
            waveform,
            record.sample_rate,
            max_new_tokens=args.caption_max_new_tokens,
        )
        status, events, canonical = parse_caption_generation(raw)
        item = {
            "run_fingerprint": run_fingerprint,
            "scene_id": scene_id,
            "mixture_path": record.mixture_path,
            "raw_generation": raw,
            "parse_status": status,
            "events": events,
            "caption_for_planner": canonical,
        }
        _append_jsonl(captions_path, item)
        captions[scene_id] = item

    vocabulary = sorted(
        {event.label for _, record in selected for event in record.events}
    )
    for _index, record in pending_records:
        caption_item = captions[record.scene_id]
        planner_prompt = build_planner_prompt(
            str(caption_item["caption_for_planner"]), record.question
        )
        assert generator is not None
        raw = generator.generate_text(
            planner_prompt,
            None,
            None,
            max_new_tokens=args.planner_max_new_tokens,
        )
        parsed = parse_planner_generation(raw, record.question)
        diagnostics = planner_diagnostics(
            record,
            decision=str(parsed["decision"]),
            source_phrase=str(parsed["source_phrase"]),
            vocabulary=vocabulary,
        )
        item = {
            "run_fingerprint": run_fingerprint,
            "id": record.sample_id,
            "scene_id": record.scene_id,
            "scene_family_id": record.scene_family_id,
            "variant_id": record.variant_id,
            "relation": record.relation,
            "question_semantics_id": record.question_semantics_id,
            "question": record.question,
            "caption_parse_status": caption_item["parse_status"],
            "raw_planner_generation": raw,
            **parsed,
            "diagnostics": diagnostics,
        }
        _append_jsonl(items_path, item)
        planner_items[record.sample_id] = item

    ordered_items = [planner_items[sample_id] for sample_id in selected_ids]
    caption_values = [captions[scene_id] for scene_id in sorted(scene_to_index)]
    answerable = [
        item
        for item in ordered_items
        if item["diagnostics"]["expected_decision_post_hoc"] == "extract"
    ]
    no_evidence = [
        item
        for item in ordered_items
        if item["diagnostics"]["expected_decision_post_hoc"] == "no_evidence"
    ]

    def mean_boolean(items: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
        if not items:
            return None
        return float(sum(bool(item["diagnostics"][key]) for item in items) / len(items))

    summary = {
        "record_count_↑": len(ordered_items),
        "scene_count_↑": len(caption_values),
        "caption_valid_json_rate_↑": sum(
            item["parse_status"] == "valid" for item in caption_values
        )
        / len(caption_values),
        "planner_valid_json_rate_↑": sum(
            item["parse_status"] == "valid" for item in ordered_items
        )
        / len(ordered_items),
        "planner_fallback_rate_↓": sum(
            bool(item["fallback_used"]) for item in ordered_items
        )
        / len(ordered_items),
        "decision_accuracy_↑": mean_boolean(ordered_items, "decision_correct_↑"),
        "answerable_label_set_exact_match_↑": mean_boolean(
            answerable, "normalized_label_set_exact_match_↑"
        ),
        "no_evidence_decision_accuracy_↑": mean_boolean(
            no_evidence, "decision_correct_↑"
        ),
        "test_records_accessed_↓": sum(
            record.split.startswith("test") for _, record in selected
        ),
    }
    report = {
        "format": FORMAT_VERSION,
        "run_fingerprint": run_fingerprint,
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "selected_record_ids_sha256": _sha256_json(selected_ids),
        "paper_result_eligible": (
            args.max_records is None and "devpilot" not in manifest.name
        ),
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
        "model_input_contract": _model_input_contract(),
        "caption_prompt_version": CAPTION_PROMPT_VERSION,
        "planner_prompt_version": PLANNER_PROMPT_VERSION,
        "caption_prompt": CAPTION_PROMPT,
        "summary": summary,
        "scene_captions_jsonl": str(captions_path),
        "planner_items_jsonl": str(items_path),
        "items": ordered_items,
    }
    _atomic_json(report_path, report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
