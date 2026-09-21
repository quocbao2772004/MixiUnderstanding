#!/usr/bin/env python3
"""Evaluate caption/planner prompts through the frozen AudioSep actuator."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.qces.metrics import (
    scale_dependent_sdr,
    scale_invariant_sdr,
)
from mixi_understanding.scripts.cache_audiosep_clap_features import (
    file_identity,
    source_tree_identity,
)
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    _v5_item_metrics,
    encode_prompts,
    summarize_v5_items,
    write_item_artifacts,
)
from mixi_understanding.scripts.generate_qces_v5_caption_planner import (
    FORMAT_VERSION as PLANNER_FORMAT_VERSION,
    planner_diagnostics,
)


FORMAT_VERSION = "qces_v5_caption_planner_audiosep_eval_v1"
MODE = "caption_planner_to_frozen_audiosep__non_oracle"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--planner-report", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--allow-test-split", action="store_true")
    parser.add_argument("--no-render-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.text_batch_size <= 0:
        parser.error("--text-batch-size must be positive")
    if args.max_records < 0:
        parser.error("--max-records must be non-negative")
    return args


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_planner_items(
    planner_report: Path,
    *,
    manifest: Path,
    record_ids: Sequence[str],
) -> tuple[Dict[str, Mapping[str, Any]], Dict[str, Any]]:
    report_path = planner_report.resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("format") != PLANNER_FORMAT_VERSION:
        raise ValueError("unsupported caption/planner report format")
    manifest_sha = sha256_file(manifest)
    if payload.get("manifest_sha256") != manifest_sha:
        raise ValueError("caption/planner report is bound to another manifest")
    report_manifest = payload.get("manifest")
    if (
        not isinstance(report_manifest, str)
        or Path(report_manifest).resolve() != manifest
    ):
        raise ValueError("caption/planner manifest path does not match")
    contract = payload.get("model_input_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("caption/planner model-input contract is missing")
    caption_contract = contract.get("caption_stage")
    planner_contract = contract.get("planner_stage")
    if not isinstance(caption_contract, Mapping) or not isinstance(
        planner_contract, Mapping
    ):
        raise ValueError("caption/planner model-input contract is incomplete")
    forbidden = ("gold_answer", "answer_options", "event_labels_or_stems", "timestamps")
    if any(caption_contract.get(name) is not False for name in forbidden):
        raise ValueError("caption stage consumed a forbidden oracle input")
    if any(planner_contract.get(name) is not False for name in forbidden):
        raise ValueError("planner stage consumed a forbidden oracle input")
    if (
        planner_contract.get("predicted_caption") is not True
        or planner_contract.get("question") is not True
    ):
        raise ValueError("planner stage lacks its declared deployable inputs")

    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("caption/planner report items are missing")
    indexed: Dict[str, Mapping[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ValueError("invalid caption/planner item")
        sample_id = str(item["id"])
        if sample_id in indexed:
            raise ValueError(f"duplicate caption/planner ID: {sample_id}")
        decision = item.get("decision")
        phrase = item.get("source_phrase")
        if decision not in {"extract", "no_evidence"} or not isinstance(phrase, str):
            raise ValueError(f"invalid caption/planner decision for {sample_id}")
        if (decision == "extract") != bool(phrase.strip()):
            raise ValueError(
                f"caption/planner phrase/decision mismatch for {sample_id}"
            )
        indexed[sample_id] = item
    expected = list(record_ids)
    if not set(expected).issubset(indexed):
        missing = sorted(set(expected) - set(indexed))
        raise ValueError(
            "caption/planner report is missing selected manifest IDs: "
            f"missing={missing[:3]}"
        )
    return indexed, {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "run_fingerprint": payload.get("run_fingerprint"),
        "manifest_sha256": manifest_sha,
        "model_input_contract": contract,
        "caption_prompt_version": payload.get("caption_prompt_version"),
        "planner_prompt_version": payload.get("planner_prompt_version"),
    }


def _role_metrics(
    *,
    no_evidence: bool,
    evidence: torch.Tensor,
    anchor_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    anchor_target: torch.Tensor,
    answer_target: torch.Tensor,
) -> Dict[str, float | None]:
    result: Dict[str, float | None] = {
        "anchor_si_sdr_db_↑": None,
        "answer_si_sdr_db_↑": None,
        "weakest_role_si_sdr_db_↑": None,
        "anchor_sd_sdr_db_↑": None,
        "answer_sd_sdr_db_↑": None,
        "weakest_role_sd_sdr_db_↑": None,
    }
    if no_evidence:
        return result
    predicted_anchor = evidence * anchor_mask
    predicted_answer = evidence * answer_mask
    anchor_si = float(
        scale_invariant_sdr(predicted_anchor[None], anchor_target[None])[0]
    )
    answer_si = float(
        scale_invariant_sdr(predicted_answer[None], answer_target[None])[0]
    )
    anchor_sd = float(
        scale_dependent_sdr(predicted_anchor[None], anchor_target[None])[0]
    )
    answer_sd = float(
        scale_dependent_sdr(predicted_answer[None], answer_target[None])[0]
    )
    result.update(
        {
            "anchor_si_sdr_db_↑": anchor_si,
            "answer_si_sdr_db_↑": answer_si,
            "weakest_role_si_sdr_db_↑": min(anchor_si, answer_si),
            "anchor_sd_sdr_db_↑": anchor_sd,
            "answer_sd_sdr_db_↑": answer_sd,
            "weakest_role_sd_sdr_db_↑": min(anchor_sd, answer_sd),
        }
    )
    if not all(
        math.isfinite(float(value)) for value in result.values() if value is not None
    ):
        raise ValueError("a caption/planner role metric is NaN or infinite")
    return result


def summarize_items(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = summarize_v5_items(items)

    def values(key: str) -> List[float]:
        return [
            float(item["metrics"][key])
            for item in items
            if item["metrics"].get(key) is not None
        ]

    def distribution(key: str) -> Dict[str, float | None]:
        collected = values(key)
        if not collected:
            return {"mean": None, "median": None, "minimum": None}
        return {
            "mean": sum(collected) / len(collected),
            "median": float(median(collected)),
            "minimum": min(collected),
        }

    weakest_si = distribution("weakest_role_si_sdr_db_↑")
    weakest_sd = distribution("weakest_role_sd_sdr_db_↑")
    summary.update(
        {
            "weakest_role_si_sdr_answerable_mean_db_↑": weakest_si["mean"],
            "weakest_role_si_sdr_answerable_median_db_↑": weakest_si["median"],
            "weakest_role_si_sdr_answerable_minimum_db_↑": weakest_si["minimum"],
            "weakest_role_sd_sdr_answerable_mean_db_↑": weakest_sd["mean"],
            "weakest_role_sd_sdr_answerable_median_db_↑": weakest_sd["median"],
            "weakest_role_sd_sdr_answerable_minimum_db_↑": weakest_sd["minimum"],
            "planner_fallback_rate_↓": sum(
                bool(item["planner"]["fallback_used"]) for item in items
            )
            / len(items),
            "planner_valid_json_rate_↑": sum(
                item["planner"]["parse_status"] == "valid" for item in items
            )
            / len(items),
            "planner_decision_accuracy_↑": sum(
                float(item["metrics"]["planner_decision_correct_↑"]) for item in items
            )
            / len(items),
            "planner_label_set_exact_match_↑": sum(
                float(item["metrics"]["planner_label_set_exact_match_↑"])
                for item in items
            )
            / len(items),
        }
    )
    return summary


def metadata(record: QCESV5Record) -> Dict[str, Any]:
    return {
        "id": record.sample_id,
        "split": record.split,
        "evaluation_axis": record.evaluation_axis,
        "scene_id": record.scene_id,
        "scene_family_id": record.scene_family_id,
        "variant_id": record.variant_id,
        "counterfactual_group_id": record.counterfactual_group_id,
        "question_semantics_id": record.question_semantics_id,
        "paraphrase_family_id": record.paraphrase_family_id,
        "question_index": record.question_index,
        "question_type": record.question_type,
        "relation": record.relation,
        "primary_counterfactual_probe": record.primary_counterfactual_probe,
        "mention_order_variant": record.mention_order_variant,
        "same_label_repeat": record.same_label_repeat,
        "semantic_overlap": record.semantic_overlap,
        "hard_case_tags": list(record.hard_case_tags),
        "question": record.question,
        "no_evidence": record.no_evidence,
    }


def grouped_summaries(items: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    return {
        group: summarize_items([item for item in items if str(item[field]) == group])
        for group in sorted({str(item[field]) for item in items})
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = args.manifest.resolve()
    dataset = QCESManifestDataset(manifest, crop_samples=None)
    records = [record for record in dataset.records if isinstance(record, QCESV5Record)]
    if len(records) != len(dataset.records):
        raise ValueError("caption/planner evaluator requires pure QCES-v5")
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise ValueError("selected QCES-v5 record set is empty")
    if (
        any(record.split.startswith("test") for record in records)
        and not args.allow_test_split
    ):
        raise SystemExit("test split is sealed; pass --allow-test-split explicitly")

    planner_items, planner_provenance = load_planner_items(
        args.planner_report,
        manifest=manifest,
        record_ids=[record.sample_id for record in records],
    )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = sorted(
        {
            str(planner_items[record.sample_id]["source_phrase"])
            for record in records
            if planner_items[record.sample_id]["decision"] == "extract"
        }
    )
    embeddings = (
        encode_prompts(
            args.audiosep_root.resolve(),
            args.audiosep_checkpoint.resolve(),
            prompts,
            batch_size=args.text_batch_size,
        )
        if prompts
        else {}
    )
    separator = _load_separator(args, device) if prompts else None

    items: List[Dict[str, Any]] = []
    separator_calls = 0
    vocabulary = sorted({event.label for record in records for event in record.events})
    with torch.inference_mode():
        for index, record in enumerate(records):
            example = dataset[index]
            mixture = example.mixture.to(device)
            target = example.evidence.to(device)
            target_residual = example.residual.to(device)
            planner = planner_items[record.sample_id]
            phrase = str(planner["source_phrase"])
            if planner["decision"] == "no_evidence":
                evidence = torch.zeros_like(mixture)
            else:
                assert separator is not None
                condition = embeddings[phrase][None].to(device)
                evidence = separator(
                    {"mixture": mixture[None, None], "condition": condition}
                )["waveform"][0, 0]
                separator_calls += 1
            residual = mixture - evidence
            metrics, descriptives = _v5_item_metrics(
                no_evidence=record.no_evidence,
                evidence=evidence,
                mixture=mixture,
                target=target,
                target_residual=target_residual,
            )
            metrics.update(
                _role_metrics(
                    no_evidence=record.no_evidence,
                    evidence=evidence,
                    anchor_mask=example.anchor_mask.to(device),
                    answer_mask=example.answer_mask.to(device),
                    anchor_target=example.anchor_stem.to(device),
                    answer_target=example.answer_stem.to(device),
                )
            )
            planner_scores = planner_diagnostics(
                record,
                decision=str(planner["decision"]),
                source_phrase=phrase,
                vocabulary=vocabulary,
            )
            metrics["planner_decision_correct_↑"] = float(
                planner_scores["decision_correct_↑"]
            )
            metrics["planner_label_set_exact_match_↑"] = float(
                planner_scores["normalized_label_set_exact_match_↑"]
            )
            metrics["planner_label_recall_↑"] = float(
                planner_scores["normalized_label_recall_↑"]
            )
            item = {
                **metadata(record),
                "mode": MODE,
                "baseline_access": "non_oracle",
                "prompt": phrase,
                "planner": {
                    "decision": planner["decision"],
                    "parse_status": planner["parse_status"],
                    "fallback_used": planner["fallback_used"],
                    "caption_parse_status": planner["caption_parse_status"],
                },
                "metrics": metrics,
                "descriptives": descriptives,
            }
            write_item_artifacts(
                question_dir=(
                    output_dir
                    / record.scene_id
                    / f"q{record.question_index}_{record.question_type}"
                ),
                item=item,
                evidence=evidence,
                residual=residual,
                sample_rate=record.sample_rate,
                render_audio=not args.no_render_audio,
            )
            items.append(item)

    summary = summarize_items(items)
    report = {
        "format": FORMAT_VERSION,
        "schema_versions": sorted({record.schema_version for record in records}),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "selected_record_count": len(records),
        "selection": {
            "max_records_in_manifest_order": args.max_records,
            "paper_result_eligible": (
                args.max_records == 0 and "devpilot" not in manifest.name
            ),
        },
        "mode": MODE,
        "mode_registry": {
            "access": "non_oracle",
            "caption_audio": True,
            "planner_question": True,
            "uses_oracle_semantics": False,
            "uses_oracle_time": False,
            "uses_gold_answer_or_options": False,
        },
        "planner_provenance": planner_provenance,
        "audiosep_frozen": True,
        "audiosep_checkpoint": file_identity(args.audiosep_checkpoint.resolve()),
        "audiosep_config": file_identity(args.audiosep_config.resolve()),
        "audiosep_source_tree": source_tree_identity(args.audiosep_root.resolve()),
        "device": str(device),
        "rendered_audio": not args.no_render_audio,
        "separator_calls": separator_calls,
        "behavior_descriptives": {
            "planner_extract_rate": sum(
                item["planner"]["decision"] == "extract" for item in items
            )
            / len(items),
        },
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
        "protocol_limitations": [
            "Planner exact-label diagnostics are post-hoc only and never become model inputs.",
            "The caption and planner use one frozen AF3 checkpoint; this is a strong pipeline control, not an independent human caption.",
            "R=X-E makes mixture consistency arithmetic; QA sufficiency/leakage must be audited separately.",
        ],
        "summary": summary,
        "summaries_by_relation": grouped_summaries(items, "relation"),
        "summaries_by_variant": grouped_summaries(items, "variant_id"),
        "items": items,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
