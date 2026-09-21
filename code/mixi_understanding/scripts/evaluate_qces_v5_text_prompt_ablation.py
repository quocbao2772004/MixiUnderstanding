#!/usr/bin/env python3
"""Evaluate canonical text-prompt AudioSep ablations for QCES-v5.

This is an oracle-diagnostic script, not a deployable result.  It keeps the
AudioSep condition on the real text-encoder manifold, ranks a small canonical
prompt bank by frozen AudioSep response under oracle evidence windows, and can
optionally reuse a trained QCES checkpoint only for its temporal gate.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from torch.utils.data import DataLoader

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset, collate_qces
from mixi_understanding.qces.model import load_qces_checkpoint
from mixi_understanding.qces.signal import qces_linear_interpolate_1d
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    _v5_item_metrics,
    describe,
    encode_prompts,
    oracle_gate,
    summarize_v5_items,
    write_item_artifacts,
)
from mixi_understanding.scripts.evaluate_qces import (
    forward_evaluation_batch,
    load_evaluation_foundation_cache,
)
from mixi_understanding.scripts.train_qces import add_foundation_features


FORMAT_VERSION = "qces_v5_text_prompt_ablation_v1"
MODE_BEST_ORACLE_GATE = "candidate_best_text__oracle_gate"
MODE_BEST_NO_GATE = "candidate_best_text__no_gate"
MODE_BEST_QCES_GATE = "candidate_best_text__qces_predicted_gate"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--no-render-audio", action="store_true")
    parser.add_argument("--render-item-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--qces-gate-checkpoint",
        type=Path,
        help="Optional QCES checkpoint used only to supply a predicted temporal gate.",
    )
    parser.add_argument(
        "--foundation-feature-cache",
        type=Path,
        help="Required when --qces-gate-checkpoint was trained with audiosep_clap.",
    )
    parser.add_argument(
        "--qces-no-evidence-threshold",
        type=float,
        default=0.5,
        help="Descriptive threshold for the optional QCES gate checkpoint.",
    )
    args = parser.parse_args(argv)
    if args.text_batch_size <= 0:
        parser.error("--text-batch-size must be positive")
    if args.max_records < 0:
        parser.error("--max-records must be non-negative")
    if not 0.0 < args.qces_no_evidence_threshold < 1.0:
        parser.error("--qces-no-evidence-threshold must be in (0, 1)")
    if args.foundation_feature_cache is not None and args.qces_gate_checkpoint is None:
        parser.error("--foundation-feature-cache is only used with --qces-gate-checkpoint")
    if len(args.render_item_id) != len(set(args.render_item_id)):
        parser.error("--render-item-id values must be unique")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_in_order(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def evidence_labels(record: QCESV5Record) -> List[str]:
    if record.no_evidence:
        return []
    events = sorted(
        (record.event_by_id(event_id) for event_id in record.evidence_event_ids),
        key=lambda event: (event.onset_seconds, event.event_id),
    )
    return unique_in_order(event.label for event in events)


def join_labels(labels: Sequence[str], *, natural: bool = True) -> str:
    descriptions = [describe(label).replace("_", " ") for label in labels]
    if not descriptions:
        return ""
    if len(descriptions) == 1:
        return descriptions[0]
    if natural:
        return ", ".join(descriptions[:-1]) + " and " + descriptions[-1]
    return " and ".join(descriptions)


def candidate_prompts(record: QCESV5Record) -> List[str]:
    labels = evidence_labels(record)
    if not labels:
        return []
    joined = join_labels(labels)
    and_joined = join_labels(labels, natural=False)
    prompts = [
        joined,
        and_joined,
        f"the sound of {joined}",
        f"the sounds of {joined}",
        f"audio of {joined}",
        f"{joined} sound",
    ]
    if len(labels) > 1:
        first = describe(labels[0]).replace("_", " ")
        second = describe(labels[1]).replace("_", " ")
        prompts.extend(
            [
                f"{first} then {second}",
                f"{first} followed by {second}",
                f"{first} and then {second}",
            ]
        )
    for label in labels:
        phrase = describe(label).replace("_", " ")
        prompts.extend([phrase, f"the sound of {phrase}"])
    return unique_in_order(prompt.strip() for prompt in prompts if prompt.strip())


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
        "answer": record.answer,
        "no_evidence": record.no_evidence,
        "target_labels": evidence_labels(record),
    }


def summarize_by_mode(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for mode in sorted({str(item["mode"]) for item in items}):
        result[mode] = summarize_v5_items(
            [item for item in items if str(item["mode"]) == mode]
        )
    return result


def load_qces_gate_model(
    args: argparse.Namespace, device: torch.device
) -> Any | None:
    if args.qces_gate_checkpoint is None:
        return None
    checkpoint = torch.load(
        args.qces_gate_checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    model = load_qces_checkpoint(
        checkpoint,
        map_location=device,
        audiosep_repository_root=str(args.audiosep_root.resolve()),
        audiosep_config_path=str(args.audiosep_config.resolve()),
        audiosep_checkpoint_path=str(args.audiosep_checkpoint.resolve()),
    ).eval()
    return model


def qces_predicted_gates(
    args: argparse.Namespace,
    *,
    dataset: QCESManifestDataset,
    records: Sequence[QCESV5Record],
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any] | None]:
    model = load_qces_gate_model(args, device)
    if model is None:
        return {}, None
    cache = None
    if getattr(model.config, "foundation_feature_mode", "none") == "audiosep_clap":
        if args.foundation_feature_cache is None:
            raise SystemExit(
                "--foundation-feature-cache is required for this QCES gate checkpoint"
            )
        all_records = [
            record for record in dataset.records if isinstance(record, QCESV5Record)
        ]
        cache = load_evaluation_foundation_cache(
            args, args.manifest.resolve(), all_records
        )
    tokenizer = StableHashTokenizer(
        model.config.vocab_size, model.config.max_question_tokens
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda examples: collate_qces(examples, tokenizer=tokenizer),
    )
    render_ids = {record.sample_id for record in records}
    gates: Dict[str, torch.Tensor] = {}
    no_evidence_probabilities: Dict[str, float] = {}
    with torch.inference_mode():
        for index, raw_batch in enumerate(loader):
            record = dataset.records[index]
            if record.sample_id not in render_ids:
                continue
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in raw_batch.items()
            }
            add_foundation_features(batch, cache, device)
            output = forward_evaluation_batch(model, batch)
            gate = qces_linear_interpolate_1d(
                output.composition.evidence_probability[:, None],
                batch["mixture"].size(-1),
            )[0, 0].clamp(0.0, 1.0)
            gates[record.sample_id] = gate.detach()
            no_evidence_probabilities[record.sample_id] = float(
                output.composition.no_evidence_logit[0].sigmoid()
            )
    return gates, {
        "checkpoint": str(args.qces_gate_checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.qces_gate_checkpoint.resolve()),
        "foundation_feature_cache": (
            str(args.foundation_feature_cache.resolve())
            if args.foundation_feature_cache is not None
            else None
        ),
        "no_evidence_threshold": args.qces_no_evidence_threshold,
        "no_evidence_probabilities": no_evidence_probabilities,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
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

    dataset = QCESManifestDataset(args.manifest.resolve(), crop_samples=None)
    records = [
        record for record in dataset.records if isinstance(record, QCESV5Record)
    ]
    if len(records) != len(dataset.records):
        raise SystemExit("this ablation requires a pure QCES-v5 manifest")
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise SystemExit("selected record set is empty")
    selected_ids = {record.sample_id for record in records}
    unknown_render = sorted(set(args.render_item_id) - selected_ids)
    if unknown_render:
        raise SystemExit("render IDs are absent from selection: " + ", ".join(unknown_render))

    prompt_bank_by_id = {record.sample_id: candidate_prompts(record) for record in records}
    prompts = sorted({prompt for bank in prompt_bank_by_id.values() for prompt in bank})
    prompt_embeddings = encode_prompts(
        args.audiosep_root.resolve(),
        args.audiosep_checkpoint.resolve(),
        prompts,
        batch_size=args.text_batch_size,
    ) if prompts else {}
    separator = _load_separator(args, device) if prompts else None
    qces_gates, qces_gate_provenance = qces_predicted_gates(
        args, dataset=dataset, records=records, device=device
    )

    items: List[Dict[str, Any]] = []
    prompt_selection_items: List[Dict[str, Any]] = []
    separator_calls = 0
    render_ids = set(args.render_item_id)
    with torch.inference_mode():
        for index, record in enumerate(records):
            example = dataset[index]
            mixture = example.mixture.to(device)
            target = example.evidence.to(device)
            target_residual = example.residual.to(device)
            oracle_window = oracle_gate(record, mixture.numel(), device)
            raw_by_prompt: Dict[str, torch.Tensor] = {}
            if not record.no_evidence:
                assert separator is not None
                for prompt in prompt_bank_by_id[record.sample_id]:
                    condition = prompt_embeddings[prompt][None].to(device)
                    raw = separator(
                        {"mixture": mixture[None, None], "condition": condition}
                    )["waveform"][0, 0]
                    raw_by_prompt[prompt] = raw
                    separator_calls += 1
            if record.no_evidence:
                selected_prompt = ""
                selected_prompt_score = None
                selected_raw = torch.zeros_like(mixture)
            else:
                scored: List[Dict[str, Any]] = []
                for prompt, raw in raw_by_prompt.items():
                    gated = raw * oracle_window
                    metrics, _ = _v5_item_metrics(
                        no_evidence=False,
                        evidence=gated,
                        mixture=mixture,
                        target=target,
                        target_residual=target_residual,
                    )
                    scored.append(
                        {
                            "prompt": prompt,
                            "evidence_sd_sdri_db_↑": metrics[
                                "evidence_sd_sdri_db_↑"
                            ],
                            "evidence_sd_sdr_db_↑": metrics[
                                "evidence_sd_sdr_db_↑"
                            ],
                            "evidence_si_sdri_db_↑": metrics[
                                "evidence_si_sdri_db_↑"
                            ],
                        }
                    )
                scored.sort(
                    key=lambda item: (
                        float(item["evidence_sd_sdri_db_↑"]),
                        float(item["evidence_si_sdri_db_↑"]),
                        item["prompt"],
                    ),
                    reverse=True,
                )
                selected_prompt = str(scored[0]["prompt"])
                selected_prompt_score = scored[0]
                selected_raw = raw_by_prompt[selected_prompt]
                prompt_selection_items.append(
                    {
                        **metadata(record),
                        "selected_prompt": selected_prompt,
                        "selected_prompt_score": selected_prompt_score,
                        "candidate_prompts": scored,
                    }
                )

            mode_to_evidence: Dict[str, torch.Tensor] = {
                MODE_BEST_ORACLE_GATE: selected_raw * oracle_window,
                MODE_BEST_NO_GATE: selected_raw,
            }
            if qces_gates:
                qces_gate = qces_gates[record.sample_id]
                mode_to_evidence[MODE_BEST_QCES_GATE] = selected_raw * qces_gate
            for mode, evidence in mode_to_evidence.items():
                if record.no_evidence:
                    evidence = torch.zeros_like(mixture)
                residual = mixture - evidence
                metrics, descriptives = _v5_item_metrics(
                    no_evidence=record.no_evidence,
                    evidence=evidence,
                    mixture=mixture,
                    target=target,
                    target_residual=target_residual,
                )
                item = {
                    **metadata(record),
                    "mode": mode,
                    "baseline_access": "oracle_diagnostic",
                    "prompt": selected_prompt,
                    "prompt_selection_score": selected_prompt_score,
                    "metrics": metrics,
                    "descriptives": descriptives,
                    "qces_gate_available": bool(qces_gates),
                }
                should_render = not args.no_render_audio and (
                    not render_ids or record.sample_id in render_ids
                )
                write_item_artifacts(
                    question_dir=(
                        output_dir
                        / mode
                        / record.scene_id
                        / f"q{record.question_index}_{record.question_type}"
                    ),
                    item=item,
                    evidence=evidence,
                    residual=residual,
                    sample_rate=record.sample_rate,
                    render_audio=should_render,
                )
                items.append(item)

    report = {
        "format": FORMAT_VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "schema_versions": sorted({record.schema_version for record in records}),
        "selected_record_count": len(records),
        "prompt_bank": {
            "candidate_prompt_count": sum(len(bank) for bank in prompt_bank_by_id.values()),
            "unique_prompt_count": len(prompts),
            "selection": (
                "oracle response-aware: maximize evidence SD-SDRi after applying "
                "the annotated evidence window"
            ),
        },
        "mode_registry": {
            MODE_BEST_ORACLE_GATE: {
                "uses_oracle_semantics": True,
                "uses_oracle_time": True,
                "description": "Best canonical text prompt under oracle temporal gate.",
            },
            MODE_BEST_NO_GATE: {
                "uses_oracle_semantics": True,
                "uses_oracle_time": False,
                "description": "Same selected text prompt without a temporal gate.",
            },
            MODE_BEST_QCES_GATE: {
                "uses_oracle_semantics": True,
                "uses_oracle_time": False,
                "uses_qces_predicted_time": bool(qces_gates),
                "description": "Same selected text prompt with a learned QCES gate.",
            },
        },
        "qces_gate": qces_gate_provenance,
        "audiosep_frozen": True,
        "audiosep_checkpoint": str(args.audiosep_checkpoint.resolve()),
        "audiosep_checkpoint_sha256": sha256_file(args.audiosep_checkpoint.resolve()),
        "device": str(device),
        "separator_calls": separator_calls,
        "rendered_audio": not args.no_render_audio,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "protocol_limitations": [
            "This is an oracle diagnostic because prompt selection uses target waveform metrics.",
            "No-evidence rows are zeroed from the annotation and are not a deployable decision result.",
            "candidate_best_text__qces_predicted_gate still uses oracle-selected text; it isolates the temporal gate only.",
        ],
        "summaries_by_mode": summarize_by_mode(items),
        "prompt_selection_items": prompt_selection_items,
        "items": items,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summaries_by_mode"], indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
