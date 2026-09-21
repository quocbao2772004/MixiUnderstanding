#!/usr/bin/env python3
"""Render reproducible frozen-AudioSep baselines on QCES v3/v4/v5.

QCES v3/v4 retain their historical three-mode report exactly.  QCES v5 uses
an explicitly versioned protocol that adds a no-separation baseline, labels
which systems use oracle information, and emits direction-marked metrics plus
counterfactual-family metadata.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Union

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf
import torch

from mixi_understanding.data.qces_schema import QCESRecord
from mixi_understanding.data.qces_v4_schema import QCESV4Record
from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.qces.metrics import (
    scale_dependent_sdr,
    scale_invariant_sdr,
)
from mixi_understanding.qces.separators import AudioSepConditionedAdapter


MODES = ("question", "oracle_semantic", "oracle_semantic_oracle_gate")
# Do not change ``MODES``: it is the historical v3/v4 mode order and therefore
# part of the legacy report contract.
MIXTURE_MODE = "mixture_no_separation"
RAW_QUESTION_MODE = "raw_question_text__weak_non_oracle"
ORACLE_SEMANTIC_MODE = "oracle_semantic"
ORACLE_SEMANTIC_GATE_MODE = "oracle_semantic_oracle_gate"
V5_MODES = (
    MIXTURE_MODE,
    RAW_QUESTION_MODE,
    ORACLE_SEMANTIC_MODE,
    ORACLE_SEMANTIC_GATE_MODE,
)
V5_MODE_REGISTRY: Dict[str, Dict[str, Any]] = {
    MIXTURE_MODE: {
        "access": "non_oracle",
        "uses_audiosep": False,
        "uses_oracle_semantics": False,
        "uses_oracle_time": False,
        "uses_oracle_no_evidence_annotation": False,
        "description": "Copy the complete mixture to the evidence output.",
    },
    RAW_QUESTION_MODE: {
        "access": "non_oracle",
        "uses_audiosep": True,
        "uses_oracle_semantics": False,
        "uses_oracle_time": False,
        "uses_oracle_no_evidence_annotation": False,
        "description": (
            "Feed the raw relational question to frozen AudioSep as a weak "
            "text-query baseline."
        ),
    },
    ORACLE_SEMANTIC_MODE: {
        "access": "oracle_upper_bound",
        "uses_audiosep": True,
        "uses_oracle_semantics": True,
        "uses_oracle_time": False,
        "uses_oracle_no_evidence_annotation": True,
        "description": (
            "Feed annotated evidence-role labels to frozen AudioSep without "
            "a temporal gate."
        ),
    },
    ORACLE_SEMANTIC_GATE_MODE: {
        "access": "oracle_upper_bound",
        "uses_audiosep": True,
        "uses_oracle_semantics": True,
        "uses_oracle_time": True,
        "uses_oracle_no_evidence_annotation": True,
        "description": (
            "Feed annotated role labels and mask the result with annotated "
            "evidence intervals."
        ),
    },
}
SupportedRecord = Union[QCESRecord, QCESV4Record, QCESV5Record]
DESCRIPTION_OVERRIDES = {
    "buzz": "a buzzing sound",
    "croak": "a frog croaking",
    "engine knocking": "an engine knocking",
    "jackhammer": "a jackhammer",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help=(
            "QCES-v5 development/debug cap in manifest order; 0 evaluates all. "
            "The legacy v3/v4 protocol rejects this option when nonzero."
        ),
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=64,
        help=(
            "Bound frozen CLAP text-encoding memory for QCES v5. Legacy v3/v4 "
            "retain their historical all-prompts-at-once encoding."
        ),
    )
    parser.add_argument(
        "--no-render-audio",
        action="store_true",
        help=(
            "Skip predicted evidence/residual WAV writes while retaining item "
            "metadata and the complete numeric evaluation report."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def describe(label: str) -> str:
    normalized = label.strip().lower()
    return DESCRIPTION_OVERRIDES.get(normalized, normalized)


def oracle_prompt(record: SupportedRecord) -> str:
    if isinstance(record, QCESV5Record):
        if record.no_evidence:
            if not record.absent_labels:
                raise ValueError(
                    f"{record.sample_id} has no absent-label oracle annotation"
                )
            labels = list(dict.fromkeys(record.absent_labels))
        else:
            events = sorted(
                (
                    record.event_by_id(event_id)
                    for event_id in record.evidence_event_ids
                ),
                key=lambda event: (event.onset_seconds, event.event_id),
            )
            # The v5 union supervision uses semantic classes, not event
            # multiplicity.  Two same-label roles therefore receive one text
            # class prompt instead of the artificial phrase "bell and bell".
            labels = list(dict.fromkeys(event.label for event in events))
        return " and ".join(describe(label) for label in labels)

    # Historical v3/v4 behavior below is intentionally unchanged.
    if record.no_evidence:
        assert record.absent_label is not None
        return describe(record.absent_label)
    events = sorted(
        (record.event_by_id(event_id) for event_id in record.evidence_event_ids),
        key=lambda event: event.onset_seconds,
    )
    return " and ".join(describe(event.label) for event in events)


def prompt_for(record: SupportedRecord, mode: str) -> str:
    if mode in {"question", RAW_QUESTION_MODE}:
        return record.question
    if mode in {ORACLE_SEMANTIC_MODE, ORACLE_SEMANTIC_GATE_MODE}:
        return oracle_prompt(record)
    raise ValueError(f"mode {mode!r} does not have a text prompt")


def oracle_gate(
    record: SupportedRecord, samples: int, device: torch.device
) -> torch.Tensor:
    gate = torch.zeros(samples, device=device)
    if record.no_evidence:
        return gate
    for event_id in record.evidence_event_ids:
        event = record.event_by_id(event_id)
        start = max(0, min(round(event.onset_seconds * record.sample_rate), samples))
        end = max(start, min(round(event.offset_seconds * record.sample_rate), samples))
        gate[start:end] = 1.0
    return gate


def encode_prompts(
    repository_root: Path,
    checkpoint_path: Path,
    prompts: Iterable[str],
    *,
    batch_size: int | None = None,
) -> Dict[str, torch.Tensor]:
    root_string = str(repository_root.resolve())
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    from models.clap_encoder import CLAP_Encoder

    # Build without the legacy 2.4 GB CLAP checkpoint, then load the exact
    # query-encoder weights bundled in the Hugging Face AudioSep checkpoint.
    encoder = CLAP_Encoder(pretrained_path="").eval()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("query_encoder."): value
        for key, value in payload.items()
        if key.startswith("query_encoder.")
    }
    incompatible = encoder.load_state_dict(state, strict=False)
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not key.endswith("embeddings.position_ids")
    ]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "incompatible AudioSep CLAP state: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )
    unique = sorted(set(prompts))
    if batch_size is None:
        # Historical v3/v4 and external-caller path.  Keep this exact call
        # shape because the legacy report contract predates bounded batching.
        with torch.inference_mode():
            embeddings = encoder.get_query_embed(modality="text", text=unique).cpu()
        result = dict(zip(unique, embeddings))
        del embeddings
    else:
        if batch_size <= 0:
            raise ValueError("text batch size must be positive")
        result = {}
        with torch.inference_mode():
            for start in range(0, len(unique), batch_size):
                batch = unique[start : start + batch_size]
                embeddings = encoder.get_query_embed(
                    modality="text", text=batch
                ).detach().cpu()
                if embeddings.shape != (len(batch), 512):
                    raise RuntimeError(
                        "unexpected AudioSep text embedding shape: "
                        f"{tuple(embeddings.shape)}"
                    )
                if not bool(torch.isfinite(embeddings).all()):
                    raise RuntimeError("AudioSep text embeddings contain NaN/Inf")
                result.update(dict(zip(batch, embeddings)))
                del embeddings
    del encoder, payload, state
    gc.collect()
    return result


def mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def write_item_artifacts(
    *,
    question_dir: Path,
    item: Mapping[str, Any],
    evidence: torch.Tensor,
    residual: torch.Tensor,
    sample_rate: int,
    render_audio: bool,
) -> None:
    """Write one item receipt and optionally its two listening WAVs."""

    question_dir.mkdir(parents=True, exist_ok=True)
    if render_audio:
        sf.write(
            question_dir / "predicted_evidence.wav",
            evidence.detach().cpu().numpy(),
            sample_rate,
        )
        sf.write(
            question_dir / "predicted_residual.wav",
            residual.detach().cpu().numpy(),
            sample_rate,
        )
    (question_dir / "metadata.json").write_text(
        json.dumps(item, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_separator(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    return AudioSepConditionedAdapter.from_repository(
        repository_root=args.audiosep_root,
        config_path=args.audiosep_config,
        checkpoint_path=args.audiosep_checkpoint,
        device=device,
        freeze_separator=True,
    ).ss_model.eval()


def _run_legacy(
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    dataset: QCESManifestDataset,
) -> None:
    """Run the historical v3/v4 protocol without changing its report."""

    if args.max_records:
        raise SystemExit("--max-records is supported only by the QCES-v5 protocol")
    records = [
        record
        for record in dataset.records
        if isinstance(record, (QCESRecord, QCESV4Record))
    ]
    prompts = [prompt_for(record, mode) for record in records for mode in MODES]
    prompt_embeddings = encode_prompts(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(), prompts
    )
    separator = _load_separator(args, device)

    items = []
    if dataset.examples is None:
        raise RuntimeError("legacy QCES v3/v4 data unexpectedly became lazy")
    with torch.inference_mode():
        for record, example in zip(records, dataset.examples):
            mixture = example.mixture[None, None].to(device)
            for mode in MODES:
                prompt = prompt_for(record, mode)
                condition = prompt_embeddings[prompt][None].to(device)
                evidence = separator(
                    {"mixture": mixture, "condition": condition}
                )["waveform"][0, 0]
                if mode == "oracle_semantic_oracle_gate":
                    evidence = evidence * oracle_gate(
                        record, evidence.numel(), device
                    )
                residual = mixture[0, 0] - evidence
                target = example.evidence.to(device)
                target_residual = example.residual.to(device)
                item = {
                    "id": record.sample_id,
                    "scene_id": record.scene_id,
                    "question_index": record.question_index,
                    "question": record.question,
                    "no_evidence": record.no_evidence,
                    "mode": mode,
                    "prompt": prompt,
                    "evidence_l1": float((evidence - target).abs().mean()),
                    "residual_l1": float((residual - target_residual).abs().mean()),
                    "retained_ratio": float(
                        evidence.abs().sum() / mixture.abs().sum().clamp_min(1e-8)
                    ),
                    "mixture_consistency_l1": float(
                        (mixture[0, 0] - evidence - residual).abs().mean()
                    ),
                    "evidence_si_sdr": None,
                    "mixture_si_sdr": None,
                    "evidence_si_sdri": None,
                }
                if not record.no_evidence:
                    item["evidence_si_sdr"] = float(
                        scale_invariant_sdr(evidence[None], target[None])[0]
                    )
                    item["mixture_si_sdr"] = float(
                        scale_invariant_sdr(mixture[:, 0], target[None])[0]
                    )
                    item["evidence_si_sdri"] = (
                        item["evidence_si_sdr"] - item["mixture_si_sdr"]
                    )
                question_dir = (
                    output_dir
                    / mode
                    / record.scene_id
                    / f"q{record.question_index}_{record.question_type}"
                )
                write_item_artifacts(
                    question_dir=question_dir,
                    item=item,
                    evidence=evidence,
                    residual=residual,
                    sample_rate=record.sample_rate,
                    render_audio=not args.no_render_audio,
                )
                items.append(item)

    summaries = {}
    for mode in MODES:
        subset = [item for item in items if item["mode"] == mode]
        answerable = [
            item["evidence_si_sdr"]
            for item in subset
            if item["evidence_si_sdr"] is not None
        ]
        improvements = [
            item["evidence_si_sdri"]
            for item in subset
            if item["evidence_si_sdri"] is not None
        ]
        negatives = [item for item in subset if item["no_evidence"]]
        summaries[mode] = {
            "evidence_si_sdr_answerable_db_↑": mean(answerable),
            "evidence_si_sdr_median_db_↑": (
                float(median(answerable)) if answerable else 0.0
            ),
            "evidence_si_sdr_minimum_db_↑": min(answerable) if answerable else 0.0,
            "evidence_si_sdri_answerable_db_↑": mean(improvements),
            "evidence_l1_↓": mean([item["evidence_l1"] for item in subset]),
            "residual_l1_↓": mean([item["residual_l1"] for item in subset]),
            "mean_no_evidence_retained_ratio_↓": mean(
                [item["retained_ratio"] for item in negatives]
            ),
            "maximum_mixture_consistency_l1_sanity_↓": max(
                item["mixture_consistency_l1"] for item in subset
            ),
        }
    report = {
        "format": "qces_audiosep_text_baselines_v1",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(
            args.manifest.resolve().read_bytes()
        ).hexdigest(),
        "audiosep_checkpoint": str(args.audiosep_checkpoint.resolve()),
        "device": str(device),
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "mixture_consistency_note": (
            "This is an arithmetic sanity check because residual is defined as X-E; "
            "it is not a faithfulness metric."
        ),
        "summaries": summaries,
        "items": items,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


def _v5_item_metrics(
    *,
    no_evidence: bool,
    evidence: torch.Tensor,
    mixture: torch.Tensor,
    target: torch.Tensor,
    target_residual: torch.Tensor,
) -> tuple[Dict[str, float | None], Dict[str, float]]:
    """Return direction-marked metrics and direction-free descriptives."""

    residual = mixture - evidence
    metrics: Dict[str, float | None] = {
        "evidence_l1_↓": float((evidence - target).abs().mean()),
        "residual_l1_↓": float((residual - target_residual).abs().mean()),
        "mixture_consistency_l1_sanity_↓": float(
            (mixture - evidence - residual).abs().mean()
        ),
        "no_evidence_retained_ratio_↓": None,
        "evidence_si_sdr_db_↑": None,
        "mixture_si_sdr_db_↑": None,
        "evidence_si_sdri_db_↑": None,
        "evidence_sd_sdr_db_↑": None,
        "mixture_sd_sdr_db_↑": None,
        "evidence_sd_sdri_db_↑": None,
    }
    retained_ratio = float(
        evidence.abs().sum() / mixture.abs().sum().clamp_min(1e-8)
    )
    if no_evidence:
        metrics["no_evidence_retained_ratio_↓"] = retained_ratio
    else:
        evidence_si_sdr = float(
            scale_invariant_sdr(evidence[None], target[None])[0]
        )
        mixture_si_sdr = float(
            scale_invariant_sdr(mixture[None], target[None])[0]
        )
        evidence_sd_sdr = float(
            scale_dependent_sdr(evidence[None], target[None])[0]
        )
        mixture_sd_sdr = float(
            scale_dependent_sdr(mixture[None], target[None])[0]
        )
        metrics.update(
            {
                "evidence_si_sdr_db_↑": evidence_si_sdr,
                "mixture_si_sdr_db_↑": mixture_si_sdr,
                "evidence_si_sdri_db_↑": evidence_si_sdr - mixture_si_sdr,
                "evidence_sd_sdr_db_↑": evidence_sd_sdr,
                "mixture_sd_sdr_db_↑": mixture_sd_sdr,
                "evidence_sd_sdri_db_↑": evidence_sd_sdr - mixture_sd_sdr,
            }
        )
    finite_values = [value for value in metrics.values() if value is not None]
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("a QCES-v5 baseline metric is NaN or infinite")
    return metrics, {
        "evidence_retained_ratio": retained_ratio,
        "evidence_absolute_peak": float(evidence.abs().max()),
    }


def summarize_v5_items(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate one non-empty mode/stratum with metric directions intact."""

    if not items:
        raise ValueError("cannot summarize an empty QCES-v5 item set")

    def values(key: str) -> List[float]:
        return [
            float(item["metrics"][key])
            for item in items
            if item["metrics"][key] is not None
        ]

    def answerable_summary(key: str) -> Dict[str, float | None]:
        collected = values(key)
        if not collected:
            return {
                "mean": None,
                "median": None,
                "minimum": None,
            }
        return {
            "mean": mean(collected),
            "median": float(median(collected)),
            "minimum": min(collected),
        }

    si_sdr = answerable_summary("evidence_si_sdr_db_↑")
    si_sdri = answerable_summary("evidence_si_sdri_db_↑")
    sd_sdr = answerable_summary("evidence_sd_sdr_db_↑")
    sd_sdri = answerable_summary("evidence_sd_sdri_db_↑")
    negative_retention = values("no_evidence_retained_ratio_↓")
    return {
        "record_count": len(items),
        "answerable_count": sum(not item["no_evidence"] for item in items),
        "no_evidence_count": sum(item["no_evidence"] for item in items),
        "evidence_si_sdr_answerable_mean_db_↑": si_sdr["mean"],
        "evidence_si_sdr_answerable_median_db_↑": si_sdr["median"],
        "evidence_si_sdr_answerable_minimum_db_↑": si_sdr["minimum"],
        "evidence_si_sdri_answerable_mean_db_↑": si_sdri["mean"],
        "evidence_sd_sdr_answerable_mean_db_↑": sd_sdr["mean"],
        "evidence_sd_sdr_answerable_median_db_↑": sd_sdr["median"],
        "evidence_sd_sdr_answerable_minimum_db_↑": sd_sdr["minimum"],
        "evidence_sd_sdri_answerable_mean_db_↑": sd_sdri["mean"],
        "evidence_l1_mean_↓": mean(values("evidence_l1_↓")),
        "residual_l1_mean_↓": mean(values("residual_l1_↓")),
        "no_evidence_retained_ratio_mean_↓": (
            mean(negative_retention) if negative_retention else None
        ),
        "mixture_consistency_l1_sanity_maximum_↓": max(
            values("mixture_consistency_l1_sanity_↓")
        ),
    }


def _summaries_by_mode(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        mode: summarize_v5_items([item for item in items if item["mode"] == mode])
        for mode in V5_MODES
    }


def _grouped_v5_summaries(
    items: Sequence[Mapping[str, Any]], field: str
) -> Dict[str, Any]:
    groups = sorted({str(item[field]) for item in items})
    return {
        group: _summaries_by_mode(
            [item for item in items if str(item[field]) == group]
        )
        for group in groups
    }


def _v5_metadata(record: QCESV5Record) -> Dict[str, Any]:
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


def _run_v5(
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    dataset: QCESManifestDataset,
) -> None:
    if args.max_records < 0:
        raise SystemExit("--max-records must be non-negative")
    if args.text_batch_size <= 0:
        raise SystemExit("--text-batch-size must be positive")
    records = [
        record for record in dataset.records if isinstance(record, QCESV5Record)
    ]
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise SystemExit("the selected QCES-v5 record set is empty")

    text_modes = [mode for mode in V5_MODES if mode != MIXTURE_MODE]
    prompts = [
        prompt_for(record, mode) for record in records for mode in text_modes
    ]
    prompt_embeddings = encode_prompts(
        args.audiosep_root.resolve(),
        args.audiosep_checkpoint.resolve(),
        prompts,
        batch_size=args.text_batch_size,
    )
    separator = _load_separator(args, device)

    items: List[Dict[str, Any]] = []
    separator_calls = 0
    with torch.inference_mode():
        for index, record in enumerate(records):
            # Paper-scale v5 is scene-derived and lazy; indexing is required.
            example = dataset[index]
            mixture = example.mixture.to(device)
            target = example.evidence.to(device)
            target_residual = example.residual.to(device)
            prompts_for_record = {
                mode: prompt_for(record, mode) for mode in text_modes
            }
            rendered_by_prompt: Dict[str, torch.Tensor] = {}
            for prompt in dict.fromkeys(prompts_for_record.values()):
                condition = prompt_embeddings[prompt][None].to(device)
                result = separator(
                    {
                        "mixture": mixture[None, None],
                        "condition": condition,
                    }
                )["waveform"]
                if result.shape != (1, 1, mixture.numel()):
                    raise ValueError(
                        "AudioSep returned an unexpected waveform shape: "
                        f"{tuple(result.shape)}"
                    )
                rendered_by_prompt[prompt] = result[0, 0]
                separator_calls += 1
            for mode in V5_MODES:
                prompt: str | None = None
                if mode == MIXTURE_MODE:
                    evidence = mixture
                else:
                    prompt = prompts_for_record[mode]
                    evidence = rendered_by_prompt[prompt]
                    if mode == ORACLE_SEMANTIC_GATE_MODE:
                        evidence = evidence * oracle_gate(
                            record, evidence.numel(), device
                        )
                residual = mixture - evidence
                metrics, descriptives = _v5_item_metrics(
                    no_evidence=record.no_evidence,
                    evidence=evidence,
                    mixture=mixture,
                    target=target,
                    target_residual=target_residual,
                )
                item = {
                    **_v5_metadata(record),
                    "mode": mode,
                    "baseline_access": V5_MODE_REGISTRY[mode]["access"],
                    "prompt": prompt,
                    "metrics": metrics,
                    "descriptives": descriptives,
                }
                question_dir = (
                    output_dir
                    / mode
                    / record.scene_id
                    / f"q{record.question_index}_{record.question_type}"
                )
                write_item_artifacts(
                    question_dir=question_dir,
                    item=item,
                    evidence=evidence,
                    residual=residual,
                    sample_rate=record.sample_rate,
                    render_audio=not args.no_render_audio,
                )
                items.append(item)

    summaries = _summaries_by_mode(items)
    report = {
        "format": "qces_v5_audiosep_baselines_v1",
        "schema_version": records[0].schema_version,
        "storage_modes": sorted({record.storage_mode for record in records}),
        "splits": sorted({record.split for record in records}),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(
            args.manifest.resolve().read_bytes()
        ).hexdigest(),
        "selected_record_count": len(records),
        "selection": {
            "max_records_in_manifest_order": args.max_records,
            "warning": (
                "A nonzero cap is a deterministic debug subset, not a balanced "
                "paper result; use a predeclared family-complete pilot manifest."
                if args.max_records
                else None
            ),
        },
        "audiosep_checkpoint": str(args.audiosep_checkpoint.resolve()),
        "audiosep_frozen": True,
        "rendered_audio": not args.no_render_audio,
        "device": str(device),
        "text_batch_size": args.text_batch_size,
        "separator_call_provenance": {
            "actual_calls": separator_calls,
            "maximum_without_same_prompt_reuse": len(records) * len(text_modes),
            "oracle_raw_prediction_reused_for_oracle_gate": True,
        },
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "mode_registry": V5_MODE_REGISTRY,
        "protocol_limitations": [
            (
                "raw_question_text__weak_non_oracle is intentionally weak: "
                "AudioSep expects an acoustic target description, while a QCES "
                "question specifies an implicit relational pair. It is not an "
                "implementation of question-conditioned evidence composition."
            ),
            (
                "oracle_semantic and oracle_semantic_oracle_gate consume ground-"
                "truth role labels; the gated mode also consumes ground-truth "
                "timestamps. They are upper bounds, not deployable baselines."
            ),
            (
                "These are waveform metrics only and do not establish downstream "
                "question-answering sufficiency or residual necessity."
            ),
            (
                "Residual is defined arithmetically as X-E, so mixture consistency "
                "is a sanity check rather than evidence faithfulness."
            ),
        ],
        "summaries": summaries,
        "summaries_by_relation": _grouped_v5_summaries(items, "relation"),
        "summaries_by_variant": _grouped_v5_summaries(items, "variant_id"),
        "items": items,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = QCESManifestDataset(args.manifest.resolve(), crop_samples=None)
    if all(
        isinstance(record, (QCESRecord, QCESV4Record))
        for record in dataset.records
    ):
        _run_legacy(args, device, output_dir, dataset)
    elif all(isinstance(record, QCESV5Record) for record in dataset.records):
        _run_v5(args, device, output_dir, dataset)
    else:
        raise SystemExit(
            "this baseline requires one pure QCES v3, v4, or v5 manifest"
        )


if __name__ == "__main__":
    main()
