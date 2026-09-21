#!/usr/bin/env python3
"""Evaluate frozen SAM-Audio text/span prompts on QCES v3 or QCES v5.

The QCES-v5 path keeps deployable text/predicted-span controls separate from
oracle semantic/span upper bounds, reports native SAM target *and* residual
stems, and preserves counterfactual-family metadata for paired statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torchaudio

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.qces.metrics import (
    scale_dependent_sdr,
    scale_invariant_sdr,
)


MODES = (
    "question",
    "oracle_semantic",
    "oracle_semantic_oracle_span",
    "oracle_span_only",
)
PREDICTED_SPAN_MODES = (
    "oracle_semantic_predicted_span",
    "predicted_span_only",
)
ALL_MODES = MODES + PREDICTED_SPAN_MODES
V5_MODE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "question": {
        "access": "non_oracle",
        "uses_oracle_semantics": False,
        "uses_oracle_time": False,
        "description": "Raw relational question as SAM-Audio text prompt.",
    },
    "oracle_semantic": {
        "access": "oracle_upper_bound",
        "uses_oracle_semantics": True,
        "uses_oracle_time": False,
        "description": "Annotated evidence-role labels as the text prompt.",
    },
    "oracle_semantic_oracle_span": {
        "access": "oracle_upper_bound",
        "uses_oracle_semantics": True,
        "uses_oracle_time": True,
        "description": "Annotated role labels and annotated positive spans.",
    },
    "oracle_span_only": {
        "access": "oracle_upper_bound",
        "uses_oracle_semantics": False,
        "uses_oracle_time": True,
        "description": "Annotated positive spans without a text description.",
    },
    "oracle_semantic_predicted_span": {
        "access": "mixed_oracle_semantic",
        "uses_oracle_semantics": True,
        "uses_oracle_time": False,
        "description": "Annotated role labels plus externally predicted spans.",
    },
    "predicted_span_only": {
        "access": "non_oracle_if_span_predictor_is_non_oracle",
        "uses_oracle_semantics": False,
        "uses_oracle_time": False,
        "description": "Externally predicted spans without a text description.",
    },
}
DESCRIPTION_OVERRIDES = {
    "buzz": "buzzing",
    "croak": "frog croaking",
    "engine knocking": "engine knocking",
    "jackhammer": "jackhammer",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-id")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ode-steps", type=int, default=16)
    parser.add_argument("--predicted-spans-report", type=Path)
    parser.add_argument("--modes", nargs="+", choices=ALL_MODES)
    parser.add_argument(
        "--no-render-audio",
        action="store_true",
        help="Keep numeric reports/metadata but skip predicted WAV files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_manifest(path: Path) -> List[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("expected a non-empty QCES manifest")
    schemas = {row.get("schema_version") for row in rows}
    supported_v5 = {"qces_v5", "qces_v5_scene_event_derived_v1"}
    if schemas != {"qces_v3"} and not schemas.issubset(supported_v5):
        raise ValueError(
            "expected one pure QCES v3 or QCES v5 manifest, got "
            f"schemas={sorted(schemas, key=repr)}"
        )
    return rows


def describe(label: str) -> str:
    normalized = label.strip().lower()
    return DESCRIPTION_OVERRIDES.get(normalized, normalized)


def evidence_events(record: dict) -> List[dict]:
    ids = set(record["evidence_event_ids"])
    return sorted(
        (event for event in record["events"] if event["event_id"] in ids),
        key=lambda event: event["onset_seconds"],
    )


def oracle_description(record: dict) -> str:
    if record["no_evidence"]:
        return describe(record["absent_label"])
    return " and ".join(describe(event["label"]) for event in evidence_events(record))


def prompt(record: dict, mode: str) -> str:
    if mode == "question":
        return record["question"].lower()
    if mode in {"oracle_span_only", "predicted_span_only"}:
        return ""
    return oracle_description(record)


def anchors(record: dict, mode: str, predicted_spans: Dict[str, list]):
    if mode in PREDICTED_SPAN_MODES:
        return [
            ["+", interval[0], interval[1]]
            for interval in predicted_spans[record["id"]]
        ]
    if mode not in {"oracle_semantic_oracle_span", "oracle_span_only"}:
        return None
    return [
        ["+", event["onset_seconds"], event["offset_seconds"]]
        for event in evidence_events(record)
    ]


def v5_oracle_description(record: QCESV5Record) -> str:
    """Describe the annotated role union without leaking event multiplicity."""

    if record.no_evidence:
        labels = list(record.absent_labels)
    else:
        events = sorted(
            (record.event_by_id(event_id) for event_id in record.evidence_event_ids),
            key=lambda event: (event.onset_seconds, event.event_id),
        )
        labels = [event.label for event in events]
    # SAM receives semantic classes, not the artificial phrase "bell and bell"
    # when anchor and answer are two occurrences of the same class.
    labels = list(dict.fromkeys(labels))
    if not labels:
        raise ValueError(f"{record.sample_id} has no semantic prompt labels")
    return " and ".join(describe(label) for label in labels)


def v5_prompt(record: QCESV5Record, mode: str) -> str:
    if mode == "question":
        return record.question.lower()
    if mode in {"oracle_span_only", "predicted_span_only"}:
        return ""
    return v5_oracle_description(record)


def v5_anchors(
    record: QCESV5Record,
    mode: str,
    predicted_spans: Mapping[str, Sequence[Sequence[float]]],
) -> List[List[float | str]] | None:
    if mode in PREDICTED_SPAN_MODES:
        intervals = predicted_spans[record.sample_id]
    elif mode in {"oracle_semantic_oracle_span", "oracle_span_only"}:
        intervals = sorted(
            (*record.anchor_intervals, *record.answer_intervals),
            key=lambda interval: (interval[0], interval[1]),
        )
    else:
        return None
    normalized = [
        ["+", float(interval[0]), float(interval[1])] for interval in intervals
    ]
    if any(
        float(interval[2]) > record.duration_seconds + 1e-6 for interval in normalized
    ):
        raise ValueError(f"span exceeds duration for {record.sample_id}")
    return normalized


def v5_metadata(record: QCESV5Record) -> Dict[str, Any]:
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


def validate_predicted_spans(
    report_path: Path,
    *,
    manifest: Path,
    record_ids: Sequence[str],
) -> tuple[Dict[str, list], Dict[str, Any]]:
    payload = json.loads(report_path.resolve().read_text(encoding="utf-8"))
    report_manifest_sha = payload.get("manifest_sha256")
    manifest_sha = hashlib.sha256(manifest.resolve().read_bytes()).hexdigest()
    if report_manifest_sha is not None and report_manifest_sha != manifest_sha:
        raise ValueError(
            "predicted-span report is bound to a different manifest: "
            f"expected {manifest_sha}, got {report_manifest_sha}"
        )
    report_manifest_path = payload.get("manifest")
    if report_manifest_sha is None and (
        not isinstance(report_manifest_path, str)
        or Path(report_manifest_path).resolve() != manifest.resolve()
    ):
        raise ValueError(
            "predicted-span report lacks a matching manifest SHA/path binding"
        )
    spans: Dict[str, list] = {}
    for item in payload.get("items", []):
        sample_id = item.get("id")
        if sample_id in spans:
            raise ValueError(f"duplicate predicted-span item ID: {sample_id}")
        intervals = item.get("predicted_evidence_intervals")
        if not isinstance(sample_id, str) or not isinstance(intervals, list):
            raise ValueError("invalid predicted-span report item")
        normalized = []
        for index, interval in enumerate(intervals):
            if (
                not isinstance(interval, list)
                or len(interval) != 2
                or not all(isinstance(value, (int, float)) for value in interval)
            ):
                raise ValueError(f"invalid interval for {sample_id} at index {index}")
            start, end = map(float, interval)
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or start < 0
                or end <= start
            ):
                raise ValueError(
                    f"invalid interval for {sample_id} at index {index}: {interval}"
                )
            normalized.append([start, end])
        spans[sample_id] = normalized
    missing = sorted(set(record_ids) - set(spans))
    if missing:
        raise ValueError(f"predicted-span report is missing IDs: {missing[:10]}")
    return spans, {
        "path": str(report_path.resolve()),
        "sha256": hashlib.sha256(report_path.resolve().read_bytes()).hexdigest(),
        "manifest": report_manifest_path,
        "manifest_sha256": report_manifest_sha,
    }


def resolve_audio(manifest: Path, relative: str) -> Path:
    root = manifest.resolve().parent
    result = (root / relative).resolve()
    if root not in result.parents:
        raise ValueError(f"audio path escapes dataset root: {relative}")
    return result


def load_mono(path: Path, sample_rate: int) -> torch.Tensor:
    waveform, source_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    return waveform


def si_sdr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    projection = (
        (prediction * target).sum() * target / target.square().sum().clamp_min(1e-8)
    )
    noise = prediction - projection
    return float(
        10
        * torch.log10(
            (
                projection.square().sum() / noise.square().sum().clamp_min(1e-8)
            ).clamp_min(1e-8)
        )
    )


def mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def resample_mono(
    waveform: torch.Tensor, source_rate: int, target_rate: int
) -> torch.Tensor:
    waveform = waveform.reshape(1, -1)
    if source_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, target_rate)
    return waveform


def sam_v5_item_metrics(
    *,
    no_evidence: bool,
    evidence: torch.Tensor,
    residual: torch.Tensor,
    mixture: torch.Tensor,
    target: torch.Tensor,
    target_residual: torch.Tensor,
) -> tuple[Dict[str, float | None], Dict[str, float]]:
    """Measure native SAM target/residual output with explicit directions."""

    metrics: Dict[str, float | None] = {
        "evidence_l1_↓": float((evidence - target).abs().mean()),
        "residual_l1_↓": float((residual - target_residual).abs().mean()),
        "native_mixture_consistency_l1_↓": float(
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
    retained_ratio = float(evidence.abs().sum() / mixture.abs().sum().clamp_min(1e-8))
    if no_evidence:
        metrics["no_evidence_retained_ratio_↓"] = retained_ratio
    else:
        evidence_si_sdr = float(scale_invariant_sdr(evidence[None], target[None])[0])
        mixture_si_sdr = float(scale_invariant_sdr(mixture[None], target[None])[0])
        evidence_sd_sdr = float(scale_dependent_sdr(evidence[None], target[None])[0])
        mixture_sd_sdr = float(scale_dependent_sdr(mixture[None], target[None])[0])
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
        raise ValueError("a QCES-v5 SAM metric is NaN or infinite")
    return metrics, {
        "evidence_retained_ratio": retained_ratio,
        "evidence_absolute_peak": float(evidence.abs().max()),
        "residual_absolute_peak": float(residual.abs().max()),
    }


def summarize_v5_items(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("cannot summarize an empty QCES-v5 SAM item set")

    def values(key: str) -> List[float]:
        return [
            float(item["metrics"][key])
            for item in items
            if item["metrics"][key] is not None
        ]

    def distribution(key: str) -> Dict[str, float | None]:
        collected = values(key)
        if not collected:
            return {"mean": None, "median": None, "minimum": None}
        return {
            "mean": mean(collected),
            "median": float(median(collected)),
            "minimum": min(collected),
        }

    si_sdr = distribution("evidence_si_sdr_db_↑")
    si_sdri = distribution("evidence_si_sdri_db_↑")
    sd_sdr = distribution("evidence_sd_sdr_db_↑")
    sd_sdri = distribution("evidence_sd_sdri_db_↑")
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
        "native_mixture_consistency_l1_mean_↓": mean(
            values("native_mixture_consistency_l1_↓")
        ),
        "native_mixture_consistency_l1_maximum_↓": max(
            values("native_mixture_consistency_l1_↓")
        ),
    }


def grouped_v5_summaries(
    items: Sequence[Mapping[str, Any]], modes: Sequence[str], field: str
) -> Dict[str, Any]:
    return {
        str(group): {
            mode: summarize_v5_items(
                [
                    item
                    for item in items
                    if str(item[field]) == str(group) and item["mode"] == mode
                ]
            )
            for mode in modes
        }
        for group in sorted({str(item[field]) for item in items})
    }


def write_item_artifacts(
    *,
    question_dir: Path,
    item: Mapping[str, Any],
    evidence: torch.Tensor,
    residual: torch.Tensor,
    sample_rate: int,
    render_audio: bool,
) -> None:
    question_dir.mkdir(parents=True, exist_ok=True)
    if render_audio:
        torchaudio.save(
            question_dir / "predicted_evidence.wav", evidence[None], sample_rate
        )
        torchaudio.save(
            question_dir / "predicted_residual.wav", residual[None], sample_rate
        )
    (question_dir / "metadata.json").write_text(
        json.dumps(dict(item), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def hash_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def load_sam_audio(model_path: Path, device: torch.device):
    repository_root = CODE_ROOT / "baseline" / "sam-audio"
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from sam_audio import SAMAudio, SAMAudioProcessor

    processor = SAMAudioProcessor.from_pretrained(str(model_path))
    # The visual encoder, rankers, and built-in span predictor are not used by
    # these declared text/span controls. Avoid loading unrelated models.
    model = (
        SAMAudio.from_pretrained(
            str(model_path),
            vision_encoder={"dim": 1024, "batch_size": 0, "name": "none"},
            visual_ranker=None,
            text_ranker=None,
            span_predictor=None,
        )
        .eval()
        .to(device)
    )
    return processor, model


def separate_once(
    *,
    processor,
    model,
    mixture: torch.Tensor,
    description: str,
    span_prompt: Sequence[Sequence[float | str]] | None,
    device: torch.device,
    seed: int,
    ode_opt: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = processor(
        audios=[mixture],
        descriptions=[description],
        anchors=[span_prompt] if span_prompt is not None else None,
    ).to(device)
    torch.manual_seed(seed)
    result = model.separate(
        batch,
        predict_spans=False,
        reranking_candidates=1,
        ode_opt=ode_opt,
    )
    return (
        result.target[0].reshape(-1).cpu(),
        result.residual[0].reshape(-1).cpu(),
    )


def run_v5(
    *,
    args: argparse.Namespace,
    manifest: Path,
    output_dir: Path,
    device: torch.device,
    model_path: Path,
    modes: Sequence[str],
    predicted_spans: Mapping[str, Sequence[Sequence[float]]],
    predicted_span_provenance: Mapping[str, Any] | None,
) -> None:
    dataset = QCESManifestDataset(manifest, crop_samples=None)
    selected = [
        (index, record)
        for index, record in enumerate(dataset.records)
        if isinstance(record, QCESV5Record)
    ]
    if len(selected) != len(dataset.records):
        raise ValueError("expected a pure QCES-v5 manifest")
    if args.sample_id:
        selected = [pair for pair in selected if pair[1].sample_id == args.sample_id]
        if not selected:
            raise SystemExit(f"sample ID not found: {args.sample_id}")
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("the selected QCES-v5 record set is empty")

    processor, model = load_sam_audio(model_path, device)
    sample_rate = int(processor.audio_sampling_rate)
    ode_opt = {
        "method": "midpoint",
        "options": {"step_size": 1 / args.ode_steps},
    }
    items: List[Dict[str, Any]] = []
    separator_calls = 0
    empty_span_abstentions = 0
    with torch.inference_mode():
        for dataset_index, record in selected:
            example = dataset[dataset_index]
            mixture = resample_mono(example.mixture, record.sample_rate, sample_rate)
            target = resample_mono(example.evidence, record.sample_rate, sample_rate)[0]
            target_residual = resample_mono(
                example.residual, record.sample_rate, sample_rate
            )[0]
            for mode in modes:
                description = v5_prompt(record, mode)
                span_prompt = v5_anchors(record, mode, predicted_spans)
                # An explicitly empty oracle/predicted positive-span set is an
                # abstention decision. Calling SAM with neither text nor span is
                # undefined, and calling it with an absent oracle label would
                # defeat the declared no-evidence gate.
                empty_span_abstention = span_prompt == [] and mode != "question"
                if empty_span_abstention:
                    evidence = torch.zeros_like(mixture[0])
                    residual = mixture[0].clone()
                    empty_span_abstentions += 1
                else:
                    evidence, residual = separate_once(
                        processor=processor,
                        model=model,
                        mixture=mixture,
                        description=description,
                        span_prompt=span_prompt,
                        device=device,
                        seed=sample_seed(args.seed, record.sample_id),
                        ode_opt=ode_opt,
                    )
                    separator_calls += 1
                samples = min(
                    evidence.numel(),
                    residual.numel(),
                    mixture.size(-1),
                    target.numel(),
                    target_residual.numel(),
                )
                evidence = evidence[:samples]
                residual = residual[:samples]
                original = mixture[0, :samples]
                target_clip = target[:samples]
                target_residual_clip = target_residual[:samples]
                metrics, descriptives = sam_v5_item_metrics(
                    no_evidence=record.no_evidence,
                    evidence=evidence,
                    residual=residual,
                    mixture=original,
                    target=target_clip,
                    target_residual=target_residual_clip,
                )
                item = {
                    **v5_metadata(record),
                    "mode": mode,
                    "baseline_access": V5_MODE_REGISTRY[mode]["access"],
                    "description": description,
                    "anchors": span_prompt,
                    "empty_span_abstention": empty_span_abstention,
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
                    sample_rate=sample_rate,
                    render_audio=not args.no_render_audio,
                )
                items.append(item)

    summaries = {
        mode: summarize_v5_items([item for item in items if item["mode"] == mode])
        for mode in modes
    }
    checkpoint_path = model_path / "checkpoint.pt"
    config_path = model_path / "config.json"
    if not checkpoint_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "SAM model directory must contain checkpoint.pt and config.json"
        )
    report = {
        "format": "qces_v5_sam_audio_baselines_v2",
        "schema_versions": sorted({record.schema_version for _, record in selected}),
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "selected_record_count": len(selected),
        "selection": {
            "sample_id": args.sample_id,
            "limit_in_manifest_order": args.limit,
            "warning": (
                "A nonzero limit is a debug subset, not a balanced paper result."
                if args.limit
                else None
            ),
        },
        "model": str(model_path),
        "model_checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": hash_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        },
        "model_config": {
            "path": str(config_path.resolve()),
            "sha256": hash_file(config_path),
        },
        "sam_audio_frozen": True,
        "sample_rate": sample_rate,
        "ode_steps": args.ode_steps,
        "base_seed": args.seed,
        "seed_strategy": "same base_plus_sha256_sample_id noise across modes",
        "reranking_candidates": 1,
        "device": str(device),
        "rendered_audio": not args.no_render_audio,
        "separator_calls": separator_calls,
        "empty_span_abstentions": empty_span_abstentions,
        "predicted_span_provenance": predicted_span_provenance,
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
        "mode_registry": {mode: V5_MODE_REGISTRY[mode] for mode in modes},
        "native_residual": True,
        "protocol_limitations": [
            (
                "question is an intentionally weak raw-question control because "
                "relational questions do not necessarily name the answer source."
            ),
            (
                "oracle semantic/span modes consume annotations and are upper "
                "bounds, not deployable systems."
            ),
            (
                "Waveform metrics do not establish downstream QA sufficiency or "
                "residual necessity; score the rendered stems with held-out auditors."
            ),
            (
                "Unlike QCES arithmetic R=X-E, this report evaluates SAM-Audio's "
                "native residual and reports its native mixture-consistency error."
            ),
        ],
        "summaries": summaries,
        "summaries_by_relation": grouped_v5_summaries(items, modes, "relation"),
        "summaries_by_variant": grouped_v5_summaries(items, modes, "variant_id"),
        "items": items,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


def run_legacy_v3(
    *,
    args: argparse.Namespace,
    manifest: Path,
    output_dir: Path,
    device: torch.device,
    model_path: Path,
    records: Sequence[dict],
    modes: Sequence[str],
    predicted_spans: Mapping[str, Sequence[Sequence[float]]],
) -> None:
    processor, model = load_sam_audio(model_path, device)
    sample_rate = int(processor.audio_sampling_rate)
    ode_opt = {
        "method": "midpoint",
        "options": {"step_size": 1 / args.ode_steps},
    }
    items: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for record in records:
            mixture = load_mono(
                resolve_audio(manifest, record["mixture_path"]), sample_rate
            )
            target = load_mono(
                resolve_audio(manifest, record["evidence_stem_path"]), sample_rate
            )[0]
            target_residual = load_mono(
                resolve_audio(manifest, record["residual_stem_path"]), sample_rate
            )[0]
            for mode in modes:
                description = prompt(record, mode)
                span_prompt = anchors(record, mode, dict(predicted_spans))
                evidence, residual = separate_once(
                    processor=processor,
                    model=model,
                    mixture=mixture,
                    description=description,
                    span_prompt=span_prompt,
                    device=device,
                    seed=sample_seed(args.seed, record["id"]),
                    ode_opt=ode_opt,
                )
                samples = min(
                    evidence.numel(),
                    residual.numel(),
                    mixture.size(-1),
                    target.numel(),
                    target_residual.numel(),
                )
                evidence = evidence[:samples]
                residual = residual[:samples]
                original = mixture[0, :samples]
                target_clip = target[:samples]
                residual_target_clip = target_residual[:samples]
                item: Dict[str, Any] = {
                    "id": record["id"],
                    "scene_id": record["scene_id"],
                    "question_index": record["question_index"],
                    "question": record["question"],
                    "no_evidence": record["no_evidence"],
                    "mode": mode,
                    "description": description,
                    "anchors": span_prompt,
                    "evidence_l1": float((evidence - target_clip).abs().mean()),
                    "residual_l1": float(
                        (residual - residual_target_clip).abs().mean()
                    ),
                    "retained_ratio": float(
                        evidence.abs().sum() / original.abs().sum().clamp_min(1e-8)
                    ),
                    "mixture_consistency_l1": float(
                        (original - evidence - residual).abs().mean()
                    ),
                    "evidence_si_sdr": None,
                    "mixture_si_sdr": None,
                    "evidence_si_sdri": None,
                }
                if not record["no_evidence"]:
                    item["evidence_si_sdr"] = si_sdr(evidence, target_clip)
                    item["mixture_si_sdr"] = si_sdr(original, target_clip)
                    item["evidence_si_sdri"] = (
                        item["evidence_si_sdr"] - item["mixture_si_sdr"]
                    )
                write_item_artifacts(
                    question_dir=(
                        output_dir
                        / mode
                        / record["scene_id"]
                        / f"q{record['question_index']}_{record['question_type']}"
                    ),
                    item=item,
                    evidence=evidence,
                    residual=residual,
                    sample_rate=sample_rate,
                    render_audio=not args.no_render_audio,
                )
                items.append(item)
    summaries = {}
    for mode in modes:
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
            "evidence_si_sdr_answerable": mean(answerable),
            "evidence_si_sdr_median": (
                float(median(answerable)) if answerable else 0.0
            ),
            "evidence_si_sdr_minimum": min(answerable) if answerable else 0.0,
            "evidence_si_sdri_answerable": mean(improvements),
            "evidence_l1": mean([item["evidence_l1"] for item in subset]),
            "residual_l1": mean([item["residual_l1"] for item in subset]),
            "mean_no_evidence_retained_ratio": mean(
                [item["retained_ratio"] for item in negatives]
            ),
            "mean_mixture_consistency_l1": mean(
                [item["mixture_consistency_l1"] for item in subset]
            ),
        }
    report = {
        "format": "qces_sam_audio_baselines_v1",
        "manifest": str(manifest),
        "model": str(model_path),
        "sample_rate": sample_rate,
        "ode_steps": args.ode_steps,
        "base_seed": args.seed,
        "seed_strategy": "base_plus_sha256_sample_id",
        "reranking_candidates": 1,
        "device": str(device),
        "rendered_audio": not args.no_render_audio,
        "summaries": summaries,
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
    if args.limit < 0 or args.ode_steps <= 0:
        raise SystemExit("limit must be non-negative and ode-steps must be positive")
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

    manifest = args.manifest.resolve()
    raw_records = read_manifest(manifest)
    is_v5 = raw_records[0]["schema_version"] != "qces_v3"
    raw_selected = raw_records
    if args.sample_id:
        raw_selected = [
            record for record in raw_selected if record["id"] == args.sample_id
        ]
        if not raw_selected:
            raise SystemExit(f"sample ID not found: {args.sample_id}")
    if args.limit:
        raw_selected = raw_selected[: args.limit]
    modes = list(args.modes) if args.modes else list(MODES)
    if args.predicted_spans_report and args.modes is None:
        modes.extend(PREDICTED_SPAN_MODES)
    predicted_spans: Dict[str, list] = {}
    predicted_span_provenance = None
    if args.predicted_spans_report:
        predicted_spans, predicted_span_provenance = validate_predicted_spans(
            args.predicted_spans_report,
            manifest=manifest,
            record_ids=[record["id"] for record in raw_selected],
        )
    elif any(mode in PREDICTED_SPAN_MODES for mode in modes):
        raise SystemExit("predicted-span modes require --predicted-spans-report")

    model_path = args.model.resolve()
    if is_v5:
        run_v5(
            args=args,
            manifest=manifest,
            output_dir=output_dir,
            device=device,
            model_path=model_path,
            modes=modes,
            predicted_spans=predicted_spans,
            predicted_span_provenance=predicted_span_provenance,
        )
    else:
        run_legacy_v3(
            args=args,
            manifest=manifest,
            output_dir=output_dir,
            device=device,
            model_path=model_path,
            records=raw_selected,
            modes=modes,
            predicted_spans=predicted_spans,
        )


if __name__ == "__main__":
    main()
