#!/usr/bin/env python3
"""Diagnose one union prompt versus role-factorized AudioSep oracle prompts.

This is an offline, answerable-only upper-bound diagnostic.  It does not use a
learned QCES composer and it does not modify AudioSep.  Its matched-temporal
factorial separates three paths:

* A: one timeline-ordered union text prompt under the oracle union window;
* B: separate anchor/answer prompts, both under that same union window;
* C: the same separate raw predictions under their role-specific windows.

The ungated pair is included as a secondary separator-only diagnostic.  When
the two roles have the same semantic prompt, the role-factorized path reuses a
single AudioSep prediction and never doubles it in overlapping windows.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import soundfile as sf
import torch

from mixi_understanding.data.qces_v4_schema import QCESV4Record
from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.qces.metrics import (
    scale_dependent_sdr,
    scale_invariant_sdr,
)
from mixi_understanding.qces.separators import AudioSepConditionedAdapter
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    describe,
    encode_prompts,
)


SUPPORTED_RECORD_TYPES = (QCESV4Record, QCESV5Record)

UNION_RAW = "union_text__ungated"
DUAL_RAW = "dual_role_text__ungated_linear_add"
UNION_GATED = "union_text__oracle_union_window"
DUAL_UNION_GATED = "dual_role_text__oracle_union_window_linear_add"
DUAL_GATED = "dual_role_text__oracle_role_windows_linear_add"
MODES = (UNION_RAW, DUAL_RAW, UNION_GATED, DUAL_UNION_GATED, DUAL_GATED)

PRIMARY_PAIR = (UNION_GATED, DUAL_GATED)
SEMANTIC_FACTOR_PAIR = (UNION_GATED, DUAL_UNION_GATED)
ROLE_WINDOW_PAIR = (DUAL_UNION_GATED, DUAL_GATED)
SECONDARY_PAIR = (UNION_RAW, DUAL_RAW)

METRIC_DIRECTIONS = {
    "evidence_si_sdr_db": "↑",
    "evidence_sd_sdr_db": "↑",
    "evidence_si_sdri_db": "↑",
    "evidence_sd_sdri_db": "↑",
    "evidence_l1": "↓",
    "residual_l1": "↓",
    "oracle_windowed_anchor_si_sdr_db": "↑",
    "oracle_windowed_answer_si_sdr_db": "↑",
    "oracle_windowed_weakest_role_si_sdr_db": "↑",
    "oracle_windowed_anchor_sd_sdr_db": "↑",
    "oracle_windowed_answer_sd_sdr_db": "↑",
    "oracle_windowed_weakest_role_sd_sdr_db": "↑",
    "mixture_consistency_l1_sanity": "↓",
}

# A question can repeat the same target event pair under a reversed surface or
# another relation.  These role-symmetric metrics are invariant to swapping
# anchor and answer and are therefore safe for one-unit-per-waveform summaries.
ACOUSTIC_UNIT_METRIC_DIRECTIONS = {
    metric: direction
    for metric, direction in METRIC_DIRECTIONS.items()
    if not (
        metric.startswith("oracle_windowed_anchor_")
        or metric.startswith("oracle_windowed_answer_")
    )
}


@dataclass(frozen=True)
class OraclePrompts:
    union: str
    anchor: str
    answer: str
    anchor_label: str
    answer_label: str

    @property
    def same_role_prompt(self) -> bool:
        return self.anchor == self.answer


@dataclass(frozen=True)
class RenderedMode:
    evidence: torch.Tensor
    oracle_windowed_anchor: torch.Tensor
    oracle_windowed_answer: torch.Tensor


def parse_args() -> argparse.Namespace:
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
        help="Answerable-record cap for a debug run; 0 evaluates all records.",
    )
    parser.add_argument(
        "--listening-count",
        type=int,
        default=0,
        help="Render this many deterministic hard-case listening folders.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def oracle_prompts(record: QCESV4Record | QCESV5Record) -> OraclePrompts:
    """Return the current timeline-union prompt and two role prompts."""

    if record.no_evidence:
        raise ValueError("dual-role oracle prompts require an answerable record")
    if len(record.anchor_event_ids) != 1 or len(record.answer_event_ids) != 1:
        raise ValueError("each answerable record must expose one event per role")
    anchor = record.event_by_id(record.anchor_event_ids[0])
    answer = record.event_by_id(record.answer_event_ids[0])
    evidence_events = sorted(
        (record.event_by_id(event_id) for event_id in record.evidence_event_ids),
        key=lambda event: (event.onset_seconds, event.event_id),
    )
    return OraclePrompts(
        union=" and ".join(describe(event.label) for event in evidence_events),
        anchor=describe(anchor.label),
        answer=describe(answer.label),
        anchor_label=anchor.label,
        answer_label=answer.label,
    )


def paint_intervals(
    intervals: Iterable[tuple[float, float]],
    sample_rate: int,
    samples: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    gate = torch.zeros(samples, dtype=torch.float32, device=device)
    for onset, offset in intervals:
        start = max(0, min(round(onset * sample_rate), samples))
        end = max(start, min(round(offset * sample_rate), samples))
        gate[start:end] = 1.0
    return gate


def render_unique_prompts(
    separator: torch.nn.Module,
    mixture: torch.Tensor,
    conditions: Mapping[str, torch.Tensor],
    prompts: Sequence[str],
) -> tuple[Dict[str, torch.Tensor], int]:
    """Render each distinct prompt once for one mixture."""

    if mixture.ndim != 1:
        raise ValueError("mixture must have shape [samples]")
    rendered: Dict[str, torch.Tensor] = {}
    calls = 0
    for prompt in dict.fromkeys(prompts):
        condition = conditions[prompt]
        if condition.ndim != 1:
            condition = condition.reshape(-1)
        result = separator(
            {
                "mixture": mixture[None, None],
                "condition": condition[None].to(mixture.device),
            }
        )
        waveform: Any = result["waveform"] if isinstance(result, dict) else result
        if waveform.ndim == 3 and waveform.size(1) == 1:
            waveform = waveform[:, 0]
        if waveform.shape != mixture[None].shape:
            raise ValueError(
                f"AudioSep returned {tuple(waveform.shape)}, expected "
                f"{tuple(mixture[None].shape)}"
            )
        rendered[prompt] = waveform[0].detach()
        calls += 1
    return rendered, calls


def compose_modes(
    union_raw: torch.Tensor,
    anchor_raw: torch.Tensor,
    answer_raw: torch.Tensor,
    anchor_gate: torch.Tensor,
    answer_gate: torch.Tensor,
    *,
    same_role_prompt: bool,
) -> Dict[str, RenderedMode]:
    """Build paired evidence stems without amplitude clipping.

    Linear addition is the correct fusion if the two role predictions are
    ideal source estimates.  Sample-wise clipping against the mixture would be
    invalid under phase cancellation.  For identical semantic prompts, one raw
    estimate may already contain both same-class instances; it is therefore
    reused and union-gated instead of added to itself.
    """

    shapes = {
        tuple(tensor.shape)
        for tensor in (union_raw, anchor_raw, answer_raw, anchor_gate, answer_gate)
    }
    if len(shapes) != 1:
        raise ValueError(f"all rendering tensors must share shape, got {shapes}")
    union_gate = torch.maximum(anchor_gate, answer_gate)
    if same_role_prompt:
        dual_raw = anchor_raw
        dual_union_gated = anchor_raw * union_gate
        dual_gated = anchor_raw * union_gate
    else:
        dual_raw = anchor_raw + answer_raw
        # Factorial B: both semantic estimates receive exactly the same binary
        # union window.  A -> B therefore changes semantic factorization while
        # holding the temporal information fixed.
        dual_union_gated = dual_raw * union_gate
        dual_gated = anchor_raw * anchor_gate + answer_raw * answer_gate

    union_roles = (
        union_raw * anchor_gate,
        union_raw * answer_gate,
    )
    dual_roles = (
        anchor_raw * anchor_gate,
        answer_raw * answer_gate,
    )
    return {
        UNION_RAW: RenderedMode(union_raw, *union_roles),
        DUAL_RAW: RenderedMode(dual_raw, *dual_roles),
        UNION_GATED: RenderedMode(union_raw * union_gate, *union_roles),
        DUAL_UNION_GATED: RenderedMode(dual_union_gated, *dual_roles),
        DUAL_GATED: RenderedMode(dual_gated, *dual_roles),
    }


def _metric_value(
    function: Any, prediction: torch.Tensor, target: torch.Tensor
) -> float:
    return float(function(prediction[None], target[None])[0])


def evaluate_mode(
    rendered: RenderedMode,
    mixture: torch.Tensor,
    target_evidence: torch.Tensor,
    target_residual: torch.Tensor,
    target_anchor: torch.Tensor,
    target_answer: torch.Tensor,
) -> Dict[str, float]:
    evidence = rendered.evidence
    residual = mixture - evidence
    evidence_si_sdr = _metric_value(
        scale_invariant_sdr, evidence, target_evidence
    )
    evidence_sd_sdr = _metric_value(
        scale_dependent_sdr, evidence, target_evidence
    )
    mixture_si_sdr = _metric_value(
        scale_invariant_sdr, mixture, target_evidence
    )
    mixture_sd_sdr = _metric_value(
        scale_dependent_sdr, mixture, target_evidence
    )
    anchor_si_sdr = _metric_value(
        scale_invariant_sdr, rendered.oracle_windowed_anchor, target_anchor
    )
    answer_si_sdr = _metric_value(
        scale_invariant_sdr, rendered.oracle_windowed_answer, target_answer
    )
    anchor_sd_sdr = _metric_value(
        scale_dependent_sdr, rendered.oracle_windowed_anchor, target_anchor
    )
    answer_sd_sdr = _metric_value(
        scale_dependent_sdr, rendered.oracle_windowed_answer, target_answer
    )
    values = {
        "evidence_si_sdr_db_↑": evidence_si_sdr,
        "evidence_sd_sdr_db_↑": evidence_sd_sdr,
        "evidence_si_sdri_db_↑": evidence_si_sdr - mixture_si_sdr,
        "evidence_sd_sdri_db_↑": evidence_sd_sdr - mixture_sd_sdr,
        "evidence_l1_↓": float((evidence - target_evidence).abs().mean()),
        "residual_l1_↓": float((residual - target_residual).abs().mean()),
        "oracle_windowed_anchor_si_sdr_db_↑": anchor_si_sdr,
        "oracle_windowed_answer_si_sdr_db_↑": answer_si_sdr,
        "oracle_windowed_weakest_role_si_sdr_db_↑": min(
            anchor_si_sdr, answer_si_sdr
        ),
        "oracle_windowed_anchor_sd_sdr_db_↑": anchor_sd_sdr,
        "oracle_windowed_answer_sd_sdr_db_↑": answer_sd_sdr,
        "oracle_windowed_weakest_role_sd_sdr_db_↑": min(
            anchor_sd_sdr, answer_sd_sdr
        ),
        "mixture_consistency_l1_sanity_↓": float(
            (mixture - evidence - residual).abs().mean()
        ),
        "evidence_absolute_peak_descriptive": float(evidence.abs().max()),
        "evidence_retained_ratio_descriptive": float(
            evidence.abs().sum() / mixture.abs().sum().clamp_min(1e-8)
        ),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("a waveform metric is non-finite")
    return values


def linear_percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_mode(
    items: Sequence[Mapping[str, Any]],
    mode: str,
    metric_directions: Mapping[str, str] = METRIC_DIRECTIONS,
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for metric, direction in metric_directions.items():
        item_key = f"{metric}_{direction}"
        values = [float(item["metrics"][mode][item_key]) for item in items]
        summary[f"{metric}_mean_{direction}"] = sum(values) / len(values)
        summary[f"{metric}_median_{direction}"] = float(median(values))
        if direction == "↑":
            summary[f"{metric}_p10_{direction}"] = linear_percentile(values, 0.1)
            summary[f"{metric}_minimum_{direction}"] = min(values)
        else:
            summary[f"{metric}_p90_{direction}"] = linear_percentile(values, 0.9)
            summary[f"{metric}_maximum_{direction}"] = max(values)
    return summary


def paired_mode_advantage(
    items: Sequence[Mapping[str, Any]],
    reference_mode: str,
    candidate_mode: str,
    *,
    label: str,
    metric_directions: Mapping[str, str] = METRIC_DIRECTIONS,
) -> Dict[str, float]:
    """Return paired values signed so positive favors ``candidate_mode``."""

    summary: Dict[str, float] = {}
    for metric, direction in metric_directions.items():
        if metric == "mixture_consistency_l1_sanity":
            continue
        key = f"{metric}_{direction}"
        advantages = []
        for item in items:
            reference = float(item["metrics"][reference_mode][key])
            candidate = float(item["metrics"][candidate_mode][key])
            advantages.append(
                candidate - reference if direction == "↑" else reference - candidate
            )
        summary[f"{label}_advantage_{metric}_mean_↑"] = sum(advantages) / len(
            advantages
        )
        summary[f"{label}_win_rate_by_{metric}_↑"] = sum(
            value > 0.0 for value in advantages
        ) / len(advantages)
    return summary


def paired_dual_advantage(
    items: Sequence[Mapping[str, Any]],
    union_mode: str,
    dual_mode: str,
    metric_directions: Mapping[str, str] = METRIC_DIRECTIONS,
) -> Dict[str, float]:
    """Backward-compatible dual-over-union paired summary."""

    return paired_mode_advantage(
        items,
        union_mode,
        dual_mode,
        label="dual",
        metric_directions=metric_directions,
    )


def _role_overlap(record: QCESV4Record | QCESV5Record) -> bool:
    return any(
        max(anchor[0], answer[0]) < min(anchor[1], answer[1])
        for anchor in record.anchor_intervals
        for answer in record.answer_intervals
    )


def acoustic_comparison_key(item: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    """Return the waveform-level unit shared by relation/surface duplicates."""

    event_ids = item.get("evidence_event_ids")
    if not isinstance(event_ids, (list, tuple)) or not event_ids:
        raise ValueError("acoustic aggregation requires evidence_event_ids")
    return str(item["scene_id"]), tuple(sorted(str(value) for value in event_ids))


def unique_acoustic_items(
    items: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep one deterministic representative per acoustic comparison.

    Only role-symmetric metrics enter the acoustic-unit summary.  We verify
    those metrics are identical before removing duplicated question rows.
    """

    groups: Dict[tuple[str, tuple[str, ...]], list[Mapping[str, Any]]] = {}
    for item in items:
        groups.setdefault(acoustic_comparison_key(item), []).append(item)
    representatives: list[Mapping[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: str(item["id"]))
        representative = group[0]
        for other in group[1:]:
            for mode in MODES:
                for metric, direction in ACOUSTIC_UNIT_METRIC_DIRECTIONS.items():
                    metric_key = f"{metric}_{direction}"
                    left = float(representative["metrics"][mode][metric_key])
                    right = float(other["metrics"][mode][metric_key])
                    if abs(left - right) > 1e-7:
                        raise ValueError(
                            "acoustic duplicate disagrees on "
                            f"{key}/{mode}/{metric_key}: {left} versus {right}"
                        )
        representatives.append(representative)
    return representatives


def _one_group(
    subset: Sequence[Mapping[str, Any]],
    *,
    metric_directions: Mapping[str, str],
    question_row_count: int | None = None,
) -> Dict[str, Any]:
    result = {
        "count": len(subset),
        "modes": {
            mode: summarize_mode(subset, mode, metric_directions) for mode in MODES
        },
        "factorial_matched_temporal": {
            "A_mode": UNION_GATED,
            "B_mode": DUAL_UNION_GATED,
            "C_mode": DUAL_GATED,
            "A_to_B_role_factorized_semantic_extraction_advantage": paired_mode_advantage(
                subset,
                *SEMANTIC_FACTOR_PAIR,
                label="B_over_A",
                metric_directions=metric_directions,
            ),
            "B_to_C_role_window_alignment_advantage": paired_mode_advantage(
                subset,
                *ROLE_WINDOW_PAIR,
                label="C_over_B",
                metric_directions=metric_directions,
            ),
            "A_to_C_total_advantage": paired_mode_advantage(
                subset,
                *PRIMARY_PAIR,
                label="C_over_A",
                metric_directions=metric_directions,
            ),
        },
        # Preserve the original keys for downstream readers of v1 reports.
        "primary_oracle_windowed_dual_advantage": paired_dual_advantage(
            subset, *PRIMARY_PAIR, metric_directions
        ),
        "secondary_ungated_dual_advantage": paired_dual_advantage(
            subset, *SECONDARY_PAIR, metric_directions
        ),
    }
    if question_row_count is not None:
        result["question_row_count"] = question_row_count
    return result


def summarize_items(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise ValueError("cannot summarize an empty evaluation")
    result: Dict[str, Any] = {
        "overall": _one_group(items, metric_directions=METRIC_DIRECTIONS)
    }
    for field, output_name in (
        ("relation", "by_relation"),
        ("role_windows_overlap", "by_role_windows_overlap"),
        ("same_role_label", "by_same_role_label"),
    ):
        values = sorted({str(item[field]) for item in items})
        result[output_name] = {
            value: _one_group(
                [item for item in items if str(item[field]) == value],
                metric_directions=METRIC_DIRECTIONS,
            )
            for value in values
        }
    return result


def summarize_unique_acoustic_items(
    items: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize each waveform target once, including within each stratum."""

    if not items:
        raise ValueError("cannot summarize an empty evaluation")

    def one_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        units = unique_acoustic_items(rows)
        return _one_group(
            units,
            metric_directions=ACOUSTIC_UNIT_METRIC_DIRECTIONS,
            question_row_count=len(rows),
        )

    result: Dict[str, Any] = {
        "aggregation_unit": "(scene_id, unordered evidence_event_ids)",
        "metric_scope": (
            "Role-symmetric metrics only; directional anchor/answer metrics are "
            "omitted because question roles can swap for one acoustic unit."
        ),
        "overall": one_group(items),
    }
    for field, output_name in (
        ("relation", "by_relation"),
        ("role_windows_overlap", "by_role_windows_overlap"),
        ("same_role_label", "by_same_role_label"),
    ):
        values = sorted({str(item[field]) for item in items})
        result[output_name] = {
            value: one_group(
                [item for item in items if str(item[field]) == value]
            )
            for value in values
        }
    return result


def separator_call_provenance(
    items: Sequence[Mapping[str, Any]], physical_cached_calls: int
) -> Dict[str, Any]:
    """Separate deploy-time effective calls from evaluator physical forwards."""

    if physical_cached_calls < 0:
        raise ValueError("physical_cached_calls must be non-negative")
    effective_a = len(items)
    effective_dual = sum(
        len({str(item["anchor_prompt"]), str(item["answer_prompt"])})
        for item in items
    )
    physical_uncached_with_bc_reuse = sum(
        len(
            {
                str(item["union_prompt"]),
                str(item["anchor_prompt"]),
                str(item["answer_prompt"]),
            }
        )
        for item in items
    )
    if physical_cached_calls > physical_uncached_with_bc_reuse:
        raise ValueError(
            "cached physical calls cannot exceed uncached per-record calls"
        )
    return {
        "effective_calls_if_mode_deployed_independently_no_inter_record_cache": {
            "A_union_semantic_union_window": effective_a,
            "B_dual_semantic_union_window": effective_dual,
            "C_dual_semantic_role_windows": effective_dual,
        },
        "physical_calls_for_joint_factorial_evaluator": {
            "without_inter_record_cache_B_and_C_share_raw_stems": (
                physical_uncached_with_bc_reuse
            ),
            "with_consecutive_scene_prompt_cache_B_and_C_share_raw_stems": (
                physical_cached_calls
            ),
        },
        "interpretation": (
            "Effective mode counts describe independent deployment cost and must "
            "not be summed to recover physical evaluator calls. The joint evaluator "
            "renders each distinct union/anchor/answer prompt once per active scene; "
            "B and C reuse the same raw role stems and differ only in gating."
        ),
    }


def listening_priority(
    record: QCESV4Record | QCESV5Record,
) -> tuple[int, int, str, str]:
    prompts = oracle_prompts(record)
    return (
        -int(prompts.anchor_label == prompts.answer_label),
        -int(_role_overlap(record)),
        record.relation,
        record.sample_id,
    )


def write_listening_case(
    directory: Path,
    record: QCESV4Record | QCESV5Record,
    prompts: OraclePrompts,
    mixture: torch.Tensor,
    target_evidence: torch.Tensor,
    target_anchor: torch.Tensor,
    target_answer: torch.Tensor,
    rendered: Mapping[str, RenderedMode],
    metrics: Mapping[str, Mapping[str, float]],
) -> None:
    case = directory / record.sample_id
    case.mkdir(parents=True, exist_ok=False)

    def write(name: str, waveform: torch.Tensor) -> None:
        sf.write(
            case / name,
            waveform.detach().cpu().numpy(),
            record.sample_rate,
            subtype="FLOAT",
        )

    write("00_mixture.wav", mixture)
    write("01_target_evidence.wav", target_evidence)
    write("02_target_anchor.wav", target_anchor)
    write("03_target_answer.wav", target_answer)
    for index, mode in enumerate(MODES, start=10):
        write(f"{index}_{mode}.wav", rendered[mode].evidence)
    metadata = {
        "id": record.sample_id,
        "question": record.question,
        "answer": record.answer,
        "relation": record.relation,
        "anchor_label": prompts.anchor_label,
        "answer_label": prompts.answer_label,
        "union_prompt": prompts.union,
        "anchor_prompt": prompts.anchor,
        "answer_prompt": prompts.answer,
        "same_role_label": prompts.anchor_label == prompts.answer_label,
        "role_windows_overlap": _role_overlap(record),
        "factorial_mode_legend": {
            "A_union_semantic_union_window": UNION_GATED,
            "B_dual_semantic_union_window": DUAL_UNION_GATED,
            "C_dual_semantic_role_windows": DUAL_GATED,
        },
        "metrics": metrics,
    }
    (case / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.max_records < 0 or args.listening_count < 0:
        raise SystemExit("--max-records and --listening-count must be non-negative")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False

    manifest = args.manifest.resolve()
    dataset = QCESManifestDataset(manifest, crop_samples=None)
    if not all(
        isinstance(record, SUPPORTED_RECORD_TYPES) for record in dataset.records
    ):
        raise SystemExit("this diagnostic requires a pure QCES v4 or v5 manifest")
    all_answerable_indices = [
        index
        for index, record in enumerate(dataset.records)
        if isinstance(record, SUPPORTED_RECORD_TYPES) and not record.no_evidence
    ]
    selected_indices = all_answerable_indices[
        : args.max_records if args.max_records else None
    ]
    if not selected_indices:
        raise SystemExit("manifest has no selected answerable records")
    selected_records = [dataset.records[index] for index in selected_indices]
    prompts_by_id = {
        record.sample_id: oracle_prompts(record)  # type: ignore[arg-type]
        for record in selected_records
    }
    all_prompts = [
        prompt
        for prompts in prompts_by_id.values()
        for prompt in (prompts.union, prompts.anchor, prompts.answer)
    ]
    checkpoint = args.audiosep_checkpoint.resolve()
    prompt_conditions = encode_prompts(
        args.audiosep_root.resolve(), checkpoint, all_prompts
    )
    separator_adapter = AudioSepConditionedAdapter.from_repository(
        repository_root=args.audiosep_root,
        config_path=args.audiosep_config,
        checkpoint_path=checkpoint,
        device=device,
        freeze_separator=True,
    )
    separator = separator_adapter.ss_model.eval()

    listen_records = sorted(
        selected_records, key=listening_priority  # type: ignore[arg-type]
    )[: args.listening_count]
    listen_ids = {record.sample_id for record in listen_records}
    listening_dir = output_dir / "listening"
    if listen_ids:
        listening_dir.mkdir(parents=True, exist_ok=False)

    items = []
    raw_separator_calls = 0
    active_mixture_path: str | None = None
    active_raw_by_prompt: Dict[str, torch.Tensor] = {}
    for position, index in enumerate(selected_indices, start=1):
        record = dataset.records[index]
        assert isinstance(record, SUPPORTED_RECORD_TYPES)
        example = dataset[index]
        prompts = prompts_by_id[record.sample_id]
        mixture = example.mixture.to(device)
        if active_mixture_path != record.mixture_path:
            active_mixture_path = record.mixture_path
            active_raw_by_prompt = {}
        required = (prompts.union, prompts.anchor, prompts.answer)
        missing = [
            prompt
            for prompt in dict.fromkeys(required)
            if prompt not in active_raw_by_prompt
        ]
        physical_calls_this_record = 0
        if missing:
            new_raw, calls = render_unique_prompts(
                separator, mixture, prompt_conditions, missing
            )
            active_raw_by_prompt.update(
                {prompt: waveform.cpu() for prompt, waveform in new_raw.items()}
            )
            raw_separator_calls += calls
            physical_calls_this_record = calls
        union_raw = active_raw_by_prompt[prompts.union].to(device)
        anchor_raw = active_raw_by_prompt[prompts.anchor].to(device)
        answer_raw = active_raw_by_prompt[prompts.answer].to(device)
        anchor_gate = paint_intervals(
            record.anchor_intervals,
            record.sample_rate,
            mixture.numel(),
            device=device,
        )
        answer_gate = paint_intervals(
            record.answer_intervals,
            record.sample_rate,
            mixture.numel(),
            device=device,
        )
        rendered = compose_modes(
            union_raw,
            anchor_raw,
            answer_raw,
            anchor_gate,
            answer_gate,
            same_role_prompt=prompts.same_role_prompt,
        )
        target_evidence = example.evidence.to(device)
        target_residual = example.residual.to(device)
        target_anchor = example.anchor_stem.to(device)
        target_answer = example.answer_stem.to(device)
        metrics = {
            mode: evaluate_mode(
                rendered[mode],
                mixture,
                target_evidence,
                target_residual,
                target_anchor,
                target_answer,
            )
            for mode in MODES
        }
        item = {
            "id": record.sample_id,
            "scene_id": record.scene_id,
            "anchor_event_ids": list(record.anchor_event_ids),
            "answer_event_ids": list(record.answer_event_ids),
            "evidence_event_ids": list(record.evidence_event_ids),
            "split": record.split,
            "relation": record.relation,
            "question": record.question,
            "answer": record.answer,
            "anchor_label": prompts.anchor_label,
            "answer_label": prompts.answer_label,
            "union_prompt": prompts.union,
            "anchor_prompt": prompts.anchor,
            "answer_prompt": prompts.answer,
            "anchor_intervals_seconds": [
                list(span) for span in record.anchor_intervals
            ],
            "answer_intervals_seconds": [
                list(span) for span in record.answer_intervals
            ],
            "same_role_label": prompts.anchor_label == prompts.answer_label,
            "same_role_prompt": prompts.same_role_prompt,
            "separator_call_provenance": {
                "effective_calls_if_deployed_independently": {
                    "A": 1,
                    "B": 1 if prompts.same_role_prompt else 2,
                    "C": 1 if prompts.same_role_prompt else 2,
                },
                "physical_raw_calls_this_joint_cached_evaluation_row": (
                    physical_calls_this_record
                ),
                "physical_prompts_rendered_this_row": list(missing),
            },
            "role_windows_overlap": _role_overlap(record),
            "scene_semantic_overlap": getattr(record, "semantic_overlap", None),
            "metrics": metrics,
        }
        items.append(item)
        if record.sample_id in listen_ids:
            write_listening_case(
                listening_dir,
                record,
                prompts,
                mixture,
                target_evidence,
                target_anchor,
                target_answer,
                rendered,
                metrics,
            )
        if position % 10 == 0 or position == len(selected_indices):
            print(
                f"evaluated {position}/{len(selected_indices)} answerable records; "
                f"AudioSep calls={raw_separator_calls}",
                flush=True,
            )

    audiosep_config = args.audiosep_config.resolve()
    unique_summary = summarize_unique_acoustic_items(items)
    if sum(
        int(
            item["separator_call_provenance"][
                "physical_raw_calls_this_joint_cached_evaluation_row"
            ]
        )
        for item in items
    ) != raw_separator_calls:
        raise AssertionError("per-row physical call provenance does not sum to total")
    call_provenance = separator_call_provenance(items, raw_separator_calls)
    report = {
        "format": "qces_audiosep_dual_role_oracle_v2_matched_temporal_factorial",
        "scope": "offline_answerable_only_oracle_semantic_and_temporal_diagnostic",
        "claim_boundary": (
            "Frozen AudioSep backend diagnostic; not a learned QCES result, not "
            "an AudioSep improvement, and not a deployable method comparison."
        ),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "audiosep_checkpoint": str(checkpoint),
        "audiosep_checkpoint_sha256": sha256_file(checkpoint),
        "audiosep_checkpoint_size_bytes": checkpoint.stat().st_size,
        "audiosep_config": str(audiosep_config),
        "audiosep_config_sha256": sha256_file(audiosep_config),
        "device": str(device),
        "deterministic_algorithms": True,
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
        "protocol": {
            "union_prompt": (
                "Evidence-role labels joined in event-onset order, matching the "
                "existing oracle semantic target convention."
            ),
            "primary_pair": list(PRIMARY_PAIR),
            "matched_temporal_factorial": {
                "A": {
                    "mode": UNION_GATED,
                    "formula": "S(union_text) * union_gate",
                    "effective_calls_per_record": 1,
                },
                "B": {
                    "mode": DUAL_UNION_GATED,
                    "formula": (
                        "(S(anchor_text) + S(answer_text)) * union_gate for "
                        "distinct prompts; one reused stem * union_gate for "
                        "identical prompts"
                    ),
                    "effective_calls_per_record": "2 distinct / 1 identical",
                },
                "C": {
                    "mode": DUAL_GATED,
                    "formula": (
                        "S(anchor_text) * anchor_gate + S(answer_text) * "
                        "answer_gate; one reused stem * union_gate for identical "
                        "prompts"
                    ),
                    "effective_calls_per_record": "2 distinct / 1 identical",
                },
                "interpretation": {
                    "A_to_B": (
                        "Role-factorized semantic extraction plus linear waveform "
                        "fusion with the same oracle union window. This does not "
                        "isolate the text embedding from the fusion topology."
                    ),
                    "B_to_C": (
                        "Additional oracle role-to-window alignment with identical "
                        "semantic raw stems."
                    ),
                    "A_to_C": "Total semantic-plus-role-window difference.",
                },
            },
            "secondary_pair": list(SECONDARY_PAIR),
            "dual_fusion": (
                "Role-specific raw stems are multiplied by their oracle windows "
                "and added linearly; no amplitude clipping is applied."
            ),
            "same_prompt_rule": (
                "If anchor and answer prompts are identical, one raw prediction "
                "is reused and union-gated, never added to itself."
            ),
            "residual_definition": "R = X - E",
            "mixture_consistency_note": (
                "Mixture consistency is an arithmetic sanity check, not semantic "
                "faithfulness."
            ),
            "acoustic_unit_aggregation": (
                "One unit per (scene_id, unordered evidence_event_ids). Only "
                "role-symmetric metrics are aggregated because anchor/answer roles "
                "can swap across questions with the same waveform target."
            ),
        },
        "counts": {
            "manifest_records": len(dataset.records),
            "manifest_answerable_records": len(all_answerable_indices),
            "manifest_no_evidence_records_excluded": (
                len(dataset.records) - len(all_answerable_indices)
            ),
            "evaluated_answerable_records": len(items),
            "debug_subset": len(items) != len(all_answerable_indices),
            "same_role_label_records": sum(item["same_role_label"] for item in items),
            "overlapping_role_window_records": sum(
                item["role_windows_overlap"] for item in items
            ),
            "unique_acoustic_comparison_units": unique_summary["overall"]["count"],
            "uncached_union_semantic_calls": len(items),
            "uncached_dual_role_semantic_calls": sum(
                1 if item["same_role_prompt"] else 2 for item in items
            ),
            "raw_separator_calls_with_consecutive_scene_prompt_cache": (
                raw_separator_calls
            ),
            "listening_cases": len(listen_ids),
        },
        "separator_call_provenance": call_provenance,
        "summary": summarize_items(items),
        "summary_unique_acoustic_units": unique_summary,
        "items": items,
    }
    report_path = output_dir / "dual_role_oracle_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if listen_ids:
        checklist = [
            "# Dual-role AudioSep listening checklist",
            "",
            "Listen blind to the numbered WAVs before reading `metadata.json`.",
            "",
            "- Does each prediction retain both anchor and answer events?",
            "- Under the same union window, does B improve A or add bleed/noise?",
            "- With identical raw role stems, does C improve B by role alignment?",
            "- In overlap cases, are both simultaneous events audible?",
            "- In same-label cases, is either occurrence attenuated or duplicated?",
            "- Is there audible boundary chopping from the oracle windows?",
            "",
            "Metric arrows: SI-SDR/SD-SDR/SI-SDRi ↑; L1 ↓.",
            "",
            "Selected cases (same-label and overlapping-role cases are prioritized):",
            "",
            *[
                f"- `{record.sample_id}`: {record.question}"
                for record in listen_records
            ],
            "",
        ]
        (listening_dir / "README.md").write_text(
            "\n".join(checklist), encoding="utf-8"
        )
    print(json.dumps(report["summary"]["overall"], indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
