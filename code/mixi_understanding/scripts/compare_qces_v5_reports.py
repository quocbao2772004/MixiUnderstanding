#!/usr/bin/env python3
"""Paired scene-family bootstrap comparison for two QCES-v5 reports.

The reference is normally the union model and the candidate is normally the
dual-role model.  Every reported oriented delta is signed so that a positive
value means the candidate is better, including metrics whose native direction
is lower-is-better.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Mapping, Sequence


FORMAT = "qces_v5_paired_scene_family_bootstrap_v1"

# Per-item names emitted by evaluate_qces.py.  Keep directions explicit so a
# publication table cannot silently interpret an error metric backwards.
METRIC_DIRECTIONS: Dict[str, str] = {
    "evidence_sd_sdri": "↑",
    "evidence_sd_sdr": "↑",
    "weakest_role_sd_sdr": "↑",
    "anchor_sd_sdr": "↑",
    "answer_sd_sdr": "↑",
    "evidence_si_sdri": "↑",
    "evidence_si_sdr": "↑",
    "weakest_role_si_sdr": "↑",
    "anchor_si_sdr": "↑",
    "answer_si_sdr": "↑",
    "temporal_iou": "↑",
    "no_evidence_correct": "↑",
    "evidence_l1": "↓",
    "residual_l1": "↓",
    "mixture_consistency_l1": "↓",
}

DEFAULT_METRICS = (
    "evidence_sd_sdri",
    "evidence_sd_sdr",
    "weakest_role_sd_sdr",
    "temporal_iou",
    "no_evidence_correct",
    "evidence_l1",
    "residual_l1",
    "mixture_consistency_l1",
)

PAIR_METADATA_FIELDS = (
    "scene_family_id",
    "scene_id",
    "question_type",
    "question_semantics_id",
    "relation",
    "variant_id",
    "counterfactual_group_id",
    "paraphrase_family_id",
    "evaluation_axis",
    "primary_counterfactual_probe",
    "same_role_label",
    "same_label_repeat",
    "role_windows_overlap",
    "semantic_overlap",
    "hard_case_tags",
    "no_evidence",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-name", default="union")
    parser.add_argument("--candidate-name", default="dual")
    parser.add_argument(
        "--metric",
        action="append",
        dest="metrics",
        choices=tuple(METRIC_DIRECTIONS),
        help="Per-item metric to compare; repeat as needed (default: primary set).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--allow-partial-overlap",
        action="store_true",
        help=(
            "Compare the item-ID intersection. By default unequal item sets are "
            "rejected to prevent a silently unpaired paper comparison."
        ),
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_report(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: report must be a JSON object")
    if payload.get("format") != "qces_question_swap_eval_v1":
        raise ValueError(f"{path}: unsupported report format {payload.get('format')!r}")
    schema = payload.get("schema_version")
    if not isinstance(schema, str) or not schema.startswith("qces_v5_"):
        raise ValueError(f"{path}: expected a QCES-v5 report, got {schema!r}")
    if not isinstance(payload.get("items"), list):
        raise ValueError(f"{path}: report.items must be a list")
    return payload


def _index_items(report: Mapping[str, Any], *, label: str) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for position, value in enumerate(report["items"]):
        if not isinstance(value, dict):
            raise ValueError(f"{label}: items[{position}] must be an object")
        item_id = value.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(
                f"{label}: items[{position}].id must be a non-empty string"
            )
        if item_id in indexed:
            raise ValueError(f"{label}: duplicate item id {item_id!r}")
        family = value.get("scene_family_id")
        if not isinstance(family, str) or not family:
            raise ValueError(f"{label}: item {item_id!r} lacks scene_family_id")
        indexed[item_id] = value
    return indexed


def _validate_pair_metadata(
    item_id: str, reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> None:
    for field in PAIR_METADATA_FIELDS:
        if reference.get(field) != candidate.get(field):
            raise ValueError(
                f"item {item_id!r}: paired metadata mismatch for {field!r}: "
                f"{reference.get(field)!r} != {candidate.get(field)!r}"
            )


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def _metric_seed(seed: int, slice_name: str, metric: str) -> int:
    payload = f"{seed}\0{slice_name}\0{metric}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _compare_metric(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    metric: str,
    direction: str,
    slice_name: str,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> Dict[str, Any] | None:
    by_family: Dict[str, list[tuple[float, float]]] = defaultdict(list)
    for reference, candidate in pairs:
        reference_value = _numeric(reference.get(metric))
        candidate_value = _numeric(candidate.get(metric))
        if reference_value is None or candidate_value is None:
            continue
        by_family[str(reference["scene_family_id"])].append(
            (reference_value, candidate_value)
        )
    if not by_family:
        return None

    sign = 1.0 if direction == "↑" else -1.0
    family_statistics = []
    for family in sorted(by_family):
        values = by_family[family]
        reference_mean = mean(value[0] for value in values)
        candidate_mean = mean(value[1] for value in values)
        raw_delta = candidate_mean - reference_mean
        family_statistics.append(
            (reference_mean, candidate_mean, raw_delta, sign * raw_delta)
        )

    point_reference = mean(value[0] for value in family_statistics)
    point_candidate = mean(value[1] for value in family_statistics)
    raw_delta = mean(value[2] for value in family_statistics)
    oriented_delta = mean(value[3] for value in family_statistics)

    bootstrap_seed = _metric_seed(seed, slice_name, metric)
    generator = random.Random(bootstrap_seed)
    family_count = len(family_statistics)
    bootstrap_deltas = []
    for _ in range(bootstrap_samples):
        bootstrap_deltas.append(
            mean(
                family_statistics[generator.randrange(family_count)][3]
                for _ in range(family_count)
            )
        )
    bootstrap_deltas.sort()
    alpha = (1.0 - confidence_level) / 2.0

    return {
        "metric": metric,
        "metric_direction": direction,
        "reference_family_macro_mean": point_reference,
        "candidate_family_macro_mean": point_candidate,
        "raw_candidate_minus_reference": raw_delta,
        "oriented_delta_positive_means_candidate_better": oriented_delta,
        "confidence_interval": {
            "level": confidence_level,
            "method": "paired_percentile_scene_family_bootstrap",
            "lower": _quantile(bootstrap_deltas, alpha),
            "upper": _quantile(bootstrap_deltas, 1.0 - alpha),
        },
        "bootstrap_probability_oriented_delta_gt_zero": (
            sum(value > 0.0 for value in bootstrap_deltas) / bootstrap_samples
        ),
        "paired_items": sum(len(values) for values in by_family.values()),
        "scene_families": family_count,
        "bootstrap_seed": bootstrap_seed,
    }


def _available_slices(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]
) -> Dict[str, Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]]:
    predicates: Sequence[tuple[str, Callable[[Mapping[str, Any]], bool]]] = (
        ("all", lambda item: True),
        ("answerable", lambda item: item.get("no_evidence") is False),
        ("no_evidence", lambda item: item.get("no_evidence") is True),
        ("same_role_label", lambda item: item.get("same_role_label") is True),
        (
            "different_role_label",
            lambda item: item.get("same_role_label") is False,
        ),
        (
            "same_label_repeat",
            lambda item: item.get("same_label_repeat") is True,
        ),
        (
            "primary_counterfactual_probe",
            lambda item: item.get("primary_counterfactual_probe") is True,
        ),
        (
            "counterfactual_variants",
            lambda item: isinstance(item.get("variant_id"), str)
            and item.get("variant_id") != "base",
        ),
    )
    result = {}
    for name, predicate in predicates:
        selected = [pair for pair in pairs if predicate(pair[0])]
        if selected:
            result[name] = selected
    return result


def compare_reports(
    reference_report: Mapping[str, Any],
    candidate_report: Mapping[str, Any],
    *,
    reference_name: str,
    candidate_name: str,
    metrics: Sequence[str] | None = None,
    bootstrap_samples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 2026,
    allow_partial_overlap: bool = False,
) -> Dict[str, Any]:
    """Compare matched report items using a paired family-cluster bootstrap."""

    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    selected_metrics = tuple(metrics or DEFAULT_METRICS)
    unknown_metrics = set(selected_metrics) - set(METRIC_DIRECTIONS)
    if unknown_metrics:
        raise ValueError(f"unknown metrics: {sorted(unknown_metrics)}")
    if len(set(selected_metrics)) != len(selected_metrics):
        raise ValueError("metrics must not contain duplicates")

    reference_items = _index_items(reference_report, label=reference_name)
    candidate_items = _index_items(candidate_report, label=candidate_name)
    only_reference = sorted(set(reference_items) - set(candidate_items))
    only_candidate = sorted(set(candidate_items) - set(reference_items))
    if (only_reference or only_candidate) and not allow_partial_overlap:
        raise ValueError(
            "report item-ID sets differ; pass --allow-partial-overlap only for "
            f"a diagnostic comparison ({len(only_reference)} reference-only, "
            f"{len(only_candidate)} candidate-only)"
        )
    shared_ids = sorted(set(reference_items) & set(candidate_items))
    if not shared_ids:
        raise ValueError("reports have no shared item IDs")

    pairs = []
    for item_id in shared_ids:
        reference = reference_items[item_id]
        candidate = candidate_items[item_id]
        _validate_pair_metadata(item_id, reference, candidate)
        pairs.append((reference, candidate))

    slices: Dict[str, Any] = {}
    present_metric_count = 0
    for slice_name, slice_pairs in _available_slices(pairs).items():
        metric_results = {}
        for metric in selected_metrics:
            result = _compare_metric(
                slice_pairs,
                metric=metric,
                direction=METRIC_DIRECTIONS[metric],
                slice_name=slice_name,
                bootstrap_samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed,
            )
            if result is not None:
                metric_results[f"{metric}_{METRIC_DIRECTIONS[metric]}"] = result
                present_metric_count += 1
        slices[slice_name] = {
            "paired_items": len(slice_pairs),
            "scene_families": len(
                {str(reference["scene_family_id"]) for reference, _ in slice_pairs}
            ),
            "metrics": metric_results,
        }
    if not present_metric_count:
        raise ValueError("none of the selected metrics has a finite paired value")

    return {
        "format": FORMAT,
        "comparison": {
            "reference_name": reference_name,
            "candidate_name": candidate_name,
            "positive_oriented_delta_means": f"{candidate_name} is better",
            "native_raw_delta": f"{candidate_name} minus {reference_name}",
        },
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
        "estimator": {
            "unit": "scene_family_id",
            "point_estimate": "family_macro_mean_of_paired_item_differences",
            "bootstrap": "paired resampling of scene families with replacement",
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "seed": seed,
            "per_metric_seed_is_order_invariant": True,
        },
        "pairing": {
            "strict_item_set_match_required": not allow_partial_overlap,
            "exact_item_set_match": not only_reference and not only_candidate,
            "reference_items": len(reference_items),
            "candidate_items": len(candidate_items),
            "shared_items": len(shared_ids),
            "shared_scene_families": len(
                {str(reference["scene_family_id"]) for reference, _ in pairs}
            ),
            "reference_only_item_ids": only_reference,
            "candidate_only_item_ids": only_candidate,
            "paired_metadata_fields_verified": list(PAIR_METADATA_FIELDS),
        },
        "requested_metrics": [
            f"{metric}_{METRIC_DIRECTIONS[metric]}" for metric in selected_metrics
        ],
        "slices": slices,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reference_path = args.reference_report.resolve()
    candidate_path = args.candidate_report.resolve()
    output_path = args.output.resolve()
    if output_path in (reference_path, candidate_path):
        raise ValueError("output must not overwrite either source evaluation report")
    result = compare_reports(
        _load_report(reference_path),
        _load_report(candidate_path),
        reference_name=args.reference_name,
        candidate_name=args.candidate_name,
        metrics=args.metrics,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        seed=args.seed,
        allow_partial_overlap=args.allow_partial_overlap,
    )
    result["source_reports"] = {
        "reference": {
            "path": str(reference_path),
            "sha256": _sha256(reference_path),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
        },
    }
    _atomic_write_json(output_path, result)

    concise = {
        metric: {
            "reference": values["reference_family_macro_mean"],
            "candidate": values["candidate_family_macro_mean"],
            "oriented_delta_positive_is_better": values[
                "oriented_delta_positive_means_candidate_better"
            ],
            "ci_lower": values["confidence_interval"]["lower"],
            "ci_upper": values["confidence_interval"]["upper"],
        }
        for metric, values in result["slices"]["all"]["metrics"].items()
    }
    print(json.dumps(concise, indent=2, sort_keys=True))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
