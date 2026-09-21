#!/usr/bin/env python3
"""Evaluate BEATs event-presence headroom on oracle QCES evidence/complements.

This gate is independent of a learned QCES checkpoint.  It asks whether an
AudioSet-fine-tuned frozen encoder assigns required event labels more strongly
to E* than R*.  Gold labels select output coordinates after inference only.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.beats_auditor import (
    BEATsAudioSetAuditor,
    score_qces_streams,
)
from mixi_understanding.qces.data import QCESManifestDataset


FORMAT = "qces_beats_oracle_event_auditor_v1"
ORACLE_METRIC_DIRECTIONS = {
    "required_probability_mixture": "↑",
    "oracle_required_probability_evidence": "↑",
    "oracle_required_probability_residual": "↓",
    "oracle_evidence_residual_contrast": "↑",
    "oracle_vs_mixture_sufficiency_delta": "↑",
    "mixture_vs_oracle_residual_necessity_delta": "↑",
    "oracle_excluded_probability_evidence": "↓",
    "oracle_excluded_suppression_delta": "↑",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    baseline = PROJECT_ROOT / "code/baseline/beats-unilm"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beats-root", type=Path, default=baseline)
    parser.add_argument(
        "--beats-checkpoint",
        type=Path,
        default=baseline / "checkpoint/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    )
    parser.add_argument(
        "--audioset-labels",
        type=Path,
        default=baseline / "checkpoint/class_labels_indices.csv",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stream-batch-size", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-independent-families", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if (
        args.stream_batch_size <= 0
        or args.bootstrap_samples <= 0
        or args.minimum_independent_families <= 1
    ):
        parser.error(
            "batch size, bootstrap samples, and family minimum must be positive"
        )
    return args


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(sum(values) / len(values))


def cluster_bootstrap_interval(
    items: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    cluster_key: str,
    samples: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in items:
        cluster = item.get(cluster_key)
        value = item.get(metric)
        if not isinstance(cluster, str) or not cluster:
            raise ValueError(f"item lacks cluster key: {cluster_key}")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        grouped[cluster].append(float(value))
    if len(grouped) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    clusters = sorted(grouped)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = [rng.choice(clusters) for _ in clusters]
        values = [value for cluster in selected for value in grouped[cluster]]
        draws.append(_mean(values))
    draws.sort()

    def percentile(fraction: float) -> float:
        position = fraction * (len(draws) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        return draws[lower] * (1.0 - weight) + draws[upper] * weight

    return {
        "point_estimate": _mean(
            [value for values in grouped.values() for value in values]
        ),
        "lower_95": percentile(0.025),
        "upper_95": percentile(0.975),
        "clusters_↑": len(clusters),
        "bootstrap_samples_↑": samples,
    }


def summarize_oracle_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("oracle BEATs evaluation has no answerable items")
    summary = {}
    for metric in ORACLE_METRIC_DIRECTIONS:
        values = [
            float(item[metric])
            for item in items
            if isinstance(item.get(metric), (int, float))
            and math.isfinite(float(item[metric]))
        ]
        if values:
            summary[metric] = _mean(values)
    sign_rate = _mean(
        [float(item["oracle_evidence_residual_contrast"] > 0.0) for item in items]
    )
    return {
        "answerable_records_↑": len(items),
        "oracle_evidence_beats_residual_rate_↑": sign_rate,
        "summary": summary,
        "summary_with_directions": {
            **{
                f"{metric}_{ORACLE_METRIC_DIRECTIONS[metric]}": value
                for metric, value in summary.items()
            },
            "oracle_evidence_beats_residual_rate_↑": sign_rate,
        },
    }


def build_gates(
    family_count: int,
    minimum_families: int,
    contrast_interval: Mapping[str, float],
    necessity_interval: Mapping[str, float],
) -> list[dict[str, Any]]:
    return [
        {
            "metric": "independent_scene_families",
            "direction": "↑",
            "value": family_count,
            "operator": ">=",
            "threshold": minimum_families,
            "passed": family_count >= minimum_families,
        },
        {
            "metric": "oracle_evidence_residual_contrast_lower_95",
            "direction": "↑",
            "value": contrast_interval["lower_95"],
            "operator": ">",
            "threshold": 0.0,
            "passed": contrast_interval["lower_95"] > 0.0,
        },
        {
            "metric": "mixture_vs_oracle_residual_necessity_lower_95",
            "direction": "↑",
            "value": necessity_interval["lower_95"],
            "operator": ">",
            "threshold": 0.0,
            "passed": necessity_interval["lower_95"] > 0.0,
        },
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output}; use --overwrite")
    for path in (
        args.manifest,
        args.beats_root,
        args.beats_checkpoint,
        args.audioset_labels,
    ):
        if not path.resolve().exists():
            raise SystemExit(f"required input does not exist: {path.resolve()}")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dataset = QCESManifestDataset(args.manifest.resolve(), crop_samples=None)
    if not dataset.records or not all(
        isinstance(record, QCESV5Record) for record in dataset.records
    ):
        raise SystemExit("BEATs oracle audit requires a QCES-v5 manifest")
    labels = {event.label for record in dataset.records for event in record.events}
    auditor = BEATsAudioSetAuditor.from_assets(
        args.beats_root,
        args.beats_checkpoint,
        args.audioset_labels,
        labels,
        device,
    )

    scene_probabilities: dict[str, torch.Tensor] = {}
    oracle_probabilities: dict[
        tuple[str, tuple[str, ...]], tuple[torch.Tensor, torch.Tensor]
    ] = {}
    items = []
    with torch.inference_mode():
        for index, record in enumerate(dataset.records):
            if record.no_evidence:
                continue
            key = (record.scene_id, tuple(sorted(record.evidence_event_ids)))
            needs_scene = record.scene_id not in scene_probabilities
            needs_oracle = key not in oracle_probabilities
            if needs_scene or needs_oracle:
                example = dataset[index]
            if needs_scene:
                scene_probabilities[record.scene_id] = auditor.score(
                    example.mixture[None],
                    record.sample_rate,
                    args.stream_batch_size,
                )[0]
            if needs_oracle:
                pair = auditor.score(
                    torch.stack((example.evidence, example.residual)),
                    record.sample_rate,
                    args.stream_batch_size,
                )
                oracle_probabilities[key] = (pair[0], pair[1])
                if (
                    len(oracle_probabilities) == 1
                    or len(oracle_probabilities) % 25 == 0
                ):
                    print(
                        "BEATs oracle progress: "
                        f"scene mixtures={len(scene_probabilities)} ↑, "
                        f"unique evidence pairs={len(oracle_probabilities)} ↑",
                        flush=True,
                    )
            mixture_probability = scene_probabilities[record.scene_id]
            evidence_probability, residual_probability = oracle_probabilities[key]
            probabilities = torch.stack(
                (
                    mixture_probability,
                    evidence_probability,
                    residual_probability,
                    evidence_probability,
                    residual_probability,
                )
            )
            required_labels = [
                record.event_by_id(event_id).label
                for event_id in record.evidence_event_ids
            ]
            required_set = set(required_labels)
            evidence_event_ids = set(record.evidence_event_ids)
            residual_event_labels = sorted(
                {
                    event.label
                    for event in record.events
                    if event.event_id not in evidence_event_ids
                }
            )
            excluded_labels = sorted(
                {event.label for event in record.events} - required_set
            )
            scored = score_qces_streams(
                probabilities,
                required_labels,
                excluded_labels,
                auditor.label_indices,
                residual_event_labels,
            )
            items.append(
                {
                    "id": record.sample_id,
                    "scene_id": record.scene_id,
                    "scene_family_id": record.scene_family_id,
                    "variant_id": record.variant_id,
                    "relation": record.relation,
                    **{
                        key: value
                        for key, value in scored.items()
                        if not key.startswith("predicted_")
                    },
                }
            )

    summary = summarize_oracle_items(items)
    contrast_interval = cluster_bootstrap_interval(
        items,
        "oracle_evidence_residual_contrast",
        cluster_key="scene_family_id",
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    necessity_interval = cluster_bootstrap_interval(
        items,
        "mixture_vs_oracle_residual_necessity_delta",
        cluster_key="scene_family_id",
        samples=args.bootstrap_samples,
        seed=args.seed + 1,
    )
    family_count = len({item["scene_family_id"] for item in items})
    gates = build_gates(
        family_count,
        args.minimum_independent_families,
        contrast_interval,
        necessity_interval,
    )
    report = {
        "format": FORMAT,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(
            args.manifest.resolve().read_bytes()
        ).hexdigest(),
        "device": str(device),
        "auditor": auditor.provenance,
        "records_total_↑": len(dataset.records),
        "answerable_records_scored_↑": len(items),
        "unique_scene_mixture_forwards_↓": len(scene_probabilities),
        "unique_oracle_pair_forwards_↓": len(oracle_probabilities),
        "summary": summary,
        "cluster_bootstrap": {
            "unit": "scene_family_id",
            "oracle_evidence_residual_contrast_↑": contrast_interval,
            "mixture_vs_oracle_residual_necessity_delta_↑": necessity_interval,
        },
        "gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates),
        "paper_result_eligible": all(gate["passed"] for gate in gates),
        "claim_boundary": (
            "event-presence headroom for oracle stems; not predicted QCES quality, "
            "not generative QA, and not causal identification"
        ),
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "items": items,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps({**summary["summary_with_directions"], "gates": gates}, indent=2))
    print(f"wrote {output}")
    if not report["all_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
