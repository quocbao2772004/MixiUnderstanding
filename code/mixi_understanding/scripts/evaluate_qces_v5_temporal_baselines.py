#!/usr/bin/env python3
"""Evaluate non-semantic temporal masking controls on QCES-v5 mixtures."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _v5_item_metrics,
    summarize_v5_items,
    write_item_artifacts,
)
from mixi_understanding.scripts.evaluate_qces_v5_caption_planner_audiosep import (
    _role_metrics,
    metadata,
)


FORMAT_VERSION = "qces_v5_temporal_only_baselines_v1"
PAPER_TRAIN_MANIFEST_SHA256 = (
    "08856a7ecd65234a0b8a9d41d1cda4f72738bf5f8e51a777b56bdc2abebd4cfa"
)
PAPER_TRAIN_ANSWERABLE_COUNT = 9345
PAPER_TRAIN_UNION_DURATION_SECONDS = {
    "mean": 1.936536496789725,
    "median": 1.9293437500000001,
    "p90_nearest_rank_floor_index": 2.2480625,
    "maximum": 2.4883124999999993,
}
RANDOM_MODE = "random_compact_span__non_oracle"
ENERGY_MODE = "maximum_energy_compact_span__non_oracle"
ACTIVITY_MODE = "top_energy_activity_frames__non_oracle"
PREDICTED_MODE = "question_predicted_temporal_mask__non_oracle"
ORACLE_MODE = "oracle_temporal_mask__upper_bound"
BASE_MODES = (RANDOM_MODE, ENERGY_MODE, ACTIVITY_MODE, ORACLE_MODE)
MODE_REGISTRY: Dict[str, Dict[str, Any]] = {
    RANDOM_MODE: {
        "access": "non_oracle",
        "uses_question": False,
        "uses_oracle_time": False,
        "description": "One deterministic random contiguous compact span.",
    },
    ENERGY_MODE: {
        "access": "non_oracle",
        "uses_question": False,
        "uses_oracle_time": False,
        "description": "Contiguous compact span with maximum mixture energy.",
    },
    ACTIVITY_MODE: {
        "access": "non_oracle",
        "uses_question": False,
        "uses_oracle_time": False,
        "description": "Highest-energy non-overlapping activity frames within a fixed budget.",
    },
    PREDICTED_MODE: {
        "access": "non_oracle",
        "uses_question": True,
        "uses_oracle_time": False,
        "description": "Mask the mixture using spans predicted by a QCES checkpoint.",
    },
    ORACLE_MODE: {
        "access": "oracle_upper_bound",
        "uses_question": True,
        "uses_oracle_time": True,
        "description": "Mask the mixture using annotated anchor/answer intervals.",
    },
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--budget-seconds",
        type=float,
        default=2.25,
        help="Frozen from the dev-train p90 evidence-union duration.",
    )
    parser.add_argument("--activity-frame-seconds", type=float, default=0.02)
    parser.add_argument("--predicted-spans-report", type=Path)
    parser.add_argument("--modes", nargs="+", choices=tuple(MODE_REGISTRY))
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--allow-test-split", action="store_true")
    parser.add_argument("--no-render-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.budget_seconds <= 0 or args.activity_frame_seconds <= 0:
        parser.error("temporal durations must be positive")
    if args.max_records < 0:
        parser.error("--max-records must be non-negative")
    modes = tuple(dict.fromkeys(args.modes or BASE_MODES))
    if args.predicted_spans_report and args.modes is None:
        modes = (*modes[:-1], PREDICTED_MODE, modes[-1])
    if PREDICTED_MODE in modes and args.predicted_spans_report is None:
        parser.error(f"{PREDICTED_MODE} requires --predicted-spans-report")
    args.modes = modes
    return args


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:8], "big")) % (2**63)


def paint_intervals(
    intervals: Sequence[Sequence[float]], sample_rate: int, samples: int
) -> torch.Tensor:
    mask = torch.zeros(samples, dtype=torch.float32)
    for interval in intervals:
        if len(interval) != 2:
            raise ValueError("every temporal interval must have two elements")
        start_seconds, end_seconds = map(float, interval)
        if (
            not math.isfinite(start_seconds)
            or not math.isfinite(end_seconds)
            or start_seconds < 0
            or end_seconds <= start_seconds
            or end_seconds > samples / sample_rate + 1e-6
        ):
            raise ValueError(f"invalid temporal interval: {interval}")
        start = max(0, min(round(start_seconds * sample_rate), samples))
        end = max(start, min(round(end_seconds * sample_rate), samples))
        mask[start:end] = 1.0
    return mask


def random_compact_mask(
    samples: int, sample_rate: int, budget_seconds: float, seed: int
) -> torch.Tensor:
    width = min(samples, max(1, round(budget_seconds * sample_rate)))
    start = random.Random(seed).randrange(samples - width + 1)
    mask = torch.zeros(samples, dtype=torch.float32)
    mask[start : start + width] = 1.0
    return mask


def maximum_energy_compact_mask(
    mixture: torch.Tensor, sample_rate: int, budget_seconds: float
) -> torch.Tensor:
    samples = mixture.numel()
    width = min(samples, max(1, round(budget_seconds * sample_rate)))
    if width == samples:
        return torch.ones(samples, dtype=torch.float32)
    # Evaluate starts every 10 ms and include the final legal start. This is
    # deterministic, cheap, and independent of question/annotation fields.
    hop = max(1, round(0.01 * sample_rate))
    starts = list(range(0, samples - width + 1, hop))
    if starts[-1] != samples - width:
        starts.append(samples - width)
    cumulative = F.pad(mixture.float().square().cumsum(dim=0), (1, 0))
    start_tensor = torch.tensor(starts, dtype=torch.long)
    energies = cumulative[start_tensor + width] - cumulative[start_tensor]
    best_start = starts[int(torch.argmax(energies))]
    mask = torch.zeros(samples, dtype=torch.float32)
    mask[best_start : best_start + width] = 1.0
    return mask


def top_energy_activity_mask(
    mixture: torch.Tensor,
    sample_rate: int,
    budget_seconds: float,
    frame_seconds: float,
) -> torch.Tensor:
    samples = mixture.numel()
    frame = max(1, round(frame_seconds * sample_rate))
    frames = math.ceil(samples / frame)
    padded = F.pad(mixture.float().square(), (0, frames * frame - samples))
    energies = padded.reshape(frames, frame).mean(dim=1)
    selected_count = min(frames, max(1, math.ceil(budget_seconds / frame_seconds)))
    # Stable tie-breaking by frame index: add a tiny descending deterministic
    # offset well below float32 audio-energy resolution.
    tie_break = torch.arange(frames, 0, -1, dtype=energies.dtype) * 1e-12
    selected = torch.topk(energies + tie_break, selected_count).indices
    mask = torch.zeros(frames, frame, dtype=torch.float32)
    mask[selected] = 1.0
    return mask.reshape(-1)[:samples]


def load_predicted_spans(
    path: Path,
    *,
    manifest: Path,
    records: Sequence[QCESV5Record],
) -> tuple[Dict[str, Sequence[Sequence[float]]], Dict[str, Any]]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    manifest_sha = sha256_file(manifest)
    if payload.get("manifest_sha256") != manifest_sha:
        raise ValueError("predicted-span report is bound to another manifest")
    if (
        not isinstance(payload.get("manifest"), str)
        or Path(payload["manifest"]).resolve() != manifest
    ):
        raise ValueError("predicted-span report manifest path mismatch")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("predicted-span report items are missing")
    indexed: Dict[str, Sequence[Sequence[float]]] = {}
    for item in items:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ValueError("invalid predicted-span report item")
        sample_id = str(item["id"])
        intervals = item.get("predicted_evidence_intervals")
        if sample_id in indexed or not isinstance(intervals, list):
            raise ValueError("duplicate/invalid predicted-span item")
        indexed[sample_id] = intervals
    missing = sorted({record.sample_id for record in records} - set(indexed))
    if missing:
        raise ValueError(f"predicted-span report is missing IDs: {missing[:3]}")
    checkpoint = payload.get("checkpoint")
    checkpoint_identity = None
    if isinstance(checkpoint, str) and Path(checkpoint).resolve().is_file():
        checkpoint_path = Path(checkpoint).resolve()
        checkpoint_identity = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
        }
    return indexed, {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "manifest_sha256": manifest_sha,
        "checkpoint": checkpoint_identity,
    }


def temporal_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted = prediction >= 0.5
    expected = target >= 0.5
    union = (predicted | expected).sum()
    if int(union) == 0:
        return 1.0
    return float((predicted & expected).sum() / union)


def summarize_items(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary = summarize_v5_items(items)

    def values(key: str) -> List[float]:
        return [
            float(item["metrics"][key])
            for item in items
            if item["metrics"].get(key) is not None
        ]

    def mean(key: str) -> Optional[float]:
        collected = values(key)
        return sum(collected) / len(collected) if collected else None

    summary.update(
        {
            "answerable_temporal_iou_mean_↑": mean("answerable_temporal_iou_↑"),
            "weakest_role_si_sdr_answerable_mean_db_↑": mean(
                "weakest_role_si_sdr_db_↑"
            ),
            "weakest_role_sd_sdr_answerable_mean_db_↑": mean(
                "weakest_role_sd_sdr_db_↑"
            ),
        }
    )
    return summary


def grouped_summaries(
    items: Sequence[Mapping[str, Any]], modes: Sequence[str], field: str
) -> Dict[str, Any]:
    return {
        group: {
            mode: summarize_items(
                [
                    item
                    for item in items
                    if item["mode"] == mode and str(item[field]) == group
                ]
            )
            for mode in modes
        }
        for group in sorted({str(item[field]) for item in items})
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    manifest = args.manifest.resolve()
    dataset = QCESManifestDataset(manifest, crop_samples=None)
    records = [record for record in dataset.records if isinstance(record, QCESV5Record)]
    if len(records) != len(dataset.records):
        raise ValueError("temporal baselines require one pure QCES-v5 manifest")
    if args.max_records:
        records = records[: args.max_records]
    if not records:
        raise ValueError("selected QCES-v5 record set is empty")
    if (
        any(record.split.startswith("test") for record in records)
        and not args.allow_test_split
    ):
        raise SystemExit("test split is sealed; pass --allow-test-split explicitly")
    predicted_spans: Dict[str, Sequence[Sequence[float]]] = {}
    predicted_provenance = None
    if args.predicted_spans_report:
        predicted_spans, predicted_provenance = load_predicted_spans(
            args.predicted_spans_report,
            manifest=manifest,
            records=records,
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        example = dataset[index]
        target_mask = torch.maximum(example.anchor_mask, example.answer_mask)
        for mode in args.modes:
            if mode == RANDOM_MODE:
                mask = random_compact_mask(
                    record.num_samples,
                    record.sample_rate,
                    args.budget_seconds,
                    sample_seed(args.seed, record.sample_id),
                )
            elif mode == ENERGY_MODE:
                mask = maximum_energy_compact_mask(
                    example.mixture, record.sample_rate, args.budget_seconds
                )
            elif mode == ACTIVITY_MODE:
                mask = top_energy_activity_mask(
                    example.mixture,
                    record.sample_rate,
                    args.budget_seconds,
                    args.activity_frame_seconds,
                )
            elif mode == PREDICTED_MODE:
                mask = paint_intervals(
                    predicted_spans[record.sample_id],
                    record.sample_rate,
                    record.num_samples,
                )
            elif mode == ORACLE_MODE:
                mask = paint_intervals(
                    (*record.anchor_intervals, *record.answer_intervals),
                    record.sample_rate,
                    record.num_samples,
                )
            else:  # pragma: no cover - argparse/constant invariant
                raise AssertionError(mode)
            evidence = example.mixture * mask
            residual = example.mixture - evidence
            metrics, descriptives = _v5_item_metrics(
                no_evidence=record.no_evidence,
                evidence=evidence,
                mixture=example.mixture,
                target=example.evidence,
                target_residual=example.residual,
            )
            metrics.update(
                _role_metrics(
                    no_evidence=record.no_evidence,
                    evidence=evidence,
                    anchor_mask=example.anchor_mask,
                    answer_mask=example.answer_mask,
                    anchor_target=example.anchor_stem,
                    answer_target=example.answer_stem,
                )
            )
            metrics["answerable_temporal_iou_↑"] = (
                temporal_iou(mask, target_mask) if not record.no_evidence else None
            )
            item = {
                **metadata(record),
                "mode": mode,
                "baseline_access": MODE_REGISTRY[mode]["access"],
                "metrics": metrics,
                "descriptives": {
                    **descriptives,
                    "retained_duration_seconds": float(mask.sum() / record.sample_rate),
                },
            }
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
                render_audio=not args.no_render_audio,
            )
            items.append(item)

    summaries = {
        mode: summarize_items([item for item in items if item["mode"] == mode])
        for mode in args.modes
    }
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
        "modes": list(args.modes),
        "mode_registry": {mode: MODE_REGISTRY[mode] for mode in args.modes},
        "budget": {
            "seconds": args.budget_seconds,
            "source": "frozen full paper-train p90 evidence-union duration rounded to 2.25 s",
            "paper_train_manifest_sha256": PAPER_TRAIN_MANIFEST_SHA256,
            "paper_train_answerable_count": PAPER_TRAIN_ANSWERABLE_COUNT,
            "paper_train_union_duration_seconds": PAPER_TRAIN_UNION_DURATION_SECONDS,
            "activity_frame_seconds": args.activity_frame_seconds,
        },
        "seed": args.seed,
        "predicted_span_provenance": predicted_provenance,
        "rendered_audio": not args.no_render_audio,
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
        "protocol_limitations": [
            "Random/energy controls do not consume the question and therefore test temporal/energy shortcuts only.",
            "The oracle temporal mask consumes annotated intervals and is an upper bound, not deployable.",
            "All modes mask the mixture without semantic source separation; R=X-E is arithmetic.",
        ],
        "summaries": summaries,
        "summaries_by_relation": grouped_summaries(items, args.modes, "relation"),
        "summaries_by_variant": grouped_summaries(items, args.modes, "variant_id"),
        "items": items,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
