#!/usr/bin/env python3
"""Audit QCES temporal and no-evidence shortcuts without loading audio.

The two baselines are deliberately weak lookup models.  For every held-out
scene they average the binary target masks in all *other* scenes, grouped by
either the relation/question type or the normalized exact question.  An
unseen group falls back to the training-fold global prototype.  Consequently
the reported scores cannot result from memorizing another question attached
to the same waveform/scene.

This file intentionally depends only on the Python standard library.  It reads
raw JSONL fields rather than a version-specific QCES schema so that the same
audit can be run on qces_v3 and compatible future manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MASK_THRESHOLD = 0.5
NO_EVIDENCE_THRESHOLD = 0.5
SLOT_BINS = 8

UNION_MIOU = "answerable_union_temporal_miou_↑"
ANCHOR_MIOU = "answerable_anchor_temporal_miou_↑"
ANSWER_MIOU = "answerable_answer_temporal_miou_↑"
NO_EVIDENCE_ACCURACY = "no_evidence_binary_accuracy_↑"
NO_EVIDENCE_RECALL = "no_evidence_recall_↑"
ANSWERABLE_RECALL = "answerable_recall_↑"
SEEN_KEY_RATE = "held_out_prototype_key_seen_rate_↑"
FALLBACK_RATE = "global_prototype_fallback_rate_↓"

BOOTSTRAP_METRICS = (
    UNION_MIOU,
    ANCHOR_MIOU,
    ANSWER_MIOU,
    NO_EVIDENCE_ACCURACY,
    NO_EVIDENCE_RECALL,
    ANSWERABLE_RECALL,
)


@dataclass(frozen=True)
class Example:
    """Version-independent representation used by the shortcut audit."""

    record_id: str
    scene_id: str
    operator: str
    question_key: str
    no_evidence: bool
    anchor_mask: Tuple[int, ...]
    answer_mask: Tuple[int, ...]
    anchor_slots: Tuple[int, ...]
    answer_slots: Tuple[int, ...]

    @property
    def union_mask(self) -> Tuple[int, ...]:
        return tuple(int(a or b) for a, b in zip(self.anchor_mask, self.answer_mask))


@dataclass(frozen=True)
class Prediction:
    """One leave-one-scene-out prediction."""

    predicted_no_evidence: bool
    anchor_mask: Tuple[int, ...]
    answer_mask: Tuple[int, ...]
    key_seen: bool

    @property
    def union_mask(self) -> Tuple[int, ...]:
        return tuple(int(a or b) for a, b in zip(self.anchor_mask, self.answer_mask))


@dataclass(frozen=True)
class EvaluatedRow:
    example: Example
    prediction: Prediction


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args(argv)
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be at least 1")
    return args


def normalize_question(question: str) -> str:
    """Normalize formatting while intentionally retaining all label tokens."""

    normalized = unicodedata.normalize("NFKC", question).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _nested_get(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first(payload: Mapping[str, Any], paths: Sequence[str]) -> Any:
    for path in paths:
        value = _nested_get(payload, path)
        if value is not None:
            return value
    return None


def _required_text(payload: Mapping[str, Any], paths: Sequence[str], context: str) -> str:
    value = _first(payload, paths)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires a non-empty field from {list(paths)}")
    return value.strip()


def _duration_seconds(payload: Mapping[str, Any], context: str) -> float:
    value = _first(
        payload,
        (
            "duration_seconds",
            "audio_duration_seconds",
            "duration",
            "timeline.duration_seconds",
        ),
    )
    if value is None:
        num_samples = _first(payload, ("num_samples", "audio.num_samples"))
        sample_rate = _first(payload, ("sample_rate", "audio.sample_rate"))
        if num_samples is not None and sample_rate is not None:
            value = float(num_samples) / float(sample_rate)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} has no numeric duration")
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"{context} duration must be finite and positive")
    return duration


def _parse_intervals(value: Any, context: str) -> Tuple[Tuple[float, float], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        value = value.get("intervals", value.get("spans", value.get("timestamps")))
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list of intervals")
    if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
        value = [value]
    result: List[Tuple[float, float]] = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            start = _first(
                item,
                ("start", "onset", "start_seconds", "onset_seconds"),
            )
            end = _first(
                item,
                ("end", "offset", "end_seconds", "offset_seconds"),
            )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = item
        else:
            raise ValueError(f"{context}[{index}] is not a two-ended interval")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
        ):
            raise ValueError(f"{context}[{index}] endpoints must be numeric")
        start_float, end_float = float(start), float(end)
        if (
            not math.isfinite(start_float)
            or not math.isfinite(end_float)
            or start_float < 0.0
            or end_float <= start_float
        ):
            raise ValueError(f"{context}[{index}] must satisfy 0 <= start < end")
        result.append((start_float, end_float))
    return tuple(result)


def intervals_to_mask(
    intervals: Sequence[Tuple[float, float]], duration: float, frames: int
) -> Tuple[int, ...]:
    """Rasterize intervals onto a duration-normalized, fixed-size frame grid."""

    mask = [0] * frames
    for start, end in intervals:
        if start >= duration or end > duration + 1e-6:
            raise ValueError(
                f"interval {(start, end)} lies outside duration {duration}"
            )
        start = min(start, duration)
        end = min(end, duration)
        # A frame is positive if the half-open interval overlaps it.  Small
        # epsilons keep exact frame boundaries from spilling into neighbors.
        first_frame = int(math.floor((start / duration) * frames + 1e-12))
        final_frame = int(math.ceil((end / duration) * frames - 1e-12))
        first_frame = min(frames - 1, max(0, first_frame))
        final_frame = min(frames, max(first_frame + 1, final_frame))
        for frame in range(first_frame, final_frame):
            mask[frame] = 1
    return tuple(mask)


def _interval_slots(
    intervals: Sequence[Tuple[float, float]], duration: float
) -> Tuple[int, ...]:
    slots = {
        min(SLOT_BINS - 1, int((((start + end) / 2.0) / duration) * SLOT_BINS))
        for start, end in intervals
    }
    return tuple(sorted(slots))


def _no_evidence(payload: Mapping[str, Any]) -> bool:
    value = _first(
        payload,
        (
            "no_evidence",
            "labels.no_evidence",
            "target.no_evidence",
            "evidence.no_evidence",
        ),
    )
    if value is not None:
        if not isinstance(value, bool):
            raise ValueError("no_evidence must be boolean")
        return value
    answerable = _first(payload, ("answerable", "labels.answerable"))
    if answerable is not None:
        if not isinstance(answerable, bool):
            raise ValueError("answerable must be boolean")
        return not answerable
    answer = _first(payload, ("answer", "target.answer"))
    if isinstance(answer, str):
        normalized = normalize_question(answer).replace("_", " ")
        return normalized in {"no evidence", "none", "not answerable"}
    return False


def _load_examples(manifest: Path, frames: int) -> Tuple[List[Example], List[str]]:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    examples: List[Example] = []
    schema_versions: set[str] = set()
    record_ids: set[str] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number} is not valid JSON: {error}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            context = f"line {line_number}"
            record_id = _required_text(
                payload, ("id", "sample_id", "question_id"), context
            )
            if record_id in record_ids:
                raise ValueError(f"duplicate record id: {record_id}")
            record_ids.add(record_id)
            scene_id = _required_text(
                payload, ("scene_id", "scene.id", "group_id"), context
            )
            question = _required_text(
                payload, ("question", "query", "qa.question"), context
            )
            operator = _required_text(
                payload,
                # qces_v4 deliberately decouples relation="after" from the
                # answerability label; prefer it over the legacy question_type.
                ("relation", "question_type", "operator"),
                context,
            ).casefold()
            duration = _duration_seconds(payload, context)
            anchor_intervals = _parse_intervals(
                _first(
                    payload,
                    ("anchor_intervals",),
                ),
                f"{context}.anchor_intervals",
            )
            answer_intervals = _parse_intervals(
                _first(
                    payload,
                    ("answer_intervals",),
                ),
                f"{context}.answer_intervals",
            )
            no_evidence = _no_evidence(payload)
            if no_evidence and (anchor_intervals or answer_intervals):
                raise ValueError(f"{context}: no-evidence target has role intervals")
            if not no_evidence and (not anchor_intervals or not answer_intervals):
                raise ValueError(f"{context}: answerable target lacks a role interval")
            schema = payload.get("schema_version")
            if isinstance(schema, str):
                schema_versions.add(schema)
            examples.append(
                Example(
                    record_id=record_id,
                    scene_id=scene_id,
                    operator=operator,
                    question_key=normalize_question(question),
                    no_evidence=no_evidence,
                    anchor_mask=intervals_to_mask(anchor_intervals, duration, frames),
                    answer_mask=intervals_to_mask(answer_intervals, duration, frames),
                    anchor_slots=_interval_slots(anchor_intervals, duration),
                    answer_slots=_interval_slots(answer_intervals, duration),
                )
            )
    if not examples:
        raise ValueError("manifest contains no records")
    examples.sort(key=lambda row: (row.scene_id, row.record_id))
    scenes = {row.scene_id for row in examples}
    if len(scenes) < 2:
        raise ValueError("leave-one-scene-out evaluation requires at least two scenes")
    return examples, sorted(schema_versions)


def _average_mask(rows: Sequence[Example], role: str, frames: int) -> Tuple[int, ...]:
    if not rows:
        return (0,) * frames
    totals = [0] * frames
    for row in rows:
        mask = row.anchor_mask if role == "anchor" else row.answer_mask
        for index, value in enumerate(mask):
            totals[index] += value
    return tuple(int(total / len(rows) >= MASK_THRESHOLD) for total in totals)


def _prototype_prediction(
    train_rows: Sequence[Example], key: str, key_fn: Callable[[Example], str], frames: int
) -> Prediction:
    grouped_rows = [row for row in train_rows if key_fn(row) == key]
    key_seen = bool(grouped_rows)
    prototype_rows = grouped_rows if grouped_rows else list(train_rows)
    no_evidence_rate = statistics.fmean(row.no_evidence for row in prototype_rows)
    return Prediction(
        predicted_no_evidence=no_evidence_rate >= NO_EVIDENCE_THRESHOLD,
        anchor_mask=_average_mask(prototype_rows, "anchor", frames),
        answer_mask=_average_mask(prototype_rows, "answer", frames),
        key_seen=key_seen,
    )


def _evaluate_loso(
    examples: Sequence[Example], key_fn: Callable[[Example], str], frames: int
) -> List[EvaluatedRow]:
    scenes = sorted({row.scene_id for row in examples})
    results: List[EvaluatedRow] = []
    for held_scene in scenes:
        train_rows = [row for row in examples if row.scene_id != held_scene]
        test_rows = [row for row in examples if row.scene_id == held_scene]
        for row in test_rows:
            results.append(
                EvaluatedRow(
                    example=row,
                    prediction=_prototype_prediction(
                        train_rows, key_fn(row), key_fn, frames
                    ),
                )
            )
    return results


def _binary_iou(predicted: Sequence[int], target: Sequence[int]) -> float:
    intersection = sum(bool(a) and bool(b) for a, b in zip(predicted, target))
    union = sum(bool(a) or bool(b) for a, b in zip(predicted, target))
    return float(intersection / union) if union else 1.0


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    materialized = list(values)
    return statistics.fmean(materialized) if materialized else None


def _metric_values(rows: Sequence[EvaluatedRow]) -> Dict[str, Optional[float]]:
    answerable = [row for row in rows if not row.example.no_evidence]
    positives = [row for row in rows if row.example.no_evidence]
    negatives = answerable
    correct = [
        float(row.prediction.predicted_no_evidence == row.example.no_evidence)
        for row in rows
    ]
    return {
        UNION_MIOU: _mean_or_none(
            _binary_iou(row.prediction.union_mask, row.example.union_mask)
            for row in answerable
        ),
        ANCHOR_MIOU: _mean_or_none(
            _binary_iou(row.prediction.anchor_mask, row.example.anchor_mask)
            for row in answerable
        ),
        ANSWER_MIOU: _mean_or_none(
            _binary_iou(row.prediction.answer_mask, row.example.answer_mask)
            for row in answerable
        ),
        NO_EVIDENCE_ACCURACY: _mean_or_none(correct),
        NO_EVIDENCE_RECALL: _mean_or_none(
            float(row.prediction.predicted_no_evidence) for row in positives
        ),
        ANSWERABLE_RECALL: _mean_or_none(
            float(not row.prediction.predicted_no_evidence) for row in negatives
        ),
        SEEN_KEY_RATE: _mean_or_none(float(row.prediction.key_seen) for row in rows),
        FALLBACK_RATE: _mean_or_none(float(not row.prediction.key_seen) for row in rows),
    }


def _per_scene_metrics(rows: Sequence[EvaluatedRow]) -> Dict[str, Dict[str, Optional[float]]]:
    grouped: Dict[str, List[EvaluatedRow]] = defaultdict(list)
    for row in rows:
        grouped[row.example.scene_id].append(row)
    return {scene: _metric_values(grouped[scene]) for scene in sorted(grouped)}


def _quantile(values: Sequence[float], probability: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _scene_bootstrap(
    evaluated: Mapping[str, Sequence[EvaluatedRow]], samples: int, seed: int
) -> Dict[str, Any]:
    baseline_names = sorted(evaluated)
    scenes = sorted(
        {row.example.scene_id for rows in evaluated.values() for row in rows}
    )
    by_baseline_scene: Dict[str, Dict[str, List[EvaluatedRow]]] = {}
    for baseline in baseline_names:
        grouped: Dict[str, List[EvaluatedRow]] = defaultdict(list)
        for row in evaluated[baseline]:
            grouped[row.example.scene_id].append(row)
        by_baseline_scene[baseline] = grouped

    draws: Dict[str, Dict[str, List[float]]] = {
        baseline: {metric: [] for metric in BOOTSTRAP_METRICS}
        for baseline in baseline_names
    }
    deltas: Dict[str, List[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    rng = random.Random(seed)
    for _ in range(samples):
        sampled_scenes = [rng.choice(scenes) for _ in scenes]
        sampled_metrics: Dict[str, Dict[str, Optional[float]]] = {}
        for baseline in baseline_names:
            sampled_rows = [
                row
                for scene in sampled_scenes
                for row in by_baseline_scene[baseline][scene]
            ]
            sampled_metrics[baseline] = _metric_values(sampled_rows)
            for metric in BOOTSTRAP_METRICS:
                value = sampled_metrics[baseline][metric]
                if value is not None:
                    draws[baseline][metric].append(value)
        if len(baseline_names) == 2:
            relation_value = sampled_metrics["relation_question_type_only"]
            text_value = sampled_metrics["normalized_exact_question_text_only"]
            for metric in BOOTSTRAP_METRICS:
                left, right = relation_value[metric], text_value[metric]
                if left is not None and right is not None:
                    deltas[metric].append(right - left)

    intervals: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for baseline in baseline_names:
        point = _metric_values(evaluated[baseline])
        intervals[baseline] = {
            metric: {
                "point_estimate": point[metric],
                "ci95_low": _quantile(draws[baseline][metric], 0.025),
                "ci95_high": _quantile(draws[baseline][metric], 0.975),
            }
            for metric in BOOTSTRAP_METRICS
        }
    relation_point = _metric_values(evaluated["relation_question_type_only"])
    text_point = _metric_values(evaluated["normalized_exact_question_text_only"])
    delta_intervals = {
        metric: {
            "point_estimate": (
                None
                if relation_point[metric] is None or text_point[metric] is None
                else text_point[metric] - relation_point[metric]
            ),
            "ci95_low": _quantile(deltas[metric], 0.025),
            "ci95_high": _quantile(deltas[metric], 0.975),
        }
        for metric in BOOTSTRAP_METRICS
    }
    return {
        "method": "paired nonparametric bootstrap over scene clusters",
        "confidence_level": 0.95,
        "samples": samples,
        "seed": seed,
        "baseline_intervals": intervals,
        "paired_delta_normalized_question_minus_relation": delta_intervals,
        "delta_interpretation": (
            "All bootstrapped metrics are ↑, so a positive delta favors the "
            "normalized exact-question baseline."
        ),
    }


def _entropy(labels: Sequence[str]) -> float:
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _mutual_information(first: Sequence[str], second: Sequence[str]) -> Dict[str, float]:
    if len(first) != len(second):
        raise ValueError("mutual-information inputs have different lengths")
    if not first:
        return {"mutual_information_bits_↓": 0.0, "normalized_mi_↓": 0.0}
    joint = Counter(zip(first, second))
    first_counts, second_counts = Counter(first), Counter(second)
    total = len(first)
    mi = 0.0
    for (left, right), count in joint.items():
        probability = count / total
        mi += probability * math.log2(
            (count * total) / (first_counts[left] * second_counts[right])
        )
    first_entropy, second_entropy = _entropy(first), _entropy(second)
    denominator = math.sqrt(first_entropy * second_entropy)
    return {
        "mutual_information_bits_↓": mi,
        "normalized_mi_↓": mi / denominator if denominator > 0.0 else 0.0,
        "operator_entropy_bits": first_entropy,
        "slot_entropy_bits": second_entropy,
    }


def _slot_operator_diagnostics(examples: Sequence[Example]) -> Dict[str, Any]:
    answerable = [row for row in examples if not row.no_evidence]
    operators = [row.operator for row in answerable]
    anchor_slots = [",".join(map(str, row.anchor_slots)) for row in answerable]
    answer_slots = [",".join(map(str, row.answer_slots)) for row in answerable]
    role_pairs = [
        f"anchor={','.join(map(str, row.anchor_slots))}|"
        f"answer={','.join(map(str, row.answer_slots))}"
        for row in answerable
    ]
    return {
        "interpretation": (
            "Lower MI is safer: high MI means the question operator predicts the "
            "target's coarse temporal slot without audio."
        ),
        "slot_definition": (
            f"interval-center bin on a duration-normalized {SLOT_BINS}-slot grid"
        ),
        "answerable_records": len(answerable),
        "operator_to_anchor_slot": _mutual_information(operators, anchor_slots),
        "operator_to_answer_slot": _mutual_information(operators, answer_slots),
        "operator_to_role_pair_slots": _mutual_information(operators, role_pairs),
    }


def _same_scene_diversity(examples: Sequence[Example]) -> Dict[str, Any]:
    grouped: Dict[str, List[Example]] = defaultdict(list)
    for row in examples:
        if not row.no_evidence:
            grouped[row.scene_id].append(row)
    scene_pairwise_iou: Dict[str, Optional[float]] = {}
    scene_unique_ratio: Dict[str, Optional[float]] = {}
    for scene in sorted(grouped):
        rows = grouped[scene]
        pairwise = [
            _binary_iou(rows[left].union_mask, rows[right].union_mask)
            for left in range(len(rows))
            for right in range(left + 1, len(rows))
        ]
        scene_pairwise_iou[scene] = _mean_or_none(pairwise)
        scene_unique_ratio[scene] = (
            len({row.union_mask for row in rows}) / len(rows) if rows else None
        )
    mean_iou = _mean_or_none(
        value for value in scene_pairwise_iou.values() if value is not None
    )
    unique_ratio = _mean_or_none(
        value for value in scene_unique_ratio.values() if value is not None
    )
    return {
        "interpretation": (
            "Higher diversity/unique ratio and lower pairwise IoU mean that "
            "questions in the same scene demand more distinct evidence."
        ),
        "answerable_union_pairwise_iou_↓": mean_iou,
        "answerable_union_diversity_one_minus_iou_↑": (
            None if mean_iou is None else 1.0 - mean_iou
        ),
        "answerable_unique_union_mask_ratio_↑": unique_ratio,
        "per_scene_pairwise_iou_↓": scene_pairwise_iou,
        "per_scene_unique_union_mask_ratio_↑": scene_unique_ratio,
    }


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 10)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    return value


def evaluate_manifest(
    manifest: Path, frames: int, bootstrap_samples: int, seed: int
) -> Dict[str, Any]:
    examples, schema_versions = _load_examples(manifest, frames)
    evaluated = {
        "relation_question_type_only": _evaluate_loso(
            examples, lambda row: row.operator, frames
        ),
        "normalized_exact_question_text_only": _evaluate_loso(
            examples, lambda row: row.question_key, frames
        ),
    }
    scenes = sorted({row.scene_id for row in examples})
    answerable_count = sum(not row.no_evidence for row in examples)
    report: Dict[str, Any] = {
        "format": "qces_no_audio_shortcut_audit_v1",
        "manifest": str(manifest.resolve()),
        "configuration": {
            "frames": frames,
            "mask_prototype_threshold": MASK_THRESHOLD,
            "no_evidence_prototype_threshold": NO_EVIDENCE_THRESHOLD,
            "cross_validation": "leave-one-scene-out",
            "unseen_key_policy": "training-fold global prototype",
            "question_normalization": (
                "Unicode NFKC + casefold + whitespace collapse; label tokens retained"
            ),
        },
        "dataset": {
            "schema_versions": schema_versions,
            "records": len(examples),
            "scenes": len(scenes),
            "answerable_records": answerable_count,
            "no_evidence_records": len(examples) - answerable_count,
        },
        "baselines": {
            baseline: {
                "metrics": _metric_values(rows),
                "per_scene_metrics": _per_scene_metrics(rows),
            }
            for baseline, rows in evaluated.items()
        },
        "scene_clustered_bootstrap_95ci": _scene_bootstrap(
            evaluated, bootstrap_samples, seed
        ),
        "shortcut_diagnostics": {
            "slot_operator_mutual_information": _slot_operator_diagnostics(examples),
            "same_scene_target_diversity": _same_scene_diversity(examples),
        },
        "metric_direction_legend": {
            "↑": "higher is better",
            "↓": "lower is better",
        },
    }
    return _round_floats(report)


def _format_metric(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_table(report: Mapping[str, Any]) -> None:
    print("\nNo-audio leave-one-scene-out shortcut baselines")
    print(
        "Baseline                              "
        "Union mIoU ↑  Anchor mIoU ↑  Answer mIoU ↑  No-evidence acc ↑"
    )
    for baseline in (
        "relation_question_type_only",
        "normalized_exact_question_text_only",
    ):
        metrics = report["baselines"][baseline]["metrics"]
        print(
            f"{baseline:<37}"
            f"{_format_metric(metrics[UNION_MIOU]):>12}  "
            f"{_format_metric(metrics[ANCHOR_MIOU]):>13}  "
            f"{_format_metric(metrics[ANSWER_MIOU]):>13}  "
            f"{_format_metric(metrics[NO_EVIDENCE_ACCURACY]):>17}"
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    report = evaluate_manifest(
        args.manifest, args.frames, args.bootstrap_samples, args.seed
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    _print_table(report)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
