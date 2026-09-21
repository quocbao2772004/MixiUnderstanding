#!/usr/bin/env python3
"""Train a 200-class open-event calibration/reranking layer.

This sidecar is designed for the failure observed in the 200-label probe:
expanding the label bank creates many false-positive event labels.  The method
is deliberately two-stage:

1. label-level presence calibration over the separator/proposal features;
2. inventory-aware answer reranking after filtering/merging proposals.

The script trains on an open label-bank cache and evaluates on another cache.
It does not modify the core QCES or Claude-generated code.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from mixi_understanding.qces.event_proposals import EventProposal
from mixi_understanding.scripts.evaluate_qces_open_event_v2 import (
    EvalItem,
    accuracy,
    build_eval_items,
    label_text,
    load_scene_rows,
    nearest_legacy,
    occurrence_index,
    semantic_events,
    temporal_overlap,
)
from mixi_understanding.scripts.infer_qces_open_event_qa import (
    OpenAnswer,
    OpenProgram,
    event_payload,
    load_inventory,
)


DEFAULT_PROPOSAL_HEAD = (
    PROJECT_ROOT / "outputs/qces_v6_proposal_head_iou_v2/train_val_v1/proposal_head.pt"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--eval-cache", type=Path, required=True)
    parser.add_argument("--proposal-head", type=Path, default=DEFAULT_PROPOSAL_HEAD)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decode-threshold", type=float, default=0.08)
    parser.add_argument("--max-events", type=int, default=260)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-eval-scenes", type=int, default=0)
    parser.add_argument("--max-items-per-scene", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def load_cache(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def scene_ids_for_cache(cache: Mapping[str, Any], scene_rows: Mapping[str, Any], max_scenes: int) -> list[str]:
    scene_ids = [scene_id for scene_id in cache["scenes"].keys() if scene_id in scene_rows]
    return scene_ids[:max_scenes] if max_scenes else scene_ids


def gt_labels_for_scene(row: Mapping[str, Any]) -> set[str]:
    return {
        str(event["label"])
        for event in row.get("events") or []
        if event.get("event_kind", "semantic") == "semantic"
    }


def proposals_by_label(events: Sequence[EventProposal]) -> dict[str, list[EventProposal]]:
    result: dict[str, list[EventProposal]] = defaultdict(list)
    for event in events:
        result[event.label].append(event)
    for label in result:
        result[label].sort(key=lambda item: (item.onset_seconds, item.offset_seconds, -item.confidence))
    return result


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def label_stats(
    cache: Mapping[str, Any],
    scene_id: str,
    inventory: Sequence[EventProposal],
) -> dict[str, dict[str, float]]:
    entry = cache["scenes"][scene_id]
    labels = list(entry["labels"])
    props = proposals_by_label(inventory)
    stems = entry["stems"].float()
    energy = stems[:, 0]
    clap_mix = entry.get("clap_mixture_similarity")
    clap_stem = entry.get("clap_stem_similarity")
    result: dict[str, dict[str, float]] = {}
    for index, label in enumerate(labels):
        label_props = props.get(label, [])
        confs = [event.confidence for event in label_props]
        durations = [max(event.duration_seconds, 0.0) for event in label_props]
        stem_energy = energy[index]
        if clap_mix is not None:
            mix_row = clap_mix.float()[index]
            mix_max = safe_float(mix_row.max())
            mix_mean = safe_float(mix_row.mean())
        else:
            mix_max = mix_mean = 0.0
        if clap_stem is not None:
            matrix = clap_stem.float()
            diag = safe_float(matrix[index, index])
            if matrix.shape[0] > 1:
                others = torch.cat([matrix[index, :index], matrix[index, index + 1 :]])
                margin = diag - safe_float(others.max())
            else:
                margin = 0.0
        else:
            diag = margin = 0.0
        result[label] = {
            "n_props": float(len(label_props)),
            "max_conf": max(confs) if confs else 0.0,
            "mean_conf": float(np.mean(confs)) if confs else 0.0,
            "sum_conf": float(np.sum(confs)) if confs else 0.0,
            "max_duration": max(durations) if durations else 0.0,
            "sum_duration": float(np.sum(durations)) if durations else 0.0,
            "stem_log_energy_max": safe_float(torch.log10(stem_energy.max().clamp_min(1e-8))),
            "stem_log_energy_mean": safe_float(torch.log10(stem_energy.mean().clamp_min(1e-8))),
            "clap_mix_max": mix_max,
            "clap_mix_mean": mix_mean,
            "clap_stem_diag": diag,
            "clap_stem_margin": margin,
        }
    return result


LABEL_FEATURE_KEYS = (
    "n_props",
    "max_conf",
    "mean_conf",
    "sum_conf",
    "max_duration",
    "sum_duration",
    "stem_log_energy_max",
    "stem_log_energy_mean",
    "clap_mix_max",
    "clap_mix_mean",
    "clap_stem_diag",
    "clap_stem_margin",
)


def label_feature_vector(stats: Mapping[str, float]) -> list[float]:
    return [safe_float(stats.get(key, 0.0)) for key in LABEL_FEATURE_KEYS]


def decode_all(
    cache_path: Path,
    cache: Mapping[str, Any],
    scene_ids: Sequence[str],
    proposal_head: Path,
    *,
    threshold: float,
    max_events: int,
    device: str,
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[EventProposal, ...]]]:
    labels_by_scene: dict[str, tuple[str, ...]] = {}
    events_by_scene: dict[str, tuple[EventProposal, ...]] = {}
    for index, scene_id in enumerate(scene_ids, start=1):
        labels, inventory = load_inventory(
            cache_path=cache_path,
            proposal_head_path=proposal_head,
            scene_id=scene_id,
            threshold=threshold,
            max_events=max_events,
            device_text=device,
        )
        labels_by_scene[scene_id] = labels
        events_by_scene[scene_id] = inventory
        if index % 10 == 0 or index == len(scene_ids):
            print(f"decoded {cache_path.name} scene {index}/{len(scene_ids)}", flush=True)
    return labels_by_scene, events_by_scene


def fit_presence_model(
    cache: Mapping[str, Any],
    scene_rows: Mapping[str, Mapping[str, Any]],
    scene_ids: Sequence[str],
    events_by_scene: Mapping[str, Sequence[EventProposal]],
) -> tuple[LogisticRegression, dict[str, dict[str, dict[str, float]]]]:
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    stats_by_scene: dict[str, dict[str, dict[str, float]]] = {}
    for scene_id in scene_ids:
        stats = label_stats(cache, scene_id, events_by_scene[scene_id])
        stats_by_scene[scene_id] = stats
        present = gt_labels_for_scene(scene_rows[scene_id])
        for label, row in stats.items():
            x_rows.append(label_feature_vector(row))
            y_rows.append(int(label in present))
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64))
    return model, stats_by_scene


def predict_presence(
    model: LogisticRegression,
    stats: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    labels = list(stats.keys())
    if not labels:
        return {}
    x = np.asarray([label_feature_vector(stats[label]) for label in labels], dtype=np.float32)
    probs = model.predict_proba(x)[:, 1]
    return dict(zip(labels, map(float, probs)))


def merge_label_events(events: Sequence[EventProposal], *, gap: float) -> list[EventProposal]:
    if not events:
        return []
    ordered = sorted(events, key=lambda item: (item.onset_seconds, item.offset_seconds, -item.confidence))
    merged: list[EventProposal] = []
    current = ordered[0]
    for event in ordered[1:]:
        if event.label == current.label and event.onset_seconds <= current.offset_seconds + gap:
            current = EventProposal(
                label=current.label,
                onset_seconds=min(current.onset_seconds, event.onset_seconds),
                offset_seconds=max(current.offset_seconds, event.offset_seconds),
                confidence=max(current.confidence, event.confidence),
            )
        else:
            merged.append(current)
            current = event
    merged.append(current)
    return merged


def filtered_inventory(
    events: Sequence[EventProposal],
    presence: Mapping[str, float],
    *,
    top_k: int,
    presence_thr: float,
    prop_conf_thr: float,
    merge_gap: float,
    force_labels: Sequence[str] = (),
) -> tuple[EventProposal, ...]:
    force = {label for label in force_labels if label}
    ranked_labels = {
        label
        for label, _score in sorted(presence.items(), key=lambda item: item[1], reverse=True)[:top_k]
    }
    keep_labels = force | ranked_labels | {label for label, score in presence.items() if score >= presence_thr}
    grouped: dict[str, list[EventProposal]] = defaultdict(list)
    for event in events:
        if event.label in keep_labels and (event.confidence >= prop_conf_thr or event.label in force):
            grouped[event.label].append(event)
    merged: list[EventProposal] = []
    for label_events in grouped.values():
        merged.extend(merge_label_events(label_events, gap=merge_gap))
    return tuple(sorted(merged, key=lambda item: (item.onset_seconds, item.offset_seconds, -item.confidence)))


def relation_candidate_features(
    program: OpenProgram,
    anchor: EventProposal | None,
    candidate: EventProposal,
    presence: Mapping[str, float],
    stats: Mapping[str, Mapping[str, float]],
) -> list[float]:
    row = stats.get(candidate.label, {})
    score = presence.get(candidate.label, 0.0)
    duration = max(candidate.duration_seconds, 0.0)
    op_after = float(program.operation == "after")
    op_before = float(program.operation == "before")
    op_between = float(program.operation == "between")
    if anchor is None:
        delta = 0.0
        overlap = 0.0
    else:
        delta = candidate.onset_seconds - anchor.onset_seconds
        overlap = temporal_overlap(anchor, candidate)
    return [
        candidate.confidence,
        duration,
        score,
        safe_float(row.get("n_props", 0.0)),
        safe_float(row.get("max_conf", 0.0)),
        safe_float(row.get("sum_conf", 0.0)),
        safe_float(row.get("clap_mix_max", 0.0)),
        safe_float(row.get("clap_stem_diag", 0.0)),
        safe_float(row.get("clap_stem_margin", 0.0)),
        op_after,
        op_before,
        op_between,
        delta,
        abs(delta),
        overlap,
    ]


def anchor_for_program(program: OpenProgram, events: Sequence[EventProposal]) -> EventProposal | None:
    if not program.anchor_label:
        return None
    candidates = [
        event for event in events if event.label == program.anchor_label
    ]
    candidates.sort(key=lambda item: (item.onset_seconds, item.offset_seconds, -item.confidence))
    if len(candidates) < program.anchor_ordinal:
        return None
    return candidates[program.anchor_ordinal - 1]


def relation_candidates(program: OpenProgram, events: Sequence[EventProposal]) -> tuple[EventProposal | None, list[EventProposal]]:
    anchor = anchor_for_program(program, events)
    if program.operation == "after" and anchor is not None:
        return anchor, [
            event for event in events
            if event.label != anchor.label and event.onset_seconds > anchor.onset_seconds + 1e-6
        ]
    if program.operation == "before" and anchor is not None:
        return anchor, [
            event for event in events
            if event.label != anchor.label and event.onset_seconds < anchor.onset_seconds - 1e-6
        ]
    if program.operation == "between" and anchor is not None and program.target_label:
        rights = [
            event for event in events
            if event.label == program.target_label and event.onset_seconds > anchor.onset_seconds + 1e-6
        ]
        if not rights:
            return anchor, []
        right = min(rights, key=lambda item: item.onset_seconds)
        return anchor, [
            event for event in events
            if event.label not in {anchor.label, right.label}
            and event.onset_seconds > anchor.onset_seconds + 1e-6
            and event.onset_seconds < right.onset_seconds - 1e-6
        ]
    return anchor, []


def fit_answer_model(
    items: Sequence[EvalItem],
    events_by_scene: Mapping[str, Sequence[EventProposal]],
    presence_by_scene: Mapping[str, Mapping[str, float]],
    stats_by_scene: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> HistGradientBoostingClassifier | None:
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    weights: list[float] = []
    for item in items:
        if item.family not in {"after", "before", "between"}:
            continue
        program = item.program
        events = events_by_scene[item.scene_id]
        anchor, candidates = relation_candidates(program, events)
        for candidate in candidates:
            x_rows.append(
                relation_candidate_features(
                    program,
                    anchor,
                    candidate,
                    presence_by_scene[item.scene_id],
                    stats_by_scene[item.scene_id],
                )
            )
            positive = int(candidate.label == item.expected_answer)
            y_rows.append(positive)
            weights.append(8.0 if positive else 1.0)
    if len(set(y_rows)) < 2:
        return None
    model = HistGradientBoostingClassifier(
        max_iter=160,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        random_state=2032,
    )
    model.fit(
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.int64),
        sample_weight=np.asarray(weights, dtype=np.float32),
    )
    return model


@dataclass(frozen=True)
class Config:
    top_k: int
    presence_thr: float
    prop_conf_thr: float
    merge_gap: float
    exists_thr: float


def answer_relation(
    item: EvalItem,
    events: Sequence[EventProposal],
    presence: Mapping[str, float],
    stats: Mapping[str, Mapping[str, float]],
    answer_model: HistGradientBoostingClassifier | None,
) -> OpenAnswer:
    anchor, candidates = relation_candidates(item.program, events)
    if anchor is None:
        return OpenAnswer("no_evidence", True, (), "cal_anchor_missing")
    if not candidates:
        return OpenAnswer("no_evidence", True, (anchor,), "cal_no_relation_candidate")
    if answer_model is None:
        # Conservative fallback: temporal proximity with calibrated presence.
        def score(candidate: EventProposal) -> float:
            delta = abs(candidate.onset_seconds - anchor.onset_seconds)
            return presence.get(candidate.label, 0.0) + candidate.confidence - delta - 2.0 * temporal_overlap(anchor, candidate)
        answer = max(candidates, key=score)
    else:
        x = np.asarray(
            [
                relation_candidate_features(item.program, anchor, event, presence, stats)
                for event in candidates
            ],
            dtype=np.float32,
        )
        scores = answer_model.predict_proba(x)[:, 1]
        best = int(np.argmax(scores))
        answer = candidates[best]
    return OpenAnswer(
        answer.label,
        False,
        tuple(sorted((anchor, answer), key=lambda event: event.onset_seconds)),
        "cal_relation_reranker",
    )


def execute_calibrated(
    item: EvalItem,
    raw_events: Sequence[EventProposal],
    presence: Mapping[str, float],
    stats: Mapping[str, Mapping[str, float]],
    answer_model: HistGradientBoostingClassifier | None,
    config: Config,
) -> OpenAnswer:
    program = item.program
    force_labels = [label for label in (program.anchor_label, program.target_label) if label]
    events = filtered_inventory(
        raw_events,
        presence,
        top_k=config.top_k,
        presence_thr=config.presence_thr,
        prop_conf_thr=config.prop_conf_thr,
        merge_gap=config.merge_gap,
        force_labels=force_labels,
    )
    if not events:
        return OpenAnswer("no_evidence", True, (), "cal_empty_inventory")
    if program.operation == "exists":
        score = presence.get(str(program.target_label), 0.0)
        occ = [event for event in events if event.label == program.target_label]
        yes = score >= config.exists_thr and bool(occ)
        return OpenAnswer("yes" if yes else "no", not yes, tuple(occ), "cal_exists_presence")
    if program.operation == "count":
        score = presence.get(str(program.target_label), 0.0)
        occ = [event for event in events if event.label == program.target_label]
        if score < config.exists_thr:
            occ = []
        return OpenAnswer(str(len(occ)), not bool(occ), tuple(occ), "cal_count_merged")
    if program.operation == "first_event":
        event = min(events, key=lambda item: (item.onset_seconds, -presence.get(item.label, 0.0), -item.confidence))
        return OpenAnswer(event.label, False, (event,), "cal_first")
    if program.operation == "last_event":
        event = max(events, key=lambda item: (item.onset_seconds, presence.get(item.label, 0.0), item.confidence))
        return OpenAnswer(event.label, False, (event,), "cal_last")
    if program.operation == "longest_event":
        event = max(events, key=lambda item: (item.duration_seconds * (0.5 + presence.get(item.label, 0.0)), item.confidence))
        return OpenAnswer(event.label, False, (event,), "cal_longest")
    if program.operation in {"after", "before", "between"}:
        return answer_relation(item, events, presence, stats, answer_model)
    return OpenAnswer("unsupported", True, (), "cal_unsupported")


def eval_items(
    items: Sequence[EvalItem],
    events_by_scene: Mapping[str, Sequence[EventProposal]],
    presence_by_scene: Mapping[str, Mapping[str, float]],
    stats_by_scene: Mapping[str, Mapping[str, Mapping[str, float]]],
    answer_model: HistGradientBoostingClassifier | None,
    config: Config,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        answer = execute_calibrated(
            item,
            events_by_scene[item.scene_id],
            presence_by_scene[item.scene_id],
            stats_by_scene[item.scene_id],
            answer_model,
            config,
        )
        ok = answer.answer == item.expected_answer and answer.no_evidence == item.expected_no_evidence
        rows.append(
            {
                "sample_id": item.sample_id,
                "scene_id": item.scene_id,
                "family": item.family,
                "question": item.question,
                "expected_answer": item.expected_answer,
                "expected_no_evidence": item.expected_no_evidence,
                "answer": answer.answer,
                "no_evidence": answer.no_evidence,
                "ok": ok,
                "reason": answer.reason,
                "program": asdict(item.program),
                "evidence": [event_payload(event) for event in answer.evidence],
            }
        )
    return rows


def tune_config(
    items: Sequence[EvalItem],
    events_by_scene: Mapping[str, Sequence[EventProposal]],
    presence_by_scene: Mapping[str, Mapping[str, float]],
    stats_by_scene: Mapping[str, Mapping[str, Mapping[str, float]]],
    answer_model: HistGradientBoostingClassifier | None,
) -> tuple[Config, list[dict[str, Any]]]:
    grid = [
        Config(top_k, presence_thr, prop_conf_thr, merge_gap, exists_thr)
        for top_k in (4, 6, 8, 12, 20, 32, 48)
        for presence_thr in (0.05, 0.10, 0.20, 0.35, 0.50, 0.65)
        for prop_conf_thr in (0.06, 0.10, 0.16, 0.24)
        for merge_gap in (0.10, 0.25, 0.45, 0.70)
        for exists_thr in (0.20, 0.35, 0.50, 0.65)
    ]
    best_config = grid[0]
    best_score = -1.0
    rows: list[dict[str, Any]] = []
    # Tune on at most the first 1500 generated items to keep this fast.
    tune_items = list(items[:1500])
    for index, config in enumerate(grid, start=1):
        result = eval_items(tune_items, events_by_scene, presence_by_scene, stats_by_scene, answer_model, config)
        overall = accuracy(result, "ok")
        family_acc = {}
        for family in {row["family"] for row in result}:
            subset = [row for row in result if row["family"] == family]
            family_acc[family] = accuracy(subset, "ok")
        relation_mean = float(np.mean([family_acc.get(name, 0.0) for name in ("after", "before", "between")]))
        exists_mean = float(np.mean([family_acc.get(name, 0.0) for name in ("exists_pos", "exists_neg")]))
        used_scenes = sorted({item.scene_id for item in tune_items})
        filtered_label_counts = []
        for scene_id in used_scenes:
            filtered_events = filtered_inventory(
                events_by_scene[scene_id],
                presence_by_scene[scene_id],
                top_k=config.top_k,
                presence_thr=config.presence_thr,
                prop_conf_thr=config.prop_conf_thr,
                merge_gap=config.merge_gap,
            )
            filtered_label_counts.append(len({event.label for event in filtered_events}))
        avg_filtered_labels = float(np.mean(filtered_label_counts)) if filtered_label_counts else 0.0
        compactness = float(1.0 / (1.0 + max(avg_filtered_labels, 0.0) / 12.0))
        score = 0.40 * overall + 0.30 * relation_mean + 0.20 * exists_mean + 0.10 * compactness
        row = {
            "config": asdict(config),
            "overall_acc_↑": overall,
            "relation_mean_↑": relation_mean,
            "exists_mean_↑": exists_mean,
            "avg_filtered_labels_↓": avg_filtered_labels,
            "compactness_↑": compactness,
            "score_↑": score,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_config = config
        if index % 200 == 0:
            print(f"tune {index}/{len(grid)} best_score={best_score:.4f} best={best_config}", flush=True)
    rows.sort(key=lambda item: item["score_↑"], reverse=True)
    return best_config, rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = []
    counts = Counter(row["family"] for row in rows)
    for family, count in sorted(counts.items()):
        subset = [row for row in rows if row["family"] == family]
        families.append({"family": family, "n": count, "acc_↑": accuracy(subset, "ok")})
    return {"items": len(rows), "acc_↑": accuracy(rows, "ok"), "families": families}


def inventory_summary(
    scene_rows: Mapping[str, Mapping[str, Any]],
    events_by_scene: Mapping[str, Sequence[EventProposal]],
    presence_by_scene: Mapping[str, Mapping[str, float]],
    config: Config,
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for scene_id, scene_row in scene_rows.items():
        raw_events = tuple(events_by_scene.get(scene_id, ()))
        presence = presence_by_scene.get(scene_id, {})
        filtered_events = filtered_inventory(
            raw_events,
            presence,
            top_k=config.top_k,
            presence_thr=config.presence_thr,
            prop_conf_thr=config.prop_conf_thr,
            merge_gap=config.merge_gap,
        )
        raw_labels = {event.label for event in raw_events}
        filtered_labels = {event.label for event in filtered_events}
        true_labels = gt_labels_for_scene(scene_row)
        fp_labels = filtered_labels - true_labels
        missed_labels = true_labels - filtered_labels
        rows.append(
            {
                "raw_events": float(len(raw_events)),
                "filtered_events": float(len(filtered_events)),
                "raw_labels": float(len(raw_labels)),
                "filtered_labels": float(len(filtered_labels)),
                "true_labels": float(len(true_labels)),
                "false_positive_labels": float(len(fp_labels)),
                "missed_true_labels": float(len(missed_labels)),
                "label_recall": float(len(filtered_labels & true_labels) / max(len(true_labels), 1)),
            }
        )
    if not rows:
        return {"scenes": 0}

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    return {
        "scenes": len(rows),
        "avg_raw_events": mean("raw_events"),
        "avg_filtered_events": mean("filtered_events"),
        "avg_raw_labels": mean("raw_labels"),
        "avg_filtered_labels": mean("filtered_labels"),
        "avg_true_labels": mean("true_labels"),
        "avg_false_positive_labels": mean("false_positive_labels"),
        "avg_missed_true_labels": mean("missed_true_labels"),
        "avg_label_recall_↑": mean("label_recall"),
    }


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cache = load_cache(args.train_cache.resolve())
    eval_cache = load_cache(args.eval_cache.resolve())
    train_rows_all = load_scene_rows(Path(train_cache["manifest"]).expanduser().resolve())
    eval_rows_all = load_scene_rows(Path(eval_cache["manifest"]).expanduser().resolve())
    train_scene_ids = scene_ids_for_cache(train_cache, train_rows_all, args.max_train_scenes)
    eval_scene_ids = scene_ids_for_cache(eval_cache, eval_rows_all, args.max_eval_scenes)
    train_rows = OrderedDict((scene_id, train_rows_all[scene_id]) for scene_id in train_scene_ids)
    eval_rows = OrderedDict((scene_id, eval_rows_all[scene_id]) for scene_id in eval_scene_ids)

    print(f"train scenes={len(train_scene_ids)} eval scenes={len(eval_scene_ids)}", flush=True)
    train_labels, train_events = decode_all(
        args.train_cache.resolve(), train_cache, train_scene_ids, args.proposal_head.resolve(),
        threshold=args.decode_threshold, max_events=args.max_events, device=args.device,
    )
    eval_labels, eval_events = decode_all(
        args.eval_cache.resolve(), eval_cache, eval_scene_ids, args.proposal_head.resolve(),
        threshold=args.decode_threshold, max_events=args.max_events, device=args.device,
    )

    presence_model, train_stats = fit_presence_model(train_cache, train_rows, train_scene_ids, train_events)
    eval_stats = {
        scene_id: label_stats(eval_cache, scene_id, eval_events[scene_id])
        for scene_id in eval_scene_ids
    }
    train_presence = {
        scene_id: predict_presence(presence_model, train_stats[scene_id])
        for scene_id in train_scene_ids
    }
    eval_presence = {
        scene_id: predict_presence(presence_model, eval_stats[scene_id])
        for scene_id in eval_scene_ids
    }

    train_items = build_eval_items(train_rows, train_labels, max_items_per_scene=args.max_items_per_scene)
    eval_items_list = build_eval_items(eval_rows, eval_labels, max_items_per_scene=args.max_items_per_scene)
    print(f"train items={len(train_items)} eval items={len(eval_items_list)}", flush=True)

    # Build answer reranker on lightly merged, high-recall proposals.
    provisional_config = Config(top_k=80, presence_thr=0.02, prop_conf_thr=0.04, merge_gap=0.25, exists_thr=0.20)
    train_events_for_ranker = {
        scene_id: filtered_inventory(
            train_events[scene_id],
            train_presence[scene_id],
            top_k=provisional_config.top_k,
            presence_thr=provisional_config.presence_thr,
            prop_conf_thr=provisional_config.prop_conf_thr,
            merge_gap=provisional_config.merge_gap,
        )
        for scene_id in train_scene_ids
    }
    answer_model = fit_answer_model(train_items, train_events_for_ranker, train_presence, train_stats)
    best_config, tuning_rows = tune_config(
        train_items,
        train_events,
        train_presence,
        train_stats,
        answer_model,
    )
    print(f"best config={best_config}", flush=True)

    train_pred = eval_items(train_items, train_events, train_presence, train_stats, answer_model, best_config)
    eval_pred = eval_items(eval_items_list, eval_events, eval_presence, eval_stats, answer_model, best_config)
    train_summary = summarize(train_pred)
    eval_summary = summarize(eval_pred)
    train_inventory_summary = inventory_summary(train_rows, train_events, train_presence, best_config)
    eval_inventory_summary = inventory_summary(eval_rows, eval_events, eval_presence, best_config)
    summary = {
        "format": "qces_open_event_v3_calibrated",
        "train_cache": str(args.train_cache.resolve()),
        "eval_cache": str(args.eval_cache.resolve()),
        "proposal_head": str(args.proposal_head.resolve()),
        "decode_threshold": args.decode_threshold,
        "max_events": args.max_events,
        "train_scenes": len(train_scene_ids),
        "eval_scenes": len(eval_scene_ids),
        "train_items": len(train_items),
        "eval_items": len(eval_items_list),
        "train_unique_labels": len({label for labels in train_labels.values() for label in labels}),
        "eval_unique_labels": len({label for labels in eval_labels.values() for label in labels}),
        "best_config": asdict(best_config),
        "train": train_summary,
        "eval": eval_summary,
        "train_inventory": train_inventory_summary,
        "eval_inventory": eval_inventory_summary,
    }
    write_jsonl(output_dir / "train_predictions.jsonl", train_pred)
    write_jsonl(output_dir / "eval_predictions.jsonl", eval_pred)
    (output_dir / "tuning_grid_top.json").write_text(
        json.dumps(tuning_rows[:50], ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "models.pkl").open("wb") as handle:
        pickle.dump(
            {
                "presence_model": presence_model,
                "answer_model": answer_model,
                "label_feature_keys": LABEL_FEATURE_KEYS,
                "best_config": best_config,
            },
            handle,
        )
    lines = [
        "# QCES open-event v3 calibrated 200-class experiment",
        "",
        f"- train scenes: {summary['train_scenes']}",
        f"- eval scenes: {summary['eval_scenes']}",
        f"- train unique labels: {summary['train_unique_labels']}",
        f"- eval unique labels: {summary['eval_unique_labels']}",
        f"- best config: `{summary['best_config']}`",
        "",
        "| split | items | accuracy ↑ |",
        "|---|---:|---:|",
        f"| train | {train_summary['items']} | {train_summary['acc_↑']:.3f} |",
        f"| eval | {eval_summary['items']} | {eval_summary['acc_↑']:.3f} |",
        "",
        "## Inventory filtering",
        "",
        "| split | avg labels kept ↓ | avg false labels ↓ | avg missed true labels ↓ | label recall ↑ |",
        "|---|---:|---:|---:|---:|",
        (
            f"| train | {train_inventory_summary['avg_filtered_labels']:.2f} | "
            f"{train_inventory_summary['avg_false_positive_labels']:.2f} | "
            f"{train_inventory_summary['avg_missed_true_labels']:.2f} | "
            f"{train_inventory_summary['avg_label_recall_↑']:.3f} |"
        ),
        (
            f"| eval | {eval_inventory_summary['avg_filtered_labels']:.2f} | "
            f"{eval_inventory_summary['avg_false_positive_labels']:.2f} | "
            f"{eval_inventory_summary['avg_missed_true_labels']:.2f} | "
            f"{eval_inventory_summary['avg_label_recall_↑']:.3f} |"
        ),
        "",
        "## Eval by family",
        "",
        "| family | n | acc ↑ |",
        "|---|---:|---:|",
    ]
    for row in eval_summary["families"]:
        lines.append(f"| {row['family']} | {row['n']} | {row['acc_↑']:.3f} |")
    lines.append("")
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
