"""Acoustic-cleanliness audit and source planning for AudioSet Strong.

This module deliberately stays independent from detector training.  It uses
only official strong timestamps, the fixed ontology and materialization
metadata.  No QA answer, model prediction or downstream metric participates in
source selection.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from mixi_understanding.qces.supported_ontology import StrongEvent


FORMAT = "qces_acoustic_cleanliness_v1"
DEFAULT_CLIP_DURATION_SECONDS = 10.0

TIER_NAMES = {
    0: "fully_isolated_all_strong_labels",
    1: "selected_ontology_isolated",
    2: "low_selected_distractor_overlap",
    3: "selected_distractor_ambiguous",
}


@dataclass(frozen=True)
class MaterializedAudio:
    """Availability metadata for one already-materialized AudioSet video."""

    video_id: str
    metadata_split: str
    audio_path: str
    audio_exists: bool
    protocol_splits: tuple[str, ...] = ()
    audio_sha256: str = ""


def _stable_hash(seed: int, *values: Any) -> str:
    payload = ":".join([FORMAT, str(seed), *(str(value) for value in values)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def interval_union_length(intervals: Iterable[tuple[float, float]]) -> float:
    """Return union length, ignoring empty intervals."""

    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not ordered:
        return 0.0
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def build_hierarchy_ancestors(
    hierarchy_descendants: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    ancestors: dict[str, set[str]] = defaultdict(set)
    for ancestor, descendants in hierarchy_descendants.items():
        for descendant in descendants:
            ancestors[descendant].add(ancestor)
    return dict(ancestors)


def _intersection(target: StrongEvent, other: StrongEvent) -> tuple[float, float] | None:
    start = max(target.onset_seconds, other.onset_seconds)
    end = min(target.offset_seconds, other.offset_seconds)
    return (start, end) if end > start else None


def _nearest_margins(
    target: StrongEvent,
    others: Sequence[StrongEvent],
) -> tuple[float | None, float | None]:
    """Distance to nearest different event on each side; ``None`` means clear."""

    if any(_intersection(target, other) is not None for other in others):
        return 0.0, 0.0
    prior_offsets = [
        other.offset_seconds
        for other in others
        if other.offset_seconds <= target.onset_seconds
    ]
    later_onsets = [
        other.onset_seconds
        for other in others
        if other.onset_seconds >= target.offset_seconds
    ]
    left = (
        target.onset_seconds - max(prior_offsets)
        if prior_offsets
        else None
    )
    right = (
        min(later_onsets) - target.offset_seconds
        if later_onsets
        else None
    )
    return left, right


def _effective_min_margin(
    left: float | None,
    right: float | None,
    *,
    clear_value: float,
) -> float:
    return min(
        clear_value if left is None else left,
        clear_value if right is None else right,
    )


def _tier(
    *,
    any_overlap_fraction: float,
    selected_overlap_fraction: float,
    effective_margin_seconds: float,
    minimum_isolation_margin_seconds: float,
    maximum_low_overlap_fraction: float,
) -> int:
    tolerance = 1e-12
    if (
        any_overlap_fraction <= tolerance
        and effective_margin_seconds + tolerance >= minimum_isolation_margin_seconds
    ):
        return 0
    # For a fixed 200-way detector, an overlapping AudioSet parent or another
    # unselected label is acoustic context, not a competing target.  Preserve
    # the broad overlap audit, but prioritize absence of *selected* distractors.
    if selected_overlap_fraction <= tolerance:
        return 1
    if selected_overlap_fraction <= maximum_low_overlap_fraction + tolerance:
        return 2
    return 3


def audit_video_events(
    *,
    metadata_split: str,
    video_id: str,
    events: Sequence[StrongEvent],
    selected_mids: set[str],
    hierarchy_descendants: Mapping[str, set[str]] | None = None,
    hierarchy_ancestors: Mapping[str, set[str]] | None = None,
    materialized: MaterializedAudio | None = None,
    minimum_isolation_margin_seconds: float = 0.25,
    maximum_low_overlap_fraction: float = 0.10,
    clip_duration_seconds: float = DEFAULT_CLIP_DURATION_SECONDS,
) -> list[dict[str, Any]]:
    """Audit every selected-label event in one strongly annotated video.

    Same-label repetitions are intentionally not acoustic distractors.  Every
    *different* MID is counted by the broad overlap measure, while a separate
    measure counts only distractors that are also in the selected ontology.
    """

    if minimum_isolation_margin_seconds < 0:
        raise ValueError("minimum_isolation_margin_seconds must be non-negative")
    if not 0 <= maximum_low_overlap_fraction <= 1:
        raise ValueError("maximum_low_overlap_fraction must be in [0, 1]")
    hierarchy_descendants = hierarchy_descendants or {}
    ancestors = hierarchy_ancestors or build_hierarchy_ancestors(hierarchy_descendants)

    ordered = sorted(
        events,
        key=lambda event: (
            event.onset_seconds,
            event.offset_seconds,
            event.mid,
            event.segment_id,
        ),
    )
    rows: list[dict[str, Any]] = []
    for event_index, target in enumerate(ordered):
        if target.mid not in selected_mids:
            continue
        different = [event for event in ordered if event.mid != target.mid]
        selected_distractors = [
            event for event in different if event.mid in selected_mids
        ]
        related_mids = set(hierarchy_descendants.get(target.mid, set())) | set(
            ancestors.get(target.mid, set())
        )
        hierarchy_related = [
            event for event in different if event.mid in related_mids
        ]

        def overlap_payload(pool: Sequence[StrongEvent]) -> tuple[float, float, list[str]]:
            intersections = [
                intersection
                for other in pool
                if (intersection := _intersection(target, other)) is not None
            ]
            seconds = interval_union_length(intersections)
            fraction = seconds / max(target.duration_seconds, 1e-12)
            mids = sorted(
                {
                    other.mid
                    for other in pool
                    if _intersection(target, other) is not None
                }
            )
            return seconds, min(1.0, fraction), mids

        any_seconds, any_fraction, any_mids = overlap_payload(different)
        selected_seconds, selected_fraction, selected_overlap_mids = overlap_payload(
            selected_distractors
        )
        related_seconds, related_fraction, related_overlap_mids = overlap_payload(
            hierarchy_related
        )
        left_margin, right_margin = _nearest_margins(target, different)
        selected_left, selected_right = _nearest_margins(target, selected_distractors)
        effective_margin = _effective_min_margin(
            left_margin,
            right_margin,
            clear_value=clip_duration_seconds,
        )
        tier_id = _tier(
            any_overlap_fraction=any_fraction,
            selected_overlap_fraction=selected_fraction,
            effective_margin_seconds=effective_margin,
            minimum_isolation_margin_seconds=minimum_isolation_margin_seconds,
            maximum_low_overlap_fraction=maximum_low_overlap_fraction,
        )
        uid = _stable_hash(
            0,
            metadata_split,
            target.segment_id,
            target.mid,
            f"{target.onset_seconds:.6f}",
            f"{target.offset_seconds:.6f}",
            event_index,
        )[:24]
        rows.append(
            {
                "format": FORMAT,
                "metadata_split": metadata_split,
                "video_id": video_id,
                "segment_id": target.segment_id,
                "event_uid": uid,
                "event_index": event_index,
                "mid": target.mid,
                "label": target.label,
                "display_name": target.display_name,
                "onset_seconds": target.onset_seconds,
                "offset_seconds": target.offset_seconds,
                "duration_seconds": target.duration_seconds,
                "any_different_overlap_seconds": any_seconds,
                "any_different_overlap_fraction": any_fraction,
                "selected_distractor_overlap_seconds": selected_seconds,
                "selected_distractor_overlap_fraction": selected_fraction,
                "hierarchy_related_overlap_seconds": related_seconds,
                "hierarchy_related_overlap_fraction": related_fraction,
                "overlapping_different_mids": any_mids,
                "overlapping_different_mid_count": len(any_mids),
                "overlapping_selected_distractor_mids": selected_overlap_mids,
                "overlapping_selected_distractor_mid_count": len(
                    selected_overlap_mids
                ),
                "overlapping_hierarchy_related_mids": related_overlap_mids,
                "left_isolation_margin_seconds": left_margin,
                "right_isolation_margin_seconds": right_margin,
                "effective_min_isolation_margin_seconds": effective_margin,
                "selected_left_isolation_margin_seconds": selected_left,
                "selected_right_isolation_margin_seconds": selected_right,
                "fully_isolated": tier_id == 0,
                "selected_ontology_isolated": selected_fraction <= 1e-12,
                "clean_source_eligible": tier_id <= 2,
                "ambiguity_tier": tier_id,
                "ambiguity_tier_name": TIER_NAMES[tier_id],
                "materialized": materialized is not None,
                "materialized_audio_exists": bool(
                    materialized and materialized.audio_exists
                ),
                "materialized_audio_path": (
                    materialized.audio_path if materialized else ""
                ),
                "materialized_protocol_splits": (
                    list(materialized.protocol_splits) if materialized else []
                ),
                "clip_duration_seconds": clip_duration_seconds,
            }
        )
    return rows


def candidate_sort_key(row: Mapping[str, Any], *, seed: int) -> tuple[Any, ...]:
    """Stable quality order: acoustic tier first, availability only within tier."""

    return (
        int(row["ambiguity_tier"]),
        not bool(row.get("materialized_audio_exists")),
        float(row["selected_distractor_overlap_fraction"]),
        float(row["any_different_overlap_fraction"]),
        -float(row["effective_min_isolation_margin_seconds"]),
        -float(row["duration_seconds"]),
        _stable_hash(seed, row["metadata_split"], row["video_id"], row["event_uid"]),
    )


def best_event_per_label_video(
    rows: Iterable[Mapping[str, Any]], *, seed: int
) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Keep one deterministic event for each split/label/video support unit."""

    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for source in rows:
        row = dict(source)
        key = (str(row["metadata_split"]), str(row["label"]), str(row["video_id"]))
        current = result.get(key)
        if current is None or candidate_sort_key(row, seed=seed) < candidate_sort_key(
            current, seed=seed
        ):
            result[key] = row
    return result


def _crop_annotations(
    *,
    events: Sequence[StrongEvent],
    crop_start: float,
    crop_end: float,
    selected_mids: set[str],
) -> list[dict[str, Any]]:
    annotations = []
    for event in sorted(
        events,
        key=lambda value: (
            value.onset_seconds,
            value.offset_seconds,
            value.mid,
            value.segment_id,
        ),
    ):
        if event.offset_seconds <= crop_start or event.onset_seconds >= crop_end:
            continue
        annotations.append(
            {
                "segment_id": event.segment_id,
                "mid": event.mid,
                "label": event.label,
                "display_name": event.display_name,
                "selected_ontology_label": event.mid in selected_mids,
                "source_onset_seconds": event.onset_seconds,
                "source_offset_seconds": event.offset_seconds,
                "crop_onset_seconds": max(0.0, event.onset_seconds - crop_start),
                "crop_offset_seconds": min(crop_end, event.offset_seconds) - crop_start,
            }
        )
    return annotations


def build_crop_source_plan(
    *,
    best_candidates: Mapping[tuple[str, str, str], Mapping[str, Any]],
    events_by_split: Mapping[str, Mapping[str, Sequence[StrongEvent]]],
    selected_mids: set[str],
    target_train_videos_per_class: int = 100,
    target_eval_videos_per_class: int = 20,
    crop_padding_seconds: float = 0.25,
    seed: int = 2028,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a deterministic per-class source plan without downstream leakage.

    The ``coverage_label`` is used only to satisfy class support.  Every emitted
    crop carries *all* intersecting official strong annotations; consequently a
    multi-label crop is never silently converted into a single-target sample.
    """

    if target_train_videos_per_class <= 0 or target_eval_videos_per_class <= 0:
        raise ValueError("per-class targets must be positive")
    if crop_padding_seconds < 0:
        raise ValueError("crop_padding_seconds must be non-negative")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for (split, label, _), row in best_candidates.items():
        grouped[(split, label)].append(row)

    result: list[dict[str, Any]] = []
    counts_by_tier: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    fallback_classes: dict[str, set[str]] = defaultdict(set)
    for (split, label), candidates in sorted(grouped.items()):
        target = (
            target_train_videos_per_class
            if split == "train"
            else target_eval_videos_per_class
        )
        for rank, source in enumerate(
            sorted(candidates, key=lambda row: candidate_sort_key(row, seed=seed))[:target],
            start=1,
        ):
            row = dict(source)
            left = row.get("left_isolation_margin_seconds")
            right = row.get("right_isolation_margin_seconds")
            left_padding = crop_padding_seconds if left is None else min(
                crop_padding_seconds, max(0.0, float(left))
            )
            right_padding = crop_padding_seconds if right is None else min(
                crop_padding_seconds, max(0.0, float(right))
            )
            clip_duration = float(row.get("clip_duration_seconds", 10.0))
            crop_start = max(0.0, float(row["onset_seconds"]) - left_padding)
            crop_end = min(clip_duration, float(row["offset_seconds"]) + right_padding)
            annotations = _crop_annotations(
                events=events_by_split[split][str(row["video_id"])],
                crop_start=crop_start,
                crop_end=crop_end,
                selected_mids=selected_mids,
            )
            selected_annotations = [
                annotation
                for annotation in annotations
                if annotation["selected_ontology_label"]
            ]
            tier = int(row["ambiguity_tier"])
            counts_by_tier[split][TIER_NAMES[tier]] += 1
            if tier > 2:
                fallback_classes[split].add(label)
            protocol_splits = list(row.get("materialized_protocol_splits") or ())
            if split == "eval":
                split_lock = "test"
            elif protocol_splits:
                split_lock = "+".join(sorted(set(protocol_splits)))
            else:
                split_lock = "unassigned_train_pool"
            result.append(
                {
                    "format": FORMAT,
                    "metadata_split": split,
                    "video_id": row["video_id"],
                    "segment_id": row["segment_id"],
                    "source_event_uid": row["event_uid"],
                    "coverage_mid": row["mid"],
                    "coverage_label": label,
                    "coverage_rank": rank,
                    "ambiguity_tier": tier,
                    "ambiguity_tier_name": TIER_NAMES[tier],
                    "fully_isolated": bool(row["fully_isolated"]),
                    "clean_source_eligible": bool(row["clean_source_eligible"]),
                    "any_different_overlap_fraction": row[
                        "any_different_overlap_fraction"
                    ],
                    "selected_distractor_overlap_fraction": row[
                        "selected_distractor_overlap_fraction"
                    ],
                    "event_onset_seconds": row["onset_seconds"],
                    "event_offset_seconds": row["offset_seconds"],
                    "crop_start_seconds": crop_start,
                    "crop_end_seconds": crop_end,
                    "materialized": bool(row.get("materialized")),
                    "materialized_audio_exists": bool(
                        row.get("materialized_audio_exists")
                    ),
                    "materialized_audio_path": row.get("materialized_audio_path", ""),
                    "split_lock": split_lock,
                    "strong_annotations": annotations,
                    "selected_ontology_annotations": selected_annotations,
                    "supervision_policy": (
                        "complete_intersecting_strong_annotations;"
                        "coverage_label_is_planning_only;no_single_target_assumption"
                    ),
                    "selection_key": _stable_hash(
                        seed, split, label, row["video_id"], row["event_uid"]
                    ),
                }
            )

    summary = {
        "planned_rows": len(result),
        "planned_unique_videos": len(
            {(row["metadata_split"], row["video_id"]) for row in result}
        ),
        "materialized_rows": sum(row["materialized_audio_exists"] for row in result),
        "materialized_unique_videos": len(
            {
                (row["metadata_split"], row["video_id"])
                for row in result
                if row["materialized_audio_exists"]
            }
        ),
        "ambiguity_tiers": {
            split: dict(sorted(values.items()))
            for split, values in sorted(counts_by_tier.items())
        },
        "classes_requiring_selected_distractor_or_ambiguous_fallback": {
            split: len(labels) for split, labels in sorted(fallback_classes.items())
        },
        "leakage_controls": {
            "uses_qa_questions_or_answers": False,
            "uses_model_predictions_or_scores": False,
            "uses_downstream_qa_target_label": False,
            "uses_detector_target_performance": False,
            "coverage_label_used_only_for_support_quota": True,
            "all_intersecting_strong_annotations_emitted": True,
            "single_label_crop_assumption": False,
            "official_eval_locked_to_test": True,
        },
    }
    return result, summary


def summarize_class_support(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    best_candidates: Mapping[tuple[str, str, str], Mapping[str, Any]],
    train_target: int = 100,
    eval_target: int = 20,
) -> list[dict[str, Any]]:
    """Report raw, fully isolated and low-overlap unique-video support."""

    by_label_split: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for (split, label, _), row in best_candidates.items():
        by_label_split[(label, split)].append(row)
    output: list[dict[str, Any]] = []
    for selected in selected_rows:
        label = str(selected["label"])
        row: dict[str, Any] = {
            "mid": str(selected["mid"]),
            "label": label,
            "display_name": str(selected.get("display_name") or label),
        }
        for split in ("train", "eval"):
            candidates = by_label_split.get((label, split), [])
            row[f"raw_{split}_unique_videos"] = len(candidates)
            row[f"isolated_{split}_unique_videos"] = sum(
                int(candidate["ambiguity_tier"]) == 0 for candidate in candidates
            )
            row[f"nonoverlap_{split}_unique_videos"] = sum(
                float(candidate["any_different_overlap_fraction"]) <= 1e-12
                for candidate in candidates
            )
            row[f"selected_isolated_{split}_unique_videos"] = sum(
                bool(candidate["selected_ontology_isolated"])
                for candidate in candidates
            )
            row[f"clean_{split}_unique_videos"] = sum(
                int(candidate["ambiguity_tier"]) <= 2 for candidate in candidates
            )
            row[f"materialized_{split}_unique_videos"] = sum(
                bool(candidate.get("materialized_audio_exists"))
                for candidate in candidates
            )
            row[f"materialized_selected_isolated_{split}_unique_videos"] = sum(
                bool(candidate.get("materialized_audio_exists"))
                and bool(candidate["selected_ontology_isolated"])
                for candidate in candidates
            )
            row[f"materialized_clean_{split}_unique_videos"] = sum(
                bool(candidate.get("materialized_audio_exists"))
                and int(candidate["ambiguity_tier"]) <= 2
                for candidate in candidates
            )
            row[f"isolated_{split}_seconds"] = sum(
                float(candidate["duration_seconds"])
                for candidate in candidates
                if int(candidate["ambiguity_tier"]) == 0
            )
        row["isolated_train_100_feasible"] = (
            row["isolated_train_unique_videos"] >= train_target
        )
        row["isolated_eval_20_feasible"] = (
            row["isolated_eval_unique_videos"] >= eval_target
        )
        row["isolated_100_20_feasible"] = (
            row["isolated_train_100_feasible"]
            and row["isolated_eval_20_feasible"]
        )
        row["selected_isolated_train_100_feasible"] = (
            row["selected_isolated_train_unique_videos"] >= train_target
        )
        row["selected_isolated_eval_20_feasible"] = (
            row["selected_isolated_eval_unique_videos"] >= eval_target
        )
        row["selected_isolated_100_20_feasible"] = (
            row["selected_isolated_train_100_feasible"]
            and row["selected_isolated_eval_20_feasible"]
        )
        row["clean_train_100_feasible"] = row["clean_train_unique_videos"] >= train_target
        row["clean_eval_20_feasible"] = row["clean_eval_unique_videos"] >= eval_target
        row["clean_100_20_feasible"] = (
            row["clean_train_100_feasible"] and row["clean_eval_20_feasible"]
        )
        row["materialized_selected_isolated_100_20_ready"] = (
            row["materialized_selected_isolated_train_unique_videos"] >= train_target
            and row["materialized_selected_isolated_eval_unique_videos"] >= eval_target
        )
        row["materialized_clean_100_20_ready"] = (
            row["materialized_clean_train_unique_videos"] >= train_target
            and row["materialized_clean_eval_unique_videos"] >= eval_target
        )
        output.append(row)
    return output


def load_materialized_audio(
    manifest_paths: Sequence[Path], *, project_root: Path
) -> dict[tuple[str, str], MaterializedAudio]:
    """Merge availability from existing manifests without changing split locks."""

    aggregate: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest_path in sorted({path.resolve() for path in manifest_paths if path.is_file()}):
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                video_id = str(row.get("video_id") or "").strip()
                if not video_id:
                    continue
                upstream = str(
                    row.get("protocol_upstream_split")
                    or row.get("hf_split")
                    or row.get("metadata_split")
                    or ""
                ).lower()
                metadata_split = "eval" if upstream in {"eval", "test"} else "train"
                audio_path = str(row.get("mixture_path") or row.get("audio_path") or "")
                resolved = Path(audio_path)
                if audio_path and not resolved.is_absolute():
                    resolved = project_root / resolved
                exists = bool(audio_path and resolved.is_file())
                key = (metadata_split, video_id)
                target = aggregate.setdefault(
                    key,
                    {
                        "paths": set(),
                        "existing_paths": set(),
                        "protocol_splits": set(),
                        "sha256": set(),
                    },
                )
                if audio_path:
                    target["paths"].add(audio_path)
                    if exists:
                        target["existing_paths"].add(audio_path)
                protocol_split = str(row.get("protocol_split") or row.get("split") or "")
                if protocol_split:
                    target["protocol_splits"].add(protocol_split)
                if row.get("audio_sha256"):
                    target["sha256"].add(str(row["audio_sha256"]))

    output: dict[tuple[str, str], MaterializedAudio] = {}
    for (metadata_split, video_id), row in sorted(aggregate.items()):
        candidates = sorted(row["existing_paths"] or row["paths"])
        output[(metadata_split, video_id)] = MaterializedAudio(
            video_id=video_id,
            metadata_split=metadata_split,
            audio_path=candidates[0] if candidates else "",
            audio_exists=bool(row["existing_paths"]),
            protocol_splits=tuple(sorted(row["protocol_splits"])),
            audio_sha256=(sorted(row["sha256"])[0] if row["sha256"] else ""),
        )
    return output


def selected_rows_from_tsv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    selected = [row for row in rows if not row.get("status") or row["status"] == "selected"]
    if not selected or not all(row.get("mid") and row.get("label") for row in selected):
        raise ValueError(f"invalid selected ontology TSV: {path}")
    labels = [row["label"] for row in selected]
    mids = [row["mid"] for row in selected]
    if len(labels) != len(set(labels)) or len(mids) != len(set(mids)):
        raise ValueError("ontology labels and MIDs must be unique")
    return selected


def support_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def distribution(field: str) -> dict[str, float | int]:
        values = sorted(float(row[field]) for row in rows)
        return {
            "min": values[0] if values else 0,
            "median": values[len(values) // 2] if values else 0,
            "max": values[-1] if values else 0,
        }

    return {
        "selected_classes": len(rows),
        "classes_with_isolated_train_100": sum(
            bool(row["isolated_train_100_feasible"]) for row in rows
        ),
        "classes_with_isolated_eval_20": sum(
            bool(row["isolated_eval_20_feasible"]) for row in rows
        ),
        "classes_with_isolated_100_20": sum(
            bool(row["isolated_100_20_feasible"]) for row in rows
        ),
        "classes_with_selected_isolated_train_100": sum(
            bool(row["selected_isolated_train_100_feasible"]) for row in rows
        ),
        "classes_with_selected_isolated_eval_20": sum(
            bool(row["selected_isolated_eval_20_feasible"]) for row in rows
        ),
        "classes_with_selected_isolated_100_20": sum(
            bool(row["selected_isolated_100_20_feasible"]) for row in rows
        ),
        "classes_with_clean_100_20": sum(
            bool(row["clean_100_20_feasible"]) for row in rows
        ),
        "classes_materialized_selected_isolated_100_20_ready": sum(
            bool(row["materialized_selected_isolated_100_20_ready"]) for row in rows
        ),
        "classes_materialized_clean_100_20_ready": sum(
            bool(row["materialized_clean_100_20_ready"]) for row in rows
        ),
        "isolated_train_unique_videos": distribution("isolated_train_unique_videos"),
        "isolated_eval_unique_videos": distribution("isolated_eval_unique_videos"),
        "selected_isolated_train_unique_videos": distribution(
            "selected_isolated_train_unique_videos"
        ),
        "selected_isolated_eval_unique_videos": distribution(
            "selected_isolated_eval_unique_videos"
        ),
        "clean_train_unique_videos": distribution("clean_train_unique_videos"),
        "clean_eval_unique_videos": distribution("clean_eval_unique_videos"),
        "materialized_selected_isolated_train_unique_videos": distribution(
            "materialized_selected_isolated_train_unique_videos"
        ),
        "materialized_selected_isolated_eval_unique_videos": distribution(
            "materialized_selected_isolated_eval_unique_videos"
        ),
    }


def event_json(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
