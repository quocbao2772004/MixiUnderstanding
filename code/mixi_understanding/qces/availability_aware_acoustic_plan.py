"""Availability-aware AudioSet-Strong crop-contract selection.

The original acoustic-cleanliness audit ranks every official strong event,
including clips that may no longer exist in a particular audio mirror.  This
sidecar joins those acoustic candidates to a pinned, metadata-only availability
index *before* applying the 100/20 per-class quota.  It therefore never spends
quota on a metadata-only source and never uses detector/QA outcomes to choose
data.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping, Sequence

from mixi_understanding.qces.acoustic_cleanliness import (
    TIER_NAMES,
    best_event_per_label_video,
    build_crop_source_plan,
    candidate_sort_key,
)
from mixi_understanding.qces.supported_ontology import StrongEvent


FORMAT = "qces_availability_aware_acoustic_plan_v1"


class AvailabilityAwarePlanError(ValueError):
    """Raised when availability cannot be joined without ambiguity/leakage."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def metadata_split_from_hf_split(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"train", "bal_train", "unbal_train"}:
        return "train"
    if normalized in {"test", "eval", "evaluation"}:
        return "eval"
    raise AvailabilityAwarePlanError(
        f"cannot map availability hf_split={value!r} to official strong split"
    )


def _normalized_labels(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return tuple(sorted({str(item) for item in values if str(item)}))


def _location_identity(entry: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(entry.get("source_route") or ""),
        str(entry.get("parquet_url") or ""),
        int(entry.get("row_group", -1)),
        int(entry.get("row_index", -1)),
    )


def normalize_availability_entries(
    entries: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Validate and group physical mirror locations by official split/video."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    physical_locations: set[tuple[str, str, int, int]] = set()
    route_video_locations: set[tuple[str, str, str, str, int, int]] = set()
    for raw in entries:
        row = dict(raw)
        video_id = str(row.get("video_id") or "")
        source_route = str(row.get("source_route") or "")
        hf_split = str(row.get("hf_split") or "")
        revision = str(row.get("hf_revision") or "")
        dataset = str(row.get("hf_dataset") or "")
        parquet_url = str(row.get("parquet_url") or "")
        row_group = int(row.get("row_group", -1))
        row_index = int(row.get("row_index", -1))
        if (
            not video_id
            or not source_route
            or not revision
            or not dataset
            or not parquet_url
            or row_group < 0
            or row_index < 0
        ):
            raise AvailabilityAwarePlanError(
                f"incomplete availability entry for video_id={video_id!r}"
            )
        split = metadata_split_from_hf_split(hf_split)
        physical = _location_identity(row)
        if physical in physical_locations:
            raise AvailabilityAwarePlanError(
                f"duplicate physical availability location: {physical}"
            )
        physical_locations.add(physical)
        route_video = (
            split,
            video_id,
            source_route,
            parquet_url,
            row_group,
            row_index,
        )
        if route_video in route_video_locations:
            raise AvailabilityAwarePlanError(
                f"duplicate route/video availability entry: {route_video}"
            )
        route_video_locations.add(route_video)
        row["labels"] = list(_normalized_labels(row.get("labels")))
        row["human_labels"] = list(_normalized_labels(row.get("human_labels")))
        row["metadata_split"] = split
        grouped[(split, video_id)].append(row)
    for key, values in grouped.items():
        values.sort(key=_location_identity)
        split, video_id = key
        if split not in {"train", "eval"} or not video_id:
            raise AvailabilityAwarePlanError(f"invalid availability key: {key}")
    return grouped


def _route_rank(source_route: str, route_priority: Sequence[str]) -> tuple[int, str]:
    try:
        return route_priority.index(source_route), source_route
    except ValueError:
        return len(route_priority), source_route


def choose_location_for_mid(
    locations: Sequence[Mapping[str, Any]],
    *,
    mid: str,
    route_priority: Sequence[str],
    require_label_identity: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Choose one deterministic physical location and audit label identity."""

    label_verified = [row for row in locations if mid in set(row.get("labels") or [])]
    labels_missing = [row for row in locations if not row.get("labels")]
    label_mismatched = [
        row
        for row in locations
        if row.get("labels") and mid not in set(row.get("labels") or [])
    ]
    eligible = label_verified
    if not require_label_identity:
        eligible = [*label_verified, *labels_missing, *label_mismatched]
    if not eligible:
        return None, {
            "physical_locations": len(locations),
            "label_verified_locations": len(label_verified),
            "label_missing_locations": len(labels_missing),
            "label_mismatched_locations": len(label_mismatched),
        }
    chosen = min(
        eligible,
        key=lambda row: (
            0 if mid in set(row.get("labels") or []) else 1,
            _route_rank(str(row["source_route"]), route_priority),
            str(row["parquet_url"]),
            int(row["row_group"]),
            int(row["row_index"]),
        ),
    )
    return dict(chosen), {
        "physical_locations": len(locations),
        "label_verified_locations": len(label_verified),
        "label_missing_locations": len(labels_missing),
        "label_mismatched_locations": len(label_mismatched),
    }


def choose_canonical_location_for_video(
    locations: Sequence[Mapping[str, Any]],
    *,
    requested_mids: set[str],
    route_priority: Sequence[str],
    require_label_identity: bool,
) -> tuple[dict[str, Any] | None, set[str], dict[str, Any]]:
    """Choose one physical row for every candidate derived from one video.

    Mirrors can expose different weak-label sets for the same AudioSet ID.  A
    joint materialization plan cannot fetch one label from one route and a
    second label from another route while claiming one source row.  Prefer a
    route covering the most requested MIDs, then the configured route order;
    candidates not verified by that canonical row are excluded.
    """

    if not locations or not requested_mids:
        return None, set(), {
            "physical_locations": len(locations),
            "requested_mids": len(requested_mids),
            "supported_mids": 0,
            "full_mid_cover": False,
        }
    ranked: list[tuple[tuple[Any, ...], Mapping[str, Any], set[str]]] = []
    for row in locations:
        labels = set(str(value) for value in row.get("labels") or [])
        supported = requested_mids & labels
        if require_label_identity and not supported:
            continue
        if not require_label_identity:
            supported = set(requested_mids)
        key = (
            -len(supported),
            *_route_rank(str(row["source_route"]), route_priority),
            str(row["parquet_url"]),
            int(row["row_group"]),
            int(row["row_index"]),
        )
        ranked.append((key, row, supported))
    if not ranked:
        return None, set(), {
            "physical_locations": len(locations),
            "requested_mids": len(requested_mids),
            "supported_mids": 0,
            "full_mid_cover": False,
        }
    _, chosen, supported = min(ranked, key=lambda value: value[0])
    return dict(chosen), set(supported), {
        "physical_locations": len(locations),
        "requested_mids": len(requested_mids),
        "supported_mids": len(supported),
        "full_mid_cover": supported == requested_mids,
    }


def _candidate_counts(
    candidates: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    selected_labels: Sequence[str],
    train_target: int,
    eval_target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for (split, label, _), row in candidates.items():
        grouped[(split, label)].append(row)
    rows: list[dict[str, Any]] = []
    deficits: list[dict[str, Any]] = []
    for split, target in (("train", train_target), ("eval", eval_target)):
        for label in sorted(selected_labels):
            values = grouped.get((split, label), [])
            by_tier = Counter(int(row["ambiguity_tier"]) for row in values)
            record = {
                "metadata_split": split,
                "label": label,
                "target": target,
                "available_unique_videos": len(values),
                "tier0_unique_videos": by_tier[0],
                "tier0_1_unique_videos": by_tier[0] + by_tier[1],
                "tier0_2_unique_videos": by_tier[0] + by_tier[1] + by_tier[2],
                "tier3_unique_videos": by_tier[3],
                "deficit": max(0, target - len(values)),
            }
            rows.append(record)
            if record["deficit"]:
                deficits.append(record)
    return rows, deficits


def _attach_availability_contracts(
    crop_rows: Iterable[Mapping[str, Any]],
    *,
    candidate_location: Mapping[
        tuple[str, str, str, str], Mapping[str, Any]
    ],
    partition: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, Counter[str]],
    set[str],
]:
    if partition not in {"primary", "reserve"}:
        raise ValueError(f"unsupported selection partition: {partition}")
    output: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    route_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "eval": Counter(),
    }
    selection_keys: set[str] = set()
    for raw in crop_rows:
        row = dict(raw)
        split = str(row["metadata_split"])
        key = (
            split,
            str(row["coverage_label"]),
            str(row["video_id"]),
            str(row["source_event_uid"]),
        )
        location = candidate_location.get(key)
        if location is None:
            raise AvailabilityAwarePlanError(
                f"selected {partition} crop lost its availability location: {key}"
            )
        selection_key = str(row["selection_key"])
        if selection_key in selection_keys:
            raise AvailabilityAwarePlanError(
                f"duplicate {partition} selection key: {selection_key}"
            )
        selection_keys.add(selection_key)
        source_route = str(location["source_route"])
        route_counts[split][source_route] += 1
        row.update(
            {
                "format": FORMAT,
                "selection_partition": partition,
                "materialization_source_route": source_route,
                "materialization_hf_dataset": str(location["hf_dataset"]),
                "materialization_hf_revision": str(location["hf_revision"]),
                "materialization_hf_split": str(location["hf_split"]),
                "availability_route_label_verified": str(row["coverage_mid"])
                in set(location.get("labels") or []),
                "availability_location": {
                    field: location[field]
                    for field in (
                        "source_route",
                        "hf_dataset",
                        "hf_revision",
                        "hf_split",
                        "video_id",
                        "parquet_url",
                        "row_group",
                        "row_index",
                        "labels",
                        "human_labels",
                        "shard_provenance",
                    )
                    if field in location
                },
            }
        )
        if partition == "reserve":
            row["reserve_rank"] = int(row["coverage_rank"])
        output[split].append(row)
    return output, route_counts, selection_keys


def build_availability_aware_plan(
    *,
    cleanliness_rows: Iterable[Mapping[str, Any]],
    availability_entries: Iterable[Mapping[str, Any]],
    events_by_split: Mapping[str, Mapping[str, Sequence[StrongEvent]]],
    selected_label_to_mid: Mapping[str, str],
    route_priority: Sequence[str],
    target_train_videos_per_class: int = 100,
    target_eval_videos_per_class: int = 20,
    reserve_train_videos_per_class: int = 100,
    reserve_eval_videos_per_class: int = 20,
    crop_padding_seconds: float = 0.25,
    seed: int = 2028,
    require_label_identity: bool = True,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Reselect exact class quotas from materializable official sources only."""

    if not selected_label_to_mid:
        raise AvailabilityAwarePlanError("selected ontology is empty")
    if len(set(selected_label_to_mid.values())) != len(selected_label_to_mid):
        raise AvailabilityAwarePlanError("selected ontology has duplicate MIDs")
    if reserve_train_videos_per_class < 0 or reserve_eval_videos_per_class < 0:
        raise AvailabilityAwarePlanError("reserve quotas must be non-negative")
    availability = normalize_availability_entries(availability_entries)
    selected_mids = set(selected_label_to_mid.values())
    enriched_rows: list[dict[str, Any]] = []
    availability_audit = Counter()
    candidate_location: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    observed_label_mid: dict[str, set[str]] = defaultdict(set)
    cleanliness_by_video: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in cleanliness_rows:
        row = dict(raw)
        split = str(row.get("metadata_split") or "")
        video_id = str(row.get("video_id") or "")
        label = str(row.get("label") or "")
        mid = str(row.get("mid") or "")
        event_uid = str(row.get("event_uid") or "")
        if label not in selected_label_to_mid:
            continue
        observed_label_mid[label].add(mid)
        if selected_label_to_mid[label] != mid:
            raise AvailabilityAwarePlanError(
                f"cleanliness label/MID mismatch: {label} -> {mid}, "
                f"expected {selected_label_to_mid[label]}"
            )
        cleanliness_by_video[(split, video_id)].append(row)

    for (split, video_id), video_rows in sorted(cleanliness_by_video.items()):
        locations = availability.get((split, video_id), [])
        if not locations:
            availability_audit["events_without_available_video"] += len(video_rows)
            continue
        requested_mids = {str(row["mid"]) for row in video_rows}
        availability_audit["label_mismatched_locations"] += sum(
            not requested_mids.issubset(
                {str(value) for value in location.get("labels") or []}
            )
            for location in locations
        )
        chosen, supported_mids, audit = choose_canonical_location_for_video(
            locations,
            requested_mids=requested_mids,
            route_priority=route_priority,
            require_label_identity=require_label_identity,
        )
        availability_audit["physical_locations_considered"] += int(
            audit["physical_locations"]
        )
        if chosen is None:
            availability_audit["events_without_label_verified_route"] += len(
                video_rows
            )
            continue
        if bool(audit["full_mid_cover"]):
            availability_audit["videos_with_full_canonical_mid_cover"] += 1
        else:
            availability_audit["videos_with_partial_canonical_mid_cover"] += 1
        for row in video_rows:
            label = str(row["label"])
            mid = str(row["mid"])
            event_uid = str(row["event_uid"])
            verified = mid in supported_mids
            if require_label_identity and not verified:
                availability_audit[
                    "events_excluded_by_canonical_route_label_conflict"
                ] += 1
                continue
            availability_audit["available_events"] += 1
            row["availability_location"] = chosen
            row["availability_route_label_verified"] = verified
            enriched_rows.append(row)
            candidate_location[(split, label, video_id, event_uid)] = chosen

    missing_ontology_rows = sorted(
        label for label in selected_label_to_mid if not observed_label_mid.get(label)
    )
    if missing_ontology_rows:
        raise AvailabilityAwarePlanError(
            f"cleanliness audit has no rows for selected labels: {missing_ontology_rows}"
        )
    canonical_location_by_video: dict[tuple[str, str], tuple[str, str, int, int]] = {}
    for (split, _label, video_id, _event_uid), location in candidate_location.items():
        identity = _location_identity(location)
        key = (split, video_id)
        previous = canonical_location_by_video.get(key)
        if previous is not None and previous != identity:
            raise AvailabilityAwarePlanError(
                f"non-canonical physical locations remain for {split}/{video_id}"
            )
        canonical_location_by_video[key] = identity
    best = best_event_per_label_video(enriched_rows, seed=seed)
    per_class, deficits = _candidate_counts(
        best,
        selected_labels=tuple(selected_label_to_mid),
        train_target=target_train_videos_per_class,
        eval_target=target_eval_videos_per_class,
    )
    crop_rows, source_summary = build_crop_source_plan(
        best_candidates=best,
        events_by_split=events_by_split,
        selected_mids=selected_mids,
        target_train_videos_per_class=target_train_videos_per_class,
        target_eval_videos_per_class=target_eval_videos_per_class,
        crop_padding_seconds=crop_padding_seconds,
        seed=seed,
    )
    primary, route_counts, selection_keys = _attach_availability_contracts(
        crop_rows,
        candidate_location=candidate_location,
        partition="primary",
    )

    # Reserve selection is computed from the same pre-QA acoustic ranking, but
    # only after removing every source video used by the primary plan.  This
    # makes later AudioSep/audibility rejection refill deterministic and free
    # of outcome-dependent reselection or source reuse.
    primary_sources_by_split = {
        split: {str(row["video_id"]) for row in primary[split]}
        for split in ("train", "eval")
    }
    reserve_candidates = {
        key: row
        for key, row in best.items()
        if str(key[2]) not in primary_sources_by_split[str(key[0])]
    }
    reserve_support, reserve_deficits = _candidate_counts(
        reserve_candidates,
        selected_labels=tuple(selected_label_to_mid),
        train_target=reserve_train_videos_per_class,
        eval_target=reserve_eval_videos_per_class,
    )
    if reserve_train_videos_per_class and reserve_eval_videos_per_class:
        reserve_crop_rows, reserve_source_summary = build_crop_source_plan(
            best_candidates=reserve_candidates,
            events_by_split=events_by_split,
            selected_mids=selected_mids,
            target_train_videos_per_class=reserve_train_videos_per_class,
            target_eval_videos_per_class=reserve_eval_videos_per_class,
            crop_padding_seconds=crop_padding_seconds,
            seed=seed,
        )
    else:
        # build_crop_source_plan intentionally requires positive quotas; allow
        # callers to disable both reserve splits without weakening primary.
        if reserve_train_videos_per_class != reserve_eval_videos_per_class:
            raise AvailabilityAwarePlanError(
                "reserve quotas must both be positive or both be zero"
            )
        reserve_crop_rows = []
        reserve_source_summary = {
            "planned_rows": 0,
            "planned_unique_videos": 0,
            "disabled": True,
        }
    reserve, reserve_route_counts, reserve_selection_keys = (
        _attach_availability_contracts(
            reserve_crop_rows,
            candidate_location=candidate_location,
            partition="reserve",
        )
    )
    if selection_keys & reserve_selection_keys:
        raise AvailabilityAwarePlanError(
            "primary and reserve selection keys overlap"
        )
    output: dict[str, list[dict[str, Any]]] = {
        "train": primary["train"],
        "eval": primary["eval"],
        "reserve_train": reserve["train"],
        "reserve_eval": reserve["eval"],
    }

    selected_counts = Counter(
        (split, str(row["coverage_label"]))
        for split in ("train", "eval")
        for row in primary[split]
    )
    quota_mismatches = []
    for split, target in (
        ("train", target_train_videos_per_class),
        ("eval", target_eval_videos_per_class),
    ):
        for label in sorted(selected_label_to_mid):
            observed = selected_counts[(split, label)]
            if observed != target:
                quota_mismatches.append(
                    {
                        "metadata_split": split,
                        "label": label,
                        "expected": target,
                        "observed": observed,
                    }
                )
    train_sources = {str(row["video_id"]) for row in primary["train"]}
    eval_sources = {str(row["video_id"]) for row in primary["eval"]}
    overlap = train_sources & eval_sources
    if overlap:
        raise AvailabilityAwarePlanError(
            f"selected train/eval source overlap: {len(overlap)}; "
            f"examples={sorted(overlap)[:10]}"
        )
    reserve_train_sources = {
        str(row["video_id"]) for row in reserve["train"]
    }
    reserve_eval_sources = {
        str(row["video_id"]) for row in reserve["eval"]
    }
    primary_reserve_overlap = (
        (train_sources & reserve_train_sources)
        | (eval_sources & reserve_eval_sources)
        | (train_sources & reserve_eval_sources)
        | (eval_sources & reserve_train_sources)
    )
    reserve_cross_split_overlap = reserve_train_sources & reserve_eval_sources
    if primary_reserve_overlap or reserve_cross_split_overlap:
        raise AvailabilityAwarePlanError(
            "primary/reserve source overlap violates refill independence: "
            f"primary_reserve={sorted(primary_reserve_overlap)[:10]}, "
            f"reserve_cross_split={sorted(reserve_cross_split_overlap)[:10]}"
        )
    plan_hash = _sha256(
        sorted(
            [*primary["train"], *primary["eval"]],
            key=lambda row: (
                str(row["metadata_split"]),
                str(row["coverage_label"]),
                int(row["coverage_rank"]),
                str(row["selection_key"]),
            ),
        )
    )
    reserve_plan_hash = _sha256(
        sorted(
            [*reserve["train"], *reserve["eval"]],
            key=lambda row: (
                str(row["metadata_split"]),
                str(row["coverage_label"]),
                int(row["reserve_rank"]),
                str(row["selection_key"]),
            ),
        )
    )
    report = {
        "format": FORMAT,
        "audit_passes": not deficits and not quota_mismatches,
        "plan_sha256": plan_hash,
        "reserve_plan_sha256": reserve_plan_hash,
        "selected_classes": len(selected_label_to_mid),
        "configuration": {
            "target_train_videos_per_class": target_train_videos_per_class,
            "target_eval_videos_per_class": target_eval_videos_per_class,
            "reserve_train_videos_per_class": reserve_train_videos_per_class,
            "reserve_eval_videos_per_class": reserve_eval_videos_per_class,
            "crop_padding_seconds": crop_padding_seconds,
            "seed": seed,
            "route_priority": list(route_priority),
            "require_route_label_identity": require_label_identity,
        },
        "availability": {
            "indexed_split_video_pairs": len(availability),
            "indexed_physical_locations": sum(map(len, availability.values())),
            **dict(sorted(availability_audit.items())),
        },
        "per_class_support": per_class,
        "per_class_reserve_support_after_primary_source_exclusion": reserve_support,
        "deficits": deficits,
        "reserve_deficits": reserve_deficits,
        "quota_mismatches": quota_mismatches,
        "planned_crop_rows": {
            "train": len(output["train"]),
            "eval": len(output["eval"]),
            "total": len(output["train"]) + len(output["eval"]),
        },
        "planned_unique_source_videos": {
            "train": len(train_sources),
            "eval": len(eval_sources),
            "total": len(train_sources) + len(eval_sources),
        },
        "reserve_crop_rows": {
            "train": len(reserve["train"]),
            "eval": len(reserve["eval"]),
            "total": len(reserve["train"]) + len(reserve["eval"]),
        },
        "reserve_unique_source_videos": {
            "train": len(reserve_train_sources),
            "eval": len(reserve_eval_sources),
            "total": len(reserve_train_sources) + len(reserve_eval_sources),
        },
        "route_counts": {
            split: dict(sorted(route_counts[split].items()))
            for split in ("train", "eval")
        },
        "reserve_route_counts": {
            split: dict(sorted(reserve_route_counts[split].items()))
            for split in ("train", "eval")
        },
        "train_eval_source_overlap": 0,
        "primary_reserve_source_overlap": 0,
        "reserve_train_eval_source_overlap": 0,
        "selection_key_count": len(selection_keys),
        "reserve_selection_key_count": len(reserve_selection_keys),
        "source_plan_summary": source_summary,
        "reserve_source_plan_summary": reserve_source_summary,
        "invariants": {
            "availability_join_before_quota_selection": True,
            "exact_label_mid_identity": True,
            "all_intersecting_strong_annotations_preserved": True,
            "no_qa_or_detector_outcome_used": True,
            "official_eval_locked_to_test": True,
            "one_physical_location_per_official_split_video": True,
            "reserve_ranked_before_audiosep_or_audibility_outcomes": True,
            "reserve_sources_disjoint_from_primary_sources": True,
        },
    }
    return output, report


def refill_rejected_primary_rows(
    *,
    primary_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    reserve_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    rejected_selection_keys: set[str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Deterministically replace rejected primary rows from a frozen reserve.

    The reserve ordering and physical locations must already have been frozen
    before any AudioSep/audibility outcome.  Refill therefore consumes no QA
    signal and never reruns candidate selection with a new seed.
    """

    output: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    all_primary_keys = {
        str(row["selection_key"])
        for split in ("train", "eval")
        for row in primary_by_split.get(split, [])
    }
    unknown = rejected_selection_keys - all_primary_keys
    if unknown:
        raise AvailabilityAwarePlanError(
            f"rejected keys are absent from primary plan: {sorted(unknown)[:10]}"
        )
    primary_sources = {
        str(row["video_id"])
        for split in ("train", "eval")
        for row in primary_by_split.get(split, [])
    }
    reserve_sources = {
        str(row["video_id"])
        for split in ("train", "eval")
        for row in reserve_by_split.get(split, [])
    }
    overlap = primary_sources & reserve_sources
    if overlap:
        raise AvailabilityAwarePlanError(
            f"reserve source overlaps frozen primary plan: {sorted(overlap)[:10]}"
        )

    replacements: list[dict[str, Any]] = []
    deficits: list[dict[str, Any]] = []
    for split in ("train", "eval"):
        primary_rows = [dict(row) for row in primary_by_split.get(split, [])]
        reserve_rows = [dict(row) for row in reserve_by_split.get(split, [])]
        rejected_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        kept: list[dict[str, Any]] = []
        for row in primary_rows:
            if str(row["selection_key"]) in rejected_selection_keys:
                rejected_by_label[str(row["coverage_label"])].append(row)
            else:
                kept.append(row)
        reserve_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in reserve_rows:
            reserve_by_label[str(row["coverage_label"])].append(row)
        for label, rejected in sorted(rejected_by_label.items()):
            candidates = sorted(
                reserve_by_label.get(label, []),
                key=lambda row: (
                    int(row.get("reserve_rank") or row.get("coverage_rank") or 0),
                    str(row["selection_key"]),
                ),
            )
            if len(candidates) < len(rejected):
                deficits.append(
                    {
                        "metadata_split": split,
                        "label": label,
                        "rejected": len(rejected),
                        "reserve_available": len(candidates),
                        "deficit": len(rejected) - len(candidates),
                    }
                )
            for original, candidate in zip(
                sorted(
                    rejected,
                    key=lambda row: (
                        int(row["coverage_rank"]),
                        str(row["selection_key"]),
                    ),
                ),
                candidates,
            ):
                replacement = dict(candidate)
                replacement.update(
                    {
                        "selection_partition": "primary_refill_from_frozen_reserve",
                        "coverage_rank": int(original["coverage_rank"]),
                        "replaced_primary_selection_key": str(
                            original["selection_key"]
                        ),
                    }
                )
                kept.append(replacement)
                replacements.append(replacement)
        output[split] = sorted(
            kept,
            key=lambda row: (
                str(row["coverage_label"]),
                int(row["coverage_rank"]),
                str(row["selection_key"]),
            ),
        )
    output_sources = {
        str(row["video_id"])
        for split in ("train", "eval")
        for row in output[split]
    }
    replacement_sources = {str(row["video_id"]) for row in replacements}
    retained_primary_sources = output_sources - replacement_sources
    if retained_primary_sources & replacement_sources:
        raise AvailabilityAwarePlanError(
            "refill unexpectedly reused a retained primary source"
        )
    return output, {
        "format": FORMAT,
        "audit_passes": not deficits,
        "rejected_primary_rows": len(rejected_selection_keys),
        "replacement_rows": len(replacements),
        "deficits": deficits,
        "retained_primary_refill_source_overlap": 0,
        "selection_policy": "frozen_reserve_rank_then_selection_key",
        "uses_qa_outcomes": False,
        "changes_seed_after_rejection": False,
    }
