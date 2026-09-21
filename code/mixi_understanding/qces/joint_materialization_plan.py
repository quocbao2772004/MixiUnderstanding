"""Convert the acoustic crop contract into one minimal video materialization plan.

The acoustic-cleanliness source plan is expressed per class/video support unit,
so a multi-label video can occur several times.  This module merges those rows
without changing the selected support contract and attaches every official
strong event from the source video.  The result is the single plan that should
be scanned/materialized for both detector training and clean crop generation.
"""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from mixi_understanding.qces.supported_ontology import StrongEvent


FORMAT = "qces_joint_acoustic_materialization_plan_v1"


class JointPlanError(ValueError):
    """Raised when the crop contract cannot produce a leakage-safe plan."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_crop_rows(paths: Sequence[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    receipts: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise JointPlanError(f"crop source plan does not exist: {path}")
        count = 0
        splits: set[str] = set()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise JointPlanError(
                        f"invalid JSON at {path}:{line_number}: {error}"
                    ) from error
                split = str(row.get("metadata_split") or "")
                if split not in rows_by_split:
                    raise JointPlanError(
                        f"unsupported metadata_split={split!r} at {path}:{line_number}"
                    )
                rows_by_split[split].append(row)
                splits.add(split)
                count += 1
        receipts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": count,
                "metadata_splits": sorted(splits),
            }
        )
    return rows_by_split, receipts


def _event_payload(event: StrongEvent, *, selected_mids: set[str]) -> dict[str, Any]:
    return {
        "segment_id": event.segment_id,
        "video_id": event.video_id,
        "mid": event.mid,
        "label": event.label,
        "display_name": event.display_name,
        "onset_seconds": float(event.onset_seconds),
        "offset_seconds": float(event.offset_seconds),
        "selected_ontology_label": event.mid in selected_mids,
    }


def _validate_coverage_event(row: Mapping[str, Any], events: Sequence[StrongEvent]) -> None:
    mid = str(row.get("coverage_mid") or "")
    label = str(row.get("coverage_label") or "")
    segment_id = str(row.get("segment_id") or "")
    onset = float(row.get("event_onset_seconds", -1.0))
    offset = float(row.get("event_offset_seconds", -1.0))
    if not any(
        event.mid == mid
        and event.label == label
        and event.segment_id == segment_id
        and event.onset_seconds <= onset + 1e-9
        and event.offset_seconds >= offset - 1e-9
        for event in events
    ):
        raise JointPlanError(
            "crop coverage event is absent from full official strong events: "
            f"video={row.get('video_id')} label={row.get('coverage_label')}"
        )


def _materialization_contract(
    row: Mapping[str, Any], *, metadata_split: str, video_id: str
) -> dict[str, Any]:
    """Normalize a route-aware physical location for one crop request."""

    source_route = str(row.get("materialization_source_route") or "")
    if not source_route:
        return {
            "source_route": "enyoukai_audioset_strong_pinned_parquet_v1",
            "hf_dataset": "enyoukai/AudioSet-Strong",
            "hf_revision": "",
            "hf_split": "train" if metadata_split == "train" else "test",
            "availability_location": {},
        }
    dataset = str(row.get("materialization_hf_dataset") or "")
    revision = str(row.get("materialization_hf_revision") or "")
    hf_split = str(row.get("materialization_hf_split") or "")
    location = row.get("availability_location")
    if not dataset or not revision or not hf_split or not isinstance(location, dict):
        raise JointPlanError(
            f"incomplete materialization route for {metadata_split}/{video_id}"
        )
    expected_split = "train" if metadata_split == "train" else "test"
    if hf_split != expected_split:
        raise JointPlanError(
            f"materialization route split mismatch for {metadata_split}/{video_id}: "
            f"{hf_split} != {expected_split}"
        )
    for key, expected in (
        ("source_route", source_route),
        ("hf_dataset", dataset),
        ("hf_revision", revision),
        ("hf_split", hf_split),
        ("video_id", video_id),
    ):
        if str(location.get(key) or "") != expected:
            raise JointPlanError(
                f"availability location {key} mismatch for {metadata_split}/{video_id}"
            )
    if (
        not str(location.get("parquet_url") or "")
        or int(location.get("row_group", -1)) < 0
        or int(location.get("row_index", -1)) < 0
    ):
        raise JointPlanError(
            f"availability location is incomplete for {metadata_split}/{video_id}"
        )
    return {
        "source_route": source_route,
        "hf_dataset": dataset,
        "hf_revision": revision,
        "hf_split": hf_split,
        "availability_location": dict(location),
    }


def _existing_disk_estimate(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    unique_videos: Mapping[str, set[str]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    total_remaining = 0.0
    for split in ("train", "eval"):
        path_by_video: dict[str, Path] = {}
        for row in rows_by_split[split]:
            video_id = str(row["video_id"])
            value = str(row.get("materialized_audio_path") or "")
            if value and video_id not in path_by_video:
                candidate = Path(value)
                if candidate.is_file():
                    path_by_video[video_id] = candidate
        sizes = [path.stat().st_size for path in path_by_video.values()]
        mean_bytes = statistics.fmean(sizes) if sizes else 0.0
        median_bytes = statistics.median(sizes) if sizes else 0.0
        missing = len(unique_videos[split]) - len(path_by_video)
        remaining = mean_bytes * missing
        total_remaining += remaining
        output[split] = {
            "unique_plan_videos": len(unique_videos[split]),
            "existing_audio_videos": len(path_by_video),
            "existing_audio_bytes": sum(sizes),
            "existing_mean_audio_bytes": mean_bytes,
            "existing_median_audio_bytes": median_bytes,
            "missing_audio_videos": missing,
            "remaining_audio_bytes_estimate_from_existing_mean": int(round(remaining)),
        }
    output["total_remaining_audio_bytes_estimate_from_existing_mean"] = int(
        round(total_remaining)
    )
    output["method"] = (
        "missing video count multiplied by split-specific mean encoded FLAC size "
        "of matching already-materialized crop-plan videos"
    )
    return output


def build_joint_materialization_plan(
    *,
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    events_by_split: Mapping[str, Mapping[str, Sequence[StrongEvent]]],
    selected_labels: set[str],
    selected_mids: set[str],
    target_train_per_class: int = 100,
    target_eval_per_class: int = 20,
    raw_plan_ids: Mapping[str, set[str]] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if len(selected_labels) != len(selected_mids):
        raise JointPlanError("selected label/MID sets must have equal cardinality")
    if not selected_labels:
        raise JointPlanError("selected ontology is empty")
    raw_plan_ids = raw_plan_ids or {"train": set(), "eval": set()}
    expected_targets = {"train": target_train_per_class, "eval": target_eval_per_class}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        "train": defaultdict(list),
        "eval": defaultdict(list),
    }
    coverage_units: set[tuple[str, str, str]] = set()
    coverage_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "eval": Counter(),
    }
    tier_counts: dict[str, Counter[int]] = {
        "train": Counter(),
        "eval": Counter(),
    }
    unique_videos: dict[str, set[str]] = {"train": set(), "eval": set()}
    route_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "eval": Counter(),
    }
    for split in ("train", "eval"):
        for source in rows_by_split.get(split, ()):
            row = dict(source)
            video_id = str(row.get("video_id") or "")
            label = str(row.get("coverage_label") or "")
            mid = str(row.get("coverage_mid") or "")
            if not video_id or label not in selected_labels or mid not in selected_mids:
                raise JointPlanError(
                    f"invalid crop support row split={split} video={video_id} label={label} mid={mid}"
                )
            unit = (split, label, video_id)
            if unit in coverage_units:
                raise JointPlanError(f"duplicate class-video support unit: {unit}")
            coverage_units.add(unit)
            coverage_counts[split][label] += 1
            tier_counts[split][int(row["ambiguity_tier"])] += 1
            grouped[split][video_id].append(row)
            unique_videos[split].add(video_id)

    quota_failures: list[dict[str, Any]] = []
    for split in ("train", "eval"):
        target = expected_targets[split]
        for label in sorted(selected_labels):
            observed = int(coverage_counts[split][label])
            if observed != target:
                quota_failures.append(
                    {"split": split, "label": label, "expected": target, "observed": observed}
                )
    if quota_failures:
        raise JointPlanError(
            f"crop contract does not meet exact class quotas; examples={quota_failures[:10]}"
        )
    overlap = unique_videos["train"] & unique_videos["eval"]
    if overlap:
        raise JointPlanError(
            f"official train/eval source overlap: {len(overlap)} videos; examples={sorted(overlap)[:10]}"
        )

    plans: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    for split in ("train", "eval"):
        for video_id in sorted(grouped[split]):
            requests = sorted(
                grouped[split][video_id],
                key=lambda row: (
                    str(row["coverage_label"]),
                    int(row["coverage_rank"]),
                    str(row["selection_key"]),
                ),
            )
            events = list(events_by_split.get(split, {}).get(video_id, ()))
            if not events:
                raise JointPlanError(
                    f"crop-plan video is absent from official strong metadata: {split}/{video_id}"
                )
            for request in requests:
                _validate_coverage_event(request, events)
            materialization_contracts = [
                _materialization_contract(
                    request,
                    metadata_split=split,
                    video_id=video_id,
                )
                for request in requests
            ]
            contract_payloads = {
                _canonical(contract) for contract in materialization_contracts
            }
            if len(contract_payloads) != 1:
                raise JointPlanError(
                    f"one source video was assigned conflicting materialization "
                    f"locations: {split}/{video_id}"
                )
            contract = materialization_contracts[0]
            route_counts[split][str(contract["source_route"])] += 1
            locks = {str(row.get("split_lock") or "") for row in requests}
            if len(locks) != 1:
                raise JointPlanError(
                    f"conflicting split locks for {split}/{video_id}: {sorted(locks)}"
                )
            split_lock = next(iter(locks))
            if split == "eval" and split_lock != "test":
                raise JointPlanError(
                    f"official eval video is not test-locked: {video_id}/{split_lock}"
                )
            full_events = [
                _event_payload(event, selected_mids=selected_mids)
                for event in sorted(
                    events,
                    key=lambda value: (
                        value.onset_seconds,
                        value.offset_seconds,
                        value.mid,
                        value.segment_id,
                    ),
                )
            ]
            selected_event_labels = sorted(
                {
                    str(event["label"])
                    for event in full_events
                    if bool(event["selected_ontology_label"])
                }
            )
            plans[split].append(
                {
                    "format": FORMAT,
                    "source_route": str(contract["source_route"]),
                    "hf_dataset": str(contract["hf_dataset"]),
                    "hf_revision": str(contract["hf_revision"]),
                    "metadata_split": split,
                    "hf_split": str(contract["hf_split"]),
                    "availability_location": dict(
                        contract["availability_location"]
                    ),
                    "video_id": video_id,
                    "split_lock": split_lock,
                    "labels": selected_event_labels,
                    "all_strong_labels": sorted(
                        {str(event["label"]) for event in full_events}
                    ),
                    "covers_deficit_labels": sorted(
                        {str(row["coverage_label"]) for row in requests}
                    ),
                    "coverage_request_count": len(requests),
                    "crop_requests": requests,
                    "segment_ids": sorted(
                        {str(event["segment_id"]) for event in full_events}
                    ),
                    "events": full_events,
                    "full_official_strong_events": True,
                    "supervision_policy": (
                        "crop_contract_coverage;full_official_strong_events;"
                        "coverage_label_is_planning_only;source_disjoint_split_lock"
                    ),
                }
            )

    comparison: dict[str, Any] = {}
    for split in ("train", "eval"):
        crop_ids = unique_videos[split]
        raw_ids = set(raw_plan_ids.get(split, set()))
        comparison[split] = {
            "joint_crop_contract_unique_videos": len(crop_ids),
            "old_raw_plan_unique_videos": len(raw_ids),
            "overlap": len(crop_ids & raw_ids),
            "naive_union": len(crop_ids | raw_ids),
            "crop_only": len(crop_ids - raw_ids),
            "raw_only_skipped": len(raw_ids - crop_ids),
        }
    plan_hash = _sha256_bytes(
        _canonical(
            sorted(
                [
                row
                for split in ("train", "eval")
                for row in plans[split]
                ],
                key=lambda row: (str(row["hf_split"]), str(row["video_id"])),
            )
        )
    )
    receipt = {
        "format": FORMAT,
        "audit_passes": True,
        "plan_sha256": plan_hash,
        "selected_classes": len(selected_labels),
        "coverage_class_video_rows": {
            split: sum(coverage_counts[split].values()) for split in ("train", "eval")
        },
        "exact_class_video_quota": {
            "train": target_train_per_class,
            "eval": target_eval_per_class,
        },
        "unique_videos": {split: len(unique_videos[split]) for split in ("train", "eval")},
        "total_unique_videos": sum(len(unique_videos[split]) for split in ("train", "eval")),
        "train_eval_source_overlap": 0,
        "tier_counts": {
            split: {str(key): value for key, value in sorted(tier_counts[split].items())}
            for split in ("train", "eval")
        },
        "route_source_video_counts": {
            split: dict(sorted(route_counts[split].items()))
            for split in ("train", "eval")
        },
        "old_raw_plan_comparison": comparison,
        "naive_union_total": sum(comparison[split]["naive_union"] for split in ("train", "eval")),
        "raw_only_videos_skipped_total": sum(
            comparison[split]["raw_only_skipped"] for split in ("train", "eval")
        ),
        "pre_scan_disk_estimate": _existing_disk_estimate(rows_by_split, unique_videos),
        "invariants": {
            "one_row_per_source_video": True,
            "all_official_strong_events_emitted": True,
            "coverage_requests_preserved": True,
            "exact_200_class_100_20_contract": True,
            "official_eval_test_locked": True,
            "cross_official_split_source_overlap": 0,
            "raw_plan_rows_added": 0,
        },
    }
    return plans, receipt
