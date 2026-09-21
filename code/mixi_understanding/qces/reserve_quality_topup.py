"""Deterministic quality-gated refill from a frozen QCES reserve plan.

The availability-aware planner freezes primary and reserve crop requests before
AudioSep is run.  This module consumes only the terminal decision of that fixed
acoustic quality gate (gold/silver/rejected); it deliberately ignores QA,
detector accuracy, and the numeric AudioSep/CLAP scores.  Accepted reserve rows
are consumed in their frozen acoustic order and unevaluated rows are emitted as
the next bounded materialization wave.

The pending wave is also converted to the route-aware joint schema accepted by
``audioset_plan_materializer.py``.  This keeps a single physical download per
source video even when that source covers more than one class deficit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mixi_understanding.qces.supported_ontology import StrongEvent


FORMAT = "qces_reserve_quality_topup_v1"
JOINT_FORMAT = "qces_joint_acoustic_materialization_plan_v1"
DEFAULT_ACCEPTED_TIERS = ("gold", "silver")


class ReserveTopupError(ValueError):
    """Raised when a refill would weaken the frozen data contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ReserveTopupError(f"JSONL input does not exist: {resolved}")
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReserveTopupError(
                    f"invalid JSON at {resolved}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ReserveTopupError(
                    f"JSONL row is not an object at {resolved}:{line_number}"
                )
            rows.append(value)
    return rows


def flatten_crop_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Accept either flat crop rows or existing joint materialization rows."""

    output: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        requests = row.get("crop_requests")
        if isinstance(requests, list):
            for request in requests:
                if not isinstance(request, Mapping):
                    raise ReserveTopupError("joint crop_requests contains a non-object")
                request_row = dict(request)
                # A joint row is authoritative for these transport fields if a
                # historical nested request omitted them.
                for field, nested_field in (
                    ("metadata_split", "metadata_split"),
                    ("video_id", "video_id"),
                    ("materialization_source_route", "source_route"),
                    ("materialization_hf_dataset", "hf_dataset"),
                    ("materialization_hf_revision", "hf_revision"),
                    ("materialization_hf_split", "hf_split"),
                    ("availability_location", "availability_location"),
                    ("split_lock", "split_lock"),
                ):
                    if not request_row.get(field) and row.get(nested_field) is not None:
                        request_row[field] = row[nested_field]
                output.append(request_row)
        else:
            output.append(row)
    return output


def _selection_key(row: Mapping[str, Any]) -> str:
    for field in ("selection_key", "materialization_item_id", "item_id", "source_id"):
        value = str(row.get(field) or "")
        if value:
            return value
    crop_request = row.get("crop_request")
    if isinstance(crop_request, Mapping):
        value = str(crop_request.get("selection_key") or "")
        if value:
            return value
    raise ReserveTopupError("row has no selection/materialization identity")


def _reserve_sort_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("ambiguity_tier", 99)),
        int(row.get("reserve_rank") or row.get("coverage_rank") or 0),
        _selection_key(row),
    )


def _primary_sort_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return (int(row.get("coverage_rank") or 0), _selection_key(row))


def _normalize_plan(
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_partition: str,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, set[str]],
]:
    normalized: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    by_key: dict[str, dict[str, Any]] = {}
    sources: dict[str, set[str]] = {"train": set(), "eval": set()}
    support_units: set[tuple[str, str, str]] = set()
    for split in ("train", "eval"):
        for source in flatten_crop_rows(rows_by_split.get(split, ())):
            row = dict(source)
            row_split = str(row.get("metadata_split") or "")
            label = str(row.get("coverage_label") or "")
            mid = str(row.get("coverage_mid") or "")
            video_id = str(row.get("video_id") or "")
            key = _selection_key(row)
            partition = str(row.get("selection_partition") or "")
            if row_split != split or not label or not mid or not video_id:
                raise ReserveTopupError(
                    f"incomplete {expected_partition} crop row: "
                    f"split={row_split!r}, label={label!r}, video={video_id!r}"
                )
            if expected_partition == "primary" and partition not in {"", "primary"}:
                raise ReserveTopupError(
                    f"non-primary row supplied as primary: {key}/{partition}"
                )
            if expected_partition == "reserve" and partition != "reserve":
                raise ReserveTopupError(
                    f"non-reserve row supplied as reserve: {key}/{partition}"
                )
            unit = (split, label, video_id)
            if unit in support_units:
                raise ReserveTopupError(f"duplicate class-video support unit: {unit}")
            if key in by_key:
                raise ReserveTopupError(f"duplicate selection key: {key}")
            support_units.add(unit)
            by_key[key] = row
            sources[split].add(video_id)
            normalized[split].append(row)
    return normalized, by_key, sources


def targets_from_primary_contract(
    primary_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, int]]:
    """Derive a smoke-test quota from the rows deliberately placed in the smoke."""

    output: dict[str, dict[str, int]] = {"train": {}, "eval": {}}
    for split in ("train", "eval"):
        counts = Counter(
            str(row["coverage_label"])
            for row in flatten_crop_rows(primary_by_split.get(split, ()))
        )
        output[split] = {label: int(count) for label, count in sorted(counts.items())}
    return output


def constant_targets(
    primary_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    train_target: int = 100,
    eval_target: int = 20,
) -> dict[str, dict[str, int]]:
    """Build the production 100/20 target over the primary ontology."""

    if train_target <= 0 or eval_target <= 0:
        raise ReserveTopupError("train/eval targets must be positive")
    labels = {
        str(row["coverage_label"])
        for split in ("train", "eval")
        for row in flatten_crop_rows(primary_by_split.get(split, ()))
    }
    if not labels:
        raise ReserveTopupError("primary contract is empty")
    return {
        "train": {label: int(train_target) for label in sorted(labels)},
        "eval": {label: int(eval_target) for label in sorted(labels)},
    }


def _quality_protocol(row: Mapping[str, Any]) -> str:
    value = str(row.get("quality_protocol") or "")
    if value:
        return value
    gate = row.get("quality_gate")
    if isinstance(gate, Mapping):
        value = str(gate.get("quality_protocol") or "")
        if value:
            return value
    provenance = row.get("provenance")
    if isinstance(provenance, Mapping):
        value = str(provenance.get("quality_protocol") or "")
        if value:
            return value
    raise ReserveTopupError(
        f"quality row {_selection_key(row)} has no fixed quality protocol"
    )


def collect_quality_outcomes(
    rows: Iterable[Mapping[str, Any]],
    *,
    accepted_tiers: Sequence[str] = DEFAULT_ACCEPTED_TIERS,
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Normalize terminal gate outcomes without reading numeric quality scores."""

    allowed = {str(value) for value in accepted_tiers}
    if not allowed or "rejected" in allowed:
        raise ReserveTopupError("accepted tiers must be non-empty and exclude rejected")
    outcomes: dict[str, dict[str, Any]] = {}
    protocols: set[str] = set()
    for raw in rows:
        row = dict(raw)
        key = _selection_key(row)
        tier = str(row.get("acceptance_tier") or row.get("cleanliness_tier") or "")
        accepted_value = row.get("accepted")
        if not tier or not isinstance(accepted_value, bool):
            raise ReserveTopupError(
                f"quality row {key} lacks terminal accepted/tier fields"
            )
        accepted = bool(accepted_value)
        if accepted != (tier in allowed):
            raise ReserveTopupError(
                f"quality row {key} has inconsistent accepted={accepted}, tier={tier}"
            )
        protocol = _quality_protocol(row)
        protocols.add(protocol)
        compact = {"accepted": accepted, "acceptance_tier": tier, "protocol": protocol}
        previous = outcomes.get(key)
        if previous is not None and previous != compact:
            raise ReserveTopupError(f"conflicting quality outcomes for {key}")
        outcomes[key] = compact
    if len(protocols) > 1:
        raise ReserveTopupError(
            f"quality results mix fixed protocols: {sorted(protocols)}"
        )
    return outcomes, protocols


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


def _transport_contract(row: Mapping[str, Any]) -> dict[str, Any]:
    video_id = str(row["video_id"])
    contract = {
        "source_route": str(row.get("materialization_source_route") or ""),
        "hf_dataset": str(row.get("materialization_hf_dataset") or ""),
        "hf_revision": str(row.get("materialization_hf_revision") or ""),
        "hf_split": str(row.get("materialization_hf_split") or ""),
        "availability_location": dict(row.get("availability_location") or {}),
    }
    location = contract["availability_location"]
    if (
        not contract["source_route"]
        or not contract["hf_dataset"]
        or not contract["hf_revision"]
        or contract["hf_split"] not in {"train", "test"}
        or not location
    ):
        raise ReserveTopupError(f"incomplete reserve transport contract: {video_id}")
    for field, expected in (
        ("source_route", contract["source_route"]),
        ("hf_dataset", contract["hf_dataset"]),
        ("hf_revision", contract["hf_revision"]),
        ("hf_split", contract["hf_split"]),
        ("video_id", video_id),
    ):
        if str(location.get(field) or "") != str(expected):
            raise ReserveTopupError(
                f"reserve location {field} mismatch for {video_id}"
            )
    if (
        not str(location.get("parquet_url") or "")
        or int(location.get("row_group", -1)) < 0
        or int(location.get("row_index", -1)) < 0
    ):
        raise ReserveTopupError(f"incomplete reserve physical location: {video_id}")
    return contract


def build_joint_topup_plan(
    *,
    pending_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    events_by_split: Mapping[str, Mapping[str, Sequence[StrongEvent]]],
    selected_mids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    """Merge pending class rows into materializer-ready source-video rows."""

    output: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    for split in ("train", "eval"):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pending_by_split.get(split, ()):
            grouped[str(row["video_id"])].append(dict(row))
        for video_id in sorted(grouped):
            requests = sorted(grouped[video_id], key=_reserve_sort_key)
            contracts = {_canonical(_transport_contract(row)) for row in requests}
            if len(contracts) != 1:
                raise ReserveTopupError(
                    f"reserve source has conflicting physical routes: {split}/{video_id}"
                )
            contract = _transport_contract(requests[0])
            expected_hf_split = "train" if split == "train" else "test"
            if contract["hf_split"] != expected_hf_split:
                raise ReserveTopupError(
                    f"reserve split mismatch for {split}/{video_id}: "
                    f"{contract['hf_split']} != {expected_hf_split}"
                )
            locks = {str(row.get("split_lock") or "") for row in requests}
            if len(locks) != 1:
                raise ReserveTopupError(
                    f"conflicting reserve split locks: {split}/{video_id}"
                )
            split_lock = next(iter(locks))
            if split == "eval" and split_lock != "test":
                raise ReserveTopupError(
                    f"reserve eval source is not test-locked: {video_id}/{split_lock}"
                )
            events = sorted(
                events_by_split.get(split, {}).get(video_id, ()),
                key=lambda event: (
                    event.onset_seconds,
                    event.offset_seconds,
                    event.mid,
                    event.segment_id,
                ),
            )
            if not events:
                raise ReserveTopupError(
                    f"reserve source absent from strong metadata: {split}/{video_id}"
                )
            for request in requests:
                if not any(
                    event.segment_id == str(request["segment_id"])
                    and event.mid == str(request["coverage_mid"])
                    and event.label == str(request["coverage_label"])
                    and event.onset_seconds <= float(request["event_onset_seconds"]) + 1e-9
                    and event.offset_seconds >= float(request["event_offset_seconds"]) - 1e-9
                    for event in events
                ):
                    raise ReserveTopupError(
                        "reserve coverage event absent from strong metadata: "
                        f"{split}/{video_id}/{request['coverage_label']}"
                    )
            event_rows = [
                _event_payload(event, selected_mids=selected_mids) for event in events
            ]
            output[split].append(
                {
                    "format": JOINT_FORMAT,
                    "topup_format": FORMAT,
                    "source_route": contract["source_route"],
                    "hf_dataset": contract["hf_dataset"],
                    "hf_revision": contract["hf_revision"],
                    "metadata_split": split,
                    "hf_split": contract["hf_split"],
                    "availability_location": contract["availability_location"],
                    "video_id": video_id,
                    "split_lock": split_lock,
                    "labels": sorted(
                        {event["label"] for event in event_rows if event["selected_ontology_label"]}
                    ),
                    "all_strong_labels": sorted({event["label"] for event in event_rows}),
                    "covers_deficit_labels": sorted(
                        {str(row["coverage_label"]) for row in requests}
                    ),
                    "coverage_request_count": len(requests),
                    "crop_requests": requests,
                    "segment_ids": sorted({event["segment_id"] for event in event_rows}),
                    "events": event_rows,
                    "full_official_strong_events": True,
                    "supervision_policy": (
                        "frozen_reserve_quality_topup;full_official_strong_events;"
                        "fixed_acoustic_gate_only;source_disjoint_split_lock"
                    ),
                }
            )
    return output


def build_reserve_quality_topup(
    *,
    primary_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    reserve_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    quality_rows: Iterable[Mapping[str, Any]],
    targets_by_split_label: Mapping[str, Mapping[str, int]],
    events_by_split: Mapping[str, Mapping[str, Sequence[StrongEvent]]],
    overdraw_factor: float = 2.0,
    accepted_tiers: Sequence[str] = DEFAULT_ACCEPTED_TIERS,
    require_terminal_primary: bool = True,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    """Select accepted refills and the next deterministic reserve wave."""

    if overdraw_factor < 1.0 or not math.isfinite(overdraw_factor):
        raise ReserveTopupError("overdraw_factor must be finite and >= 1")
    primary, primary_by_key, primary_sources = _normalize_plan(
        primary_by_split, expected_partition="primary"
    )
    reserve, reserve_by_key, reserve_sources = _normalize_plan(
        reserve_by_split, expected_partition="reserve"
    )
    duplicate_keys = set(primary_by_key) & set(reserve_by_key)
    if duplicate_keys:
        raise ReserveTopupError(
            f"primary/reserve selection-key overlap: {sorted(duplicate_keys)[:10]}"
        )
    primary_all_sources = primary_sources["train"] | primary_sources["eval"]
    reserve_all_sources = reserve_sources["train"] | reserve_sources["eval"]
    source_overlap = primary_all_sources & reserve_all_sources
    primary_cross = primary_sources["train"] & primary_sources["eval"]
    reserve_cross = reserve_sources["train"] & reserve_sources["eval"]
    if source_overlap or primary_cross or reserve_cross:
        raise ReserveTopupError(
            "source-disjoint contract failed: "
            f"primary_reserve={sorted(source_overlap)[:5]}, "
            f"primary_cross={sorted(primary_cross)[:5]}, "
            f"reserve_cross={sorted(reserve_cross)[:5]}"
        )
    outcomes, protocols = collect_quality_outcomes(
        quality_rows, accepted_tiers=accepted_tiers
    )
    known_keys = set(primary_by_key) | set(reserve_by_key)
    unknown_outcomes = set(outcomes) - known_keys
    if unknown_outcomes:
        raise ReserveTopupError(
            f"quality outcomes are absent from frozen plans: {sorted(unknown_outcomes)[:10]}"
        )
    missing_primary = set(primary_by_key) - set(outcomes)
    if require_terminal_primary and missing_primary:
        raise ReserveTopupError(
            f"primary quality audit is incomplete: {len(missing_primary)} rows; "
            f"examples={sorted(missing_primary)[:10]}"
        )

    primary_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    reserve_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for split in ("train", "eval"):
        for row in primary[split]:
            primary_grouped[(split, str(row["coverage_label"]))].append(row)
        for row in reserve[split]:
            reserve_grouped[(split, str(row["coverage_label"]))].append(row)

    artifacts: dict[str, dict[str, list[dict[str, Any]]]] = {
        "accepted_primary": {"train": [], "eval": []},
        "accepted_reserve_topups": {"train": [], "eval": []},
        "pending_reserve_requests": {"train": [], "eval": []},
        "joint_materialization": {"train": [], "eval": []},
    }
    per_class: list[dict[str, Any]] = []
    shortages: list[dict[str, Any]] = []
    labels_and_mids = {
        (str(row["coverage_label"]), str(row["coverage_mid"]))
        for row in [*primary["train"], *primary["eval"], *reserve["train"], *reserve["eval"]]
    }
    mid_by_label: dict[str, str] = {}
    for label, mid in labels_and_mids:
        previous = mid_by_label.setdefault(label, mid)
        if previous != mid:
            raise ReserveTopupError(f"label maps to multiple MIDs: {label}")

    for split in ("train", "eval"):
        targets = targets_by_split_label.get(split, {})
        if not targets:
            raise ReserveTopupError(f"target contract is empty for {split}")
        extra_primary_labels = {
            label for (row_split, label) in primary_grouped if row_split == split
        } - set(targets)
        if extra_primary_labels:
            raise ReserveTopupError(
                f"primary {split} labels absent from target contract: "
                f"{sorted(extra_primary_labels)[:10]}"
            )
        for label in sorted(targets):
            target = int(targets[label])
            if target <= 0:
                raise ReserveTopupError(f"non-positive target for {split}/{label}")
            planned_primary = sorted(
                primary_grouped.get((split, label), []), key=_primary_sort_key
            )
            if len(planned_primary) != target:
                raise ReserveTopupError(
                    f"primary contract is not exact for {split}/{label}: "
                    f"{len(planned_primary)} != {target}"
                )
            accepted_primary = [
                row
                for row in planned_primary
                if outcomes.get(_selection_key(row), {}).get("accepted") is True
            ]
            artifacts["accepted_primary"][split].extend(accepted_primary)
            remaining = target - len(accepted_primary)

            candidates = sorted(
                reserve_grouped.get((split, label), []), key=_reserve_sort_key
            )
            accepted_reserve: list[dict[str, Any]] = []
            pending_candidates: list[dict[str, Any]] = []
            saw_unknown = False
            evaluated_after_gap: list[str] = []
            for row in candidates:
                outcome = outcomes.get(_selection_key(row))
                if outcome is None:
                    saw_unknown = True
                    pending_candidates.append(row)
                    continue
                if saw_unknown:
                    evaluated_after_gap.append(_selection_key(row))
                    continue
                if outcome["accepted"] and remaining > 0:
                    accepted_reserve.append(row)
                    remaining -= 1
            if evaluated_after_gap:
                raise ReserveTopupError(
                    "reserve quality evaluation skipped earlier frozen candidates for "
                    f"{split}/{label}: {evaluated_after_gap[:5]}"
                )
            artifacts["accepted_reserve_topups"][split].extend(accepted_reserve)
            request_count = min(
                len(pending_candidates), int(math.ceil(remaining * overdraw_factor))
            )
            pending = pending_candidates[:request_count]
            for request_rank, row in enumerate(pending, start=1):
                emitted = dict(row)
                emitted.update(
                    {
                        "topup_format": FORMAT,
                        "topup_request_rank": request_rank,
                        "topup_deficit_before_wave": remaining,
                        "topup_overdraw_factor": float(overdraw_factor),
                        "topup_selection_policy": (
                            "ambiguity_tier_then_frozen_reserve_rank_then_selection_key"
                        ),
                    }
                )
                artifacts["pending_reserve_requests"][split].append(emitted)
            shortfall_after_available = max(0, remaining - len(pending_candidates))
            if shortfall_after_available:
                shortages.append(
                    {
                        "metadata_split": split,
                        "label": label,
                        "missing_after_known_acceptance": remaining,
                        "unevaluated_reserve_available": len(pending_candidates),
                        "hard_shortage": shortfall_after_available,
                    }
                )
            per_class.append(
                {
                    "metadata_split": split,
                    "label": label,
                    "target": target,
                    "primary_accepted": len(accepted_primary),
                    "primary_rejected": sum(
                        outcomes.get(_selection_key(row), {}).get("accepted") is False
                        for row in planned_primary
                    ),
                    "reserve_accepted_selected": len(accepted_reserve),
                    "deficit_before_pending_wave": remaining,
                    "pending_reserve_requested": len(pending),
                    "unevaluated_reserve_available": len(pending_candidates),
                    "reserve_tier_counts": dict(
                        sorted(Counter(int(row["ambiguity_tier"]) for row in candidates).items())
                    ),
                    "quota_ready": remaining == 0,
                }
            )

    selected_mids = set(mid_by_label.values())
    artifacts["joint_materialization"] = build_joint_topup_plan(
        pending_by_split=artifacts["pending_reserve_requests"],
        events_by_split=events_by_split,
        selected_mids=selected_mids,
    )

    final_sources = {
        split: {
            str(row["video_id"])
            for kind in ("accepted_primary", "accepted_reserve_topups")
            for row in artifacts[kind][split]
        }
        for split in ("train", "eval")
    }
    final_overlap = final_sources["train"] & final_sources["eval"]
    if final_overlap:
        raise ReserveTopupError(
            f"accepted topup creates train/eval source overlap: {sorted(final_overlap)[:10]}"
        )
    counts = {
        kind: {
            split: len(artifacts[kind][split]) for split in ("train", "eval")
        }
        for kind in artifacts
    }
    receipt = {
        "format": FORMAT,
        "audit_passes": not shortages,
        "configuration": {
            "overdraw_factor": float(overdraw_factor),
            "accepted_tiers": list(accepted_tiers),
            "require_terminal_primary": bool(require_terminal_primary),
            "selection_order": (
                "ambiguity_tier_then_frozen_reserve_rank_then_selection_key"
            ),
        },
        "fixed_quality_protocols": sorted(protocols),
        "targets": {
            split: {label: int(value) for label, value in sorted(targets_by_split_label[split].items())}
            for split in ("train", "eval")
        },
        "counts": counts,
        "quality_outcomes": {
            "total": len(outcomes),
            "accepted": sum(bool(value["accepted"]) for value in outcomes.values()),
            "rejected": sum(not bool(value["accepted"]) for value in outcomes.values()),
            "missing_primary": len(missing_primary),
        },
        "per_class": per_class,
        "hard_shortages": shortages,
        "primary_reserve_source_overlap": 0,
        "train_eval_accepted_source_overlap": 0,
        "pending_class_video_rows": sum(
            counts["pending_reserve_requests"].values()
        ),
        "pending_joint_source_video_rows": sum(
            counts["joint_materialization"].values()
        ),
        "artifacts_sha256": {
            kind: _sha256(
                [
                    *artifacts[kind]["train"],
                    *artifacts[kind]["eval"],
                ]
            )
            for kind in artifacts
        },
        "invariants": {
            "reserve_order_frozen_before_quality_outcomes": True,
            "uses_only_fixed_acoustic_acceptance": True,
            "numeric_quality_scores_used_for_selection": False,
            "qa_or_detector_outcomes_used_for_selection": False,
            "primary_and_reserve_sources_disjoint": True,
            "official_train_and_eval_sources_disjoint": True,
            "joint_rows_preserve_exact_preindexed_routes": True,
            "overdraw_declared_before_reserve_quality_evaluation": True,
        },
    }
    return artifacts, receipt


__all__ = [
    "DEFAULT_ACCEPTED_TIERS",
    "FORMAT",
    "JOINT_FORMAT",
    "ReserveTopupError",
    "build_joint_topup_plan",
    "build_reserve_quality_topup",
    "collect_quality_outcomes",
    "constant_targets",
    "flatten_crop_rows",
    "load_jsonl",
    "sha256_file",
    "targets_from_primary_contract",
]
