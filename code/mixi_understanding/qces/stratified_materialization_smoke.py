"""Deterministic, route-bounded smoke plans for AudioSet materialization.

The availability-aware crop contract is ultimately converted to one joint row
per physical source video.  A smoke run must exercise every active mirror
without accidentally becoming a full download.  This module selects at most
two source rows per route, preserves their exact pre-indexed locations, and
records the split/label/tier coverage used for the selection.

Native sample rate is deliberately *not* inferred from a mirror name.  When a
previous materialization manifest is supplied, its decoded sample-rate field
can be used to include 44.1 and 48 kHz sources.  Without such a manifest the
selector falls back to route, split, class and ambiguity-tier diversity.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORMAT = "qces_stratified_materialization_smoke_v1"
TARGET_NATIVE_SAMPLE_RATES = (44_100, 48_000)


class StratifiedSmokeError(ValueError):
    """Raised when a bounded smoke contract cannot be made auditable."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_key(value: Any, *, seed: int) -> str:
    return hashlib.sha256(
        str(seed).encode("ascii") + b"\0" + _canonical(value)
    ).hexdigest()


def route_slug(route: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", route.lower()).strip("_")
    if not slug:
        raise StratifiedSmokeError(f"cannot derive slug for source route {route!r}")
    return slug


def load_jsonl(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise StratifiedSmokeError(f"input JSONL does not exist: {path}")
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise StratifiedSmokeError(
                        f"invalid JSON at {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise StratifiedSmokeError(
                        f"expected object at {path}:{line_number}"
                    )
                rows.append(row)
                count += 1
        receipts.append(
            {"path": str(path), "sha256": sha256_file(path), "rows": count}
        )
    return rows, receipts


def _profile(row: Mapping[str, Any]) -> dict[str, Any]:
    route = str(row.get("source_route") or "")
    split = str(row.get("metadata_split") or "")
    hf_split = str(row.get("hf_split") or "")
    video_id = str(row.get("video_id") or "")
    dataset = str(row.get("hf_dataset") or "")
    revision = str(row.get("hf_revision") or "")
    location = row.get("availability_location")
    events = row.get("events")
    requests = row.get("crop_requests")
    if (
        not route
        or split not in {"train", "eval"}
        or hf_split not in {"train", "test"}
        or not video_id
        or not dataset
        or not revision
        or not isinstance(location, Mapping)
        or not isinstance(events, list)
        or not events
        or not isinstance(requests, list)
        or not requests
    ):
        raise StratifiedSmokeError(
            f"incomplete route-aware joint plan row for video_id={video_id!r}"
        )
    expected_hf_split = "train" if split == "train" else "test"
    if hf_split != expected_hf_split:
        raise StratifiedSmokeError(
            f"metadata/HF split mismatch for {video_id}: {split}/{hf_split}"
        )
    for field, expected in (
        ("source_route", route),
        ("hf_dataset", dataset),
        ("hf_revision", revision),
        ("hf_split", hf_split),
        ("video_id", video_id),
    ):
        if str(location.get(field) or "") != expected:
            raise StratifiedSmokeError(
                f"availability_location {field} mismatch for {video_id}"
            )
    if (
        not str(location.get("parquet_url") or "")
        or int(location.get("row_group", -1)) < 0
        or int(location.get("row_index", -1)) < 0
    ):
        raise StratifiedSmokeError(
            f"pre-indexed location is incomplete for {video_id}"
        )
    labels = sorted(
        {
            str(value.get("coverage_label") or "")
            for value in requests
            if isinstance(value, Mapping) and str(value.get("coverage_label") or "")
        }
    )
    tiers = sorted(
        {
            int(value["ambiguity_tier"])
            for value in requests
            if isinstance(value, Mapping) and "ambiguity_tier" in value
        }
    )
    if not labels or not tiers:
        raise StratifiedSmokeError(
            f"crop requests lack coverage label/tier for {video_id}"
        )
    return {
        "source_route": route,
        "metadata_split": split,
        "hf_split": hf_split,
        "video_id": video_id,
        "hf_dataset": dataset,
        "hf_revision": revision,
        "coverage_labels": labels,
        "ambiguity_tiers": tiers,
        "crop_requests": len(requests),
    }


def sample_rates_from_manifests(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Read only explicitly decoded source rates from materialized manifests."""

    output: dict[str, int] = {}
    for row in rows:
        video_id = str(row.get("video_id") or row.get("source_video_id") or "")
        if not video_id:
            continue
        rates: set[int] = set()
        for field in ("source_sample_rate", "input_sample_rate", "sample_rate"):
            value = row.get(field)
            if value is not None and int(value) > 0:
                rates.add(int(value))
        for crop in row.get("crop_records") or []:
            if not isinstance(crop, Mapping):
                continue
            for field in ("source_sample_rate", "input_sample_rate", "sample_rate"):
                value = crop.get(field)
                if value is not None and int(value) > 0:
                    rates.add(int(value))
        if not rates:
            continue
        if len(rates) != 1:
            raise StratifiedSmokeError(
                f"materialized manifest has conflicting sample rates for {video_id}: "
                f"{sorted(rates)}"
            )
        rate = next(iter(rates))
        previous = output.get(video_id)
        if previous is not None and previous != rate:
            raise StratifiedSmokeError(
                f"sample rate changed across manifests for {video_id}: {previous}/{rate}"
            )
        output[video_id] = rate
    return output


def _constraint_mask(profile: Mapping[str, Any], sample_rates: Mapping[str, int]) -> int:
    mask = 1 if str(profile["metadata_split"]) == "train" else 2
    rate = sample_rates.get(str(profile["video_id"]))
    if rate == 44_100:
        mask |= 4
    elif rate == 48_000:
        mask |= 8
    return mask


def _option_key(option: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -len(option["labels"]),
        -len(option["tiers"]),
        str(option["stable"]),
    )


def _route_options(
    candidates: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    target_count: int,
    sample_rates: Mapping[str, int],
    seed: int,
    representatives_per_constraint: int = 16,
    options_per_mask: int = 16,
) -> list[dict[str, Any]]:
    """Compress a large route to diverse exact-size options over four bits."""

    buckets: dict[int, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for value in candidates:
        buckets[_constraint_mask(value[1], sample_rates)].append(value)
    reduced: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for mask in sorted(buckets):
        values = sorted(
            buckets[mask],
            key=lambda value: (
                int(min(value[1]["ambiguity_tiers"])),
                stable_key(value[0], seed=seed),
                str(value[1]["video_id"]),
            ),
        )
        chosen: list[tuple[dict[str, Any], dict[str, Any]]] = []
        labels: set[str] = set()
        tiers: set[int] = set()
        while values and len(chosen) < representatives_per_constraint:
            best = min(
                values,
                key=lambda value: (
                    -len(set(value[1]["coverage_labels"]) - labels),
                    -len(set(value[1]["ambiguity_tiers"]) - tiers),
                    stable_key(value[0], seed=seed),
                ),
            )
            values.remove(best)
            chosen.append(best)
            labels.update(best[1]["coverage_labels"])
            tiers.update(best[1]["ambiguity_tiers"])
        reduced.extend(chosen)
    by_mask: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for combination in itertools.combinations(reduced, target_count):
        video_ids = [str(value[1]["video_id"]) for value in combination]
        if len(video_ids) != len(set(video_ids)):
            continue
        mask = 0
        labels: set[str] = set()
        tiers: set[int] = set()
        for row, profile in combination:
            mask |= _constraint_mask(profile, sample_rates)
            labels.update(profile["coverage_labels"])
            tiers.update(profile["ambiguity_tiers"])
        stable = hashlib.sha256(
            "|".join(
                sorted(stable_key(row, seed=seed) for row, _ in combination)
            ).encode("ascii")
        ).hexdigest()
        option = {
            "values": tuple(combination),
            "mask": mask,
            "labels": frozenset(labels),
            "tiers": frozenset(tiers),
            "stable": stable,
        }
        by_mask[mask].append(option)
    output: list[dict[str, Any]] = []
    for mask in sorted(by_mask):
        output.extend(sorted(by_mask[mask], key=_option_key)[:options_per_mask])
    if not output:
        raise StratifiedSmokeError("route option compression produced no smoke subset")
    return output


def _distinct_label_assignment(profiles: Sequence[Mapping[str, Any]]) -> list[str] | None:
    """Return one unique coverage label per selected row when a matching exists."""

    label_to_row: dict[str, int] = {}

    def assign(row_index: int, visited: set[str]) -> bool:
        for label in sorted(set(profiles[row_index]["coverage_labels"])):
            if label in visited:
                continue
            visited.add(label)
            previous = label_to_row.get(label)
            if previous is None or assign(previous, visited):
                label_to_row[label] = row_index
                return True
        return False

    for index in range(len(profiles)):
        if not assign(index, set()):
            return None
    output = [""] * len(profiles)
    for label, index in label_to_row.items():
        output[index] = label
    return output if all(output) else None


def select_stratified_smoke(
    rows: Iterable[Mapping[str, Any]],
    *,
    records_per_route: int = 2,
    seed: int = 2028,
    known_sample_rates: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one or two exact source-video rows per active route."""

    if records_per_route not in {1, 2}:
        raise StratifiedSmokeError("records_per_route must be 1 or 2")
    sample_rates = {
        str(key): int(value) for key, value in (known_sample_rates or {}).items()
    }
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    source_contract: dict[str, tuple[str, str]] = {}
    seen_canonical: dict[str, bytes] = {}
    split_by_source: dict[str, str] = {}
    for raw in rows:
        row = dict(raw)
        profile = _profile(row)
        video_id = str(profile["video_id"])
        payload = _canonical(row)
        if video_id in seen_canonical:
            if seen_canonical[video_id] != payload:
                raise StratifiedSmokeError(
                    f"conflicting duplicate source video in input: {video_id}"
                )
            continue
        if video_id in split_by_source and split_by_source[video_id] != profile["metadata_split"]:
            raise StratifiedSmokeError(
                f"source video crosses train/eval: {video_id}"
            )
        seen_canonical[video_id] = payload
        split_by_source[video_id] = str(profile["metadata_split"])
        route = str(profile["source_route"])
        contract = (str(profile["hf_dataset"]), str(profile["hf_revision"]))
        previous = source_contract.get(route)
        if previous is not None and previous != contract:
            raise StratifiedSmokeError(
                f"route has conflicting dataset/revision: {route}"
            )
        source_contract[route] = contract
        grouped[route].append((row, profile))
    if not grouped:
        raise StratifiedSmokeError("joint primary plan is empty")
    slugs = [route_slug(route) for route in grouped]
    if len(slugs) != len(set(slugs)):
        raise StratifiedSmokeError("active source routes have colliding slugs")

    route_order = sorted(grouped)
    route_options = {
        route: _route_options(
            grouped[route],
            target_count=min(records_per_route, len(grouped[route])),
            sample_rates=sample_rates,
            seed=seed,
        )
        for route in route_order
    }
    # Keep a small deterministic Pareto beam for every four-bit coverage mask.
    beam: dict[int, list[dict[str, Any]]] = {
        0: [
            {
                "values": tuple(),
                "mask": 0,
                "labels": frozenset(),
                "tiers": frozenset(),
                "stable": "",
            }
        ]
    }
    for route in route_order:
        updated: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for existing_values in beam.values():
            for existing in existing_values:
                for option in route_options[route]:
                    mask = int(existing["mask"]) | int(option["mask"])
                    combined = {
                        "values": tuple(existing["values"]) + tuple(option["values"]),
                        "mask": mask,
                        "labels": frozenset(set(existing["labels"]) | set(option["labels"])),
                        "tiers": frozenset(set(existing["tiers"]) | set(option["tiers"])),
                        "stable": hashlib.sha256(
                            (str(existing["stable"]) + str(option["stable"])).encode("ascii")
                        ).hexdigest(),
                    }
                    updated[mask].append(combined)
        beam = {
            mask: sorted(values, key=_option_key)[:32]
            for mask, values in updated.items()
        }
    available_splits = sorted(
        {str(profile["metadata_split"]) for values in grouped.values() for _, profile in values}
    )
    split_mask = (1 if "train" in available_splits else 0) | (
        2 if "eval" in available_splits else 0
    )
    known_target_rates = sorted(
        {
            rate
            for video_id, rate in sample_rates.items()
            if video_id in seen_canonical and rate in TARGET_NATIVE_SAMPLE_RATES
        }
    )
    rate_mask = (4 if 44_100 in known_target_rates else 0) | (
        8 if 48_000 in known_target_rates else 0
    )

    def global_key(option: Mapping[str, Any]) -> tuple[Any, ...]:
        mask = int(option["mask"])
        return (
            -int((mask & split_mask) == split_mask),
            -int((mask & rate_mask).bit_count()),
            -len(option["labels"]),
            -len(option["tiers"]),
            str(option["stable"]),
        )

    chosen_global = min(
        [option for values in beam.values() for option in values],
        key=global_key,
    )
    selected = list(chosen_global["values"])
    selected_labels = set(chosen_global["labels"])
    selected_tiers = set(chosen_global["tiers"])
    selected_by_route: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for value in selected:
        selected_by_route[str(value[1]["source_route"])].append(value)
    route_summaries: dict[str, Any] = {}
    for route in route_order:
        candidates = grouped[route]
        route_selected = selected_by_route[route]
        route_summaries[route] = {
            "slug": route_slug(route),
            "hf_dataset": source_contract[route][0],
            "hf_revision": source_contract[route][1],
            "available_source_videos": len(candidates),
            "available_metadata_splits": sorted(
                {str(value[1]["metadata_split"]) for value in candidates}
            ),
            "selected_source_videos": len(route_selected),
            "selected_video_ids": sorted(
                str(value[1]["video_id"]) for value in route_selected
            ),
            "selected_metadata_splits": sorted(
                {str(value[1]["metadata_split"]) for value in route_selected}
            ),
            "selected_crop_requests": sum(
                int(value[1]["crop_requests"]) for value in route_selected
            ),
        }

    selected.sort(
        key=lambda value: (
            str(value[1]["source_route"]),
            str(value[1]["metadata_split"]),
            str(value[1]["video_id"]),
        )
    )
    selected_rows = [value[0] for value in selected]
    selected_profiles = [value[1] for value in selected]
    selected_ids = [str(value["video_id"]) for value in selected_profiles]
    if len(selected_ids) != len(set(selected_ids)):
        raise StratifiedSmokeError("selected smoke sources overlap")
    selected_by_split = {
        split: {
            str(value["video_id"])
            for value in selected_profiles
            if str(value["metadata_split"]) == split
        }
        for split in ("train", "eval")
    }
    cross_split_overlap = selected_by_split["train"] & selected_by_split["eval"]
    if cross_split_overlap:
        raise StratifiedSmokeError(
            f"selected train/eval source overlap: {sorted(cross_split_overlap)}"
        )
    selected_splits = sorted(
        {str(profile["metadata_split"]) for profile in selected_profiles}
    )
    split_diversity_pass = set(selected_splits) == set(available_splits)
    available_labels = {
        str(label)
        for values in grouped.values()
        for _, profile in values
        for label in profile["coverage_labels"]
    }
    assigned_labels = _distinct_label_assignment(selected_profiles)
    selected_primary_labels = assigned_labels or [
        sorted(profile["coverage_labels"])[0] for profile in selected_profiles
    ]
    label_diversity_feasible = len(available_labels) >= len(selected_profiles)
    label_diversity_pass = (
        len(set(selected_primary_labels)) == len(selected_profiles)
        if label_diversity_feasible
        else True
    )
    selected_known_rates = sorted(
        {
            sample_rates[video_id]
            for video_id in selected_ids
            if video_id in sample_rates
        }
    )
    native_rate_pass = set(known_target_rates).issubset(selected_known_rates)
    report = {
        "format": FORMAT,
        "audit_passes": bool(
            split_diversity_pass
            and label_diversity_pass
            and native_rate_pass
            and not cross_split_overlap
        ),
        "configuration": {
            "records_per_route": records_per_route,
            "seed": seed,
            "target_native_sample_rates_if_known": list(TARGET_NATIVE_SAMPLE_RATES),
            "sample_rate_policy": (
                "decoded_manifest_only;never_infer_from_route;otherwise_route_split_label_tier"
            ),
        },
        "active_routes": len(grouped),
        "selected_source_videos": len(selected_profiles),
        "selected_crop_requests": sum(
            int(value["crop_requests"]) for value in selected_profiles
        ),
        "available_metadata_splits": available_splits,
        "selected_metadata_splits": selected_splits,
        "split_counts": dict(
            sorted(Counter(str(value["metadata_split"]) for value in selected_profiles).items())
        ),
        "selected_stratification_labels": selected_primary_labels,
        "selected_unique_stratification_labels": len(set(selected_primary_labels)),
        "selected_coverage_labels": sorted(selected_labels),
        "selected_ambiguity_tiers": sorted(selected_tiers),
        "known_target_native_sample_rates": known_target_rates,
        "selected_known_native_sample_rates": selected_known_rates,
        "selected_video_sample_rates": {
            video_id: sample_rates[video_id]
            for video_id in selected_ids
            if video_id in sample_rates
        },
        "train_eval_source_overlap": len(cross_split_overlap),
        "route_selection": route_summaries,
        "invariants": {
            "one_or_two_source_videos_per_active_route": True,
            "exact_active_route_coverage": len(route_summaries) == len(grouped),
            "available_split_diversity_preserved": split_diversity_pass,
            "unique_stratification_label_per_row_when_feasible": label_diversity_pass,
            "known_44k1_48k_covered_when_feasible": native_rate_pass,
            "sample_rate_never_inferred_from_route": True,
            "cross_split_source_overlap": len(cross_split_overlap),
            "preindexed_locations_preserved_verbatim": True,
        },
    }
    return selected_rows, report


def aggregate_scan_receipts(
    *,
    smoke_receipt: Mapping[str, Any],
    route_receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and aggregate route-specific materializer scan receipts."""

    expected_routes = smoke_receipt.get("route_outputs")
    if not isinstance(expected_routes, Mapping) or not expected_routes:
        raise StratifiedSmokeError("smoke receipt has no route_outputs")
    missing = sorted(set(expected_routes) - set(route_receipts))
    extra = sorted(set(route_receipts) - set(expected_routes))
    if missing or extra:
        raise StratifiedSmokeError(
            f"route receipt mismatch: missing={missing}, extra={extra}"
        )
    estimate_fields = (
        "video_id_scan_compressed_column_bytes_estimate",
        "matching_audio_row_groups_compressed_column_bytes_estimate",
        "total_compressed_column_bytes_estimate",
        "total_full_shard_payload_bytes_no_retry_ceiling",
        "planned_audio_disk_bytes_estimate",
    )
    totals = Counter()
    routes: dict[str, Any] = {}
    all_safe = True
    for route in sorted(expected_routes):
        expected = expected_routes[route]
        receipt = route_receipts[route]
        if not bool(receipt.get("scan_only")):
            raise StratifiedSmokeError(f"route {route} receipt is not scan-only")
        if not bool(receipt.get("require_preindexed_locations")):
            raise StratifiedSmokeError(
                f"route {route} did not require pre-indexed locations"
            )
        if int(receipt.get("plan_rows", -1)) != int(expected["rows"]):
            raise StratifiedSmokeError(f"route {route} plan row count changed")
        plan_files = receipt.get("plan_files") or []
        hashes = {str(value.get("sha256") or "") for value in plan_files}
        if str(expected["sha256"]) not in hashes:
            raise StratifiedSmokeError(f"route {route} plan hash is absent from scan receipt")
        if str(receipt.get("hf_dataset") or "") != str(expected["hf_dataset"]):
            raise StratifiedSmokeError(f"route {route} dataset contract changed")
        if str(receipt.get("resolved_revision") or "") != str(expected["hf_revision"]):
            raise StratifiedSmokeError(f"route {route} revision contract changed")
        validated_rows = int(
            (receipt.get("preindexed_location_summary") or {}).get(
                "validated_plan_rows", 0
            )
        )
        if validated_rows != int(expected["rows"]):
            raise StratifiedSmokeError(
                f"route {route} did not validate every pre-indexed row: "
                f"{validated_rows} != {expected['rows']}"
            )
        if int(receipt.get("plan_rows_unaccounted_for", 0)) != 0:
            raise StratifiedSmokeError(f"route {route} has unaccounted plan rows")
        estimates = receipt.get("estimates") or {}
        route_estimates = {}
        for field in estimate_fields:
            value = int(estimates.get(field, 0))
            route_estimates[field] = value
            totals[field] += value
        safe = bool((receipt.get("disk_preflight") or {}).get("safe"))
        all_safe = all_safe and safe
        routes[route] = {
            "slug": str(expected["slug"]),
            "plan_rows": int(receipt["plan_rows"]),
            "preindexed_rows_validated": validated_rows,
            "disk_preflight_safe": safe,
            "estimates": route_estimates,
        }
    return {
        "format": "qces_stratified_materialization_scan_cost_v1",
        "audit_passes": all_safe,
        "routes": routes,
        "totals": {field: int(totals[field]) for field in estimate_fields},
        "all_route_disk_preflights_safe": all_safe,
        "notes": (
            "compressed-column estimates exclude protocol overhead, range-cache "
            "effects and retry transfers"
        ),
    }


def audit_remote_smoke_manifests(
    *,
    smoke_receipt: Mapping[str, Any],
    route_manifests: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Prove that the bounded smoke actually read each selected remote row.

    Existing manifests are intentionally forbidden here.  They are useful for
    the later full transfer, but a route-equivalence smoke has to exercise the
    exact Parquet row and verify decoded crop timestamps on every active route.
    """

    expected_routes = smoke_receipt.get("route_outputs")
    selections = smoke_receipt.get("route_selection")
    if not isinstance(expected_routes, Mapping) or not isinstance(selections, Mapping):
        raise StratifiedSmokeError("smoke receipt lacks route contracts/selections")
    missing = sorted(set(expected_routes) - set(route_manifests))
    extra = sorted(set(route_manifests) - set(expected_routes))
    if missing or extra:
        raise StratifiedSmokeError(
            f"remote manifest route mismatch: missing={missing}, extra={extra}"
        )
    rates = Counter()
    routes: dict[str, Any] = {}
    all_video_ids: set[str] = set()
    total_crops = 0
    for route in sorted(expected_routes):
        contract = expected_routes[route]
        transactions = [dict(value) for value in route_manifests[route]]
        expected_ids = set(str(value) for value in selections[route]["selected_video_ids"])
        observed_ids = {str(value.get("video_id") or "") for value in transactions}
        if observed_ids != expected_ids:
            raise StratifiedSmokeError(
                f"route {route} materialized video IDs changed: "
                f"expected={sorted(expected_ids)}, observed={sorted(observed_ids)}"
            )
        if all_video_ids & observed_ids:
            raise StratifiedSmokeError(
                f"remote smoke source reused across routes: {sorted(all_video_ids & observed_ids)}"
            )
        all_video_ids.update(observed_ids)
        route_crop_count = 0
        route_rates: Counter[int] = Counter()
        for transaction in transactions:
            provenance = transaction.get("source_provenance") or {}
            if str(provenance.get("transport") or "") != "hf_parquet_row_group":
                raise StratifiedSmokeError(
                    f"route {route}/{transaction.get('video_id')} did not read remote Parquet"
                )
            for field, expected in (
                ("source_route", route),
                ("hf_dataset", str(contract["hf_dataset"])),
                ("hf_revision", str(contract["hf_revision"])),
            ):
                if str(provenance.get(field) or "") != expected:
                    raise StratifiedSmokeError(
                        f"remote provenance {field} changed for {transaction.get('video_id')}"
                    )
            crops = transaction.get("crop_records")
            if not isinstance(crops, list) or not crops:
                raise StratifiedSmokeError(
                    f"remote transaction has no crops: {transaction.get('video_id')}"
                )
            if int(transaction.get("crop_item_count", -1)) != len(crops):
                raise StratifiedSmokeError(
                    f"crop count mismatch for {transaction.get('video_id')}"
                )
            for crop in crops:
                rate = int(crop.get("sample_rate", 0))
                if rate <= 0:
                    raise StratifiedSmokeError("remote crop lacks decoded sample rate")
                rates[rate] += 1
                route_rates[rate] += 1
                retained = float(crop.get("coverage_event_retained_fraction", -1.0))
                if retained < 1.0 - 1e-9:
                    raise StratifiedSmokeError(
                        f"coverage timestamp clipped for {crop.get('materialization_item_id')}: "
                        f"{retained}"
                    )
                label = str(crop.get("coverage_label") or "")
                mid = str(crop.get("coverage_mid") or "")
                matching_events = [
                    event
                    for event in crop.get("all_strong_events") or []
                    if str(event.get("label") or "") == label
                    and str(event.get("audioset_mid") or "") == mid
                    and float(event.get("offset_seconds", 0.0))
                    > float(event.get("onset_seconds", 0.0))
                ]
                if not matching_events:
                    raise StratifiedSmokeError(
                        f"coverage event/timestamp absent after crop: "
                        f"{crop.get('materialization_item_id')}"
                    )
                crop_provenance = crop.get("source_provenance") or {}
                if str(crop_provenance.get("transport") or "") != "hf_parquet_row_group":
                    raise StratifiedSmokeError(
                        f"crop reused non-remote source: {crop.get('materialization_item_id')}"
                    )
            route_crop_count += len(crops)
        if route_crop_count != int(contract["crop_requests"]):
            raise StratifiedSmokeError(
                f"route {route} crop request count changed: "
                f"{route_crop_count} != {contract['crop_requests']}"
            )
        total_crops += route_crop_count
        routes[route] = {
            "source_videos": len(transactions),
            "crop_requests": route_crop_count,
            "native_sample_rate_counts": {
                str(key): value for key, value in sorted(route_rates.items())
            },
            "transport": "hf_parquet_row_group",
            "coverage_timestamps_complete": True,
        }
    if total_crops != int(smoke_receipt["selected_crop_requests"]):
        raise StratifiedSmokeError(
            f"total remote crops changed: {total_crops} != "
            f"{smoke_receipt['selected_crop_requests']}"
        )
    required_rates = set(
        int(value) for value in smoke_receipt.get("known_target_native_sample_rates") or []
    )
    observed_rates = set(rates)
    rate_pass = required_rates.issubset(observed_rates)
    return {
        "format": "qces_stratified_remote_materialization_smoke_audit_v1",
        "audit_passes": rate_pass,
        "active_routes": len(routes),
        "source_videos": len(all_video_ids),
        "crop_requests": total_crops,
        "native_sample_rate_counts": {
            str(key): value for key, value in sorted(rates.items())
        },
        "known_target_native_sample_rates_required": sorted(required_rates),
        "known_target_native_sample_rates_observed": sorted(
            required_rates & observed_rates
        ),
        "routes": routes,
        "invariants": {
            "existing_manifest_reuse": 0,
            "all_sources_read_from_remote_parquet": True,
            "exact_selected_video_ids": True,
            "exact_selected_crop_requests": True,
            "coverage_event_and_timestamp_retained": True,
            "known_44k1_48k_covered_when_declared": rate_pass,
        },
    }
