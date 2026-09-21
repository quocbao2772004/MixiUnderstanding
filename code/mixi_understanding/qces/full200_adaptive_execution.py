"""Deterministic execution contract for the full 200-class acoustic source bank.

The existing availability and joint plans are immutable inputs.  This module
only partitions their primary source-video rows into bounded transport/cleaning
chunks and freezes an evaluation cohort before any AudioSep, detector, or QA
outcome exists.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FORMAT = "qces_full200_adaptive_execution_v1"
REGISTRY_FORMAT = "qces_full200_primary_chunk_registry_v1"
EVAL_FORMAT = "qces_full200_fixed_eval_contract_v1"
PROGRESS_FORMAT = "qces_full200_primary_progress_v1"


class Full200ExecutionError(RuntimeError):
    """Raised when an input violates the frozen 200-class contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"))
        handle.write(b"\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(_canonical_bytes(dict(row)))
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Full200ExecutionError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise Full200ExecutionError(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise Full200ExecutionError(
                    f"expected object at {path}:{line_number}"
                )
            rows.append(value)
    return rows


def _slug(value: str) -> str:
    output = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not output:
        raise Full200ExecutionError(f"cannot slug route {value!r}")
    return output


def _selection_key(row: Mapping[str, Any]) -> str:
    value = str(
        row.get("selection_key")
        or row.get("materialization_item_id")
        or row.get("item_id")
        or ""
    )
    if not value:
        raise Full200ExecutionError("crop row has no selection key")
    return value


def _video_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("video_id") or row.get("source_video_id") or "")
    if not value:
        raise Full200ExecutionError("row has no source video id")
    return value


def _validate_flat_plan(
    rows: Sequence[Mapping[str, Any]], *, split: str, partition: str
) -> tuple[set[str], set[str], Counter[str]]:
    keys: set[str] = set()
    sources: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        if str(row.get("metadata_split") or "") != split:
            raise Full200ExecutionError(f"wrong split in {split} {partition} plan")
        if str(row.get("selection_partition") or "") != partition:
            raise Full200ExecutionError(
                f"wrong selection partition in {split} {partition} plan"
            )
        key = _selection_key(row)
        if key in keys:
            raise Full200ExecutionError(f"duplicate selection key: {key}")
        keys.add(key)
        sources.add(_video_id(row))
        label = str(row.get("coverage_label") or "")
        if not label:
            raise Full200ExecutionError(f"missing coverage label for {key}")
        counts[label] += 1
    return keys, sources, counts


def _joint_crop_keys(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        crops = row.get("crop_requests")
        if not isinstance(crops, list) or not crops:
            raise Full200ExecutionError("joint row has no crop requests")
        for crop in crops:
            if not isinstance(crop, Mapping):
                raise Full200ExecutionError("invalid nested crop request")
            key = _selection_key(crop)
            if key in keys:
                raise Full200ExecutionError(f"duplicate joint crop key: {key}")
            keys.add(key)
    return keys


def partition_joint_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_videos: int,
    maximum_crops: int,
) -> list[list[dict[str, Any]]]:
    """Pack complete Parquet row groups without splitting their selected rows.

    The materializer reads the compressed audio column at row-group granularity.
    Keeping a row group in exactly one execution chunk prevents a large hidden
    network multiplier when label-ordered plan rows point into the same group.
    Rows in synthetic/unit-test plans without a location remain individual
    atomic transport units.
    """

    if maximum_videos < 1 or maximum_crops < 1:
        raise ValueError("chunk bounds must be positive")
    transport_units: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    transport_order: list[tuple[Any, ...]] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        location = row.get("availability_location")
        if isinstance(location, Mapping):
            parquet_url = str(location.get("parquet_url") or "")
            row_group = location.get("row_group")
        else:
            parquet_url = ""
            row_group = None
        key: tuple[Any, ...]
        if parquet_url and isinstance(row_group, int):
            key = (parquet_url, int(row_group))
        else:
            key = ("__individual_row__", index)
        if key not in transport_units:
            transport_units[key] = []
            transport_order.append(key)
        transport_units[key].append(row)

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_crops = 0
    for key in transport_order:
        unit = transport_units[key]
        crop_count = sum(len(row.get("crop_requests") or []) for row in unit)
        if crop_count < len(unit):
            raise Full200ExecutionError("joint row has no crop requests")
        should_flush = bool(current) and (
            len(current) + len(unit) > maximum_videos
            or current_crops + crop_count > maximum_crops
        )
        if should_flush:
            chunks.append(current)
            current = []
            current_crops = 0
        current.extend(unit)
        current_crops += crop_count
    if current:
        chunks.append(current)
    return chunks


def build_execution_contract(
    *,
    availability_receipt_path: Path,
    joint_receipt_path: Path,
    primary_train_path: Path,
    primary_eval_path: Path,
    reserve_train_path: Path,
    reserve_eval_path: Path,
    output_dir: Path,
    maximum_videos_per_chunk: int = 320,
    maximum_crops_per_chunk: int = 384,
) -> dict[str, Any]:
    availability_receipt_path = availability_receipt_path.resolve()
    joint_receipt_path = joint_receipt_path.resolve()
    output_dir = output_dir.resolve()
    availability = load_json(availability_receipt_path)
    joint = load_json(joint_receipt_path)
    if not availability.get("audit_passes") or not joint.get("audit_passes"):
        raise Full200ExecutionError("upstream availability/joint receipt did not pass")
    if int(availability.get("selected_classes", -1)) != 200:
        raise Full200ExecutionError("availability contract is not 200 classes")
    if int(joint.get("selected_classes", -1)) != 200:
        raise Full200ExecutionError("joint contract is not 200 classes")
    expected_quota = {"train": 100, "eval": 20}
    if dict(joint.get("exact_class_video_quota") or {}) != expected_quota:
        raise Full200ExecutionError("joint contract is not exact 100/20")
    if int(joint.get("train_eval_source_overlap", -1)) != 0:
        raise Full200ExecutionError("joint train/eval sources overlap")

    primary_paths = {
        "train": primary_train_path.resolve(),
        "eval": primary_eval_path.resolve(),
    }
    reserve_paths = {
        "train": reserve_train_path.resolve(),
        "eval": reserve_eval_path.resolve(),
    }
    primary: dict[str, list[dict[str, Any]]] = {}
    reserve: dict[str, list[dict[str, Any]]] = {}
    primary_keys: dict[str, set[str]] = {}
    reserve_keys: dict[str, set[str]] = {}
    primary_sources: dict[str, set[str]] = {}
    reserve_sources: dict[str, set[str]] = {}
    class_counts: dict[str, Counter[str]] = {}
    reserve_counts: dict[str, Counter[str]] = {}
    for split in ("train", "eval"):
        primary[split] = load_jsonl(primary_paths[split])
        reserve[split] = load_jsonl(reserve_paths[split])
        (
            primary_keys[split],
            primary_sources[split],
            class_counts[split],
        ) = _validate_flat_plan(primary[split], split=split, partition="primary")
        (
            reserve_keys[split],
            reserve_sources[split],
            reserve_counts[split],
        ) = _validate_flat_plan(reserve[split], split=split, partition="reserve")
        expected_labels = set(class_counts[split])
        if len(expected_labels) != 200:
            raise Full200ExecutionError(f"{split} primary has {len(expected_labels)} classes")
        mismatches = {
            label: count
            for label, count in class_counts[split].items()
            if count != expected_quota[split]
        }
        if mismatches:
            raise Full200ExecutionError(
                f"{split} primary quota mismatches: {list(mismatches.items())[:5]}"
            )
    if primary_sources["train"] & primary_sources["eval"]:
        raise Full200ExecutionError("primary source leakage")
    if reserve_sources["train"] & reserve_sources["eval"]:
        raise Full200ExecutionError("reserve source leakage")
    all_primary_sources = primary_sources["train"] | primary_sources["eval"]
    all_reserve_sources = reserve_sources["train"] | reserve_sources["eval"]
    if all_primary_sources & all_reserve_sources:
        raise Full200ExecutionError("primary/reserve source overlap")
    if (primary_keys["train"] | primary_keys["eval"]) & (
        reserve_keys["train"] | reserve_keys["eval"]
    ):
        raise Full200ExecutionError("primary/reserve selection-key overlap")

    registry: list[dict[str, Any]] = []
    joint_keys: dict[str, set[str]] = {"train": set(), "eval": set()}
    route_outputs = joint.get("route_outputs")
    if not isinstance(route_outputs, Mapping) or not route_outputs:
        raise Full200ExecutionError("joint receipt has no route outputs")
    for route, raw_contract in sorted(route_outputs.items()):
        if not isinstance(raw_contract, Mapping):
            raise Full200ExecutionError(f"invalid route contract: {route}")
        route_slug = str(raw_contract.get("slug") or _slug(str(route)))
        dataset = str(raw_contract.get("hf_dataset") or "")
        revision = str(raw_contract.get("hf_revision") or "")
        split_contracts = raw_contract.get("splits")
        if not dataset or not revision or not isinstance(split_contracts, Mapping):
            raise Full200ExecutionError(f"incomplete immutable route: {route}")
        for split, plan_contract in sorted(split_contracts.items()):
            if split not in {"train", "eval"} or not isinstance(plan_contract, Mapping):
                raise Full200ExecutionError(f"invalid route split: {route}/{split}")
            plan_path = Path(str(plan_contract.get("path") or "")).resolve()
            declared_hash = str(plan_contract.get("sha256") or "")
            if sha256_file(plan_path) != declared_hash:
                raise Full200ExecutionError(f"route-plan hash mismatch: {plan_path}")
            rows = load_jsonl(plan_path)
            if len(rows) != int(plan_contract.get("rows", -1)):
                raise Full200ExecutionError(f"route-plan row mismatch: {plan_path}")
            for row in rows:
                if str(row.get("metadata_split") or "") != split:
                    raise Full200ExecutionError(f"route split mismatch: {plan_path}")
                if str(row.get("source_route") or "") != str(route):
                    raise Full200ExecutionError(f"route identity mismatch: {plan_path}")
            joint_keys[split].update(_joint_crop_keys(rows))
            chunks = partition_joint_rows(
                rows,
                maximum_videos=maximum_videos_per_chunk,
                maximum_crops=maximum_crops_per_chunk,
            )
            for index, chunk_rows in enumerate(chunks):
                chunk_id = f"{split}-{route_slug}-{index:04d}"
                chunk_path = output_dir / "plans" / split / f"{chunk_id}.jsonl"
                atomic_jsonl(chunk_path, chunk_rows)
                crops = [
                    dict(crop)
                    for row in chunk_rows
                    for crop in row.get("crop_requests") or []
                ]
                labels = Counter(str(crop.get("coverage_label") or "") for crop in crops)
                transport_groups = {
                    (
                        str((row.get("availability_location") or {}).get("parquet_url") or ""),
                        int((row.get("availability_location") or {}).get("row_group", -1)),
                    )
                    for row in chunk_rows
                }
                registry.append(
                    {
                        "chunk_id": chunk_id,
                        "metadata_split": split,
                        "source_route": str(route),
                        "route_slug": route_slug,
                        "hf_dataset": dataset,
                        "hf_revision": revision,
                        "plan_path": str(chunk_path),
                        "plan_sha256": sha256_file(chunk_path),
                        "source_videos": len(chunk_rows),
                        "crop_requests": len(crops),
                        "transport_row_groups": len(transport_groups),
                        "class_count": len(labels),
                        "class_crop_counts": dict(sorted(labels.items())),
                        "raw_crop_retention": (
                            "temporary_until_terminal_acoustic_receipt"
                            if split == "train"
                            else "persistent_fixed_eval_cohort"
                        ),
                    }
                )
    for split in ("train", "eval"):
        if joint_keys[split] != primary_keys[split]:
            raise Full200ExecutionError(
                f"joint/flat primary mismatch for {split}: "
                f"joint_only={len(joint_keys[split] - primary_keys[split])}, "
                f"flat_only={len(primary_keys[split] - joint_keys[split])}"
            )

    registry_path = output_dir / "primary_chunk_registry.jsonl"
    atomic_jsonl(registry_path, registry)
    fixed_eval_path = output_dir / "fixed_eval_candidates.jsonl"
    atomic_jsonl(fixed_eval_path, primary["eval"])
    evaluation_contract = {
        "format": EVAL_FORMAT,
        "selected_before_acoustic_gate": True,
        "selected_before_detector_or_qa_training": True,
        "candidate_eligibility_uses_audiosep": False,
        "audiosep_grade_is_report_only": True,
        "official_metadata_split": "eval",
        "official_upstream_split": "test",
        "class_count": 200,
        "candidate_rows": len(primary["eval"]),
        "unique_source_videos": len(primary_sources["eval"]),
        "candidate_manifest": {
            "path": str(fixed_eval_path),
            "sha256": sha256_file(fixed_eval_path),
        },
        "required_grade_field": "acceptance_tier",
        "grade_values": ["gold", "silver", "rejected"],
        "downstream_reporting": [
            "fixed_all_candidates_model_independent",
            "gold_only_diagnostic",
            "gold_plus_silver_diagnostic",
        ],
        "manual_audibility_subset_required_before_locked_test_claim": True,
        "train_eval_source_overlap": 0,
    }
    atomic_json(output_dir / "evaluation_contract.json", evaluation_contract)

    contract = {
        "format": FORMAT,
        "audit_passes": True,
        "selected_classes": 200,
        "primary_quota_per_class": expected_quota,
        "primary_crop_rows": {
            split: len(primary[split]) for split in ("train", "eval")
        },
        "primary_unique_source_videos": {
            split: len(primary_sources[split]) for split in ("train", "eval")
        },
        "reserve_crop_rows": {
            split: len(reserve[split]) for split in ("train", "eval")
        },
        "reserve_unique_source_videos": {
            split: len(reserve_sources[split]) for split in ("train", "eval")
        },
        "chunking": {
            "maximum_videos_per_chunk": maximum_videos_per_chunk,
            "maximum_crops_per_chunk": maximum_crops_per_chunk,
            "chunks": len(registry),
            "chunks_by_split": dict(Counter(row["metadata_split"] for row in registry)),
        },
        "adaptive_quality_policy": {
            "initial_partition": "primary_exact_100_train_20_eval",
            "reserve_staged_not_eagerly_downloaded": True,
            "overdraw_factor_per_observed_deficit": 2.0,
            "reserve_order": "ambiguity_tier_then_frozen_reserve_rank_then_selection_key",
            "gate_values_persisted": ["gold", "silver", "rejected"],
            "gate_thresholds_never_relaxed_for_shortage": True,
        },
        "storage_policy": {
            "train_raw_crops": "bounded_scratch_then_delete_after_terminal_receipt",
            "accepted_train_stems": "persistent",
            "eval_raw_crops": "persistent_fixed_cohort",
            "all_quality_outcomes": "persistent",
        },
        "inputs": {
            "availability_receipt": {
                "path": str(availability_receipt_path),
                "sha256": sha256_file(availability_receipt_path),
            },
            "joint_receipt": {
                "path": str(joint_receipt_path),
                "sha256": sha256_file(joint_receipt_path),
            },
            "primary": {
                split: {
                    "path": str(primary_paths[split]),
                    "sha256": sha256_file(primary_paths[split]),
                }
                for split in ("train", "eval")
            },
            "reserve": {
                split: {
                    "path": str(reserve_paths[split]),
                    "sha256": sha256_file(reserve_paths[split]),
                }
                for split in ("train", "eval")
            },
        },
        "outputs": {
            "chunk_registry": {
                "path": str(registry_path),
                "sha256": sha256_file(registry_path),
                "rows": len(registry),
            },
            "evaluation_contract": str(output_dir / "evaluation_contract.json"),
        },
        "invariants": {
            "exact_200_class_100_20_primary_contract": True,
            "primary_reserve_source_overlap": 0,
            "train_eval_source_overlap": 0,
            "qa_or_detector_outcomes_used_for_selection": False,
            "evaluation_candidate_set_frozen_before_audiosep": True,
            "source_video_rows_not_split_across_chunks": True,
            "parquet_row_groups_not_split_across_chunks": True,
        },
    }
    atomic_json(output_dir / "execution_contract.json", contract)
    return contract


def build_progress(*, execution_dir: Path) -> dict[str, Any]:
    execution_dir = execution_dir.resolve()
    registry = load_jsonl(execution_dir / "primary_chunk_registry.jsonl")
    completed: list[dict[str, Any]] = []
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    by_route: dict[str, Counter[str]] = defaultdict(Counter)
    for row in registry:
        chunk_id = str(row["chunk_id"])
        split = str(row["metadata_split"])
        receipt_path = (
            execution_dir / "primary_clean" / split / chunk_id / "source_bank_receipt.json"
        )
        if not receipt_path.is_file():
            continue
        receipt = load_json(receipt_path)
        expected = int(row["crop_requests"])
        if int(receipt.get("input_items", -1)) != expected:
            raise Full200ExecutionError(f"terminal receipt mismatch: {chunk_id}")
        accepted = int(receipt.get("accepted_items", -1))
        rejected = int(receipt.get("rejected_items", -1))
        if accepted < 0 or rejected < 0 or accepted + rejected != expected:
            raise Full200ExecutionError(f"invalid terminal counts: {chunk_id}")
        completed.append(row)
        for counter in (by_split[split], by_route[str(row["source_route"])]):
            counter["chunks"] += 1
            counter["crop_requests"] += expected
            counter["accepted"] += accepted
            counter["rejected"] += rejected
    total_crops = sum(int(row["crop_requests"]) for row in registry)
    done_crops = sum(int(row["crop_requests"]) for row in completed)
    stat = os.statvfs(execution_dir)
    progress = {
        "format": PROGRESS_FORMAT,
        "complete": len(completed) == len(registry),
        "chunks_completed": len(completed),
        "chunks_total": len(registry),
        "crop_requests_completed": done_crops,
        "crop_requests_total": total_crops,
        "percent": 100.0 * done_crops / max(1, total_crops),
        "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        "by_route": {key: dict(value) for key, value in sorted(by_route.items())},
        "free_disk_bytes": int(stat.f_bavail * stat.f_frsize),
    }
    atomic_json(execution_dir / "progress.json", progress)
    return progress
