#!/usr/bin/env python3
"""Build a deterministic AudioSep-quality reserve top-up wave for QCES."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from mixi_understanding.qces.audioset_plan_materializer import (
    atomic_bytes,
    atomic_json,
    atomic_jsonl,
)
from mixi_understanding.qces.reserve_quality_topup import (
    ReserveTopupError,
    build_reserve_quality_topup,
    constant_targets,
    flatten_crop_rows,
    load_jsonl,
    sha256_file,
    targets_from_primary_contract,
)
from mixi_understanding.qces.supported_ontology import load_strong_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN_DIR = PROJECT_ROOT / "outputs/qces_availability_aware_acoustic_plan_v2"
DEFAULT_METADATA_DIR = PROJECT_ROOT / "code/baseline/PretrainedSED/hf_dataset_gen/metadata"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_reserve_quality_topup_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-plan",
        type=Path,
        action="append",
        required=True,
        help=(
            "Repeatable flat crop plan or joint smoke plan. The nested "
            "crop_requests are flattened automatically."
        ),
    )
    parser.add_argument(
        "--reserve-plan",
        type=Path,
        action="append",
        default=None,
        help="Repeatable frozen reserve crop plan; defaults to v2 train/eval reserve.",
    )
    parser.add_argument(
        "--quality-audit",
        type=Path,
        action="append",
        required=True,
        help="Repeatable terminal quality_audit.jsonl (accepted and rejected rows).",
    )
    parser.add_argument(
        "--source-bank",
        type=Path,
        action="append",
        default=[],
        help="Optional accepted source_bank.jsonl for identity/outcome cross-checking.",
    )
    parser.add_argument("--metadata-dir", type=Path, default=DEFAULT_METADATA_DIR)
    parser.add_argument("--target-train-per-class", type=int, default=100)
    parser.add_argument("--target-eval-per-class", type=int, default=20)
    parser.add_argument(
        "--target-from-primary-contract",
        action="store_true",
        help="Smoke-only: derive each class quota from the supplied primary rows.",
    )
    parser.add_argument("--overdraw-factor", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.target_train_per_class <= 0 or args.target_eval_per_class <= 0:
        parser.error("class targets must be positive")
    if args.overdraw_factor < 1.0:
        parser.error("--overdraw-factor must be >= 1")
    return args


def _load_grouped(paths: Sequence[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    receipts: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        raw_rows = load_jsonl(path)
        rows = flatten_crop_rows(raw_rows)
        for row in rows:
            split = str(row.get("metadata_split") or "")
            if split not in grouped:
                raise ReserveTopupError(
                    f"unsupported metadata_split={split!r} in {path}"
                )
            grouped[split].append(row)
        receipts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "input_rows": len(raw_rows),
                "crop_rows": len(rows),
            }
        )
    return grouped, receipts


def _tsv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "metadata_split",
        "label",
        "target",
        "primary_accepted",
        "primary_rejected",
        "reserve_accepted_selected",
        "deficit_before_pending_wave",
        "pending_reserve_requested",
        "unevaluated_reserve_available",
        "quota_ready",
    )
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(str(row.get(field, "")) for field in fields))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not result:
        raise ReserveTopupError(f"cannot create route slug for {value!r}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reserve_paths = tuple(
        args.reserve_plan
        or (
            DEFAULT_PLAN_DIR / "crop_source_reserve_train.jsonl",
            DEFAULT_PLAN_DIR / "crop_source_reserve_eval.jsonl",
        )
    )
    output_dir = args.output_dir.resolve()
    core_paths = {
        "accepted_primary_train": output_dir / "accepted_primary_train.jsonl",
        "accepted_primary_eval": output_dir / "accepted_primary_eval.jsonl",
        "accepted_reserve_train": output_dir / "accepted_reserve_topups_train.jsonl",
        "accepted_reserve_eval": output_dir / "accepted_reserve_topups_eval.jsonl",
        "pending_train": output_dir / "reserve_topup_requests_train.jsonl",
        "pending_eval": output_dir / "reserve_topup_requests_eval.jsonl",
        "joint_train": output_dir / "joint_materialization_plan_train.jsonl",
        "joint_eval": output_dir / "joint_materialization_plan_eval.jsonl",
        "receipt": output_dir / "reserve_topup_receipt.json",
        "table": output_dir / "per_class_topup.tsv",
        "markdown": output_dir / "reserve_topup_receipt.md",
    }
    existing = [str(path) for path in core_paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"outputs already exist; pass --overwrite: {existing}")

    try:
        primary, primary_receipts = _load_grouped(tuple(args.primary_plan))
        reserve, reserve_receipts = _load_grouped(reserve_paths)
        quality_paths = tuple(args.quality_audit) + tuple(args.source_bank)
        quality_rows: list[dict[str, Any]] = []
        quality_receipts: list[dict[str, Any]] = []
        for raw_path in quality_paths:
            path = raw_path.resolve()
            rows = load_jsonl(path)
            quality_rows.extend(rows)
            quality_receipts.append(
                {"path": str(path), "sha256": sha256_file(path), "rows": len(rows)}
            )
        if args.target_from_primary_contract:
            targets = targets_from_primary_contract(primary)
            target_mode = "primary_contract_smoke_only"
        else:
            targets = constant_targets(
                primary,
                train_target=int(args.target_train_per_class),
                eval_target=int(args.target_eval_per_class),
            )
            target_mode = "fixed_production_100_20"
        _, events_by_split = load_strong_metadata(args.metadata_dir.resolve())
        artifacts, receipt = build_reserve_quality_topup(
            primary_by_split=primary,
            reserve_by_split=reserve,
            quality_rows=quality_rows,
            targets_by_split_label=targets,
            events_by_split=events_by_split,
            overdraw_factor=float(args.overdraw_factor),
        )
    except (OSError, ValueError, ReserveTopupError) as error:
        print(f"ERROR: {error}", flush=True)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping = (
        ("accepted_primary", "train", core_paths["accepted_primary_train"]),
        ("accepted_primary", "eval", core_paths["accepted_primary_eval"]),
        ("accepted_reserve_topups", "train", core_paths["accepted_reserve_train"]),
        ("accepted_reserve_topups", "eval", core_paths["accepted_reserve_eval"]),
        ("pending_reserve_requests", "train", core_paths["pending_train"]),
        ("pending_reserve_requests", "eval", core_paths["pending_eval"]),
        ("joint_materialization", "train", core_paths["joint_train"]),
        ("joint_materialization", "eval", core_paths["joint_eval"]),
    )
    outputs: dict[str, Any] = {}
    for kind, split, path in mapping:
        rows = artifacts[kind][split]
        atomic_jsonl(path, rows)
        outputs[f"{kind}_{split}"] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
        }

    route_groups: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "eval"):
        for row in artifacts["joint_materialization"][split]:
            route_groups.setdefault(str(row["source_route"]), []).append(row)
    route_outputs: dict[str, Any] = {}
    slugs: set[str] = set()
    for route, rows in sorted(route_groups.items()):
        slug = _slug(route)
        if slug in slugs:
            raise ReserveTopupError(f"duplicate route slug: {slug}")
        slugs.add(slug)
        path = output_dir / "routes" / slug / "joint_materialization_plan.jsonl"
        if path.exists() and not args.overwrite:
            raise SystemExit(f"route output exists; pass --overwrite: {path}")
        atomic_jsonl(path, rows)
        datasets = {str(row["hf_dataset"]) for row in rows}
        revisions = {str(row["hf_revision"]) for row in rows}
        if len(datasets) != 1 or len(revisions) != 1:
            raise ReserveTopupError(f"route contract is not immutable: {route}")
        route_outputs[route] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "source_videos": len({str(row["video_id"]) for row in rows}),
            "hf_dataset": next(iter(datasets)),
            "hf_revision": next(iter(revisions)),
        }

    receipt["target_mode"] = target_mode
    receipt["inputs"] = {
        "primary_plans": primary_receipts,
        "reserve_plans": reserve_receipts,
        "quality_results": quality_receipts,
        "strong_metadata_dir": str(args.metadata_dir.resolve()),
    }
    receipt["outputs"] = outputs
    receipt["route_outputs"] = route_outputs
    atomic_bytes(core_paths["table"], _tsv(receipt["per_class"]))
    atomic_json(core_paths["receipt"], receipt)
    deficits = sum(
        int(row["deficit_before_pending_wave"]) for row in receipt["per_class"]
    )
    markdown = "\n".join(
        (
            "# QCES deterministic reserve top-up",
            "",
            f"- Audit: **{'PASS' if receipt['audit_passes'] else 'FAIL'}**",
            f"- Target mode: `{target_mode}`",
            f"- Fixed quality protocols: `{', '.join(receipt['fixed_quality_protocols'])}`",
            f"- Remaining accepted-source deficit: {deficits}",
            f"- Pending reserve class-video rows: {receipt['pending_class_video_rows']}",
            f"- Pending joint source-video rows: {receipt['pending_joint_source_video_rows']}",
            f"- Overdraw factor: {args.overdraw_factor:g}x",
            f"- Hard reserve shortages: {len(receipt['hard_shortages'])}",
            "",
            "Selection uses only the frozen ambiguity tier/reserve rank and the fixed "
            "AudioSep acoustic acceptance bit. QA and detector outcomes are excluded.",
            "",
        )
    )
    atomic_bytes(core_paths["markdown"], markdown.encode("utf-8"))
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0 if receipt["audit_passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
