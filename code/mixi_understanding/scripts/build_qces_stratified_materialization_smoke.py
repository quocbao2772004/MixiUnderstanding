#!/usr/bin/env python3
"""Build a deterministic 1--2-source-per-route post-index smoke plan."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from mixi_understanding.qces.audioset_plan_materializer import (
    atomic_bytes,
    atomic_json,
    atomic_jsonl,
)
from mixi_understanding.qces.stratified_materialization_smoke import (
    StratifiedSmokeError,
    load_jsonl,
    route_slug,
    sample_rates_from_manifests,
    select_stratified_smoke,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_stratified_materialization_smoke_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        action="append",
        required=True,
        help="Repeatable route-aware joint primary plan JSONL (never a reserve plan).",
    )
    parser.add_argument(
        "--known-materialized-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional decoded materialization manifest used only for real native "
            "sample-rate stratification."
        ),
    )
    parser.add_argument("--records-per-route", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    receipt_path = output_dir / "stratified_smoke_receipt.json"
    combined_path = output_dir / "smoke_materialization_manifest.jsonl"
    markdown_path = output_dir / "stratified_smoke_receipt.md"
    if not args.overwrite:
        existing = [
            str(path)
            for path in (receipt_path, combined_path, markdown_path)
            if path.exists()
        ]
        if existing:
            raise SystemExit(f"outputs exist; pass --overwrite: {existing}")
    try:
        rows, input_receipts = load_jsonl(tuple(args.plan))
        known_rows, known_receipts = load_jsonl(
            tuple(args.known_materialized_manifest)
        ) if args.known_materialized_manifest else ([], [])
        known_rates = sample_rates_from_manifests(known_rows)
        selected, report = select_stratified_smoke(
            rows,
            records_per_route=int(args.records_per_route),
            seed=int(args.seed),
            known_sample_rates=known_rates,
        )
    except StratifiedSmokeError as error:
        raise SystemExit(f"ERROR: {error}") from error

    route_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        route_rows[str(row["source_route"])].append(row)
    route_outputs: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(combined_path, selected)
    for route in sorted(route_rows):
        rows_for_route = sorted(
            route_rows[route],
            key=lambda row: (
                str(row["metadata_split"]),
                str(row["video_id"]),
            ),
        )
        slug = route_slug(route)
        path = output_dir / "routes" / slug / "smoke_materialization_plan.jsonl"
        atomic_jsonl(path, rows_for_route)
        datasets = {str(row["hf_dataset"]) for row in rows_for_route}
        revisions = {str(row["hf_revision"]) for row in rows_for_route}
        if len(datasets) != 1 or len(revisions) != 1:
            raise SystemExit(f"ERROR: route contract changed while writing: {route}")
        route_outputs[route] = {
            "slug": slug,
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(rows_for_route),
            "crop_requests": sum(
                len(row.get("crop_requests") or []) for row in rows_for_route
            ),
            "metadata_splits": sorted(
                {str(row["metadata_split"]) for row in rows_for_route}
            ),
            "coverage_labels": sorted(
                {
                    str(request["coverage_label"])
                    for row in rows_for_route
                    for request in row.get("crop_requests") or []
                }
            ),
            "hf_dataset": next(iter(datasets)),
            "hf_revision": next(iter(revisions)),
        }
    report["inputs"] = {
        "joint_primary_plans": input_receipts,
        "known_materialized_manifests": known_receipts,
    }
    report["outputs"] = {
        "combined_manifest": {
            "path": str(combined_path),
            "sha256": sha256_file(combined_path),
            "rows": len(selected),
        }
    }
    report["route_outputs"] = route_outputs
    atomic_json(receipt_path, report)
    lines = [
        "# Stratified post-index materialization smoke",
        "",
        f"- Audit: **{'PASS' if report['audit_passes'] else 'FAIL'}**",
        f"- Active routes: {report['active_routes']}",
        f"- Selected source videos: {report['selected_source_videos']}",
        f"- Selected crop requests: {report['selected_crop_requests']}",
        f"- Splits: {', '.join(report['selected_metadata_splits'])}",
        f"- Unique stratification labels: {report['selected_unique_stratification_labels']}",
        f"- Ambiguity tiers: {report['selected_ambiguity_tiers']}",
        f"- Known native rates selected: {report['selected_known_native_sample_rates']}",
        f"- Train/eval source overlap: {report['train_eval_source_overlap']}",
        "",
        "Sample rate is never guessed from a route. Without a decoded manifest,",
        "the smoke is stratified by route, split, coverage label and ambiguity tier.",
        "",
    ]
    atomic_bytes(markdown_path, ("\n".join(lines)).encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(report["audit_passes"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
