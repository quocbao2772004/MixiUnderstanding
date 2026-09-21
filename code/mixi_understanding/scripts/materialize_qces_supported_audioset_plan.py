#!/usr/bin/env python3
"""Robustly materialize the exact supported-ontology AudioSet-Strong plans.

Unlike the legacy streaming materializer, this command is exact and resumable:
it locates planned video IDs by reading only the Parquet ``video_id`` column,
then fetches audio only from matching row groups.  A bounded smoke run exits 2
until every planned row has been materialized; ``--scan-only`` is the one
intentional non-materializing mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from huggingface_hub import HfApi

from mixi_understanding.qces.audioset_plan_materializer import (
    DEFAULT_PARQUET_PATTERN,
    MaterializationConfig,
    MaterializationError,
    RetryPolicy,
    run_materialization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN_DIR = PROJECT_ROOT / "outputs/qces_supported_ontology_200_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_supported_ontology_200_materialized_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        action="append",
        dest="plans",
        help="Repeatable plan JSONL. Defaults to both selected train/eval plans.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--existing-manifest",
        type=Path,
        action="append",
        default=[],
        help="Repeatable existing manifest whose matching audio is hash/soundfile-validated and reused.",
    )
    parser.add_argument(
        "--parquet-pattern",
        default=DEFAULT_PARQUET_PATTERN,
        help="fsspec pattern with optional {split}/{revision}; local patterns support tests.",
    )
    parser.add_argument("--hf-dataset", default="enyoukai/AudioSet-Strong")
    parser.add_argument("--hf-revision", default="main")
    parser.add_argument(
        "--storage-mode",
        choices=["full_clips", "requested_crops"],
        default="full_clips",
        help="Store full sources or only crop requests embedded in the joint plan.",
    )
    parser.add_argument(
        "--resolved-revision",
        default="",
        help="Explicit immutable revision (also used for local test data); otherwise resolve HF revision.",
    )
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument(
        "--require-preindexed-locations",
        action="store_true",
        help=(
            "Fail unless every plan row embeds an exact, matching availability_location; "
            "never fall back to scanning another mirror."
        ),
    )
    parser.add_argument(
        "--max-new-videos",
        type=int,
        default=0,
        help="Bound a smoke run. Incomplete materialization deliberately exits 2.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-initial-seconds", type=float, default=0.5)
    parser.add_argument("--retry-max-seconds", type=float, default=8.0)
    parser.add_argument("--block-size-mib", type=int, default=1)
    parser.add_argument(
        "--min-free-disk-gib",
        type=float,
        default=10.0,
        help="Abort before audio reads if the estimate would leave less free disk.",
    )
    parser.add_argument("--disk-estimate-safety-factor", type=float, default=1.15)
    parser.add_argument("--scan-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_new_videos < 0:
        parser.error("--max-new-videos must be >= 0")
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1")
    if args.block_size_mib < 1:
        parser.error("--block-size-mib must be >= 1")
    if args.min_free_disk_gib < 0:
        parser.error("--min-free-disk-gib must be >= 0")
    if args.disk_estimate_safety_factor < 1:
        parser.error("--disk-estimate-safety-factor must be >= 1")
    if args.scan_workers < 1:
        parser.error("--scan-workers must be >= 1")
    return args


def resolve_revision(args: argparse.Namespace) -> str:
    if args.resolved_revision:
        return str(args.resolved_revision)
    if not str(args.parquet_pattern).startswith("hf://"):
        return "local-unpinned"
    last_error: OSError | None = None
    for attempt in range(1, int(args.max_retries) + 1):
        try:
            info = HfApi().dataset_info(args.hf_dataset, revision=args.hf_revision)
            if not info.sha:
                raise MaterializationError(
                    f"could not resolve immutable revision for "
                    f"{args.hf_dataset}@{args.hf_revision}"
                )
            return str(info.sha)
        except OSError as error:
            last_error = error
            if attempt >= int(args.max_retries):
                break
            delay = min(
                float(args.retry_max_seconds),
                float(args.retry_initial_seconds) * (2 ** (attempt - 1)),
            )
            print(
                f"retry {attempt}/{int(args.max_retries) - 1} resolving "
                f"{args.hf_dataset}@{args.hf_revision}: {error}; "
                f"sleeping {delay:.2f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
    raise MaterializationError(
        f"failed to resolve immutable revision for {args.hf_dataset}@"
        f"{args.hf_revision} after {args.max_retries} attempts: {last_error}"
    ) from last_error


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plans = tuple(args.plans or [
        DEFAULT_PLAN_DIR / "materialization_plan_train.jsonl",
        DEFAULT_PLAN_DIR / "materialization_plan_eval.jsonl",
    ])
    try:
        config = MaterializationConfig(
            plan_paths=plans,
            output_dir=args.output_dir,
            parquet_pattern=args.parquet_pattern,
            hf_dataset=str(args.hf_dataset),
            resolved_revision=resolve_revision(args),
            storage_mode=str(args.storage_mode),
            existing_manifest_paths=tuple(args.existing_manifest),
            require_preindexed_locations=bool(args.require_preindexed_locations),
            scan_only=bool(args.scan_only),
            max_new_videos=int(args.max_new_videos),
            block_size_bytes=int(args.block_size_mib) * (1 << 20),
            minimum_free_disk_bytes=int(float(args.min_free_disk_gib) * (1 << 30)),
            disk_estimate_safety_factor=float(args.disk_estimate_safety_factor),
            scan_workers=int(args.scan_workers),
            retry=RetryPolicy(
                max_attempts=int(args.max_retries),
                initial_delay_seconds=float(args.retry_initial_seconds),
                maximum_delay_seconds=float(args.retry_max_seconds),
            ),
        )
        receipt = run_materialization(config)
    except (MaterializationError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    if args.scan_only:
        return 0 if bool(receipt["disk_preflight"]["safe"]) else 3
    return 0 if bool(receipt["complete"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
