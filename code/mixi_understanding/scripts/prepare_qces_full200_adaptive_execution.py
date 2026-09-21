#!/usr/bin/env python3
"""Freeze bounded execution chunks for the 200-class QCES source bank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mixi_understanding.qces.full200_adaptive_execution import (
    Full200ExecutionError,
    build_execution_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_DIR = PROJECT_ROOT / "outputs/qces_availability_aware_acoustic_plan_v2"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--availability-receipt",
        type=Path,
        default=PLAN_DIR / "availability_aware_plan_receipt.json",
    )
    parser.add_argument(
        "--joint-receipt",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/qces_joint_acoustic_materialization_plan_v2/joint_materialization_receipt.json"
        ),
    )
    parser.add_argument("--primary-train", type=Path, default=PLAN_DIR / "crop_source_plan_train.jsonl")
    parser.add_argument("--primary-eval", type=Path, default=PLAN_DIR / "crop_source_plan_eval.jsonl")
    parser.add_argument("--reserve-train", type=Path, default=PLAN_DIR / "crop_source_reserve_train.jsonl")
    parser.add_argument("--reserve-eval", type=Path, default=PLAN_DIR / "crop_source_reserve_eval.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1",
    )
    parser.add_argument("--maximum-videos-per-chunk", type=int, default=320)
    parser.add_argument("--maximum-crops-per-chunk", type=int, default=384)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        receipt = build_execution_contract(
            availability_receipt_path=args.availability_receipt,
            joint_receipt_path=args.joint_receipt,
            primary_train_path=args.primary_train,
            primary_eval_path=args.primary_eval,
            reserve_train_path=args.reserve_train,
            reserve_eval_path=args.reserve_eval,
            output_dir=args.output_dir,
            maximum_videos_per_chunk=args.maximum_videos_per_chunk,
            maximum_crops_per_chunk=args.maximum_crops_per_chunk,
        )
    except (OSError, ValueError, Full200ExecutionError) as error:
        print(f"ERROR: {error}")
        return 1
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
