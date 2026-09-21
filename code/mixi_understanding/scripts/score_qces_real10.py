#!/usr/bin/env python3
"""Validate and score one non-QA QCES-Real-10 prediction set.

The command defaults to ``real_dev``.  Real-test additionally requires the
explicit authorization flag and a renderer receipt whose exact method-freeze
artifact remains content-valid.  The output is created exclusively and is
never overwritten.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.real10_scoring import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    Real10ScoringError,
    score_qces_real10,
    write_score_report_exclusive,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("real_dev", "real_test"), default="real_dev"
    )
    parser.add_argument(
        "--allow-real-test",
        action="store_true",
        help=(
            "Authorize scoring a real-test render only after its receipt proves "
            "the exact pre-test method-freeze gate."
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if args.bootstrap_replicates <= 0:
        raise Real10ScoringError("--bootstrap-replicates must be positive")
    if args.split == "real_dev" and args.allow_real_test:
        raise Real10ScoringError("--allow-real-test is invalid with --split real_dev")
    report = score_qces_real10(
        scoring_manifest_path=args.scoring_manifest,
        dataset_root=args.dataset_root,
        prediction_root=args.prediction_root,
        split=args.split,
        allow_real_test=args.allow_real_test,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    write_score_report_exclusive(args.output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except (FileExistsError, FileNotFoundError, Real10ScoringError) as error:
        raise SystemExit(f"QCES-Real-10 scoring failed: {error}") from error
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "split": report["split"],
                "all_integrity_gates_passed": report["integrity_gates"][
                    "all_integrity_gates_passed"
                ],
                "records_scored_↑": report["overall"]["counts"]["records_↑"],
                "creator_clusters_↑": report["overall"]["counts"][
                    "authoritative_creator_clusters_↑"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
