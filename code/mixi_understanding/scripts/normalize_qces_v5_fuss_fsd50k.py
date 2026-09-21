#!/usr/bin/env python3
"""Normalize official FUSS/FSD50K upstream metadata for the QCES v5 gate.

The command is offline and never downloads audio.  By default it requires the
listed FUSS WAV files and computes their duration and SHA-256.  Use
``--metadata-only`` for a safe join smoke test before the FUSS source archive is
available; that mode intentionally does not emit ``fuss_sources.jsonl``.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.data.qces_v5_source_ledger import FUSS_SPLITS  # noqa: E402
from mixi_understanding.data.qces_v5_upstream_normalizer import (  # noqa: E402
    UpstreamNormalizationError,
    normalize_official_upstream,
    write_normalized_outputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fuss-root",
        type=Path,
        required=True,
        help="root against which normalized FUSS audio_path values are relative",
    )
    parser.add_argument(
        "--fuss-data-dir",
        type=Path,
        required=True,
        help=(
            "directory containing <split>_{foreground,background}.txt and "
            "<split>/sound/*.wav"
        ),
    )
    parser.add_argument(
        "--fsd50k-ground-truth-dir",
        type=Path,
        required=True,
        help="official extracted FSD50K.ground_truth directory",
    )
    parser.add_argument(
        "--fsd50k-metadata-dir",
        type=Path,
        required=True,
        help="official extracted FSD50K.metadata directory",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=FUSS_SPLITS,
        default=list(FUSS_SPLITS),
        help="FUSS split subset; the paper conversion uses all three",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "join lists to labels/licenses without requiring audio; no normalized "
            "FUSS manifest or release-readiness claim is emitted"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    result = normalize_official_upstream(
        fuss_root=args.fuss_root,
        fuss_data_dir=args.fuss_data_dir,
        fsd50k_ground_truth_dir=args.fsd50k_ground_truth_dir,
        fsd50k_metadata_dir=args.fsd50k_metadata_dir,
        splits=args.splits,
        metadata_only=args.metadata_only,
    )
    paths = write_normalized_outputs(
        result,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    metrics = result["report"]["metrics"]
    print(
        "PASS (normalization only): "
        f"listed={metrics['fuss_listed_source_count']['value']}, "
        f"normalized={metrics['normalized_fsd50k_row_count']['value']}, "
        f"rejected={metrics['rejected_source_count']['value']}; "
        f"report={paths['report']}"
    )
    if args.metadata_only:
        print(
            "NO RELEASE CLAIM: metadata-only mode deliberately emitted no "
            "fuss_sources.jsonl"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileExistsError, OSError, UpstreamNormalizationError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
