#!/usr/bin/env python3
"""Refresh and print the full-200 primary source-bank progress receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from mixi_understanding.qces.full200_adaptive_execution import (
    Full200ExecutionError,
    build_progress,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execution-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1",
    )
    args = parser.parse_args(argv)
    try:
        progress = build_progress(execution_dir=args.execution_dir)
    except (OSError, ValueError, Full200ExecutionError) as error:
        print(f"ERROR: {error}")
        return 1
    print(json.dumps(progress, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
