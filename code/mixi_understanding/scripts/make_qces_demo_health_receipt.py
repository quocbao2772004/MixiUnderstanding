#!/usr/bin/env python3
"""Bind a QCES checkpoint to explicit anti-collapse metrics for the demo."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.demo_contract import (
    GATE_PROFILES,
    DemoContractError,
    build_health_receipt,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(GATE_PROFILES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        receipt = build_health_receipt(
            args.checkpoint.resolve(), args.evaluation_report.resolve(), args.profile
        )
    except DemoContractError as error:
        raise SystemExit(str(error)) from error
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not receipt["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
