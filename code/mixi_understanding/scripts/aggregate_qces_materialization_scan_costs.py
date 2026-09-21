#!/usr/bin/env python3
"""Aggregate validated route-specific scan-only materializer receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from mixi_understanding.qces.audioset_plan_materializer import atomic_json
from mixi_understanding.qces.stratified_materialization_smoke import (
    StratifiedSmokeError,
    aggregate_scan_receipts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument(
        "--scan-root",
        type=Path,
        required=True,
        help="Directory containing <route-slug>/materialization_receipt.json.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StratifiedSmokeError(f"receipt does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StratifiedSmokeError(f"receipt is not an object: {path}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        smoke = _json(args.smoke_receipt.resolve())
        routes = smoke.get("route_outputs") or {}
        route_receipts = {
            str(route): _json(
                args.scan_root.resolve()
                / str(contract["slug"])
                / "materialization_receipt.json"
            )
            for route, contract in routes.items()
        }
        report = aggregate_scan_receipts(
            smoke_receipt=smoke,
            route_receipts=route_receipts,
        )
    except (KeyError, json.JSONDecodeError, StratifiedSmokeError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(report["audit_passes"]) else 3


if __name__ == "__main__":
    raise SystemExit(main())
