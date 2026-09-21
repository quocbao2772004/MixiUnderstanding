#!/usr/bin/env python3
"""Audit that every bounded route smoke row was remotely decoded exactly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from mixi_understanding.qces.audioset_plan_materializer import atomic_json
from mixi_understanding.qces.stratified_materialization_smoke import (
    StratifiedSmokeError,
    audit_remote_smoke_manifests,
    load_jsonl,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-receipt", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
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
        route_manifests = {}
        for route, contract in (smoke.get("route_outputs") or {}).items():
            path = (
                args.materialized_root.resolve()
                / str(contract["slug"])
                / "audioset_strong_plan_manifest.jsonl"
            )
            rows, _ = load_jsonl((path,))
            route_manifests[str(route)] = rows
        report = audit_remote_smoke_manifests(
            smoke_receipt=smoke,
            route_manifests=route_manifests,
        )
    except (KeyError, json.JSONDecodeError, StratifiedSmokeError) as error:
        raise SystemExit(f"ERROR: {error}") from error
    atomic_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if bool(report["audit_passes"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
