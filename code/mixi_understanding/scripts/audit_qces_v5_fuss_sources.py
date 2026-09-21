#!/usr/bin/env python3
"""Audit a pinned FUSS/FSD50K metadata join for the QCES v5 paper route.

This command is intentionally offline.  It never downloads upstream data and
never emits a source receipt unless ``--finalize`` is requested and every
selected local FUSS source matches its declared SHA-256 digest.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.data.qces_v5_source_ledger import (  # noqa: E402
    SourceAuditError,
    atomic_json,
    audit_source_ledger,
)


PLAN_NAME = "qces_v5_fuss_source_plan.json"
COMPLIANCE_NAME = "qces_v5_fuss_source_compliance.json"
RECEIPT_NAME = "qces_v5_fuss_source_receipt.json"
BLOCKED_RECEIPT_FORMAT = "qces_v5_source_receipt_blocked_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fuss-root",
        type=Path,
        required=True,
        help="local pinned FUSS v1.3 root containing the normalized manifest",
    )
    parser.add_argument(
        "--fsd50k-root",
        type=Path,
        required=True,
        help="local pinned FSD50K metadata root (labels/provenance only)",
    )
    parser.add_argument(
        "--pins-json",
        type=Path,
        required=True,
        help="pin record with manifest, license, DOI, revision, and file hashes",
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        required=True,
        help="disjoint seen/held-out/nuisance label selection",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="root against which receipt audio_path values are made relative",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for plan, compliance report, and optional receipt",
    )
    parser.add_argument("--seed", type=int, default=314_159)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="verify every selected local audio hash and emit a usable receipt",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this command's existing output artifacts",
    )
    return parser.parse_args(argv)


def _preflight_outputs(paths: Sequence[Path], *, overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"output exists: {rendered}; use --overwrite")


def _blocked_receipt(
    *, plan: Mapping[str, Any], compliance: Mapping[str, Any]
) -> dict[str, Any]:
    """Write an unmistakably non-consumable tombstone over a stale receipt."""

    return {
        "format": BLOCKED_RECEIPT_FORMAT,
        "profile": plan.get("profile"),
        "source_route": plan.get("source_route"),
        "acquisition_complete": False,
        "release_ready": False,
        "metadata_gate_passed": plan.get("metadata_gate_passed", False),
        "reason": (
            "No valid finalization exists; inspect the current compliance JSON. "
            "This tombstone cannot be consumed as an audited source ledger."
        ),
        "input_fingerprint": compliance.get("input_fingerprint"),
    }


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    plan_path = output_dir / PLAN_NAME
    compliance_path = output_dir / COMPLIANCE_NAME
    receipt_path = output_dir / RECEIPT_NAME
    # Include the receipt in preflight even for plan-only runs. Otherwise an
    # old success receipt could survive beside a newly overwritten plan.
    outputs = [plan_path, compliance_path, receipt_path]
    _preflight_outputs(outputs, overwrite=args.overwrite)

    try:
        plan, compliance, receipt = audit_source_ledger(
            fuss_root=args.fuss_root,
            fsd50k_root=args.fsd50k_root,
            pins_path=args.pins_json,
            selection_path=args.selection_json,
            project_root=args.project_root,
            seed=args.seed,
            verify_audio=args.finalize,
        )
    except (SourceAuditError, OSError):
        # Explicit overwrite means fail closed: an input/pin error during a
        # re-audit must not leave a previously valid receipt consumable.
        if args.overwrite and receipt_path.exists():
            atomic_json(
                receipt_path,
                {
                    "format": BLOCKED_RECEIPT_FORMAT,
                    "acquisition_complete": False,
                    "release_ready": False,
                    "reason": "Audit aborted before a valid plan was produced.",
                },
                overwrite=True,
            )
        raise
    atomic_json(plan_path, plan, overwrite=args.overwrite)
    atomic_json(compliance_path, compliance, overwrite=args.overwrite)

    if args.finalize:
        if receipt is None:
            # If --overwrite was explicitly authorized, replace any prior valid
            # receipt so a failed re-audit cannot leave a stale success artifact.
            atomic_json(
                receipt_path,
                _blocked_receipt(plan=plan, compliance=compliance),
                overwrite=args.overwrite,
            )
            print(
                f"BLOCKED: receipt not emitted; inspect {compliance_path}",
                file=sys.stderr,
            )
            return 2
        atomic_json(receipt_path, receipt, overwrite=args.overwrite)
        print(f"PASS: finalized audited receipt: {receipt_path}")
        return 0

    if args.overwrite and receipt_path.exists():
        atomic_json(
            receipt_path,
            _blocked_receipt(plan=plan, compliance=compliance),
            overwrite=True,
        )
    if not plan["metadata_gate_passed"]:
        print(
            f"BLOCKED: metadata gate failed; inspect {compliance_path}",
            file=sys.stderr,
        )
        return 2
    print(
        "PASS (metadata only): plan is feasible; no audio receipt was emitted. "
        f"Plan: {plan_path}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileExistsError, SourceAuditError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
