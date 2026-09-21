#!/usr/bin/env python3
"""Derive packet-bound canonical 10 s WAVs from the pinned TACOS archive."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.data.qces_v5_tacos import TacosAuditError  # noqa: E402
from mixi_understanding.data.qces_v5_tacos_audio import (  # noqa: E402
    extract_tacos_subset,
    write_extraction_artifacts,
)


COMPLIANCE_NAME = "qces_v5_tacos_audio_compliance.json"
RECEIPT_NAME = "qces_v5_tacos_audio_receipt.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-packet", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--audio-archive", type=Path, required=True)
    parser.add_argument("--audio-output-dir", type=Path, required=True)
    parser.add_argument("--receipt-output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--maximum-duration-error-seconds", type=float, default=0.01)
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="replace an existing derived WAV only when it differs from the canonical bytes",
    )
    parser.add_argument(
        "--overwrite-receipts",
        action="store_true",
        help="replace this command's JSON/JSONL receipt artifacts",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    receipt_output_dir = args.receipt_output_dir.resolve()
    compliance_path = receipt_output_dir / COMPLIANCE_NAME
    receipt_path = receipt_output_dir / RECEIPT_NAME
    existing = [path for path in (compliance_path, receipt_path) if path.exists()]
    if existing and not args.overwrite_receipts:
        raise FileExistsError(
            "receipt outputs exist; use --overwrite-receipts: "
            + ", ".join(map(str, existing))
        )
    compliance, receipts = extract_tacos_subset(
        packet_path=args.annotation_packet.resolve(),
        plan_path=args.source_plan.resolve(),
        archive_path=args.audio_archive.resolve(),
        output_dir=args.audio_output_dir.resolve(),
        project_root=args.project_root.resolve(),
        overwrite_audio=args.overwrite_audio,
        maximum_duration_error_seconds=args.maximum_duration_error_seconds,
    )
    write_extraction_artifacts(
        compliance_path=compliance_path,
        receipt_path=receipt_path,
        compliance=compliance,
        receipts=receipts,
        overwrite=args.overwrite_receipts,
    )
    print(
        f"PASS: derived/verified {len(receipts)} packet-bound TACOS WAV files "
        "(mono 32 kHz, 320000 frames each)"
    )
    print(f"compliance: {compliance_path}")
    print(f"receipt: {receipt_path}")
    print("submission real-data gate: false (human verification remains required)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileExistsError, OSError, TacosAuditError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
