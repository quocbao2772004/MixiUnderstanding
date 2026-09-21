#!/usr/bin/env python3
"""Finalize two-pass TACOS annotations into the QCES real QA manifest."""
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

from mixi_understanding.data.qces_v5_tacos import TacosAuditError  # noqa: E402
from mixi_understanding.data.qces_v5_tacos_finalize import (  # noqa: E402
    finalize_human_annotations,
    write_finalization_artifacts,
)


OUTPUT_NAMES = (
    "qces_v5_tacos_human_compliance.json",
    "qces_v5_tacos_verified_scenes.jsonl",
    "qces_v5_tacos_real_qa.jsonl",
    "qces_v5_tacos_adjudication_audit.jsonl",
    "qces_real10_inference.jsonl",
    "qces_real10_scoring.jsonl",
    "qces_real10_inference_real_dev.jsonl",
    "qces_real10_inference_real_test.jsonl",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-packet", type=Path, required=True)
    parser.add_argument("--audio-receipt", type=Path, required=True)
    parser.add_argument("--response-root", type=Path, required=True)
    parser.add_argument("--rater-a-id", required=True)
    parser.add_argument("--rater-b-id", required=True)
    parser.add_argument("--adjudicator-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--target-real-dev-scenes", type=int, default=20)
    parser.add_argument("--target-real-test-scenes", type=int, default=100)
    parser.add_argument(
        "--maximum-boundary-difference-seconds", type=float, default=0.25
    )
    parser.add_argument("--minimum-interval-iou", type=float, default=0.80)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    project_root = args.project_root.resolve()
    existing = [
        output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "finalization outputs exist; use --overwrite: "
            + ", ".join(map(str, existing))
        )
    compliance, scenes, qa_rows, audit_rows = finalize_human_annotations(
        packet_path=args.annotation_packet.resolve(),
        audio_receipt_path=args.audio_receipt.resolve(),
        response_root=args.response_root.resolve(),
        rater_a_id=args.rater_a_id,
        rater_b_id=args.rater_b_id,
        adjudicator_id=args.adjudicator_id,
        project_root=project_root,
        target_real_dev_scenes=args.target_real_dev_scenes,
        target_real_test_scenes=args.target_real_test_scenes,
        maximum_boundary_difference_seconds=args.maximum_boundary_difference_seconds,
        minimum_interval_iou=args.minimum_interval_iou,
    )
    write_finalization_artifacts(
        output_dir=output_dir,
        project_root=project_root,
        compliance=compliance,
        scenes=scenes,
        qa_rows=qa_rows,
        audit_rows=audit_rows,
        overwrite=args.overwrite,
    )
    print(f"resolved scenes ↑: {len(scenes)}")
    print(f"verified QA rows ↑: {len(qa_rows)}")
    print(f"authoritative inference rows ↑: {len(qa_rows)}")
    print(f"authoritative scoring rows ↑: {len(qa_rows)}")
    print(
        "submission real-data gate: "
        + str(compliance["submission_real_data_gate_passed"]).lower()
    )
    return 0 if compliance["submission_real_data_gate_passed"] else 2


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileExistsError, OSError, TacosAuditError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
