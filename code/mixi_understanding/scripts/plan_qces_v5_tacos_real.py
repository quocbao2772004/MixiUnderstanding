#!/usr/bin/env python3
"""Audit TACOS metadata and create the QCES v5 real-set annotation packet."""
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

from mixi_understanding.data.qces_v5_tacos import (  # noqa: E402
    DEFAULT_BENCHMARK_WINDOW_SECONDS,
    TacosAuditError,
    atomic_json,
    atomic_jsonl,
    build_tacos_real_plan,
)


PLAN_NAME = "qces_v5_tacos_real_source_plan.json"
COMPLIANCE_NAME = "qces_v5_tacos_real_compliance.json"
PACKET_NAME = "qces_v5_tacos_annotation_packet.jsonl"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--strong-csv", type=Path, required=True)
    parser.add_argument("--weak-csv", type=Path, required=True)
    parser.add_argument("--development-split-csv", type=Path, required=True)
    parser.add_argument("--test-split-csv", type=Path, required=True)
    parser.add_argument("--record-json", type=Path, required=True)
    parser.add_argument("--qces-label-selection", type=Path, required=True)
    parser.add_argument(
        "--semantic-model-snapshot",
        type=Path,
        required=True,
        help=(
            "local sentence-transformers/all-MiniLM-L6-v2 snapshot at the "
            "pinned c9745ed1 revision; network aliases are rejected"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--audio-archive",
        type=Path,
        help="optional official audio.zip; verifies checksum and selected members",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--development-core-scenes", type=int, default=20)
    parser.add_argument("--test-core-scenes", type=int, default=100)
    parser.add_argument("--minimum-region-seconds", type=float, default=0.25)
    parser.add_argument("--minimum-onset-gap-seconds", type=float, default=0.35)
    parser.add_argument("--overlap-fraction", type=float, default=0.5)
    parser.add_argument(
        "--benchmark-window-seconds",
        type=float,
        default=DEFAULT_BENCHMARK_WINDOW_SECONDS,
    )
    parser.add_argument("--maximum-pairwise-cosine", type=float, default=0.80)
    parser.add_argument("--cosine-round-decimals", type=int, default=6)
    parser.add_argument("--semantic-embedding-batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    plan_path = output_dir / PLAN_NAME
    compliance_path = output_dir / COMPLIANCE_NAME
    packet_path = output_dir / PACKET_NAME
    existing = [
        path for path in (plan_path, compliance_path, packet_path) if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "outputs exist; use --overwrite: " + ", ".join(map(str, existing))
        )
    plan, compliance, packet = build_tacos_real_plan(
        metadata_path=args.metadata_csv.resolve(),
        strong_path=args.strong_csv.resolve(),
        weak_path=args.weak_csv.resolve(),
        development_split_path=args.development_split_csv.resolve(),
        test_split_path=args.test_split_csv.resolve(),
        record_json_path=args.record_json.resolve(),
        qces_label_selection_path=args.qces_label_selection.resolve(),
        semantic_model_snapshot_path=args.semantic_model_snapshot.resolve(),
        seed=args.seed,
        development_core_count=args.development_core_scenes,
        test_core_count=args.test_core_scenes,
        minimum_region_seconds=args.minimum_region_seconds,
        minimum_onset_gap_seconds=args.minimum_onset_gap_seconds,
        overlap_fraction=args.overlap_fraction,
        benchmark_window_seconds=args.benchmark_window_seconds,
        maximum_pairwise_cosine=args.maximum_pairwise_cosine,
        cosine_round_decimals=args.cosine_round_decimals,
        semantic_embedding_batch_size=args.semantic_embedding_batch_size,
        audio_archive_path=(
            args.audio_archive.resolve() if args.audio_archive is not None else None
        ),
    )
    atomic_json(plan_path, plan, overwrite=args.overwrite)
    atomic_json(compliance_path, compliance, overwrite=args.overwrite)
    atomic_jsonl(packet_path, packet, overwrite=args.overwrite)
    metrics = compliance["metrics"]
    print(f"PASS: wrote {len(packet)} CC0 TACOS fixed-10s candidate scenes")
    print(
        f"post-{args.maximum_pairwise_cosine:.2f}/hash capacity: "
        f"real_dev={metrics['post_hash_real_dev_creators_↑']} creators, "
        f"real_test={metrics['post_hash_real_test_creators_↑']} creators"
    )
    print(f"plan: {plan_path}")
    print(f"compliance: {compliance_path}")
    print(f"annotation packet: {packet_path}")
    print("submission real-data gate: false (human verification remains required)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (FileExistsError, OSError, TacosAuditError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
