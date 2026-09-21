#!/usr/bin/env python3
"""Resumably fetch exact AudioSet source rows through HF Dataset Viewer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from mixi_understanding.qces.audioset_dataset_viewer_backend import (
    DatasetViewerTransportError,
    ViewerRetryPolicy,
)
from mixi_understanding.qces.audioset_dataset_viewer_materializer import (
    ViewerMaterializationConfig,
    run_viewer_materialization,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed row-level AudioSet transport. The output source manifest "
            "can be passed to the existing crop materializer as --existing-manifest."
        )
    )
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument(
        "--availability",
        type=Path,
        action="append",
        required=True,
        help="Availability fragment/JSON/JSONL; repeat for multiple files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-new-rows",
        type=int,
        default=0,
        help="Bound a smoke run; zero fetches all requested rows.",
    )
    parser.add_argument("--minimum-free-gib", type=float, default=10.0)
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--initial-delay-seconds", type=float, default=0.5)
    parser.add_argument("--maximum-delay-seconds", type=float, default=8.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing an optional HF token; never persisted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get(str(args.token_env)) or None
    config = ViewerMaterializationConfig(
        binding_path=args.binding,
        availability_paths=tuple(args.availability),
        output_dir=args.output_dir,
        max_new_rows=int(args.max_new_rows),
        minimum_free_disk_bytes=int(float(args.minimum_free_gib) * (1 << 30)),
        token=token,
        retry=ViewerRetryPolicy(
            max_attempts=int(args.max_attempts),
            initial_delay_seconds=float(args.initial_delay_seconds),
            maximum_delay_seconds=float(args.maximum_delay_seconds),
        ),
        timeout_seconds=float(args.timeout_seconds),
    )
    try:
        receipt = run_viewer_materialization(config)
    except DatasetViewerTransportError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}))
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if receipt["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

