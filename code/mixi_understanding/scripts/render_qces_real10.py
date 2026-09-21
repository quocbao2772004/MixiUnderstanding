#!/usr/bin/env python3
"""Render label-free QCES-Real-10 evidence/residual predictions.

The default operation renders only ``real_dev``.  Before the one permitted
``real_test`` render, first run this program on the frozen dev configuration
with ``--write-method-freeze-receipt``.  The test invocation then needs both
``--allow-real-test`` and that exact receipt.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.real10_prediction import (
    Real10RenderError,
    RenderSettings,
    prepare_render_inputs,
    render_prediction_set,
    write_method_freeze_receipt,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--foundation-cache", type=Path, required=True)
    parser.add_argument("--qces-checkpoint", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--split", choices=("real_dev", "real_test"), default="real_dev"
    )
    parser.add_argument("--role-threshold", type=float, default=0.5)
    parser.add_argument("--no-evidence-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--allow-real-test",
        action="store_true",
        help="Explicitly authorize the single post-freeze real-test render.",
    )
    parser.add_argument(
        "--method-freeze-receipt",
        type=Path,
        help="Exact dev-created receipt required with --allow-real-test.",
    )
    parser.add_argument(
        "--write-method-freeze-receipt",
        type=Path,
        help=(
            "Validate a real-dev setup, exclusively write its method-freeze "
            "receipt, and exit without constructing the separator."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only a bit-exact, fully revalidated completed output.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    settings = RenderSettings(
        split=args.split,
        role_threshold=args.role_threshold,
        no_evidence_threshold=args.no_evidence_threshold,
        seed=args.seed,
        device_type=args.device,
    )
    if args.write_method_freeze_receipt is not None:
        if args.output_dir is not None:
            raise SystemExit(
                "--write-method-freeze-receipt cannot be combined with --output-dir"
            )
        if args.split != "real_dev":
            raise SystemExit("method freeze must use --split real_dev")
        if (
            args.allow_real_test
            or args.method_freeze_receipt is not None
            or args.resume
        ):
            raise SystemExit(
                "method-freeze creation cannot use real-test or resume flags"
            )
    elif args.output_dir is None:
        raise SystemExit("rendering requires --output-dir")

    try:
        prepared = prepare_render_inputs(
            manifest_path=args.manifest,
            foundation_cache_dir=args.foundation_cache,
            qces_checkpoint_path=args.qces_checkpoint,
            audiosep_root=args.audiosep_root,
            audiosep_config_path=args.audiosep_config,
            audiosep_checkpoint_path=args.audiosep_checkpoint,
            settings=settings,
        )
        if args.write_method_freeze_receipt is not None:
            receipt = write_method_freeze_receipt(
                args.write_method_freeze_receipt,
                prepared=prepared,
                settings=settings,
            )
            print(
                json.dumps(
                    {
                        "method_freeze_receipt": str(
                            args.write_method_freeze_receipt.resolve()
                        ),
                        "method_identity_sha256": receipt["method_identity_sha256"],
                        "real_test_render_count_before_freeze_↓": 0,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        assert args.output_dir is not None
        result = render_prediction_set(
            prepared=prepared,
            settings=settings,
            output_dir=args.output_dir,
            allow_real_test=args.allow_real_test,
            method_freeze_receipt_path=args.method_freeze_receipt,
            resume=args.resume,
        )
    except (
        FileExistsError,
        FileNotFoundError,
        OSError,
        Real10RenderError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"QCES-Real-10 render failed: {error}") from error
    print(
        json.dumps(
            {
                "output_dir": str(result.output_dir),
                "resumed_exact_output": result.resumed,
                "run_identity_sha256": result.receipt["run_identity_sha256"],
                "counts": result.receipt["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
