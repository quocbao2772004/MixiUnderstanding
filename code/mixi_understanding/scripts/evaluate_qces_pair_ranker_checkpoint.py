#!/usr/bin/env python3
"""Evaluate a trained QCES pair-ranker checkpoint on full validation items."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.train_qces_detector_pair_ranker import (  # noqa: E402
    PairRanker,
    build_examples,
    conservative_metrics_with_skips,
    evaluate,
    make_device,
    save_predictions,
)
from mixi_understanding.scripts.evaluate_qces_detector_inventory_qa import write_json  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frame-probs", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--none-score-bias", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def namespace_from_checkpoint_args(raw: dict[str, Any]) -> Namespace:
    values = dict(raw)
    for key in ("train_frame_probs", "val_frame_probs", "train_manifest", "val_manifest", "output_dir"):
        if key in values:
            values[key] = Path(values[key])
    return Namespace(**values)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = make_device(args.device)

    ckpt = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    train_args = namespace_from_checkpoint_args(ckpt.get("args_jsonable") or ckpt.get("args") or {})
    frame_probs = args.frame_probs or train_args.val_frame_probs
    manifest = args.manifest or train_args.val_manifest
    none_bias = float(args.none_score_bias if args.none_score_bias is not None else ckpt["report"].get("selected_none_score_bias", 0.0))

    frame_payload = torch.load(frame_probs.resolve(), map_location="cpu", weights_only=False)
    labels = list(frame_payload["labels"])
    feature_dim = int(ckpt.get("feature_dim", 15))
    model = PairRanker(
        num_labels=len(labels),
        feature_dim=feature_dim,
        embed_dim=int(train_args.embed_dim),
        hidden_dim=int(train_args.hidden_dim),
        dropout=float(train_args.dropout),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    val_examples, val_skipped = build_examples(
        manifest_path=manifest.resolve(),
        frame_payload=frame_payload,
        args=train_args,
        split_name="val",
        max_scenes=int(train_args.max_val_scenes),
    )
    metrics, rows = evaluate(model, val_examples, device, none_score_bias=none_bias)
    report = {
        "format": "qces_pair_ranker_checkpoint_eval_v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "frame_probs": str(frame_probs.resolve()),
        "manifest": str(manifest.resolve()),
        "none_score_bias": none_bias,
        "items": len(rows),
        "val_skipped": len(val_skipped),
        "val_skip_reasons": {reason: sum(row["reason"] == reason for row in val_skipped) for reason in sorted({row["reason"] for row in val_skipped})},
        "metrics": metrics,
        "metrics_full_conservative": conservative_metrics_with_skips(metrics, val_skipped),
    }
    write_json(output_dir / "eval_report.json", report)
    save_predictions(output_dir / "val_predictions.jsonl", rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
