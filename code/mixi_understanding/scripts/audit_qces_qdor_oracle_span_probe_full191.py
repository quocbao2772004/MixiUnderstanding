#!/usr/bin/env python3
"""Run the Q-DOR oracle-span representation gate on explicit full191 splits.

This sidecar keeps the historical AudioSet regrouping probe unchanged.  It
uses the leakage-safe scene-id lists emitted by the full191 builder and asks a
single diagnostic question: when the gold event interval is supplied, can a
small probe recover the event label from the frozen R1 dense representation?
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from mixi_understanding.scripts.audit_qces_qdor_oracle_span_probe import (
    ORACLE_GATE_THRESHOLD,
    OracleSpanProbe,
    _atomic_json,
    evaluate,
    load_examples,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root", type=Path, action="append", dest="dense_roots", required=True)
    parser.add_argument("--train-scene-list", type=Path, required=True)
    parser.add_argument("--dev-scene-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--only-isolated-events", action="store_true")
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def read_scene_list(path: Path) -> set[str]:
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not values:
        raise ValueError(f"empty scene list: {path}")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise SystemExit("epochs, batch-size, and patience must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    indexes = [
        json.loads((root.resolve() / "index.json").read_text(encoding="utf-8"))
        for root in args.dense_roots
    ]
    labels = list(indexes[0]["labels"])
    if any(list(index["labels"]) != labels for index in indexes[1:]):
        raise RuntimeError("dense roots use different label orderings")
    train_ids = read_scene_list(args.train_scene_list.resolve())
    dev_ids = read_scene_list(args.dev_scene_list.resolve())
    overlap = train_ids & dev_ids
    if overlap:
        raise RuntimeError(f"train/dev scene leakage: {len(overlap)} overlapping ids")

    (train_x, train_y), (dev_x, dev_y), data_audit = load_examples(
        [root.resolve() for root in args.dense_roots],
        train_ids,
        dev_ids,
        labels,
        only_isolated_events=args.only_isolated_events,
    )
    missing_train = len(train_ids) - int(data_audit["scenes"].get("train", 0))
    missing_dev = len(dev_ids) - int(data_audit["scenes"].get("dev", 0))
    if missing_train or missing_dev:
        raise RuntimeError(
            f"dense coverage incomplete: missing train={missing_train}, dev={missing_dev}"
        )

    device = torch.device(args.device)
    model = OracleSpanProbe(train_x.shape[1], args.hidden_dim, len(labels)).to(device)
    counts = torch.bincount(train_y, minlength=len(labels)).float()
    sample_weight = counts.clamp_min(1).reciprocal()[train_y]
    sampler = WeightedRandomSampler(
        sample_weight,
        len(train_y),
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=args.batch_size, sampler=sampler)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for value, target in loader:
            value, target = value.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(value), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, dev_x, dev_y, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if best is None or row["top1_accuracy"] > best["top1_accuracy"]:
            best = dict(row)
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    assert best is not None
    report = {
        "format": "qces_qdor_oracle_span_probe_full191_v1",
        "status": "representation_gate_only_not_deployable",
        "input_contract": "gold interval -> pooled frozen R1 hidden + detector logits; no label input",
        "seed": args.seed,
        "labels": len(labels),
        "only_isolated_events": args.only_isolated_events,
        "dense_roots": [str(root.resolve()) for root in args.dense_roots],
        "scene_lists": {
            "train": str(args.train_scene_list.resolve()),
            "dev": str(args.dev_scene_list.resolve()),
        },
        "data": data_audit,
        "best": best,
        "oracle_span_answer_top1": float(best["top1_accuracy"]),
        "history": history,
        "gate": {
            "metric": "full191 dev oracle-span top1 accuracy",
            "threshold": ORACLE_GATE_THRESHOLD,
            "passes": bool(best["top1_accuracy"] >= ORACLE_GATE_THRESHOLD),
        },
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), "best": best, "gate": report["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
