#!/usr/bin/env python3
"""Measure whether dense BEATs features can identify an oracle event window.

This is a representation gate, not a deployable QCES result.  The probe sees
the gold event interval but never the gold label as input.  It is intentionally
run on the clean AudioSet train/dev grouping before spending time on the full
question-conditioned reasoner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


FRAME_HOP_SECONDS = 0.04
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ORACLE_GATE_THRESHOLD = 0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dense-root",
        type=Path,
        action="append",
        dest="dense_roots",
        default=[],
        help="Dense v2 root; repeat for scene-disjoint train/dev exports.",
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_clean_detector_protocol_v1_current",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_qdor_oracle_span_probe/current/report.json",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--only-isolated-events",
        action="store_true",
        help="Keep events that do not overlap a differently labelled annotation.",
    )
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _read_scene_ids(path: Path) -> set[str]:
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if str(row.get("protocol_source")) == "audioset_strong" or "audioset" in str(
                    row.get("source_route", "")
                ).lower():
                    result.add(str(row["scene_id"]))
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _iter_shards(root: Path) -> Iterable[dict[str, Any]]:
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    for item in index["shards"]:
        relative = item.get("path") or item.get("filename")
        yield torch.load(root / relative, map_location="cpu", weights_only=False)


def _event_vector(
    features: torch.Tensor,
    logits: torch.Tensor,
    onset: float,
    offset: float,
    valid_frames: int,
) -> torch.Tensor | None:
    start = max(0, min(valid_frames - 1, int(math.floor(onset / FRAME_HOP_SECONDS))))
    end = max(start + 1, min(valid_frames, int(math.ceil(offset / FRAME_HOP_SECONDS))))
    if valid_frames <= 0 or end <= start:
        return None
    hidden = features[start:end].float()
    scores = logits[start:end].float()
    # Mean and max preserve sustained and transient evidence respectively.
    return torch.cat(
        (hidden.mean(0), hidden.amax(0), scores.mean(0), scores.amax(0)), dim=0
    )


def load_examples(
    dense_roots: list[Path],
    train_scene_ids: set[str],
    dev_scene_ids: set[str],
    labels: list[str],
    *,
    only_isolated_events: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], dict[str, Any]]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    values: dict[str, list[torch.Tensor]] = {"train_x": [], "train_y": [], "dev_x": [], "dev_y": []}
    observed_scenes = Counter()
    skipped = Counter()
    seen_scene_ids: set[str] = set()
    for dense_root in dense_roots:
        for shard in _iter_shards(dense_root):
            for row_index, scene_id in enumerate(shard["scene_ids"]):
                if scene_id in seen_scene_ids:
                    raise RuntimeError(f"duplicate scene across dense roots: {scene_id}")
                seen_scene_ids.add(scene_id)
                split = (
                    "train"
                    if scene_id in train_scene_ids
                    else "dev"
                    if scene_id in dev_scene_ids
                    else ""
                )
                if not split:
                    continue
                observed_scenes[split] += 1
                valid_frames = int(shard["valid_frames"][row_index])
                scene_events = [
                    event
                    for event in shard["gold_events"][row_index]
                    if str(event.get("event_kind", "semantic")) == "semantic"
                ]
                for event in scene_events:
                    label = str(event.get("label", ""))
                    if label not in label_to_index:
                        skipped["unknown_label"] += 1
                        continue
                    if only_isolated_events and any(
                        str(other.get("label", "")) != label
                        and min(
                            float(event["offset_seconds"]),
                            float(other["offset_seconds"]),
                        )
                        > max(
                            float(event["onset_seconds"]),
                            float(other["onset_seconds"]),
                        )
                        for other in scene_events
                        if other is not event
                    ):
                        skipped["overlaps_different_label"] += 1
                        continue
                    vector = _event_vector(
                        shard["features"][row_index],
                        shard["logits"][row_index],
                        float(event["onset_seconds"]),
                        float(event["offset_seconds"]),
                        valid_frames,
                    )
                    if vector is None:
                        skipped["empty_window"] += 1
                        continue
                    values[f"{split}_x"].append(vector)
                    values[f"{split}_y"].append(
                        torch.tensor(label_to_index[label])
                    )
    if not values["train_x"] or not values["dev_x"]:
        raise RuntimeError("no train/dev oracle-window examples were recovered")
    train = (torch.stack(values["train_x"]), torch.stack(values["train_y"]).long())
    dev = (torch.stack(values["dev_x"]), torch.stack(values["dev_y"]).long())
    audit = {
        "scenes": dict(observed_scenes),
        "events": {"train": len(train[1]), "dev": len(dev[1])},
        "classes": {
            "train": int(train[1].unique().numel()),
            "dev": int(dev[1].unique().numel()),
        },
        "skipped": dict(skipped),
    }
    return train, dev, audit


class OracleSpanProbe(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict[str, float]:
    model.eval()
    logits: list[torch.Tensor] = []
    for start in range(0, len(y), 512):
        logits.append(model(x[start : start + 512].to(device)).cpu())
    score = torch.cat(logits)
    ranks = score.argsort(dim=1, descending=True)
    top1 = float((ranks[:, 0] == y).float().mean())
    top5 = float((ranks[:, :5] == y[:, None]).any(1).float().mean())
    top10 = float((ranks[:, :10] == y[:, None]).any(1).float().mean())
    per_class = []
    for label in y.unique().tolist():
        mask = y == label
        per_class.append(float((ranks[mask, 0] == y[mask]).float().mean()))
    return {
        "top1_accuracy": top1,
        "top5_accuracy": top5,
        "top10_accuracy": top10,
        "macro_top1_accuracy_observed_classes": float(np.mean(per_class)),
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dense_roots = args.dense_roots or [
        PROJECT_ROOT
        / "outputs/qces_qdor_dense_features/current_checkpoint_audioset_train"
    ]
    indexes = [
        json.loads((path / "index.json").read_text(encoding="utf-8"))
        for path in dense_roots
    ]
    if any(index["labels"] != indexes[0]["labels"] for index in indexes[1:]):
        raise RuntimeError("dense roots use different label orderings")
    index = indexes[0]
    labels = list(index["labels"])
    train_ids = _read_scene_ids(args.clean_root / "detector_manifest_train.jsonl")
    dev_ids = _read_scene_ids(args.clean_root / "detector_manifest_dev.jsonl")
    (train_x, train_y), (dev_x, dev_y), data_audit = load_examples(
        dense_roots,
        train_ids,
        dev_ids,
        labels,
        only_isolated_events=args.only_isolated_events,
    )

    device = torch.device(args.device)
    model = OracleSpanProbe(train_x.shape[1], args.hidden_dim, len(labels)).to(device)
    counts = torch.bincount(train_y, minlength=len(labels)).float()
    sample_weight = counts.clamp_min(1).reciprocal()[train_y]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(sample_weight, len(train_y), replacement=True, generator=generator)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

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
        if best is None or metrics["top1_accuracy"] > best["top1_accuracy"]:
            best = dict(row)
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    assert best is not None
    report = {
        "format": "qces_qdor_oracle_span_probe_v1",
        "status": "representation_gate_only_not_deployable",
        "input_contract": "oracle interval -> pooled BEATs hidden + detector logits; no label input",
        "warning": "The current detector encoder saw the AudioSet official-train pool before the clean dev regrouping; this probe is diagnostic, not paper test evidence.",
        "seed": args.seed,
        "only_isolated_events": args.only_isolated_events,
        "labels": len(labels),
        "dense_roots": [str(path.resolve()) for path in dense_roots],
        "data": data_audit,
        "best": best,
        # Canonical producer/consumer contract consumed by
        # train_qces_qdor_dense.load_oracle_gate.  Keep ``best`` unchanged for
        # diagnostics and historical receipt readers.
        "oracle_span_answer_top1": float(best["top1_accuracy"]),
        "history": history,
        "gate": {
            "metric": "dev oracle-span top1 accuracy",
            "threshold": ORACLE_GATE_THRESHOLD,
            "passes": best["top1_accuracy"] >= ORACLE_GATE_THRESHOLD,
        },
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "best": best, "gate": report["gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
