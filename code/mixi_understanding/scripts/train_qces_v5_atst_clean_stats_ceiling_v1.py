#!/usr/bin/env python3
"""Train a clean-component ATST statistics teacher and measure its ceiling."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import models.prediction_wrapper as prediction_wrapper_module
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.prediction_wrapper import PredictionsWrapper
from mixi_understanding.scripts.audit_qces_v5_matched_clean_mixture_gap_v1 import (
    NUM_FRAMES,
    event_rows,
    load_component_canvas,
    read_scenes,
)
from mixi_understanding.scripts.audit_qces_v5_targeted_logit_calibration_v1 import compact_metric
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _atomic_torch, _sha256_file


FORMAT = "qces_v5_atst_clean_stats_ceiling_v1"


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full188_tiered_realistic_v5"
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=data / "detector_scene_manifest_tiered_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=data / "detector_scene_manifest_tiered_dev.jsonl")
    parser.add_argument("--matched-manifest", type=Path, default=Path("/var/tmp/qces_v5_matched_eval_v1/detector_scene_manifest_matched_eval.jsonl"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/var/tmp/qces_v5_atst_clean_stats_ceiling_v1"))
    parser.add_argument("--output-dir", type=Path, default=base / "v5_atst_clean_stats_ceiling_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7501)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class StatsHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value.float())


@torch.inference_mode()
def export_stats(
    backbone: PredictionsWrapper,
    events: Sequence[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    amp: bool,
    split: str,
) -> dict[str, Any]:
    stats: list[torch.Tensor] = []
    targets: list[int] = []
    for begin in range(0, len(events), batch_size):
        batch_events = events[begin : begin + batch_size]
        waveforms = torch.stack([load_component_canvas(event) for event in batch_events]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            mel = backbone.mel_forward(waveforms)
            features = backbone.model(mel)
        if int(features.shape[1]) != NUM_FRAMES:
            features = F.interpolate(
                features.transpose(1, 2), size=NUM_FRAMES, mode="linear", align_corners=False
            ).transpose(1, 2)
        for index, event in enumerate(batch_events):
            start = max(0, min(NUM_FRAMES - 1, int(event["onset_frame"])))
            end = max(start + 1, min(NUM_FRAMES, int(event["offset_frame"])))
            span = features[index, start:end].float()
            stats.append(torch.cat((span.mean(0), span.amax(0), span.std(0, unbiased=False))).cpu().half())
            targets.append(int(event["label_id"]))
        processed = min(begin + len(batch_events), len(events))
        if processed == len(batch_events) or processed % 512 < len(batch_events):
            print(f"clean_stats_export split={split} {processed}/{len(events)}", flush=True)
    return {
        "format": FORMAT + "_cache",
        "split": split,
        "event_id": [event["event_id"] for event in events],
        "scene_id": [event["scene_id"] for event in events],
        "features": torch.stack(stats),
        "targets": torch.tensor(targets, dtype=torch.long),
    }


@torch.inference_mode()
def evaluate(model: StatsHead, cache: dict[str, Any], device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    scores: list[torch.Tensor] = []
    features = cache["features"]
    for begin in range(0, len(features), batch_size):
        scores.append(model(features[begin : begin + batch_size].to(device)).cpu())
    return compact_metric(torch.cat(scores), cache["targets"].long())


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = {
        "train": args.train_manifest.resolve(),
        "dev": args.dev_manifest.resolve(),
        "matched_eval": args.matched_manifest.resolve(),
    }
    split_events = {split: event_rows(read_scenes(path)) for split, path in manifest_paths.items()}
    prediction_wrapper_module.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
    backbone = PredictionsWrapper(
        ATSTWrapper(), checkpoint="ATST-F_strong_1", n_classes_strong=188,
        n_classes_weak=188, seq_model_type=None, head_type="linear",
    ).to(device)
    backbone.eval().requires_grad_(False)
    caches: dict[str, dict[str, Any]] = {}
    cache_paths: dict[str, Path] = {}
    for split in ("train", "dev", "matched_eval"):
        cache = export_stats(
            backbone, split_events[split], device=device,
            batch_size=args.feature_batch_size, amp=args.amp, split=split,
        )
        path = cache_dir / f"clean_stats_{split}.pt"
        _atomic_torch(cache, path)
        caches[split] = cache
        cache_paths[split] = path
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    input_dim = int(caches["train"]["features"].shape[-1])
    model = StatsHead(input_dim, args.hidden_dim, 188).to(device)
    counts = Counter(caches["train"]["targets"].tolist())
    class_weight = torch.tensor(
        [1.0 / math.sqrt(max(counts.get(index, 1), 1)) for index in range(188)], device=device
    )
    class_weight = (class_weight / class_weight.mean()).clamp(0.5, 2.5)
    train_loader = DataLoader(
        TensorDataset(caches["train"]["features"], caches["train"]["targets"]),
        batch_size=args.train_batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    best_key: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_dev: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, targets, weight=class_weight, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            correct += int(logits.argmax(1).eq(targets).sum())
            seen += int(targets.numel())
        scheduler.step()
        dev = evaluate(model, caches["dev"], device, args.train_batch_size)
        row = {
            "epoch": epoch, "train_loss": total_loss / seen,
            "train_top1": correct / seen, "dev": dev,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (dev["top1_accuracy_↑"], dev["macro_top1_accuracy_↑"], dev["top5_accuracy_↑"])
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_dev = dev
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break
    if best_state is None or best_dev is None:
        raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state)
    matched = evaluate(model, caches["matched_eval"], device, args.train_batch_size)
    checkpoint_path = output_dir / "atst_clean_stats_ceiling_v1_best.pt"
    _atomic_torch({
        "format": FORMAT, "model_state_dict": best_state, "input_dim": input_dim,
        "hidden_dim": args.hidden_dim, "classes": 188, "best_epoch": best_epoch,
        "dev_metrics": best_dev, "matched_eval_metrics": matched,
    }, checkpoint_path)
    gates = {
        "clean_dev_top1_ge_0_75": best_dev["top1_accuracy_↑"] >= 0.75,
        "clean_matched_eval_top1_ge_0_70": matched["top1_accuracy_↑"] >= 0.70,
        "clean_matched_eval_top5_ge_0_90": matched["top5_accuracy_↑"] >= 0.90,
    }
    report = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "clean-component semantic ceiling; matched evaluation read only after dev checkpoint selection",
        "backbone": "official ATST-F Strong frozen",
        "representation": "oracle active-span raw ATST mean+max+std",
        "data": {split: {"events": len(events), "scenes": len(set(e["scene_id"] for e in events))} for split, events in split_events.items()},
        "best_epoch": best_epoch,
        "dev": best_dev,
        "matched_eval": matched,
        "history": history,
        "gates": {"passed": all(gates.values()), "checks": gates},
        "decision": "distill_clean_teacher_into_mixture" if all(gates.values()) else "representation_or_taxonomy_still_limits_clean_semantics",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "artifacts": {
            split: {"cache_sha256": _sha256_file(cache_paths[split]), "manifest_sha256": _sha256_file(manifest_paths[split])}
            for split in cache_paths
        },
    }
    _atomic_json(report, output_dir / "receipt.json")
    print(json.dumps({
        "complete": True, "best_epoch": best_epoch, "dev": best_dev,
        "matched_eval": matched, "gates": report["gates"], "decision": report["decision"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
