#!/usr/bin/env python3
"""Frozen ATST-Frame Strong oracle-interval semantic architecture screen."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import models.prediction_wrapper as prediction_wrapper_module
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.prediction_wrapper import PredictionsWrapper
from mixi_understanding.scripts.train_qces_local_semantic_r3 import LocalSemanticHead
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    SceneDataset,
    SceneItem,
    collate,
    load_ontology,
    load_scene_manifest,
    make_device,
    set_seed,
)
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _atomic_torch, _sha256_file


FORMAT = "qces_v5_atst_oracle_semantic_screen_v1"
CACHE_FORMAT = "qces_v5_atst_dense_feature_cache_v1"
NUM_FRAMES = 250


def parse_args() -> argparse.Namespace:
    data = Path("/var/tmp/qces_full188_tiered_realistic_v5")
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=data / "detector_scene_manifest_tiered_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=data / "detector_scene_manifest_tiered_dev.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--output-dir", type=Path, default=base / "v5_atst_oracle_semantic_screen_v1")
    parser.add_argument("--cache-dir", type=Path, default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6604)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--boundary-jitter-frames", type=int, default=2)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def semantic_events(row: SceneItem) -> list[Mapping[str, Any]]:
    return [event for event in row.events if str(event.get("event_kind", "semantic")) == "semantic"]


def source_audit(train_manifest: Path, dev_manifest: Path) -> dict[str, Any]:
    def values(path: Path) -> tuple[set[str], set[str]]:
        sources: set[str] = set(); hashes: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for event in row["events"]:
                sources.add(str(event.get("source_id") or ""))
                hashes.add(str(event.get("source_sha256") or ""))
        return sources - {""}, hashes - {""}
    train_sources, train_hashes = values(train_manifest)
    dev_sources, dev_hashes = values(dev_manifest)
    result = {
        "train_sources": len(train_sources),
        "dev_sources": len(dev_sources),
        "source_id_overlap": len(train_sources & dev_sources),
        "source_sha256_overlap": len(train_hashes & dev_hashes),
    }
    if result["source_id_overlap"] or result["source_sha256_overlap"]:
        raise ValueError(f"train/dev source leakage: {result}")
    return result


@torch.inference_mode()
def export_cache(
    backbone: PredictionsWrapper,
    rows: Sequence[SceneItem],
    device: torch.device,
    *,
    batch_size: int,
    num_workers: int,
    amp: bool,
    split: str,
) -> dict[str, Any]:
    loader = DataLoader(
        SceneDataset(rows, audio_root=Path("/"), fixed_seconds=10.0),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )
    features: list[torch.Tensor] = []
    processed = 0
    backbone.eval()
    for waveforms, batch_rows in loader:
        audio = waveforms.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            mel = backbone.mel_forward(audio)
            value = backbone.model(mel)
        if value.shape[1] != NUM_FRAMES:
            value = F.interpolate(value.transpose(1, 2), size=NUM_FRAMES, mode="linear", align_corners=False).transpose(1, 2)
        features.append(value.cpu().half())
        processed += len(batch_rows)
        if processed == len(batch_rows) or processed % 256 < len(batch_rows):
            print(f"atst_feature_export split={split} {processed}/{len(rows)}", flush=True)
    return {
        "format": CACHE_FORMAT,
        "split": split,
        "scene_id": [row.scene_id for row in rows],
        "features": torch.cat(features),
        "intervals": [
            torch.tensor([[float(event["onset_seconds"]) / row.duration_seconds, float(event["offset_seconds"]) / row.duration_seconds] for event in semantic_events(row)], dtype=torch.float32)
            for row in rows
        ],
        "labels": [torch.tensor([int(event["label_id"]) for event in semantic_events(row)], dtype=torch.long) for row in rows],
    }


class CachedSceneDataset(Dataset[dict[str, Any]]):
    def __init__(self, cache: Mapping[str, Any]) -> None:
        self.features = cache["features"]
        self.intervals = cache["intervals"]
        self.labels = cache["labels"]

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"features": self.features[index], "intervals": self.intervals[index], "labels": self.labels[index]}


def cached_collate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "features": torch.stack([row["features"] for row in rows]),
        "intervals": [row["intervals"] for row in rows],
        "labels": [row["labels"] for row in rows],
    }


def collect_spans(
    features: torch.Tensor,
    intervals: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    *,
    jitter: int,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences: list[torch.Tensor] = []; targets: list[int] = []
    for batch_index, (scene_intervals, scene_labels) in enumerate(zip(intervals, labels)):
        for interval, label in zip(scene_intervals, scene_labels):
            start = int(math.floor(float(interval[0]) * NUM_FRAMES)); end = int(math.ceil(float(interval[1]) * NUM_FRAMES))
            start = max(0, min(NUM_FRAMES - 1, start)); end = max(start + 1, min(NUM_FRAMES, end))
            if training and jitter:
                start += random.randint(-jitter, jitter); end += random.randint(-jitter, jitter)
                start = max(0, min(NUM_FRAMES - 1, start)); end = max(start + 1, min(NUM_FRAMES, end))
            sequences.append(features[batch_index, start:end]); targets.append(int(label))
    maximum = max(value.shape[0] for value in sequences)
    padded = features.new_zeros(len(sequences), maximum, features.shape[-1])
    mask = torch.zeros(len(sequences), maximum, dtype=torch.bool, device=features.device)
    for index, value in enumerate(sequences):
        padded[index, : value.shape[0]] = value; mask[index, : value.shape[0]] = True
    return padded, mask, torch.tensor(targets, dtype=torch.long, device=features.device)


@torch.inference_mode()
def evaluate(head: LocalSemanticHead, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    head.eval(); scores: list[torch.Tensor] = []; targets: list[torch.Tensor] = []
    for batch in loader:
        value = batch["features"].to(device=device, dtype=torch.float32)
        spans, mask, target = collect_spans(value, batch["intervals"], batch["labels"], jitter=0, training=False)
        scores.append(head(spans, mask).cpu()); targets.append(target.cpu())
    score = torch.cat(scores); target = torch.cat(targets); ordering = score.argsort(dim=-1, descending=True)
    class_total: Counter[int] = Counter(target.tolist()); class_correct: Counter[int] = Counter()
    for label, predicted in zip(target.tolist(), ordering[:, 0].tolist()):
        class_correct[label] += int(label == predicted)
    return {
        "events": int(target.numel()),
        "observed_classes": len(class_total),
        "top1_accuracy_\u2191": float(ordering[:, 0].eq(target).float().mean()),
        "top5_accuracy_\u2191": float((ordering[:, :5] == target[:, None]).any(dim=-1).float().mean()),
        "top20_accuracy_\u2191": float((ordering[:, :20] == target[:, None]).any(dim=-1).float().mean()),
        "macro_top1_accuracy_\u2191": float(sum(class_correct[key] / value for key, value in class_total.items()) / len(class_total)),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    output_dir = args.output_dir.resolve(); cache_dir = args.cache_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True); cache_dir.mkdir(parents=True, exist_ok=True)
    labels = load_ontology(args.ontology.resolve()); label_map = {label: index for index, label in enumerate(labels)}
    train_rows = load_scene_manifest(args.train_manifest.resolve(), label_map); dev_rows = load_scene_manifest(args.dev_manifest.resolve(), label_map)
    if args.max_train_scenes: train_rows = train_rows[: args.max_train_scenes]
    if args.max_dev_scenes: dev_rows = dev_rows[: args.max_dev_scenes]
    audit = source_audit(args.train_manifest.resolve(), args.dev_manifest.resolve())
    device = make_device(args.device)
    # PretrainedSED keeps this as a process-CWD-relative module constant.  Pin it
    # to the checked official resource so launches from the QCES root cannot
    # silently download/load a different file under ./resources.
    prediction_wrapper_module.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
    backbone = PredictionsWrapper(
        ATSTWrapper(), checkpoint="ATST-F_strong_1", n_classes_strong=len(labels), n_classes_weak=len(labels), seq_model_type=None, head_type="linear"
    ).to(device)
    backbone.eval().requires_grad_(False)
    cache_paths = {"train": cache_dir / "atst_features_train.pt", "dev": cache_dir / "atst_features_dev.pt"}
    caches = {}
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        cache = export_cache(backbone, rows, device, batch_size=args.feature_batch_size, num_workers=args.num_workers, amp=args.amp, split=split)
        _atomic_torch(cache, cache_paths[split]); caches[split] = cache
    del backbone
    if device.type == "cuda": torch.cuda.empty_cache()
    train_dataset = CachedSceneDataset(caches["train"]); dev_dataset = CachedSceneDataset(caches["dev"])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, generator=generator, num_workers=0, collate_fn=cached_collate)
    dev_loader = DataLoader(dev_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=0, collate_fn=cached_collate)
    input_dim = int(caches["train"]["features"].shape[-1])
    head = LocalSemanticHead(input_dim, args.hidden_dim, len(labels)).to(device)
    counts = Counter(int(label) for values in caches["train"]["labels"] for label in values.tolist())
    weight = torch.tensor([1.0 / math.sqrt(max(counts.get(index, 1), 1)) for index in range(len(labels))], device=device)
    weight = (weight / weight.mean()).clamp(0.5, 2.5)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    best = None; best_epoch = 0; best_state = None; stale = 0; history = []
    for epoch in range(1, args.epochs + 1):
        head.train(); total = correct = seen = 0
        for batch in train_loader:
            value = batch["features"].to(device=device, dtype=torch.float32)
            spans, mask, target = collect_spans(value, batch["intervals"], batch["labels"], jitter=args.boundary_jitter_frames, training=True)
            logits = head(spans, mask)
            loss = F.cross_entropy(logits, target, weight=weight, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0); optimizer.step()
            total += float(loss.detach()) * target.numel(); correct += int(logits.argmax(dim=-1).eq(target).sum()); seen += int(target.numel())
        scheduler.step(); metrics = evaluate(head, dev_loader, device)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_loss": total / max(seen, 1), "train_top1": correct / max(seen, 1), "dev": metrics}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        key = (float(metrics["top1_accuracy_\u2191"]), float(metrics["macro_top1_accuracy_\u2191"]), float(metrics["top5_accuracy_\u2191"]))
        if best is None or key > best:
            best = key; best_epoch = epoch; best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}; stale = 0
        else: stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True); break
    if best_state is None: raise RuntimeError("no checkpoint")
    checkpoint_path = output_dir / "atst_oracle_semantic_head_v1_best.pt"
    best_metrics = history[best_epoch - 1]["dev"]
    _atomic_torch({"format": FORMAT, "model_state_dict": best_state, "input_dim": input_dim, "hidden_dim": args.hidden_dim, "labels": labels, "best_epoch": best_epoch, "best_metrics": best_metrics}, checkpoint_path)
    gates = {
        "oracle_top1_ge_0_65": float(best_metrics["top1_accuracy_\u2191"]) >= 0.65,
        "oracle_macro_top1_ge_0_60": float(best_metrics["macro_top1_accuracy_\u2191"]) >= 0.60,
        "oracle_top5_ge_0_90": float(best_metrics["top5_accuracy_\u2191"]) >= 0.90,
    }
    atst_resource = PRETRAINED_SED_ROOT / "resources/ATST-F_strong_1.pt"
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen-backbone oracle-interval architecture screen; not deployable end-to-end",
        "qa_question_or_answer_used_as_input": False,
        "data": {"train_scenes": len(train_rows), "dev_scenes": len(dev_rows), "train_events": sum(counts.values()), "classes": len(labels), "identity_audit": audit},
        "backbone": "ATST-F_strong_1 frozen",
        "beats_reference": {"full191_last2_blocks_oracle_top1": 0.5539906025, "v5_interval_curriculum_oracle_top1": 0.5803},
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
        "gates": gates,
        "decision": "integrate_atst_with_predicted_slots" if all(gates.values()) else "atst_oracle_semantics_insufficient",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "atst_resource_sha256": _sha256_file(atst_resource),
        "cache": {split: {"path": str(path), "sha256": _sha256_file(path)} for split, path in cache_paths.items()},
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "best_metrics": best_metrics, "gates": gates, "decision": receipt["decision"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
