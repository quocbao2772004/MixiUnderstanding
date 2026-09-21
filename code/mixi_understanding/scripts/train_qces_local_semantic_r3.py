#!/usr/bin/env python3
"""R3 local semantic adapter for the full191 QCES pipeline.

R1 remains the frozen detector/presence path.  This script creates a separate
copy of its BEATs encoder and trains only the final transformer blocks plus an
attention-pooling 191-way softmax head.  The supervision is event-local: the
classifier sees the waveform and a gold interval, but never the gold label as
input.  This directly addresses the measured top-1 semantic bottleneck
without changing R1's no-evidence calibration.
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
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline" / "PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(value))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    SceneDataset,
    SceneItem,
    collate,
    interpolate_sequence,
    load_model,
    load_ontology,
    load_scene_manifest,
    make_device,
    set_seed,
)


FORMAT = "qces_local_semantic_r3_v1"
FRAME_HOP_SECONDS = 0.04
DEFAULT_DATA = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
DEFAULT_R1 = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1/pretrainedsed_beats_qces_detector.pt"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/local_semantic_r3_v1"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_DATA / "detector_scene_manifest_multi_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=DEFAULT_DATA / "detector_scene_manifest_multi_dev.jsonl")
    parser.add_argument("--ontology", type=Path, default=DEFAULT_DATA / "ontology_191.txt")
    parser.add_argument("--r1-checkpoint", type=Path, default=DEFAULT_R1)
    parser.add_argument("--audio-root", type=Path, default=Path("/"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-name", default="BEATs_strong_1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=2)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--boundary-jitter-frames", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-3)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2053)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.num_workers < 0:
        parser.error("epochs/batch-size must be positive and num-workers non-negative")
    if not 1 <= args.unfreeze_last_blocks <= 4:
        parser.error("use 1--4 final blocks for conservative local adaptation")
    return args


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class LocalSemanticHead(nn.Module):
    """Frame projection plus learned pooling; no query/label is supplied."""

    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int) -> None:
        super().__init__()
        self.frame = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, spans: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.frame(spans)
        valid = mask.unsqueeze(-1)
        denominator = valid.sum(1).clamp_min(1)
        mean = (hidden * valid).sum(1) / denominator
        maximum = hidden.masked_fill(~valid, -1e4).amax(1)
        attention_logits = self.attention(hidden).squeeze(-1).masked_fill(~mask, -1e4)
        attention = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        attended = (hidden * attention).sum(1)
        return self.classifier(torch.cat((mean, maximum, attended), dim=-1))


def encoder_features(model: nn.Module, waveforms: torch.Tensor) -> torch.Tensor:
    mel = model.mel_forward(waveforms)
    features = model.model(mel)
    features = interpolate_sequence(features, model.seq_len)
    return model.seq_model(features)


def semantic_events(row: SceneItem) -> list[Mapping[str, Any]]:
    return [event for event in row.events if str(event.get("event_kind", "semantic")) == "semantic"]


def collect_spans(
    features: torch.Tensor,
    rows: Sequence[SceneItem],
    *,
    jitter_frames: int,
    training: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences: list[torch.Tensor] = []
    targets: list[int] = []
    time_steps = int(features.shape[1])
    for batch_index, row in enumerate(rows):
        duration = max(float(row.duration_seconds), 1e-6)
        for event in semantic_events(row):
            start = max(0, min(time_steps - 1, int(math.floor(float(event["onset_seconds"]) / duration * time_steps))))
            end = max(start + 1, min(time_steps, int(math.ceil(float(event["offset_seconds"]) / duration * time_steps))))
            if training and jitter_frames > 0:
                start += random.randint(-jitter_frames, jitter_frames)
                end += random.randint(-jitter_frames, jitter_frames)
                start = max(0, min(time_steps - 1, start))
                end = max(start + 1, min(time_steps, end))
            sequences.append(features[batch_index, start:end])
            targets.append(int(event["label_id"]))
    if not sequences:
        raise RuntimeError("batch has no semantic events")
    max_length = max(sequence.shape[0] for sequence in sequences)
    padded = features.new_zeros((len(sequences), max_length, features.shape[-1]))
    mask = torch.zeros((len(sequences), max_length), dtype=torch.bool, device=features.device)
    for index, sequence in enumerate(sequences):
        padded[index, : sequence.shape[0]] = sequence
        mask[index, : sequence.shape[0]] = True
    return padded, mask, torch.tensor(targets, dtype=torch.long, device=features.device)


def topk_metrics(score: torch.Tensor, target: torch.Tensor, num_labels: int) -> dict[str, float]:
    ranks = score.argsort(dim=1, descending=True)
    result: dict[str, float] = {}
    for k in (1, 5, 10):
        result[f"top{k}_accuracy"] = float((ranks[:, :k] == target[:, None]).any(1).float().mean())
    per_class: list[float] = []
    for label_id in target.unique().tolist():
        selected = target == int(label_id)
        per_class.append(float((ranks[selected, 0] == target[selected]).float().mean()))
    result["macro_top1_accuracy_observed_classes"] = float(np.mean(per_class))
    result["observed_classes"] = int(target.unique().numel())
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    head: LocalSemanticHead,
    loader: DataLoader,
    device: torch.device,
    num_labels: int,
) -> dict[str, float]:
    model.eval()
    head.eval()
    scores: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for waveforms, rows in loader:
        features = encoder_features(model, waveforms.to(device, non_blocking=True))
        spans, mask, target = collect_spans(features, rows, jitter_frames=0, training=False)
        scores.append(head(spans, mask).cpu())
        targets.append(target.cpu())
    return topk_metrics(torch.cat(scores), torch.cat(targets), num_labels)


def trainable_adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in trainable}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    checkpoint_path = output_dir / "local_semantic_r3_best.pt"
    report_path = output_dir / "report.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    labels = load_ontology(args.ontology.resolve())
    train_rows = load_scene_manifest(args.train_manifest.resolve(), {label: index for index, label in enumerate(labels)})
    dev_rows = load_scene_manifest(args.dev_manifest.resolve(), {label: index for index, label in enumerate(labels)})
    if args.max_train_scenes:
        train_rows = train_rows[: args.max_train_scenes]
    if args.max_dev_scenes:
        dev_rows = dev_rows[: args.max_dev_scenes]
    if {row.scene_id for row in train_rows} & {row.scene_id for row in dev_rows}:
        raise RuntimeError("train/dev scene leakage")

    device = make_device(args.device)
    model = load_model(
        len(labels),
        args.checkpoint_name,
        device,
        unfreeze_last_blocks=args.unfreeze_last_blocks,
    )
    r1 = torch.load(args.r1_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if list(r1.get("labels", [])) != labels:
        raise RuntimeError("R1 checkpoint/ontology label mismatch")
    model.load_state_dict(r1["model_state_dict"], strict=True)
    model.strong_head.requires_grad_(False)
    model.weak_head.requires_grad_(False)
    head = LocalSemanticHead(model.strong_head.in_features, args.hidden_dim, len(labels)).to(device)

    trainable_backbone = [parameter for parameter in model.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable_backbone, "lr": args.backbone_lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        SceneDataset(train_rows, audio_root=args.audio_root.resolve(), fixed_seconds=10.0),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )
    dev_loader = DataLoader(
        SceneDataset(dev_rows, audio_root=args.audio_root.resolve(), fixed_seconds=10.0),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    class_counts = Counter(
        int(event["label_id"])
        for row in train_rows
        for event in semantic_events(row)
    )
    raw_weight = torch.tensor(
        [1.0 / math.sqrt(max(class_counts.get(index, 1), 1)) for index in range(len(labels))],
        dtype=torch.float32,
        device=device,
    )
    class_weight = (raw_weight / raw_weight.mean()).clamp(0.5, 2.0)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    stale = 0
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    for epoch in range(1, args.epochs + 1):
        model.eval()
        # Activate dropout only in the trainable final transformer blocks.
        for layer_index in range(12 - args.unfreeze_last_blocks, 12):
            model.model.beats.encoder.layers[layer_index].train()
        head.train()
        losses: list[float] = []
        seen_events = 0
        for step, (waveforms, rows) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                features = encoder_features(model, waveforms.to(device, non_blocking=True))
                spans, mask, target = collect_spans(
                    features,
                    rows,
                    jitter_frames=args.boundary_jitter_frames,
                    training=True,
                )
                score = head(spans, mask)
                loss = F.cross_entropy(
                    score,
                    target,
                    weight=class_weight,
                    label_smoothing=args.label_smoothing,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_backbone + list(head.parameters()), 5.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            seen_events += int(target.numel())
            if step % 100 == 0:
                print(
                    json.dumps({"epoch": epoch, "step": step, "events": seen_events, "loss": float(np.mean(losses[-100:]))}),
                    flush=True,
                )
        metrics = evaluate(model, head, dev_loader, device, len(labels))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_events": seen_events, **metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        improved = best is None or row["top1_accuracy"] >= best["top1_accuracy"] + args.early_stopping_min_delta
        if improved:
            best = dict(row)
            stale = 0
            atomic_torch(
                checkpoint_path,
                {
                    "format": FORMAT,
                    "labels": labels,
                    "r1_checkpoint": str(args.r1_checkpoint.resolve()),
                    "unfreeze_last_blocks": args.unfreeze_last_blocks,
                    "trainable_parameter_names": trainable_names,
                    "encoder_adapter_state_dict": trainable_adapter_state(model),
                    "local_head_state_dict": {name: value.detach().cpu() for name, value in head.state_dict().items()},
                    "head_config": {"input_dim": model.strong_head.in_features, "hidden_dim": args.hidden_dim},
                    "best": best,
                },
            )
        else:
            stale += 1
        if args.early_stopping_patience > 0 and stale >= args.early_stopping_patience:
            print(json.dumps({"early_stopping": True, "epoch": epoch, "best": best}), flush=True)
            break

    assert best is not None
    report = {
        "format": FORMAT,
        "status": "complete",
        "purpose": "local semantic classification only; R1 detector/no-evidence path remains frozen",
        "labels": len(labels),
        "data": {
            "train_scenes": len(train_rows),
            "dev_scenes": len(dev_rows),
            "train_events": sum(class_counts.values()),
            "train_classes": len(class_counts),
        },
        "configuration": vars(args) | {
            "train_manifest": str(args.train_manifest.resolve()),
            "dev_manifest": str(args.dev_manifest.resolve()),
            "ontology": str(args.ontology.resolve()),
            "r1_checkpoint": str(args.r1_checkpoint.resolve()),
            "output_dir": str(output_dir),
            "audio_root": str(args.audio_root.resolve()),
            "device": str(device),
        },
        "best": best,
        "gate": {
            "metric": "full191 dev oracle-span top1 accuracy",
            "threshold": 0.75,
            "passes": bool(best["top1_accuracy"] >= 0.75),
        },
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path), "best": best, "gate": report["gate"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
