#!/usr/bin/env python3
"""Source-disjoint generalization pilot for the V5 joint scene separator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from mixi_understanding.scripts.train_qces_v5_joint_convtasnet_micro_v1 import (
    CHECKPOINT_FORMAT,
    SAMPLE_RATE,
    SAMPLES,
    SceneDataset,
    SceneRef,
    atomic_json,
    atomic_torch,
    evaluate,
    model_forward,
    sha256_file,
    training_loss,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAT = "qces_v5_joint_convtasnet_generalization_pilot_receipt_v1"


def parse_args() -> argparse.Namespace:
    data = Path("/var/tmp/qces_full188_tiered_realistic_v5")
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, default=data / "detector_scene_manifest_tiered_train.jsonl")
    parser.add_argument("--dev-manifest", type=Path, default=data / "detector_scene_manifest_tiered_dev.jsonl")
    parser.add_argument("--output-dir", type=Path, default=base / "v5_joint_convtasnet_pilot_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8221)
    parser.add_argument("--max-train-scenes", type=int, default=256)
    parser.add_argument("--max-dev-scenes", type=int, default=128)
    parser.add_argument("--train-audit-scenes", type=int, default=32)
    parser.add_argument("--num-sources", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--enc-num-feats", type=int, default=128)
    parser.add_argument("--mask-num-feats", type=int, default=64)
    parser.add_argument("--mask-hidden-feats", type=int, default=128)
    parser.add_argument("--mask-num-layers", type=int, default=6)
    parser.add_argument("--mask-num-stacks", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def stable_scenes(path: Path, maximum: int, num_sources: int, seed: int) -> list[SceneRef]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if 1 < len(row["events"]) <= num_sources]
    # Hash selection preserves the natural sparse/moderate/hard distribution;
    # no target waveform or QA answer is used to select examples.
    rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['scene_id']}".encode()).hexdigest())
    return [SceneRef(row) for row in rows[:maximum]]


def source_ids(scenes: list[SceneRef]) -> set[str]:
    return {str(event["source_id"]) for scene in scenes for event in scene.row["events"]}


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "items"}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_scenes = stable_scenes(args.train_manifest.resolve(), args.max_train_scenes, args.num_sources, args.seed)
    dev_scenes = stable_scenes(args.dev_manifest.resolve(), args.max_dev_scenes, args.num_sources, args.seed + 1)
    overlap = source_ids(train_scenes) & source_ids(dev_scenes)
    if overlap:
        raise ValueError(f"train/dev original source leakage: {len(overlap)}")
    train_dataset = SceneDataset(train_scenes, args.num_sources)
    train_audit_dataset = SceneDataset(train_scenes[: args.train_audit_scenes], args.num_sources)
    dev_dataset = SceneDataset(dev_scenes, args.num_sources)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    architecture = {
        "num_sources": args.num_sources,
        "enc_num_feats": args.enc_num_feats,
        "mask_num_feats": args.mask_num_feats,
        "mask_num_hidden_feats": args.mask_hidden_feats,
        "mask_num_layers": args.mask_num_layers,
        "mask_num_stacks": args.mask_num_stacks,
    }
    model = torchaudio.models.ConvTasNet(
        num_sources=args.num_sources,
        enc_num_feats=args.enc_num_feats,
        msk_num_feats=args.mask_num_feats,
        msk_num_hidden_feats=args.mask_hidden_feats,
        msk_num_layers=args.mask_num_layers,
        msk_num_stacks=args.mask_num_stacks,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.03)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train_dataset, batch_size=1, shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == "cuda")
    baseline = evaluate(model, dev_dataset, device, args.amp)
    atomic_json(baseline, output_dir / "baseline_dev_metrics.json")
    print(json.dumps({"baseline_dev": compact(baseline)}, sort_keys=True), flush=True)
    best_key = -math.inf
    best_epoch = 0
    best_dev = baseline
    best_train = evaluate(model, train_audit_dataset, device, args.amp)
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for sample in loader:
            mixture = sample["mixture"].to(device)
            targets = sample["targets"][0].to(device)
            num_active = int(sample["num_active"].item())
            prediction = model_forward(model, mixture, args.amp)[0]
            loss, details = training_loss(prediction, mixture[0], targets, num_active)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for key, value in details.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
        scheduler.step()
        dev_metrics = evaluate(model, dev_dataset, device, args.amp)
        train_metrics = evaluate(model, train_audit_dataset, device, args.amp)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": {key: value / max(1, batches) for key, value in totals.items()},
            "train_audit": compact(train_metrics),
            "dev": compact(dev_metrics),
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = float(dev_metrics["si_sdri_vs_raw_mixture_db_\u2191"]["median"])
        if key > best_key + 0.05:
            best_key = key
            best_epoch = epoch
            best_dev = dev_metrics
            best_train = train_metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break
    checkpoint_path = output_dir / "joint_convtasnet_pilot_v1_best.pt"
    atomic_torch({
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": best_state,
        "architecture": architecture,
        "best_epoch": best_epoch,
        "sample_rate": SAMPLE_RATE,
        "samples": SAMPLES,
    }, checkpoint_path)
    gates = {
        "dev_median_si_sdri_vs_raw_mixture_ge_1dB": float(best_dev["si_sdri_vs_raw_mixture_db_\u2191"]["median"]) >= 1.0,
        "dev_positive_fraction_vs_raw_mixture_ge_0_65": float(best_dev["si_sdri_vs_raw_mixture_db_\u2191"]["positive_fraction"]) >= 0.65,
        "dev_overlap_median_within_2dB_of_oracle_span": float(best_dev["overlap_events_si_sdri_vs_oracle_span_db_\u2191"]["median"]) >= -2.0,
    }
    decision = "scale_to_full_v5" if all(gates.values()) else "do_not_scale_analyze_generalization_gap"
    receipt = {
        "format": FORMAT,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "source-disjoint pilot; not a full-dataset result",
        "arguments": vars(args) | {
            "train_manifest": str(args.train_manifest.resolve()),
            "dev_manifest": str(args.dev_manifest.resolve()),
            "output_dir": str(output_dir),
        },
        "architecture": architecture,
        "data_audit": {
            "train_scenes": len(train_scenes),
            "dev_scenes": len(dev_scenes),
            "train_original_sources": len(source_ids(train_scenes)),
            "dev_original_sources": len(source_ids(dev_scenes)),
            "train_dev_original_source_overlap": len(overlap),
            "train_profiles": {profile: sum(s.row.get("layout_profile") == profile for s in train_scenes) for profile in ("sparse", "moderate", "hard")},
            "dev_profiles": {profile: sum(s.row.get("layout_profile") == profile for s in dev_scenes) for profile in ("sparse", "moderate", "hard")},
        },
        "baseline_dev": baseline,
        "best_epoch": best_epoch,
        "best_train_audit": best_train,
        "best_dev": best_dev,
        "history": history,
        "gates": gates,
        "decision": decision,
        "checkpoint": str(checkpoint_path),
    }
    receipt_path = output_dir / "training_receipt.json"
    atomic_json(receipt, receipt_path)
    receipt["artifacts"] = {"checkpoint_sha256": sha256_file(checkpoint_path), "receipt": str(receipt_path)}
    atomic_json(receipt, receipt_path)
    print(json.dumps({"receipt": str(receipt_path), "best_epoch": best_epoch, "gates": gates, "decision": decision}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
