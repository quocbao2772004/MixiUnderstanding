#!/usr/bin/env python3
"""Rehearsal fine-tuning for the V5 main and V4 stress localizer domains.

This is an independent sidecar: historical trainers and checkpoints are not
modified.  Training samples V5 realistic scenes and V4 high-overlap scenes at
a fixed ratio.  A checkpoint is selected using the harmonic mean of the two
dev F1 scores, with one shared objectness threshold and the mask decoder fixed
before this run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from mixi_understanding.qces.overlap_event_slots_v2 import (
    OverlapEventSlotsV2,
    OverlapEventSlotsV2LossWeights,
    overlap_event_slots_v2_loss,
)
from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1Config,
    RelationalEventSlotsV1LossWeights,
)
from mixi_understanding.scripts.evaluate_qces_slot_mask_interval_decoder_v1 import _decode
from mixi_understanding.scripts.train_qces_overlap_event_slots_v2 import (
    CHECKPOINT_FORMAT,
    OverlapSceneSlotDataset,
    _to_device,
    collate_overlap_slots,
    collect_predictions,
    evaluate_objectness_threshold,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _atomic_torch,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import assert_dense_identity_disjoint_v2


FORMAT = "qces_overlap_event_slots_rehearsal_receipt_v1"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _grid(value: str) -> list[float]:
    result = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if not result or any(item < 0.0 or item > 1.0 for item in result):
        raise argparse.ArgumentTypeError("thresholds must lie in [0,1]")
    return result


def _harmonic(left: float, right: float) -> float:
    return 2.0 * left * right / max(left + right, 1e-12)


def _joint_row(
    main_predictions: Mapping[str, Mapping[str, Any]],
    stress_predictions: Mapping[str, Mapping[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    main = evaluate_objectness_threshold(main_predictions, objectness_threshold=threshold)
    stress = evaluate_objectness_threshold(stress_predictions, objectness_threshold=threshold)
    main_f1 = float(main["slot_f1_iou50"])
    stress_f1 = float(stress["slot_f1_iou50"])
    return {
        "objectness_threshold": threshold,
        "main": main,
        "stress": stress,
        "harmonic_f1_iou50": _harmonic(main_f1, stress_f1),
        "minimum_f1_iou50": min(main_f1, stress_f1),
        "mean_matched_iou": (
            float(main["slot_mean_matched_iou"])
            + float(stress["slot_mean_matched_iou"])
        )
        / 2.0,
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    main = row["main"]
    stress = row["stress"]
    return (
        float(row["harmonic_f1_iou50"]),
        float(row["minimum_f1_iou50"]),
        min(
            float(main["slot_precision_iou50"]),
            float(main["slot_recall_iou50"]),
            float(stress["slot_precision_iou50"]),
            float(stress["slot_recall_iou50"]),
        ),
        float(row["mean_matched_iou"]),
    )


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-train-index", type=Path, required=True)
    parser.add_argument("--main-dev-index", type=Path, required=True)
    parser.add_argument("--stress-train-index", type=Path, required=True)
    parser.add_argument("--stress-dev-index", type=Path, required=True)
    parser.add_argument("--main-train-scenes", type=Path, required=True)
    parser.add_argument("--main-dev-scenes", type=Path, required=True)
    parser.add_argument("--stress-train-scenes", type=Path, required=True)
    parser.add_argument("--stress-dev-scenes", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=base / "overlap_event_slots_v2_rehearsal_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=5202)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--main-sampling-ratio", type=float, default=0.75)
    parser.add_argument("--mask-threshold", type=float, default=0.6)
    parser.add_argument("--mask-weight", type=float, default=1.0)
    parser.add_argument(
        "--objectness-grid", type=_grid, default=_grid("0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    )
    parser.add_argument("--shard-cache-size", type=int, default=24)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-scenes-per-domain", type=int, default=0)
    parser.add_argument("--max-dev-scenes-per-domain", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.main_sampling_ratio < 1.0:
        raise ValueError("main-sampling-ratio must lie strictly in (0,1)")
    if not 0.0 <= args.mask_threshold <= 1.0 or not 0.0 <= args.mask_weight <= 1.0:
        raise ValueError("mask decoder values must lie in [0,1]")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    main_train = load_scene_list(args.main_train_scenes)
    main_dev = load_scene_list(args.main_dev_scenes)
    stress_train = load_scene_list(args.stress_train_scenes)
    stress_dev = load_scene_list(args.stress_dev_scenes)
    if args.max_train_scenes_per_domain > 0:
        main_train = main_train[: args.max_train_scenes_per_domain]
        stress_train = stress_train[: args.max_train_scenes_per_domain]
    if args.max_dev_scenes_per_domain > 0:
        main_dev = main_dev[: args.max_dev_scenes_per_domain]
        stress_dev = stress_dev[: args.max_dev_scenes_per_domain]

    index_paths = [
        args.main_train_index,
        args.main_dev_index,
        args.stress_train_index,
        args.stress_dev_index,
    ]
    store = DenseFeatureStore(index_paths, cache_size=args.shard_cache_size)
    identity_audit = assert_dense_identity_disjoint_v2(
        store,
        {"combined_train": main_train + stress_train, "combined_dev": main_dev + stress_dev},
    )
    main_train_dataset = OverlapSceneSlotDataset(store, main_train, preload=args.preload)
    stress_train_dataset = OverlapSceneSlotDataset(store, stress_train, preload=args.preload)
    main_dev_dataset = OverlapSceneSlotDataset(store, main_dev, preload=args.preload)
    stress_dev_dataset = OverlapSceneSlotDataset(store, stress_dev, preload=args.preload)
    combined_train = ConcatDataset([main_train_dataset, stress_train_dataset])

    main_weight = args.main_sampling_ratio / len(main_train_dataset)
    stress_weight = (1.0 - args.main_sampling_ratio) / len(stress_train_dataset)
    sample_weights = torch.tensor(
        [main_weight] * len(main_train_dataset) + [stress_weight] * len(stress_train_dataset),
        dtype=torch.double,
    )
    samples_per_epoch = round(len(main_train_dataset) / args.main_sampling_ratio)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=generator,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_overlap_slots,
    }
    train_loader = DataLoader(combined_train, sampler=sampler, **loader_options)
    main_dev_loader = DataLoader(main_dev_dataset, shuffle=False, **loader_options)
    stress_dev_loader = DataLoader(stress_dev_dataset, shuffle=False, **loader_options)

    init_path = args.init_checkpoint.resolve()
    init_payload = torch.load(init_path, map_location="cpu", weights_only=True)
    config = RelationalEventSlotsV1Config(**init_payload["config"])
    if int(config.feature_dim) != int(store.feature_dim or 0):
        raise ValueError("checkpoint and dense feature dimensions differ")
    model = OverlapEventSlotsV2(config)
    model.load_state_dict(init_payload["model_state_dict"], strict=True)
    device = _device(args.device)
    model.to(device)

    base_weights = RelationalEventSlotsV1LossWeights()
    overlap_weights = OverlapEventSlotsV2LossWeights()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_path = output_dir / "overlap_event_slots_v2_rehearsal_best.pt"
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_selection: dict[str, Any] | None = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Counter[str] = Counter()
        examples = 0
        domain_counts: Counter[str] = Counter()
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(batch["features"], batch["valid_mask"])
                losses = overlap_event_slots_v2_loss(
                    outputs,
                    batch["target_intervals"],
                    batch["target_event_masks"],
                    batch["frame_event_target"],
                    batch["frame_onset_target"],
                    batch["valid_mask"],
                    base_weights=base_weights,
                    overlap_weights=overlap_weights,
                )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            count = int(batch["features"].shape[0])
            examples += count
            for scene_id in raw_batch["scene_id"]:
                domain_counts["main" if str(scene_id).startswith("tiered_realistic_v5_") else "stress"] += 1
            for name, value in losses.items():
                totals[name] += float(value.detach()) * count
        scheduler.step()

        main_raw = collect_predictions(model, main_dev_loader, device)
        stress_raw = collect_predictions(model, stress_dev_loader, device)
        main_decoded = _decode(
            main_raw, mask_threshold=args.mask_threshold, mask_weight=args.mask_weight
        )
        stress_decoded = _decode(
            stress_raw, mask_threshold=args.mask_threshold, mask_weight=args.mask_weight
        )
        grid = [_joint_row(main_decoded, stress_decoded, value) for value in args.objectness_grid]
        selection = max(grid, key=_selection_key)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_examples": examples,
            "sampled_domains": dict(domain_counts),
            "train": {name: value / max(examples, 1) for name, value in totals.items()},
            "selection": selection,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        current_key = _selection_key(selection)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_selection = selection
            stale_epochs = 0
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT,
                    "epoch": epoch,
                    "config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "objectness_threshold": float(selection["objectness_threshold"]),
                    "mask_decoder": {
                        "component": "connected_component_containing_global_peak",
                        "mask_threshold": args.mask_threshold,
                        "mask_weight": args.mask_weight,
                    },
                    "selection_metrics": selection,
                    "labels": list(store.labels or []),
                    "dense_index_sha256": store.index_sha256,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    if best_selection is None:
        raise RuntimeError("training produced no checkpoint")
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "v5_main_v4_stress_rehearsal_with_fixed_mask_decoder",
        "answer_label_used_as_model_input": False,
        "historical_code_or_checkpoint_modified": False,
        "arguments": _jsonable(vars(args)),
        "config": asdict(config),
        "base_loss_weights": asdict(base_weights),
        "overlap_loss_weights": asdict(overlap_weights),
        "sampling": {
            "main_probability": args.main_sampling_ratio,
            "stress_probability": 1.0 - args.main_sampling_ratio,
            "samples_per_epoch": samples_per_epoch,
        },
        "selection_protocol": {
            "primary": "harmonic_mean_of_main_and_stress_f1_iou50",
            "single_shared_objectness_threshold": True,
            "mask_decoder_fixed_before_training": True,
            "paper_eligible": False,
            "reason": "development dev sets are used for checkpoint and threshold selection",
        },
        "initialization": {"checkpoint": str(init_path), "sha256": _sha256_file(init_path)},
        "splits": {
            "main_train": len(main_train),
            "stress_train": len(stress_train),
            "main_dev": len(main_dev),
            "stress_dev": len(stress_dev),
        },
        "identity_audit": identity_audit,
        "best_epoch": best_epoch,
        "best_selection": best_selection,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "history": history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "best_selection": best_selection}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
