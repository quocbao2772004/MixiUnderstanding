#!/usr/bin/env python3
"""Train overlap-aware class-agnostic slots on Gold Natural v3.

This sidecar leaves all historical Claude/QCES trainers and checkpoints
untouched.  Checkpoint selection uses a source-disjoint half of dev; decoder
calibration uses the other half.  Answer labels are never model inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from mixi_understanding.qces.overlap_event_slots_v2 import (
    OverlapEventSlotsV2,
    OverlapEventSlotsV2LossWeights,
    overlap_event_slots_v2_loss,
)
from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1Config,
    RelationalEventSlotsV1LossWeights,
    interval_iou_matrix,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _atomic_torch,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import (
    assert_dense_identity_disjoint_v2,
)
from mixi_understanding.scripts.train_qces_relational_event_slots_v1 import (
    FIXED_AUDIO_SECONDS,
    FRAME_HOP_SECONDS,
    NUM_FRAMES,
    SceneSlotDataset,
    collate_scene_slots,
    split_source_disjoint_scenes,
)


FORMAT = "qces_overlap_event_slots_training_receipt_v2"
CHECKPOINT_FORMAT = "qces_overlap_event_slots_checkpoint_v2"


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


class OverlapSceneSlotDataset(SceneSlotDataset):
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = super().__getitem__(index)
        scene_id = str(row["scene_id"])
        events = self.gold[scene_id]["events"]
        valid_frames = int(self.gold[scene_id]["valid_frames"])
        masks = torch.zeros(len(events), NUM_FRAMES, dtype=torch.float32)
        for event_index, event in enumerate(events):
            start = max(
                0,
                min(
                    valid_frames - 1,
                    int(math.floor(float(event["onset_seconds"]) / FRAME_HOP_SECONDS + 1e-8)),
                ),
            )
            end = max(
                start + 1,
                min(
                    valid_frames,
                    int(math.ceil(float(event["offset_seconds"]) / FRAME_HOP_SECONDS - 1e-8)),
                ),
            )
            masks[event_index, start:end] = 1.0
        row["target_event_masks"] = masks
        return row


def collate_overlap_slots(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch = collate_scene_slots(rows)
    batch["target_event_masks"] = [row["target_event_masks"] for row in rows]
    return batch


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key in (
        "features",
        "detector_logits",
        "valid_mask",
        "frame_event_target",
        "frame_onset_target",
    ):
        result[key] = result[key].to(device, non_blocking=True)
    result["target_intervals"] = [value.to(device, non_blocking=True) for value in batch["target_intervals"]]
    result["target_event_masks"] = [
        value.to(device, non_blocking=True) for value in batch["target_event_masks"]
    ]
    return result


@torch.inference_mode()
def collect_predictions(
    model: OverlapEventSlotsV2,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    model.eval()
    collected: dict[str, dict[str, Any]] = {}
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        outputs = model(batch["features"], batch["valid_mask"])
        confidence = outputs["objectness_logits"].sigmoid().cpu()
        intervals = outputs["intervals"].cpu()
        mask_probability = outputs["slot_mask_logits"].sigmoid().cpu()
        for index, scene_id in enumerate(raw_batch["scene_id"]):
            collected[str(scene_id)] = {
                "confidence": confidence[index],
                "intervals": intervals[index],
                "mask_probability": mask_probability[index],
                "gold_intervals": raw_batch["target_intervals"][index].float(),
                "gold_event_masks": raw_batch["target_event_masks"][index].float(),
            }
    return collected


def evaluate_objectness_threshold(
    predictions: Mapping[str, Mapping[str, Any]], *, objectness_threshold: float
) -> dict[str, float | int]:
    if not 0.0 <= objectness_threshold <= 1.0:
        raise ValueError("objectness_threshold must be in [0,1]")
    predicted_count = gold_count = true_positive_30 = true_positive_50 = 0
    exact_count = 0
    matched_iou_sum = matched_count = 0
    count_absolute_error = 0
    mask_iou_sum = mask_iou_count = 0
    for record in predictions.values():
        keep = record["confidence"] >= objectness_threshold
        predicted = record["intervals"][keep]
        predicted_masks = record["mask_probability"][keep]
        gold = record["gold_intervals"]
        gold_masks = record["gold_event_masks"]
        predicted_count += int(predicted.shape[0])
        gold_count += int(gold.shape[0])
        exact_count += int(predicted.shape[0] == gold.shape[0])
        count_absolute_error += abs(int(predicted.shape[0]) - int(gold.shape[0]))
        if predicted.shape[0] == 0 or gold.shape[0] == 0:
            continue
        iou = interval_iou_matrix(predicted.float(), gold.float())
        rows, columns = linear_sum_assignment((1.0 - iou).numpy())
        values = iou[rows, columns]
        true_positive_30 += int((values >= 0.30).sum())
        true_positive_50 += int((values >= 0.50).sum())
        matched_iou_sum += float(values.sum())
        matched_count += int(values.numel())
        binary_masks = predicted_masks[rows] >= 0.5
        target_masks = gold_masks[columns] >= 0.5
        intersection = (binary_masks & target_masks).sum(dim=1).float()
        union = (binary_masks | target_masks).sum(dim=1).float().clamp_min(1)
        mask_iou_sum += float((intersection / union).sum())
        mask_iou_count += len(rows)

    def metrics_at(true_positive: int) -> tuple[float, float, float]:
        precision = true_positive / max(predicted_count, 1)
        recall = true_positive / max(gold_count, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        return precision, recall, f1

    precision_30, recall_30, f1_30 = metrics_at(true_positive_30)
    precision_50, recall_50, f1_50 = metrics_at(true_positive_50)
    return {
        "decoder": "query_specific_intervals_no_temporal_nms",
        "objectness_threshold": objectness_threshold,
        "num_scenes": len(predictions),
        "predicted_events": predicted_count,
        "gold_events": gold_count,
        "slot_precision_iou30": precision_30,
        "slot_recall_iou30": recall_30,
        "slot_f1_iou30": f1_30,
        "slot_precision_iou50": precision_50,
        "slot_recall_iou50": recall_50,
        "slot_f1_iou50": f1_50,
        "slot_exact_count_accuracy": exact_count / max(len(predictions), 1),
        "slot_count_mae": count_absolute_error / max(len(predictions), 1),
        "slot_mean_matched_iou": matched_iou_sum / max(matched_count, 1),
        "matched_slot_mask_iou50_threshold": mask_iou_sum / max(mask_iou_count, 1),
    }


def _threshold_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    precision = float(metrics["slot_precision_iou50"])
    recall = float(metrics["slot_recall_iou50"])
    return (
        min(precision, recall),
        float(metrics["slot_f1_iou50"]),
        float(metrics["slot_exact_count_accuracy"]),
        float(metrics["slot_mean_matched_iou"]),
        -abs(float(metrics["objectness_threshold"]) - 0.5),
    )


def _checkpoint_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["slot_f1_iou50"]),
        min(
            float(metrics["slot_precision_iou50"]),
            float(metrics["slot_recall_iou50"]),
        ),
        float(metrics["slot_mean_matched_iou"]),
        float(metrics["slot_exact_count_accuracy"]),
    )


def _parse_threshold_grid(value: str) -> list[float]:
    values = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if not values or any(not 0.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("thresholds must be in [0,1]")
    return values


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_gold_natural_v3"
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-index", type=Path, default=base / "dense_overlap_gold_natural_v3_train/index.json"
    )
    parser.add_argument(
        "--dev-index", type=Path, default=base / "dense_overlap_gold_natural_v3_dev/index.json"
    )
    parser.add_argument("--train-scenes", type=Path, default=data / "scene_ids_overlap_train.txt")
    parser.add_argument("--dev-scenes", type=Path, default=data / "scene_ids_overlap_dev.txt")
    parser.add_argument("--output-dir", type=Path, default=base / "overlap_event_slots_v2")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=5201)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=2)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument(
        "--dev-protocol",
        choices=("full_dev_diagnostic", "source_disjoint_calibration_selection"),
        default="full_dev_diagnostic",
        help=(
            "Gold Natural v3 dev is nearly one source-connected component. "
            "Use full_dev_diagnostic for model development; a later locked "
            "source-disjoint test is required for paper metrics."
        ),
    )
    parser.add_argument(
        "--objectness-grid",
        type=_parse_threshold_grid,
        default=_parse_threshold_grid("0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90"),
    )
    parser.add_argument("--shard-cache-size", type=int, default=20)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output directory is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(args.seed)

    train_ids = load_scene_list(args.train_scenes)
    dev_ids = load_scene_list(args.dev_scenes)
    if args.max_train_scenes > 0:
        train_ids = train_ids[: args.max_train_scenes]
    if args.max_dev_scenes > 0:
        dev_ids = dev_ids[: args.max_dev_scenes]
    store = DenseFeatureStore([args.train_index, args.dev_index], cache_size=args.shard_cache_size)
    if args.dev_protocol == "source_disjoint_calibration_selection":
        calibration_ids, selection_ids = split_source_disjoint_scenes(
            store, dev_ids, seed=args.seed, calibration_fraction=args.calibration_fraction
        )
        if min(len(calibration_ids), len(selection_ids)) < max(10, int(0.10 * len(dev_ids))):
            raise ValueError(
                "source-connected dev split is degenerate; use "
                "--dev-protocol full_dev_diagnostic and reserve a new locked test"
            )
        identity_audit = assert_dense_identity_disjoint_v2(
            store,
            {"train": train_ids, "calibration": calibration_ids, "selection": selection_ids},
        )
    else:
        calibration_ids = selection_ids = dev_ids
        identity_audit = assert_dense_identity_disjoint_v2(
            store, {"train": train_ids, "dev": dev_ids}
        )

    train_dataset = OverlapSceneSlotDataset(store, train_ids, preload=args.preload)
    calibration_dataset = OverlapSceneSlotDataset(store, calibration_ids, preload=args.preload)
    selection_dataset = (
        calibration_dataset
        if args.dev_protocol == "full_dev_diagnostic"
        else OverlapSceneSlotDataset(store, selection_ids, preload=args.preload)
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_overlap_slots,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    calibration_loader = DataLoader(calibration_dataset, shuffle=False, **loader_options)
    selection_loader = DataLoader(selection_dataset, shuffle=False, **loader_options)

    config = RelationalEventSlotsV1Config(
        feature_dim=int(store.feature_dim or 0),
        hidden_dim=args.hidden_dim,
        num_slots=args.num_slots,
        num_heads=args.num_heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward_dim=args.hidden_dim * 3,
        dropout=args.dropout,
    )
    device = _device(args.device)
    model = OverlapEventSlotsV2(config).to(device)
    initialization: dict[str, Any] = {"checkpoint": None}
    if args.init_checkpoint is not None:
        init_path = args.init_checkpoint.resolve()
        payload = torch.load(init_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(payload["model_state_dict"], strict=False)
        allowed_missing = {
            "slot_mask_query.0.weight",
            "slot_mask_query.0.bias",
            "slot_mask_query.1.weight",
            "slot_mask_query.1.bias",
            "slot_mask_memory.0.weight",
            "slot_mask_memory.0.bias",
            "slot_mask_memory.1.weight",
            "slot_mask_memory.1.bias",
        }
        if set(missing) not in (set(), allowed_missing) or unexpected:
            raise ValueError(f"incompatible init checkpoint: missing={missing}, unexpected={unexpected}")
        initialization = {
            "checkpoint": str(init_path),
            "sha256": _sha256_file(init_path),
            "missing_new_mask_parameters": sorted(missing),
        }

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
    checkpoint_path = output_dir / "overlap_event_slots_v2_best.pt"
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_selection: dict[str, Any] | None = None
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Counter[str] = Counter()
        examples = 0
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
            for name, value in losses.items():
                totals[name] += float(value.detach()) * count
        scheduler.step()

        calibration_predictions = collect_predictions(model, calibration_loader, device)
        calibration_grid = [
            evaluate_objectness_threshold(calibration_predictions, objectness_threshold=value)
            for value in args.objectness_grid
        ]
        calibration_metrics = max(calibration_grid, key=_threshold_key)
        selected_threshold = float(calibration_metrics["objectness_threshold"])
        if args.dev_protocol == "full_dev_diagnostic":
            selection_metrics = dict(calibration_metrics)
        else:
            selection_predictions = collect_predictions(model, selection_loader, device)
            selection_metrics = evaluate_objectness_threshold(
                selection_predictions, objectness_threshold=selected_threshold
            )
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": {name: value / max(examples, 1) for name, value in totals.items()},
            "calibration": calibration_metrics,
            "selection": selection_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        current_key = _checkpoint_key(selection_metrics)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_selection = dict(selection_metrics)
            stale_epochs = 0
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT,
                    "epoch": epoch,
                    "config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "objectness_threshold": selected_threshold,
                    "calibration_metrics": calibration_metrics,
                    "selection_metrics": selection_metrics,
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
    success_gates = {
        "slot_precision_iou50_ge_0_85": float(best_selection["slot_precision_iou50"]) >= 0.85,
        "slot_recall_iou50_ge_0_90": float(best_selection["slot_recall_iou50"]) >= 0.90,
        "slot_f1_iou50_ge_0_87": float(best_selection["slot_f1_iou50"]) >= 0.87,
        "slot_mean_matched_iou_ge_0_85": float(best_selection["slot_mean_matched_iou"]) >= 0.85,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "overlap_aware_class_agnostic_query_slots_with_per_slot_masks",
        "answer_label_used_as_model_input": False,
        "temporal_nms_used": False,
        "config": asdict(config),
        "base_loss_weights": asdict(base_weights),
        "overlap_loss_weights": asdict(overlap_weights),
        "arguments": _jsonable(vars(args)),
        "evaluation_protocol": {
            "name": args.dev_protocol,
            "threshold_and_checkpoint_selected_on_same_dev": (
                args.dev_protocol == "full_dev_diagnostic"
            ),
            "paper_eligible": False,
            "required_next": "freeze checkpoint and decoder, then evaluate on a new source-disjoint locked test",
        },
        "initialization": initialization,
        "splits": {
            "train_scenes": len(train_ids),
            "calibration_scenes": len(calibration_ids),
            "selection_scenes": len(selection_ids),
        },
        "identity_audit": identity_audit,
        "best_epoch": best_epoch,
        "best_selection": best_selection,
        "success_gates": success_gates,
        "all_success_gates_pass": all(success_gates.values()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "history": history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(
        json.dumps(
            {
                "complete": True,
                "best_epoch": best_epoch,
                "best_selection": best_selection,
                "success_gates": success_gates,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
