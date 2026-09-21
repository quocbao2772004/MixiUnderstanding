#!/usr/bin/env python3
"""Fine-tune only the DETR query branch with polyphonic supervision.

The frozen v1 encoder, input projection, frame-event head, and onset head are
kept byte-identical.  Training exposure is 50% historical non-overlap, 25%
pair-overlap, and 25% triple-overlap.  Checkpoint selection macro-averages
non-overlap and overlap proposal recall so overlap cannot improve by silently
destroying sequential performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from torch.utils.data import ConcatDataset, DataLoader

from mixi_understanding.qces.relational_event_slots_v1 import (
    RelationalEventSlotsV1,
    RelationalEventSlotsV1Config,
    RelationalEventSlotsV1LossWeights,
    relational_event_slots_v1_loss,
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
from mixi_understanding.scripts.train_qces_relational_event_slots_v1 import (
    CHECKPOINT_FORMAT,
    SceneSlotDataset,
    _one_to_one_ious,
    _to_device,
    collate_scene_slots,
    collect_predictions,
)


FORMAT = "qces_polyphonic_query_branch_training_receipt_v1"
POLYPHONIC_CHECKPOINT_FORMAT = "qces_polyphonic_query_branch_checkpoint_v1"
TRAINABLE_PREFIXES = (
    "decoder.",
    "slot_queries.",
    "objectness_head.",
    "box_head.",
)


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/multievent"
    overlap = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--initial-checkpoint", type=Path,
        default=base / "relational_event_slots_v1/relational_event_slots_v1_best.pt",
    )
    parser.add_argument("--nonoverlap-train-index", type=Path, default=base / "dense_multi_train_v2/index.json")
    parser.add_argument("--nonoverlap-dev-index", type=Path, default=base / "dense_multi_dev_v2/index.json")
    parser.add_argument("--overlap-train-index", type=Path, default=base / "dense_overlap_query_train_v1/index.json")
    parser.add_argument("--overlap-dev-index", type=Path, default=base / "dense_overlap_query_dev_v1/index.json")
    parser.add_argument("--nonoverlap-train-scenes", type=Path, default=data / "scene_ids_train.txt")
    parser.add_argument("--nonoverlap-dev-scenes", type=Path, default=data / "scene_ids_dev.txt")
    parser.add_argument("--overlap-train-scenes", type=Path, default=overlap / "scene_ids_overlap_train.txt")
    parser.add_argument("--overlap-dev-scenes", type=Path, default=overlap / "scene_ids_overlap_dev.txt")
    parser.add_argument(
        "--output-dir", type=Path,
        default=base / "polyphonic_query_branch_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2097)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _seed(seed: int) -> None:
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


def _state_hash(state: Mapping[str, torch.Tensor], names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _configure_trainable(model: RelationalEventSlotsV1) -> tuple[list[str], list[str]]:
    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(TRAINABLE_PREFIXES):
            parameter.requires_grad_(True)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if not trainable or any(not name.startswith(TRAINABLE_PREFIXES) for name in trainable):
        raise RuntimeError(f"unexpected trainable parameter set: {trainable[:10]}")
    return trainable, frozen


def _set_query_train_mode(model: RelationalEventSlotsV1) -> None:
    model.eval()
    model.decoder.train()
    model.slot_queries.train()
    model.objectness_head.train()
    model.box_head.train()


def _proposal_metrics(predictions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    objectness_positive_sum = 0.0
    for record in predictions.values():
        gold = list(record["gold_intervals"])
        ranked = sorted(record["slots"], key=lambda row: float(row["score"]), reverse=True)
        all_intervals = [[float(row["start"]), float(row["end"])] for row in ranked]
        count_intervals = all_intervals[: len(gold)]
        all_iou = _one_to_one_ious(all_intervals, gold)
        count_iou = _one_to_one_ious(count_intervals, gold)
        all30 = sum(value >= 0.30 for value in all_iou)
        all50 = sum(value >= 0.50 for value in all_iou)
        count30 = sum(value >= 0.30 for value in count_iou)
        count50 = sum(value >= 0.50 for value in count_iou)
        totals["scenes"] += 1
        totals["gold"] += len(gold)
        totals["all30"] += all30
        totals["all50"] += all50
        totals["count30"] += count30
        totals["count50"] += count50
        totals["all_scene30"] += int(all30 == len(gold))
        totals["all_scene50"] += int(all50 == len(gold))
        totals["count_scene30"] += int(count30 == len(gold))
        totals["count_scene50"] += int(count50 == len(gold))
        objectness_positive_sum += sum(float(row["score"]) for row in ranked[: len(gold)])
    gold_count = max(int(totals["gold"]), 1)
    scene_count = max(int(totals["scenes"]), 1)
    return {
        "scenes": int(totals["scenes"]),
        "gold_events": int(totals["gold"]),
        "all8_event_recall_iou30_↑": totals["all30"] / gold_count,
        "all8_event_recall_iou50_↑": totals["all50"] / gold_count,
        "top_gold_count_event_recall_iou30_↑": totals["count30"] / gold_count,
        "top_gold_count_event_recall_iou50_↑": totals["count50"] / gold_count,
        "all8_all_events_scene_accuracy_iou30_↑": totals["all_scene30"] / scene_count,
        "all8_all_events_scene_accuracy_iou50_↑": totals["all_scene50"] / scene_count,
        "top_gold_count_all_events_scene_accuracy_iou30_↑": totals["count_scene30"] / scene_count,
        "top_gold_count_all_events_scene_accuracy_iou50_↑": totals["count_scene50"] / scene_count,
        "mean_top_gold_count_objectness": objectness_positive_sum / gold_count,
    }


def _selection_key(nonoverlap: Mapping[str, Any], overlap: Mapping[str, Any]) -> tuple[float, ...]:
    macro_top50 = 0.5 * (
        float(nonoverlap["top_gold_count_event_recall_iou50_↑"])
        + float(overlap["top_gold_count_event_recall_iou50_↑"])
    )
    macro_all50 = 0.5 * (
        float(nonoverlap["all8_event_recall_iou50_↑"])
        + float(overlap["all8_event_recall_iou50_↑"])
    )
    overlap_scene50 = float(overlap["all8_all_events_scene_accuracy_iou50_↑"])
    nonoverlap_scene50 = float(nonoverlap["all8_all_events_scene_accuracy_iou50_↑"])
    return macro_top50, macro_all50, overlap_scene50, nonoverlap_scene50


def main() -> None:
    args = parse_args()
    for name in ("epochs", "patience", "batch_size"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed(args.seed)

    initial_path = args.initial_checkpoint.resolve()
    initial_sha = _sha256_file(initial_path)
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if initial.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("initial checkpoint is not relational-event-slots v1")
    indexes = [
        args.nonoverlap_train_index.resolve(), args.nonoverlap_dev_index.resolve(),
        args.overlap_train_index.resolve(), args.overlap_dev_index.resolve(),
    ]
    store = DenseFeatureStore(indexes, cache_size=32)
    if list(initial.get("labels") or []) != list(store.labels or []):
        raise ValueError("checkpoint and dense indexes disagree on label order")
    scene_ids = {
        "nonoverlap_train": load_scene_list(args.nonoverlap_train_scenes),
        "nonoverlap_dev": load_scene_list(args.nonoverlap_dev_scenes),
        "overlap_train": load_scene_list(args.overlap_train_scenes),
        "overlap_dev": load_scene_list(args.overlap_dev_scenes),
    }
    identity = assert_dense_identity_disjoint_v2(
        store,
        {
            "train": scene_ids["nonoverlap_train"] + scene_ids["overlap_train"],
            "dev": scene_ids["nonoverlap_dev"] + scene_ids["overlap_dev"],
        },
    )
    datasets = {
        name: SceneSlotDataset(store, ids, preload=args.preload)
        for name, ids in scene_ids.items()
    }
    if len(datasets["nonoverlap_train"]) != len(datasets["overlap_train"]):
        raise RuntimeError("50/25/25 exposure requires equal non-overlap and total overlap scenes")
    train_dataset = ConcatDataset((datasets["nonoverlap_train"], datasets["overlap_train"]))
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_scene_slots,
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_args)
    nonoverlap_dev_loader = DataLoader(datasets["nonoverlap_dev"], shuffle=False, **loader_args)
    overlap_dev_loader = DataLoader(datasets["overlap_dev"], shuffle=False, **loader_args)

    config = RelationalEventSlotsV1Config(**initial["config"])
    device = _device(args.device)
    model = RelationalEventSlotsV1(config).to(device)
    model.load_state_dict(initial["model_state_dict"], strict=True)
    trainable_names, frozen_names = _configure_trainable(model)
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_frozen_hash = _state_hash(initial_state, frozen_names)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    weights = RelationalEventSlotsV1LossWeights(
        objectness=1.0, box_l1=5.0, box_iou=2.0,
        frame_bce=0.0, frame_dice=0.0, onset_bce=0.0,
    )
    checkpoint_path = output_dir / "polyphonic_query_branch_best.pt"

    def save_checkpoint(epoch: int, selection_metrics: Mapping[str, Any]) -> None:
        current_state = model.state_dict()
        current_frozen_hash = _state_hash(current_state, frozen_names)
        if current_frozen_hash != initial_frozen_hash:
            raise RuntimeError("a frozen parameter changed during query tuning")
        _atomic_torch(
            {
                "format": POLYPHONIC_CHECKPOINT_FORMAT,
                "epoch": epoch,
                "config": asdict(config),
                "model_state_dict": current_state,
                "labels": list(store.labels or []),
                "initial_checkpoint": str(initial_path),
                "initial_checkpoint_sha256": initial_sha,
                "trainable_parameter_names": trainable_names,
                "frozen_parameter_names": frozen_names,
                "initial_frozen_parameter_sha256": initial_frozen_hash,
                "current_frozen_parameter_sha256": current_frozen_hash,
                "historical_hysteresis_decoder": {
                    "boundary_low_threshold": float(initial["boundary_low_threshold"]),
                    "presence_high_threshold": float(initial["presence_high_threshold"]),
                },
                "selection_metrics": dict(selection_metrics),
            },
            checkpoint_path,
        )

    # Epoch zero is an eligible fallback.  Query tuning must beat the frozen
    # checkpoint under the same macro validation policy, not merely finish.
    baseline_nonoverlap = _proposal_metrics(
        collect_predictions(model, nonoverlap_dev_loader, device)
    )
    baseline_overlap = _proposal_metrics(
        collect_predictions(model, overlap_dev_loader, device)
    )
    baseline_key = _selection_key(baseline_nonoverlap, baseline_overlap)
    baseline = {
        "epoch": 0,
        "learning_rate": 0.0,
        "train": None,
        "nonoverlap_dev": baseline_nonoverlap,
        "overlap_dev": baseline_overlap,
        "selection_key": list(baseline_key),
    }
    print(json.dumps({"baseline": baseline}, ensure_ascii=False, sort_keys=True), flush=True)
    save_checkpoint(0, baseline)
    history = []
    best_key: tuple[float, ...] = baseline_key
    best_epoch = 0
    best_metrics: dict[str, Any] = baseline
    stale = 0

    for epoch in range(1, args.epochs + 1):
        _set_query_train_mode(model)
        totals: Counter[str] = Counter()
        examples = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(batch["features"], batch["valid_mask"])
                losses = relational_event_slots_v1_loss(
                    outputs,
                    batch["target_intervals"],
                    batch["frame_event_target"],
                    batch["frame_onset_target"],
                    batch["valid_mask"],
                    weights=weights,
                    no_object_weight=0.2,
                )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            count = int(batch["features"].shape[0])
            examples += count
            for name, value in losses.items():
                totals[name] += float(value.detach()) * count
        scheduler.step()
        train_metrics = {name: value / max(examples, 1) for name, value in totals.items()}
        nonoverlap_predictions = collect_predictions(model, nonoverlap_dev_loader, device)
        overlap_predictions = collect_predictions(model, overlap_dev_loader, device)
        nonoverlap_metrics = _proposal_metrics(nonoverlap_predictions)
        overlap_metrics = _proposal_metrics(overlap_predictions)
        key = _selection_key(nonoverlap_metrics, overlap_metrics)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "nonoverlap_dev": nonoverlap_metrics,
            "overlap_dev": overlap_metrics,
            "selection_key": list(key),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = row
            stale = 0
            save_checkpoint(epoch, row)
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    if best_metrics is None:
        raise RuntimeError("training produced no checkpoint")
    checkpoint_sha = _sha256_file(checkpoint_path)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "frozen_encoder_and_hysteresis_heads_plus_overlap_supervised_detr_query_branch",
        "answer_label_used": False,
        "arguments": _jsonable(vars(args)),
        "initial_checkpoint": str(initial_path),
        "initial_checkpoint_sha256": initial_sha,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "frozen_parameter_sha256": initial_frozen_hash,
        "exposure": {
            "nonoverlap": len(datasets["nonoverlap_train"]) / len(train_dataset),
            "pair_overlap": 0.5 * len(datasets["overlap_train"]) / len(train_dataset),
            "triple_overlap": 0.5 * len(datasets["overlap_train"]) / len(train_dataset),
        },
        "identity_audit": identity,
        "loss_weights": asdict(weights),
        "selection_policy": "lexicographic macro top-gold-count recall50, macro all8 recall50, overlap scene50, nonoverlap scene50",
        "baseline_metrics": baseline,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "best_metrics": best_metrics, "checkpoint_sha256": checkpoint_sha}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
