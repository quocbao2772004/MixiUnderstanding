#!/usr/bin/env python3
"""Train class-aware RED/EPN event proposals on frozen QCES BEATs features.

This is a new sidecar experiment.  It does not modify the historical detector,
class-agnostic temporal slots, pair ranker, or separator checkpoints.
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

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from mixi_understanding.qces.benchmark_integrity import semantic_events
from mixi_understanding.qces.class_aware_red_epn_v1 import (
    ClassAwareRedEpnV1,
    ClassAwareRedEpnV1Config,
    ClassAwareRedEpnV1LossWeights,
    class_aware_red_epn_v1_loss,
    decode_class_aware_proposals,
    interval_iou,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    NUM_FRAMES,
    _atomic_json,
    _atomic_torch,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import (
    assert_dense_identity_disjoint_v2,
)


FORMAT = "qces_class_aware_red_epn_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_class_aware_red_epn_checkpoint_v1"
FRAME_HOP_SECONDS = 0.04
FIXED_AUDIO_SECONDS = 10.0


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/multievent"
    overlap = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nonoverlap-train-index", type=Path, default=base / "dense_multi_train_v2/index.json"
    )
    parser.add_argument(
        "--nonoverlap-dev-index", type=Path, default=base / "dense_multi_dev_v2/index.json"
    )
    parser.add_argument(
        "--overlap-train-index", type=Path, default=base / "dense_overlap_query_train_v2/index.json"
    )
    parser.add_argument(
        "--overlap-dev-index", type=Path, default=base / "dense_overlap_query_dev_v2/index.json"
    )
    parser.add_argument(
        "--nonoverlap-train-scenes", type=Path, default=data / "scene_ids_train.txt"
    )
    parser.add_argument(
        "--nonoverlap-dev-scenes", type=Path, default=data / "scene_ids_dev.txt"
    )
    parser.add_argument(
        "--overlap-train-scenes", type=Path, default=overlap / "scene_ids_overlap_train.txt"
    )
    parser.add_argument(
        "--overlap-dev-scenes", type=Path, default=overlap / "scene_ids_overlap_dev.txt"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=base / "class_aware_red_epn_v1"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2171)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--context-dim", type=int, default=256)
    parser.add_argument("--epn-hidden-dim", type=int, default=128)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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


def _target_tensors(
    events: Sequence[Mapping[str, Any]],
    label_to_id: Mapping[str, int],
    valid_frames: int,
    num_classes: int,
) -> dict[str, torch.Tensor | list[dict[str, float | int]]]:
    frame_presence = torch.zeros(NUM_FRAMES, num_classes, dtype=torch.float32)
    frame_onset = torch.zeros_like(frame_presence)
    frame_offset = torch.zeros_like(frame_presence)
    duration_since = torch.zeros_like(frame_presence)
    duration_to = torch.zeros_like(frame_presence)
    clip_presence = torch.zeros(num_classes, dtype=torch.float32)
    frame_center = (
        (torch.arange(NUM_FRAMES, dtype=torch.float32) + 0.5) / float(NUM_FRAMES)
    )[:, None].expand(-1, num_classes).clone()
    gold_events: list[dict[str, float | int]] = []
    for event in events:
        label = str(event["label"])
        if label not in label_to_id:
            raise ValueError(f"event label absent from dense ontology: {label}")
        label_id = int(label_to_id[label])
        start_seconds = float(event["onset_seconds"])
        end_seconds = float(event["offset_seconds"])
        if not 0.0 <= start_seconds < end_seconds <= FIXED_AUDIO_SECONDS + 1e-6:
            raise ValueError(f"invalid event interval: {event}")
        start = start_seconds / FIXED_AUDIO_SECONDS
        end = min(end_seconds / FIXED_AUDIO_SECONDS, 1.0)
        start_frame = max(
            0, min(valid_frames - 1, int(math.floor(start_seconds / FRAME_HOP_SECONDS)))
        )
        end_frame = max(
            start_frame + 1,
            min(valid_frames, int(math.ceil(end_seconds / FRAME_HOP_SECONDS))),
        )
        frame_presence[start_frame:end_frame, label_id] = 1.0
        frame_onset[start_frame, label_id] = 1.0
        if start_frame > 0:
            frame_onset[start_frame - 1, label_id] = max(
                float(frame_onset[start_frame - 1, label_id]), 0.35
            )
        if start_frame + 1 < valid_frames:
            frame_onset[start_frame + 1, label_id] = max(
                float(frame_onset[start_frame + 1, label_id]), 0.35
            )
        offset_frame = max(start_frame, min(valid_frames - 1, end_frame - 1))
        frame_offset[offset_frame, label_id] = 1.0
        if offset_frame > 0:
            frame_offset[offset_frame - 1, label_id] = max(
                float(frame_offset[offset_frame - 1, label_id]), 0.35
            )
        if offset_frame + 1 < valid_frames:
            frame_offset[offset_frame + 1, label_id] = max(
                float(frame_offset[offset_frame + 1, label_id]), 0.35
            )
        centers = frame_center[start_frame:end_frame, label_id]
        # Same-label simultaneous instances are intrinsically ambiguous in a
        # mono mixture.  Keep the shorter duration target where they collide.
        new_since = (centers - start).clamp_min(0.0)
        new_to = (end - centers).clamp_min(0.0)
        old_duration = (
            duration_since[start_frame:end_frame, label_id]
            + duration_to[start_frame:end_frame, label_id]
        )
        new_duration = new_since + new_to
        replace = (old_duration <= 0) | (new_duration < old_duration)
        duration_since[start_frame:end_frame, label_id] = torch.where(
            replace, new_since, duration_since[start_frame:end_frame, label_id]
        )
        duration_to[start_frame:end_frame, label_id] = torch.where(
            replace, new_to, duration_to[start_frame:end_frame, label_id]
        )
        clip_presence[label_id] = 1.0
        gold_events.append({"label_id": label_id, "start": start, "end": end})
    return {
        "frame_presence": frame_presence,
        "frame_onset": frame_onset,
        "frame_offset": frame_offset,
        "duration_since_onset": duration_since,
        "duration_to_offset": duration_to,
        "frame_center": frame_center,
        "clip_presence": clip_presence,
        "gold_events": gold_events,
    }


class ClassAwareSceneDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        store: DenseFeatureStore,
        scene_ids: Sequence[str],
        *,
        preload: bool,
        max_scenes: int = 0,
    ) -> None:
        selected = list(dict.fromkeys(str(value) for value in scene_ids))
        if max_scenes > 0:
            selected = selected[:max_scenes]
        if not selected:
            raise ValueError("class-aware dataset cannot be empty")
        missing = sorted(set(selected) - store.scene_ids)
        if missing:
            raise ValueError(f"dense store misses scenes: {missing[:5]}")
        self.store = store
        self.scene_ids = selected
        self.preloaded = store.preload(selected) if preload else None
        labels = list(store.labels or [])
        self.label_to_id = {label: index for index, label in enumerate(labels)}
        self.targets: dict[str, dict[str, Any]] = {}
        for scene_id in selected:
            metadata = store.metadata(scene_id)
            valid_frames = int(metadata["valid_frames"])
            events = semantic_events(metadata)
            if not events:
                raise ValueError(f"{scene_id}: no semantic events")
            self.targets[scene_id] = _target_tensors(
                events, self.label_to_id, valid_frames, len(labels)
            ) | {"valid_frames": valid_frames}

    def __len__(self) -> int:
        return len(self.scene_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        scene_id = self.scene_ids[index]
        dense = self.preloaded[scene_id] if self.preloaded is not None else self.store.get(scene_id)
        target = self.targets[scene_id]
        return {
            "scene_id": scene_id,
            "features": dense["features"].float(),
            "detector_logits": dense["logits"].float(),
            "valid_mask": torch.arange(NUM_FRAMES) < int(target["valid_frames"]),
            "frame_presence": target["frame_presence"],
            "frame_onset": target["frame_onset"],
            "frame_offset": target["frame_offset"],
            "duration_since_onset": target["duration_since_onset"],
            "duration_to_offset": target["duration_to_offset"],
            "frame_center": target["frame_center"],
            "clip_presence": target["clip_presence"],
            "gold_events": target["gold_events"],
        }


def collate_class_aware(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "features",
        "detector_logits",
        "valid_mask",
        "frame_presence",
        "frame_onset",
        "frame_offset",
        "duration_since_onset",
        "duration_to_offset",
        "frame_center",
        "clip_presence",
    )
    result = {key: torch.stack([row[key] for row in rows]) for key in tensor_keys}
    result["scene_id"] = [str(row["scene_id"]) for row in rows]
    result["gold_events"] = [list(row["gold_events"]) for row in rows]
    return result


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    result = dict(batch)
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device, non_blocking=True)
    return result


def _matched_count(
    proposals: Sequence[Mapping[str, Any]],
    gold_events: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
    threshold: float,
) -> tuple[int, list[int]]:
    predicted = list(proposals[:top_k])
    hit_gold: list[int] = []
    for label_id in sorted({int(row["label_id"]) for row in gold_events}):
        prediction_index = [
            index for index, row in enumerate(predicted) if int(row["label_id"]) == label_id
        ]
        gold_index = [
            index for index, row in enumerate(gold_events) if int(row["label_id"]) == label_id
        ]
        if not prediction_index or not gold_index:
            continue
        matrix = np.asarray(
            [
                [
                    interval_iou(
                        [predicted[p]["start"], predicted[p]["end"]],
                        [gold_events[g]["start"], gold_events[g]["end"]],
                    )
                    for g in gold_index
                ]
                for p in prediction_index
            ],
            dtype=np.float32,
        )
        rows, columns = linear_sum_assignment(1.0 - matrix)
        for row, column in zip(rows, columns):
            if float(matrix[row, column]) >= threshold:
                hit_gold.append(gold_index[int(column)])
    unique = sorted(set(hit_gold))
    return len(unique), unique


@torch.inference_mode()
def evaluate(
    model: ClassAwareRedEpnV1,
    loader: DataLoader,
    device: torch.device,
    *,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    counter: Counter[str] = Counter()
    class_hit20 = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)
    direct_oracle_top1 = 0
    raw_oracle_top1 = 0
    for raw in loader:
        batch = _to_device(raw, device)
        outputs = model(batch["features"], batch["detector_logits"], batch["valid_mask"])
        for batch_index, gold in enumerate(raw["gold_events"]):
            proposals = decode_class_aware_proposals(outputs, batch_index, max_events=20)
            counter["scenes"] += 1
            counter["events"] += len(gold)
            for item in gold:
                class_total[int(item["label_id"])] += 1
                start = max(0, min(NUM_FRAMES - 1, int(math.floor(float(item["start"]) * NUM_FRAMES))))
                end = max(
                    start + 1,
                    min(NUM_FRAMES, int(math.ceil(float(item["end"]) * NUM_FRAMES))),
                )
                label_id = int(item["label_id"])
                direct_prediction = outputs["direct_presence_logits"][batch_index, start:end].mean(0).argmax()
                raw_prediction = batch["detector_logits"][batch_index, start:end].mean(0).argmax()
                direct_oracle_top1 += int(direct_prediction.item() == label_id)
                raw_oracle_top1 += int(raw_prediction.item() == label_id)
            for top_k in (5, 8, 20):
                for suffix, threshold in (("30", 0.30), ("50", 0.50)):
                    hit, hit_index = _matched_count(
                        proposals, gold, top_k=top_k, threshold=threshold
                    )
                    counter[f"top{top_k}_{suffix}"] += hit
                    counter[f"top{top_k}_scene_{suffix}"] += int(hit == len(gold))
                    if top_k == 20 and suffix == "50":
                        for gold_index in hit_index:
                            class_hit20[int(gold[gold_index]["label_id"])] += 1
    events = max(int(counter["events"]), 1)
    scenes = max(int(counter["scenes"]), 1)
    observed = class_total > 0
    result: dict[str, Any] = {
        "scenes": int(counter["scenes"]),
        "gold_events": int(counter["events"]),
        "classes_observed": int(observed.sum()),
        "direct_oracle_interval_top1_accuracy_↑": direct_oracle_top1 / events,
        "raw_detector_oracle_interval_top1_accuracy_↑": raw_oracle_top1 / events,
        "top20_macro_label_iou50_recall_↑": float(
            (class_hit20[observed].float() / class_total[observed].float()).mean()
        ),
    }
    for top_k in (5, 8, 20):
        for suffix in ("30", "50"):
            result[f"top{top_k}_label_iou{suffix}_recall_↑"] = (
                counter[f"top{top_k}_{suffix}"] / events
            )
            result[f"top{top_k}_all_events_scene_iou{suffix}_accuracy_↑"] = (
                counter[f"top{top_k}_scene_{suffix}"] / scenes
            )
    return result


def _selection_key(nonoverlap: Mapping[str, Any], overlap: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        0.5
        * (
            float(nonoverlap["top8_label_iou50_recall_↑"])
            + float(overlap["top8_label_iou50_recall_↑"])
        ),
        float(overlap["top8_label_iou50_recall_↑"]),
        0.5
        * (
            float(nonoverlap["direct_oracle_interval_top1_accuracy_↑"])
            + float(overlap["direct_oracle_interval_top1_accuracy_↑"])
        ),
        0.5
        * (
            float(nonoverlap["top20_macro_label_iou50_recall_↑"])
            + float(overlap["top20_macro_label_iou50_recall_↑"])
        ),
    )


def main() -> None:
    args = parse_args()
    for name in ("epochs", "patience", "batch_size"):
        if int(getattr(args, name)) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)

    indexes = [
        args.nonoverlap_train_index.resolve(),
        args.nonoverlap_dev_index.resolve(),
        args.overlap_train_index.resolve(),
        args.overlap_dev_index.resolve(),
    ]
    store = DenseFeatureStore(indexes, cache_size=32)
    labels = list(store.labels or [])
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
        name: ClassAwareSceneDataset(
            store,
            ids,
            preload=args.preload,
            max_scenes=args.max_train_scenes if "train" in name else args.max_dev_scenes,
        )
        for name, ids in scene_ids.items()
    }
    if len(datasets["nonoverlap_train"]) != len(datasets["overlap_train"]):
        raise RuntimeError("training requires equal non-overlap and overlap exposure")
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_class_aware,
    }
    train_loader = DataLoader(
        ConcatDataset((datasets["nonoverlap_train"], datasets["overlap_train"])),
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_args,
    )
    nonoverlap_dev_loader = DataLoader(
        datasets["nonoverlap_dev"], shuffle=False, **loader_args
    )
    overlap_dev_loader = DataLoader(datasets["overlap_dev"], shuffle=False, **loader_args)
    device = _device(args.device)
    config = ClassAwareRedEpnV1Config(
        feature_dim=int(store.feature_dim),
        num_classes=len(labels),
        context_dim=args.context_dim,
        epn_hidden_dim=args.epn_hidden_dim,
    )
    model = ClassAwareRedEpnV1(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_weights = ClassAwareRedEpnV1LossWeights()
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    stale = 0
    checkpoint_path = output_dir / "class_aware_red_epn_best.pt"

    baseline = {
        "epoch": 0,
        "train": None,
        "nonoverlap_dev": evaluate(model, nonoverlap_dev_loader, device, num_classes=len(labels)),
        "overlap_dev": evaluate(model, overlap_dev_loader, device, num_classes=len(labels)),
    }
    baseline["selection_key"] = list(
        _selection_key(baseline["nonoverlap_dev"], baseline["overlap_dev"])
    )
    print(json.dumps({"baseline": baseline}, ensure_ascii=False, sort_keys=True), flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals: Counter[str] = Counter()
        batches = 0
        for raw in train_loader:
            batch = _to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                outputs = model(
                    batch["features"], batch["detector_logits"], batch["valid_mask"]
                )
                losses = class_aware_red_epn_v1_loss(
                    outputs, batch, batch["valid_mask"], weights=loss_weights
                )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            for name, value in losses.items():
                totals[name] += float(value.detach())
            batches += 1
        scheduler.step()
        nonoverlap = evaluate(model, nonoverlap_dev_loader, device, num_classes=len(labels))
        overlap_metrics = evaluate(model, overlap_dev_loader, device, num_classes=len(labels))
        key = _selection_key(nonoverlap, overlap_metrics)
        row = {
            "epoch": epoch,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": {name: value / max(batches, 1) for name, value in totals.items()},
            "nonoverlap_dev": nonoverlap,
            "overlap_dev": overlap_metrics,
            "selection_key": list(key),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_metrics, stale = key, epoch, row, 0
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT,
                    "epoch": epoch,
                    "config": asdict(config),
                    "labels": labels,
                    "model_state_dict": model.state_dict(),
                    "selection_metrics": row,
                    "dense_indexes": [str(path) for path in indexes],
                    "dense_index_sha256": {str(path): _sha256_file(path) for path in indexes},
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_metrics is None:
        raise RuntimeError("class-aware RED/EPN training produced no checkpoint")
    gates = {
        "macro_top8_label_iou50_recall_ge_0_70": best_key[0] >= 0.70,
        "overlap_top8_label_iou50_recall_ge_0_60": best_key[1] >= 0.60,
        "macro_oracle_interval_top1_ge_0_60": best_key[2] >= 0.60,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "frozen BEATs features/logits + class-aware RED + single-GRU EPN",
        "answer_label_used_as_input": False,
        "arguments": _jsonable(vars(args)),
        "config": asdict(config),
        "loss_weights": asdict(loss_weights),
        "labels": labels,
        "classes": len(labels),
        "data": {name: len(dataset) for name, dataset in datasets.items()},
        "identity_audit": identity,
        "baseline": baseline,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "success_gates": gates,
        "all_success_gates_pass": all(gates.values()),
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
                "selection_key": list(best_key),
                "all_success_gates_pass": receipt["all_success_gates_pass"],
                "checkpoint_sha256": receipt["checkpoint_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

