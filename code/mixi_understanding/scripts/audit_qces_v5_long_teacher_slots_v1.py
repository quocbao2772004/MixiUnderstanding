#!/usr/bin/env python3
"""Audit a clean long-view semantic teacher on frozen predicted slot masks.

No training is performed.  For every frozen event slot, mean/std/max BEATs
statistics are pooled inside the pre-registered mask component and scored by
the existing long-source teacher.  The result is compared with pooled R1 on
the same slots and the same objectness threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mixi_understanding.qces.overlap_event_slots_v2 import OverlapEventSlotsV2
from mixi_understanding.qces.relational_event_slots_v1 import RelationalEventSlotsV1Config
from mixi_understanding.scripts.train_qces_long_short_semantic_teacher_v2 import (
    LongHeadConfig,
    LongSemanticHead,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _atomic_torch,
    _device,
    _sha256_file,
    load_scene_list,
)
from mixi_understanding.scripts.train_qces_v4_slot_semantic_head_v1 import NONE_LABEL
from mixi_understanding.scripts.train_qces_v5_rehearsal_slot_semantic_head_v2 import (
    SemanticSceneDataset,
    _component_intervals_and_weights,
    _event_metrics,
    collate_semantic,
)


FORMAT = "qces_v5_long_teacher_slot_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--localizer-checkpoint", type=Path, required=True)
    parser.add_argument("--long-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _masked_stats(features: torch.Tensor, component: torch.Tensor) -> torch.Tensor:
    weight = component / component.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = torch.einsum("bst,btd->bsd", weight, features)
    centered = features[:, None] - mean[:, :, None]
    variance = torch.einsum("bst,bstd->bsd", weight, centered.square())
    expanded = features[:, None].expand(-1, component.shape[1], -1, -1)
    masked = expanded.masked_fill(component[..., None] <= 0.0, float("-inf"))
    maximum = masked.amax(dim=2)
    return torch.cat((mean, variance.clamp_min(1e-6).sqrt(), maximum), dim=-1)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    index_path = args.index.resolve()
    scene_path = args.scenes.resolve()
    base_path = args.base_cache.resolve()
    localizer_path = args.localizer_checkpoint.resolve()
    teacher_path = args.long_teacher_checkpoint.resolve()
    base_cache = torch.load(base_path, map_location="cpu", weights_only=True)
    scene_ids = load_scene_list(scene_path)
    if list(base_cache["scene_id"]) != scene_ids:
        raise ValueError("base cache and scene list order differ")
    store = DenseFeatureStore([index_path], cache_size=20)
    dataset = SemanticSceneDataset(store, scene_ids, preload=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_semantic,
    )

    localizer_checkpoint = torch.load(localizer_path, map_location="cpu", weights_only=True)
    config = RelationalEventSlotsV1Config(**localizer_checkpoint["config"])
    objectness_threshold = float(localizer_checkpoint["objectness_threshold"])
    mask_threshold = float(localizer_checkpoint["mask_decoder"]["mask_threshold"])
    localizer = OverlapEventSlotsV2(config)
    localizer.load_state_dict(localizer_checkpoint["model_state_dict"], strict=True)
    teacher_checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=True)
    teacher = LongSemanticHead(LongHeadConfig(**teacher_checkpoint["long_config"]))
    teacher.load_state_dict(teacher_checkpoint["long_model_state_dict"], strict=True)
    device = _device(args.device)
    localizer.to(device).eval()
    teacher.to(device).eval()

    teacher_logits: list[torch.Tensor] = []
    teacher_embeddings: list[torch.Tensor] = []
    observed_objectness: list[torch.Tensor] = []
    observed_intervals: list[torch.Tensor] = []
    processed = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True).float()
        detector = batch["detector_logits"].to(device, non_blocking=True).float()
        valid = batch["valid_mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(args.amp and device.type == "cuda"),
        ):
            outputs = localizer(features, valid)
        mask_probability = outputs["slot_mask_logits"].sigmoid()
        intervals, component_cpu = _component_intervals_and_weights(
            mask_probability, threshold=mask_threshold
        )
        component = component_cpu.to(device, non_blocking=True)
        component = component * valid[:, None].to(component.dtype)
        stats = _masked_stats(features, component)
        weight = component / component.sum(dim=-1, keepdim=True).clamp_min(1.0)
        pooled_r1 = torch.einsum("bst,btc->bsc", weight, detector)
        batch_size, slots, _ = stats.shape
        logits, embedding = teacher(
            stats.reshape(batch_size * slots, -1),
            pooled_r1.reshape(batch_size * slots, -1),
        )
        teacher_logits.append(logits.reshape(batch_size, slots, -1).cpu())
        teacher_embeddings.append(embedding.reshape(batch_size, slots, -1).cpu().half())
        observed_objectness.append(outputs["objectness_logits"].sigmoid().cpu().half())
        observed_intervals.append(intervals.half())
        processed += batch_size
        if processed == batch_size or processed % 512 < batch_size:
            print(f"long_teacher_slots {processed}/{len(dataset)}", flush=True)

    logits = torch.cat(teacher_logits)
    embedding = torch.cat(teacher_embeddings)
    objectness = torch.cat(observed_objectness)
    intervals = torch.cat(observed_intervals)
    if not torch.allclose(objectness.float(), base_cache["objectness"].float(), atol=2e-3, rtol=2e-3):
        raise ValueError("localizer objectness does not reproduce the frozen base cache")
    if not torch.allclose(intervals.float(), base_cache["intervals"].float(), atol=5e-3, rtol=2e-3):
        raise ValueError("mask intervals do not reproduce the frozen base cache")
    long_scores = F.pad(logits.float(), (0, 1), value=-1e9)
    pooled_r1 = F.pad(base_cache["slot_input"][..., -NONE_LABEL:].float(), (0, 1), value=-1e9)
    long_metrics = _event_metrics(
        base_cache,
        long_scores,
        objectness_threshold=objectness_threshold,
        learned=False,
    )
    r1_metrics = _event_metrics(
        base_cache,
        pooled_r1,
        objectness_threshold=objectness_threshold,
        learned=False,
    )
    gates = {
        "long_teacher_top1_improves_pooled_r1": float(long_metrics["label_top1_given_iou50_↑"])
        > float(r1_metrics["label_top1_given_iou50_↑"]),
        "long_teacher_joint_f1_improves_pooled_r1": float(long_metrics["joint_label_iou50_f1_↑"])
        > float(r1_metrics["joint_label_iou50_f1_↑"]),
    }
    feature_path = args.output.resolve().with_name(args.output.stem + "_features.pt")
    _atomic_torch(
        {
            "format": "qces_v5_long_teacher_slot_features_v1",
            "scene_id": list(base_cache["scene_id"]),
            "teacher_logits": logits.half(),
            "teacher_embedding": embedding,
        },
        feature_path,
    )
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "split": args.split_name,
        "scenes": len(scene_ids),
        "classes": NONE_LABEL,
        "objectness_threshold": objectness_threshold,
        "mask_threshold": mask_threshold,
        "pooled_r1": r1_metrics,
        "long_teacher": long_metrics,
        "gates": gates,
        "all_quality_gates_pass": all(gates.values()),
        "base_cache": str(base_path),
        "base_cache_sha256": _sha256_file(base_path),
        "localizer_checkpoint": str(localizer_path),
        "localizer_checkpoint_sha256": _sha256_file(localizer_path),
        "long_teacher_checkpoint": str(teacher_path),
        "long_teacher_checkpoint_sha256": _sha256_file(teacher_path),
        "features": str(feature_path),
        "features_sha256": _sha256_file(feature_path),
    }
    _atomic_json(receipt, args.output.resolve())
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
