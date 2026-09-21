#!/usr/bin/env python3
"""Evaluate a frozen overlap-slot checkpoint on a dense-feature split.

This sidecar is intentionally evaluation-only.  It sweeps one global
objectness threshold on the requested development split, reports the frozen
checkpoint threshold as a separate row, and breaks the selected result down by
V5 layout profile.  It never updates model parameters.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.overlap_event_slots_v2 import OverlapEventSlotsV2
from mixi_understanding.qces.relational_event_slots_v1 import RelationalEventSlotsV1Config
from mixi_understanding.scripts.train_qces_overlap_event_slots_v2 import (
    OverlapSceneSlotDataset,
    _checkpoint_key,
    _parse_threshold_grid,
    collate_overlap_slots,
    collect_predictions,
    evaluate_objectness_threshold,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _device,
    _sha256_file,
    load_scene_list,
)


FORMAT = "qces_overlap_event_slots_frozen_evaluation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Optional raw manifest for layout-profile breakdown.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shard-cache-size", type=int, default=20)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--objectness-grid",
        type=_parse_threshold_grid,
        default=_parse_threshold_grid("0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.index.resolve()
    checkpoint_path = args.checkpoint.resolve()
    scene_list_path = args.scene_list.resolve()
    store = DenseFeatureStore([index_path], cache_size=args.shard_cache_size)
    scene_ids = load_scene_list(scene_list_path)
    missing = sorted(set(scene_ids) - store.scene_ids)
    if missing:
        raise ValueError(f"scene list contains {len(missing)} ids absent from dense index")

    dataset = OverlapSceneSlotDataset(store, scene_ids, preload=args.preload)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_overlap_slots,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = RelationalEventSlotsV1Config(**payload["config"])
    if int(config.feature_dim) != int(store.feature_dim or -1):
        raise ValueError("checkpoint and dense feature dimensions differ")
    model = OverlapEventSlotsV2(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = _device(args.device)
    model.to(device)

    predictions = collect_predictions(model, loader, device)
    grid = [
        evaluate_objectness_threshold(predictions, objectness_threshold=value)
        for value in args.objectness_grid
    ]
    selected = max(grid, key=_checkpoint_key)
    frozen_threshold = float(payload.get("objectness_threshold", selected["objectness_threshold"]))
    frozen = evaluate_objectness_threshold(predictions, objectness_threshold=frozen_threshold)

    profile_by_scene: dict[str, str] = {}
    manifest_path: Path | None = None
    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            profile_by_scene[str(row["scene_id"])] = str(row.get("layout_profile") or "unspecified")
        if missing_profiles := sorted(set(scene_ids) - set(profile_by_scene)):
            raise ValueError(f"manifest misses {len(missing_profiles)} evaluated scenes")
    else:
        profile_by_scene = {
            scene_id: str(store.metadata(scene_id).get("layout_profile") or "unspecified")
            for scene_id in scene_ids
        }
    profiles: dict[str, dict[str, Any]] = {}
    profile_names = sorted(set(profile_by_scene.values()))
    for profile in profile_names:
        subset = {
            scene_id: predictions[scene_id]
            for scene_id in scene_ids
            if profile_by_scene[scene_id] == profile
        }
        profiles[profile] = evaluate_objectness_threshold(
            subset,
            objectness_threshold=float(selected["objectness_threshold"]),
        )

    gates = {
        "slot_precision_iou50_ge_0.85": float(selected["slot_precision_iou50"]) >= 0.85,
        "slot_recall_iou50_ge_0.90": float(selected["slot_recall_iou50"]) >= 0.90,
        "slot_f1_iou50_ge_0.87": float(selected["slot_f1_iou50"]) >= 0.87,
        "slot_mean_matched_iou_ge_0.85": float(selected["slot_mean_matched_iou"]) >= 0.85,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "diagnostic_threshold_selected_on_this_split": True,
        "paper_eligible": False,
        "index": str(index_path),
        "index_sha256": _sha256_file(index_path),
        "scene_list": str(scene_list_path),
        "scene_list_sha256": _sha256_file(scene_list_path),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "manifest_sha256": _sha256_file(manifest_path) if manifest_path is not None else None,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "scenes": len(scene_ids),
        "selected": selected,
        "frozen_checkpoint_threshold": frozen,
        "by_layout_profile_at_selected_threshold": profiles,
        "threshold_grid": grid,
        "success_gates": gates,
        "all_success_gates_pass": all(gates.values()),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(receipt, args.output.resolve())
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
