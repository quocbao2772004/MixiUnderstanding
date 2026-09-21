#!/usr/bin/env python3
"""Decode event intervals from already-trained per-slot masks.

The V2 localizer predicts both a box and a dense mask for every slot, but its
historical evaluator uses only the box as the timestamp.  This evaluation-only
sidecar converts the connected mask component containing the slot's peak into
an interval and optionally fuses that interval with the box.  No parameter is
updated.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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


FORMAT = "qces_slot_mask_interval_decoder_evaluation_v1"


def _float_grid(value: str) -> list[float]:
    values = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if not values or any(not 0.0 <= item <= 1.0 for item in values):
        raise argparse.ArgumentTypeError("grid values must lie in [0,1]")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--objectness-grid", type=_parse_threshold_grid, default=_parse_threshold_grid("0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95"))
    parser.add_argument("--mask-threshold-grid", type=_float_grid, default=_float_grid("0.3,0.4,0.5,0.6,0.7"))
    parser.add_argument("--mask-weight-grid", type=_float_grid, default=_float_grid("0.5,0.75,1.0"))
    parser.add_argument("--fixed-objectness-threshold", type=float)
    parser.add_argument("--fixed-mask-threshold", type=float)
    parser.add_argument("--fixed-mask-weight", type=float)
    return parser.parse_args()


def _peak_component_interval(probability: torch.Tensor, threshold: float) -> torch.Tensor:
    if probability.ndim != 1 or probability.numel() < 1:
        raise ValueError("slot mask must be a non-empty vector")
    frames = int(probability.numel())
    peak = int(probability.argmax())
    active = probability >= threshold
    if not bool(active[peak]):
        active[peak] = True
    left = peak
    while left > 0 and bool(active[left - 1]):
        left -= 1
    right = peak + 1
    while right < frames and bool(active[right]):
        right += 1
    return torch.tensor([left / frames, right / frames], dtype=torch.float32)


def _decode(
    predictions: Mapping[str, Mapping[str, Any]], *, mask_threshold: float, mask_weight: float
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for scene_id, record in predictions.items():
        masks = record["mask_probability"].float()
        mask_intervals = torch.stack(
            [_peak_component_interval(mask, mask_threshold) for mask in masks], dim=0
        )
        boxes = record["intervals"].float()
        fused = (1.0 - mask_weight) * boxes + mask_weight * mask_intervals
        fused[:, 0] = fused[:, 0].clamp(0.0, 1.0)
        fused[:, 1] = fused[:, 1].clamp(0.0, 1.0)
        fused[:, 1] = fused[:, 1].maximum(fused[:, 0] + 1.0 / masks.shape[-1]).clamp_max(1.0)
        result[scene_id] = {**record, "intervals": fused}
    return result


def _profiles(manifest: Path | None, scene_ids: list[str]) -> dict[str, str]:
    if manifest is None:
        return {scene_id: "unspecified" for scene_id in scene_ids}
    result: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["scene_id"])] = str(row.get("layout_profile") or "unspecified")
    if missing := set(scene_ids) - set(result):
        raise ValueError(f"manifest misses {len(missing)} evaluated scenes")
    return result


def main() -> None:
    args = parse_args()
    fixed_values = (
        args.fixed_objectness_threshold,
        args.fixed_mask_threshold,
        args.fixed_mask_weight,
    )
    if any(value is not None for value in fixed_values) and not all(value is not None for value in fixed_values):
        raise ValueError("all three fixed decoder values must be supplied together")
    index_path = args.index.resolve()
    checkpoint_path = args.checkpoint.resolve()
    scene_list_path = args.scene_list.resolve()
    manifest_path = args.manifest.resolve() if args.manifest else None
    store = DenseFeatureStore([index_path], cache_size=20)
    scene_ids = load_scene_list(scene_list_path)
    if missing := set(scene_ids) - store.scene_ids:
        raise ValueError(f"dense store misses {len(missing)} scenes")
    dataset = OverlapSceneSlotDataset(store, scene_ids, preload=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_overlap_slots)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = OverlapEventSlotsV2(RelationalEventSlotsV1Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = _device(args.device)
    model.to(device)
    raw = collect_predictions(model, loader, device)

    baseline_grid = [
        evaluate_objectness_threshold(raw, objectness_threshold=value)
        for value in args.objectness_grid
    ]
    baseline = max(baseline_grid, key=_checkpoint_key)
    candidates: list[dict[str, Any]] = []
    if all(value is not None for value in fixed_values):
        decoder_grid = [
            (
                float(args.fixed_mask_threshold),
                float(args.fixed_mask_weight),
                float(args.fixed_objectness_threshold),
            )
        ]
        selection_mode = "fixed_from_external_split"
    else:
        decoder_grid = [
            (mask_threshold, mask_weight, objectness_threshold)
            for mask_threshold in args.mask_threshold_grid
            for mask_weight in args.mask_weight_grid
            for objectness_threshold in args.objectness_grid
        ]
        selection_mode = "diagnostic_grid_on_this_split"
    cache: dict[tuple[float, float], dict[str, dict[str, Any]]] = {}
    for mask_threshold, mask_weight, objectness_threshold in decoder_grid:
        key = (mask_threshold, mask_weight)
        decoded = cache.setdefault(
            key,
            _decode(raw, mask_threshold=mask_threshold, mask_weight=mask_weight),
        )
        metrics = evaluate_objectness_threshold(decoded, objectness_threshold=objectness_threshold)
        candidates.append(
            {
                "decoder": {
                    "mask_component": "connected_component_containing_global_peak",
                    "mask_threshold": mask_threshold,
                    "mask_weight": mask_weight,
                    "box_weight": 1.0 - mask_weight,
                    "objectness_threshold": objectness_threshold,
                },
                "metrics": metrics,
            }
        )
    selected = max(candidates, key=lambda row: _checkpoint_key(row["metrics"]))
    decoded = cache[(selected["decoder"]["mask_threshold"], selected["decoder"]["mask_weight"])]
    profile_by_scene = _profiles(manifest_path, scene_ids)
    profile_metrics: dict[str, Any] = {}
    for profile in sorted(set(profile_by_scene.values())):
        subset = {scene_id: decoded[scene_id] for scene_id in scene_ids if profile_by_scene[scene_id] == profile}
        profile_metrics[profile] = evaluate_objectness_threshold(
            subset,
            objectness_threshold=float(selected["decoder"]["objectness_threshold"]),
        )
    gates = {
        "f1_iou50_ge_0.87": float(selected["metrics"]["slot_f1_iou50"]) >= 0.87,
        "mean_iou_ge_0.85": float(selected["metrics"]["slot_mean_matched_iou"]) >= 0.85,
        "improves_box_f1": float(selected["metrics"]["slot_f1_iou50"]) > float(baseline["slot_f1_iou50"]),
        "improves_box_mean_iou": float(selected["metrics"]["slot_mean_matched_iou"]) > float(baseline["slot_mean_matched_iou"]),
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "selection_mode": selection_mode,
        "paper_eligible": False,
        "index": str(index_path),
        "index_sha256": _sha256_file(index_path),
        "scene_list": str(scene_list_path),
        "scene_list_sha256": _sha256_file(scene_list_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "scenes": len(scene_ids),
        "box_baseline": baseline,
        "selected": selected,
        "by_layout_profile": profile_metrics,
        "gates": gates,
        "all_quality_gates_pass": all(gates.values()),
        "candidate_count": len(candidates),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(receipt, args.output.resolve())
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
