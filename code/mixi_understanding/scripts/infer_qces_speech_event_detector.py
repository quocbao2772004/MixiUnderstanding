#!/usr/bin/env python3
"""Apply an existing QCES BEATs temporal head to a new scene manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from mixi_understanding.scripts.train_qces_speech_event_detector import (
    LABELS,
    PROJECT_ROOT,
    _atomic_text,
    _event_metrics,
    _extract_feature,
    _load_model,
    _read_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = _read_jsonl(args.scene_manifest.resolve())
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    if tuple(checkpoint["labels"]) != LABELS:
        raise RuntimeError("checkpoint labels do not match the 11-class detector")
    feature_dim = int(checkpoint["feature_dim"])
    head = nn.Sequential(
        nn.LayerNorm(feature_dim), nn.Linear(feature_dim, feature_dim // 2), nn.GELU(),
        nn.Dropout(0.15), nn.Linear(feature_dim // 2, len(LABELS)),
    ).to(device)
    head.load_state_dict(checkpoint["head_state_dict"])
    head.eval()
    backbone = _load_model(device)
    features = []
    for index, scene in enumerate(scenes, start=1):
        features.append(_extract_feature(backbone, scene, device).float())
        print(f"features {index}/{len(scenes)}", flush=True)
    with torch.no_grad():
        probs = torch.sigmoid(head(torch.stack(features).to(device))).cpu()
    thresholds = torch.tensor([float(checkpoint["class_thresholds"][label]) for label in LABELS])
    receipt = {
        "format": "qces_speech_event_inference_receipt_v1", "complete": True,
        "checkpoint": str(args.checkpoint.resolve().relative_to(PROJECT_ROOT)),
        "labels": list(LABELS), "scene_count": len(scenes), "splits": {},
    }
    for split in ("val", "test"):
        indices = [i for i, scene in enumerate(scenes) if scene["split"] == split]
        split_scenes = [scenes[i] for i in indices]
        split_probs = probs[indices]
        metrics, rows = _event_metrics(split_scenes, split_probs, thresholds)
        receipt["splits"][split] = metrics
        _atomic_text(output / f"detector_predictions_{split}.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    receipt["scene_manifest_sha256"] = hashlib.sha256(args.scene_manifest.resolve().read_bytes()).hexdigest()
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt["splits"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
