#!/usr/bin/env python3
"""Export frozen BEATs transformer tokens on their original time-frequency grid."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from torch.utils.data import DataLoader

from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    SceneDataset,
    collate,
    load_model,
    load_ontology,
    load_scene_manifest,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json,
    _atomic_torch,
    _sha256_file,
)


FORMAT = "qces_beats_tf_grid_index_v1"
SHARD_FORMAT = "qces_beats_tf_grid_shard_v1"
TIME_PATCHES = 62
FREQUENCY_PATCHES = 8
FEATURE_DIM = 768


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_semantic_sufficient_v4"
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=data / "detector_scene_manifest_overlap_dev.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrainedsed-checkpoint-name", default="BEATs_strong_1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def extract_grid(model: torch.nn.Module, waveforms: torch.Tensor) -> torch.Tensor:
    mel = model.mel_forward(waveforms)
    source = mel.transpose(2, 3)
    patch = model.model.beats.patch_embedding(source)
    if tuple(patch.shape[-2:]) != (TIME_PATCHES, FREQUENCY_PATCHES):
        raise ValueError(f"unexpected BEATs patch grid: {tuple(patch.shape)}")
    tokens = model.model(mel)
    expected = (waveforms.shape[0], TIME_PATCHES * FREQUENCY_PATCHES, FEATURE_DIM)
    if tuple(tokens.shape) != expected:
        raise ValueError(f"unexpected BEATs tokens: {tuple(tokens.shape)} != {expected}")
    return tokens.reshape(waveforms.shape[0], TIME_PATCHES, FREQUENCY_PATCHES, FEATURE_DIM)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    index_path = output_dir / "index.json"
    if index_path.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {index_path}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_ontology(args.ontology.resolve())
    label_to_id = {label: index for index, label in enumerate(labels)}
    scenes = load_scene_manifest(args.manifest.resolve(), label_to_id, args.max_scenes)
    if not scenes:
        raise ValueError("empty scene manifest")

    checkpoint = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if checkpoint.get("labels") != labels:
        raise ValueError("detector checkpoint and ontology labels differ")
    device = make_device(args.device)
    model = load_model(len(labels), args.pretrainedsed_checkpoint_name, device, unfreeze_last_blocks=0)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False)
    loader = DataLoader(
        SceneDataset(scenes, audio_root=PROJECT_ROOT, fixed_seconds=10.0),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", collate_fn=collate,
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    pending: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    processed = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        shard_index = len(shards)
        name = f"shard-{run_id}-{shard_index:05d}.pt"
        path = output_dir / name
        payload = {
            "format": SHARD_FORMAT, "run_id": run_id, "shard_index": shard_index,
            "scene_ids": [row["scene_id"] for row in pending],
            "grid_features": torch.stack([row["grid_features"] for row in pending]),
            "gold_events": [row["gold_events"] for row in pending],
        }
        expected = (len(pending), TIME_PATCHES, FREQUENCY_PATCHES, FEATURE_DIM)
        if tuple(payload["grid_features"].shape) != expected or payload["grid_features"].dtype != torch.float16:
            raise ValueError("invalid TF-grid shard tensor")
        _atomic_torch(payload, path)
        shards.append({
            "index": shard_index, "path": name, "scene_count": len(pending),
            "first_scene_id": pending[0]["scene_id"], "last_scene_id": pending[-1]["scene_id"],
            "bytes": path.stat().st_size, "sha256": _sha256_file(path),
        })
        print(f"wrote shard={shard_index + 1} scenes={len(pending)} total={processed}", flush=True)
        pending = []

    for waveforms, batch_scenes in loader:
        grid = extract_grid(model, waveforms.to(device, non_blocking=True)).half().cpu()
        for batch_index, scene in enumerate(batch_scenes):
            pending.append({
                "scene_id": scene.scene_id,
                "grid_features": grid[batch_index].contiguous(),
                "gold_events": list(scene.events),
            })
            processed += 1
            if len(pending) >= args.shard_size:
                flush()
        if processed == len(batch_scenes) or processed % 128 < len(batch_scenes):
            print(f"export {processed}/{len(scenes)}", flush=True)
    flush()
    if processed != len(scenes):
        raise RuntimeError("export count mismatch")
    index = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "run_id": run_id, "scene_count": processed,
        "labels": labels, "grid": {
            "time_patches": TIME_PATCHES, "frequency_patches": FREQUENCY_PATCHES,
            "feature_dim": FEATURE_DIM,
            "token_order": "reshape exact from [B,496,768] to [B,time=62,frequency=8,768]",
            "input_mel_shape": [128, 998],
        },
        "manifest": str(args.manifest.resolve()), "manifest_sha256": _sha256_file(args.manifest.resolve()),
        "ontology": str(args.ontology.resolve()), "ontology_sha256": _sha256_file(args.ontology.resolve()),
        "detector_checkpoint": str(args.detector_checkpoint.resolve()),
        "detector_checkpoint_sha256": _sha256_file(args.detector_checkpoint.resolve()),
        "qa_question_or_answer_used_as_input": False,
        "shards": shards,
    }
    _atomic_json(index, index_path)
    print(json.dumps({"complete": True, "scene_count": processed, "shards": len(shards), "index": str(index_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
