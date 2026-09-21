#!/usr/bin/env python3
"""Export dense frame probabilities from a trained QCES PretrainedSED detector."""

from __future__ import annotations

import argparse
import json
import sys
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
    run_model,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200/partial_as_fuss_e4_unfreeze1_highthr_resume/pretrainedsed_beats_qces_detector.pt"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/detector_manifest_val.jsonl"
DEFAULT_ONTOLOGY = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/ontology_200_trainable.txt"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200/partial_as_fuss_e4_unfreeze1_highthr_resume/val_frame_probs.pt"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--audio-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-name", default="BEATs_strong_1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--float32", action="store_true", help="Store probabilities as float32 instead of float16.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"output exists: {output}; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    device = make_device(args.device)
    labels = load_ontology(args.ontology.resolve())
    label_to_id = {label: index for index, label in enumerate(labels)}
    rows = load_scene_manifest(args.manifest.resolve(), label_to_id, args.max_scenes)
    if not rows:
        raise SystemExit("empty manifest after ontology filtering")

    checkpoint = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=False)
    ckpt_labels = checkpoint.get("labels")
    if ckpt_labels != labels:
        raise ValueError(
            f"checkpoint/ontology labels mismatch: checkpoint={len(ckpt_labels or [])}, ontology={len(labels)}"
        )

    model = load_model(len(labels), args.checkpoint_name, device, unfreeze_last_blocks=0)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    loader = DataLoader(
        SceneDataset(rows, audio_root=args.audio_root.resolve()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    predictions = run_model(model, loader, labels, device)

    scenes: dict[str, dict[str, Any]] = {}
    dtype = torch.float32 if args.float32 else torch.float16
    for row in predictions:
        scenes[str(row["scene_id"])] = {
            "split": str(row.get("split") or ""),
            "duration_seconds": float(row["duration_seconds"]),
            "gold_events": list(row["gold_events"]),
            "probs": row["probs"].to(dtype=dtype).cpu(),
        }

    payload = {
        "format": "qces_detector_frame_probs_v1",
        "detector_checkpoint": str(args.detector_checkpoint.resolve()),
        "manifest": str(args.manifest.resolve()),
        "ontology": str(args.ontology.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "labels": labels,
        "scenes": scenes,
        "dtype": str(dtype),
    }
    torch.save(payload, output)
    summary = {
        "format": payload["format"],
        "output": str(output),
        "scenes": len(scenes),
        "labels": len(labels),
        "dtype": str(dtype),
        "detector_checkpoint": payload["detector_checkpoint"],
        "manifest": payload["manifest"],
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
