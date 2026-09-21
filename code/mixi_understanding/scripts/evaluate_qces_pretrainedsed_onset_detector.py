#!/usr/bin/env python3
"""Evaluate a trained QCES PretrainedSED onset detector checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch
from torch.utils.data import DataLoader

from train_qces_pretrainedsed_detector import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_DATASET_ROOT,
    SceneDataset,
    collate,
    load_ontology,
    load_scene_manifest,
    make_device,
    read_jsonl,
    save_json,
)
from train_qces_pretrainedsed_onset_detector import (
    OnsetDecoderConfig,
    compact_decoded_rows,
    load_initialized_model,
    run_model,
    score_config,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_onset_detector_30class/full801_val102_e6_tiny_v1/pretrainedsed_beats_qces_onset_detector.pt"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_AUDIT_DIR / "ontology_ready_train.txt")
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint_path.resolve(), map_location="cpu", weights_only=False)
    labels = list(checkpoint.get("labels") or load_ontology(args.ontology.resolve()))
    report = checkpoint.get("report") or {}
    init_checkpoint = Path(report.get("init_checkpoint") or "")
    if not init_checkpoint.exists():
        raise SystemExit(f"missing init checkpoint recorded in onset report: {init_checkpoint}")

    device = make_device(args.device)
    model = load_initialized_model(labels, init_checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    label_to_id = {label: index for index, label in enumerate(labels)}
    scene_rows = load_scene_manifest(args.detector_manifest.resolve(), label_to_id)
    loader = DataLoader(
        SceneDataset(scene_rows, audio_root=args.audio_root.resolve()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    predictions = run_model(model, loader, labels, device)
    qa_rows = [
        row
        for row in read_jsonl(args.qa_manifest.resolve())
        if str(row.get("scene_id") or "") in {scene.scene_id for scene in scene_rows}
    ]
    config = OnsetDecoderConfig(**checkpoint["best_config"])
    summary, decoded_rows, open_rows, option_rows = score_config(
        predictions,
        labels,
        qa_rows,
        config,
        event_iou_threshold=args.event_iou_threshold,
    )
    output = {
        "format": "qces_pretrainedsed_onset_detector_eval_v1",
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "detector_manifest": str(args.detector_manifest.resolve()),
        "qa_manifest": str(args.qa_manifest.resolve()),
        "ontology": str(args.ontology.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "device": str(device),
        "scenes": len(scene_rows),
        "labels": labels,
        "summary": summary,
    }
    save_json(output_dir / "eval_report.json", output)
    write_jsonl(output_dir / "predicted_events.jsonl", compact_decoded_rows(decoded_rows))
    write_jsonl(output_dir / "qa_predictions.jsonl", open_rows)
    write_jsonl(output_dir / "option_aware_predictions.jsonl", option_rows)

    md = [
        "# QCES PretrainedSED onset detector evaluation",
        "",
        f"- scenes: {len(scene_rows)}",
        f"- QA rows: {summary['open_qa']['items']}",
        f"- config: `{json.dumps(summary['config'], sort_keys=True)}`",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| event F1 @ IoU {args.event_iou_threshold:.2f} ↑ | {summary['detector']['event_iou']['f1_↑']:.3f} |",
        f"| frame F1 ↑ | {summary['detector']['frame']['f1_↑']:.3f} |",
        f"| scene-label F1 ↑ | {summary['detector']['scene_label']['f1_↑']:.3f} |",
        f"| open QA accuracy ↑ | {summary['open_qa']['accuracy_↑']:.3f} |",
        f"| open answerable accuracy ↑ | {summary['open_qa']['answerable_accuracy_↑']:.3f} |",
        f"| open no-evidence accuracy ↑ | {summary['open_qa']['no_evidence_accuracy_↑']:.3f} |",
        f"| option QA accuracy ↑ | {summary['option_qa']['accuracy_↑']:.3f} |",
        f"| option answerable accuracy ↑ | {summary['option_qa']['answerable_accuracy_↑']:.3f} |",
        f"| option no-evidence accuracy ↑ | {summary['option_qa']['no_evidence_accuracy_↑']:.3f} |",
    ]
    (output_dir / "eval_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
