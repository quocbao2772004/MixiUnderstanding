#!/usr/bin/env python3
"""Evaluate a trained QCES PretrainedSED detector checkpoint on one split."""

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
    SceneItem,
    collate,
    load_model,
    load_ontology,
    load_scene_manifest,
    make_device,
    probs_to_events,
    read_jsonl,
    run_model,
    save_json,
    summarize_detector,
    summarize_qa,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--detector-manifest", type=Path, required=True)
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_AUDIT_DIR / "ontology_ready_train.txt")
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--event-iou-threshold", type=float, default=0.30)
    parser.add_argument("--min-duration", type=float, default=0.08)
    parser.add_argument("--merge-gap", type=float, default=0.12)
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
    label_to_id = {label: index for index, label in enumerate(labels)}
    report = checkpoint.get("report") or {}
    model_checkpoint_name = str(report.get("checkpoint") or "BEATs_strong_1")
    threshold = float(args.threshold if args.threshold >= 0 else checkpoint.get("best_threshold", report.get("best_threshold", 0.5)))

    device = make_device(args.device)
    model = load_model(len(labels), model_checkpoint_name, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

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
    detector_summary = summarize_detector(
        predictions,
        labels,
        threshold=threshold,
        event_iou_threshold=args.event_iou_threshold,
        min_duration=args.min_duration,
        merge_gap=args.merge_gap,
    )

    pred_events_by_scene: dict[str, list[dict[str, Any]]] = {}
    prediction_rows: list[dict[str, Any]] = []
    for row in predictions:
        scene = SceneItem(
            scene_id=str(row["scene_id"]),
            split=str(row["split"]),
            mixture_path="",
            duration_seconds=float(row["duration_seconds"]),
            sample_rate=16_000,
            events=tuple(row["gold_events"]),
        )
        events = probs_to_events(
            row["probs"],
            scene,
            labels,
            threshold=threshold,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
        )
        pred_events_by_scene[scene.scene_id] = events
        prediction_rows.append(
            {
                "scene_id": scene.scene_id,
                "split": scene.split,
                "threshold": threshold,
                "predicted_events": events,
                "gold_events": row["gold_events"],
            }
        )

    qa_rows = [row for row in read_jsonl(args.qa_manifest.resolve()) if str(row.get("scene_id") or "") in pred_events_by_scene]
    qa_summary = summarize_qa(qa_rows, pred_events_by_scene)
    summary = {
        "format": "qces_pretrainedsed_detector_eval_v1",
        "checkpoint_path": str(args.checkpoint_path.resolve()),
        "detector_manifest": str(args.detector_manifest.resolve()),
        "qa_manifest": str(args.qa_manifest.resolve()),
        "ontology": str(args.ontology.resolve()),
        "audio_root": str(args.audio_root.resolve()),
        "device": str(device),
        "threshold": threshold,
        "labels": labels,
        "scenes": len(scene_rows),
        "detector": detector_summary,
        "qa": {key: value for key, value in qa_summary.items() if key != "rows"},
    }
    save_json(output_dir / "eval_report.json", summary)
    write_jsonl(output_dir / "predicted_events.jsonl", prediction_rows)
    write_jsonl(output_dir / "qa_predictions.jsonl", qa_summary["rows"])

    md = [
        "# QCES PretrainedSED detector evaluation",
        "",
        f"- scenes: {len(scene_rows)}",
        f"- QA rows: {qa_summary['items']}",
        f"- threshold: {threshold:.2f}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| scene-label F1 ↑ | {detector_summary['scene_label']['f1_↑']:.3f} |",
        f"| scene-label precision ↑ | {detector_summary['scene_label']['precision_↑']:.3f} |",
        f"| scene-label recall ↑ | {detector_summary['scene_label']['recall_↑']:.3f} |",
        f"| avg labels kept ↓ | {detector_summary['scene_label']['avg_labels_kept_↓']:.2f} |",
        f"| avg false-positive labels ↓ | {detector_summary['scene_label']['avg_false_positive_labels_↓']:.2f} |",
        f"| event F1 @ IoU {args.event_iou_threshold:.2f} ↑ | {detector_summary['event_iou']['f1_↑']:.3f} |",
        f"| frame F1 ↑ | {detector_summary['frame']['f1_↑']:.3f} |",
        f"| QA accuracy ↑ | {qa_summary['accuracy_↑']:.3f} |",
        f"| answerable accuracy ↑ | {qa_summary['answerable_accuracy_↑']:.3f} |",
        f"| no-evidence accuracy ↑ | {qa_summary['no_evidence_accuracy_↑']:.3f} |",
        "",
        "| relation | items | accuracy ↑ |",
        "|---|---:|---:|",
    ]
    for relation, row in qa_summary["by_relation"].items():
        md.append(f"| {relation} | {row['items']} | {row['accuracy_↑']:.3f} |")
    (output_dir / "eval_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
