#!/usr/bin/env python3
"""Select deployment thresholds from cached dev scores and report locked test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mixi_understanding.scripts.evaluate_human_event_beats_live_v1 import (
    PROJECT_ROOT,
    RAW_TO_LABEL,
    THRESHOLD_GRID,
    data_counts,
    metrics,
    read_jsonl,
)


def load_scores(path: Path) -> list[np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return [row.float().numpy() for row in payload["scores"]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_clean_detector_protocol_v1_current",
    )
    parser.add_argument(
        "--score-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_live_human_events_beats_v1",
    )
    args = parser.parse_args()

    protocol = args.protocol_dir.resolve()
    output = args.score_dir.resolve()
    splits = {
        split: read_jsonl(protocol / f"detector_manifest_{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    score_rows = {
        split: load_scores(output / f"{split}_frame_scores.pt")
        for split in ("dev", "test")
    }

    selected: dict[str, float] = {}
    dev_grid: dict[str, Any] = {}
    test_results: dict[str, Any] = {}
    for label_index, label in enumerate(RAW_TO_LABEL.values()):
        candidates = [
            metrics(splits["dev"], score_rows["dev"], label_index, label, threshold)
            for threshold in THRESHOLD_GRID
        ]
        # The live UI needs usable intervals, not only clip-level presence.
        # Select on dev event F1; balanced accuracy and IoU only break ties.
        best = max(
            candidates,
            key=lambda row: (
                row["event_f1_↑"],
                row["clip_balanced_accuracy_↑"],
                row["matched_mean_iou_↑"],
                -row["threshold"],
            ),
        )
        selected[label] = float(best["threshold"])
        dev_grid[label] = {"selected": best, "grid": candidates}
        test_results[label] = metrics(
            splits["test"],
            score_rows["test"],
            label_index,
            label,
            selected[label],
        )

    macro_names = (
        "clip_accuracy_↑",
        "clip_balanced_accuracy_↑",
        "event_f1_↑",
        "matched_mean_iou_↑",
        "frame_f1_↑",
    )
    macro = {
        name: float(np.mean([row[name] for row in test_results.values()]))
        for name in macro_names
    }
    receipt = {
        "format": "qces_live_human_events_beats_deployment_v1",
        "complete": True,
        "model": "public PretrainedSED BEATs-Strong (unmodified)",
        "qces_finetuning": False,
        "classes": list(RAW_TO_LABEL.values()),
        "data_scene_counts": {split: len(rows) for split, rows in splits.items()},
        "data": {split: data_counts(rows) for split, rows in splits.items()},
        "selection": "per-class maximum dev event F1; balanced accuracy and matched IoU tie-breakers",
        "test_annotations_used_for_threshold_selection": False,
        "selected_thresholds": selected,
        "dev": dev_grid,
        "test": test_results,
        "test_macro": macro,
        "metric_definitions": {
            "clip_accuracy": "presence/absence accuracy; inflated by class imbalance",
            "clip_balanced_accuracy": "mean of positive recall and negative recall",
            "event_f1": "greedy same-class event matching at temporal IoU >= 0.30",
            "matched_mean_iou": "mean temporal IoU over matched true-positive events only",
            "frame_f1": "40 ms frame-level activity F1",
        },
        "limitations": [
            "AudioSet-Strong-derived evaluation is not guaranteed disjoint from public BEATs pretraining.",
            "Matched mean IoU excludes missed events and must be read together with event recall/F1.",
        ],
    }
    (output / "deployment_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"thresholds": selected, "test": test_results, "macro": macro}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
