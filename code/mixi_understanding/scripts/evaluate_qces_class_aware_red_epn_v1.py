#!/usr/bin/env python3
"""Audit semantic candidate ceilings of a class-aware RED/EPN checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
from torch.utils.data import DataLoader

from mixi_understanding.qces.class_aware_red_epn_v1 import (
    ClassAwareRedEpnV1,
    ClassAwareRedEpnV1Config,
    decode_class_aware_proposals,
)
from mixi_understanding.scripts.train_qces_class_aware_red_epn_v1 import (
    ClassAwareSceneDataset,
    collate_class_aware,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    NUM_FRAMES,
    _atomic_json,
    _device,
    _sha256_file,
    load_scene_list,
)


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=base / "class_aware_red_epn_v1_full/class_aware_red_epn_best.pt",
    )
    parser.add_argument("--nonoverlap-index", type=Path, default=base / "dense_multi_dev_v2/index.json")
    parser.add_argument(
        "--overlap-index", type=Path, default=base / "dense_overlap_query_dev_v2/index.json"
    )
    parser.add_argument(
        "--nonoverlap-scenes",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/multievent/scene_ids_dev.txt",
    )
    parser.add_argument(
        "--overlap-scenes",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/qces_full191_overlap_query_train_dev_v2/scene_ids_overlap_dev.txt",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


@torch.inference_mode()
def audit_split(
    model: ClassAwareRedEpnV1,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    totals: Counter[str] = Counter()
    for raw in loader:
        features = raw["features"].to(device, non_blocking=True)
        logits = raw["detector_logits"].to(device, non_blocking=True)
        valid = raw["valid_mask"].to(device, non_blocking=True)
        outputs = model(features, logits, valid)
        for batch_index, events in enumerate(raw["gold_events"]):
            proposals = decode_class_aware_proposals(
                outputs, batch_index, max_classes=64, max_events=40
            )
            unique_labels: list[int] = []
            for proposal in proposals:
                label_id = int(proposal["label_id"])
                if label_id not in unique_labels:
                    unique_labels.append(label_id)
            totals["scenes"] += 1
            totals["events"] += len(events)
            all_scene_hits: dict[int, bool] = {k: True for k in (1, 5, 8, 20)}
            for event in events:
                label_id = int(event["label_id"])
                start = max(
                    0,
                    min(NUM_FRAMES - 1, int(math.floor(float(event["start"]) * NUM_FRAMES))),
                )
                end = max(
                    start + 1,
                    min(NUM_FRAMES, int(math.ceil(float(event["end"]) * NUM_FRAMES))),
                )
                pooled = outputs["direct_presence_logits"][batch_index, start:end].mean(0)
                ordering = pooled.argsort(descending=True).tolist()
                for top_k in (1, 5, 8, 20):
                    interval_hit = label_id in ordering[:top_k]
                    proposal_hit = label_id in unique_labels[:top_k]
                    totals[f"interval_top{top_k}"] += int(interval_hit)
                    totals[f"proposal_label_top{top_k}"] += int(proposal_hit)
                    all_scene_hits[top_k] = all_scene_hits[top_k] and proposal_hit
            for top_k, hit in all_scene_hits.items():
                totals[f"all_labels_scene_top{top_k}"] += int(hit)
    event_count = max(int(totals["events"]), 1)
    scene_count = max(int(totals["scenes"]), 1)
    result: dict[str, Any] = {
        "scenes": int(totals["scenes"]),
        "gold_events": int(totals["events"]),
    }
    for top_k in (1, 5, 8, 20):
        result[f"oracle_interval_semantic_top{top_k}_accuracy_↑"] = (
            totals[f"interval_top{top_k}"] / event_count
        )
        result[f"proposal_label_only_recall_at_{top_k}_↑"] = (
            totals[f"proposal_label_top{top_k}"] / event_count
        )
        result[f"all_gold_labels_scene_recall_at_{top_k}_↑"] = (
            totals[f"all_labels_scene_top{top_k}"] / scene_count
        )
    return result


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ClassAwareRedEpnV1Config(**dict(payload["config"]))
    model = ClassAwareRedEpnV1(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = _device(args.device)
    model.to(device)

    store = DenseFeatureStore(
        [args.nonoverlap_index.resolve(), args.overlap_index.resolve()], cache_size=16
    )
    datasets = {
        "nonoverlap_dev": ClassAwareSceneDataset(
            store, load_scene_list(args.nonoverlap_scenes), preload=True
        ),
        "overlap_dev": ClassAwareSceneDataset(
            store, load_scene_list(args.overlap_scenes), preload=True
        ),
    }
    loader_args = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_class_aware,
    }
    metrics = {
        name: audit_split(model, DataLoader(dataset, **loader_args), device)
        for name, dataset in datasets.items()
    }
    output = (
        args.output.resolve()
        if args.output is not None
        else checkpoint_path.parent / "semantic_candidate_audit.json"
    )
    receipt: Mapping[str, Any] = {
        "format": "qces_class_aware_red_epn_semantic_candidate_audit_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "answer_label_used_as_input": False,
        "metrics": metrics,
    }
    _atomic_json(receipt, output)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
