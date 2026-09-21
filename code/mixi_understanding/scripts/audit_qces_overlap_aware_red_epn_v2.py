#!/usr/bin/env python3
"""Measure event and adjacent-pair recall of overlap-aware RED/EPN v2."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    interval_iou,
)
from mixi_understanding.scripts.train_qces_class_aware_red_epn_v1 import (
    ClassAwareSceneDataset,
    collate_class_aware,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _device,
    _sha256_file,
    load_scene_list,
)


FORMAT = "qces_overlap_aware_red_epn_audit_v2"
DEFAULT_BASE = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
)
DEFAULT_DATA = PROJECT_ROOT / "outputs/qces_full188_overlap_gold_natural_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_BASE / "overlap_aware_red_epn_v2/overlap_aware_red_epn_v2_best.pt",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=DEFAULT_BASE / "dense_overlap_gold_natural_v3_dev/index.json",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        default=DEFAULT_DATA / "scene_ids_overlap_dev.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BASE / "overlap_aware_red_epn_v2/adjacent_pair_audit.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 8, 20])
    return parser.parse_args()


def adjacent_index_pairs(events: Sequence[Mapping[str, Any]]) -> list[tuple[int, int]]:
    ordered = sorted(
        enumerate(events),
        key=lambda row: (float(row[1]["start"]), float(row[1]["end"]), row[0]),
    )
    pairs: list[tuple[int, int]] = []
    for left, right in zip(ordered, ordered[1:]):
        if math.isclose(
            float(left[1]["start"]), float(right[1]["start"]), abs_tol=1e-9
        ):
            continue
        pairs.append((left[0], right[0]))
    return pairs


def event_hit(
    proposals: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
    *,
    top_k: int,
    iou_threshold: float,
) -> bool:
    for proposal in proposals[:top_k]:
        if int(proposal["label_id"]) != int(event["label_id"]):
            continue
        if interval_iou(
            (float(proposal["start"]), float(proposal["end"])),
            (float(event["start"]), float(event["end"])),
        ) >= iou_threshold:
            return True
    return False


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ClassAwareRedEpnV1Config(**dict(payload["config"]))
    model = ClassAwareRedEpnV1(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = _device(args.device)
    model.to(device).eval()

    store = DenseFeatureStore([args.dense_index.resolve()], cache_size=16)
    dataset = ClassAwareSceneDataset(
        store, load_scene_list(args.scene_list.resolve()), preload=True
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_class_aware,
    )
    top_k_values = sorted(set(int(value) for value in args.top_k))
    if not top_k_values or min(top_k_values) < 1 or max(top_k_values) > config.num_classes:
        raise ValueError("top-k values must be between 1 and the ontology size")
    max_candidates = max(top_k_values)
    max_classes = max(32, max_candidates)

    totals: Counter[str] = Counter()
    for raw in loader:
        features = raw["features"].to(device, non_blocking=True)
        logits = raw["detector_logits"].to(device, non_blocking=True)
        valid = raw["valid_mask"].to(device, non_blocking=True)
        outputs = model(features, logits, valid)
        for batch_index, events in enumerate(raw["gold_events"]):
            proposals = decode_class_aware_proposals(
                outputs,
                batch_index,
                max_classes=max_classes,
                max_events=max_candidates,
            )
            pairs = adjacent_index_pairs(events)
            totals["scenes"] += 1
            totals["events"] += len(events)
            totals["pairs"] += len(pairs)
            for top_k in top_k_values:
                hits = [
                    event_hit(
                        proposals,
                        event,
                        top_k=top_k,
                        iou_threshold=args.iou_threshold,
                    )
                    for event in events
                ]
                totals[f"event_hits_{top_k}"] += sum(hits)
                totals[f"all_scene_hits_{top_k}"] += int(all(hits))
                totals[f"pair_hits_{top_k}"] += sum(
                    hits[left] and hits[right] for left, right in pairs
                )

    metrics: dict[str, Any] = {
        "scenes": int(totals["scenes"]),
        "events": int(totals["events"]),
        "adjacent_pairs": int(totals["pairs"]),
    }
    for top_k in top_k_values:
        metrics[f"event_label_iou030_recall@{top_k}_↑"] = totals[
            f"event_hits_{top_k}"
        ] / max(totals["events"], 1)
        metrics[f"adjacent_pair_joint_iou030_recall@{top_k}_↑"] = totals[
            f"pair_hits_{top_k}"
        ] / max(totals["pairs"], 1)
        metrics[f"all_events_scene_iou030_recall@{top_k}_↑"] = totals[
            f"all_scene_hits_{top_k}"
        ] / max(totals["scenes"], 1)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_epoch": int(payload["epoch"]),
        "answer_label_used_as_model_input": False,
        "gold_used_only_for_post_inference_scoring": True,
        "iou_threshold": args.iou_threshold,
        "metrics": metrics,
    }
    _atomic_json(receipt, args.output.resolve())
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
