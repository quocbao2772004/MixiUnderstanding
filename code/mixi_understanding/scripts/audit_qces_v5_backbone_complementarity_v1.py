#!/usr/bin/env python3
"""Paired oracle-span error audit for the frozen BEATs and ATST semantic heads."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from mixi_understanding.scripts.train_qces_local_semantic_r3 import LocalSemanticHead
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json
from mixi_understanding.scripts.train_qces_v4_slot_semantic_head_v1 import ResidualSlotSemanticHead
from mixi_understanding.scripts.train_qces_v5_atst_oracle_semantic_screen_v1 import collect_spans


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
NUM_CLASSES = 188


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, default=Path("/var/tmp/qces_full188_tiered_realistic_v5/ontology_188.txt"))
    parser.add_argument("--manifest", type=Path, default=Path("/var/tmp/qces_full188_tiered_realistic_v5/detector_scene_manifest_tiered_dev.jsonl"))
    parser.add_argument("--beats-cache", type=Path, default=BASE / "v5_interval_curriculum_semantic_v3/interval_curriculum_main_dev.pt")
    parser.add_argument("--beats-checkpoint", type=Path, default=BASE / "v5_interval_curriculum_semantic_v3/interval_curriculum_semantic_v3_best.pt")
    parser.add_argument("--atst-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1/atst_features_dev.pt"))
    parser.add_argument("--atst-checkpoint", type=Path, default=BASE / "v5_atst_oracle_semantic_screen_v1/atst_oracle_semantic_head_v1_best.pt")
    parser.add_argument("--output", type=Path, default=BASE / "v5_atst_oracle_semantic_screen_v1/backbone_complementarity_audit_v1.json")
    return parser.parse_args()


def infer_beats(cache: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    model = ResidualSlotSemanticHead(
        int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), NUM_CLASSES, float(checkpoint["dropout"])
    )
    # Stage 1 is the checkpoint selected only with oracle intervals.  Using it
    # avoids mixing predicted-slot rehearsal effects into this architecture audit.
    model.load_state_dict(checkpoint["stage1_state_dict"], strict=True)
    model.eval()
    keep = cache["oracle_exact"]
    values = cache["oracle_input"][keep].float()
    target = cache["oracle_labels"][keep].long()
    output = []
    with torch.inference_mode():
        for start in range(0, len(values), 256):
            output.append(model(values[start : start + 256])[:, :NUM_CLASSES])
    return torch.cat(output), target


def infer_atst(cache: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    model = LocalSemanticHead(int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]), NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    outputs, targets = [], []
    with torch.inference_mode():
        for scene_index in range(int(cache["features"].shape[0])):
            value = cache["features"][scene_index : scene_index + 1].float()
            spans, mask, target = collect_spans(
                value, [cache["intervals"][scene_index]], [cache["labels"][scene_index]], jitter=0, training=False
            )
            outputs.append(model(spans, mask)); targets.append(target)
    return torch.cat(outputs), torch.cat(targets)


def flatten_metadata(manifest: Path, expected_scene_ids: list[str]) -> list[dict[str, Any]]:
    by_scene: dict[str, Mapping[str, Any]] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line); by_scene[str(row["scene_id"])] = row
    result = []
    for scene_id in expected_scene_ids:
        row = by_scene[scene_id]
        for event in row["events"]:
            if str(event.get("event_kind", "semantic")) != "semantic":
                continue
            result.append({
                "scene_id": scene_id,
                "event_id": str(event["event_id"]),
                "label_id": int(event["label_id"]),
                "active_sir_db": float(event.get("active_sir_db", 0.0)),
                "overlap_fraction": float(event.get("overlap_fraction", 0.0)),
                "duration_seconds": float(event["offset_seconds"]) - float(event["onset_seconds"]),
                "crop_was_capped": bool(event.get("crop_was_capped", False)),
                "source_path": str(event.get("source_path", "")),
            })
    return result


def metric_group(indices: list[int], target: torch.Tensor, beats: torch.Tensor, atst: torch.Tensor) -> dict[str, Any]:
    index = torch.tensor(indices, dtype=torch.long)
    truth = target[index]; left = beats[index]; right = atst[index]
    left_ok = left.eq(truth); right_ok = right.eq(truth)
    return {
        "events": len(indices),
        "beats_top1_accuracy_\u2191": float(left_ok.float().mean()),
        "atst_top1_accuracy_\u2191": float(right_ok.float().mean()),
        "either_backbone_oracle_top1_\u2191": float((left_ok | right_ok).float().mean()),
        "both_wrong_rate_\u2193": float((~left_ok & ~right_ok).float().mean()),
    }


def binned(
    metadata: list[Mapping[str, Any]], target: torch.Tensor, beats: torch.Tensor, atst: torch.Tensor,
    assign: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata): groups[assign(row)].append(index)
    return {key: metric_group(value, target, beats, atst) for key, value in sorted(groups.items())}


def top_confusions(predicted: torch.Tensor, target: torch.Tensor, labels: list[str], limit: int = 25) -> list[dict[str, Any]]:
    counts = Counter((int(gold), int(pred)) for gold, pred in zip(target, predicted) if gold != pred)
    return [
        {"gold": labels[gold], "predicted": labels[pred], "count": count}
        for (gold, pred), count in counts.most_common(limit)
    ]


def main() -> None:
    args = parse_args()
    labels = [line.strip() for line in args.ontology.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != NUM_CLASSES: raise ValueError(f"expected {NUM_CLASSES} labels, found {len(labels)}")
    beats_cache = torch.load(args.beats_cache, map_location="cpu", weights_only=True)
    atst_cache = torch.load(args.atst_cache, map_location="cpu", weights_only=True)
    if list(beats_cache["scene_id"]) != list(atst_cache["scene_id"]): raise ValueError("scene order mismatch")
    beats_logits, beats_target = infer_beats(beats_cache, torch.load(args.beats_checkpoint, map_location="cpu", weights_only=True))
    atst_logits, atst_target = infer_atst(atst_cache, torch.load(args.atst_checkpoint, map_location="cpu", weights_only=True))
    if not torch.equal(beats_target, atst_target): raise ValueError("event/label order mismatch")
    metadata = flatten_metadata(args.manifest, list(beats_cache["scene_id"]))
    if [int(row["label_id"]) for row in metadata] != beats_target.tolist(): raise ValueError("manifest/cache event mismatch")
    beats_pred = beats_logits.argmax(dim=-1); atst_pred = atst_logits.argmax(dim=-1)
    beats_ok = beats_pred.eq(beats_target); atst_ok = atst_pred.eq(atst_target)
    consensus_wrong = Counter(
        (int(gold), int(left)) for gold, left, right in zip(beats_target, beats_pred, atst_pred) if gold != left and left == right
    )
    all_indices = list(range(len(metadata)))
    receipt = {
        "format": "qces_v5_backbone_complementarity_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "paired oracle-span diagnostic on the fixed V5 dev set; either-backbone is an oracle ceiling, not a deployable score",
        "data": {"scenes": len(beats_cache["scene_id"]), "events": len(metadata), "classes": len(labels)},
        "overall": metric_group(all_indices, beats_target, beats_pred, atst_pred) | {
            "both_correct": int((beats_ok & atst_ok).sum()),
            "beats_only_correct": int((beats_ok & ~atst_ok).sum()),
            "atst_only_correct": int((~beats_ok & atst_ok).sum()),
            "both_wrong": int((~beats_ok & ~atst_ok).sum()),
        },
        "by_active_sir_db": binned(metadata, beats_target, beats_pred, atst_pred, lambda x: "<0" if x["active_sir_db"] < 0 else "0-5" if x["active_sir_db"] < 5 else "5-10" if x["active_sir_db"] < 10 else "10-20" if x["active_sir_db"] < 20 else ">=20"),
        "by_overlap_fraction": binned(metadata, beats_target, beats_pred, atst_pred, lambda x: "zero" if x["overlap_fraction"] == 0 else "(0,0.25]" if x["overlap_fraction"] <= 0.25 else ">0.25"),
        "by_duration_seconds": binned(metadata, beats_target, beats_pred, atst_pred, lambda x: "<=0.5" if x["duration_seconds"] <= 0.5 else "(0.5,1.5]" if x["duration_seconds"] <= 1.5 else ">1.5"),
        "by_crop_capped": binned(metadata, beats_target, beats_pred, atst_pred, lambda x: str(x["crop_was_capped"]).lower()),
        "beats_top_confusions": top_confusions(beats_pred, beats_target, labels),
        "atst_top_confusions": top_confusions(atst_pred, atst_target, labels),
        "consensus_wrong_confusions": [
            {"gold": labels[gold], "predicted": labels[pred], "count": count}
            for (gold, pred), count in consensus_wrong.most_common(25)
        ],
        "per_class": [],
    }
    for label_id, label in enumerate(labels):
        indices = torch.where(beats_target.eq(label_id))[0].tolist()
        receipt["per_class"].append({"label": label} | metric_group(indices, beats_target, beats_pred, atst_pred))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(receipt, args.output)
    print(json.dumps({"complete": True, "output": str(args.output), "overall": receipt["overall"], "consensus_wrong_top5": receipt["consensus_wrong_confusions"][:5]}, indent=2))


if __name__ == "__main__":
    main()
