#!/usr/bin/env python3
"""Triage V5 classes using clean-source semantics and unused source reserves."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from mixi_understanding.scripts.train_qces_long_short_semantic_teacher_v2 import LongHeadConfig, LongSemanticHead
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, default=Path("/var/tmp/qces_full188_tiered_realistic_v5/ontology_188.txt"))
    parser.add_argument("--train-manifest", type=Path, default=Path("/var/tmp/qces_full188_tiered_realistic_v5/detector_scene_manifest_tiered_train.jsonl"))
    parser.add_argument("--dev-manifest", type=Path, default=Path("/var/tmp/qces_full188_tiered_realistic_v5/detector_scene_manifest_tiered_dev.jsonl"))
    parser.add_argument("--clean-train-cache", type=Path, default=BASE / "long_short_semantic_teacher_v2/full_source_stats_train.pt")
    parser.add_argument("--clean-dev-cache", type=Path, default=BASE / "long_short_semantic_teacher_v2/full_source_stats_dev.pt")
    parser.add_argument("--teacher-checkpoint", type=Path, default=BASE / "long_short_semantic_teacher_v2/long_short_semantic_teacher_v2_best.pt")
    parser.add_argument("--sourcebank-root", type=Path, default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1/primary_clean")
    parser.add_argument("--output-dir", type=Path, default=BASE / "v5_semantic_sufficiency_audit_v1")
    return parser.parse_args()


def selected_sources(path: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        for event in row["events"]: result[str(event["label"])].add(str(event["source_id"]))
    return result


def available_sources(root: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for manifest in root.rglob("source_bank.jsonl"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            if not bool(row.get("accepted")) or str(row.get("acceptance_tier", "")).lower() != "gold": continue
            result[str(row["label"])].add(str(row["source_id"]))
    return result


@torch.inference_mode()
def clean_predictions(cache_path: Path, checkpoint_path: Path) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = LongSemanticHead(LongHeadConfig(**payload["long_config"]))
    model.load_state_dict(payload["long_model_state_dict"], strict=True); model.eval()
    outputs = []
    for start in range(0, len(cache["label_id"]), 512):
        logits, _ = model(cache["fixed_stats"][start : start + 512].float(), cache["r1_logits"][start : start + 512].float())
        outputs.append(logits)
    return torch.cat(outputs), cache["label_id"].long(), list(map(str, cache["source_id"]))


def main() -> None:
    args = parse_args(); output_dir = args.output_dir.resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    labels = [value.strip() for value in args.ontology.read_text(encoding="utf-8").splitlines() if value.strip()]
    train_selected = selected_sources(args.train_manifest); dev_selected = selected_sources(args.dev_manifest)
    # Only the official-train sourcebank may replenish train/dev.  The eval
    # partition remains locked and is counted separately for auditing only.
    available = available_sources(args.sourcebank_root / "train")
    locked_eval = available_sources(args.sourcebank_root / "eval")
    logits, target, source_ids = clean_predictions(args.clean_dev_cache, args.teacher_checkpoint)
    predicted = logits.argmax(-1); top5 = logits.topk(5, dim=-1).indices
    per_class = []
    for label_id, label in enumerate(labels):
        indices = torch.where(target.eq(label_id))[0]
        correct = predicted[indices].eq(label_id)
        confusion = Counter(int(value) for value in predicted[indices][~correct].tolist())
        selected = train_selected[label] | dev_selected[label]
        reserve = available[label] - selected
        accuracy = float(correct.float().mean()) if len(indices) else 0.0
        if accuracy < 0.60 and len(reserve) >= 20:
            action = "enrich_existing_gold_reserve"
        elif accuracy < 0.60:
            action = "manual_taxonomy_and_audibility_review"
        elif len(train_selected[label]) < 40:
            action = "increase_train_coverage"
        else:
            action = "keep"
        per_class.append({
            "label": label, "label_id": label_id,
            "clean_dev_sources": int(len(indices)),
            "clean_top1_accuracy_\u2191": accuracy,
            "clean_top5_accuracy_\u2191": float((top5[indices] == label_id).any(-1).float().mean()) if len(indices) else 0.0,
            "v5_train_unique_sources": len(train_selected[label]), "v5_dev_unique_sources": len(dev_selected[label]),
            "gold_train_sourcebank_unique_sources": len(available[label]), "unused_gold_train_reserve": len(reserve),
            "locked_gold_eval_sources": len(locked_eval[label]),
            "top_wrong_predictions": [{"label": labels[key], "count": value} for key, value in confusion.most_common(5)],
            "action": action,
        })
    action_counts = Counter(row["action"] for row in per_class)
    receipt: dict[str, Any] = {
        "format": "qces_v5_semantic_sufficiency_audit_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "data/ontology triage only; model accuracy is not used to silently delete test classes",
        "data": {"classes": len(labels), "clean_dev_sources": len(target), "sourcebank_manifests": len(list(args.sourcebank_root.rglob('source_bank.jsonl')))},
        "overall": {
            "clean_top1_accuracy_\u2191": float(predicted.eq(target).float().mean()),
            "clean_top5_accuracy_\u2191": float((top5 == target[:, None]).any(-1).float().mean()),
            "classes_below_0_60_top1": sum(row["clean_top1_accuracy_\u2191"] < 0.60 for row in per_class),
            "classes_below_0_50_top1": sum(row["clean_top1_accuracy_\u2191"] < 0.50 for row in per_class),
            "actions": dict(action_counts),
        },
        "per_class": per_class,
        "decision": "targeted_data_enrichment_before_any_more_model_training",
    }
    _atomic_json(receipt, output_dir / "report.json")
    weakest = sorted(per_class, key=lambda row: (row["clean_top1_accuracy_\u2191"], -row["unused_gold_train_reserve"]))[:40]
    overall = receipt["overall"]
    accuracy_key = "clean_top1_accuracy_\u2191"
    lines = [
        "# V5 semantic sufficiency audit", "",
        f"- Clean top-1: {overall['clean_top1_accuracy_↑']:.3f} ↑; top-5: {overall['clean_top5_accuracy_↑']:.3f} ↑",
        f"- Classes below 0.60 top-1: {overall['classes_below_0_60_top1']}/188",
        f"- Actions: {dict(action_counts)}", "",
        "| Class | Clean top-1 ↑ | Train src | Dev src | Unused Gold reserve | Action |", "|---|---:|---:|---:|---:|---|",
    ]
    lines.extend(f"| {row['label']} | {row[accuracy_key]:.3f} | {row['v5_train_unique_sources']} | {row['v5_dev_unique_sources']} | {row['unused_gold_train_reserve']} | {row['action']} |" for row in weakest)
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"complete": True, "output": str(output_dir), "overall": receipt["overall"], "weakest": weakest[:10]}, indent=2))


if __name__ == "__main__":
    main()
