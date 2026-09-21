#!/usr/bin/env python3
"""Evaluate the frozen BEATs long-view clean teacher on matched official eval."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import torch

from mixi_understanding.scripts.audit_qces_full_source_semantic_ceiling_v1 import load_pairs
from mixi_understanding.scripts.audit_qces_v5_targeted_logit_calibration_v1 import compact_metric
from mixi_understanding.scripts.train_qces_long_short_semantic_teacher_v2 import (
    LongHeadConfig,
    LongSemanticHead,
    export_long_cache,
)
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import load_model, load_ontology, make_device
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _atomic_torch, _sha256_file


FORMAT = "qces_v5_long_teacher_matched_eval_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    data = Path("/var/tmp/qces_v5_matched_eval_v1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-manifest", type=Path, default=data / "detector_scene_manifest_matched_eval.jsonl")
    parser.add_argument("--component-manifest", type=Path, default=data / "event_components_matched_eval.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--teacher-checkpoint", type=Path, default=base / "long_short_semantic_teacher_v2/long_short_semantic_teacher_v2_best.pt")
    parser.add_argument("--cache", type=Path, default=Path("/var/tmp/qces_v5_long_teacher_matched_eval_v1/full_source_stats.pt"))
    parser.add_argument("--output", type=Path, default=base / "long_short_semantic_teacher_v2/matched_eval_v1.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained-checkpoint", default="BEATs_strong_1")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    labels = load_ontology(args.ontology.resolve())
    rows = load_pairs(
        args.component_manifest.resolve(), args.scene_manifest.resolve(), labels
    )
    device = make_device(args.device)
    detector_payload = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if list(detector_payload.get("labels") or []) != labels:
        raise ValueError("detector ontology mismatch")
    backbone = load_model(len(labels), args.pretrained_checkpoint, device)
    backbone.load_state_dict(detector_payload["model_state_dict"], strict=True)
    backbone.eval().requires_grad_(False)
    cache = export_long_cache(
        backbone, rows, device=device, batch_size=args.batch_size,
        num_workers=args.num_workers, amp=args.amp, split="matched_eval",
    )
    cache_path = args.cache.resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch(cache, cache_path)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload = torch.load(args.teacher_checkpoint.resolve(), map_location="cpu", weights_only=True)
    model = LongSemanticHead(LongHeadConfig(**payload["long_config"])).to(device)
    model.load_state_dict(payload["long_model_state_dict"], strict=True)
    model.eval()
    scores: list[torch.Tensor] = []
    for begin in range(0, len(rows), 256):
        logits, _ = model(
            cache["fixed_stats"][begin : begin + 256].to(device),
            cache["r1_logits"][begin : begin + 256].to(device),
        )
        scores.append(logits.cpu())
    teacher_scores = torch.cat(scores)
    targets = cache["label_id"].long()
    baseline_scores = cache["r1_logits"].float()
    baseline = compact_metric(baseline_scores, targets)
    teacher = compact_metric(teacher_scores, targets)
    gates = {
        "matched_top1_ge_0_70": teacher["top1_accuracy_↑"] >= 0.70,
        "matched_top5_ge_0_90": teacher["top5_accuracy_↑"] >= 0.90,
        "improves_frozen_r1_top1_by_0_05": teacher["top1_accuracy_↑"] >= baseline["top1_accuracy_↑"] + 0.05,
    }
    report = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one-shot matched official-eval full clean source; no mixture or QA answer input",
        "data": {"unique_sources": len(rows), "classes": len(set(row.label for row in rows))},
        "frozen_r1_baseline": baseline,
        "long_teacher": teacher,
        "delta": {
            "top1": teacher["top1_accuracy_↑"] - baseline["top1_accuracy_↑"],
            "top5": teacher["top5_accuracy_↑"] - baseline["top5_accuracy_↑"],
        },
        "gates": {"passed": all(gates.values()), "checks": gates},
        "decision": "expand_clean_teacher_training_diversity_then_distill" if all(gates.values()) else "clean_teacher_domain_generalization_insufficient",
        "artifacts": {
            "scene_manifest_sha256": _sha256_file(args.scene_manifest.resolve()),
            "component_manifest_sha256": _sha256_file(args.component_manifest.resolve()),
            "cache_sha256": _sha256_file(cache_path),
            "teacher_checkpoint_sha256": _sha256_file(args.teacher_checkpoint.resolve()),
            "detector_checkpoint_sha256": _sha256_file(args.detector_checkpoint.resolve()),
        },
    }
    _atomic_json(report, args.output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
