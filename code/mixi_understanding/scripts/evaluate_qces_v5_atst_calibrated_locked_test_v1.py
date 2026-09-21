#!/usr/bin/env python3
"""One-shot locked-test evaluation of frozen V5 targeted logit calibration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import torch
from torch.utils.data import DataLoader

import models.prediction_wrapper as prediction_wrapper_module
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.prediction_wrapper import PredictionsWrapper
from mixi_understanding.scripts.audit_qces_v5_targeted_logit_calibration_v1 import compact_metric
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    load_ontology,
    load_scene_manifest,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _atomic_torch, _sha256_file
from mixi_understanding.scripts.train_qces_v5_atst_oracle_semantic_screen_v1 import (
    CachedSceneDataset,
    cached_collate,
    export_cache,
)
from mixi_understanding.scripts.train_qces_v5_atst_targeted_enrichment_v1 import (
    load_head,
    predictions,
)


FORMAT = "qces_v5_atst_calibrated_locked_test_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    data = PROJECT_ROOT / "outputs/qces_full188_tiered_realistic_v5"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, default=data / "detector_scene_manifest_locked_test_filtered.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--calibration", type=Path, default=base / "v5_atst_targeted_enrichment_v1/logit_calibration_audit_v1.json")
    parser.add_argument("--enrichment-build-receipt", type=Path, default=Path("/var/tmp/qces_v5_targeted_enrichment_v1/build_receipt.json"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=base / "v5_atst_oracle_semantic_screen_v1/atst_oracle_semantic_head_v1_best.pt")
    parser.add_argument("--enriched-checkpoint", type=Path, default=base / "v5_atst_targeted_enrichment_v1/atst_targeted_enrichment_v1_best.pt")
    parser.add_argument("--test-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_locked_test_v1/atst_features_test.pt"))
    parser.add_argument("--output", type=Path, default=base / "v5_atst_targeted_enrichment_v1/locked_test_v1.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--evaluation-name", default="locked_test")
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def scene_bootstrap(
    base_correct: np.ndarray,
    fused_correct: np.ndarray,
    scene_index: np.ndarray,
    *,
    num_scenes: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, num_scenes, size=num_scenes)
        weights = np.bincount(sampled, minlength=num_scenes)
        event_weights = weights[scene_index]
        denominator = event_weights.sum()
        values[index] = (
            ((fused_correct - base_correct) * event_weights).sum() / max(denominator, 1)
        )
    return {
        "replicates": replicates,
        "mean_delta": float(values.mean()),
        "ci95_lower": float(np.quantile(values, 0.025)),
        "ci95_upper": float(np.quantile(values, 0.975)),
        "probability_delta_gt_0": float((values > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    calibration_path = args.calibration.resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("decision") != "freeze_calibration_then_evaluate_locked_test":
        raise ValueError("dev calibration was not frozen for locked-test evaluation")
    alpha = float(calibration["selected"]["alpha"])
    weak_bias = float(calibration["selected"]["weak_shared_bias"])
    build_receipt = json.loads(args.enrichment_build_receipt.resolve().read_text(encoding="utf-8"))
    if build_receipt["fixed_evaluation"]["hard_identity_overlap"] != 0:
        raise ValueError("enrichment overlaps fixed evaluation")

    labels = load_ontology(args.ontology.resolve())
    label_map = {label: index for index, label in enumerate(labels)}
    weak_labels = list(build_receipt["weak_labels"])
    weak_ids = {label_map[label] for label in weak_labels}
    other_ids = set(range(len(labels))) - weak_ids
    test_manifest = args.test_manifest.resolve()
    rows = load_scene_manifest(test_manifest, label_map)
    device = make_device(args.device)

    prediction_wrapper_module.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
    backbone = PredictionsWrapper(
        ATSTWrapper(), checkpoint="ATST-F_strong_1",
        n_classes_strong=len(labels), n_classes_weak=len(labels),
        seq_model_type=None, head_type="linear",
    ).to(device)
    backbone.eval().requires_grad_(False)
    cache = export_cache(
        backbone, rows, device, batch_size=args.feature_batch_size,
        num_workers=args.num_workers, amp=args.amp, split=args.evaluation_name,
    )
    cache_path = args.test_cache.resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch(cache, cache_path)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    loader = DataLoader(
        CachedSceneDataset(cache), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=cached_collate,
    )
    baseline, base_payload = load_head(args.baseline_checkpoint.resolve(), device)
    enriched, enrich_payload = load_head(args.enriched_checkpoint.resolve(), device)
    if list(base_payload["labels"]) != labels or list(enrich_payload["labels"]) != labels:
        raise ValueError("checkpoint ontology mismatch")
    base_scores, targets = predictions(baseline, loader, device)
    enrich_scores, targets_enriched = predictions(enriched, loader, device)
    if not torch.equal(targets, targets_enriched):
        raise RuntimeError("test target order changed")
    weak_index = sorted(weak_ids)
    fused_scores = base_scores.clone()
    fused_scores[:, weak_index] = (
        (1.0 - alpha) * base_scores[:, weak_index]
        + alpha * enrich_scores[:, weak_index]
        + weak_bias
    )

    result = {
        "baseline": {
            "all": compact_metric(base_scores, targets),
            "weak16": compact_metric(base_scores, targets, weak_ids),
            "other172": compact_metric(base_scores, targets, other_ids),
        },
        "frozen_calibrated": {
            "all": compact_metric(fused_scores, targets),
            "weak16": compact_metric(fused_scores, targets, weak_ids),
            "other172": compact_metric(fused_scores, targets, other_ids),
        },
    }
    delta = {
        scope: {
            key: result["frozen_calibrated"][scope][key] - result["baseline"][scope][key]
            for key in ("top1_accuracy_↑", "top5_accuracy_↑", "macro_top1_accuracy_↑")
        }
        for scope in ("all", "weak16", "other172")
    }
    events_per_scene = [len(value) for value in cache["labels"]]
    scene_index = np.concatenate([
        np.full(count, index, dtype=np.int64) for index, count in enumerate(events_per_scene)
    ])
    base_correct = base_scores.argmax(1).eq(targets).numpy().astype(np.float64)
    fused_correct = fused_scores.argmax(1).eq(targets).numpy().astype(np.float64)
    bootstrap = scene_bootstrap(
        base_correct, fused_correct, scene_index, num_scenes=len(rows),
        replicates=args.bootstrap_replicates, seed=args.seed,
    )
    report: dict[str, Any] = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": f"one-shot {args.evaluation_name}; oracle event intervals; semantic diagnostic, not end-to-end QA",
        "frozen_before_test": {
            "alpha": alpha,
            "weak_shared_bias": weak_bias,
            "weak_classes": len(weak_ids),
            "nonweak_classes": len(other_ids),
        },
        "data": {
            "scenes": len(rows),
            "events": int(targets.numel()),
            "classes": len(labels),
            "enrichment_test_hard_identity_overlap": 0,
        },
        "metrics": result,
        "delta": delta,
        "scene_bootstrap_top1_delta": bootstrap,
        "interpretation": (
            "frozen calibration generalized to locked test"
            if delta["all"]["top1_accuracy_↑"] > 0
            else "dev calibration did not improve locked test"
        ),
        "artifacts": {
            "test_manifest_sha256": _sha256_file(test_manifest),
            "test_cache_sha256": _sha256_file(cache_path),
            "calibration_sha256": _sha256_file(calibration_path),
            "baseline_checkpoint_sha256": _sha256_file(args.baseline_checkpoint.resolve()),
            "enriched_checkpoint_sha256": _sha256_file(args.enriched_checkpoint.resolve()),
        },
    }
    _atomic_json(report, args.output.resolve())
    print(json.dumps({
        "baseline": result["baseline"],
        "frozen_calibrated": result["frozen_calibrated"],
        "delta": delta,
        "bootstrap": bootstrap,
        "interpretation": report["interpretation"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
