#!/usr/bin/env python3
"""Corrected held-out evaluation for a frozen target-invariant checkpoint.

The original run accidentally treated compact component files as already
mixture-aligned.  This script only remeasures the frozen checkpoint: compact
components are reinserted at their event onset before ATST inference.  It does
not train, tune, or select a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    load_scene_manifest,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _sha256_file
from mixi_understanding.scripts.train_qces_v5_oracle_group_experts_v1 import (
    group_lookup,
    ordering_metrics,
)
from mixi_understanding.scripts.train_qces_v5_target_invariant_atst_v1 import (
    TargetInvariantExperts,
    build_backbone,
    component_rows,
    evaluate_rows,
    old_expert_baseline,
    stripped_eval,
)


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    output = base / "v5_target_invariant_atst_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=output / "target_invariant_atst_v1_best.pt")
    parser.add_argument("--receipt", type=Path, default=output / "receipt.json")
    parser.add_argument("--matched-manifest", type=Path, default=Path("/var/tmp/qces_v5_matched_eval_v1/detector_scene_manifest_matched_eval.jsonl"))
    parser.add_argument("--previous-experts", type=Path, default=base / "v5_oracle_group_experts_v1/oracle_group_experts_v1_best.pt")
    parser.add_argument("--previous-matched-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_matched_eval_v1/atst_features_matched_eval.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    labels = list(payload["labels"])
    groups = [list(map(int, group)) for group in payload["groups"]]
    label_to_group, _ = group_lookup(groups, len(labels))
    label_to_id = {label: index for index, label in enumerate(labels)}
    previous_path = args.previous_experts.resolve()
    previous = torch.load(previous_path, map_location="cpu", weights_only=False)
    device = make_device(args.device)
    backbone = build_backbone(len(labels), device)
    state = backbone.model.state_dict()
    state.update(payload["atst_trainable_state_dict"])
    backbone.model.load_state_dict(state, strict=True)
    model = TargetInvariantExperts(
        int(payload["input_dim"]), int(payload["hidden_dim"]), groups, previous["model_state_dict"]
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    matched_rows = load_scene_manifest(args.matched_manifest.resolve(), label_to_id)
    mixture = evaluate_rows(
        backbone, model, matched_rows, groups, label_to_group, device,
        batch_size=args.batch_size, num_workers=args.num_workers, amp=args.amp,
    )
    clean = evaluate_rows(
        backbone, model, component_rows(matched_rows), groups, label_to_group, device,
        batch_size=args.batch_size, num_workers=args.num_workers, amp=args.amp,
        component_canvas=True,
    )
    old = old_expert_baseline(previous_path, args.previous_matched_cache.resolve(), device)
    if not torch.equal(old["targets"], mixture["targets"]):
        raise RuntimeError("old/new matched event order differs")
    heavy = mixture["heavy_mask"]
    old_heavy = ordering_metrics(old["ordering"][heavy], old["targets"][heavy])
    heavy_delta = float(
        mixture["heavy_overlap_oracle_group"]["top1_accuracy_↑"]
        - old_heavy["top1_accuracy_↑"]
    )
    checks = {
        "clean_matched_oracle_group_top1_ge_0_75": float(clean["oracle_group"]["top1_accuracy_↑"]) >= 0.75,
        "mixture_matched_oracle_group_top1_ge_0_68": float(mixture["oracle_group"]["top1_accuracy_↑"]) >= 0.68,
        "mixture_matched_oracle_group_top5_ge_0_93": float(mixture["oracle_group"]["top5_accuracy_↑"]) >= 0.93,
        "heavy_overlap_oracle_group_gain_ge_0_08": heavy_delta >= 0.08,
        "deployable_router_beats_flat_atst_by_0_05": float(mixture["predicted_router"]["top1_accuracy_↑"]) >= 0.51625 + 0.05,
    }
    receipt_path = args.receipt.resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    prior_checks = receipt.get("gates", {}).get("checks", {})
    structural = {
        key: bool(value)
        for key, value in prior_checks.items()
        if key in {
            "source_pool_passed", "remix_waveforms_valid", "backbone_gradient_nonzero",
            "frozen_backbone_has_no_gradient", "exact_trainable_block_contract",
        }
    }
    all_checks = {**structural, **checks}
    receipt["measurement_correction"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "compact component files must be reinserted at mixture onset before applying mixture-aligned intervals",
        "training_changed": False,
        "checkpoint_selection_changed": False,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    receipt["matched_held_out"] = {
        "new_mixture": stripped_eval(mixture),
        "new_clean_components": stripped_eval(clean),
        "previous_oracle_group_experts": old["metrics"],
        "previous_oracle_group_experts_heavy_overlap_ge_0_30": old_heavy,
        "heavy_overlap_oracle_group_top1_delta": heavy_delta,
    }
    receipt["gates"] = {"passed": all(all_checks.values()), "checks": all_checks}
    receipt["decision"] = (
        "accept_target_invariant_representation"
        if receipt["gates"]["passed"]
        else "stop_188_fine_grained_and_revise_ontology_or_hierarchy"
    )
    _atomic_json(receipt, receipt_path)
    print(json.dumps({
        "complete": True,
        "training_changed": False,
        "checkpoint_selection_changed": False,
        "matched": receipt["matched_held_out"],
        "gates": receipt["gates"],
        "decision": receipt["decision"],
        "receipt": str(receipt_path),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
