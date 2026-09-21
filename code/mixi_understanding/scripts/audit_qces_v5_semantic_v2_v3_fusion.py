#!/usr/bin/env python3
"""No-training complementarity audit for V2 slot and V3 interval semantics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _device, _sha256_file
from mixi_understanding.scripts.train_qces_v4_slot_semantic_head_v1 import NONE_LABEL, ResidualSlotSemanticHead
from mixi_understanding.scripts.train_qces_v5_rehearsal_slot_semantic_head_v2 import _event_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAT = "qces_v5_semantic_v2_v3_fusion_audit_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-dir", type=Path, default=base / "v5_rehearsal_slot_semantic_head_v2")
    parser.add_argument("--v3-dir", type=Path, default=base / "v5_interval_curriculum_semantic_v3")
    parser.add_argument("--output", type=Path, default=base / "v5_interval_curriculum_semantic_v3/v2_v3_fusion_audit.json")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


@torch.inference_mode()
def logits(model: ResidualSlotSemanticHead, value: torch.Tensor, device: torch.device) -> torch.Tensor:
    flat = value.reshape(-1, value.shape[-1])
    rows = []
    model.eval()
    for start in range(0, flat.shape[0], 512):
        rows.append(model(flat[start : start + 512].to(device)).cpu())
    return torch.cat(rows).reshape(value.shape[0], value.shape[1], NONE_LABEL + 1)


def harmonic(left: float, right: float) -> float:
    return 2.0 * left * right / max(left + right, 1e-12)


def score(main: Mapping[str, Any], stress: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "main": dict(main),
        "stress": dict(stress),
        "harmonic_joint_f1": harmonic(float(main["joint_label_iou50_f1_\u2191"]), float(stress["joint_label_iou50_f1_\u2191"])),
        "harmonic_label_top1": harmonic(float(main["label_top1_given_iou50_\u2191"]), float(stress["label_top1_given_iou50_\u2191"])),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = _device(args.device)
    v2_dir = args.v2_dir.resolve(); v3_dir = args.v3_dir.resolve()
    v2_checkpoint_path = v2_dir / "slot_semantic_head_v2_best.pt"
    v3_checkpoint_path = v3_dir / "interval_curriculum_semantic_v3_best.pt"
    v2_saved = torch.load(v2_checkpoint_path, map_location="cpu", weights_only=True)
    v3_saved = torch.load(v3_checkpoint_path, map_location="cpu", weights_only=True)
    v2_model = ResidualSlotSemanticHead(v2_saved["input_dim"], v2_saved["hidden_dim"], NONE_LABEL, v2_saved["dropout"])
    v2_model.load_state_dict(v2_saved["model_state_dict"], strict=True); v2_model.to(device)
    v3_model = ResidualSlotSemanticHead(v3_saved["input_dim"], v3_saved["hidden_dim"], NONE_LABEL, v3_saved["dropout"])
    v3_model.load_state_dict(v3_saved["model_state_dict"], strict=True); v3_model.to(device)
    domains = {}
    for domain in ("main", "stress"):
        old = torch.load(v2_dir / f"frozen_slots_{domain}_dev.pt", map_location="cpu", weights_only=True)
        rich = torch.load(v3_dir / f"interval_curriculum_{domain}_dev.pt", map_location="cpu", weights_only=True)
        if list(old["scene_id"]) != list(rich["scene_id"]):
            raise ValueError(f"scene order mismatch: {domain}")
        if not torch.allclose(old["objectness"].float(), rich["objectness"].float(), atol=1e-3):
            raise ValueError(f"objectness mismatch: {domain}")
        domains[domain] = {
            "cache": rich,
            "v2": logits(v2_model, old["slot_input"], device),
            "v3": logits(v3_model, rich["predicted_input"], device),
        }
    objectness_threshold = float(v2_saved["objectness_threshold"])
    rows = []
    for step in range(11):
        alpha = step / 10.0
        metrics = {}
        for domain, values in domains.items():
            fused = (1.0 - alpha) * values["v2"] + alpha * values["v3"]
            metrics[domain] = _event_metrics(values["cache"], fused, objectness_threshold=objectness_threshold, learned=True)
        rows.append({"alpha_v3": alpha, **score(metrics["main"], metrics["stress"])})
    best = max(rows, key=lambda row: (row["harmonic_joint_f1"], row["harmonic_label_top1"]))
    baseline = rows[0]
    complementarity = {}
    for domain, values in domains.items():
        cache = values["cache"]
        # Slot-level agreement is diagnostic only; end-to-end event metrics
        # above remain the selection criterion.
        keep = cache["objectness"].float() >= objectness_threshold
        target = cache["target_slot_labels"].long()
        valid = keep & target.lt(NONE_LABEL) & cache["target_slot_ious"].float().ge(0.5)
        p2 = values["v2"].argmax(dim=-1); p3 = values["v3"].argmax(dim=-1)
        complementarity[domain] = {
            "matched_slots": int(valid.sum()),
            "v2_correct_v3_wrong": int((valid & p2.eq(target) & ~p3.eq(target)).sum()),
            "v2_wrong_v3_correct": int((valid & ~p2.eq(target) & p3.eq(target)).sum()),
            "both_correct": int((valid & p2.eq(target) & p3.eq(target)).sum()),
            "both_wrong": int((valid & ~p2.eq(target) & ~p3.eq(target)).sum()),
        }
    gates = {
        "harmonic_joint_gain_ge_0_005": float(best["harmonic_joint_f1"]) >= float(baseline["harmonic_joint_f1"]) + 0.005,
        "stress_top1_not_worse_than_v2": float(best["stress"]["label_top1_given_iou50_\u2191"]) >= float(baseline["stress"]["label_top1_given_iou50_\u2191"]),
        "main_top1_improves_v2": float(best["main"]["label_top1_given_iou50_\u2191"]) > float(baseline["main"]["label_top1_given_iou50_\u2191"]),
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_performed": False,
        "qa_question_or_answer_used_as_input": False,
        "rows": rows,
        "v2_baseline": baseline,
        "best": best,
        "complementarity": complementarity,
        "gates": gates,
        "decision": "train_gated_residual_fusion" if all(gates.values()) else "do_not_train_fusion",
        "artifacts": {
            "v2_checkpoint_sha256": _sha256_file(v2_checkpoint_path),
            "v3_checkpoint_sha256": _sha256_file(v3_checkpoint_path),
        },
    }
    _atomic_json(receipt, args.output.resolve())
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
