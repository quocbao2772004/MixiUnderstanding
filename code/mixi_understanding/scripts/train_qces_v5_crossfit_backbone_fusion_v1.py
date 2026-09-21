#!/usr/bin/env python3
"""Cross-fitted classwise log-linear fusion of frozen BEATs and ATST experts.

The fusion has only one mixture coefficient and one bias per class.  Every
reported prediction is made by a calibrator that did not train on that scene.
The locked test is deliberately not loaded by this architecture screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from mixi_understanding.scripts.audit_qces_v5_backbone_complementarity_v1 import infer_atst, infer_beats
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _atomic_torch, _sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
NUM_CLASSES = 188


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats-cache", type=Path, default=BASE / "v5_interval_curriculum_semantic_v3/interval_curriculum_main_dev.pt")
    parser.add_argument("--beats-checkpoint", type=Path, default=BASE / "v5_interval_curriculum_semantic_v3/interval_curriculum_semantic_v3_best.pt")
    parser.add_argument("--atst-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1/atst_features_dev.pt"))
    parser.add_argument("--atst-checkpoint", type=Path, default=BASE / "v5_atst_oracle_semantic_screen_v1/atst_oracle_semantic_head_v1_best.pt")
    parser.add_argument("--output-dir", type=Path, default=BASE / "v5_crossfit_backbone_fusion_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--seed", type=int, default=6701)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class ClasswiseLogLinearFusion(torch.nn.Module):
    """Regularized product-of-experts with 2*C trainable scalars."""

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.alpha_logit = torch.nn.Parameter(torch.zeros(classes))
        self.bias = torch.nn.Parameter(torch.zeros(classes))

    def forward(self, beats: torch.Tensor, atst: torch.Tensor) -> torch.Tensor:
        left = F.log_softmax(beats.float(), dim=-1)
        right = F.log_softmax(atst.float(), dim=-1)
        alpha = self.alpha_logit.sigmoid()
        return alpha * left + (1.0 - alpha) * right + self.bias


def scene_folds(atst_cache: dict[str, Any], folds: int) -> torch.Tensor:
    values = []
    for scene_id, labels in zip(atst_cache["scene_id"], atst_cache["labels"]):
        digest = int(hashlib.sha256(str(scene_id).encode()).hexdigest()[:16], 16)
        values.extend([digest % folds] * int(labels.numel()))
    return torch.tensor(values, dtype=torch.long)


def fit(
    beats: torch.Tensor, atst: torch.Tensor, target: torch.Tensor, indices: torch.Tensor,
    *, epochs: int, learning_rate: float, weight_decay: float, seed: int,
) -> tuple[ClasswiseLogLinearFusion, list[float]]:
    torch.manual_seed(seed)
    model = ClasswiseLogLinearFusion(NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.02)
    history = []
    for _ in range(epochs):
        logits = model(beats[indices], atst[indices])
        loss = F.cross_entropy(logits, target[indices], label_smoothing=0.01)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
        history.append(float(loss.detach()))
    return model.eval(), history


def metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float | int]:
    order = logits.argsort(dim=-1, descending=True)
    return {
        "events": int(target.numel()),
        "top1_accuracy_\u2191": float(order[:, 0].eq(target).float().mean()),
        "top5_accuracy_\u2191": float((order[:, :5] == target[:, None]).any(dim=-1).float().mean()),
        "nll_\u2193": float(F.cross_entropy(logits, target)),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    beats_cache = torch.load(args.beats_cache, map_location="cpu", weights_only=True)
    atst_cache = torch.load(args.atst_cache, map_location="cpu", weights_only=True)
    if list(beats_cache["scene_id"]) != list(atst_cache["scene_id"]): raise ValueError("scene order mismatch")
    beats, target = infer_beats(beats_cache, torch.load(args.beats_checkpoint, map_location="cpu", weights_only=True))
    atst, atst_target = infer_atst(atst_cache, torch.load(args.atst_checkpoint, map_location="cpu", weights_only=True))
    if not torch.equal(target, atst_target): raise ValueError("target mismatch")
    folds = scene_folds(atst_cache, args.folds)
    if len(folds) != len(target): raise ValueError("fold/event mismatch")
    crossfit = torch.empty_like(beats)
    fold_reports = []
    all_indices = torch.arange(len(target))
    for fold in range(args.folds):
        train_indices = all_indices[folds.ne(fold)]; eval_indices = all_indices[folds.eq(fold)]
        model, history = fit(
            beats, atst, target, train_indices, epochs=args.epochs,
            learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=args.seed + fold,
        )
        with torch.inference_mode(): crossfit[eval_indices] = model(beats[eval_indices], atst[eval_indices])
        fold_reports.append({
            "fold": fold, "train_events": int(train_indices.numel()), "eval_events": int(eval_indices.numel()),
            "train_scenes": len({str(scene) for scene, value in zip(atst_cache["scene_id"], range(len(atst_cache["scene_id"]))) if int(hashlib.sha256(str(scene).encode()).hexdigest()[:16], 16) % args.folds != fold}),
            "eval_scenes": sum(int(hashlib.sha256(str(scene).encode()).hexdigest()[:16], 16) % args.folds == fold for scene in atst_cache["scene_id"]),
            "final_train_loss": history[-1], "metrics": metrics(crossfit[eval_indices], target[eval_indices]),
        })
        print(json.dumps(fold_reports[-1], sort_keys=True), flush=True)
    average = 0.5 * F.log_softmax(beats, dim=-1) + 0.5 * F.log_softmax(atst, dim=-1)
    beats_ok = beats.argmax(-1).eq(target); atst_ok = atst.argmax(-1).eq(target)
    crossfit_metrics = metrics(crossfit, target)
    full_model, full_history = fit(
        beats, atst, target, all_indices, epochs=args.epochs,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay, seed=args.seed + args.folds,
    )
    checkpoint_path = output_dir / "classwise_loglinear_fusion_v1.pt"
    _atomic_torch({
        "format": "qces_v5_classwise_loglinear_fusion_checkpoint_v1",
        "model_state_dict": full_model.state_dict(), "classes": NUM_CLASSES,
        "calibration_split": "V5 dev only", "locked_test_used": False,
    }, checkpoint_path)
    gates = {
        "crossfit_top1_ge_0_65": float(crossfit_metrics["top1_accuracy_\u2191"]) >= 0.65,
        "crossfit_improves_both_experts": float(crossfit_metrics["top1_accuracy_\u2191"]) > max(float(beats_ok.float().mean()), float(atst_ok.float().mean())),
    }
    receipt = {
        "format": "qces_v5_crossfit_backbone_fusion_training_receipt_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "scene-grouped 5-fold cross-fitted classwise log-linear product of frozen BEATs and ATST experts",
        "qa_question_or_answer_used_as_input": False,
        "locked_test_loaded_or_used": False,
        "trainable_parameters": sum(value.numel() for value in full_model.parameters()),
        "data": {"scenes": len(atst_cache["scene_id"]), "events": len(target), "classes": NUM_CLASSES, "folds": args.folds},
        "single_experts": {"beats": metrics(beats, target), "atst": metrics(atst, target)},
        "fixed_equal_logprob_average": metrics(average, target),
        "either_expert_oracle_ceiling": {"top1_accuracy_\u2191": float((beats_ok | atst_ok).float().mean())},
        "crossfit_fusion": crossfit_metrics,
        "fold_reports": fold_reports,
        "gates": gates,
        "decision": "export_locked_test_features_and_evaluate_once" if all(gates.values()) else "close_dual_backbone_fusion",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "full_calibration_final_loss": full_history[-1],
        "arguments": vars(args) | {"beats_cache": str(args.beats_cache), "beats_checkpoint": str(args.beats_checkpoint), "atst_cache": str(args.atst_cache), "atst_checkpoint": str(args.atst_checkpoint), "output_dir": str(args.output_dir)},
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "crossfit_fusion": crossfit_metrics, "gates": gates, "decision": receipt["decision"]}, indent=2))


if __name__ == "__main__":
    main()
