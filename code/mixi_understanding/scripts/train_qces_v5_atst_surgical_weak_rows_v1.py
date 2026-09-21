#!/usr/bin/env python3
"""Adapt only the 16 weak ATST classifier rows with full V5 rehearsal.

This is the anti-forgetting follow-up to targeted-enrichment v1.  The complete
baseline head is restored; all shared representation layers and 172 output
rows are immutable.  Only the final linear rows for the 16 audited classes may
change.  Original V5 examples remain in every epoch as negative/positive
rehearsal, while the new Gold tier supplies additional positives.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mixi_understanding.scripts.train_qces_local_semantic_r3 import LocalSemanticHead
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _atomic_torch, _sha256_file
from mixi_understanding.scripts.train_qces_v5_atst_oracle_semantic_screen_v1 import (
    CachedSceneDataset,
    cached_collate,
    collect_spans,
)
from mixi_understanding.scripts.train_qces_v5_atst_targeted_enrichment_v1 import (
    combine_caches,
    evaluate_slices,
    load_head,
)


FORMAT = "qces_v5_atst_surgical_weak_rows_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-cache-dir", type=Path, default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1"))
    parser.add_argument("--enrichment-root", type=Path, default=Path("/var/tmp/qces_v5_targeted_enrichment_v1"))
    parser.add_argument("--enrichment-cache", type=Path, default=Path("/var/tmp/qces_v5_targeted_enrichment_v1/atst_features_enrich.pt"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=base / "v5_atst_oracle_semantic_screen_v1/atst_oracle_semantic_head_v1_best.pt")
    parser.add_argument("--output-dir", type=Path, default=base / "v5_atst_surgical_weak_rows_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6604)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--boundary-jitter-frames", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    build_receipt_path = args.enrichment_root.resolve() / "build_receipt.json"
    build_receipt = json.loads(build_receipt_path.read_text(encoding="utf-8"))
    if not build_receipt.get("quality_gates", {}).get("passed"):
        raise ValueError("targeted enrichment did not pass build gates")
    weak_labels = list(build_receipt["weak_labels"])

    baseline_checkpoint = args.baseline_checkpoint.resolve()
    head, baseline_payload = load_head(baseline_checkpoint, device)
    labels = list(baseline_payload["labels"])
    label_to_id = {label: index for index, label in enumerate(labels)}
    weak_ids = {label_to_id[label] for label in weak_labels}
    all_ids = set(range(len(labels)))

    base_train_path = args.v5_cache_dir.resolve() / "atst_features_train.pt"
    dev_path = args.v5_cache_dir.resolve() / "atst_features_dev.pt"
    enrich_path = args.enrichment_cache.resolve()
    base_train = torch.load(base_train_path, map_location="cpu", weights_only=False)
    dev_cache = torch.load(dev_path, map_location="cpu", weights_only=False)
    enrich_cache = torch.load(enrich_path, map_location="cpu", weights_only=False)
    combined = combine_caches(base_train, enrich_cache)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        CachedSceneDataset(combined),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=cached_collate,
    )
    dev_loader = DataLoader(
        CachedSceneDataset(dev_cache),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=cached_collate,
    )
    baseline_metrics = evaluate_slices(head, dev_loader, device, weak_ids, all_ids)

    for parameter in head.parameters():
        parameter.requires_grad_(False)
    final_layer = head.classifier[-1]
    if not isinstance(final_layer, torch.nn.Linear) or final_layer.out_features != len(labels):
        raise TypeError("unexpected LocalSemanticHead classifier contract")
    final_layer.weight.requires_grad_(True)
    final_layer.bias.requires_grad_(True)
    row_mask = torch.zeros(len(labels), 1, device=device)
    row_mask[list(sorted(weak_ids))] = 1.0
    final_layer.weight.register_hook(lambda grad: grad * row_mask)
    final_layer.bias.register_hook(lambda grad: grad * row_mask.squeeze(1))
    frozen_nonweak_weight = final_layer.weight.detach().cpu().clone()[
        [index for index in range(len(labels)) if index not in weak_ids]
    ]
    frozen_nonweak_bias = final_layer.bias.detach().cpu().clone()[
        [index for index in range(len(labels)) if index not in weak_ids]
    ]

    base_counts = Counter(
        int(label) for values in base_train["labels"] for label in values.tolist()
    )
    class_weight = torch.tensor(
        [1.0 / math.sqrt(max(base_counts.get(index, 1), 1)) for index in range(len(labels))],
        device=device,
    )
    class_weight = (class_weight / class_weight.mean()).clamp(0.5, 2.5)
    # Adam rather than AdamW: decoupled weight decay would move the supposedly
    # frozen 172 rows even after their gradients are masked.
    optimizer = torch.optim.Adam(
        [final_layer.weight, final_layer.bias], lr=args.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )

    best_key = (
        float(baseline_metrics["all"]["top1_accuracy_↑"]),
        float(baseline_metrics["targeted_weak_16"]["top1_accuracy_↑"]),
        float(baseline_metrics["all"]["top5_accuracy_↑"]),
    )
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    best_metrics = baseline_metrics
    history: list[dict[str, Any]] = [{"epoch": 0, "dev": baseline_metrics}]
    stale = 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for batch in train_loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            spans, mask, target = collect_spans(
                features,
                batch["intervals"],
                batch["labels"],
                jitter=args.boundary_jitter_frames,
                training=True,
            )
            logits = head(spans, mask)
            loss = F.cross_entropy(
                logits, target, weight=class_weight, label_smoothing=args.label_smoothing
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([final_layer.weight, final_layer.bias], 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * target.numel()
            correct += int(logits.argmax(dim=-1).eq(target).sum())
            seen += int(target.numel())
        scheduler.step()
        current = evaluate_slices(head, dev_loader, device, weak_ids, all_ids)
        summary = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(seen, 1),
            "train_top1": correct / max(seen, 1),
            "dev_top1": current["all"]["top1_accuracy_↑"],
            "dev_weak16_top1": current["targeted_weak_16"]["top1_accuracy_↑"],
            "dev_other172_top1": current["other_172"]["top1_accuracy_↑"],
            "dev_top5": current["all"]["top5_accuracy_↑"],
        }
        history.append({**summary, "dev": current})
        print(json.dumps(summary, sort_keys=True), flush=True)
        key = (
            float(current["all"]["top1_accuracy_↑"]),
            float(current["targeted_weak_16"]["top1_accuracy_↑"]),
            float(current["all"]["top5_accuracy_↑"]),
        )
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = current
            best_state = {
                name: value.detach().cpu().clone() for name, value in head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    checkpoint_path = output_dir / "atst_surgical_weak_rows_v1_best.pt"
    _atomic_torch(
        {
            "format": FORMAT,
            "model_state_dict": best_state,
            "input_dim": int(baseline_payload["input_dim"]),
            "hidden_dim": int(baseline_payload["hidden_dim"]),
            "labels": labels,
            "weak_labels": weak_labels,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
        },
        checkpoint_path,
    )

    nonweak = [index for index in range(len(labels)) if index not in weak_ids]
    saved_nonweak_weight = best_state["classifier.4.weight"][nonweak]
    saved_nonweak_bias = best_state["classifier.4.bias"][nonweak]
    invariance = {
        "nonweak_weight_max_abs_change": float((saved_nonweak_weight - frozen_nonweak_weight).abs().max()),
        "nonweak_bias_max_abs_change": float((saved_nonweak_bias - frozen_nonweak_bias).abs().max()),
        "shared_state_exactly_unchanged": all(
            torch.equal(value, baseline_payload["model_state_dict"][name])
            for name, value in best_state.items()
            if name not in {"classifier.4.weight", "classifier.4.bias"}
        ),
    }
    delta = {
        scope: {
            key: float(best_metrics[scope][key]) - float(baseline_metrics[scope][key])
            for key in ("top1_accuracy_↑", "top5_accuracy_↑", "macro_top1_accuracy_↑")
        }
        for scope in ("all", "targeted_weak_16", "other_172")
    }
    gates = {
        "overall_top1_strictly_improves": delta["all"]["top1_accuracy_↑"] > 0.0,
        "targeted_weak_top1_improves_at_least_0_05": delta["targeted_weak_16"]["top1_accuracy_↑"] >= 0.05,
        "other_172_top1_does_not_drop_over_0_005": delta["other_172"]["top1_accuracy_↑"] >= -0.005,
        "overall_top5_does_not_drop_over_0_005": delta["all"]["top5_accuracy_↑"] >= -0.005,
        "frozen_parameters_bitwise_unchanged": (
            invariance["nonweak_weight_max_abs_change"] == 0.0
            and invariance["nonweak_bias_max_abs_change"] == 0.0
            and invariance["shared_state_exactly_unchanged"]
        ),
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "baseline initialization + 16 final-classifier rows only + full V5 rehearsal",
        "trainable_parameter_policy": {
            "trainable": "classifier.4 rows for the 16 audited weak classes",
            "frozen": "all shared layers and 172 non-target classifier rows",
            "optimizer": "Adam without decoupled weight decay",
        },
        "data": {
            "base_train_scenes": len(base_train["scene_id"]),
            "enrichment_scenes": len(enrich_cache["scene_id"]),
            "dev_scenes": len(dev_cache["scene_id"]),
            "locked_test_evaluated": False,
        },
        "baseline": baseline_metrics,
        "best": best_metrics,
        "delta": delta,
        "best_epoch": best_epoch,
        "invariance_audit": invariance,
        "gates": {"passed": all(gates.values()), "checks": gates},
        "decision": "retain_surgical_enrichment" if all(gates.values()) else "do_not_integrate_surgical_enrichment",
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "baseline_checkpoint": str(baseline_checkpoint),
        "baseline_checkpoint_sha256": _sha256_file(baseline_checkpoint),
        "enrichment_build_receipt_sha256": _sha256_file(build_receipt_path),
        "enrichment_cache_sha256": _sha256_file(enrich_path),
        "fixed_dev_cache_sha256": _sha256_file(dev_path),
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({
        "complete": True,
        "best_epoch": best_epoch,
        "baseline_top1": baseline_metrics["all"]["top1_accuracy_↑"],
        "best_top1": best_metrics["all"]["top1_accuracy_↑"],
        "weak_top1_delta": delta["targeted_weak_16"]["top1_accuracy_↑"],
        "other_top1_delta": delta["other_172"]["top1_accuracy_↑"],
        "gates_passed": all(gates.values()),
        "decision": receipt["decision"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
