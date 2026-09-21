#!/usr/bin/env python3
"""Controlled ATST semantic retrain on V5 plus Gold targeted enrichment.

The frozen V5 dev cache, architecture, optimizer recipe, and seed are inherited
from the ATST architecture screen.  Only the training cache changes: a
leakage-audited Gold enrichment tier for 16 weak classes is appended.  Loss
weights remain computed from the original V5 train distribution so adding data
does not silently down-weight the targeted classes.
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
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline/PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models.prediction_wrapper as prediction_wrapper_module
from models.atstframe.ATSTF_wrapper import ATSTWrapper
from models.prediction_wrapper import PredictionsWrapper
from mixi_understanding.scripts.train_qces_local_semantic_r3 import LocalSemanticHead
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    load_ontology,
    load_scene_manifest,
    make_device,
    set_seed,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json,
    _atomic_torch,
    _sha256_file,
)
from mixi_understanding.scripts.train_qces_v5_atst_oracle_semantic_screen_v1 import (
    CachedSceneDataset,
    cached_collate,
    collect_spans,
    export_cache,
)


FORMAT = "qces_v5_atst_targeted_enrichment_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5-cache-dir", type=Path, default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1"))
    parser.add_argument("--baseline-dir", type=Path, default=base / "v5_atst_oracle_semantic_screen_v1")
    parser.add_argument("--enrichment-root", type=Path, default=Path("/var/tmp/qces_v5_targeted_enrichment_v1"))
    parser.add_argument("--enrichment-cache", type=Path, default=Path("/var/tmp/qces_v5_targeted_enrichment_v1/atst_features_enrich.pt"))
    parser.add_argument("--ontology", type=Path, default=PROJECT_ROOT / "outputs/qces_full188_tiered_realistic_v5/ontology_188.txt")
    parser.add_argument("--output-dir", type=Path, default=base / "v5_atst_targeted_enrichment_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=6604)
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--boundary-jitter-frames", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def combine_caches(base: Mapping[str, Any], enrich: Mapping[str, Any]) -> dict[str, Any]:
    if int(base["features"].shape[-1]) != int(enrich["features"].shape[-1]):
        raise ValueError("base/enrichment ATST feature dimensions differ")
    scene_ids = list(base["scene_id"]) + list(enrich["scene_id"])
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("base/enrichment scene ids overlap")
    return {
        "format": FORMAT + "_combined_cache",
        "split": "train_plus_targeted_enrich",
        "scene_id": scene_ids,
        "features": torch.cat([base["features"], enrich["features"]]),
        "intervals": list(base["intervals"]) + list(enrich["intervals"]),
        "labels": list(base["labels"]) + list(enrich["labels"]),
    }


@torch.inference_mode()
def predictions(
    head: LocalSemanticHead,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    head.eval()
    score_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for batch in loader:
        value = batch["features"].to(device=device, dtype=torch.float32)
        spans, mask, target = collect_spans(
            value, batch["intervals"], batch["labels"], jitter=0, training=False
        )
        score_rows.append(head(spans, mask).cpu())
        target_rows.append(target.cpu())
    return torch.cat(score_rows), torch.cat(target_rows)


def metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    label_subset: set[int] | None = None,
) -> dict[str, Any]:
    keep = torch.ones_like(targets, dtype=torch.bool)
    if label_subset is not None:
        keep = torch.tensor([int(value) in label_subset for value in targets.tolist()])
    score = scores[keep]
    target = targets[keep]
    ordering = score.argsort(dim=-1, descending=True)
    totals: Counter[int] = Counter(target.tolist())
    correct: Counter[int] = Counter()
    for gold, pred in zip(target.tolist(), ordering[:, 0].tolist(), strict=True):
        correct[gold] += int(gold == pred)
    return {
        "events": int(target.numel()),
        "observed_classes": len(totals),
        "top1_accuracy_↑": float(ordering[:, 0].eq(target).float().mean()),
        "top5_accuracy_↑": float((ordering[:, :5] == target[:, None]).any(dim=-1).float().mean()),
        "top20_accuracy_↑": float((ordering[:, :20] == target[:, None]).any(dim=-1).float().mean()),
        "macro_top1_accuracy_↑": float(
            sum(correct[label] / count for label, count in totals.items()) / len(totals)
        ),
        "per_class_top1": {
            str(label): {"correct": correct[label], "total": count, "accuracy": correct[label] / count}
            for label, count in sorted(totals.items())
        },
    }


def evaluate_slices(
    head: LocalSemanticHead,
    loader: DataLoader,
    device: torch.device,
    weak_ids: set[int],
    all_ids: set[int],
) -> dict[str, Any]:
    score, target = predictions(head, loader, device)
    return {
        "all": metrics(score, target),
        "targeted_weak_16": metrics(score, target, label_subset=weak_ids),
        "other_172": metrics(score, target, label_subset=all_ids - weak_ids),
    }


def load_head(path: Path, device: torch.device) -> tuple[LocalSemanticHead, Mapping[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    head = LocalSemanticHead(
        int(payload["input_dim"]), int(payload["hidden_dim"]), len(payload["labels"])
    ).to(device)
    head.load_state_dict(payload["model_state_dict"])
    return head, payload


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_ontology(args.ontology.resolve())
    if len(labels) != 188:
        raise ValueError("expected frozen 188-class ontology")
    label_map = {label: index for index, label in enumerate(labels)}
    enrichment_root = args.enrichment_root.resolve()
    build_receipt_path = enrichment_root / "build_receipt.json"
    build_receipt = json.loads(build_receipt_path.read_text(encoding="utf-8"))
    if not build_receipt.get("quality_gates", {}).get("passed"):
        raise ValueError("enrichment build quality gates did not pass")
    if build_receipt["fixed_evaluation"]["hard_identity_overlap"] != 0:
        raise ValueError("enrichment identity leakage")
    weak_labels = list(build_receipt["weak_labels"])
    weak_ids = {label_map[label] for label in weak_labels}
    all_ids = set(range(len(labels)))

    v5_cache_dir = args.v5_cache_dir.resolve()
    base_train_path = v5_cache_dir / "atst_features_train.pt"
    dev_path = v5_cache_dir / "atst_features_dev.pt"
    base_train = torch.load(base_train_path, map_location="cpu", weights_only=False)
    dev_cache = torch.load(dev_path, map_location="cpu", weights_only=False)

    enrichment_manifest = enrichment_root / "detector_scene_manifest_targeted_enrich.jsonl"
    enrich_cache_path = args.enrichment_cache.resolve()
    device = make_device(args.device)
    prediction_wrapper_module.RESOURCES_FOLDER = str(PRETRAINED_SED_ROOT / "resources")
    enrich_rows = load_scene_manifest(enrichment_manifest, label_map)
    backbone = PredictionsWrapper(
        ATSTWrapper(),
        checkpoint="ATST-F_strong_1",
        n_classes_strong=len(labels),
        n_classes_weak=len(labels),
        seq_model_type=None,
        head_type="linear",
    ).to(device)
    backbone.eval().requires_grad_(False)
    enrich_cache = export_cache(
        backbone,
        enrich_rows,
        device,
        batch_size=args.feature_batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        split="enrich",
    )
    enrich_cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_torch(enrich_cache, enrich_cache_path)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    combined = combine_caches(base_train, enrich_cache)
    train_dataset = CachedSceneDataset(combined)
    dev_dataset = CachedSceneDataset(dev_cache)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=cached_collate,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=cached_collate,
    )

    baseline_checkpoint = args.baseline_dir.resolve() / "atst_oracle_semantic_head_v1_best.pt"
    baseline_head, baseline_payload = load_head(baseline_checkpoint, device)
    if list(baseline_payload["labels"]) != labels:
        raise ValueError("baseline ontology differs")
    baseline_metrics = evaluate_slices(baseline_head, dev_loader, device, weak_ids, all_ids)
    del baseline_head

    input_dim = int(combined["features"].shape[-1])
    head = LocalSemanticHead(input_dim, args.hidden_dim, len(labels)).to(device)
    # Preserve the original V5 loss weighting.  Otherwise appending examples
    # would paradoxically reduce the class weight of exactly the weak labels.
    base_counts = Counter(
        int(label) for values in base_train["labels"] for label in values.tolist()
    )
    weight = torch.tensor(
        [1.0 / math.sqrt(max(base_counts.get(index, 1), 1)) for index in range(len(labels))],
        device=device,
    )
    weight = (weight / weight.mean()).clamp(0.5, 2.5)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )

    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        for batch in train_loader:
            value = batch["features"].to(device=device, dtype=torch.float32)
            spans, mask, target = collect_spans(
                value,
                batch["intervals"],
                batch["labels"],
                jitter=args.boundary_jitter_frames,
                training=True,
            )
            logits = head(spans, mask)
            loss = F.cross_entropy(
                logits, target, weight=weight, label_smoothing=args.label_smoothing
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * target.numel()
            correct += int(logits.argmax(dim=-1).eq(target).sum())
            seen += int(target.numel())
        scheduler.step()
        current = evaluate_slices(head, dev_loader, device, weak_ids, all_ids)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(seen, 1),
            "train_top1": correct / max(seen, 1),
            "dev": current,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (
            float(current["all"]["top1_accuracy_↑"]),
            float(current["targeted_weak_16"]["top1_accuracy_↑"]),
            float(current["all"]["top5_accuracy_↑"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in head.state_dict().items()
            }
            best_metrics = current
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    if best_state is None or best_metrics is None:
        raise RuntimeError("no checkpoint selected")
    checkpoint_path = output_dir / "atst_targeted_enrichment_v1_best.pt"
    _atomic_torch(
        {
            "format": FORMAT,
            "model_state_dict": best_state,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "labels": labels,
            "weak_labels": weak_labels,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
        },
        checkpoint_path,
    )

    delta = {
        scope: {
            key: float(best_metrics[scope][key]) - float(baseline_metrics[scope][key])
            for key in ("top1_accuracy_↑", "top5_accuracy_↑", "macro_top1_accuracy_↑")
        }
        for scope in ("all", "targeted_weak_16", "other_172")
    }
    gates = {
        "overall_top1_improves_at_least_0_01": delta["all"]["top1_accuracy_↑"] >= 0.01,
        "targeted_weak_top1_improves_at_least_0_08": delta["targeted_weak_16"]["top1_accuracy_↑"] >= 0.08,
        "overall_top5_drop_at_most_0_005": delta["all"]["top5_accuracy_↑"] >= -0.005,
        "other_172_top1_drop_at_most_0_01": delta["other_172"]["top1_accuracy_↑"] >= -0.01,
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen ATST oracle-interval semantic test of targeted Gold train enrichment",
        "controlled_change": "append leak-safe enrichment training scenes only",
        "fixed": {
            "dev_cache": str(dev_path),
            "dev_cache_sha256": _sha256_file(dev_path),
            "baseline_checkpoint": str(baseline_checkpoint),
            "baseline_checkpoint_sha256": _sha256_file(baseline_checkpoint),
            "architecture_optimizer_seed_and_class_weights": True,
            "locked_test_evaluated": False,
        },
        "data": {
            "base_train_scenes": len(base_train["scene_id"]),
            "enrichment_scenes": len(enrich_cache["scene_id"]),
            "combined_train_scenes": len(combined["scene_id"]),
            "base_train_events": sum(len(value) for value in base_train["labels"]),
            "enrichment_events": sum(len(value) for value in enrich_cache["labels"]),
            "dev_scenes": len(dev_cache["scene_id"]),
            "dev_events": sum(len(value) for value in dev_cache["labels"]),
            "weak_labels": weak_labels,
            "enrichment_build_receipt": str(build_receipt_path),
            "enrichment_build_receipt_sha256": _sha256_file(build_receipt_path),
        },
        "baseline": baseline_metrics,
        "best_epoch": best_epoch,
        "best": best_metrics,
        "delta": delta,
        "gates": {"passed": all(gates.values()), "checks": gates},
        "decision": "retain_targeted_enrichment" if all(gates.values()) else "reject_or_redesign_targeted_enrichment",
        "history": history,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "enrichment_cache": str(enrich_cache_path),
        "enrichment_cache_sha256": _sha256_file(enrich_cache_path),
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({
        "complete": True,
        "best_epoch": best_epoch,
        "baseline_top1": baseline_metrics["all"]["top1_accuracy_↑"],
        "best_top1": best_metrics["all"]["top1_accuracy_↑"],
        "weak_top1_delta": delta["targeted_weak_16"]["top1_accuracy_↑"],
        "gates": gates,
        "decision": receipt["decision"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
