#!/usr/bin/env python3
"""Measure whether targeted enrichment has calibratable semantic headroom.

The baseline remains authoritative for all 172 non-target logits.  For the 16
audited weak classes only, this audit interpolates baseline and enriched logits
and applies one shared bias.  A two-scalar family is intentionally used instead
of per-class tuning to limit validation overfitting.
"""

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

import torch
from torch.utils.data import DataLoader

from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _sha256_file
from mixi_understanding.scripts.train_qces_v5_atst_oracle_semantic_screen_v1 import (
    CachedSceneDataset,
    cached_collate,
)
from mixi_understanding.scripts.train_qces_v5_atst_targeted_enrichment_v1 import (
    load_head,
    metrics,
    predictions,
)


FORMAT = "qces_v5_targeted_logit_calibration_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1/atst_features_dev.pt"))
    parser.add_argument("--enrichment-root", type=Path, default=Path("/var/tmp/qces_v5_targeted_enrichment_v1"))
    parser.add_argument("--baseline-checkpoint", type=Path, default=base / "v5_atst_oracle_semantic_screen_v1/atst_oracle_semantic_head_v1_best.pt")
    parser.add_argument("--enriched-checkpoint", type=Path, default=base / "v5_atst_targeted_enrichment_v1/atst_targeted_enrichment_v1_best.pt")
    parser.add_argument("--output", type=Path, default=base / "v5_atst_targeted_enrichment_v1/logit_calibration_audit_v1.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def compact_metric(scores: torch.Tensor, targets: torch.Tensor, subset: set[int] | None = None) -> dict[str, Any]:
    keep = torch.ones_like(targets, dtype=torch.bool)
    if subset is not None:
        keep = torch.tensor([int(value) in subset for value in targets.tolist()])
    score = scores[keep]
    target = targets[keep]
    ordering = score.topk(k=20, dim=-1).indices
    totals: dict[int, int] = {}
    correct: dict[int, int] = {}
    for gold, pred in zip(target.tolist(), ordering[:, 0].tolist(), strict=True):
        totals[gold] = totals.get(gold, 0) + 1
        correct[gold] = correct.get(gold, 0) + int(gold == pred)
    return {
        "events": int(target.numel()),
        "observed_classes": len(totals),
        "top1_accuracy_↑": float(ordering[:, 0].eq(target).float().mean()),
        "top5_accuracy_↑": float((ordering[:, :5] == target[:, None]).any(-1).float().mean()),
        "top20_accuracy_↑": float((ordering == target[:, None]).any(-1).float().mean()),
        "macro_top1_accuracy_↑": sum(correct.get(label, 0) / count for label, count in totals.items()) / len(totals),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cache = torch.load(args.dev_cache.resolve(), map_location="cpu", weights_only=False)
    loader = DataLoader(
        CachedSceneDataset(cache), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=cached_collate,
    )
    baseline, baseline_payload = load_head(args.baseline_checkpoint.resolve(), device)
    enriched, enriched_payload = load_head(args.enriched_checkpoint.resolve(), device)
    labels = list(baseline_payload["labels"])
    if list(enriched_payload["labels"]) != labels:
        raise ValueError("checkpoint ontologies differ")
    weak_labels = json.loads(
        (args.enrichment_root.resolve() / "build_receipt.json").read_text(encoding="utf-8")
    )["weak_labels"]
    weak_ids = {labels.index(label) for label in weak_labels}
    other_ids = set(range(len(labels))) - weak_ids

    baseline_scores, targets = predictions(baseline, loader, device)
    enriched_scores, enriched_targets = predictions(enriched, loader, device)
    if not torch.equal(targets, enriched_targets):
        raise RuntimeError("dev target order changed")

    baseline_pred = baseline_scores.argmax(1)
    enriched_pred = enriched_scores.argmax(1)
    correct_base = baseline_pred.eq(targets)
    correct_enriched = enriched_pred.eq(targets)
    headroom = {
        "either_model_oracle_top1_↑": float((correct_base | correct_enriched).float().mean()),
        "both_correct": int((correct_base & correct_enriched).sum()),
        "baseline_only_correct": int((correct_base & ~correct_enriched).sum()),
        "enriched_only_correct": int((~correct_base & correct_enriched).sum()),
        "both_wrong": int((~correct_base & ~correct_enriched).sum()),
    }
    weak_mask = torch.tensor([int(value) in weak_ids for value in targets.tolist()])
    other_mask = ~weak_mask
    headroom["weak16_either_oracle_top1_↑"] = float(
        ((correct_base | correct_enriched)[weak_mask]).float().mean()
    )
    headroom["other172_either_oracle_top1_↑"] = float(
        ((correct_base | correct_enriched)[other_mask]).float().mean()
    )

    # Scale the bias grid by the actual baseline-logit dispersion rather than
    # hard-coding a model-specific numerical range.
    logit_std = float(baseline_scores.std())
    alphas = [index / 20.0 for index in range(21)]
    biases = [(-2.0 + index / 20.0) * logit_std for index in range(81)]
    rows: list[dict[str, Any]] = []
    weak_index = sorted(weak_ids)
    for alpha in alphas:
        fused = baseline_scores.clone()
        fused[:, weak_index] = (
            (1.0 - alpha) * baseline_scores[:, weak_index]
            + alpha * enriched_scores[:, weak_index]
        )
        for bias in biases:
            current = fused.clone()
            current[:, weak_index] += bias
            all_metric = compact_metric(current, targets)
            weak_metric = compact_metric(current, targets, weak_ids)
            other_metric = compact_metric(current, targets, other_ids)
            rows.append({
                "alpha": alpha,
                "weak_shared_bias": bias,
                "all_top1": all_metric["top1_accuracy_↑"],
                "all_top5": all_metric["top5_accuracy_↑"],
                "weak16_top1": weak_metric["top1_accuracy_↑"],
                "other172_top1": other_metric["top1_accuracy_↑"],
            })

    baseline_metric = {
        "all": compact_metric(baseline_scores, targets),
        "weak16": compact_metric(baseline_scores, targets, weak_ids),
        "other172": compact_metric(baseline_scores, targets, other_ids),
    }
    enriched_metric = {
        "all": compact_metric(enriched_scores, targets),
        "weak16": compact_metric(enriched_scores, targets, weak_ids),
        "other172": compact_metric(enriched_scores, targets, other_ids),
    }
    eligible = [
        row for row in rows
        if row["weak16_top1"] >= baseline_metric["weak16"]["top1_accuracy_↑"] + 0.05
        and row["other172_top1"] >= baseline_metric["other172"]["top1_accuracy_↑"] - 0.005
        and row["all_top5"] >= baseline_metric["all"]["top5_accuracy_↑"] - 0.005
    ]
    selected = max(
        eligible or rows,
        key=lambda row: (row["all_top1"], row["weak16_top1"], row["all_top5"]),
    )
    gates = {
        "constrained_operating_point_exists": bool(eligible),
        "overall_top1_improves_at_least_0_005": (
            selected["all_top1"] >= baseline_metric["all"]["top1_accuracy_↑"] + 0.005
        ),
        "weak16_top1_improves_at_least_0_05": (
            selected["weak16_top1"] >= baseline_metric["weak16"]["top1_accuracy_↑"] + 0.05
        ),
        "other172_top1_drop_at_most_0_005": (
            selected["other172_top1"] >= baseline_metric["other172"]["top1_accuracy_↑"] - 0.005
        ),
    }
    report = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "fixed V5 dev diagnostic; locked test remains untouched",
        "method": "baseline logits for 172 classes; two-scalar interpolation+bias for 16 weak classes",
        "degrees_of_freedom": 2,
        "baseline": baseline_metric,
        "full_enriched": enriched_metric,
        "complementarity": headroom,
        "grid": {
            "alphas": len(alphas),
            "biases": len(biases),
            "logit_std": logit_std,
            "operating_points": len(rows),
            "eligible_operating_points": len(eligible),
        },
        "selected": selected,
        "delta_vs_baseline": {
            "all_top1": selected["all_top1"] - baseline_metric["all"]["top1_accuracy_↑"],
            "all_top5": selected["all_top5"] - baseline_metric["all"]["top5_accuracy_↑"],
            "weak16_top1": selected["weak16_top1"] - baseline_metric["weak16"]["top1_accuracy_↑"],
            "other172_top1": selected["other172_top1"] - baseline_metric["other172"]["top1_accuracy_↑"],
        },
        "gates": {"passed": all(gates.values()), "checks": gates},
        "decision": "freeze_calibration_then_evaluate_locked_test" if all(gates.values()) else "close_targeted_enrichment_branch",
        "artifacts": {
            "dev_cache_sha256": _sha256_file(args.dev_cache.resolve()),
            "baseline_checkpoint_sha256": _sha256_file(args.baseline_checkpoint.resolve()),
            "enriched_checkpoint_sha256": _sha256_file(args.enriched_checkpoint.resolve()),
        },
    }
    _atomic_json(report, args.output.resolve())
    print(json.dumps({
        "baseline_top1": baseline_metric["all"]["top1_accuracy_↑"],
        "selected": selected,
        "delta": report["delta_vs_baseline"],
        "gates": report["gates"],
        "decision": report["decision"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
