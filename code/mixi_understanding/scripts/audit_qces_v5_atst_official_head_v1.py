#!/usr/bin/env python3
"""Audit ATST's original 447-class AudioSet-Strong head on V5 oracle spans."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from data_util.audioset_classes import as_strong_train_classes
from mixi_understanding.scripts.audit_qces_v5_backbone_complementarity_v1 import infer_atst
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, default=Path("/var/tmp/qces_full188_tiered_realistic_v5/ontology_188.txt"))
    parser.add_argument("--cache", type=Path, default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1/atst_features_dev.pt"))
    parser.add_argument("--official-checkpoint", type=Path, default=PROJECT_ROOT / "code/baseline/PretrainedSED/resources/ATST-F_strong_1.pt")
    parser.add_argument("--learned-checkpoint", type=Path, default=BASE / "v5_atst_oracle_semantic_screen_v1/atst_oracle_semantic_head_v1_best.pt")
    parser.add_argument("--output", type=Path, default=BASE / "v5_atst_oracle_semantic_screen_v1/official_447_head_audit_v1.json")
    return parser.parse_args()


def canonical(label: str) -> str:
    # The QCES builder renders AudioSet comma-separated synonyms with "and".
    # Dropping only conjunctions gives a bijection for all 188 selected labels.
    tokens = re.findall(r"[a-z0-9]+", label.lower().replace("_", " "))
    return " ".join(value for value in tokens if value not in {"and", "or"})


def metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    order = logits.argsort(dim=-1, descending=True)
    return {
        "events": int(target.numel()),
        "top1_accuracy_\u2191": float(order[:, 0].eq(target).float().mean()),
        "top5_accuracy_\u2191": float((order[:, :5] == target[:, None]).any(dim=-1).float().mean()),
        "top20_accuracy_\u2191": float((order[:, :20] == target[:, None]).any(dim=-1).float().mean()),
    }


def official_span_scores(cache: Mapping[str, Any], weight: torch.Tensor, bias: torch.Tensor, mapping: torch.Tensor) -> dict[str, torch.Tensor]:
    result: dict[str, list[torch.Tensor]] = {"mean_logit": [], "max_logit": [], "logmeanexp": []}
    for features, intervals in zip(cache["features"], cache["intervals"]):
        frame_logits = F.linear(features.float(), weight, bias)[:, mapping]
        for interval in intervals:
            start = max(0, min(249, int(torch.floor(interval[0] * 250))))
            end = max(start + 1, min(250, int(torch.ceil(interval[1] * 250))))
            current = frame_logits[start:end]
            result["mean_logit"].append(current.mean(0))
            result["max_logit"].append(current.amax(0))
            result["logmeanexp"].append(torch.logsumexp(current, dim=0) - torch.log(torch.tensor(float(len(current)))))
    return {key: torch.stack(value) for key, value in result.items()}


def main() -> None:
    args = parse_args()
    labels = [line.strip() for line in args.ontology.read_text(encoding="utf-8").splitlines() if line.strip()]
    official_lookup = {canonical(label): index for index, label in enumerate(as_strong_train_classes)}
    if len(official_lookup) != 447: raise ValueError("official AudioSet labels are not canonical-unique")
    missing = [label for label in labels if canonical(label) not in official_lookup]
    if missing: raise ValueError(f"unmapped QCES labels: {missing}")
    mapping = torch.tensor([official_lookup[canonical(label)] for label in labels], dtype=torch.long)
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    official = torch.load(args.official_checkpoint, map_location="cpu", weights_only=True)
    scores = official_span_scores(cache, official["strong_head.weight"], official["strong_head.bias"], mapping)
    learned_logits, target = infer_atst(cache, torch.load(args.learned_checkpoint, map_location="cpu", weights_only=True))
    official_reports = {key: metrics(value, target) for key, value in scores.items()}
    best_name = max(official_reports, key=lambda key: official_reports[key]["top1_accuracy_\u2191"])
    learned_ok = learned_logits.argmax(-1).eq(target); official_ok = scores[best_name].argmax(-1).eq(target)
    receipt = {
        "format": "qces_v5_atst_official_447_head_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "oracle-span diagnostic on V5 dev; no parameter training and no locked-test access",
        "mapping": {"qces_classes": len(labels), "mapped_to_official_447": len(mapping), "policy": "lowercase alphanumeric tokens with conjunctions and/or removed"},
        "learned_188_head": metrics(learned_logits, target),
        "official_447_head": official_reports,
        "best_official_pooling": best_name,
        "complementarity_with_learned_head": {
            "both_correct": int((learned_ok & official_ok).sum()),
            "learned_only_correct": int((learned_ok & ~official_ok).sum()),
            "official_only_correct": int((~learned_ok & official_ok).sum()),
            "both_wrong": int((~learned_ok & ~official_ok).sum()),
            "either_head_oracle_top1_\u2191": float((learned_ok | official_ok).float().mean()),
        },
        "decision": "use_official_head_as_prior" if official_reports[best_name]["top1_accuracy_\u2191"] >= 0.50 and int((~learned_ok & official_ok).sum()) >= 100 else "official_head_prior_insufficient",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(receipt, args.output)
    print(json.dumps({"complete": True, **receipt}, indent=2))


if __name__ == "__main__":
    main()
