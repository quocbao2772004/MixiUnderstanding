#!/usr/bin/env python3
"""Clean-pretrained, paired mixture/clean seven-expert semantic experiment.

The seven-class partition is frozen from ``v5_oracle_group_experts_v1``.  This
experiment changes only the expert representation/training recipe:

1. pretrain each 26--27 way expert on exact clean event components;
2. adapt the same experts to oracle-span mixture statistics;
3. retain clean supervision and a small clean-teacher KL term during adaptation.

Development data selects checkpoints.  Matched official-eval is read once only
after the partition and both training stages are frozen.
"""

from __future__ import annotations

import argparse
import copy
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
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mixi_understanding.scripts.train_qces_pretrainedsed_detector import set_seed
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json,
    _atomic_torch,
    _sha256_file,
)
from mixi_understanding.scripts.train_qces_v5_oracle_group_experts_v1 import (
    GroupExperts,
    expert_ordering,
    group_lookup,
    loader,
    ordering_metrics,
    per_group_metrics,
    scene_bootstrap,
)


FORMAT = "qces_v5_dualview_group_experts_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    previous = base / "v5_oracle_group_experts_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-checkpoint", type=Path, default=previous / "oracle_group_experts_v1_best.pt")
    parser.add_argument("--previous-receipt", type=Path, default=previous / "receipt.json")
    parser.add_argument("--train-mixture-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1/atst_features_train.pt"))
    parser.add_argument("--dev-mixture-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1/atst_features_dev.pt"))
    parser.add_argument("--matched-mixture-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_matched_eval_v1/atst_features_matched_eval.pt"))
    parser.add_argument("--train-clean-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_clean_stats_ceiling_v1/clean_stats_train.pt"))
    parser.add_argument("--dev-clean-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_clean_stats_ceiling_v1/clean_stats_dev.pt"))
    parser.add_argument("--matched-clean-cache", type=Path, default=Path("/var/tmp/qces_v5_atst_clean_stats_ceiling_v1/clean_stats_matched_eval.pt"))
    parser.add_argument(
        "--extra-clean-cache",
        type=Path,
        help="Optional leakage-audited extra Gold train statistics used only during clean pretraining.",
    )
    parser.add_argument("--stats-cache-dir", type=Path, default=Path("/var/tmp/qces_v5_dualview_group_experts_v1"))
    parser.add_argument("--output-dir", type=Path, default=base / "v5_dualview_group_experts_v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=9107)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--clean-epochs", type=int, default=24)
    parser.add_argument("--adapt-epochs", type=int, default=24)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--clean-learning-rate", type=float, default=3e-4)
    parser.add_argument("--adapt-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--clean-rehearsal-weight", type=float, default=0.25)
    parser.add_argument("--distillation-weight", type=float, default=0.10)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def event_targets(cache: Mapping[str, Any]) -> torch.Tensor:
    return torch.cat([value.long() for value in cache["labels"]])


def event_scene_ids(cache: Mapping[str, Any]) -> list[str]:
    return [
        str(scene_id)
        for scene_id, labels in zip(cache["scene_id"], cache["labels"], strict=True)
        for _ in range(len(labels))
    ]


def mixture_stats(cache: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[torch.Tensor] = []
    targets: list[int] = []
    scene_ids: list[str] = []
    time_steps = int(cache["features"].shape[1])
    for scene_id, features, intervals, labels in zip(
        cache["scene_id"], cache["features"], cache["intervals"], cache["labels"], strict=True
    ):
        value = features.float()
        for interval, target in zip(intervals, labels, strict=True):
            start = max(0, min(time_steps - 1, int(math.floor(float(interval[0]) * time_steps))))
            end = max(start + 1, min(time_steps, int(math.ceil(float(interval[1]) * time_steps))))
            span = value[start:end]
            rows.append(torch.cat((span.mean(0), span.amax(0), span.std(0, unbiased=False))).half())
            targets.append(int(target))
            scene_ids.append(str(scene_id))
    return {
        "format": FORMAT + "_mixture_stats",
        "features": torch.stack(rows),
        "targets": torch.tensor(targets, dtype=torch.long),
        "scene_id": scene_ids,
    }


class StatsExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, classes),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value.float())


class StatsExperts(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, groups: Sequence[Sequence[int]]) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [StatsExpert(input_dim, hidden_dim, len(group)) for group in groups]
        )


@torch.inference_mode()
def stats_ordering(
    model: StatsExperts,
    features: torch.Tensor,
    targets: torch.Tensor,
    groups: Sequence[Sequence[int]],
    label_to_group: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    maximum = max(map(len, groups))
    orders: list[torch.Tensor] = []
    device_lookup = label_to_group.to(device)
    for begin in range(0, len(features), batch_size):
        value = features[begin : begin + batch_size].to(device)
        target = targets[begin : begin + batch_size].to(device)
        target_groups = device_lookup[target]
        batch_order = torch.full((len(value), maximum), -1, dtype=torch.long, device=device)
        for group_id, (head, group) in enumerate(zip(model.heads, groups, strict=True)):
            selected = torch.nonzero(target_groups == group_id, as_tuple=False).flatten()
            if not selected.numel():
                continue
            local_order = head(value[selected]).argsort(1, descending=True)
            global_labels = torch.tensor(group, dtype=torch.long, device=device)
            batch_order[selected, : len(group)] = global_labels[local_order]
        orders.append(batch_order.cpu())
    return torch.cat(orders)


def class_weights(
    targets: torch.Tensor,
    groups: Sequence[Sequence[int]],
    device: torch.device,
) -> list[torch.Tensor]:
    counts = Counter(targets.tolist())
    result: list[torch.Tensor] = []
    for group in groups:
        weight = torch.tensor(
            [1.0 / math.sqrt(max(counts.get(label, 1), 1)) for label in group],
            dtype=torch.float32,
            device=device,
        )
        result.append((weight / weight.mean()).clamp(0.5, 2.5))
    return result


def local_lookups(
    groups: Sequence[Sequence[int]], num_classes: int, device: torch.device
) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    for group in groups:
        mapping = {int(label): index for index, label in enumerate(group)}
        result.append(
            torch.tensor([mapping.get(label, -1) for label in range(num_classes)], device=device)
        )
    return result


def evaluate_stats(
    model: StatsExperts,
    features: torch.Tensor,
    targets: torch.Tensor,
    groups: Sequence[Sequence[int]],
    label_to_group: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    order = stats_ordering(
        model, features, targets, groups, label_to_group, device=device, batch_size=batch_size
    )
    return order, ordering_metrics(order, targets)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_cache_dir = args.stats_cache_dir.resolve()
    stats_cache_dir.mkdir(parents=True, exist_ok=True)

    previous_checkpoint_path = args.groups_checkpoint.resolve()
    previous_payload = torch.load(previous_checkpoint_path, map_location="cpu", weights_only=False)
    labels = list(previous_payload["labels"])
    groups = [list(map(int, group)) for group in previous_payload["groups"]]
    if len(labels) != 188 or len(groups) != 7:
        raise ValueError("frozen 188-class/seven-group contract changed")
    label_to_group, local_maps = group_lookup(groups, len(labels))

    mixture_paths = {
        "train": args.train_mixture_cache.resolve(),
        "dev": args.dev_mixture_cache.resolve(),
        "matched": args.matched_mixture_cache.resolve(),
    }
    clean_paths = {
        "train": args.train_clean_cache.resolve(),
        "dev": args.dev_clean_cache.resolve(),
        "matched": args.matched_clean_cache.resolve(),
    }
    mixture_sequence = {
        split: torch.load(path, map_location="cpu", weights_only=False)
        for split, path in mixture_paths.items()
    }
    clean = {
        split: torch.load(path, map_location="cpu", weights_only=False)
        for split, path in clean_paths.items()
    }
    mix: dict[str, dict[str, Any]] = {}
    mix_stats_paths: dict[str, Path] = {}
    alignment: dict[str, Any] = {}
    for split in ("train", "dev", "matched"):
        stats = mixture_stats(mixture_sequence[split])
        path = stats_cache_dir / f"mixture_stats_{split}.pt"
        _atomic_torch(stats, path)
        mix[split] = stats
        mix_stats_paths[split] = path
        same_target = torch.equal(stats["targets"], clean[split]["targets"].long())
        same_scene = list(stats["scene_id"]) == list(clean[split]["scene_id"])
        alignment[split] = {
            "events": len(stats["targets"]),
            "target_order_exact": same_target,
            "scene_order_exact": same_scene,
        }
        if not same_target or not same_scene:
            raise RuntimeError(f"{split} mixture/clean pairing is not exact")
    print(json.dumps({"paired_alignment": alignment}, sort_keys=True), flush=True)

    input_dim = int(clean["train"]["features"].shape[-1])
    clean_pretrain_features = clean["train"]["features"]
    clean_pretrain_targets = clean["train"]["targets"].long()
    extra_clean_audit: dict[str, Any] | None = None
    if args.extra_clean_cache is not None:
        extra_path = args.extra_clean_cache.resolve()
        extra = torch.load(extra_path, map_location="cpu", weights_only=False)
        if int(extra["features"].shape[-1]) != input_dim:
            raise ValueError("extra clean feature dimension differs")
        if int(extra["targets"].min()) < 0 or int(extra["targets"].max()) >= len(labels):
            raise ValueError("extra clean cache contains label outside frozen ontology")
        clean_pretrain_features = torch.cat((clean_pretrain_features, extra["features"]))
        clean_pretrain_targets = torch.cat((clean_pretrain_targets, extra["targets"].long()))
        extra_clean_audit = {
            "path": str(extra_path),
            "sha256": _sha256_file(extra_path),
            "events": len(extra["targets"]),
            "observed_classes": int(extra["targets"].unique().numel()),
            "combined_clean_pretrain_events": len(clean_pretrain_targets),
        }

    model = StatsExperts(input_dim, args.hidden_dim, groups).to(device)
    weights = class_weights(clean_pretrain_targets, groups, device)
    lookup = local_lookups(groups, len(labels), device)
    device_label_to_group = label_to_group.to(device)

    # Stage 1: learn within-group class boundaries from exact clean components.
    clean_loader = DataLoader(
        TensorDataset(clean_pretrain_features, clean_pretrain_targets),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.clean_learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.clean_epochs, eta_min=args.clean_learning_rate * 0.05
    )
    clean_best_key: tuple[float, float, float] | None = None
    clean_best_epoch = 0
    clean_best_state: dict[str, torch.Tensor] | None = None
    clean_best_metrics: dict[str, Any] | None = None
    clean_history: list[dict[str, Any]] = []
    stale = 0
    for epoch in range(1, args.clean_epochs + 1):
        model.train()
        loss_sum = 0.0
        correct = seen = 0
        for features, targets in clean_loader:
            features = features.to(device)
            targets = targets.to(device)
            target_groups = device_label_to_group[targets]
            summed = torch.zeros((), device=device)
            batch_seen = batch_correct = 0
            for group_id, head in enumerate(model.heads):
                selected = torch.nonzero(target_groups == group_id, as_tuple=False).flatten()
                if not selected.numel():
                    continue
                local_target = lookup[group_id][targets[selected]]
                logits = head(features[selected])
                loss = F.cross_entropy(
                    logits,
                    local_target,
                    weight=weights[group_id],
                    label_smoothing=args.label_smoothing,
                    reduction="sum",
                )
                summed = summed + loss
                batch_seen += int(selected.numel())
                batch_correct += int(logits.argmax(1).eq(local_target).sum())
            objective = summed / max(batch_seen, 1)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(summed.detach())
            correct += batch_correct
            seen += batch_seen
        scheduler.step()
        _, metrics = evaluate_stats(
            model,
            clean["dev"]["features"],
            clean["dev"]["targets"].long(),
            groups,
            label_to_group,
            device=device,
            batch_size=args.batch_size,
        )
        row = {
            "stage": "clean_pretrain",
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "train_top1": correct / max(seen, 1),
            "dev": metrics,
        }
        clean_history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (
            float(metrics["top1_accuracy_↑"]),
            float(metrics["macro_top1_accuracy_↑"]),
            float(metrics["top5_accuracy_↑"]),
        )
        if clean_best_key is None or key > clean_best_key:
            clean_best_key = key
            clean_best_epoch = epoch
            clean_best_metrics = metrics
            clean_best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"stage": "clean_pretrain", "early_stop": True, "epoch": epoch, "best_epoch": clean_best_epoch}), flush=True)
            break
    if clean_best_state is None or clean_best_metrics is None:
        raise RuntimeError("clean pretraining selected no checkpoint")
    model.load_state_dict(clean_best_state)
    teacher = copy.deepcopy(model).to(device).eval().requires_grad_(False)

    # Stage 2: adapt to the mixture while preserving clean decision boundaries.
    paired_loader = DataLoader(
        TensorDataset(
            mix["train"]["features"],
            clean["train"]["features"],
            mix["train"]["targets"].long(),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.adapt_learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.adapt_epochs, eta_min=args.adapt_learning_rate * 0.05
    )
    adapt_best_key: tuple[float, float, float] | None = None
    adapt_best_epoch = 0
    adapt_best_state: dict[str, torch.Tensor] | None = None
    adapt_best_metrics: dict[str, Any] | None = None
    adapt_history: list[dict[str, Any]] = []
    stale = 0
    temperature = float(args.temperature)
    for epoch in range(1, args.adapt_epochs + 1):
        model.train()
        loss_sum = ce_mix_sum = ce_clean_sum = kd_sum = 0.0
        correct = seen = 0
        for mix_features, clean_features, targets in paired_loader:
            mix_features = mix_features.to(device)
            clean_features = clean_features.to(device)
            targets = targets.to(device)
            target_groups = device_label_to_group[targets]
            summed = torch.zeros((), device=device)
            batch_seen = batch_correct = 0
            batch_ce_mix = batch_ce_clean = batch_kd = 0.0
            for group_id, (head, teacher_head) in enumerate(zip(model.heads, teacher.heads, strict=True)):
                selected = torch.nonzero(target_groups == group_id, as_tuple=False).flatten()
                if not selected.numel():
                    continue
                local_target = lookup[group_id][targets[selected]]
                mix_logits = head(mix_features[selected])
                clean_logits = head(clean_features[selected])
                with torch.no_grad():
                    teacher_logits = teacher_head(clean_features[selected])
                ce_mix = F.cross_entropy(
                    mix_logits,
                    local_target,
                    weight=weights[group_id],
                    label_smoothing=args.label_smoothing,
                    reduction="sum",
                )
                ce_clean = F.cross_entropy(
                    clean_logits,
                    local_target,
                    weight=weights[group_id],
                    label_smoothing=args.label_smoothing,
                    reduction="sum",
                )
                kd = F.kl_div(
                    F.log_softmax(mix_logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1),
                    reduction="none",
                ).sum(1).sum() * (temperature ** 2)
                group_loss = (
                    ce_mix
                    + args.clean_rehearsal_weight * ce_clean
                    + args.distillation_weight * kd
                )
                summed = summed + group_loss
                batch_ce_mix += float(ce_mix.detach())
                batch_ce_clean += float(ce_clean.detach())
                batch_kd += float(kd.detach())
                batch_seen += int(selected.numel())
                batch_correct += int(mix_logits.argmax(1).eq(local_target).sum())
            objective = summed / max(batch_seen, 1)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(summed.detach())
            ce_mix_sum += batch_ce_mix
            ce_clean_sum += batch_ce_clean
            kd_sum += batch_kd
            correct += batch_correct
            seen += batch_seen
        scheduler.step()
        dev_order, dev_metrics = evaluate_stats(
            model,
            mix["dev"]["features"],
            mix["dev"]["targets"].long(),
            groups,
            label_to_group,
            device=device,
            batch_size=args.batch_size,
        )
        _, clean_retention = evaluate_stats(
            model,
            clean["dev"]["features"],
            clean["dev"]["targets"].long(),
            groups,
            label_to_group,
            device=device,
            batch_size=args.batch_size,
        )
        row = {
            "stage": "paired_adaptation",
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "train_ce_mix": ce_mix_sum / max(seen, 1),
            "train_ce_clean": ce_clean_sum / max(seen, 1),
            "train_kd": kd_sum / max(seen, 1),
            "train_mixture_top1": correct / max(seen, 1),
            "mixture_dev": dev_metrics,
            "clean_dev_retention": clean_retention,
        }
        adapt_history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (
            float(dev_metrics["top1_accuracy_↑"]),
            float(dev_metrics["macro_top1_accuracy_↑"]),
            float(dev_metrics["top5_accuracy_↑"]),
        )
        if adapt_best_key is None or key > adapt_best_key:
            adapt_best_key = key
            adapt_best_epoch = epoch
            adapt_best_metrics = dev_metrics
            adapt_best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"stage": "paired_adaptation", "early_stop": True, "epoch": epoch, "best_epoch": adapt_best_epoch}), flush=True)
            break
    if adapt_best_state is None or adapt_best_metrics is None:
        raise RuntimeError("paired adaptation selected no checkpoint")
    model.load_state_dict(adapt_best_state)

    checkpoint_path = output_dir / "dualview_group_experts_v1_best.pt"
    _atomic_torch(
        {
            "format": FORMAT,
            "model_state_dict": adapt_best_state,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "labels": labels,
            "groups": groups,
            "clean_best_epoch": clean_best_epoch,
            "adapt_best_epoch": adapt_best_epoch,
            "adapt_best_dev_metrics": adapt_best_metrics,
        },
        checkpoint_path,
    )

    # Frozen checkpoint and partition; matched split starts here.
    matched_targets = mix["matched"]["targets"].long()
    proposed_mix_order, proposed_mix_metrics = evaluate_stats(
        model,
        mix["matched"]["features"],
        matched_targets,
        groups,
        label_to_group,
        device=device,
        batch_size=args.batch_size,
    )
    proposed_clean_order, proposed_clean_metrics = evaluate_stats(
        model,
        clean["matched"]["features"],
        clean["matched"]["targets"].long(),
        groups,
        label_to_group,
        device=device,
        batch_size=args.batch_size,
    )
    clean_model = StatsExperts(input_dim, args.hidden_dim, groups).to(device)
    clean_model.load_state_dict(clean_best_state)
    clean_upper_order, clean_upper_metrics = evaluate_stats(
        clean_model,
        clean["matched"]["features"],
        clean["matched"]["targets"].long(),
        groups,
        label_to_group,
        device=device,
        batch_size=args.batch_size,
    )

    previous_model = GroupExperts(
        int(previous_payload["input_dim"]), int(previous_payload["hidden_dim"]), groups
    ).to(device)
    previous_model.load_state_dict(previous_payload["model_state_dict"])
    previous_order, previous_targets = expert_ordering(
        previous_model,
        loader(mixture_sequence["matched"], 32, shuffle=False, seed=args.seed),
        device,
        groups,
        label_to_group,
    )
    if not torch.equal(previous_targets, matched_targets):
        raise RuntimeError("previous/proposed matched event order differs")
    previous_metrics = ordering_metrics(previous_order, previous_targets)
    delta_top1 = float(proposed_mix_metrics["top1_accuracy_↑"] - previous_metrics["top1_accuracy_↑"])
    delta_top5 = float(proposed_mix_metrics["top5_accuracy_↑"] - previous_metrics["top5_accuracy_↑"])
    scene_index = np.concatenate(
        [
            np.full(len(values), index, dtype=np.int64)
            for index, values in enumerate(mixture_sequence["matched"]["labels"])
        ]
    )
    bootstrap = scene_bootstrap(
        previous_order[:, 0].eq(previous_targets).numpy(),
        proposed_mix_order[:, 0].eq(matched_targets).numpy(),
        scene_index,
        num_scenes=len(mixture_sequence["matched"]["scene_id"]),
        replicates=args.bootstrap_replicates,
        seed=args.seed + 2,
    )
    matched_per_group = per_group_metrics(
        proposed_mix_order, matched_targets, groups, label_to_group
    )
    gates = {
        "matched_top1_ge_0_65": float(proposed_mix_metrics["top1_accuracy_↑"]) >= 0.65,
        "matched_top1_improves_previous_by_0_03": delta_top1 >= 0.03,
        "matched_top5_not_worse": delta_top5 >= 0.0,
        "bootstrap_ci_lower_gt_0": float(bootstrap["ci95_lower"]) > 0.0,
        "clean_oracle_group_top1_ge_0_70": float(clean_upper_metrics["top1_accuracy_↑"]) >= 0.70,
    }
    passed = all(gates.values())
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "oracle-group semantic decision experiment; not deployable routing",
        "method": "clean group pretraining followed by paired mixture adaptation with clean rehearsal and low-weight KL",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "alignment": alignment,
        "extra_clean_pretraining": extra_clean_audit,
        "data": {
            split: {"events": len(mix[split]["targets"]), "scenes": len(mixture_sequence[split]["scene_id"])}
            for split in ("train", "dev", "matched")
        },
        "clean_pretrain": {
            "best_epoch": clean_best_epoch,
            "best_dev": clean_best_metrics,
            "matched_clean_upper_bound": clean_upper_metrics,
            "history": clean_history,
        },
        "paired_adaptation": {
            "best_epoch": adapt_best_epoch,
            "best_mixture_dev": adapt_best_metrics,
            "matched_mixture": proposed_mix_metrics,
            "matched_clean_retention": proposed_clean_metrics,
            "history": adapt_history,
        },
        "matched_comparison": {
            "previous_sequence_experts": previous_metrics,
            "proposed_dualview_stats_experts": proposed_mix_metrics,
            "delta_top1": delta_top1,
            "delta_top5": delta_top5,
            "scene_bootstrap_delta_top1": bootstrap,
            "per_group": matched_per_group,
        },
        "gates": {"passed": passed, "checks": gates},
        "decision": "retain_dualview_group_experts" if passed else "do_not_integrate_dualview_group_experts",
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "previous_checkpoint_sha256": _sha256_file(previous_checkpoint_path),
            "previous_receipt_sha256": _sha256_file(args.previous_receipt.resolve()),
            "mixture_stats": {
                split: {"path": str(path), "sha256": _sha256_file(path)}
                for split, path in mix_stats_paths.items()
            },
            "clean_caches": {
                split: {"path": str(path), "sha256": _sha256_file(path)}
                for split, path in clean_paths.items()
            },
        },
    }
    receipt_path = output_dir / "receipt.json"
    _atomic_json(receipt, receipt_path)
    print(
        json.dumps(
            {
                "complete": True,
                "clean_upper_top1": clean_upper_metrics["top1_accuracy_↑"],
                "previous_matched_top1": previous_metrics["top1_accuracy_↑"],
                "proposed_matched_top1": proposed_mix_metrics["top1_accuracy_↑"],
                "delta_top1": delta_top1,
                "bootstrap": bootstrap,
                "gates": gates,
                "decision": receipt["decision"],
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
