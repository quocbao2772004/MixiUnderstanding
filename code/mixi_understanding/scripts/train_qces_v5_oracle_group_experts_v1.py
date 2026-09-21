#!/usr/bin/env python3
"""Decision experiment for a seven-expert QCES semantic classifier.

This is deliberately an oracle-*group* experiment, not an oracle-answer
experiment.  Each event is routed only to the expert containing its gold class;
the selected expert must still choose the answer among 26--27 classes.

The class partition is learned from the ordinary V5 development split and then
frozen before the matched official-eval split is opened.  Classes that the flat
head confuses (or represents similarly in its final layer) are kept together,
which makes the future coarse routing problem realistic instead of gaming the
oracle score by separating every confusing pair.
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
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.data import DataLoader

from mixi_understanding.scripts.train_qces_local_semantic_r3 import LocalSemanticHead
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import set_seed
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json,
    _atomic_torch,
    _sha256_file,
)
from mixi_understanding.scripts.train_qces_v5_atst_oracle_semantic_screen_v1 import (
    CachedSceneDataset,
    cached_collate,
    collect_spans,
)
from mixi_understanding.scripts.train_qces_v5_atst_targeted_enrichment_v1 import load_head


FORMAT = "qces_v5_oracle_group_experts_v1"
NUM_GROUPS = 7


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-cache",
        type=Path,
        default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1/atst_features_train.pt"),
    )
    parser.add_argument(
        "--dev-cache",
        type=Path,
        default=Path("/var/tmp/qces_v5_atst_oracle_semantic_screen_v1/atst_features_dev.pt"),
    )
    parser.add_argument(
        "--matched-cache",
        type=Path,
        default=Path("/var/tmp/qces_v5_atst_matched_eval_v1/atst_features_matched_eval.pt"),
    )
    parser.add_argument(
        "--flat-checkpoint",
        type=Path,
        default=base / "v5_atst_oracle_semantic_screen_v1/atst_oracle_semantic_head_v1_best.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "v5_oracle_group_experts_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8821)
    parser.add_argument("--num-groups", type=int, default=NUM_GROUPS)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--boundary-jitter-frames", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def loader(cache: Mapping[str, Any], batch_size: int, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        CachedSceneDataset(cache),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        collate_fn=cached_collate,
    )


@torch.inference_mode()
def flat_predictions(
    head: LocalSemanticHead,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    head.eval()
    score_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    for batch in data_loader:
        features = batch["features"].to(device=device, dtype=torch.float32)
        spans, mask, target = collect_spans(
            features, batch["intervals"], batch["labels"], jitter=0, training=False
        )
        score_rows.append(head(spans, mask).cpu())
        target_rows.append(target.cpu())
    return torch.cat(score_rows), torch.cat(target_rows)


def balanced_confusion_partition(
    scores: torch.Tensor,
    targets: torch.Tensor,
    classifier_weights: torch.Tensor,
    *,
    num_groups: int,
    seed: int,
) -> tuple[list[list[int]], dict[str, float]]:
    """Cluster confusable classes together with exact balanced capacities."""
    num_classes = int(scores.shape[1])
    if num_classes < num_groups:
        raise ValueError("more groups than classes")
    topk = min(10, num_classes)
    ordering = scores.argsort(dim=1, descending=True)[:, :topk]
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    counts = np.bincount(targets.numpy(), minlength=num_classes).astype(np.float64)
    rank_weight = 1.0 / np.log2(np.arange(2, topk + 2, dtype=np.float64))
    for gold, predictions in zip(targets.tolist(), ordering.tolist(), strict=True):
        for rank, predicted in enumerate(predictions):
            if int(predicted) != int(gold):
                confusion[int(gold), int(predicted)] += rank_weight[rank]
    confusion /= np.maximum(counts[:, None], 1.0)
    confusion = 0.5 * (confusion + confusion.T)
    if float(confusion.max()) > 0:
        confusion /= float(confusion.max())

    weight = F.normalize(classifier_weights.float(), dim=1).cpu().numpy()
    representation = np.maximum(weight @ weight.T, 0.0)
    np.fill_diagonal(representation, 0.0)
    affinity = 0.80 * confusion + 0.20 * representation
    np.fill_diagonal(affinity, 1.0)

    degree = np.maximum(affinity.sum(axis=1), 1e-8)
    normalized = affinity / np.sqrt(degree[:, None] * degree[None, :])
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    embedding = eigenvectors[:, np.argsort(eigenvalues)[-num_groups:]]
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)

    initial = KMeans(n_clusters=num_groups, n_init=50, random_state=seed).fit(embedding)
    centers = initial.cluster_centers_.copy()
    base, remainder = divmod(num_classes, num_groups)
    capacities = np.asarray([base + int(index < remainder) for index in range(num_groups)])
    slots = np.repeat(np.arange(num_groups), capacities)
    assignment = np.full(num_classes, -1, dtype=np.int64)
    for _ in range(30):
        cost = ((embedding[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        rows, columns = linear_sum_assignment(cost[:, slots])
        updated = np.full(num_classes, -1, dtype=np.int64)
        updated[rows] = slots[columns]
        if np.array_equal(updated, assignment):
            break
        assignment = updated
        for group_id in range(num_groups):
            centers[group_id] = embedding[assignment == group_id].mean(axis=0)

    groups = [np.flatnonzero(assignment == group_id).tolist() for group_id in range(num_groups)]
    if sorted(label for group in groups for label in group) != list(range(num_classes)):
        raise RuntimeError("invalid class partition")
    within = sum(
        float(affinity[np.ix_(group, group)].sum() - len(group)) for group in groups
    )
    total = float(affinity.sum() - num_classes)
    audit = {
        "within_group_affinity_fraction_↑": within / max(total, 1e-8),
        "minimum_group_size": min(map(len, groups)),
        "maximum_group_size": max(map(len, groups)),
    }
    return groups, audit


def group_lookup(groups: Sequence[Sequence[int]], num_classes: int) -> tuple[torch.Tensor, list[dict[int, int]]]:
    label_to_group = torch.full((num_classes,), -1, dtype=torch.long)
    local_maps: list[dict[int, int]] = []
    for group_id, group in enumerate(groups):
        local_maps.append({int(label): index for index, label in enumerate(group)})
        label_to_group[torch.tensor(group, dtype=torch.long)] = group_id
    if bool((label_to_group < 0).any()):
        raise RuntimeError("some labels have no expert")
    return label_to_group, local_maps


class GroupExperts(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, groups: Sequence[Sequence[int]]) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [LocalSemanticHead(input_dim, hidden_dim, len(group)) for group in groups]
        )


def ordering_metrics(ordering: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    totals: Counter[int] = Counter(targets.tolist())
    correct: Counter[int] = Counter()
    for gold, predicted in zip(targets.tolist(), ordering[:, 0].tolist(), strict=True):
        correct[int(gold)] += int(gold == predicted)
    result: dict[str, Any] = {
        "events": int(targets.numel()),
        "observed_classes": len(totals),
        "top1_accuracy_↑": float(ordering[:, 0].eq(targets).float().mean()),
        "top5_accuracy_↑": float(
            (ordering[:, : min(5, ordering.shape[1])] == targets[:, None]).any(1).float().mean()
        ),
        "top20_accuracy_↑": float(
            (ordering[:, : min(20, ordering.shape[1])] == targets[:, None]).any(1).float().mean()
        ),
        "macro_top1_accuracy_↑": float(
            sum(correct[label] / count for label, count in totals.items()) / len(totals)
        ),
    }
    return result


def flat_ordering(scores: torch.Tensor) -> torch.Tensor:
    return scores.argsort(dim=1, descending=True)


def oracle_masked_flat_ordering(
    scores: torch.Tensor,
    targets: torch.Tensor,
    groups: Sequence[Sequence[int]],
    label_to_group: torch.Tensor,
) -> torch.Tensor:
    result = torch.full((targets.numel(), max(map(len, groups))), -1, dtype=torch.long)
    for group_id, group in enumerate(groups):
        selected = torch.nonzero(label_to_group[targets] == group_id, as_tuple=False).flatten()
        local = scores[selected][:, group]
        result[selected, : len(group)] = torch.tensor(group, dtype=torch.long)[
            local.argsort(1, descending=True)
        ]
    return result


@torch.inference_mode()
def expert_ordering(
    model: GroupExperts,
    data_loader: DataLoader,
    device: torch.device,
    groups: Sequence[Sequence[int]],
    label_to_group: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    orders: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    maximum = max(map(len, groups))
    for batch in data_loader:
        features = batch["features"].to(device=device, dtype=torch.float32)
        spans, mask, target = collect_spans(
            features, batch["intervals"], batch["labels"], jitter=0, training=False
        )
        batch_order = torch.full(
            (target.numel(), maximum), -1, dtype=torch.long, device=device
        )
        target_groups = label_to_group.to(device)[target]
        for group_id, (head, group) in enumerate(zip(model.heads, groups, strict=True)):
            selected = torch.nonzero(target_groups == group_id, as_tuple=False).flatten()
            if not selected.numel():
                continue
            local_order = head(spans[selected], mask[selected]).argsort(1, descending=True)
            global_labels = torch.tensor(group, dtype=torch.long, device=device)
            batch_order[selected, : len(group)] = global_labels[local_order]
        orders.append(batch_order.cpu())
        targets.append(target.cpu())
    return torch.cat(orders), torch.cat(targets)


def per_group_metrics(
    ordering: torch.Tensor,
    targets: torch.Tensor,
    groups: Sequence[Sequence[int]],
    label_to_group: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_groups = label_to_group[targets]
    for group_id, group in enumerate(groups):
        selected = target_groups == group_id
        rows.append({"group_id": group_id, "classes": len(group), **ordering_metrics(ordering[selected], targets[selected])})
    return rows


def scene_bootstrap(
    flat_correct: np.ndarray,
    expert_correct: np.ndarray,
    scene_index: np.ndarray,
    *,
    num_scenes: int,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=np.float64)
    difference = expert_correct.astype(np.float64) - flat_correct.astype(np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, num_scenes, size=num_scenes)
        weights = np.bincount(sampled, minlength=num_scenes)[scene_index]
        delta[replicate] = float((difference * weights).sum() / max(weights.sum(), 1))
    return {
        "replicates": replicates,
        "mean_delta": float(delta.mean()),
        "ci95_lower": float(np.quantile(delta, 0.025)),
        "ci95_upper": float(np.quantile(delta, 0.975)),
        "probability_delta_gt_0": float((delta > 0).mean()),
    }


def main() -> None:
    args = parse_args()
    if args.num_groups != NUM_GROUPS:
        raise ValueError("this decision experiment is locked to seven groups")
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = {
        "train": args.train_cache.resolve(),
        "dev": args.dev_cache.resolve(),
        "matched": args.matched_cache.resolve(),
    }
    caches = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in cache_paths.items()
    }
    checkpoint_path = args.flat_checkpoint.resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    flat_head, flat_payload = load_head(checkpoint_path, device)
    labels = list(flat_payload["labels"])
    num_classes = len(labels)
    if num_classes != 188:
        raise ValueError(f"expected 188 frozen classes, got {num_classes}")
    for name, cache in caches.items():
        maximum = max(int(value.max()) for value in cache["labels"] if value.numel())
        if maximum >= num_classes:
            raise ValueError(f"{name} cache contains label outside the frozen ontology")

    design_loader = loader(caches["dev"], args.batch_size, shuffle=False, seed=args.seed)
    dev_flat_scores, dev_targets = flat_predictions(flat_head, design_loader, device)
    classifier_weights = flat_payload["model_state_dict"]["classifier.4.weight"]
    groups, grouping_audit = balanced_confusion_partition(
        dev_flat_scores,
        dev_targets,
        classifier_weights,
        num_groups=args.num_groups,
        seed=args.seed,
    )
    label_to_group, local_maps = group_lookup(groups, num_classes)
    group_manifest = {
        "format": FORMAT + "_groups",
        "partition_source": "flat-head confusion and classifier geometry on V5 design-dev only",
        "matched_eval_used": False,
        "audit": grouping_audit,
        "groups": [
            {
                "group_id": group_id,
                "size": len(group),
                "label_ids": group,
                "labels": [labels[label] for label in group],
            }
            for group_id, group in enumerate(groups)
        ],
    }
    group_path = output_dir / "groups.json"
    _atomic_json(group_manifest, group_path)

    dev_flat_order = flat_ordering(dev_flat_scores)
    dev_masked_order = oracle_masked_flat_ordering(
        dev_flat_scores, dev_targets, groups, label_to_group
    )
    dev_controls = {
        "flat_188": ordering_metrics(dev_flat_order, dev_targets),
        "same_flat_head_oracle_group_mask": ordering_metrics(dev_masked_order, dev_targets),
    }
    print(json.dumps({"grouping": grouping_audit, "dev_controls": dev_controls}, sort_keys=True), flush=True)

    train_counts = Counter(
        int(label) for values in caches["train"]["labels"] for label in values.tolist()
    )
    group_weights: list[torch.Tensor] = []
    for group in groups:
        weights = torch.tensor(
            [1.0 / math.sqrt(max(train_counts.get(label, 1), 1)) for label in group],
            dtype=torch.float32,
            device=device,
        )
        group_weights.append((weights / weights.mean()).clamp(0.5, 2.5))

    model = GroupExperts(
        int(caches["train"]["features"].shape[-1]), args.hidden_dim, groups
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    train_loader = loader(caches["train"], args.batch_size, shuffle=True, seed=args.seed)
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_dev_order: torch.Tensor | None = None
    best_dev_metrics: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    stale = 0
    local_lookup = [
        torch.tensor([local_maps[group_id].get(label, -1) for label in range(num_classes)], device=device)
        for group_id in range(args.num_groups)
    ]
    device_label_to_group = label_to_group.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = 0.0
        correct = 0
        seen = 0
        for batch in train_loader:
            features = batch["features"].to(device=device, dtype=torch.float32)
            spans, mask, targets = collect_spans(
                features,
                batch["intervals"],
                batch["labels"],
                jitter=args.boundary_jitter_frames,
                training=True,
            )
            target_groups = device_label_to_group[targets]
            summed_loss = torch.zeros((), dtype=torch.float32, device=device)
            batch_seen = 0
            batch_correct = 0
            for group_id, head in enumerate(model.heads):
                selected = torch.nonzero(target_groups == group_id, as_tuple=False).flatten()
                if not selected.numel():
                    continue
                local_target = local_lookup[group_id][targets[selected]]
                logits = head(spans[selected], mask[selected])
                losses = F.cross_entropy(
                    logits,
                    local_target,
                    weight=group_weights[group_id],
                    label_smoothing=args.label_smoothing,
                    reduction="none",
                )
                summed_loss = summed_loss + losses.sum()
                batch_seen += int(selected.numel())
                batch_correct += int(logits.argmax(1).eq(local_target).sum())
            loss = summed_loss / max(batch_seen, 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_total += float(summed_loss.detach())
            correct += batch_correct
            seen += batch_seen
        scheduler.step()
        dev_order, repeated_dev_targets = expert_ordering(
            model, design_loader, device, groups, label_to_group
        )
        if not torch.equal(dev_targets, repeated_dev_targets):
            raise RuntimeError("dev event order changed")
        dev_metrics = ordering_metrics(dev_order, dev_targets)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_total / max(seen, 1),
            "train_top1": correct / max(seen, 1),
            "dev": dev_metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (
            float(dev_metrics["top1_accuracy_↑"]),
            float(dev_metrics["macro_top1_accuracy_↑"]),
            float(dev_metrics["top5_accuracy_↑"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_dev_order = dev_order.clone()
            best_dev_metrics = dev_metrics
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    if best_state is None or best_dev_order is None or best_dev_metrics is None:
        raise RuntimeError("no expert checkpoint selected")
    model.load_state_dict(best_state)
    expert_checkpoint = output_dir / "oracle_group_experts_v1_best.pt"
    _atomic_torch(
        {
            "format": FORMAT,
            "model_state_dict": best_state,
            "input_dim": int(caches["train"]["features"].shape[-1]),
            "hidden_dim": args.hidden_dim,
            "labels": labels,
            "groups": groups,
            "best_epoch": best_epoch,
            "best_dev_metrics": best_dev_metrics,
        },
        expert_checkpoint,
    )

    # The grouping and checkpoint are now frozen.  Only here is matched eval used.
    matched_loader = loader(caches["matched"], args.batch_size, shuffle=False, seed=args.seed)
    matched_flat_scores, matched_targets = flat_predictions(flat_head, matched_loader, device)
    matched_flat_order = flat_ordering(matched_flat_scores)
    matched_masked_order = oracle_masked_flat_ordering(
        matched_flat_scores, matched_targets, groups, label_to_group
    )
    matched_expert_order, repeated_matched_targets = expert_ordering(
        model, matched_loader, device, groups, label_to_group
    )
    if not torch.equal(matched_targets, repeated_matched_targets):
        raise RuntimeError("matched event order changed")

    dev_result = {
        **dev_controls,
        "trained_oracle_group_experts": best_dev_metrics,
        "per_group": per_group_metrics(best_dev_order, dev_targets, groups, label_to_group),
    }
    matched_result = {
        "flat_188": ordering_metrics(matched_flat_order, matched_targets),
        "same_flat_head_oracle_group_mask": ordering_metrics(matched_masked_order, matched_targets),
        "trained_oracle_group_experts": ordering_metrics(matched_expert_order, matched_targets),
        "per_group": per_group_metrics(
            matched_expert_order, matched_targets, groups, label_to_group
        ),
    }
    flat_matched = matched_result["flat_188"]
    expert_matched = matched_result["trained_oracle_group_experts"]
    top1_delta = float(expert_matched["top1_accuracy_↑"] - flat_matched["top1_accuracy_↑"])
    top5_delta = float(expert_matched["top5_accuracy_↑"] - flat_matched["top5_accuracy_↑"])
    scene_index = np.concatenate(
        [np.full(len(values), index, dtype=np.int64) for index, values in enumerate(caches["matched"]["labels"])]
    )
    bootstrap = scene_bootstrap(
        matched_flat_order[:, 0].eq(matched_targets).numpy(),
        matched_expert_order[:, 0].eq(matched_targets).numpy(),
        scene_index,
        num_scenes=len(caches["matched"]["scene_id"]),
        replicates=args.bootstrap_replicates,
        seed=args.seed + 1,
    )
    gates = {
        "matched_top1_gain_ge_0_05": top1_delta >= 0.05,
        "matched_top1_ge_0_65": float(expert_matched["top1_accuracy_↑"]) >= 0.65,
        "matched_top5_not_worse": top5_delta >= 0.0,
        "matched_bootstrap_ci_lower_gt_0": float(bootstrap["ci95_lower"]) > 0.0,
    }
    passed = all(gates.values())
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "oracle group only; no QA answer label, question, or matched annotation used for grouping/training",
        "not_deployable": True,
        "data": {
            "classes": num_classes,
            "groups": args.num_groups,
            "train_scenes": len(caches["train"]["scene_id"]),
            "train_events": sum(len(value) for value in caches["train"]["labels"]),
            "dev_scenes": len(caches["dev"]["scene_id"]),
            "dev_events": int(dev_targets.numel()),
            "matched_scenes": len(caches["matched"]["scene_id"]),
            "matched_events": int(matched_targets.numel()),
        },
        "partition": group_manifest,
        "best_epoch": best_epoch,
        "design_dev": dev_result,
        "matched_held_out": matched_result,
        "matched_delta": {
            "trained_experts_minus_flat_top1": top1_delta,
            "trained_experts_minus_flat_top5": top5_delta,
        },
        "scene_bootstrap_experts_minus_flat_top1": bootstrap,
        "gates": {"passed": passed, "checks": gates},
        "decision": "proceed_to_top2_router" if passed else "reject_or_redesign_seven_experts",
        "history": history,
        "artifacts": {
            "flat_checkpoint": str(checkpoint_path),
            "flat_checkpoint_sha256": _sha256_file(checkpoint_path),
            "expert_checkpoint": str(expert_checkpoint),
            "expert_checkpoint_sha256": _sha256_file(expert_checkpoint),
            "groups": str(group_path),
            "groups_sha256": _sha256_file(group_path),
            "caches": {
                name: {"path": str(path), "sha256": _sha256_file(path)}
                for name, path in cache_paths.items()
            },
        },
    }
    receipt_path = output_dir / "receipt.json"
    _atomic_json(receipt, receipt_path)
    print(
        json.dumps(
            {
                "complete": True,
                "best_epoch": best_epoch,
                "matched_flat_top1": flat_matched["top1_accuracy_↑"],
                "matched_oracle_mask_top1": matched_result["same_flat_head_oracle_group_mask"]["top1_accuracy_↑"],
                "matched_expert_top1": expert_matched["top1_accuracy_↑"],
                "matched_expert_top1_delta": top1_delta,
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
