#!/usr/bin/env python3
"""Train a label-conditioned listwise T-F matcher over semantic Top-K slots."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.relational_event_slots_v1 import interval_iou_matrix
from mixi_understanding.scripts.export_qces_beats_tf_grid_v1 import (
    FEATURE_DIM, FREQUENCY_PATCHES, TIME_PATCHES,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json, _atomic_torch, _device, _sha256_file,
)
from mixi_understanding.scripts.train_qces_tf_source_slot_semantic_v1 import (
    base_logits, load_grid,
)
from mixi_understanding.scripts.train_qces_v4_slot_semantic_head_v1 import NONE_LABEL


FORMAT = "qces_candidate_tf_matcher_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_candidate_tf_matcher_checkpoint_v1"


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    semantic = base / "v4_slot_semantic_head_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-grid-index", type=Path, default=Path("/var/tmp/qces_v4_beats_tf_train/index.json"))
    parser.add_argument("--dev-grid-index", type=Path, default=Path("/var/tmp/qces_v4_beats_tf_dev/index.json"))
    parser.add_argument("--train-slot-cache", type=Path, default=semantic / "frozen_slots_train.pt")
    parser.add_argument("--dev-slot-cache", type=Path, default=semantic / "frozen_slots_dev.pt")
    parser.add_argument("--base-semantic-checkpoint", type=Path, default=semantic / "v4_slot_semantic_head_v1_best.pt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--output-dir", type=Path, default=base / "candidate_tf_matcher_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8231)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--attention-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--temporal-temperature", type=float, default=0.02)
    parser.add_argument("--minimum-train-iou", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class MatcherDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self, grid: torch.Tensor, cache: Mapping[str, Any], logits: torch.Tensor,
        maximum: int,
    ) -> None:
        count = min(len(grid), maximum) if maximum else len(grid)
        self.grid = grid[:count]
        self.intervals = cache["intervals"][:count]
        self.objectness = cache["objectness"][:count]
        self.target_labels = cache["target_slot_labels"][:count]
        self.target_weights = cache["target_slot_weights"][:count]
        self.target_ious = cache["target_slot_ious"][:count]
        self.base_logits = logits[:count]

    def __len__(self) -> int:
        return len(self.grid)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "grid": self.grid[index], "intervals": self.intervals[index],
            "objectness": self.objectness[index], "target_labels": self.target_labels[index],
            "target_weights": self.target_weights[index], "target_ious": self.target_ious[index],
            "base_logits": self.base_logits[index],
        }


class CandidateTFMatcherV1(nn.Module):
    def __init__(
        self, *, label_prototypes: torch.Tensor, top_k: int,
        attention_dim: int, hidden_dim: int, dropout: float,
        temporal_temperature: float,
    ) -> None:
        super().__init__()
        self.top_k = int(top_k)
        self.temporal_temperature = float(temporal_temperature)
        self.grid_norm = nn.LayerNorm(FEATURE_DIM)
        self.key = nn.Linear(FEATURE_DIM, attention_dim)
        self.value = nn.Linear(FEATURE_DIM, attention_dim)
        self.label_projection = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM), nn.Linear(FEATURE_DIM, attention_dim), nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.LayerNorm(attention_dim + 5),
            nn.Linear(attention_dim + 5, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer("label_prototypes", F.normalize(label_prototypes.float(), dim=-1), persistent=True)

    def temporal_prior(self, intervals: torch.Tensor) -> torch.Tensor:
        centers = (torch.arange(TIME_PATCHES, device=intervals.device, dtype=intervals.dtype) + 0.5) / TIME_PATCHES
        start, end = intervals[..., 0, None], intervals[..., 1, None]
        prior = torch.sigmoid((centers - start) / self.temporal_temperature) * torch.sigmoid((end - centers) / self.temporal_temperature)
        return prior.clamp_min(1e-5)

    def forward(
        self, grid: torch.Tensor, intervals: torch.Tensor,
        objectness: torch.Tensor, base_logits: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        normalized_grid = self.grid_norm(grid.float())
        key = self.key(normalized_grid)
        value = self.value(normalized_grid)
        query = self.label_projection(self.label_prototypes[candidate_ids])
        compatibility = torch.einsum("bskd,btfd->bsktf", query, key) / math.sqrt(key.shape[-1])
        prior = self.temporal_prior(intervals.float())
        compatibility = compatibility + prior.log()[:, :, None, :, None]
        flattened = compatibility.flatten(-2)
        attention = flattened.softmax(dim=-1).reshape_as(compatibility)
        pooled = torch.einsum("bsktf,btfd->bskd", attention, value)
        interaction = pooled * F.normalize(query, dim=-1)
        evidence_lse = torch.logsumexp(flattened, dim=-1) - math.log(TIME_PATCHES * FREQUENCY_PATCHES)
        evidence_peak = flattened.max(dim=-1).values
        base_candidate = base_logits[..., :NONE_LABEL].gather(-1, candidate_ids)
        width = (intervals[..., 1] - intervals[..., 0]).unsqueeze(-1).expand_as(base_candidate)
        objectness_feature = objectness.unsqueeze(-1).expand_as(base_candidate)
        scalar = torch.stack((base_candidate, evidence_lse, evidence_peak, width, objectness_feature), dim=-1)
        residual = self.residual(torch.cat((interaction, scalar), dim=-1)).squeeze(-1)
        return base_candidate + residual


def candidates_with_gold(
    base: torch.Tensor, targets: torch.Tensor, *, top_k: int, inject_gold: bool,
) -> torch.Tensor:
    candidate = base[..., :NONE_LABEL].topk(top_k, dim=-1).indices
    if inject_gold:
        valid = targets != NONE_LABEL
        present = (candidate == targets.unsqueeze(-1)).any(dim=-1)
        replace = valid & ~present
        candidate = candidate.clone()
        candidate[..., -1] = torch.where(replace, targets, candidate[..., -1])
    return candidate


@torch.inference_mode()
def evaluate(
    model: CandidateTFMatcherV1, dataset: MatcherDataset,
    cache: Mapping[str, Any], device: torch.device,
    *, batch_size: int, top_k: int, objectness_threshold: float, amp: bool,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_candidates = []
    all_scores = []
    model.eval()
    for batch in loader:
        base = batch["base_logits"].to(device)
        candidate = candidates_with_gold(base, batch["target_labels"].to(device), top_k=top_k, inject_gold=False)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            scores = model(
                batch["grid"].to(device), batch["intervals"].to(device),
                batch["objectness"].to(device), base, candidate,
            )
        all_candidates.append(candidate.cpu())
        all_scores.append(scores.cpu())
    candidates = torch.cat(all_candidates)
    scores = torch.cat(all_scores)
    baseline_correct = ranker_correct = ranker_top5 = candidate_hits = 0
    localized = predicted_count = gold_count = 0
    paired = {"ranker_corrects_baseline": 0, "ranker_breaks_baseline": 0, "both_correct": 0, "both_wrong": 0}
    items = []
    for index in range(len(dataset)):
        base = dataset.base_logits[index]
        keep = (dataset.objectness[index].float() >= objectness_threshold) & (base.argmax(dim=-1) != NONE_LABEL)
        predicted_intervals = dataset.intervals[index].float()[keep]
        candidate = candidates[index][keep]
        score = scores[index][keep]
        gold_intervals = cache["gold_intervals"][index].float()
        gold_labels = cache["gold_labels"][index].long()
        predicted_count += int(predicted_intervals.shape[0])
        gold_count += int(gold_intervals.shape[0])
        if predicted_intervals.numel() == 0:
            continue
        iou = interval_iou_matrix(predicted_intervals, gold_intervals)
        rows, columns = linear_sum_assignment((1.0 - iou).numpy())
        for pred_index, gold_index in zip(rows, columns):
            if float(iou[pred_index, gold_index]) < 0.50:
                continue
            localized += 1
            gold = int(gold_labels[gold_index])
            baseline_label = int(candidate[pred_index, 0])
            order = score[pred_index].argsort(descending=True)
            ranked_labels = candidate[pred_index, order]
            ranker_label = int(ranked_labels[0])
            hit = gold in candidate[pred_index].tolist()
            candidate_hits += int(hit)
            baseline_ok = baseline_label == gold
            ranker_ok = ranker_label == gold
            baseline_correct += int(baseline_ok)
            ranker_correct += int(ranker_ok)
            ranker_top5 += int(gold in ranked_labels[:5].tolist())
            if baseline_ok and ranker_ok:
                paired["both_correct"] += 1
            elif baseline_ok:
                paired["ranker_breaks_baseline"] += 1
            elif ranker_ok:
                paired["ranker_corrects_baseline"] += 1
            else:
                paired["both_wrong"] += 1
            items.append({"scene_id": str(cache["scene_id"][index]), "gold_label_id": gold, "baseline_label_id": baseline_label, "ranker_label_id": ranker_label, "candidate_hit": hit})
    def prf(correct: int) -> dict[str, float]:
        precision = correct / max(predicted_count, 1)
        recall = correct / max(gold_count, 1)
        return {"precision_\u2191": precision, "recall_\u2191": recall, "f1_\u2191": 2 * precision * recall / max(precision + recall, 1e-12)}
    return {
        "scenes": len(dataset), "gold_events": gold_count, "predicted_events": predicted_count,
        "localized_events_iou50": localized,
        "localization_recall_iou50": localized / max(gold_count, 1),
        "candidate_recall_at_20_given_iou50_\u2191": candidate_hits / max(localized, 1),
        "baseline_top1_given_iou50_\u2191": baseline_correct / max(localized, 1),
        "ranker_top1_given_iou50_\u2191": ranker_correct / max(localized, 1),
        "ranker_top5_given_iou50_\u2191": ranker_top5 / max(localized, 1),
        "baseline_joint": prf(baseline_correct), "ranker_joint": prf(ranker_correct),
        "paired_changes": paired, "items": items,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {}
    for split, grid_path, cache_path, maximum in (
        ("train", args.train_grid_index, args.train_slot_cache, args.max_train_scenes),
        ("dev", args.dev_grid_index, args.dev_slot_cache, args.max_dev_scenes),
    ):
        ids, grid, events, index = load_grid(grid_path.resolve())
        cache = torch.load(cache_path.resolve(), map_location="cpu", weights_only=False)
        if ids != [str(value) for value in cache["scene_id"]]:
            raise ValueError(f"{split}: grid/slot order mismatch")
        payloads[split] = {"ids": ids, "grid": grid, "events": events, "index": index, "cache": cache, "maximum": maximum}
    train_sources = {str(event["source_id"]) for scene in payloads["train"]["events"] for event in scene}
    dev_sources = {str(event["source_id"]) for scene in payloads["dev"]["events"] for event in scene}
    if train_sources & dev_sources:
        raise ValueError("train/dev source leakage")
    semantic_checkpoint = torch.load(args.base_semantic_checkpoint.resolve(), map_location="cpu", weights_only=True)
    detector_checkpoint = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=False)
    bases = {split: base_logits(value["cache"], semantic_checkpoint) for split, value in payloads.items()}
    datasets = {split: MatcherDataset(value["grid"], value["cache"], bases[split], value["maximum"]) for split, value in payloads.items()}
    r1_weight = detector_checkpoint["model_state_dict"]["strong_head.weight"]
    device = _device(args.device)
    model = CandidateTFMatcherV1(
        label_prototypes=r1_weight, top_k=args.top_k,
        attention_dim=args.attention_dim, hidden_dim=args.hidden_dim,
        dropout=args.dropout, temporal_temperature=args.temporal_temperature,
    ).to(device)
    threshold = float(semantic_checkpoint["objectness_threshold"])
    baseline = evaluate(model, datasets["dev"], payloads["dev"]["cache"], device, batch_size=args.batch_size, top_k=args.top_k, objectness_threshold=threshold, amp=args.amp)
    counts = torch.bincount(payloads["train"]["cache"]["target_slot_labels"][:len(datasets["train"])].reshape(-1), minlength=NONE_LABEL + 1).float()
    median = counts[:NONE_LABEL].clamp_min(1).median()
    class_weights = (median / counts[:NONE_LABEL].clamp_min(1)).sqrt().clamp(0.5, 3.0).to(device)
    loader = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed), num_workers=0, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    best = baseline
    best_epoch = 0
    best_key = float(baseline["ranker_joint"]["f1_\u2191"])
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = weight_sum = 0.0
        for batch in loader:
            target = batch["target_labels"].long().to(device)
            iou = batch["target_ious"].float().to(device)
            valid = (target != NONE_LABEL) & (iou >= args.minimum_train_iou)
            if not bool(valid.any()):
                continue
            base = batch["base_logits"].to(device)
            candidate = candidates_with_gold(base, target, top_k=args.top_k, inject_gold=True)
            gold_position = (candidate == target.unsqueeze(-1)).float().argmax(dim=-1)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(args.amp and device.type == "cuda")):
                score = model(batch["grid"].to(device), batch["intervals"].to(device), batch["objectness"].to(device), base, candidate)
                element = F.cross_entropy(score[valid], gold_position[valid], reduction="none", label_smoothing=0.02)
                weight = batch["target_weights"].float().to(device)[valid] * class_weights[target[valid]]
                loss = (element * weight).sum() / weight.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer); scaler.update()
            loss_sum += float(loss.detach()) * float(weight.sum())
            weight_sum += float(weight.sum())
        scheduler.step()
        metrics = evaluate(model, datasets["dev"], payloads["dev"]["cache"], device, batch_size=args.batch_size, top_k=args.top_k, objectness_threshold=threshold, amp=args.amp)
        row = {"epoch": epoch, "train_loss": loss_sum / max(weight_sum, 1e-8), "learning_rate": optimizer.param_groups[0]["lr"], "dev": {key: value for key, value in metrics.items() if key != "items"}}
        history.append(row); print(json.dumps(row, sort_keys=True), flush=True)
        key = float(metrics["ranker_joint"]["f1_\u2191"])
        if key > best_key + 1e-4:
            best_key = key; best = metrics; best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True); break
    checkpoint_path = output_dir / "candidate_tf_matcher_v1_best.pt"
    _atomic_torch({"format": CHECKPOINT_FORMAT, "model_state_dict": best_state, "labels": semantic_checkpoint["labels"], "top_k": args.top_k, "config": {"attention_dim": args.attention_dim, "hidden_dim": args.hidden_dim, "dropout": args.dropout, "temporal_temperature": args.temporal_temperature}, "best_epoch": best_epoch}, checkpoint_path)
    base_top1 = float(baseline["baseline_top1_given_iou50_\u2191"])
    best_top1 = float(best["ranker_top1_given_iou50_\u2191"])
    gates = {"ranker_top1_given_iou50_ge_0_50": best_top1 >= 0.50, "ranker_top1_improves_by_ge_0_15": best_top1 >= base_top1 + 0.15, "ranker_joint_f1_ge_0_38": float(best["ranker_joint"]["f1_\u2191"]) >= 0.38}
    receipt = {"format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(), "complete": True, "paper_eligible": False, "claim_boundary": "source-disjoint development architecture screen", "qa_question_or_answer_used_as_input": False, "method": "Top20_label_conditioned_listwise_TF_attention_matcher", "data": {"train_scenes": len(datasets["train"]), "dev_scenes": len(datasets["dev"]), "classes": NONE_LABEL, "source_overlap": 0}, "baseline": baseline, "best_epoch": best_epoch, "best": best, "success_gates": gates, "decision": "proceed_to_event_graph_and_audiosep_render" if all(gates.values()) else "candidate_matcher_insufficient", "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256_file(checkpoint_path), "history": history, "inputs": {"train_grid_index_sha256": _sha256_file(args.train_grid_index.resolve()), "dev_grid_index_sha256": _sha256_file(args.dev_grid_index.resolve()), "base_semantic_checkpoint_sha256": _sha256_file(args.base_semantic_checkpoint.resolve()), "detector_checkpoint_sha256": _sha256_file(args.detector_checkpoint.resolve())}}
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "baseline": {key: value for key, value in baseline.items() if key != "items"}, "best_epoch": best_epoch, "best": {key: value for key, value in best.items() if key != "items"}, "gates": gates, "decision": receipt["decision"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
