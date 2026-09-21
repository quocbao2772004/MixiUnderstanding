#!/usr/bin/env python3
"""Train a frequency-aware residual semantic adapter on frozen V4 slots."""

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
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.scripts.export_qces_beats_tf_grid_v1 import (
    FEATURE_DIM, FORMAT as GRID_FORMAT, FREQUENCY_PATCHES, TIME_PATCHES,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    _atomic_json, _atomic_torch, _device, _sha256_file,
)
from mixi_understanding.scripts.train_qces_v4_slot_semantic_head_v1 import (
    NONE_LABEL, ResidualSlotSemanticHead, _event_metrics,
)


FORMAT = "qces_tf_source_slot_semantic_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_tf_source_slot_semantic_checkpoint_v1"


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
    parser.add_argument("--output-dir", type=Path, default=base / "tf_source_slot_semantic_v1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=8201)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--attention-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--temporal-temperature", type=float, default=0.02)
    parser.add_argument("--none-class-weight", type=float, default=0.25)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_grid(index_path: Path) -> tuple[list[str], torch.Tensor, list[list[Mapping[str, Any]]], Mapping[str, Any]]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("format") != GRID_FORMAT or not index.get("complete"):
        raise ValueError(f"bad TF-grid index: {index_path}")
    scene_ids: list[str] = []
    features = []
    events: list[list[Mapping[str, Any]]] = []
    for descriptor in index["shards"]:
        payload = torch.load(index_path.parent / descriptor["path"], map_location="cpu", weights_only=False)
        scene_ids.extend(str(value) for value in payload["scene_ids"])
        features.append(payload["grid_features"])
        events.extend(payload["gold_events"])
    grid = torch.cat(features)
    expected = (len(scene_ids), TIME_PATCHES, FREQUENCY_PATCHES, FEATURE_DIM)
    if tuple(grid.shape) != expected:
        raise ValueError(f"bad TF grid shape: {tuple(grid.shape)}")
    return scene_ids, grid, events, index


@torch.inference_mode()
def base_logits(slot_cache: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> torch.Tensor:
    model = ResidualSlotSemanticHead(
        int(checkpoint["input_dim"]), int(checkpoint["hidden_dim"]),
        NONE_LABEL, float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    values = slot_cache["slot_input"]
    return torch.cat([model(values[i : i + 256]).cpu() for i in range(0, len(values), 256)])


class TFSlotDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self, grid: torch.Tensor, slot_cache: Mapping[str, Any], frozen_base_logits: torch.Tensor,
        max_scenes: int = 0,
    ) -> None:
        count = min(len(grid), max_scenes) if max_scenes else len(grid)
        self.grid = grid[:count]
        self.slot_input = slot_cache["slot_input"][:count]
        self.intervals = slot_cache["intervals"][:count]
        self.objectness = slot_cache["objectness"][:count]
        self.labels = slot_cache["target_slot_labels"][:count]
        self.weights = slot_cache["target_slot_weights"][:count]
        self.base_logits = frozen_base_logits[:count]

    def __len__(self) -> int:
        return len(self.grid)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "grid": self.grid[index], "slot_input": self.slot_input[index],
            "intervals": self.intervals[index], "objectness": self.objectness[index],
            "labels": self.labels[index], "weights": self.weights[index],
            "base_logits": self.base_logits[index],
        }


class TFSourceSlotSemanticV1(nn.Module):
    def __init__(
        self, *, slot_dim: int, attention_dim: int, hidden_dim: int,
        dropout: float, temporal_temperature: float,
        r1_weight: torch.Tensor, r1_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.temporal_temperature = float(temporal_temperature)
        self.grid_norm = nn.LayerNorm(FEATURE_DIM)
        self.key = nn.Linear(FEATURE_DIM, attention_dim)
        self.value = nn.Linear(FEATURE_DIM, attention_dim)
        self.query = nn.Sequential(
            nn.LayerNorm(128 + 3), nn.Linear(128 + 3, attention_dim), nn.GELU(),
        )
        residual_input = (NONE_LABEL + 1) + NONE_LABEL + attention_dim + 128 + 3 + 1
        self.residual = nn.Sequential(
            nn.LayerNorm(residual_input),
            nn.Linear(residual_input, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, NONE_LABEL + 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer("r1_weight", r1_weight.float().clone(), persistent=True)
        self.register_buffer("r1_bias", r1_bias.float().clone(), persistent=True)
        self.slot_dim = int(slot_dim)

    def temporal_prior(self, intervals: torch.Tensor) -> torch.Tensor:
        centers = (torch.arange(TIME_PATCHES, device=intervals.device, dtype=intervals.dtype) + 0.5) / TIME_PATCHES
        start = intervals[..., 0, None]
        end = intervals[..., 1, None]
        tau = self.temporal_temperature
        prior = torch.sigmoid((centers - start) / tau) * torch.sigmoid((end - centers) / tau)
        return prior.clamp_min(1e-5)

    def forward(
        self, grid: torch.Tensor, slot_input: torch.Tensor, intervals: torch.Tensor,
        objectness: torch.Tensor, frozen_base_logits: torch.Tensor,
    ) -> torch.Tensor:
        raw_grid = grid.float()
        grid = self.grid_norm(raw_grid)
        slot_embedding = slot_input[..., :128].float()
        width = (intervals[..., 1] - intervals[..., 0]).unsqueeze(-1)
        geometry = torch.cat((intervals.float(), width.float()), dim=-1)
        query = self.query(torch.cat((slot_embedding, geometry), dim=-1))
        key = self.key(grid)
        value = self.value(grid)
        attention = torch.einsum("bsd,btfd->bstf", query, key) / math.sqrt(key.shape[-1])
        prior = self.temporal_prior(intervals.float())
        attention = attention + prior.log().unsqueeze(-1)
        attention = attention.flatten(-2).softmax(dim=-1).reshape_as(attention)
        pooled = torch.einsum("bstf,btfd->bsd", attention, value)

        token_logits = F.linear(raw_grid, self.r1_weight, self.r1_bias)
        frequency_lme = torch.logsumexp(token_logits, dim=2) - math.log(FREQUENCY_PATCHES)
        time_weight = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        fixed_tf_scores = torch.einsum("bst,btc->bsc", time_weight, frequency_lme)
        context = torch.cat((
            frozen_base_logits.float(), fixed_tf_scores, pooled, slot_embedding,
            geometry, objectness.float().unsqueeze(-1),
        ), dim=-1)
        return frozen_base_logits.float() + self.residual(context)


@torch.inference_mode()
def predict(
    model: TFSourceSlotSemanticV1, dataset: TFSlotDataset, device: torch.device,
    *, batch_size: int, amp: bool,
) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    parts = []
    model.eval()
    for batch in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            logits = model(
                batch["grid"].to(device), batch["slot_input"].to(device),
                batch["intervals"].to(device), batch["objectness"].to(device),
                batch["base_logits"].to(device),
            )
        parts.append(logits.cpu())
    return torch.cat(parts)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_payload: dict[str, dict[str, Any]] = {}
    for split, grid_arg, cache_arg, max_scenes in (
        ("train", args.train_grid_index, args.train_slot_cache, args.max_train_scenes),
        ("dev", args.dev_grid_index, args.dev_slot_cache, args.max_dev_scenes),
    ):
        ids, grid, events, index = load_grid(grid_arg.resolve())
        cache = torch.load(cache_arg.resolve(), map_location="cpu", weights_only=False)
        if ids != [str(value) for value in cache["scene_id"]]:
            raise ValueError(f"{split}: TF-grid/slot scene order mismatch")
        split_payload[split] = {"ids": ids, "grid": grid, "events": events, "index": index, "cache": cache, "max": max_scenes}
    train_sources = {str(event["source_id"]) for scene in split_payload["train"]["events"] for event in scene}
    dev_sources = {str(event["source_id"]) for scene in split_payload["dev"]["events"] for event in scene}
    overlap = train_sources & dev_sources
    if overlap:
        raise ValueError(f"train/dev source leakage: {len(overlap)}")

    semantic_checkpoint = torch.load(args.base_semantic_checkpoint.resolve(), map_location="cpu", weights_only=True)
    detector_checkpoint = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=False)
    if semantic_checkpoint["labels"] != detector_checkpoint["labels"]:
        raise ValueError("semantic/detector label mismatch")
    bases = {split: base_logits(payload["cache"], semantic_checkpoint) for split, payload in split_payload.items()}
    datasets = {
        split: TFSlotDataset(payload["grid"], payload["cache"], bases[split], payload["max"])
        for split, payload in split_payload.items()
    }
    device = _device(args.device)
    r1_state = detector_checkpoint["model_state_dict"]
    model = TFSourceSlotSemanticV1(
        slot_dim=int(split_payload["train"]["cache"]["slot_input"].shape[-1]),
        attention_dim=args.attention_dim, hidden_dim=args.hidden_dim,
        dropout=args.dropout, temporal_temperature=args.temporal_temperature,
        r1_weight=r1_state["strong_head.weight"], r1_bias=r1_state["strong_head.bias"],
    ).to(device)
    objectness_threshold = float(semantic_checkpoint["objectness_threshold"])
    baseline_metrics = _event_metrics(
        split_payload["dev"]["cache"], bases["dev"][: len(datasets["dev"])],
        objectness_threshold=objectness_threshold, learned=True,
    )
    initial_logits = predict(model, datasets["dev"], device, batch_size=args.batch_size, amp=args.amp)
    initial_metrics = _event_metrics(
        split_payload["dev"]["cache"], initial_logits,
        objectness_threshold=objectness_threshold, learned=True,
    )
    if initial_metrics != baseline_metrics:
        raise RuntimeError("zero residual does not exactly reproduce base semantic head")

    train_cache = split_payload["train"]["cache"]
    counts = torch.bincount(
        train_cache["target_slot_labels"][: len(datasets["train"])].reshape(-1),
        minlength=NONE_LABEL + 1,
    ).float()
    median = counts[:NONE_LABEL].clamp_min(1).median()
    class_weights = (median / counts[:NONE_LABEL].clamp_min(1)).sqrt().clamp(0.5, 3.0)
    class_weights = torch.cat((class_weights, torch.tensor([args.none_class_weight]))).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        generator=generator, num_workers=0, pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device.type == "cuda"))
    best_metrics = baseline_metrics
    best_epoch = 0
    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best_key = (float(best_metrics["joint_label_iou50_f1_\u2191"]), float(best_metrics["label_top1_given_iou50_\u2191"]))
    history = []
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_slots = 0
        for batch in train_loader:
            target = batch["labels"].long().to(device)
            match_weight = batch["weights"].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(args.amp and device.type == "cuda")):
                logits = model(
                    batch["grid"].to(device), batch["slot_input"].to(device),
                    batch["intervals"].to(device), batch["objectness"].to(device),
                    batch["base_logits"].to(device),
                )
                element = F.cross_entropy(
                    logits.reshape(-1, NONE_LABEL + 1), target.reshape(-1),
                    weight=class_weights, reduction="none", label_smoothing=0.02,
                ).reshape_as(target)
                loss = (element * match_weight).sum() / match_weight.sum().clamp_min(1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * target.numel()
            total_slots += int(target.numel())
        scheduler.step()
        dev_logits = predict(model, datasets["dev"], device, batch_size=args.batch_size, amp=args.amp)
        metrics = _event_metrics(
            split_payload["dev"]["cache"], dev_logits,
            objectness_threshold=objectness_threshold, learned=True,
        )
        row = {"epoch": epoch, "train_loss": total_loss / max(total_slots, 1), "learning_rate": optimizer.param_groups[0]["lr"], "dev": metrics}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (float(metrics["joint_label_iou50_f1_\u2191"]), float(metrics["label_top1_given_iou50_\u2191"]))
        if key > best_key:
            best_key = key
            best_metrics = metrics
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    checkpoint_path = output_dir / "tf_source_slot_semantic_v1_best.pt"
    _atomic_torch({
        "format": CHECKPOINT_FORMAT, "labels": semantic_checkpoint["labels"],
        "model_state_dict": best_state, "best_epoch": best_epoch,
        "best_metrics": best_metrics, "objectness_threshold": objectness_threshold,
        "config": {
            "attention_dim": args.attention_dim, "hidden_dim": args.hidden_dim,
            "dropout": args.dropout, "temporal_temperature": args.temporal_temperature,
        },
    }, checkpoint_path)
    gates = {
        "label_top1_given_iou50_ge_0_45": float(best_metrics["label_top1_given_iou50_\u2191"]) >= 0.45,
        "label_top1_improves_base_by_ge_0_08": float(best_metrics["label_top1_given_iou50_\u2191"]) >= float(baseline_metrics["label_top1_given_iou50_\u2191"]) + 0.08,
        "joint_f1_ge_0_34": float(best_metrics["joint_label_iou50_f1_\u2191"]) >= 0.34,
    }
    receipt = {
        "format": FORMAT, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True, "paper_eligible": False,
        "claim_boundary": "source-disjoint development architecture screen",
        "qa_question_or_answer_used_as_input": False,
        "method": "frozen_BEATs_TxF_grid_plus_slot_cross_attention_and_residual_base_semantic_head",
        "source_identity_overlap_train_dev": len(overlap),
        "data": {"train_scenes": len(datasets["train"]), "dev_scenes": len(datasets["dev"]), "classes": NONE_LABEL},
        "baseline_metrics": baseline_metrics, "best_epoch": best_epoch,
        "best_metrics": best_metrics, "success_gates": gates,
        "decision": "proceed_to_event_graph" if all(gates.values()) else "frequency_adapter_insufficient_proceed_to_pit_source_separation",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": _sha256_file(checkpoint_path),
        "history": history,
        "inputs": {
            "train_grid_index_sha256": _sha256_file(args.train_grid_index.resolve()),
            "dev_grid_index_sha256": _sha256_file(args.dev_grid_index.resolve()),
            "base_semantic_checkpoint_sha256": _sha256_file(args.base_semantic_checkpoint.resolve()),
            "detector_checkpoint_sha256": _sha256_file(args.detector_checkpoint.resolve()),
        },
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "baseline": baseline_metrics, "best_epoch": best_epoch, "best": best_metrics, "gates": gates, "decision": receipt["decision"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
