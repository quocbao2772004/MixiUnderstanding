#!/usr/bin/env python3
"""Train a permutation-invariant event-count head on frozen event slots.

The head sees only frozen slot embeddings, objectness, and predicted temporal
geometry.  It never receives labels, questions, or QA answers.  At inference,
the predicted count selects the top-N objectness slots.  Checkpoint selection
uses the harmonic mean of exact-count accuracy on V5 main and V4 stress.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

from mixi_understanding.qces.relational_event_slots_v1 import interval_iou_matrix
from mixi_understanding.scripts.train_qces_qdor_dense import _atomic_json, _atomic_torch, _device, _sha256_file


FORMAT = "qces_v5_slot_count_training_receipt_v1"
CHECKPOINT_FORMAT = "qces_v5_slot_count_checkpoint_v1"


@dataclass(frozen=True)
class CountHeadConfig:
    slot_embedding_dim: int = 128
    hidden_dim: int = 128
    max_count: int = 8
    dropout: float = 0.10


class SlotCountHead(nn.Module):
    def __init__(self, config: CountHeadConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.slot_embedding_dim + 4
        self.slot_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.scene_head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 2),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.max_count + 1),
        )

    def forward(
        self, slot_embedding: torch.Tensor, objectness: torch.Tensor, intervals: torch.Tensor
    ) -> torch.Tensor:
        probability = objectness.clamp(1e-5, 1.0 - 1e-5)
        logit = torch.logit(probability).unsqueeze(-1)
        duration = (intervals[..., 1] - intervals[..., 0]).clamp_min(0.0).unsqueeze(-1)
        value = torch.cat((slot_embedding, logit, intervals, duration), dim=-1)
        encoded = self.slot_encoder(value)
        pooled = torch.cat((encoded.mean(dim=1), encoded.max(dim=1).values), dim=-1)
        return self.scene_head(pooled)


class CountDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, cache: Mapping[str, Any], *, slot_embedding_dim: int) -> None:
        self.embedding = cache["slot_input"][..., :slot_embedding_dim].float()
        self.objectness = cache["objectness"].float()
        self.intervals = cache["intervals"].float()
        self.count = torch.tensor([len(value) for value in cache["gold_intervals"]], dtype=torch.long)

    def __len__(self) -> int:
        return int(self.count.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "embedding": self.embedding[index],
            "objectness": self.objectness[index],
            "intervals": self.intervals[index],
            "count": self.count[index],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-train-cache", type=Path, required=True)
    parser.add_argument("--main-dev-cache", type=Path, required=True)
    parser.add_argument("--stress-train-cache", type=Path, required=True)
    parser.add_argument("--stress-dev-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7201)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--slot-embedding-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--main-sampling-ratio", type=float, default=0.75)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _harmonic(left: float, right: float) -> float:
    return 2.0 * left * right / max(left + right, 1e-12)


def _localization_metrics(cache: Mapping[str, Any], predicted_count: torch.Tensor) -> dict[str, float]:
    true_positive = predicted_total = gold_total = exact = absolute_error = 0
    for index, raw_count in enumerate(predicted_count.tolist()):
        objectness = cache["objectness"][index].float()
        count = max(1, min(int(objectness.shape[0]), int(raw_count)))
        keep = objectness.argsort(descending=True)[:count]
        predicted = cache["intervals"][index].float()[keep]
        gold = cache["gold_intervals"][index].float()
        predicted_total += len(predicted)
        gold_total += len(gold)
        exact += int(len(predicted) == len(gold))
        absolute_error += abs(len(predicted) - len(gold))
        iou = interval_iou_matrix(predicted, gold)
        rows, columns = linear_sum_assignment((1.0 - iou).numpy())
        true_positive += int((iou[rows, columns] >= 0.5).sum())
    precision = true_positive / max(predicted_total, 1)
    recall = true_positive / max(gold_total, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    scenes = len(predicted_count)
    return {
        "exact_count_accuracy_↑": exact / max(scenes, 1),
        "count_mae_↓": absolute_error / max(scenes, 1),
        "localization_precision_iou50_↑": precision,
        "localization_recall_iou50_↑": recall,
        "localization_f1_iou50_↑": f1,
    }


@torch.inference_mode()
def evaluate(
    model: SlotCountHead,
    dataset: CountDataset,
    cache: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: list[torch.Tensor] = []
    model.eval()
    for batch in loader:
        logits = model(
            batch["embedding"].to(device),
            batch["objectness"].to(device),
            batch["intervals"].to(device),
        )
        predictions.append(logits.argmax(dim=-1).clamp_min(1).cpu())
    return _localization_metrics(cache, torch.cat(predictions))


def _baseline(cache: Mapping[str, Any], mode: str) -> dict[str, float]:
    if mode == "threshold_0.9":
        count = (cache["objectness"].float() >= 0.9).sum(dim=-1).clamp_min(1)
    elif mode == "round_probability_sum":
        count = cache["objectness"].float().sum(dim=-1).round().long().clamp(1, 8)
    elif mode == "oracle_count":
        count = torch.tensor([len(value) for value in cache["gold_intervals"]])
    else:
        raise ValueError(mode)
    return _localization_metrics(cache, count)


def _selection(main: Mapping[str, float], stress: Mapping[str, float]) -> dict[str, Any]:
    main_exact = float(main["exact_count_accuracy_↑"])
    stress_exact = float(stress["exact_count_accuracy_↑"])
    main_f1 = float(main["localization_f1_iou50_↑"])
    stress_f1 = float(stress["localization_f1_iou50_↑"])
    return {
        "main": dict(main),
        "stress": dict(stress),
        "harmonic_exact_count_accuracy": _harmonic(main_exact, stress_exact),
        "minimum_exact_count_accuracy": min(main_exact, stress_exact),
        "harmonic_localization_f1": _harmonic(main_f1, stress_f1),
    }


def _key(row: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        float(row["harmonic_exact_count_accuracy"]),
        float(row["harmonic_localization_f1"]),
        float(row["minimum_exact_count_accuracy"]),
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.main_sampling_ratio < 1.0:
        raise ValueError("main-sampling-ratio must lie strictly in (0,1)")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "main_train": args.main_train_cache.resolve(),
        "main_dev": args.main_dev_cache.resolve(),
        "stress_train": args.stress_train_cache.resolve(),
        "stress_dev": args.stress_dev_cache.resolve(),
    }
    caches = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in paths.items()
    }
    config = CountHeadConfig(
        slot_embedding_dim=args.slot_embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    datasets = {
        name: CountDataset(cache, slot_embedding_dim=config.slot_embedding_dim)
        for name, cache in caches.items()
    }
    combined = ConcatDataset([datasets["main_train"], datasets["stress_train"]])
    main_length = len(datasets["main_train"])
    stress_length = len(datasets["stress_train"])
    weights = torch.tensor(
        [args.main_sampling_ratio / main_length] * main_length
        + [(1.0 - args.main_sampling_ratio) / stress_length] * stress_length,
        dtype=torch.double,
    )
    samples_per_epoch = round(main_length / args.main_sampling_ratio)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler)
    train_counts = torch.cat((datasets["main_train"].count, datasets["stress_train"].count))
    class_counts = torch.bincount(train_counts, minlength=config.max_count + 1).float()
    observed = class_counts > 0
    class_weights = torch.zeros_like(class_counts)
    class_weights[observed] = class_counts[observed].sum() / class_counts[observed]
    class_weights[observed] /= class_weights[observed].mean()

    device = _device(args.device)
    model = SlotCountHead(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.05
    )
    baselines = {
        mode: _selection(_baseline(caches["main_dev"], mode), _baseline(caches["stress_dev"], mode))
        for mode in ("threshold_0.9", "round_probability_sum", "oracle_count")
    }
    print(json.dumps({"baselines": baselines}, sort_keys=True), flush=True)
    best_selection: dict[str, Any] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = seen = 0
        for batch in loader:
            logits = model(
                batch["embedding"].to(device),
                batch["objectness"].to(device),
                batch["intervals"].to(device),
            )
            target = batch["count"].to(device)
            loss = F.cross_entropy(logits, target, weight=class_weights.to(device), label_smoothing=0.02)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = len(target)
            loss_sum += float(loss.detach()) * count
            correct += int((logits.argmax(dim=-1) == target).sum())
            seen += count
        scheduler.step()
        main_metrics = evaluate(model, datasets["main_dev"], caches["main_dev"], device, args.batch_size)
        stress_metrics = evaluate(model, datasets["stress_dev"], caches["stress_dev"], device, args.batch_size)
        selection = _selection(main_metrics, stress_metrics)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / max(seen, 1),
            "train_accuracy": correct / max(seen, 1),
            "selection": selection,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = _key(selection)
        if best_key is None or key > best_key:
            best_key = key
            best_selection = selection
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            print(json.dumps({"early_stop": True, "epoch": epoch, "best_epoch": best_epoch}), flush=True)
            break

    if best_selection is None or best_state is None:
        raise RuntimeError("training produced no checkpoint")
    checkpoint_path = output_dir / "slot_count_head_v1_best.pt"
    _atomic_torch(
        {
            "format": CHECKPOINT_FORMAT,
            "config": asdict(config),
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_selection": best_selection,
        },
        checkpoint_path,
    )
    threshold = baselines["threshold_0.9"]
    gates = {
        "harmonic_exact_count_improves_threshold": float(best_selection["harmonic_exact_count_accuracy"])
        > float(threshold["harmonic_exact_count_accuracy"]),
        "main_localization_f1_not_degraded": float(best_selection["main"]["localization_f1_iou50_↑"])
        >= float(threshold["main"]["localization_f1_iou50_↑"]),
        "stress_localization_f1_not_degraded": float(best_selection["stress"]["localization_f1_iou50_↑"])
        >= float(threshold["stress"]["localization_f1_iou50_↑"]),
    }
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_eligible": False,
        "qa_question_answer_or_event_label_used_as_input": False,
        "method": "permutation_invariant_frozen_slot_count_head_then_top_n_objectness",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "config": asdict(config),
        "cache": {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in paths.items()},
        "baselines": baselines,
        "best_epoch": best_epoch,
        "best_selection": best_selection,
        "success_gates": gates,
        "all_success_gates_pass": all(gates.values()),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "history": history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "best": best_selection, "gates": gates}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
