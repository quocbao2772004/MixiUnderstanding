#!/usr/bin/env python3
"""Train a prompt-conditioned temporal grounder on frozen AudioSep outputs.

This is the next step after the structured-planner diagnostic:

    structured question fields + event inventory -> canonical text prompt
    frozen AudioSep(prompt, mixture) -> ungated evidence
    learned temporal grounder -> gate

The grounder is trained only against annotated temporal masks.  At evaluation
time it does not consume target waveform, gold answer, or gold evidence IDs.
The planner/prompt still uses the annotation-provided event inventory, so the
result is a semi-oracle diagnostic rather than the final full system.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.qces.signal import qces_linear_interpolate_1d
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    _load_separator,
    _v5_item_metrics,
    encode_prompts,
    summarize_v5_items,
    write_item_artifacts,
)
from mixi_understanding.scripts.evaluate_qces_v5_structured_planner_audiosep import (
    MODE_PLANNER_TEXT_GATE,
    MODE_PLANNER_TEXT_NO_GATE,
    Plan,
    metadata as planner_metadata,
    plan_record,
    planned_gate,
)


FORMAT_VERSION = "qces_v5_temporal_grounder_audiosep_v1"
CACHE_FORMAT = "qces_v5_temporal_grounder_audiosep_cache_v1"
MODE_SOFT_GATE = "temporal_grounder__soft_gate"
MODE_HARD_GATE = "temporal_grounder__hard_gate_train_threshold"
MODE_REGISTRY = {
    MODE_PLANNER_TEXT_NO_GATE: {
        "description": "Structured planner text prompt, no temporal gate.",
        "uses_learned_time": False,
        "uses_oracle_time": False,
    },
    MODE_PLANNER_TEXT_GATE: {
        "description": "Structured planner text prompt, annotation-planned event spans.",
        "uses_learned_time": False,
        "uses_oracle_time": True,
    },
    MODE_SOFT_GATE: {
        "description": "Structured planner text prompt, learned soft temporal gate.",
        "uses_learned_time": True,
        "uses_oracle_time": False,
    },
    MODE_HARD_GATE: {
        "description": "Structured planner text prompt, learned gate threshold selected on train.",
        "uses_learned_time": True,
        "uses_oracle_time": False,
    },
}
RELATIONS = ("after", "before", "first")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--audiosep-root", type=Path, required=True)
    parser.add_argument("--audiosep-config", type=Path, required=True)
    parser.add_argument("--audiosep-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--tversky-weight", type=float, default=0.5)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-val-records", type=int, default=0)
    parser.add_argument("--render-item-id", action="append", default=[])
    parser.add_argument("--no-render-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args(argv)
    if args.frames <= 0:
        parser.error("--frames must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.text_batch_size <= 0:
        parser.error("--text-batch-size must be positive")
    if args.max_train_records < 0 or args.max_val_records < 0:
        parser.error("--max-*-records must be non-negative")
    if len(args.render_item_id) != len(set(args.render_item_id)):
        parser.error("--render-item-id values must be unique")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_records(dataset: QCESManifestDataset, limit: int) -> list[QCESV5Record]:
    records = [
        record for record in dataset.records if isinstance(record, QCESV5Record)
    ]
    if len(records) != len(dataset.records):
        raise SystemExit("temporal grounder requires a pure QCES-v5 manifest")
    return records[:limit] if limit else records


def cache_identity(args: argparse.Namespace, manifest: Path, records: Sequence[QCESV5Record]) -> dict[str, Any]:
    return {
        "format": CACHE_FORMAT,
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest.resolve()),
        "record_ids": [record.sample_id for record in records],
        "audiosep_checkpoint": {
            "path": str(args.audiosep_checkpoint.resolve()),
            "sha256": sha256_file(args.audiosep_checkpoint.resolve()),
            "size_bytes": args.audiosep_checkpoint.resolve().stat().st_size,
        },
    }


def identity_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def build_or_load_audiosep_cache(
    *,
    args: argparse.Namespace,
    split: str,
    dataset: QCESManifestDataset,
    records: Sequence[QCESV5Record],
    device: torch.device,
) -> dict[str, Any]:
    cache_path = args.cache_dir.resolve() / f"{split}_raw_audiosep.pt"
    expected_identity = cache_identity(args, dataset.manifest_path, records)
    if cache_path.is_file() and not args.rebuild_cache:
        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict) and identity_matches(
            payload.get("identity", {}), expected_identity
        ):
            return payload
    args.cache_dir.resolve().mkdir(parents=True, exist_ok=True)

    plans = {record.sample_id: plan_record(record) for record in records}
    prompts = sorted(
        {
            plan.prompt
            for plan in plans.values()
            if not plan.no_evidence and plan.prompt
        }
    )
    prompt_embeddings = (
        encode_prompts(
            args.audiosep_root.resolve(),
            args.audiosep_checkpoint.resolve(),
            prompts,
            batch_size=args.text_batch_size,
        )
        if prompts
        else {}
    )
    separator = _load_separator(args, device) if prompts else None
    raw_by_id: dict[str, torch.Tensor] = {}
    plan_by_id: dict[str, dict[str, Any]] = {}
    separator_calls = 0
    with torch.inference_mode():
        for index, record in enumerate(records):
            example = dataset[index]
            plan = plans[record.sample_id]
            if plan.no_evidence:
                raw = torch.zeros_like(example.mixture)
            else:
                assert separator is not None
                mixture = example.mixture.to(device)
                condition = prompt_embeddings[plan.prompt][None].to(device)
                raw = separator(
                    {"mixture": mixture[None, None], "condition": condition}
                )["waveform"][0, 0].detach().cpu()
                separator_calls += 1
            raw_by_id[record.sample_id] = raw.to(torch.float16)
            plan_by_id[record.sample_id] = {
                "event_ids": list(plan.event_ids),
                "labels": list(plan.labels),
                "prompt": plan.prompt,
                "no_evidence": plan.no_evidence,
                "reason": plan.reason,
                "relation": plan.relation,
                "planned_answer_label": plan.planned_answer_label,
            }
    payload = {
        "identity": expected_identity,
        "split": split,
        "separator_calls": separator_calls,
        "plans": plan_by_id,
        "raw_audiosep_by_id": raw_by_id,
    }
    torch.save(payload, cache_path)
    return payload


def frame_pool(signal: torch.Tensor, frames: int, mode: str = "avg") -> torch.Tensor:
    value = signal.float()[None, None]
    if mode == "max":
        return F.adaptive_max_pool1d(value.abs(), frames)[0, 0]
    if mode == "avg_abs":
        return F.adaptive_avg_pool1d(value.abs(), frames)[0, 0]
    if mode == "avg_power":
        return F.adaptive_avg_pool1d(value.square(), frames)[0, 0]
    raise ValueError(mode)


def union_mask_frames(example: Any, frames: int) -> torch.Tensor:
    union = (example.anchor_mask + example.answer_mask).clamp_max(1.0)
    return F.adaptive_max_pool1d(union[None, None].float(), frames)[0, 0]


def relation_one_hot(relation: str, frames: int) -> torch.Tensor:
    values = torch.zeros(len(RELATIONS), frames, dtype=torch.float32)
    if relation in RELATIONS:
        values[RELATIONS.index(relation)].fill_(1.0)
    return values


def repeated_scalar(value: float, frames: int) -> torch.Tensor:
    return torch.full((1, frames), float(value), dtype=torch.float32)


def frame_features(
    *,
    mixture: torch.Tensor,
    raw_audiosep: torch.Tensor,
    record: QCESV5Record,
    plan: Plan,
    frames: int,
) -> torch.Tensor:
    raw_abs = frame_pool(raw_audiosep, frames, "avg_abs")
    raw_power = frame_pool(raw_audiosep, frames, "avg_power")
    raw_peak = frame_pool(raw_audiosep, frames, "max")
    mix_abs = frame_pool(mixture, frames, "avg_abs")
    mix_power = frame_pool(mixture, frames, "avg_power")
    eps = 1e-8
    log_raw_abs = torch.log(raw_abs + eps)
    log_raw_power = torch.log(raw_power + eps)
    log_raw_peak = torch.log(raw_peak + eps)
    log_mix_abs = torch.log(mix_abs + eps)
    log_mix_power = torch.log(mix_power + eps)
    log_ratio = torch.log((raw_power + eps) / (mix_power + eps))
    derivative = F.pad(log_raw_abs[1:] - log_raw_abs[:-1], (1, 0))
    time = torch.linspace(-1.0, 1.0, frames)
    prompt_label_count = len(plan.labels)
    ordinal = record.query_instance_ordinal or 0
    candidate_count = len(record.query_candidate_labels)
    return torch.cat(
        [
            log_raw_abs[None],
            log_raw_power[None],
            log_raw_peak[None],
            log_mix_abs[None],
            log_mix_power[None],
            log_ratio[None],
            derivative[None],
            time[None],
            torch.sin(math.pi * time)[None],
            torch.cos(math.pi * time)[None],
            relation_one_hot(record.relation, frames),
            repeated_scalar(1.0 if plan.no_evidence else 0.0, frames),
            repeated_scalar(float(ordinal) / 5.0, frames),
            repeated_scalar(float(prompt_label_count) / 4.0, frames),
            repeated_scalar(float(candidate_count) / 4.0, frames),
        ],
        dim=0,
    )


@dataclass(frozen=True)
class FeaturePack:
    sample_id: str
    features: torch.Tensor
    target: torch.Tensor
    no_evidence: bool


class GrounderFeatureDataset(Dataset[FeaturePack]):
    def __init__(self, packs: Sequence[FeaturePack]) -> None:
        self.packs = list(packs)

    def __len__(self) -> int:
        return len(self.packs)

    def __getitem__(self, index: int) -> FeaturePack:
        return self.packs[index]


def collate_packs(packs: Sequence[FeaturePack]) -> dict[str, Any]:
    return {
        "sample_id": [pack.sample_id for pack in packs],
        "features": torch.stack([pack.features for pack in packs]),
        "target": torch.stack([pack.target for pack in packs]),
        "no_evidence": torch.tensor(
            [float(pack.no_evidence) for pack in packs], dtype=torch.float32
        ),
    }


def make_feature_packs(
    *,
    dataset: QCESManifestDataset,
    records: Sequence[QCESV5Record],
    cache: Mapping[str, Any],
    frames: int,
) -> list[FeaturePack]:
    raw_by_id = cache["raw_audiosep_by_id"]
    packs: list[FeaturePack] = []
    for index, record in enumerate(records):
        example = dataset[index]
        plan = plan_record(record)
        raw = raw_by_id[record.sample_id].float()
        features = frame_features(
            mixture=example.mixture,
            raw_audiosep=raw,
            record=record,
            plan=plan,
            frames=frames,
        )
        target = union_mask_frames(example, frames)
        packs.append(
            FeaturePack(
                sample_id=record.sample_id,
                features=features,
                target=target,
                no_evidence=record.no_evidence,
            )
        )
    return packs


def feature_normalization(packs: Sequence[FeaturePack]) -> tuple[torch.Tensor, torch.Tensor]:
    stacked = torch.stack([pack.features for pack in packs])
    mean = stacked.mean(dim=(0, 2), keepdim=False)
    std = stacked.std(dim=(0, 2), keepdim=False).clamp_min(1e-4)
    return mean, std


def apply_normalization(
    packs: Sequence[FeaturePack], mean: torch.Tensor, std: torch.Tensor
) -> list[FeaturePack]:
    result: list[FeaturePack] = []
    for pack in packs:
        features = (pack.features - mean[:, None]) / std[:, None]
        result.append(
            FeaturePack(
                sample_id=pack.sample_id,
                features=features,
                target=pack.target,
                no_evidence=pack.no_evidence,
            )
        )
    return result


class TemporalGrounder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=5,
                        padding=2 * dilation,
                        dilation=dilation,
                    ),
                    nn.GroupNorm(8, hidden_dim),
                    nn.SiLU(),
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                )
                for dilation in (1, 2, 4, 8, 16)
            ]
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.10,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input(features)
        for block in self.convs:
            hidden = hidden + block(hidden)
        sequence = hidden.transpose(1, 2)
        sequence, _ = self.gru(sequence)
        return self.output(sequence).squeeze(-1)


def dice_loss(probability: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = 2.0 * (probability * target).sum(dim=1) + 1e-6
    denominator = probability.sum(dim=1) + target.sum(dim=1) + 1e-6
    return (1.0 - numerator / denominator).mean()


def tversky_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 0.35,
    beta: float = 0.65,
) -> torch.Tensor:
    true_positive = (probability * target).sum(dim=1)
    false_positive = (probability * (1.0 - target)).sum(dim=1)
    false_negative = ((1.0 - probability) * target).sum(dim=1)
    score = (true_positive + 1e-6) / (
        true_positive + alpha * false_positive + beta * false_negative + 1e-6
    )
    return (1.0 - score).mean()


def loss_fn(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    pos_weight: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
    tversky_weight: float,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight
    )
    probability = torch.sigmoid(logits)
    return (
        bce_weight * bce
        + dice_weight * dice_loss(probability, target)
        + tversky_weight * tversky_loss(probability, target)
    )


def frame_metrics(
    probability: torch.Tensor, target: torch.Tensor, threshold: float
) -> dict[str, float]:
    prediction = probability >= threshold
    expected = target >= 0.5
    tp = (prediction & expected).sum().float()
    fp = (prediction & ~expected).sum().float()
    fn = (~prediction & expected).sum().float()
    union = (prediction | expected).sum().float()
    if int(union) == 0:
        return {
            "frame_iou_↑": 1.0,
            "frame_precision_↑": 1.0,
            "frame_recall_↑": 1.0,
            "frame_f1_↑": 1.0,
        }
    iou = torch.where(union > 0, tp / union.clamp_min(1.0), union.new_tensor(1.0))
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "frame_iou_↑": float(iou),
        "frame_precision_↑": float(precision),
        "frame_recall_↑": float(recall),
        "frame_f1_↑": float(f1),
    }


def run_grounder(
    model: nn.Module,
    packs: Sequence[FeaturePack],
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    loader = DataLoader(
        GrounderFeatureDataset(packs),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_packs,
    )
    outputs: dict[str, torch.Tensor] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device)
            probability = torch.sigmoid(model(features)).detach().cpu()
            for sample_id, sample_probability in zip(batch["sample_id"], probability):
                outputs[sample_id] = sample_probability
    return outputs


def aggregate_frame_metrics(
    probabilities: Mapping[str, torch.Tensor],
    packs: Sequence[FeaturePack],
    threshold: float,
) -> dict[str, float]:
    rows = []
    for pack in packs:
        rows.append(frame_metrics(probabilities[pack.sample_id], pack.target, threshold))

    def mean(key: str) -> float:
        return float(sum(row[key] for row in rows) / max(len(rows), 1))

    return {
        "frame_iou_mean_↑": mean("frame_iou_↑"),
        "frame_precision_mean_↑": mean("frame_precision_↑"),
        "frame_recall_mean_↑": mean("frame_recall_↑"),
        "frame_f1_mean_↑": mean("frame_f1_↑"),
    }


def choose_threshold(
    probabilities: Mapping[str, torch.Tensor], packs: Sequence[FeaturePack]
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    best_score = -1.0
    for threshold in [i / 20 for i in range(1, 20)]:
        metrics = aggregate_frame_metrics(probabilities, packs, threshold)
        score = metrics["frame_f1_mean_↑"]
        if score > best_score:
            best_threshold = threshold
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


def train_model(
    *,
    args: argparse.Namespace,
    train_packs: Sequence[FeaturePack],
    val_packs: Sequence[FeaturePack],
    device: torch.device,
) -> tuple[TemporalGrounder, dict[str, Any]]:
    torch.manual_seed(args.seed)
    model = TemporalGrounder(train_packs[0].features.size(0), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positive = torch.stack([pack.target for pack in train_packs]).sum()
    total = torch.tensor(float(len(train_packs) * train_packs[0].target.numel()))
    pos_weight = ((total - positive) / positive.clamp_min(1.0)).clamp(1.0, 20.0).to(device)
    loader = DataLoader(
        GrounderFeatureDataset(train_packs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_packs,
        generator=torch.Generator().manual_seed(args.seed),
    )
    best_state = None
    best_score = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            features = batch["features"].to(device)
            target = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(
                logits,
                target,
                pos_weight=pos_weight,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                tversky_weight=args.tversky_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_prob = run_grounder(model, train_packs, device, args.batch_size)
        threshold, train_frame = choose_threshold(train_prob, train_packs)
        val_prob = run_grounder(model, val_packs, device, args.batch_size)
        val_frame = aggregate_frame_metrics(val_prob, val_packs, threshold)
        score = val_frame["frame_f1_mean_↑"]
        row = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / max(len(losses), 1)),
            "threshold": threshold,
            "train_frame": train_frame,
            "val_frame": val_frame,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    train_prob = run_grounder(model, train_packs, device, args.batch_size)
    threshold, train_frame = choose_threshold(train_prob, train_packs)
    val_prob = run_grounder(model, val_packs, device, args.batch_size)
    val_frame = aggregate_frame_metrics(val_prob, val_packs, threshold)
    training_report = {
        "history": history,
        "selected_threshold": threshold,
        "selected_by": "maximum validation frame F1 during training; threshold selected by train frame F1",
        "train_frame_at_selected_threshold": train_frame,
        "val_frame_at_selected_threshold": val_frame,
        "pos_weight": float(pos_weight.detach().cpu()),
    }
    return model, training_report


def gate_to_samples(probability_frames: torch.Tensor, samples: int) -> torch.Tensor:
    return qces_linear_interpolate_1d(
        probability_frames[None, None].float(), samples
    )[0, 0].clamp(0.0, 1.0)


def evaluate_waveforms(
    *,
    args: argparse.Namespace,
    dataset: QCESManifestDataset,
    records: Sequence[QCESV5Record],
    cache: Mapping[str, Any],
    probabilities: Mapping[str, torch.Tensor],
    threshold: float,
    output_dir: Path,
    split: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_by_id = cache["raw_audiosep_by_id"]
    render_ids = set(args.render_item_id)
    items: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        example = dataset[index]
        plan = plan_record(record)
        mixture = example.mixture.to(device)
        target = example.evidence.to(device)
        target_residual = example.residual.to(device)
        raw = raw_by_id[record.sample_id].float().to(device)
        planned = planned_gate(record, plan, mixture.numel(), device, 0.0)
        soft = gate_to_samples(probabilities[record.sample_id].to(device), mixture.numel())
        hard = (soft >= threshold).float()
        mode_to_evidence = {
            MODE_PLANNER_TEXT_NO_GATE: raw,
            MODE_PLANNER_TEXT_GATE: raw * planned,
            MODE_SOFT_GATE: raw * soft,
            MODE_HARD_GATE: raw * hard,
        }
        if plan.no_evidence:
            mode_to_evidence = {
                mode: torch.zeros_like(mixture) for mode in mode_to_evidence
            }
        frame_target = union_mask_frames(example, args.frames)
        frame_probability = probabilities[record.sample_id]
        frame_metric = frame_metrics(frame_probability, frame_target, threshold)
        for mode, evidence in mode_to_evidence.items():
            residual = mixture - evidence
            metrics, descriptives = _v5_item_metrics(
                no_evidence=record.no_evidence,
                evidence=evidence,
                mixture=mixture,
                target=target,
                target_residual=target_residual,
            )
            item = {
                **planner_metadata(record, plan),
                "split_evaluated": split,
                "mode": mode,
                "baseline_access": "structured_question_fields_plus_oracle_event_inventory_for_prompt_only",
                "prompt": plan.prompt,
                "temporal_grounder_threshold": threshold,
                "frame_metrics": frame_metric,
                "metrics": metrics,
                "descriptives": descriptives,
            }
            should_render = not args.no_render_audio and (
                not render_ids or record.sample_id in render_ids
            )
            write_item_artifacts(
                question_dir=(
                    output_dir
                    / split
                    / mode
                    / record.scene_id
                    / f"q{record.question_index}_{record.question_type}"
                ),
                item=item,
                evidence=evidence,
                residual=residual,
                sample_rate=record.sample_rate,
                render_audio=should_render,
            )
            items.append(item)
    return items, summarize_by_mode(items)


def summarize_by_mode(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in sorted({str(item["mode"]) for item in items}):
        subset = [item for item in items if str(item["mode"]) == mode]
        summary = summarize_v5_items(subset)
        frame_keys = (
            "frame_iou_↑",
            "frame_precision_↑",
            "frame_recall_↑",
            "frame_f1_↑",
        )
        for key in frame_keys:
            values = [float(item["frame_metrics"][key]) for item in subset]
            summary[f"{key[:-1]}mean_↑"] = float(sum(values) / max(len(values), 1))
        result[mode] = summary
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = QCESManifestDataset(args.train_manifest.resolve(), crop_samples=None)
    val_dataset = QCESManifestDataset(args.val_manifest.resolve(), crop_samples=None)
    train_records = select_records(train_dataset, args.max_train_records)
    val_records = select_records(val_dataset, args.max_val_records)
    unknown_render = sorted(set(args.render_item_id) - {record.sample_id for record in val_records})
    if unknown_render:
        raise SystemExit("render IDs are absent from val selection: " + ", ".join(unknown_render))

    train_cache = build_or_load_audiosep_cache(
        args=args,
        split="train",
        dataset=train_dataset,
        records=train_records,
        device=device,
    )
    val_cache = build_or_load_audiosep_cache(
        args=args,
        split="val",
        dataset=val_dataset,
        records=val_records,
        device=device,
    )
    train_packs = make_feature_packs(
        dataset=train_dataset,
        records=train_records,
        cache=train_cache,
        frames=args.frames,
    )
    val_packs = make_feature_packs(
        dataset=val_dataset,
        records=val_records,
        cache=val_cache,
        frames=args.frames,
    )
    mean, std = feature_normalization(train_packs)
    train_packs = apply_normalization(train_packs, mean, std)
    val_packs = apply_normalization(val_packs, mean, std)
    model, training_report = train_model(
        args=args,
        train_packs=train_packs,
        val_packs=val_packs,
        device=device,
    )
    threshold = float(training_report["selected_threshold"])
    train_probabilities = run_grounder(model, train_packs, device, args.batch_size)
    val_probabilities = run_grounder(model, val_packs, device, args.batch_size)
    checkpoint_path = output_dir / "temporal_grounder_checkpoint.pt"
    torch.save(
        {
            "format": FORMAT_VERSION,
            "model_state_dict": model.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "input_dim": train_packs[0].features.size(0),
            "hidden_dim": args.hidden_dim,
            "frames": args.frames,
            "threshold": threshold,
            "training_report": training_report,
        },
        checkpoint_path,
    )
    train_items, train_summary = evaluate_waveforms(
        args=args,
        dataset=train_dataset,
        records=train_records,
        cache=train_cache,
        probabilities=train_probabilities,
        threshold=threshold,
        output_dir=output_dir,
        split="train",
        device=device,
    )
    val_items, val_summary = evaluate_waveforms(
        args=args,
        dataset=val_dataset,
        records=val_records,
        cache=val_cache,
        probabilities=val_probabilities,
        threshold=threshold,
        output_dir=output_dir,
        split="val",
        device=device,
    )
    report = {
        "format": FORMAT_VERSION,
        "train_manifest": str(args.train_manifest.resolve()),
        "train_manifest_sha256": sha256_file(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "val_manifest_sha256": sha256_file(args.val_manifest.resolve()),
        "train_record_count": len(train_records),
        "val_record_count": len(val_records),
        "frames": args.frames,
        "audiosep_checkpoint": str(args.audiosep_checkpoint.resolve()),
        "audiosep_checkpoint_sha256": sha256_file(args.audiosep_checkpoint.resolve()),
        "device": str(device),
        "checkpoint": str(checkpoint_path.resolve()),
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "mode_registry": MODE_REGISTRY,
        "training": training_report,
        "summaries_by_split": {
            "train": train_summary,
            "val": val_summary,
        },
        "protocol_limitations": [
            "The planner/prompt still uses annotation-provided event inventory.",
            "The temporal grounder does not use target waveform, gold answer, or gold evidence IDs at evaluation.",
            "Raw AudioSep outputs are cached for efficiency but are generated from frozen AudioSep and structured-planner prompts.",
        ],
        "items": {
            "train": train_items,
            "val": val_items,
        },
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summaries_by_split"]["val"], indent=2, sort_keys=True))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
