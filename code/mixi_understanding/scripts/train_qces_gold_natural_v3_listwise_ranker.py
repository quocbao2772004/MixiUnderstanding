#!/usr/bin/env python3
"""Train a listwise adjacent-event ranker with an explicit NONE candidate.

Candidates come only from overlap-aware RED/EPN v2.  The question supplies an
anchor label and a before/after relation; the ranker chooses one relation-valid
event proposal or NONE.  Gold answer labels and timestamps supervise the loss
but never enter candidate generation or model features.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from mixi_understanding.qces.class_aware_red_epn_v1 import (
    ClassAwareRedEpnV1,
    ClassAwareRedEpnV1Config,
    decode_class_aware_proposals,
    interval_iou,
)
from mixi_understanding.scripts.audit_qces_overlap_aware_red_epn_v2 import (
    adjacent_index_pairs,
)
from mixi_understanding.scripts.train_qces_class_aware_red_epn_v1 import (
    ClassAwareSceneDataset,
    collate_class_aware,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _atomic_torch,
    _device,
    _sha256_file,
    load_scene_list,
)


FORMAT = "qces_gold_natural_v3_listwise_ranker_receipt_v1"
CHECKPOINT_FORMAT = "qces_gold_natural_v3_listwise_ranker_checkpoint_v1"
DEFAULT_BASE = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
)
DEFAULT_DATA = PROJECT_ROOT / "outputs/qces_full188_overlap_gold_natural_v3"
NUMERIC_DIM = 15


@dataclass(frozen=True)
class RankerConfig:
    num_classes: int
    max_candidates: int = 120
    numeric_dim: int = NUMERIC_DIM
    label_embedding_dim: int = 16
    relation_embedding_dim: int = 4
    hidden_dim: int = 64
    dropout: float = 0.10


class ListwisePairNoneRanker(nn.Module):
    def __init__(self, config: RankerConfig) -> None:
        super().__init__()
        self.config = config
        self.label_embedding = nn.Embedding(config.num_classes, config.label_embedding_dim)
        self.relation_embedding = nn.Embedding(2, config.relation_embedding_dim)
        candidate_dim = (
            config.numeric_dim
            + 2 * config.label_embedding_dim
            + config.relation_embedding_dim
        )
        self.candidate_mlp = nn.Sequential(
            nn.Linear(candidate_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.none_mlp = nn.Sequential(
            nn.Linear(config.label_embedding_dim + config.relation_embedding_dim + 6, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Start with a neutral listwise distribution. Independent random final
        # layers can otherwise put NONE above all 120 candidates before the
        # first update and create an easy abstention-collapse attractor.
        nn.init.zeros_(self.candidate_mlp[-1].weight)
        nn.init.zeros_(self.candidate_mlp[-1].bias)
        nn.init.zeros_(self.none_mlp[-1].weight)
        nn.init.zeros_(self.none_mlp[-1].bias)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        candidate_label = self.label_embedding(batch["candidate_label_id"])
        anchor_label = self.label_embedding(batch["anchor_label_id"])
        relation = self.relation_embedding(batch["relation_id"])
        count = batch["candidate_numeric"].shape[1]
        anchor_expand = anchor_label[:, None, :].expand(-1, count, -1)
        relation_expand = relation[:, None, :].expand(-1, count, -1)
        candidate_input = torch.cat(
            (batch["candidate_numeric"], candidate_label, anchor_expand, relation_expand),
            dim=-1,
        )
        candidate_score = self.candidate_mlp(candidate_input).squeeze(-1)
        candidate_score = candidate_score.masked_fill(~batch["candidate_mask"], -1e4)
        none_input = torch.cat((anchor_label, relation, batch["none_numeric"]), dim=-1)
        none_score = self.none_mlp(none_input)
        return torch.cat((candidate_score, none_score), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proposal-checkpoint",
        type=Path,
        default=DEFAULT_BASE / "overlap_aware_red_epn_v2/overlap_aware_red_epn_v2_best.pt",
    )
    parser.add_argument(
        "--train-index",
        type=Path,
        default=DEFAULT_BASE / "dense_overlap_gold_natural_v3_train/index.json",
    )
    parser.add_argument(
        "--dev-index",
        type=Path,
        default=DEFAULT_BASE / "dense_overlap_gold_natural_v3_dev/index.json",
    )
    parser.add_argument(
        "--train-scenes", type=Path, default=DEFAULT_DATA / "scene_ids_overlap_train.txt"
    )
    parser.add_argument(
        "--dev-scenes", type=Path, default=DEFAULT_DATA / "scene_ids_overlap_dev.txt"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_BASE / "listwise_pair_none_ranker_v1"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4199)
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--proposal-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--no-evidence-loss-weight", type=float, default=0.5)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def collect_scene_candidates(
    model: ClassAwareRedEpnV1,
    store: DenseFeatureStore,
    scene_ids: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
    max_candidates: int,
    max_scenes: int,
) -> list[dict[str, Any]]:
    dataset = ClassAwareSceneDataset(
        store, scene_ids, preload=True, max_scenes=max_scenes
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_class_aware,
    )
    rows: list[dict[str, Any]] = []
    model.eval()
    for raw in loader:
        outputs = model(
            raw["features"].to(device, non_blocking=True),
            raw["detector_logits"].to(device, non_blocking=True),
            raw["valid_mask"].to(device, non_blocking=True),
        )
        for batch_index, scene_id in enumerate(raw["scene_id"]):
            proposals = decode_class_aware_proposals(
                outputs,
                batch_index,
                max_classes=max_candidates,
                max_events=max_candidates,
            )
            rows.append(
                {
                    "scene_id": str(scene_id),
                    "events": list(raw["gold_events"][batch_index]),
                    "proposals": proposals,
                }
            )
        if len(rows) % 256 < batch_size:
            print(f"candidate inference: {len(rows)}/{len(dataset)} scenes", flush=True)
    return rows


def _overlap(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    return intersection / max(min(left[1] - left[0], right[1] - right[0]), 1e-8)


def candidate_features(
    candidate: Mapping[str, Any],
    anchor: Mapping[str, Any],
    *,
    relation: int,
    global_rank: int,
    temporal_rank: int,
    max_candidates: int,
) -> list[float]:
    candidate_start, candidate_end = float(candidate["start"]), float(candidate["end"])
    anchor_start, anchor_end = float(anchor["start"]), float(anchor["end"])
    oriented_delta = (
        candidate_start - anchor_start if relation == 1 else anchor_start - candidate_start
    )
    gap = (
        max(0.0, candidate_start - anchor_end)
        if relation == 1
        else max(0.0, anchor_start - candidate_end)
    )
    return [
        float(candidate["score"]),
        float(candidate.get("clip_score", candidate["score"])),
        candidate_end - candidate_start,
        global_rank / max(max_candidates - 1, 1),
        float(anchor["score"]),
        float(anchor.get("clip_score", anchor["score"])),
        anchor_end - anchor_start,
        oriented_delta,
        abs(oriented_delta),
        gap,
        _overlap((candidate_start, candidate_end), (anchor_start, anchor_end)),
        1.0 / (1.0 + temporal_rank),
        candidate_start,
        candidate_end,
        float(candidate["label_id"] == anchor["label_id"]),
    ]


def build_rank_examples(
    scenes: Sequence[Mapping[str, Any]],
    *,
    num_classes: int,
    max_candidates: int,
    iou_threshold: float,
) -> tuple[list[dict[str, torch.Tensor]], dict[str, int]]:
    examples: list[dict[str, torch.Tensor]] = []
    counts: Counter[str] = Counter()
    for scene in scenes:
        events = list(scene["events"])
        proposals = list(scene["proposals"])
        pairs = adjacent_index_pairs(events)
        if len(pairs) != max(len(events) - 1, 0):
            counts["scenes_skipped_tied_onset"] += 1
            continue
        qa_specs: list[tuple[int, int | None, int]] = []
        for left, right in pairs:
            qa_specs.extend(((left, right, 1), (right, left, 0)))
        order = sorted(
            range(len(events)), key=lambda index: (events[index]["start"], events[index]["end"])
        )
        if order:
            qa_specs.extend(((order[0], None, 0), (order[-1], None, 1)))

        for anchor_index, answer_index, relation in qa_specs:
            anchor_gold = events[anchor_index]
            anchor_label = int(anchor_gold["label_id"])
            anchor_candidates = [
                proposal for proposal in proposals if int(proposal["label_id"]) == anchor_label
            ]
            anchor = anchor_candidates[0] if anchor_candidates else None
            no_evidence = answer_index is None
            counts["questions"] += 1
            counts["no_evidence" if no_evidence else "answerable"] += 1
            if anchor is not None:
                counts["anchor_found"] += 1

            candidate_rows: list[tuple[int, Mapping[str, Any]]] = []
            if anchor is not None:
                for global_rank, candidate in enumerate(proposals):
                    if int(candidate["label_id"]) == anchor_label:
                        continue
                    delta = (
                        float(candidate["start"]) - float(anchor["start"])
                        if relation == 1
                        else float(anchor["start"]) - float(candidate["start"])
                    )
                    if delta > 1e-6:
                        candidate_rows.append((global_rank, candidate))
            candidate_rows = candidate_rows[:max_candidates]
            temporal_order = sorted(
                range(len(candidate_rows)),
                key=lambda index: abs(
                    float(candidate_rows[index][1]["start"])
                    - (float(anchor["start"]) if anchor is not None else 0.0)
                ),
            )
            temporal_rank = {index: rank for rank, index in enumerate(temporal_order)}

            numeric = torch.zeros(max_candidates, NUMERIC_DIM)
            candidate_ids = torch.zeros(max_candidates, dtype=torch.long)
            mask = torch.zeros(max_candidates, dtype=torch.bool)
            starts = torch.zeros(max_candidates)
            ends = torch.zeros(max_candidates)
            target = torch.zeros(max_candidates + 1)
            positive_weights: list[tuple[int, float]] = []
            answer = events[answer_index] if answer_index is not None else None
            for index, (global_rank, candidate) in enumerate(candidate_rows):
                numeric[index] = torch.tensor(
                    candidate_features(
                        candidate,
                        anchor,
                        relation=relation,
                        global_rank=global_rank,
                        temporal_rank=temporal_rank[index],
                        max_candidates=max_candidates,
                    )
                )
                candidate_ids[index] = int(candidate["label_id"])
                mask[index] = True
                starts[index], ends[index] = float(candidate["start"]), float(candidate["end"])
                if answer is not None and int(candidate["label_id"]) == int(answer["label_id"]):
                    iou = interval_iou(
                        (float(candidate["start"]), float(candidate["end"])),
                        (float(answer["start"]), float(answer["end"])),
                    )
                    if iou >= iou_threshold:
                        positive_weights.append((index, math.exp(5.0 * iou)))

            target_available = False
            if no_evidence and anchor is not None:
                target[-1] = 1.0
                target_available = True
            elif positive_weights and anchor is not None:
                total = sum(weight for _, weight in positive_weights)
                for index, weight in positive_weights:
                    target[index] = weight / total
                target_available = True
                counts["answerable_supported"] += 1
            elif not no_evidence:
                counts["answerable_unsupported"] += 1

            none_numeric = torch.tensor(
                [
                    0.0 if anchor is None else float(anchor["score"]),
                    0.0 if anchor is None else float(anchor.get("clip_score", anchor["score"])),
                    0.0 if anchor is None else float(anchor["end"]) - float(anchor["start"]),
                    len(candidate_rows) / max(max_candidates, 1),
                    0.0 if not candidate_rows else float(candidate_rows[0][1]["score"]),
                    float(anchor is not None),
                ]
            )
            examples.append(
                {
                    "candidate_numeric": numeric,
                    "candidate_label_id": candidate_ids.clamp(0, num_classes - 1),
                    "candidate_mask": mask,
                    "candidate_start": starts,
                    "candidate_end": ends,
                    "anchor_label_id": torch.tensor(anchor_label),
                    "relation_id": torch.tensor(relation),
                    "none_numeric": none_numeric,
                    "target": target,
                    "target_available": torch.tensor(target_available),
                    "no_evidence": torch.tensor(no_evidence),
                    "anchor_found": torch.tensor(anchor is not None),
                    "gold_label_id": torch.tensor(-1 if answer is None else int(answer["label_id"])),
                    "gold_start": torch.tensor(0.0 if answer is None else float(answer["start"])),
                    "gold_end": torch.tensor(0.0 if answer is None else float(answer["end"])),
                }
            )
    return examples, {key: int(value) for key, value in counts.items()}


class RankDataset(Dataset[Mapping[str, torch.Tensor]]):
    def __init__(self, rows: Sequence[Mapping[str, torch.Tensor]]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        return self.rows[index]


def to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.inference_mode()
def collect_ranker_outputs(
    model: ListwisePairNoneRanker,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    model.eval()
    collected: dict[str, list[torch.Tensor]] = {
        key: []
        for key in (
            "scores",
            "candidate_label_id",
            "candidate_start",
            "candidate_end",
            "no_evidence",
            "anchor_found",
            "gold_label_id",
            "gold_start",
            "gold_end",
        )
    }
    for raw in loader:
        batch = to_device(raw, device)
        collected["scores"].append(model(batch).float().cpu())
        for key in collected:
            if key != "scores":
                collected[key].append(raw[key].cpu())
    return {key: torch.cat(values, dim=0) for key, values in collected.items()}


def metrics_from_ranker_outputs(
    outputs: Mapping[str, torch.Tensor],
    *,
    none_bias: float,
    iou_threshold: float,
) -> dict[str, float | int]:
    counts: Counter[str] = Counter()
    scores = outputs["scores"].clone()
    scores[:, -1] += none_bias
    prediction = scores.argmax(dim=1)
    none_index = scores.shape[1] - 1
    for index in range(prediction.numel()):
        counts["questions"] += 1
        no_evidence = bool(outputs["no_evidence"][index])
        predicted_none = int(prediction[index]) == none_index
        if no_evidence:
            counts["no_evidence"] += 1
            counts["no_evidence_correct"] += int(predicted_none)
            counts["verified_no_evidence_correct"] += int(
                predicted_none and bool(outputs["anchor_found"][index])
            )
            continue
        counts["answerable"] += 1
        if predicted_none:
            continue
        candidate_index = int(prediction[index])
        label_correct = int(outputs["candidate_label_id"][index, candidate_index]) == int(
            outputs["gold_label_id"][index]
        )
        counts["answer_label_correct"] += int(label_correct)
        iou = interval_iou(
            (
                float(outputs["candidate_start"][index, candidate_index]),
                float(outputs["candidate_end"][index, candidate_index]),
            ),
            (float(outputs["gold_start"][index]), float(outputs["gold_end"][index])),
        )
        counts["evidence_correct"] += int(label_correct and iou >= iou_threshold)
    answerable = max(counts["answerable"], 1)
    no_evidence = max(counts["no_evidence"], 1)
    evidence_accuracy = counts["evidence_correct"] / answerable
    verified_noev = counts["verified_no_evidence_correct"] / no_evidence
    harmonic = (
        0.0
        if evidence_accuracy + verified_noev <= 0.0
        else 2.0 * evidence_accuracy * verified_noev / (evidence_accuracy + verified_noev)
    )
    return {
        "questions": int(counts["questions"]),
        "answerable": int(counts["answerable"]),
        "no_evidence": int(counts["no_evidence"]),
        "answerable_label_accuracy_↑": counts["answer_label_correct"] / answerable,
        "answerable_evidence_accuracy_iou030_↑": evidence_accuracy,
        "no_evidence_accuracy_↑": counts["no_evidence_correct"] / no_evidence,
        "verified_no_evidence_accuracy_↑": verified_noev,
        "balanced_evidence_noev_accuracy_↑": 0.5
        * (
            evidence_accuracy
            + verified_noev
        ),
        "harmonic_evidence_noev_accuracy_↑": harmonic,
        "none_bias": none_bias,
    }


def calibrate_none_bias(
    outputs: Mapping[str, torch.Tensor], *, iou_threshold: float
) -> dict[str, float | int]:
    best_key: tuple[float, ...] | None = None
    best: dict[str, float | int] | None = None
    for none_bias in np.linspace(-6.0, 2.0, 65):
        metrics = metrics_from_ranker_outputs(
            outputs, none_bias=float(none_bias), iou_threshold=iou_threshold
        )
        key = (
            float(metrics["harmonic_evidence_noev_accuracy_↑"]),
            min(
                float(metrics["answerable_evidence_accuracy_iou030_↑"]),
                float(metrics["verified_no_evidence_accuracy_↑"]),
            ),
            float(metrics["answerable_evidence_accuracy_iou030_↑"]),
            float(metrics["answerable_label_accuracy_↑"]),
        )
        if best_key is None or key > best_key:
            best_key, best = key, metrics
    if best is None:
        raise RuntimeError("NONE-bias calibration produced no result")
    return best


def main() -> None:
    args = parse_args()
    if not 0.0 < args.no_evidence_loss_weight <= 1.0:
        raise ValueError("--no-evidence-loss-weight must be in (0,1]")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = _device(args.device)

    proposal_payload = torch.load(
        args.proposal_checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    proposal_config = ClassAwareRedEpnV1Config(**dict(proposal_payload["config"]))
    proposal_model = ClassAwareRedEpnV1(proposal_config)
    proposal_model.load_state_dict(proposal_payload["model_state_dict"], strict=True)
    proposal_model.to(device).eval()
    store = DenseFeatureStore(
        [args.train_index.resolve(), args.dev_index.resolve()], cache_size=32
    )
    split_scenes = {
        "train": load_scene_list(args.train_scenes.resolve()),
        "dev": load_scene_list(args.dev_scenes.resolve()),
    }
    scene_candidates = {
        split: collect_scene_candidates(
            proposal_model,
            store,
            scene_ids,
            device=device,
            batch_size=args.proposal_batch_size,
            max_candidates=args.max_candidates,
            max_scenes=args.max_train_scenes if split == "train" else args.max_dev_scenes,
        )
        for split, scene_ids in split_scenes.items()
    }
    del proposal_model
    torch.cuda.empty_cache()
    examples_and_counts = {
        split: build_rank_examples(
            rows,
            num_classes=proposal_config.num_classes,
            max_candidates=args.max_candidates,
            iou_threshold=args.iou_threshold,
        )
        for split, rows in scene_candidates.items()
    }
    datasets = {
        split: RankDataset(examples) for split, (examples, _) in examples_and_counts.items()
    }
    train_indices = [
        index
        for index, row in enumerate(datasets["train"].rows)
        if bool(row["target_available"])
    ]
    train_loader = DataLoader(
        Subset(datasets["train"], train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        datasets["dev"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    config = RankerConfig(
        num_classes=proposal_config.num_classes, max_candidates=args.max_candidates
    )
    model = ListwisePairNoneRanker(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_metrics: Mapping[str, Any] | None = None
    stale = 0
    checkpoint_path = output_dir / "listwise_pair_none_ranker_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for raw in train_loader:
            batch = to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch)
            log_probability = scores.log_softmax(dim=1)
            per_example = -(batch["target"] * log_probability).sum(dim=1)
            example_weight = torch.where(
                batch["no_evidence"],
                torch.full_like(per_example, args.no_evidence_loss_weight),
                torch.ones_like(per_example),
            )
            loss = (per_example * example_weight).sum() / example_weight.sum().clamp_min(1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        metrics = calibrate_none_bias(
            collect_ranker_outputs(model, dev_loader, device),
            iou_threshold=args.iou_threshold,
        )
        key = (
            float(metrics["harmonic_evidence_noev_accuracy_↑"]),
            min(
                float(metrics["answerable_evidence_accuracy_iou030_↑"]),
                float(metrics["verified_no_evidence_accuracy_↑"]),
            ),
            float(metrics["answerable_evidence_accuracy_iou030_↑"]),
            float(metrics["answerable_label_accuracy_↑"]),
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "dev": metrics,
            "selection_key": list(key),
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch, best_metrics, stale = key, epoch, row, 0
            _atomic_torch(
                {
                    "format": CHECKPOINT_FORMAT,
                    "epoch": epoch,
                    "config": asdict(config),
                    "model_state_dict": model.state_dict(),
                    "selection_metrics": row,
                    "none_bias": float(metrics["none_bias"]),
                    "proposal_checkpoint": str(args.proposal_checkpoint.resolve()),
                    "proposal_checkpoint_sha256": _sha256_file(
                        args.proposal_checkpoint.resolve()
                    ),
                },
                checkpoint_path,
            )
        else:
            stale += 1
        if stale >= args.patience:
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    if best_metrics is None:
        raise RuntimeError("ranker training produced no checkpoint")
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "overlap-aware Top-120 proposals + listwise pair ranker + NONE",
        "answer_label_used_as_model_input": False,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "config": asdict(config),
        "proposal_checkpoint": str(args.proposal_checkpoint.resolve()),
        "proposal_checkpoint_sha256": _sha256_file(args.proposal_checkpoint.resolve()),
        "data": {
            split: {
                "scenes": len(scene_candidates[split]),
                "questions": len(datasets[split]),
                "counts": examples_and_counts[split][1],
            }
            for split in datasets
        },
        "trainable_questions": len(train_indices),
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "history": history,
    }
    _atomic_json(receipt, output_dir / "receipt.json")
    print(json.dumps({"complete": True, "best_epoch": best_epoch, "best_metrics": best_metrics}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
