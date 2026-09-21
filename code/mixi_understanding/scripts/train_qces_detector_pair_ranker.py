#!/usr/bin/env python3
"""Train a listwise QCES detector pair ranker with a NONE candidate."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.scripts.audit_qces_detector_union_proposal_recall import build_union_pool
from mixi_understanding.scripts.evaluate_qces_detector_inventory_qa import (
    QAItem,
    build_qa_items,
    interval_iou,
    intervals_iou,
    read_jsonl,
    summarize_rows,
    write_json,
)
from mixi_understanding.scripts.evaluate_qces_detector_pair_retrieval_qa import (
    Segment,
    temporal_gap,
)


DEFAULT_TRAIN_FRAME_PROBS = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200/partial_as_fuss_e4_unfreeze1_highthr_resume/train_frame_probs.pt"
)
DEFAULT_VAL_FRAME_PROBS = (
    PROJECT_ROOT
    / "outputs/qces_pretrainedsed_detector_multisource200/partial_as_fuss_e4_unfreeze1_highthr_resume/val_frame_probs.pt"
)
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/detector_manifest_train.jsonl"
DEFAULT_VAL_MANIFEST = PROJECT_ROOT / "outputs/qces_multisource_detector_trainset_v1/detector_manifest_val.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_detector_pair_ranker/multisource200_union_v1"


@dataclass(frozen=True)
class Candidate:
    anchor: Segment | None
    answer: Segment | None
    is_none: bool
    features: tuple[float, ...]
    anchor_label_id: int
    answer_label_id: int


@dataclass
class Example:
    item: QAItem
    features: torch.Tensor
    anchor_label_ids: torch.Tensor
    answer_label_ids: torch.Tensor
    target: torch.Tensor
    candidates: list[Candidate]
    has_pair_positive: bool


class PairRanker(nn.Module):
    def __init__(self, *, num_labels: int, feature_dim: int, embed_dim: int = 24, hidden_dim: int = 128, dropout: float = 0.10) -> None:
        super().__init__()
        self.none_label_id = num_labels
        self.label_embedding = nn.Embedding(num_labels + 1, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor, anchor_ids: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
        anchor_emb = self.label_embedding(anchor_ids)
        answer_emb = self.label_embedding(answer_ids)
        x = torch.cat([features, anchor_emb, answer_emb], dim=-1)
        return self.net(x).squeeze(-1)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-frame-probs", type=Path, default=DEFAULT_TRAIN_FRAME_PROBS)
    parser.add_argument("--val-frame-probs", type=Path, default=DEFAULT_VAL_FRAME_PROBS)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=Path, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2036)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-val-scenes", type=int, default=0)
    parser.add_argument("--max-answerable-per-scene", type=int, default=4)
    parser.add_argument("--max-no-evidence-per-scene", type=int, default=2)
    parser.add_argument("--top-anchor-candidates", type=int, default=16)
    parser.add_argument("--top-answer-candidates", type=int, default=240)
    parser.add_argument("--max-pair-candidates", type=int, default=512)
    parser.add_argument("--positive-iou", type=float, default=0.30)
    parser.add_argument("--none-prior-logit", type=float, default=0.0)
    parser.add_argument("--answerable-loss-weight", type=float, default=3.0)
    parser.add_argument("--no-evidence-loss-weight", type=float, default=1.0)
    parser.add_argument("--none-bias-grid", type=float, nargs="+", default=[-6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    # Union proposal settings, matching the audit defaults.
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--merge-gaps", type=float, nargs="+", default=[0.04, 0.12, 0.24])
    parser.add_argument("--min-duration", type=float, default=0.06)
    parser.add_argument("--hysteresis-high", type=float, nargs="+", default=[0.50, 0.70, 0.85])
    parser.add_argument("--hysteresis-low-ratio", type=float, nargs="+", default=[0.35, 0.50, 0.65])
    parser.add_argument("--peak-top-k-per-label", type=int, default=3)
    parser.add_argument("--peak-ratios", type=float, nargs="+", default=[0.25, 0.35, 0.50])
    parser.add_argument("--peak-min-score", type=float, default=0.02)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--max-proposals", type=int, default=320)
    parser.add_argument("--pool-progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def make_device(text: str) -> torch.device:
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(text)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gold_interval(pair: Sequence[float] | tuple[float, float] | None) -> tuple[float, float] | None:
    if pair is None or len(pair) != 2:
        return None
    return (float(pair[0]), float(pair[1]))


def segment_iou(seg: Segment | None, interval: tuple[float, float] | None) -> float:
    if seg is None or interval is None:
        return 0.0
    return interval_iou((seg.onset_seconds, seg.offset_seconds), interval)


def candidate_rank_map(pool: Sequence[Segment]) -> dict[int, int]:
    return {id(seg): index for index, seg in enumerate(pool, start=1)}


def label_first_rank(pool: Sequence[Segment]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for index, seg in enumerate(pool, start=1):
        ranks.setdefault(seg.label, index)
    return ranks


def occurrence_index(anchor: Segment, anchors_same_label: Sequence[Segment]) -> int:
    ordered = sorted(anchors_same_label, key=lambda seg: (seg.onset_seconds, seg.offset_seconds, -seg.confidence))
    for index, seg in enumerate(ordered, start=1):
        if seg is anchor:
            return index
    return 0


def count_intervening(anchor: Segment, answer: Segment, context: Sequence[Segment]) -> int:
    lo = min(anchor.onset_seconds, answer.onset_seconds)
    hi = max(anchor.onset_seconds, answer.onset_seconds)
    return sum(
        1
        for seg in context
        if lo + 1e-6 < seg.onset_seconds < hi - 1e-6
        and seg.confidence >= 0.80
        and segment_iou(seg, (anchor.onset_seconds, anchor.offset_seconds)) < 0.5
        and segment_iou(seg, (answer.onset_seconds, answer.offset_seconds)) < 0.5
    )


def pair_features(
    item: QAItem,
    anchor: Segment,
    answer: Segment,
    *,
    pool_rank: Mapping[int, int],
    label_rank: Mapping[str, int],
    anchors_same_label: Sequence[Segment],
    context: Sequence[Segment],
    max_rank: int,
) -> tuple[float, ...]:
    gap = temporal_gap(item.relation, anchor, answer)
    if gap is None:
        gap = 99.0
    pred_ord = occurrence_index(anchor, anchors_same_label)
    return (
        float(anchor.confidence),
        float(answer.confidence),
        math.log1p(max(0.0, gap)),
        min(float(gap), 10.0) / 10.0,
        min(anchor.duration_seconds, 10.0) / 10.0,
        min(answer.duration_seconds, 10.0) / 10.0,
        1.0 if item.relation == "after" else 0.0,
        1.0 if anchor.label == answer.label else 0.0,
        min(count_intervening(anchor, answer, context), 8) / 8.0,
        min(abs(pred_ord - int(item.anchor_ordinal)), 8) / 8.0,
        min(pred_ord, 8) / 8.0,
        1.0 / math.sqrt(float(pool_rank.get(id(anchor), max_rank) + 1)),
        1.0 / math.sqrt(float(pool_rank.get(id(answer), max_rank) + 1)),
        1.0 / math.sqrt(float(label_rank.get(anchor.label, max_rank) + 1)),
        1.0 / math.sqrt(float(label_rank.get(answer.label, max_rank) + 1)),
    )


def none_features(relation: str, feature_dim: int) -> tuple[float, ...]:
    values = [0.0] * feature_dim
    values[6] = 1.0 if relation == "after" else 0.0
    return tuple(values)


def build_pool_cache(
    frame_payload: Mapping[str, Any],
    args: argparse.Namespace,
    *,
    scene_ids: set[str],
) -> dict[str, list[Segment]]:
    labels = list(frame_payload["labels"])
    cache: dict[str, list[Segment]] = {}
    total = sum(1 for scene_id in frame_payload["scenes"] if str(scene_id) in scene_ids)
    done = 0
    for scene_id, entry in frame_payload["scenes"].items():
        if str(scene_id) not in scene_ids:
            continue
        cache[str(scene_id)] = build_union_pool(
            entry["probs"],
            labels,
            duration=float(entry["duration_seconds"]),
            args=args,
        )
        done += 1
        if args.pool_progress_every > 0 and (done == 1 or done % args.pool_progress_every == 0 or done == total):
            print(f"built_pool_cache {done}/{total}", flush=True)
    return cache


def build_examples(
    *,
    manifest_path: Path,
    frame_payload: Mapping[str, Any],
    args: argparse.Namespace,
    split_name: str,
    max_scenes: int,
) -> tuple[list[Example], list[dict[str, Any]]]:
    labels = list(frame_payload["labels"])
    none_label_id = len(labels)
    manifest_rows = read_jsonl(manifest_path)
    qa_items = build_qa_items(
        manifest_rows,
        max_scenes=max_scenes,
        max_answerable_per_scene=args.max_answerable_per_scene,
        max_no_evidence_per_scene=args.max_no_evidence_per_scene,
        min_gap_seconds=0.0,
    )
    needed_scene_ids = {item.scene_id for item in qa_items}
    pool_cache = build_pool_cache(frame_payload, args, scene_ids=needed_scene_ids)
    examples: list[Example] = []
    skipped: list[dict[str, Any]] = []
    feature_dim = 15
    for item in qa_items:
        pool = pool_cache.get(item.scene_id)
        if not pool:
            skipped.append({"item_id": item.item_id, "split": split_name, "reason": "empty_pool"})
            continue
        pool_by_conf = sorted(pool, key=lambda seg: seg.confidence, reverse=True)
        pool_rank = candidate_rank_map(pool_by_conf)
        label_rank = label_first_rank(pool_by_conf)
        anchors = [seg for seg in pool_by_conf if seg.label == item.anchor_label][: args.top_anchor_candidates]
        answers = pool_by_conf[: args.top_answer_candidates]
        context = pool_by_conf[: min(len(pool_by_conf), args.top_answer_candidates)]
        candidates: list[Candidate] = []
        pair_priority: list[tuple[float, Candidate]] = []
        anchors_same_label_all = [seg for seg in pool_by_conf if seg.label == item.anchor_label]
        for anchor in anchors:
            for answer in answers:
                gap = temporal_gap(item.relation, anchor, answer)
                if gap is None:
                    continue
                if anchor.label == answer.label and interval_iou(
                    (anchor.onset_seconds, anchor.offset_seconds),
                    (answer.onset_seconds, answer.offset_seconds),
                ) > 0.80:
                    continue
                features = pair_features(
                    item,
                    anchor,
                    answer,
                    pool_rank=pool_rank,
                    label_rank=label_rank,
                    anchors_same_label=anchors_same_label_all,
                    context=context,
                    max_rank=max(args.max_proposals, args.top_answer_candidates) + 1,
                )
                candidate = Candidate(
                    anchor=anchor,
                    answer=answer,
                    is_none=False,
                    features=features,
                    anchor_label_id=int(anchor.label_id),
                    answer_label_id=int(answer.label_id),
                )
                heuristic = anchor.confidence + answer.confidence - 0.05 * max(0.0, gap)
                pair_priority.append((heuristic, candidate))
        pair_priority.sort(key=lambda row: row[0], reverse=True)
        candidates = [candidate for _, candidate in pair_priority[: args.max_pair_candidates]]
        none_candidate = Candidate(
            anchor=None,
            answer=None,
            is_none=True,
            features=none_features(item.relation, feature_dim),
            anchor_label_id=none_label_id,
            answer_label_id=none_label_id,
        )
        candidates.append(none_candidate)
        if not candidates:
            skipped.append({"item_id": item.item_id, "split": split_name, "reason": "no_candidates"})
            continue

        target_values = torch.zeros(len(candidates), dtype=torch.float32)
        has_pair_positive = False
        if item.no_evidence:
            target_values[-1] = 1.0
        else:
            anchor_interval = gold_interval(item.gold_anchor_interval)
            answer_interval = gold_interval(item.gold_answer_interval)
            for index, candidate in enumerate(candidates[:-1]):
                anchor_iou = segment_iou(candidate.anchor, anchor_interval)
                answer_iou = segment_iou(candidate.answer, answer_interval)
                answer_label_ok = candidate.answer is not None and candidate.answer.label == item.answer_label
                if anchor_iou >= args.positive_iou and answer_iou >= args.positive_iou and answer_label_ok:
                    target_values[index] = max(1e-4, anchor_iou * answer_iou)
                    has_pair_positive = True
            if not has_pair_positive:
                if split_name == "train":
                    skipped.append({"item_id": item.item_id, "split": split_name, "reason": "answerable_no_positive_pair"})
                    continue
                # For validation/inference, keep the example. The model may still
                # predict the correct answer label even when no proposal reaches
                # the positive IoU threshold; evidence IoU will naturally score low.
                skipped.append({"item_id": item.item_id, "split": split_name, "reason": "answerable_no_positive_pair_included_eval"})
                target_values[-1] = 1.0
        target_values = target_values / target_values.sum().clamp_min(1e-8)
        examples.append(
            Example(
                item=item,
                features=torch.tensor([candidate.features for candidate in candidates], dtype=torch.float32),
                anchor_label_ids=torch.tensor([candidate.anchor_label_id for candidate in candidates], dtype=torch.long),
                answer_label_ids=torch.tensor([candidate.answer_label_id for candidate in candidates], dtype=torch.long),
                target=target_values,
                candidates=candidates,
                has_pair_positive=has_pair_positive,
            )
        )
    return examples, skipped


def soft_ce_loss(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target * torch.log_softmax(scores, dim=0)).sum()


def apply_none_bias(scores: torch.Tensor, bias: float) -> torch.Tensor:
    if scores.numel() == 0 or bias == 0.0:
        return scores
    adjusted = scores.clone()
    adjusted[-1] = adjusted[-1] + float(bias)
    return adjusted


def train_epoch(model: PairRanker, examples: list[Example], optimizer: torch.optim.Optimizer, device: torch.device, args: argparse.Namespace) -> float:
    model.train()
    random.shuffle(examples)
    losses: list[float] = []
    for ex in examples:
        features = ex.features.to(device)
        anchor_ids = ex.anchor_label_ids.to(device)
        answer_ids = ex.answer_label_ids.to(device)
        target = ex.target.to(device)
        scores = apply_none_bias(model(features, anchor_ids, answer_ids), args.none_prior_logit)
        item_weight = args.no_evidence_loss_weight if ex.item.no_evidence else args.answerable_loss_weight
        loss = soft_ce_loss(scores, target) * float(item_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(
    model: PairRanker,
    examples: list[Example],
    device: torch.device,
    *,
    none_score_bias: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for ex in examples:
        scores = apply_none_bias(model(
            ex.features.to(device),
            ex.anchor_label_ids.to(device),
            ex.answer_label_ids.to(device),
        ), none_score_bias).detach().cpu()
        best_index = int(torch.argmax(scores).item())
        best = ex.candidates[best_index]
        item = ex.item
        if best.is_none or best.answer is None:
            pred_answer = "no_evidence"
            pred_noev = True
            pred_intervals: tuple[tuple[float, float], ...] = ()
            pred_events: list[dict[str, Any]] = []
        else:
            pred_answer = best.answer.label
            pred_noev = False
            pred_intervals = (
                (best.anchor.onset_seconds, best.anchor.offset_seconds) if best.anchor else (0.0, 0.0),
                (best.answer.onset_seconds, best.answer.offset_seconds),
            )
            pred_events = [seg.to_dict() for seg in (best.anchor, best.answer) if seg is not None]
        evidence_iou = 1.0 if item.no_evidence and pred_noev else intervals_iou(item.gold_evidence_intervals, pred_intervals)
        answer_correct = pred_answer == item.answer and pred_noev == item.no_evidence
        rows.append(
            {
                "item_id": item.item_id,
                "scene_id": item.scene_id,
                "source_route": item.source_route,
                "relation": item.relation,
                "question": item.question,
                "gold_answer": item.answer,
                "gold_no_evidence": item.no_evidence,
                "gold_anchor_label": item.anchor_label,
                "gold_anchor_ordinal": item.anchor_ordinal,
                "gold_answer_label": item.answer_label,
                "gold_evidence_intervals": [list(x) for x in item.gold_evidence_intervals],
                "pred_answer": pred_answer,
                "pred_no_evidence": pred_noev,
                "pred_evidence_intervals": [list(x) for x in pred_intervals],
                "pred_evidence_events": pred_events,
                "answer_correct": answer_correct,
                "evidence_iou_↑": evidence_iou,
                "answer_and_evidence_iou030_correct": answer_correct and (item.no_evidence or evidence_iou >= 0.30),
                "reason": "ranker_none" if pred_noev else "ranker_pair",
                "best_score": float(scores[best_index].item()),
                "num_candidates": len(ex.candidates),
            }
        )
    return summarize_rows(rows, 0.30), rows


def save_predictions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def conservative_metrics_with_skips(metrics: Mapping[str, Any], skipped: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count skipped answerable examples as wrong with zero evidence IoU.

    The ranker is only trainable when a positive anchor-answer proposal exists. This
    helper reports the stricter end-to-end number where missing proposal examples
    are retained in the denominator.
    """

    skipped_answerable = sum(str(row.get("reason")) == "answerable_no_positive_pair" for row in skipped)
    conditional_items = int(metrics.get("items", 0))
    conditional_answerable = int(metrics.get("answerable_items", 0))
    no_evidence_items = int(metrics.get("no_evidence_items", 0))
    full_items = conditional_items + skipped_answerable
    full_answerable = conditional_answerable + skipped_answerable
    correct = float(metrics.get("accuracy_↑", 0.0)) * conditional_items
    answerable_correct = float(metrics.get("answerable_accuracy_↑", 0.0)) * conditional_answerable
    both_correct = float(metrics.get("answer_and_evidence_iou030_accuracy_↑", 0.0)) * conditional_answerable
    evidence_iou_sum = float(metrics.get("mean_evidence_iou_answerable_↑", 0.0)) * conditional_answerable
    return {
        "items": full_items,
        "answerable_items": full_answerable,
        "no_evidence_items": no_evidence_items,
        "skipped_answerable_as_wrong": skipped_answerable,
        "accuracy_↑": correct / max(full_items, 1),
        "answerable_accuracy_↑": answerable_correct / max(full_answerable, 1),
        "no_evidence_accuracy_↑": float(metrics.get("no_evidence_accuracy_↑", 0.0)),
        "mean_evidence_iou_answerable_↑": evidence_iou_sum / max(full_answerable, 1),
        "answer_and_evidence_iou030_accuracy_↑": both_correct / max(full_answerable, 1),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = make_device(args.device)

    train_payload = torch.load(args.train_frame_probs.resolve(), map_location="cpu", weights_only=False)
    val_payload = torch.load(args.val_frame_probs.resolve(), map_location="cpu", weights_only=False)
    labels = list(train_payload["labels"])
    if labels != list(val_payload["labels"]):
        raise ValueError("train/val labels mismatch")

    print("building_train_examples", flush=True)
    train_examples, train_skipped = build_examples(
        manifest_path=args.train_manifest.resolve(),
        frame_payload=train_payload,
        args=args,
        split_name="train",
        max_scenes=args.max_train_scenes,
    )
    print("building_val_examples", flush=True)
    val_examples, val_skipped = build_examples(
        manifest_path=args.val_manifest.resolve(),
        frame_payload=val_payload,
        args=args,
        split_name="val",
        max_scenes=args.max_val_scenes,
    )
    if not train_examples or not val_examples:
        raise SystemExit("empty train or val examples")
    feature_dim = int(train_examples[0].features.size(1))
    model = PairRanker(
        num_labels=len(labels),
        feature_dim=feature_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_examples, optimizer, device, args)
        metrics, _rows = evaluate(model, val_examples, device, none_score_bias=args.none_prior_logit)
        score = float(metrics["answerable_accuracy_↑"]) + float(metrics["no_evidence_accuracy_↑"]) + float(metrics["answer_and_evidence_iou030_accuracy_↑"])
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "score": score,
            "accuracy_↑": metrics["accuracy_↑"],
            "answerable_accuracy_↑": metrics["answerable_accuracy_↑"],
            "no_evidence_accuracy_↑": metrics["no_evidence_accuracy_↑"],
            "mean_evidence_iou_answerable_↑": metrics["mean_evidence_iou_answerable_↑"],
            "answer_and_evidence_iou030_accuracy_↑": metrics["answer_and_evidence_iou030_accuracy_↑"],
            "predicted_no_evidence_rate": metrics["predicted_no_evidence_rate"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    bias_sweep: list[dict[str, Any]] = []
    best_bias = float(args.none_prior_logit)
    best_bias_score = -1.0
    for bias in args.none_bias_grid:
        metrics, _rows = evaluate(model, val_examples, device, none_score_bias=float(bias))
        bias_score = (
            float(metrics["answerable_accuracy_↑"])
            + float(metrics["no_evidence_accuracy_↑"])
            + float(metrics["answer_and_evidence_iou030_accuracy_↑"])
        )
        row = {
            "none_score_bias": float(bias),
            "score": bias_score,
            "accuracy_↑": metrics["accuracy_↑"],
            "answerable_accuracy_↑": metrics["answerable_accuracy_↑"],
            "no_evidence_accuracy_↑": metrics["no_evidence_accuracy_↑"],
            "mean_evidence_iou_answerable_↑": metrics["mean_evidence_iou_answerable_↑"],
            "answer_and_evidence_iou030_accuracy_↑": metrics["answer_and_evidence_iou030_accuracy_↑"],
            "predicted_no_evidence_rate": metrics["predicted_no_evidence_rate"],
        }
        bias_sweep.append(row)
        if bias_score > best_bias_score:
            best_bias_score = bias_score
            best_bias = float(bias)
    val_metrics, val_rows = evaluate(model, val_examples, device, none_score_bias=best_bias)
    val_metrics_full_conservative = conservative_metrics_with_skips(val_metrics, val_skipped)
    report = {
        "format": "qces_detector_pair_ranker_v1",
        "train_frame_probs": str(args.train_frame_probs.resolve()),
        "val_frame_probs": str(args.val_frame_probs.resolve()),
        "train_manifest": str(args.train_manifest.resolve()),
        "val_manifest": str(args.val_manifest.resolve()),
        "labels": labels,
        "settings": jsonable_args(args),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "train_skipped": len(train_skipped),
        "val_skipped": len(val_skipped),
        "train_skip_reasons": {reason: sum(row["reason"] == reason for row in train_skipped) for reason in sorted({row["reason"] for row in train_skipped})},
        "val_skip_reasons": {reason: sum(row["reason"] == reason for row in val_skipped) for reason in sorted({row["reason"] for row in val_skipped})},
        "history": history,
        "bias_sweep": bias_sweep,
        "selected_none_score_bias": best_bias,
        "val_metrics": val_metrics,
        "val_metrics_full_conservative": val_metrics_full_conservative,
    }
    write_json(output_dir / "training_report.json", report)
    save_predictions(output_dir / "val_predictions.jsonl", val_rows)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "feature_dim": feature_dim,
            "args": vars(args),
            "args_jsonable": jsonable_args(args),
            "report": report,
        },
        output_dir / "pair_ranker.pt",
    )
    md = [
        "# QCES detector pair ranker",
        "",
        f"- train examples: {len(train_examples)}",
        f"- val examples: {len(val_examples)}",
        f"- train skipped: {len(train_skipped)}",
        f"- val skipped: {len(val_skipped)}",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| accuracy ↑ | {val_metrics['accuracy_↑']:.4f} |",
        f"| answerable accuracy ↑ | {val_metrics['answerable_accuracy_↑']:.4f} |",
        f"| no-evidence accuracy ↑ | {val_metrics['no_evidence_accuracy_↑']:.4f} |",
        f"| mean evidence IoU answerable ↑ | {val_metrics['mean_evidence_iou_answerable_↑']:.4f} |",
        f"| answer+IoU≥0.30 accuracy ↑ | {val_metrics['answer_and_evidence_iou030_accuracy_↑']:.4f} |",
        f"| predicted no-evidence rate | {val_metrics['predicted_no_evidence_rate']:.4f} |",
        "",
        "## Full conservative metrics",
        "",
        "Skipped answerable samples without a positive proposal are counted as wrong.",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| items | {val_metrics_full_conservative['items']} |",
        f"| skipped answerable as wrong | {val_metrics_full_conservative['skipped_answerable_as_wrong']} |",
        f"| accuracy ↑ | {val_metrics_full_conservative['accuracy_↑']:.4f} |",
        f"| answerable accuracy ↑ | {val_metrics_full_conservative['answerable_accuracy_↑']:.4f} |",
        f"| no-evidence accuracy ↑ | {val_metrics_full_conservative['no_evidence_accuracy_↑']:.4f} |",
        f"| mean evidence IoU answerable ↑ | {val_metrics_full_conservative['mean_evidence_iou_answerable_↑']:.4f} |",
        f"| answer+IoU≥0.30 accuracy ↑ | {val_metrics_full_conservative['answer_and_evidence_iou030_accuracy_↑']:.4f} |",
    ]
    (output_dir / "training_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(report["val_metrics"], ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
