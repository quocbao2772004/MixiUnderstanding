#!/usr/bin/env python3
"""Train a QCES-v6 proposal head with IoU-aligned objectives.

This is intentionally separate from the original proposal-head trainer.  It
keeps the same ``ProposalHead`` architecture and checkpoint payload so every
downstream QCES-v6 script can consume the produced ``proposal_head.pt`` without
code changes.

Changes vs. v1:

* frame BCE is normalised separately over active and inactive frames, instead
  of relying only on a global positive weight;
* a soft Tversky/IoU loss is applied to the actual activity used at decoding
  time: ``sigmoid(frame) * sigmoid(presence)``;
* a length calibration loss discourages under/over-short spans;
* checkpoint selection is explicitly evidence-IoU weighted.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.qces.event_proposals import ProposalHead
from mixi_understanding.qces.stem_features import FrameGrid
from mixi_understanding.scripts.train_qces_v6_proposal_head import (
    ENERGY_THRESHOLD_GRID,
    THRESHOLD_GRID,
    activity_from_head,
    build_questions,
    build_units,
    evaluate_planner,
    frame_metrics,
    load_taxonomy,
    read_manifest,
    scene_label_intervals,
)


FORMAT_VERSION = "qces_v6_proposal_head_iou_v2"
THRESHOLD_GRID_V2 = tuple(
    sorted(set((0.025, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.45, 0.60)))
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--active-bce-weight", type=float, default=1.4)
    parser.add_argument("--inactive-bce-weight", type=float, default=0.45)
    parser.add_argument("--soft-iou-weight", type=float, default=1.2)
    parser.add_argument("--coverage-weight", type=float, default=0.18)
    parser.add_argument("--presence-weight", type=float, default=0.8)
    parser.add_argument("--onset-weight", type=float, default=0.8)
    parser.add_argument("--onset-positive-weight", type=float, default=20.0)
    parser.add_argument("--tversky-alpha", type=float, default=0.35)
    parser.add_argument("--tversky-beta", type=float, default=0.75)
    parser.add_argument("--select-answer-weight", type=float, default=0.25)
    parser.add_argument("--select-noev-weight", type=float, default=0.15)
    parser.add_argument("--select-iou-weight", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.14)
    parser.add_argument("--zero-feature-group", action="append", default=[])
    parser.add_argument("--no-onset-split", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def active_inactive_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    active_weight: float,
    inactive_weight: float,
) -> torch.Tensor:
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    active = targets > 0.5
    inactive = ~active
    active_loss = loss[active].mean() if bool(active.any()) else loss.new_zeros(())
    inactive_loss = loss[inactive].mean() if bool(inactive.any()) else loss.new_zeros(())
    return active_weight * active_loss + inactive_weight * inactive_loss


def combined_activity(logits: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits) * torch.sigmoid(presence)[:, None]


def soft_tversky_loss(
    probability: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    present = targets.amax(dim=-1) > 0.5
    if not bool(present.any()):
        # For absent labels the presence/BCE terms already provide the useful
        # signal.  A Tversky loss on empty targets would only duplicate FP
        # suppression and make positive recall worse.
        return probability.new_zeros(())
    p = probability[present]
    y = targets[present]
    tp = (p * y).sum(dim=-1)
    fp = (p * (1.0 - y)).sum(dim=-1)
    fn = ((1.0 - p) * y).sum(dim=-1)
    score = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - score.mean()


def coverage_loss(
    probability: torch.Tensor,
    targets: torch.Tensor,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    present = targets.amax(dim=-1) > 0.5
    if not bool(present.any()):
        return probability.new_zeros(())
    predicted = probability[present].sum(dim=-1)
    gold = targets[present].sum(dim=-1)
    return torch.abs(torch.log((predicted + eps) / (gold + eps))).mean()


def selection_score(row: dict[str, float], args: argparse.Namespace) -> float:
    return (
        args.select_answer_weight * row["planner_answer_accuracy_↑"]
        + args.select_noev_weight * row["planner_no_evidence_accuracy_↑"]
        + args.select_iou_weight * row["planner_evidence_span_iou_↑"]
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output is not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    taxonomy = load_taxonomy(args.dataset_config.resolve())
    grid = FrameGrid(sample_rate=32_000)

    train_cache = torch.load(args.train_cache, map_location="cpu", weights_only=False)
    val_cache = torch.load(args.val_cache, map_location="cpu", weights_only=False)
    train_rows = read_manifest(args.train_manifest)
    val_rows = read_manifest(args.val_manifest)

    train_questions = build_questions(train_rows, train_cache, taxonomy)
    val_questions = build_questions(val_rows, val_cache, taxonomy)
    zeroed = tuple(args.zero_feature_group)
    train_units = build_units(
        train_questions,
        train_cache,
        scene_label_intervals(train_rows),
        grid,
        zeroed,
    )
    val_units = build_units(
        val_questions,
        val_cache,
        scene_label_intervals(val_rows),
        grid,
        zeroed,
    )
    print(
        f"train units={len(train_units)} questions={len(train_questions)} | "
        f"val units={len(val_units)} questions={len(val_questions)}",
        flush=True,
    )

    model = ProposalHead(channels=args.channels, dropout=args.dropout).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=max(args.epochs, 1)
    )
    onset_positive_weight = torch.tensor(args.onset_positive_weight, device=device)
    use_onset_split = not args.no_onset_split

    baseline_activity = activity_from_head(None, val_cache, device, zeroed)
    baseline = max(
        (
            evaluate_planner(
                val_questions,
                baseline_activity,
                grid,
                threshold,
                use_onset_split,
            )
            for threshold in ENERGY_THRESHOLD_GRID
        ),
        key=lambda row: row["planner_joint_score_↑"],
    )
    print(f"training-free energy reader: {baseline}", flush=True)

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    order = list(range(len(train_units)))
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(order)
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            chunk = order[start : start + args.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            for position in chunk:
                unit = train_units[position]
                features = unit.features.to(device)
                targets = unit.targets.to(device)
                onset_targets = unit.onset_targets.to(device)
                logits, onset_logits, presence = model(features)
                probability = combined_activity(logits, presence)
                present_targets = (targets.amax(dim=-1) > 0.5).float()

                frame_loss = active_inactive_bce(
                    logits,
                    targets,
                    active_weight=args.active_bce_weight,
                    inactive_weight=args.inactive_bce_weight,
                )
                iou_loss = soft_tversky_loss(
                    probability,
                    targets,
                    alpha=args.tversky_alpha,
                    beta=args.tversky_beta,
                )
                len_loss = coverage_loss(probability, targets)
                presence_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    presence,
                    present_targets,
                )
                onset_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    onset_logits,
                    onset_targets,
                    pos_weight=onset_positive_weight,
                )
                loss = loss + (
                    frame_loss
                    + args.soft_iou_weight * iou_loss
                    + args.coverage_weight * len_loss
                    + args.presence_weight * presence_loss
                    + args.onset_weight * onset_loss
                )
            loss = loss / max(len(chunk), 1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            total_loss += float(loss.detach())
            batches += 1
        schedule.step()

        model.eval()
        val_activity = activity_from_head(model, val_cache, device, zeroed)
        probabilities = []
        targets = []
        for unit in val_units:
            probabilities.append(
                torch.stack(
                    [val_activity[unit.scene_id][label][0] for label in unit.labels]
                )
            )
            targets.append(unit.targets)
        probability = torch.cat(probabilities)
        target = torch.cat(targets)
        sweep = [
            evaluate_planner(
                val_questions,
                val_activity,
                grid,
                threshold,
                use_onset_split,
            )
            for threshold in THRESHOLD_GRID_V2
        ]
        selected = max(sweep, key=lambda row: selection_score(row, args))
        precision, recall, f1 = frame_metrics(
            probability,
            target,
            selected["threshold"],
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "val_frame_precision_↑": precision,
            "val_frame_recall_↑": recall,
            "val_frame_f1_↑": f1,
            "val_selection_score_↑": selection_score(selected, args),
            **{f"val_{key}": value for key, value in selected.items()},
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if best is None or row["val_selection_score_↑"] > best["val_selection_score_↑"]:
            best = row
            torch.save(
                {
                    "format": FORMAT_VERSION,
                    "state_dict": model.state_dict(),
                    "channels": args.channels,
                    "dropout": args.dropout,
                    "threshold": selected["threshold"],
                    "zeroed_feature_groups": list(zeroed),
                    "onset_split": use_onset_split,
                    "epoch": epoch,
                    "selection": row,
                    "selection_weights": {
                        "answer": args.select_answer_weight,
                        "no_evidence": args.select_noev_weight,
                        "evidence_iou": args.select_iou_weight,
                    },
                },
                output_dir / "proposal_head.pt",
            )

    (output_dir / "training_report.json").write_text(
        json.dumps(
            {
                "format": FORMAT_VERSION,
                "arguments": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "train_units": len(train_units),
                "val_units": len(val_units),
                "train_questions": len(train_questions),
                "val_questions": len(val_questions),
                "training_free_energy_reader": baseline,
                "history": history,
                "best": best,
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"best: {json.dumps(best, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
