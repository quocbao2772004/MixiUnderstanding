#!/usr/bin/env python3
"""Train the label-agnostic proposal head over frozen-separator stems.

The head never sees a label identity or a class index.  It reads only the
energy/onset/shape descriptor of each returned stem and the competition between
the stems of one candidate set, and predicts which frames that stem's label is
actually active in.  Model selection is end-task aligned: the checkpoint and the
decoding threshold are chosen by the accuracy of the symbolic planner that
consumes the proposals, not by frame F1.

Training units are unique ``(scene, candidate label set)`` pairs so that the
sixteen questions sharing a scene do not enter the objective sixteen times.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.qces.event_proposals import (
    EventProposal,
    ProposalHead,
    decode_proposals,
    energy_activity,
    features_from_cache,
    zero_feature_groups,
)
from mixi_understanding.qces.question_parsing import parse_question
from mixi_understanding.qces.relational_planner import plan_from_proposals
from mixi_understanding.qces.stem_features import (
    NUM_FRAMES,
    FrameGrid,
    intervals_to_frame_targets,
)

FORMAT_VERSION = "qces_v6_proposal_head_v1"
# The grid must reach well below 0.2.  The head's activity is the product of a
# frame probability and a clip-level presence probability, so a genuinely
# present label often peaks around 0.1; a grid that starts at 0.25 silently
# turns those labels into missed detections and the planner then abstains on
# 44 percent of answerable records instead of 18 percent.
THRESHOLD_GRID = (0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60)
# The training-free reader multiplies a competition share by a normalised
# envelope, so with about six candidates its output rarely exceeds ~0.17 and the
# learned head's grid would never fire for it.  Giving the baseline its own,
# lower grid keeps the comparison honest.
ENERGY_THRESHOLD_GRID = (0.02, 0.05, 0.08, 0.12, 0.20, 0.30)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--presence-weight", type=float, default=1.0)
    parser.add_argument("--onset-weight", type=float, default=1.0)
    parser.add_argument("--onset-positive-weight", type=float, default=20.0)
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=0.0,
        help=(
            "Focal down-weighting of easy frames. About 2 percent of "
            "candidate-label frames are positive, because most candidates are "
            "absent classes whose target is all zeros; 0 keeps plain BCE."
        ),
    )
    parser.add_argument("--seed", type=int, default=2028)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--zero-feature-group",
        action="append",
        default=[],
        help=(
            "Ablation: zero every channel of this feature group in both "
            "training and evaluation. Repeatable."
        ),
    )
    parser.add_argument(
        "--no-onset-split",
        action="store_true",
        help="Ablation: decode proposals without the onset-peak split.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


@dataclass
class Unit:
    """One inference-shaped training unit."""

    scene_id: str
    labels: tuple[str, ...]
    features: torch.Tensor
    targets: torch.Tensor
    onset_targets: torch.Tensor


@dataclass
class Question:
    """One evaluation record bound to its scene's cached stems."""

    record_id: str
    scene_id: str
    labels: tuple[str, ...]
    parsed: Any
    gold_answer: str | None
    gold_no_evidence: bool
    gold_spans: tuple[tuple[float, float], ...]


def load_taxonomy(path: Path) -> tuple[str, ...]:
    composition = json.loads(path.read_text(encoding="utf-8"))["composition"]
    labels: list[str] = []
    for key in ("seen_labels", "nuisance_labels", "heldout_labels"):
        labels.extend(composition.get(key, []))
    return tuple(dict.fromkeys(labels))


def read_manifest(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def scene_label_intervals(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, list[tuple[float, float]]]]:
    """Annotated semantic event intervals per scene and label (training only)."""

    intervals: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for row in rows:
        scene_id = row["scene_id"]
        if scene_id in intervals:
            continue
        by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for event in row["events"]:
            if event.get("event_kind") != "semantic":
                continue
            by_label[event["label"]].append(
                (float(event["onset_seconds"]), float(event["offset_seconds"]))
            )
        intervals[scene_id] = {
            label: sorted(spans) for label, spans in by_label.items()
        }
    return intervals


def build_questions(
    rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, Any],
    taxonomy: Sequence[str],
) -> list[Question]:
    questions: list[Question] = []
    for row in rows:
        scene_id = row["scene_id"]
        if scene_id not in cache["scenes"]:
            continue
        parsed = parse_question(row["question"], row["answer_options"], taxonomy)
        events = {event["event_id"]: event for event in row["events"]}
        spans = tuple(
            sorted(
                (
                    float(events[event_id]["onset_seconds"]),
                    float(events[event_id]["offset_seconds"]),
                )
                for event_id in row["evidence_event_ids"]
            )
        )
        questions.append(
            Question(
                record_id=row["id"],
                scene_id=scene_id,
                labels=tuple(parsed.query_labels),
                parsed=parsed,
                gold_answer=None if row["no_evidence"] else row["answer"],
                gold_no_evidence=bool(row["no_evidence"]),
                gold_spans=spans,
            )
        )
    return questions


def build_units(
    questions: Sequence[Question],
    cache: Mapping[str, Any],
    intervals: Mapping[str, Mapping[str, list[tuple[float, float]]]],
    grid: FrameGrid,
    zeroed: Sequence[str] = (),
) -> list[Unit]:
    # The head's features are permutation-equivariant over candidates, so two
    # questions whose candidate sets differ only in mention order are the same
    # training unit and must not be counted twice.
    seen: set[tuple[str, frozenset[str]]] = set()
    units: list[Unit] = []
    for question in questions:
        key = (question.scene_id, frozenset(question.labels))
        if key in seen:
            continue
        seen.add(key)
        features, _ = scene_features(
            cache, question.scene_id, question.labels, zeroed
        )
        spans_by_label = [
            list(intervals[question.scene_id].get(label, []))
            for label in question.labels
        ]
        targets = torch.stack(
            [intervals_to_frame_targets(spans, grid) for spans in spans_by_label]
        )
        onset_targets = torch.stack(
            [onset_frame_targets(spans, grid) for spans in spans_by_label]
        )
        units.append(
            Unit(
                scene_id=question.scene_id,
                labels=question.labels,
                features=features,
                targets=targets,
                onset_targets=onset_targets,
            )
        )
    return units


def onset_frame_targets(
    spans: Sequence[tuple[float, float]], grid: FrameGrid, tolerance: int = 2
) -> torch.Tensor:
    """A narrow band around each annotated onset, for the onset branch."""

    target = torch.zeros(grid.num_frames, dtype=torch.float32)
    for onset, _ in spans:
        centre = grid.seconds_to_frame(onset)
        start = max(0, centre - tolerance)
        stop = min(grid.num_frames, centre + tolerance + 1)
        target[start:stop] = 1.0
    return target


def scene_features(
    cache: Mapping[str, Any],
    scene_id: str,
    labels: Sequence[str],
    zeroed: Sequence[str] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    entry = cache["scenes"][scene_id]
    available = set(entry["labels"])
    missing = [label for label in labels if label not in available]
    if missing:
        raise KeyError(f"{scene_id} cache lacks stems for {missing}")
    _, features, stems = features_from_cache(entry, labels)
    return zero_feature_groups(features, zeroed), stems


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    positive_weight: torch.Tensor,
    focal_gamma: float,
) -> torch.Tensor:
    """Binary cross entropy with an optional focal factor."""

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight, reduction="none"
    )
    if focal_gamma > 0.0:
        probability = torch.sigmoid(logits)
        confidence = targets * probability + (1.0 - targets) * (1.0 - probability)
        loss = loss * (1.0 - confidence).clamp_min(1e-6).pow(focal_gamma)
    return loss.mean()


def frame_metrics(
    probability: torch.Tensor, target: torch.Tensor, threshold: float
) -> tuple[float, float, float]:
    predicted = (probability >= threshold).float()
    true_positive = float((predicted * target).sum())
    predicted_positive = float(predicted.sum())
    actual_positive = float(target.sum())
    precision = true_positive / predicted_positive if predicted_positive else 1.0
    recall = true_positive / actual_positive if actual_positive else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return precision, recall, f1


def span_iou(
    predicted: Sequence[tuple[float, float]],
    gold: Sequence[tuple[float, float]],
    duration: float = 10.0,
    resolution: int = 1000,
) -> float:
    def rasterise(spans: Sequence[tuple[float, float]]) -> torch.Tensor:
        mask = torch.zeros(resolution)
        for onset, offset in spans:
            start = max(0, min(resolution, int(onset / duration * resolution)))
            stop = max(start, min(resolution, int(offset / duration * resolution)))
            mask[start:stop] = 1.0
        return mask

    left = rasterise(predicted)
    right = rasterise(gold)
    union = float(torch.clamp(left + right, max=1.0).sum())
    if union == 0.0:
        return 1.0
    return float((left * right).sum()) / union


def evaluate_planner(
    questions: Sequence[Question],
    activity_by_scene: Mapping[str, Mapping[str, torch.Tensor]],
    grid: FrameGrid,
    threshold: float,
    use_onset_split: bool = True,
) -> dict[str, float]:
    proposals_cache: dict[tuple[str, tuple[str, ...]], list[EventProposal]] = {}
    answer_correct = 0
    answerable = 0
    no_evidence_correct = 0
    iou_total = 0.0
    for question in questions:
        key = (question.scene_id, question.labels)
        if key not in proposals_cache:
            table = activity_by_scene[question.scene_id]
            activity = torch.stack([table[label][0] for label in question.labels])
            onsets = [table[label][1] for label in question.labels]
            onset_activity = (
                None if onsets[0] is None or not use_onset_split
                else torch.stack(onsets)
            )
            proposals_cache[key] = decode_proposals(
                question.labels,
                activity,
                grid,
                threshold=threshold,
                onset_activity=onset_activity,
            )
        plan = plan_from_proposals(question.parsed, proposals_cache[key])
        no_evidence_correct += int(plan.no_evidence == question.gold_no_evidence)
        if question.gold_no_evidence:
            continue
        answerable += 1
        answer_correct += int(plan.answer_label == question.gold_answer)
        iou_total += span_iou(plan.spans, question.gold_spans)
    total = len(questions)
    return {
        "planner_answer_accuracy_↑": answer_correct / max(answerable, 1),
        "planner_no_evidence_accuracy_↑": no_evidence_correct / max(total, 1),
        "planner_evidence_span_iou_↑": iou_total / max(answerable, 1),
        # Selection proxy for the end metric.  SD-SDRi is driven by getting the
        # right labels into the prompt and the right spans into the gate, and
        # span IoU already scores an abstention on an answerable record as
        # zero, so no separate abstention term is needed here.  The separator is
        # never called during model selection.
        "planner_joint_score_↑": (
            0.4 * answer_correct / max(answerable, 1)
            + 0.2 * no_evidence_correct / max(total, 1)
            + 0.4 * iou_total / max(answerable, 1)
        ),
        "threshold": threshold,
    }


def activity_from_head(
    model: ProposalHead | None,
    cache: Mapping[str, Any],
    device: torch.device,
    zeroed: Sequence[str] = (),
) -> dict[str, dict[str, tuple[torch.Tensor, torch.Tensor | None]]]:
    """Per-scene, per-label ``(activity, onset activity)`` in ``[0, 1]``."""

    result: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor | None]]] = {}
    for scene_id, entry in cache["scenes"].items():
        labels, features, rows = features_from_cache(entry)
        features = zero_feature_groups(features, zeroed)
        if model is None:
            activity = energy_activity(rows, entry["mixture"].float())
            onset = None
        else:
            with torch.inference_mode():
                logits, onset_logits, presence = model(features.to(device))
            # The presence branch multiplies the frame activity, so a label the
            # clip never contains is suppressed everywhere instead of leaving a
            # short high-confidence island for the planner to trip over.
            activity = (
                torch.sigmoid(logits) * torch.sigmoid(presence)[:, None]
            ).cpu()
            onset = torch.sigmoid(onset_logits).cpu()
        result[scene_id] = {
            label: (
                activity[position],
                None if onset is None else onset[position],
            )
            for position, label in enumerate(labels)
        }
    return result


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
        train_questions, train_cache, scene_label_intervals(train_rows), grid, zeroed
    )
    val_units = build_units(
        val_questions, val_cache, scene_label_intervals(val_rows), grid, zeroed
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
    positive_weight = torch.tensor(args.positive_weight, device=device)
    onset_weight = torch.tensor(args.onset_positive_weight, device=device)
    use_onset_split = not args.no_onset_split

    # Training-free reference on the same cache.
    baseline_activity = activity_from_head(None, val_cache, device, zeroed)
    baseline = max(
        (
            evaluate_planner(
                val_questions, baseline_activity, grid, threshold, use_onset_split
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
                logits, onset_logits, presence = model(unit.features.to(device))
                targets = unit.targets.to(device)
                loss = loss + masked_bce(
                    logits, targets, positive_weight, args.focal_gamma
                )
                loss = loss + args.presence_weight * (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        presence, (targets.amax(dim=-1) > 0.5).float()
                    )
                )
                loss = loss + args.onset_weight * (
                    torch.nn.functional.binary_cross_entropy_with_logits(
                        onset_logits,
                        unit.onset_targets.to(device),
                        pos_weight=onset_weight,
                    )
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
                val_questions, val_activity, grid, threshold, use_onset_split
            )
            for threshold in THRESHOLD_GRID
        ]
        selected = max(sweep, key=lambda row: row["planner_joint_score_↑"])
        precision, recall, f1 = frame_metrics(
            probability, target, selected["threshold"]
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "val_frame_precision_↑": precision,
            "val_frame_recall_↑": recall,
            "val_frame_f1_↑": f1,
            **{f"val_{key}": value for key, value in selected.items()},
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if best is None or row["val_planner_joint_score_↑"] > best["val_planner_joint_score_↑"]:
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
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"best: {json.dumps(best)}", flush=True)


if __name__ == "__main__":
    main()
