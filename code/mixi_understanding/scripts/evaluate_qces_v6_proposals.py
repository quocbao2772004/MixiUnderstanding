#!/usr/bin/env python3
"""Score the proposal stage of QCES-v6 on its own terms.

The pipeline table says how good the separated evidence is.  This script says
*why*, by measuring the three things the symbolic planner actually consumes:

* whether a candidate label is present in the scene at all (the ``no_evidence``
  decision reduces to this),
* how many times it occurs (the ordinal in ``the third Camera`` reduces to
  this),
* and where those occurrences start and stop.

It also reports the training-free readers side by side, which is what shows
that the separator's output energy is a weak identity signal while the
separator's own query encoder applied to that same output is a strong one.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from mixi_understanding.qces.event_proposals import (
    ProposalHead,
    decode_proposals,
    energy_activity,
    features_from_cache,
    zero_feature_groups,
)
from mixi_understanding.qces.question_parsing import parse_question
from mixi_understanding.qces.stem_features import (
    FrameGrid,
    intervals_to_frame_targets,
)

FORMAT_VERSION = "qces_v6_proposal_evaluation_v1"
MATCH_IOU = 0.5


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--stem-cache", type=Path, required=True)
    parser.add_argument("--proposal-head", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--energy-threshold", type=float, default=0.3)
    return parser.parse_args(argv)


def load_taxonomy(path: Path) -> tuple[str, ...]:
    composition = json.loads(path.read_text(encoding="utf-8"))["composition"]
    labels: list[str] = []
    for key in ("seen_labels", "nuisance_labels", "heldout_labels"):
        labels.extend(composition.get(key, []))
    return tuple(dict.fromkeys(labels))


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    if not positive or not negative:
        return None
    ordered = sorted(
        [(value, 1) for value in positive] + [(value, 0) for value in negative]
    )
    ranks: dict[int, float] = {}
    index = 0
    total_positive_rank = 0.0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][0] == ordered[index][0]:
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            if ordered[position][1] == 1:
                total_positive_rank += average_rank
        index = stop + 1
    count_positive = len(positive)
    count_negative = len(negative)
    return (
        total_positive_rank - count_positive * (count_positive + 1) / 2.0
    ) / (count_positive * count_negative)


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    start = max(left[0], right[0])
    stop = min(left[1], right[1])
    intersection = max(0.0, stop - start)
    union = (left[1] - left[0]) + (right[1] - right[0]) - intersection
    return intersection / union if union > 0 else 0.0


def match_events(
    predicted: Sequence[tuple[float, float]],
    gold: Sequence[tuple[float, float]],
) -> tuple[int, list[tuple[float, float]]]:
    """Greedy highest-IoU matching; returns hits and (onset, offset) errors."""

    remaining = list(range(len(gold)))
    hits = 0
    errors: list[tuple[float, float]] = []
    for span in sorted(predicted):
        best_index = None
        best_iou = MATCH_IOU
        for index in remaining:
            value = interval_iou(span, gold[index])
            if value >= best_iou:
                best_iou = value
                best_index = index
        if best_index is not None:
            remaining.remove(best_index)
            hits += 1
            errors.append(
                (
                    abs(span[0] - gold[best_index][0]),
                    abs(span[1] - gold[best_index][1]),
                )
            )
    return hits, errors


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device)
    grid = FrameGrid(sample_rate=32_000)
    taxonomy = load_taxonomy(args.dataset_config.resolve())
    cache = torch.load(args.stem_cache, map_location="cpu", weights_only=False)
    payload = torch.load(args.proposal_head, map_location="cpu", weights_only=False)
    head = ProposalHead(
        channels=int(payload.get("channels", 96)),
        dropout=float(payload.get("dropout", 0.1)),
    )
    head.load_state_dict(payload["state_dict"])
    head = head.to(device).eval()
    threshold = float(payload["threshold"])
    zeroed = tuple(payload.get("zeroed_feature_groups", ()))
    onset_split = bool(payload.get("onset_split", True))

    intervals: dict[str, dict[str, list[tuple[float, float]]]] = {}
    candidate_sets: dict[str, set[str]] = defaultdict(set)
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            scene_id = row["scene_id"]
            parsed = parse_question(row["question"], row["answer_options"], taxonomy)
            candidate_sets[scene_id].update(parsed.query_labels)
            if scene_id not in intervals:
                by_label: dict[str, list[tuple[float, float]]] = defaultdict(list)
                for event in row["events"]:
                    if event.get("event_kind") != "semantic":
                        continue
                    by_label[event["label"]].append(
                        (
                            float(event["onset_seconds"]),
                            float(event["offset_seconds"]),
                        )
                    )
                intervals[scene_id] = {
                    label: sorted(spans) for label, spans in by_label.items()
                }

    presence_signals: dict[str, tuple[list[float], list[float]]] = {
        name: ([], []) for name in (
            "stem_energy_ratio",
            "clap_stem_self_similarity",
            "clap_stem_self_margin",
            "clap_mixture_similarity_maximum",
            "learned_presence",
        )
    }
    frame_counts = {name: [0.0, 0.0, 0.0] for name in ("energy_reader", "learned_head")}
    event_counts = {
        name: {"hits": 0, "predicted": 0, "gold": 0, "onset": [], "offset": []}
        for name in ("energy_reader", "learned_head")
    }
    occurrence_counts = {name: [0, 0] for name in ("energy_reader", "learned_head")}

    for scene_id, entry in cache["scenes"].items():
        if scene_id not in intervals:
            continue
        labels = list(entry["labels"])
        keep = [
            position
            for position, label in enumerate(labels)
            if label in candidate_sets[scene_id]
        ]
        labels = [labels[position] for position in keep]
        mixture = entry["mixture"].float()
        clap_stem = entry["clap_stem_similarity"].float()[keep][:, keep]
        clap_mixture = entry["clap_mixture_similarity"].float()[keep]
        labels, features, rows = features_from_cache(entry, labels)
        labels = list(labels)
        features = zero_feature_groups(features, zeroed)
        with torch.inference_mode():
            logits, onset_logits, presence = head(features.to(device))
        learned = (
            torch.sigmoid(logits) * torch.sigmoid(presence)[:, None]
        ).cpu()
        learned_onsets = torch.sigmoid(onset_logits).cpu()
        free = energy_activity(rows, mixture)

        diagonal = torch.diagonal(clap_stem)
        if clap_stem.shape[0] > 1:
            masked = clap_stem - torch.diag(
                torch.full_like(diagonal, float("inf"))
            )
            best_other = masked.amax(dim=1)
        else:
            best_other = torch.zeros_like(diagonal)

        readers = {
            "energy_reader": decode_proposals(
                labels, free, grid, threshold=args.energy_threshold
            ),
            "learned_head": decode_proposals(
                labels,
                learned,
                grid,
                threshold=threshold,
                onset_activity=learned_onsets if onset_split else None,
            ),
        }
        by_reader_label: dict[str, dict[str, list[tuple[float, float]]]] = {
            name: defaultdict(list) for name in readers
        }
        for name, proposals in readers.items():
            for proposal in proposals:
                by_reader_label[name][proposal.label].append(
                    (proposal.onset_seconds, proposal.offset_seconds)
                )

        for position, label in enumerate(labels):
            gold = intervals[scene_id].get(label, [])
            bucket = 0 if gold else 1
            presence_signals["stem_energy_ratio"][bucket].append(
                float(rows[position, 0].sum() / mixture[0].sum().clamp_min(1e-8))
            )
            presence_signals["clap_stem_self_similarity"][bucket].append(
                float(diagonal[position])
            )
            presence_signals["clap_stem_self_margin"][bucket].append(
                float(diagonal[position] - best_other[position])
            )
            presence_signals["clap_mixture_similarity_maximum"][bucket].append(
                float(clap_mixture[position].max())
            )
            presence_signals["learned_presence"][bucket].append(
                float(torch.sigmoid(presence[position]))
            )

            target = intervals_to_frame_targets(gold, grid)
            for name, activity in (
                ("energy_reader", free[position]),
                ("learned_head", learned[position]),
            ):
                cut = args.energy_threshold if name == "energy_reader" else threshold
                predicted = (activity >= cut).float()
                frame_counts[name][0] += float((predicted * target).sum())
                frame_counts[name][1] += float(predicted.sum())
                frame_counts[name][2] += float(target.sum())

                spans = by_reader_label[name].get(label, [])
                hits, errors = match_events(spans, gold)
                event_counts[name]["hits"] += hits
                event_counts[name]["predicted"] += len(spans)
                event_counts[name]["gold"] += len(gold)
                event_counts[name]["onset"].extend(error[0] for error in errors)
                event_counts[name]["offset"].extend(error[1] for error in errors)
                if gold:
                    occurrence_counts[name][1] += 1
                    occurrence_counts[name][0] += int(len(spans) == len(gold))

    def ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else 0.0

    report: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "manifest": str(args.manifest.resolve()),
        "stem_cache": str(args.stem_cache.resolve()),
        "proposal_head": str(args.proposal_head.resolve()),
        "proposal_threshold": threshold,
        "energy_threshold": args.energy_threshold,
        "zeroed_feature_groups": list(zeroed),
        "onset_split": onset_split,
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
        "label_presence_auc_↑": {
            name: roc_auc(values[0], values[1])
            for name, values in presence_signals.items()
        },
        "label_presence_counts": {
            "present": len(presence_signals["stem_energy_ratio"][0]),
            "absent": len(presence_signals["stem_energy_ratio"][1]),
        },
        "readers": {},
    }
    for name in frame_counts:
        hits, predicted, gold = frame_counts[name]
        precision = ratio(hits, predicted)
        recall = ratio(hits, gold)
        events = event_counts[name]
        event_precision = ratio(events["hits"], events["predicted"])
        event_recall = ratio(events["hits"], events["gold"])
        report["readers"][name] = {
            "frame_precision_↑": precision,
            "frame_recall_↑": recall,
            "frame_f1_↑": ratio(2 * precision * recall, precision + recall),
            "event_precision_↑": event_precision,
            "event_recall_↑": event_recall,
            "event_f1_↑": ratio(
                2 * event_precision * event_recall, event_precision + event_recall
            ),
            "onset_mae_seconds_↓": (
                float(sum(events["onset"]) / len(events["onset"]))
                if events["onset"]
                else None
            ),
            "offset_mae_seconds_↓": (
                float(sum(events["offset"]) / len(events["offset"]))
                if events["offset"]
                else None
            ),
            "occurrence_count_accuracy_↑": ratio(
                occurrence_counts[name][0], occurrence_counts[name][1]
            ),
            "matched_event_count": events["hits"],
            "predicted_event_count": events["predicted"],
            "gold_event_count": events["gold"],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["label_presence_auc_↑"], indent=2, ensure_ascii=False))
    for name, values in report["readers"].items():
        print(
            f"{name:14s} frameF1 {values['frame_f1_↑']:.3f} "
            f"eventF1 {values['event_f1_↑']:.3f} "
            f"count_acc {values['occurrence_count_accuracy_↑']:.3f} "
            f"onsetMAE {values['onset_mae_seconds_↓']} "
            f"offsetMAE {values['offset_mae_seconds_↓']}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
