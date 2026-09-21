#!/usr/bin/env python3
"""Render the QCES-v6 paper figures from fingerprinted receipts and caches.

Two figures, both arguments the manuscript makes and neither decorative:

``presence`` — what the separator's output energy knows about identity versus
what the separator's own query encoder knows about it.

``example`` — one scene end to end: mixture, the candidate stems' predicted
activity, the annotated events, and the span the planner finally gated.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mixi_understanding.qces.event_proposals import (
    ProposalHead,
    decode_proposals,
    features_from_cache,
)
from mixi_understanding.qces.question_parsing import parse_question
from mixi_understanding.qces.relational_planner import plan_from_proposals
from mixi_understanding.qces.stem_features import FrameGrid


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--stem-cache", type=Path, required=True)
    parser.add_argument("--proposal-head", type=Path, required=True)
    parser.add_argument("--proposal-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--example-record", default=None)
    return parser.parse_args(argv)


def load_taxonomy(path: Path) -> tuple[str, ...]:
    composition = json.loads(path.read_text(encoding="utf-8"))["composition"]
    labels: list[str] = []
    for key in ("seen_labels", "nuisance_labels", "heldout_labels"):
        labels.extend(composition.get(key, []))
    return tuple(dict.fromkeys(labels))


def presence_figure(report: dict[str, Any], output: Path) -> None:
    names = {
        "stem_energy_ratio": "stem energy",
        "clap_mixture_similarity_maximum": "mixture$\\times$text",
        "clap_stem_self_similarity": "stem$\\times$text",
        "clap_stem_self_margin": "stem$\\times$text margin",
        "learned_presence": "learned head",
    }
    auc = report["label_presence_auc_↑"]
    keys = [key for key in names if auc.get(key) is not None]
    values = [auc[key] for key in keys]
    figure, axes = plt.subplots(figsize=(3.3, 1.9))
    colours = ["#9aa5b1" if key == "stem_energy_ratio" else "#2f6f9f" for key in keys]
    colours[-1] = "#1b3a57"
    bars = axes.barh(range(len(keys)), values, color=colours, height=0.62)
    axes.set_yticks(range(len(keys)))
    axes.set_yticklabels([names[key] for key in keys], fontsize=7)
    axes.invert_yaxis()
    axes.set_xlim(0.5, 1.0)
    axes.axvline(0.5, color="#c0392b", linewidth=0.8, linestyle=":")
    axes.set_xlabel("label-presence AUC $\\uparrow$", fontsize=7)
    axes.tick_params(axis="x", labelsize=7)
    for bar, value in zip(bars, values):
        axes.text(
            min(value + 0.008, 0.985),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=6.5,
        )
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    figure.tight_layout(pad=0.3)
    figure.savefig(output, dpi=300)
    plt.close(figure)


def example_figure(
    row: dict[str, Any],
    entry: dict[str, Any],
    head: ProposalHead,
    threshold: float,
    taxonomy: Sequence[str],
    grid: FrameGrid,
    output: Path,
) -> None:
    parsed = parse_question(row["question"], row["answer_options"], taxonomy)
    labels = tuple(parsed.query_labels)
    labels, features, _ = features_from_cache(entry, labels)
    with torch.inference_mode():
        logits, onset_logits, presence = head(features)
    activity = torch.sigmoid(logits) * torch.sigmoid(presence)[:, None]
    onsets = torch.sigmoid(onset_logits)
    proposals = decode_proposals(
        labels, activity, grid, threshold=threshold, onset_activity=onsets
    )
    plan = plan_from_proposals(parsed, proposals)

    gold: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for event in row["events"]:
        if event.get("event_kind") == "semantic":
            gold[event["label"]].append(
                (float(event["onset_seconds"]), float(event["offset_seconds"]))
            )

    shown = [label for label in labels if gold.get(label)][:4]
    for label in labels:
        if len(shown) >= 5:
            break
        if label not in shown:
            shown.append(label)
    figure, axes = plt.subplots(
        len(shown) + 1, 1, figsize=(3.4, 0.52 * len(shown) + 1.1), sharex=True
    )
    times = [grid.frame_to_seconds(frame) for frame in range(grid.num_frames)]
    for index, label in enumerate(shown):
        axis = axes[index]
        position = list(labels).index(label)
        axis.plot(times, activity[position].tolist(), color="#1b3a57", linewidth=0.9)
        axis.axhline(threshold, color="#c0392b", linewidth=0.6, linestyle=":")
        for onset, offset in gold.get(label, []):
            axis.axvspan(onset, offset, color="#f0c419", alpha=0.35, linewidth=0)
        axis.set_ylim(-0.05, 1.05)
        axis.set_yticks([])
        axis.set_ylabel(
            label.replace("_", " ")[:14], fontsize=6, rotation=0,
            ha="right", va="center",
        )
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)

    axis = axes[-1]
    for onset, offset in plan.spans:
        axis.axvspan(onset, offset, color="#2f6f9f", alpha=0.75, linewidth=0)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_ylabel("gated\nevidence", fontsize=6, rotation=0, ha="right", va="center")
    axis.set_xlabel("time (s)", fontsize=7)
    axis.tick_params(axis="x", labelsize=7)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)

    answer = plan.answer_label or "no evidence"
    figure.suptitle(
        f"{row['question']}\npredicted: {answer}   gold: {row['answer']}",
        fontsize=6.5,
    )
    figure.tight_layout(pad=0.3, rect=(0.0, 0.0, 1.0, 0.9))
    figure.savefig(output, dpi=300)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.proposal_report.read_text(encoding="utf-8"))
    presence_figure(report, args.output_dir / "fig_presence_auc.png")
    print(f"wrote {args.output_dir / 'fig_presence_auc.png'}")

    taxonomy = load_taxonomy(args.dataset_config.resolve())
    grid = FrameGrid(sample_rate=32_000)
    cache = torch.load(args.stem_cache, map_location="cpu", weights_only=False)
    payload = torch.load(args.proposal_head, map_location="cpu", weights_only=False)
    head = ProposalHead(
        channels=int(payload.get("channels", 96)),
        dropout=float(payload.get("dropout", 0.1)),
    )
    head.load_state_dict(payload["state_dict"])
    head.eval()

    chosen = None
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["scene_id"] not in cache["scenes"]:
                continue
            if args.example_record and row["id"] != args.example_record:
                continue
            if args.example_record is None and (
                row["relation"] != "after" or row["no_evidence"]
            ):
                continue
            chosen = row
            break
    if chosen is None:
        raise SystemExit("no example record matched")
    example_figure(
        chosen,
        cache["scenes"][chosen["scene_id"]],
        head,
        float(payload["threshold"]),
        taxonomy,
        grid,
        args.output_dir / "fig_example.png",
    )
    print(f"wrote {args.output_dir / 'fig_example.png'} for {chosen['id']}")


if __name__ == "__main__":
    main()
