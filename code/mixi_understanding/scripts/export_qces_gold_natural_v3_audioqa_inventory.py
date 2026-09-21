#!/usr/bin/env python3
"""Export predicted event inventories for the multi-intent mentor demo.

The exported predictions never use gold labels to select an event.  Gold scene
events are retained only for a development-set audit and demo case filtering.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.qces.class_aware_red_epn_v1 import (
    ClassAwareRedEpnV1,
    ClassAwareRedEpnV1Config,
)
from mixi_understanding.scripts.export_qces_gold_natural_v3_acoustic_ranker_demo import (
    display_label,
)
from mixi_understanding.scripts.train_qces_gold_natural_v3_acoustic_pair_ranker_v2 import (
    AcousticPairNoneRanker,
    AcousticRankerConfig,
    DEFAULT_BASE,
    DEFAULT_DATA,
    collect_acoustic_scenes,
    pad_scene_bank,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _atomic_json,
    _device,
    _sha256_file,
    load_scene_list,
)


DEFAULT_RANKER = DEFAULT_BASE / "question_conditioned_acoustic_pair_ranker_v2"
DEFAULT_OUTPUT = DEFAULT_RANKER.parent / "audioqa_inventory_demo"
SCORING_VARIANTS = (
    "proposal",
    "clip",
    "semantic",
    "proposal_semantic",
    "clip_semantic",
)


class ProposalBankDataset(Dataset[Mapping[str, torch.Tensor]]):
    def __init__(self, banks: Sequence[Mapping[str, torch.Tensor]]) -> None:
        self.banks = list(banks)

    def __len__(self) -> int:
        return len(self.banks)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        bank = self.banks[index]
        return {
            key: bank[key]
            for key in (
                "scene_features",
                "proposal_start",
                "proposal_end",
                "proposal_label_id",
                "proposal_mask",
            )
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranker-checkpoint",
        type=Path,
        default=DEFAULT_RANKER / "acoustic_pair_ranker_v2_best.pt",
    )
    parser.add_argument(
        "--proposal-checkpoint",
        type=Path,
        default=DEFAULT_BASE
        / "overlap_aware_red_epn_v2/overlap_aware_red_epn_v2_best.pt",
    )
    parser.add_argument(
        "--dev-index",
        type=Path,
        default=DEFAULT_BASE / "dense_overlap_gold_natural_v3_dev/index.json",
    )
    parser.add_argument(
        "--dev-scenes", type=Path, default=DEFAULT_DATA / "scene_ids_overlap_dev.txt"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--inventory-size", type=int, default=4)
    parser.add_argument("--max-scenes", type=int, default=0)
    return parser.parse_args()


def _write_jsonl_atomic(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _variant_score(proposal: Mapping[str, Any], variant: str) -> float:
    proposal_score = max(float(proposal["score"]), 1e-8)
    clip_score = max(float(proposal["clip_score"]), 1e-8)
    semantic_score = max(float(proposal["semantic_score"]), 1e-8)
    if variant == "proposal":
        return proposal_score
    if variant == "clip":
        return clip_score
    if variant == "semantic":
        return semantic_score
    if variant == "proposal_semantic":
        return math.sqrt(proposal_score * semantic_score)
    if variant == "clip_semantic":
        return math.sqrt(clip_score * semantic_score)
    raise ValueError(f"unknown scoring variant: {variant}")


def _select_inventory(
    proposals: Sequence[Mapping[str, Any]], variant: str, size: int
) -> list[dict[str, Any]]:
    by_label: dict[int, list[Mapping[str, Any]]] = {}
    for proposal in proposals:
        by_label.setdefault(int(proposal["label_id"]), []).append(proposal)
    ranked: list[tuple[float, int, list[Mapping[str, Any]]]] = []
    for label_id, label_proposals in by_label.items():
        class_score = max(_variant_score(row, variant) for row in label_proposals)
        ranked.append((class_score, label_id, label_proposals))
    ranked.sort(reverse=True)
    result: list[dict[str, Any]] = []
    for class_score, label_id, label_proposals in ranked[:size]:
        occurrences = sorted(
            label_proposals,
            key=lambda row: (float(row["start"]), -_variant_score(row, variant)),
        )
        result.append(
            {
                "label_id": label_id,
                "score": class_score,
                "occurrences": [dict(row) for row in occurrences],
            }
        )
    return result


def _audit(
    rows: Sequence[Mapping[str, Any]], variant: str, inventory_size: int
) -> dict[str, float | int]:
    recalls: list[float] = []
    precisions: list[float] = []
    exact = 0
    four_event_exact = 0
    four_event_scenes = 0
    for row in rows:
        gold = {str(event["label"]) for event in row["gold_events"]}
        predicted = {
            str(event["label"])
            for event in row["inventories"][variant]
        }
        intersection = len(gold & predicted)
        recalls.append(intersection / max(len(gold), 1))
        precisions.append(intersection / max(len(predicted), 1))
        exact += int(gold == predicted)
        if len(gold) == inventory_size:
            four_event_scenes += 1
            four_event_exact += int(gold == predicted)
    count = len(rows)
    return {
        "scenes": count,
        "macro_label_recall_↑": sum(recalls) / max(count, 1),
        "macro_label_precision_↑": sum(precisions) / max(count, 1),
        "exact_inventory_accuracy_↑": exact / max(count, 1),
        "four_event_scenes": four_event_scenes,
        "four_event_exact_inventory_accuracy_↑": four_event_exact
        / max(four_event_scenes, 1),
    }


def main() -> None:
    args = parse_args()
    if args.inventory_size <= 0:
        raise ValueError("inventory-size must be positive")
    device = _device(args.device)
    ranker_checkpoint = torch.load(
        args.ranker_checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    ranker_config = AcousticRankerConfig(**dict(ranker_checkpoint["config"]))
    proposal_checkpoint = torch.load(
        args.proposal_checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    proposal_config = ClassAwareRedEpnV1Config(
        **dict(proposal_checkpoint["config"])
    )
    proposal_model = ClassAwareRedEpnV1(proposal_config)
    proposal_model.load_state_dict(
        proposal_checkpoint["model_state_dict"], strict=True
    )
    proposal_model.to(device).eval()
    store = DenseFeatureStore([args.dev_index.resolve()], cache_size=16)
    scene_ids = load_scene_list(args.dev_scenes.resolve())
    scenes = collect_acoustic_scenes(
        proposal_model,
        store,
        scene_ids,
        device=device,
        batch_size=args.batch_size,
        max_proposals=ranker_config.max_proposals,
        max_scenes=args.max_scenes,
        iou_threshold=0.30,
    )
    del proposal_model
    torch.cuda.empty_cache()
    banks = [pad_scene_bank(scene, ranker_config) for scene in scenes]
    loader = DataLoader(
        ProposalBankDataset(banks),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    ranker = AcousticPairNoneRanker(ranker_config).to(device)
    ranker.load_state_dict(ranker_checkpoint["model_state_dict"], strict=True)
    ranker.eval()
    semantic_scores: list[list[float]] = []
    with torch.inference_mode():
        for raw in loader:
            features = raw["scene_features"].to(device).float()
            starts = raw["proposal_start"].to(device)
            ends = raw["proposal_end"].to(device)
            mask = raw["proposal_mask"].to(device)
            label_ids = raw["proposal_label_id"].to(device)
            _, semantic_logits = ranker.encode_spans(features, starts, ends, mask)
            own = semantic_logits.gather(-1, label_ids[..., None]).squeeze(-1).sigmoid()
            for batch_index in range(own.shape[0]):
                valid = int(mask[batch_index].sum())
                semantic_scores.append(own[batch_index, :valid].cpu().tolist())

    labels = list(store.labels or [])
    rows: list[dict[str, Any]] = []
    for scene, scores in zip(scenes, semantic_scores, strict=True):
        metadata = store.metadata(str(scene["scene_id"]))
        duration = float(metadata.get("duration_seconds", 10.0))
        proposals: list[dict[str, Any]] = []
        for proposal, semantic_score in zip(scene["proposals"], scores, strict=True):
            item = dict(proposal)
            label_id = int(item["label_id"])
            item["label"] = labels[label_id]
            item["display_label"] = display_label(labels[label_id])
            item["semantic_score"] = float(semantic_score)
            item["start_seconds"] = float(item["start"]) * duration
            item["end_seconds"] = float(item["end"]) * duration
            proposals.append(item)
        inventories: dict[str, list[dict[str, Any]]] = {}
        for variant in SCORING_VARIANTS:
            inventory = _select_inventory(proposals, variant, args.inventory_size)
            for event in inventory:
                event["label"] = labels[int(event["label_id"])]
                event["display_label"] = display_label(event["label"])
            inventories[variant] = inventory
        gold_events = [
            {
                "label": str(event["label"]),
                "display_label": display_label(str(event["label"])),
                "start_seconds": float(event["onset_seconds"]),
                "end_seconds": float(event["offset_seconds"]),
            }
            for event in metadata.get("events", [])
            if str(event.get("event_kind", "semantic")) == "semantic"
        ]
        rows.append(
            {
                "scene_id": str(scene["scene_id"]),
                "duration_seconds": duration,
                "mixture_path": str(Path(str(metadata["mixture_path"])).resolve()),
                "inventories": inventories,
                "gold_events": gold_events,
            }
        )

    audits = {
        variant: _audit(rows, variant, args.inventory_size)
        for variant in SCORING_VARIANTS
    }
    best_variant = max(
        SCORING_VARIANTS,
        key=lambda variant: (
            float(audits[variant]["four_event_exact_inventory_accuracy_↑"]),
            float(audits[variant]["macro_label_recall_↑"]),
        ),
    )
    for row in rows:
        row["predicted_inventory"] = row["inventories"][best_variant]
        predicted = {event["label"] for event in row["predicted_inventory"]}
        gold = {event["label"] for event in row["gold_events"]}
        row["audit_only"] = {
            "gold_event_count": len(gold),
            "correct_labels": len(predicted & gold),
            "exact_inventory": predicted == gold,
        }
    output = args.output_dir.resolve()
    _write_jsonl_atomic(rows, output / "scene_inventories.jsonl")
    summary = {
        "format": "qces_gold_natural_v3_audioqa_inventory_demo_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenes": len(rows),
        "classes": len(labels),
        "inventory_size": args.inventory_size,
        "labels": labels,
        "scoring_variants": audits,
        "selected_variant": best_variant,
        "selection_note": (
            "Scoring variant selected on dev four-event exact inventory accuracy; "
            "gold is never used for per-scene inference."
        ),
        "proposal_checkpoint_sha256": _sha256_file(args.proposal_checkpoint.resolve()),
        "ranker_checkpoint_sha256": _sha256_file(args.ranker_checkpoint.resolve()),
        "scene_inventories_file": str(output / "scene_inventories.jsonl"),
    }
    _atomic_json(summary, output / "summary.json")
    print(
        json.dumps(
            {
                "complete": True,
                "scenes": len(rows),
                "selected_variant": best_variant,
                "audit": audits[best_variant],
                "output_dir": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
