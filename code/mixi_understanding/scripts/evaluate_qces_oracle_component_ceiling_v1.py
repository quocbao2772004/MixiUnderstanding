#!/usr/bin/env python3
"""Measure the semantic ceiling of exact isolated overlap-v2 components.

This is a diagnostic, not a deployable system.  The frozen R1 detector sees
the exact event waveform used to render each overlap-v2 scene in two forms:

* ``isolated``: the component is left aligned in an otherwise silent clip;
* ``aligned``: the same component is placed at its original scene timestamp.

The corresponding R1 mixture logits are read from the frozen dense-feature
receipt and pooled only inside the oracle event interval.  Gold labels are
used after inference for scoring; they are never model inputs.  Comparing the
three conditions separates source/data ambiguity, position/padding effects,
and degradation caused by polyphonic overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
PRETRAINED_SED_ROOT = CODE_ROOT / "baseline" / "PretrainedSED"
for value in (CODE_ROOT, PRETRAINED_SED_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset

from mixi_understanding.scripts.train_qces_beats_event_rank_v1 import forward_logits
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    load_model,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    _sha256_file,
)


FORMAT = "qces_oracle_component_ceiling_v1"
SAMPLE_RATE = 16_000
FIXED_SECONDS = 10
FIXED_SAMPLES = SAMPLE_RATE * FIXED_SECONDS
FRAME_HOP_SECONDS = 0.04
TOP_KS = (1, 5, 8, 20)


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full191/headtransfer_r1_v1"
    overlap = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2"
    components = PROJECT_ROOT / "outputs/qces_full191_overlap_components_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-manifest",
        type=Path,
        default=components / "event_components_dev.jsonl",
    )
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        default=overlap / "detector_scene_manifest_overlap_dev.jsonl",
    )
    parser.add_argument(
        "--dense-index",
        type=Path,
        default=base / "dense_overlap_query_dev_v2/index.json",
    )
    parser.add_argument(
        "--r1-checkpoint",
        type=Path,
        default=base / "pretrainedsed_beats_qces_detector.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "oracle_component_ceiling_v1",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pretrained-checkpoint", default="BEATs_strong_1")
    parser.add_argument("--batch-size", type=int, default=4, help="Events per batch; inference sees twice this many waveforms.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True)
class ComponentRow:
    event_id: str
    scene_id: str
    label: str
    label_id: int
    component_path: Path
    onset_frame: int
    offset_frame: int
    onset_sample: int
    offset_sample: int
    cleanliness_tier: str
    duration_equalized: bool
    recipe_kind: str
    requested_overlap_fraction: float
    maximum_concurrency: int
    source_id: str
    source_sha256: str


def load_rows(
    component_manifest: Path,
    scene_manifest: Path,
    labels: Sequence[str],
    *,
    max_events: int = 0,
) -> list[ComponentRow]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    scene_rows = read_jsonl(scene_manifest)
    scene_by_id = {str(row["scene_id"]): row for row in scene_rows}
    if len(scene_by_id) != len(scene_rows):
        raise ValueError("scene manifest has duplicate scene_id")
    event_metadata: dict[str, Mapping[str, Any]] = {}
    for scene in scene_rows:
        for event in scene.get("events") or []:
            event_id = str(event["event_id"])
            if event_id in event_metadata:
                raise ValueError(f"duplicate event_id: {event_id}")
            event_metadata[event_id] = event

    result: list[ComponentRow] = []
    seen: set[str] = set()
    for raw in read_jsonl(component_manifest):
        event_id = str(raw["event_id"])
        if event_id in seen:
            raise ValueError(f"duplicate component event_id: {event_id}")
        seen.add(event_id)
        scene_id = str(raw["scene_id"])
        scene = scene_by_id.get(scene_id)
        event = event_metadata.get(event_id)
        if scene is None or event is None:
            raise ValueError(f"component lacks scene/event metadata: {event_id}")
        label = str(raw["label"])
        if label not in label_to_id:
            raise ValueError(f"unknown ontology label: {label}")
        expected = (
            str(event["label"]),
            int(event["onset_frame"]),
            int(event["offset_frame"]),
            str(event["source_id"]),
        )
        observed = (
            label,
            int(raw["onset_frame"]),
            int(raw["offset_frame"]),
            str(raw["source_id"]),
        )
        if observed != expected:
            raise ValueError(f"component/scene metadata mismatch for {event_id}")
        path = Path(str(raw["component_path"])).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        onset_sample = int(raw["onset_sample"])
        offset_sample = int(raw["offset_sample"])
        if not 0 <= onset_sample < offset_sample <= FIXED_SAMPLES:
            raise ValueError(f"invalid component bounds: {event_id}")
        result.append(
            ComponentRow(
                event_id=event_id,
                scene_id=scene_id,
                label=label,
                label_id=label_to_id[label],
                component_path=path,
                onset_frame=int(raw["onset_frame"]),
                offset_frame=int(raw["offset_frame"]),
                onset_sample=onset_sample,
                offset_sample=offset_sample,
                cleanliness_tier=str(event.get("cleanliness_tier") or "unknown").lower(),
                duration_equalized=bool(raw.get("duration_equalized")),
                recipe_kind=str(raw.get("recipe_kind") or scene.get("recipe_kind") or "unknown"),
                requested_overlap_fraction=float(raw.get("requested_overlap_fraction", scene.get("requested_overlap_fraction", 0.0))),
                maximum_concurrency=int(scene.get("maximum_concurrency") or 0),
                source_id=str(raw["source_id"]),
                source_sha256=str(raw["source_sha256"]),
            )
        )
        if max_events > 0 and len(result) >= max_events:
            break
    if not result:
        raise ValueError("component audit has no events")
    return result


def build_variants(component: torch.Tensor, row: ComponentRow) -> tuple[torch.Tensor, torch.Tensor]:
    component = component.float().reshape(-1)
    expected = row.offset_sample - row.onset_sample
    if component.numel() != expected:
        raise ValueError(
            f"{row.event_id}: component samples={component.numel()} expected={expected}"
        )
    isolated = F.pad(component, (0, FIXED_SAMPLES - component.numel()))
    aligned = component.new_zeros(FIXED_SAMPLES)
    aligned[row.onset_sample : row.offset_sample] = component
    return isolated, aligned


class ComponentDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[ComponentRow]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        waveform, sample_rate = torchaudio.load(row.component_path)
        waveform = waveform.float().mean(dim=0)
        if int(sample_rate) != SAMPLE_RATE:
            waveform = AF.resample(waveform, int(sample_rate), SAMPLE_RATE)
        isolated, aligned = build_variants(waveform, row)
        return {"isolated": isolated, "aligned": aligned, "row": row}


def collate_components(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    isolated = torch.stack([item["isolated"] for item in items])
    aligned = torch.stack([item["aligned"] for item in items])
    return {
        "waveforms": torch.cat((isolated, aligned), dim=0),
        "rows": [item["row"] for item in items],
    }


def rank_result(scores: torch.Tensor, label_id: int, labels: Sequence[str]) -> dict[str, Any]:
    ordering = scores.argsort(descending=True)
    location = torch.where(ordering == int(label_id))[0]
    if location.numel() != 1:
        raise RuntimeError("gold label must occur exactly once in ordering")
    rank = int(location.item()) + 1
    predicted_id = int(ordering[0].item())
    negative = scores.clone()
    negative[int(label_id)] = -torch.inf
    strongest_negative = float(negative.max().item())
    return {
        "rank": rank,
        "predicted_label": str(labels[predicted_id]),
        "target_logit": float(scores[int(label_id)].item()),
        "target_margin": float(scores[int(label_id)].item()) - strongest_negative,
    }


def condition_summary(items: Sequence[Mapping[str, Any]], condition: str) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot summarize empty items")
    ranks = [int(item[condition]["rank"]) for item in items]
    result: dict[str, Any] = {
        "events": len(items),
        "mean_label_rank_↓": sum(ranks) / len(ranks),
        "mean_target_margin_↑": sum(float(item[condition]["target_margin"]) for item in items) / len(items),
    }
    for top_k in TOP_KS:
        result[f"top{top_k}_accuracy_↑"] = sum(rank <= top_k for rank in ranks) / len(ranks)
    return result


def grouped_summaries(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item[key])].append(item)
    result: dict[str, Any] = {}
    for value, rows in sorted(groups.items()):
        result[value] = {
            condition: condition_summary(rows, condition)
            for condition in ("isolated", "aligned", "mixture")
        }
    return result


def top_confusions(
    items: Sequence[Mapping[str, Any]], condition: str, *, limit: int = 30
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for item in items:
        predicted = str(item[condition]["predicted_label"])
        gold = str(item["label"])
        if predicted != gold:
            counts[(gold, predicted)] += 1
    return [
        {"gold_label": gold, "predicted_label": predicted, "count_↓": count}
        for (gold, predicted), count in counts.most_common(limit)
    ]


def decision_from_summary(summary: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    isolated = summary["isolated"]
    aligned = summary["aligned"]
    mixture = summary["mixture"]
    isolated_top1 = float(isolated["top1_accuracy_↑"])
    isolated_top5 = float(isolated["top5_accuracy_↑"])
    mixture_top1 = float(mixture["top1_accuracy_↑"])
    aligned_top1 = float(aligned["top1_accuracy_↑"])
    source_gate = isolated_top1 >= 0.85 and isolated_top5 >= 0.95
    overlap_gap = isolated_top1 - mixture_top1
    position_gap = isolated_top1 - aligned_top1
    overlap_gate = overlap_gap >= 0.20
    position_gate = abs(position_gap) <= 0.05
    if source_gate and overlap_gate and position_gate:
        recommendation = "separator_first"
        reason = "isolated components are recognizable, while overlap causes a large semantic loss"
    elif isolated_top1 < 0.80 or isolated_top5 < 0.90:
        recommendation = "repair_data_or_ontology"
        reason = "the detector cannot reliably identify exact isolated components"
    elif not position_gate:
        recommendation = "repair_padding_or_position_contract"
        reason = "moving the same component on the 10-second grid changes recognition excessively"
    else:
        recommendation = "mixed_bottleneck"
        reason = "neither source quality nor overlap alone explains the observed ceiling"
    return {
        "recommendation": recommendation,
        "reason": reason,
        "source_recognizability_gate": {
            "isolated_top1_ge_0_85": isolated_top1 >= 0.85,
            "isolated_top5_ge_0_95": isolated_top5 >= 0.95,
            "passed": source_gate,
        },
        "overlap_degradation_gate": {
            "isolated_minus_mixture_top1_↑": overlap_gap,
            "threshold_ge": 0.20,
            "passed": overlap_gate,
        },
        "position_stability_gate": {
            "isolated_minus_aligned_top1": position_gap,
            "absolute_threshold_le": 0.05,
            "passed": position_gate,
        },
    }


def _class_summary(items: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["label"])].append(item)
    details = {
        label: {
            condition: condition_summary(rows, condition)
            for condition in ("isolated", "aligned", "mixture")
        }
        for label, rows in sorted(grouped.items())
    }
    ranked = sorted(
        (
            {
                "label": label,
                "events": int(values["isolated"]["events"]),
                "isolated_top1_↑": float(values["isolated"]["top1_accuracy_↑"]),
                "aligned_top1_↑": float(values["aligned"]["top1_accuracy_↑"]),
                "mixture_top1_↑": float(values["mixture"]["top1_accuracy_↑"]),
            }
            for label, values in details.items()
        ),
        key=lambda row: (row["isolated_top1_↑"], row["mixture_top1_↑"], row["label"]),
    )
    return details, ranked[:30]


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    mixture_store: DenseFeatureStore,
    labels: Sequence[str],
    device: torch.device,
    *,
    amp: bool,
) -> list[dict[str, Any]]:
    model.eval()
    items: list[dict[str, Any]] = []
    processed = 0
    for batch in loader:
        rows: list[ComponentRow] = batch["rows"]
        count = len(rows)
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = forward_logits(model, waveforms).float()
        isolated_logits = logits[:count].cpu()
        aligned_logits = logits[count:].cpu()
        for index, row in enumerate(rows):
            duration_frames = row.offset_frame - row.onset_frame
            if duration_frames <= 0:
                raise RuntimeError(f"empty event interval: {row.event_id}")
            mixture = mixture_store.get(row.scene_id)["logits"].float()
            if mixture.shape[-1] != len(labels):
                raise RuntimeError("mixture ontology dimension mismatch")
            scores = {
                "isolated": isolated_logits[index, :duration_frames].mean(dim=0),
                "aligned": aligned_logits[index, row.onset_frame : row.offset_frame].mean(dim=0),
                "mixture": mixture[row.onset_frame : row.offset_frame].mean(dim=0),
            }
            items.append(
                {
                    "event_id": row.event_id,
                    "scene_id": row.scene_id,
                    "label": row.label,
                    "label_id": row.label_id,
                    "cleanliness_tier": row.cleanliness_tier,
                    "duration_equalized": row.duration_equalized,
                    "recipe_kind": row.recipe_kind,
                    "requested_overlap_fraction": row.requested_overlap_fraction,
                    "maximum_concurrency": row.maximum_concurrency,
                    "duration_frames": duration_frames,
                    "source_id": row.source_id,
                    "source_sha256": row.source_sha256,
                    **{
                        condition: rank_result(value, row.label_id, labels)
                        for condition, value in scores.items()
                    },
                }
            )
        processed += count
        if processed == count or processed % 200 < count:
            print(f"oracle-component audit: {processed} events", flush=True)
    return items


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.max_events < 0:
        raise SystemExit("batch size must be positive and worker/event limits non-negative")
    required = (
        args.component_manifest,
        args.scene_manifest,
        args.dense_index,
        args.r1_checkpoint,
    )
    for path in required:
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    mixture_store = DenseFeatureStore([args.dense_index.resolve()], cache_size=8)
    labels = list(mixture_store.labels or [])
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError(
            f"expected a non-trivial unique frozen ontology, got {len(labels)} labels"
        )
    rows = load_rows(
        args.component_manifest.resolve(),
        args.scene_manifest.resolve(),
        labels,
        max_events=args.max_events,
    )
    missing_scenes = sorted({row.scene_id for row in rows} - set(mixture_store.scene_ids))
    if missing_scenes:
        raise ValueError(f"dense store misses scenes: {missing_scenes[:5]}")

    device = make_device(args.device)
    model = load_model(len(labels), args.pretrained_checkpoint, device, unfreeze_last_blocks=0)
    checkpoint = torch.load(args.r1_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if list(checkpoint.get("labels") or []) != labels:
        raise ValueError("R1 checkpoint ontology mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    loader = DataLoader(
        ComponentDataset(rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_components,
    )
    items = evaluate(
        model,
        loader,
        mixture_store,
        labels,
        device,
        amp=args.amp,
    )
    summary = {
        condition: condition_summary(items, condition)
        for condition in ("isolated", "aligned", "mixture")
    }
    class_details, worst_classes = _class_summary(items)
    decision = decision_from_summary(summary)
    report = {
        "format": FORMAT,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "oracle exact-component recognition diagnostic; gold labels and intervals "
            "are used for scoring only, not as model inputs or deployable predictions"
        ),
        "answer_label_used_as_model_input": False,
        "device": str(device),
        "events": len(items),
        "scenes": len({item["scene_id"] for item in items}),
        "classes": len({item["label"] for item in items}),
        "inputs": {
            "component_manifest": str(args.component_manifest.resolve()),
            "component_manifest_sha256": _sha256_file(args.component_manifest.resolve()),
            "scene_manifest": str(args.scene_manifest.resolve()),
            "scene_manifest_sha256": _sha256_file(args.scene_manifest.resolve()),
            "dense_index": str(args.dense_index.resolve()),
            "dense_index_sha256": _sha256_file(args.dense_index.resolve()),
            "r1_checkpoint": str(args.r1_checkpoint.resolve()),
            "r1_checkpoint_sha256": _sha256_file(args.r1_checkpoint.resolve()),
        },
        "conditions": {
            "isolated": "exact rendered event component, left aligned in ten seconds of silence",
            "aligned": "same exact component at its original ten-second scene position",
            "mixture": "cached frozen-R1 mixture logits pooled in the oracle event interval",
        },
        "summary": summary,
        "deltas": {
            "isolated_minus_mixture_top1_↑": summary["isolated"]["top1_accuracy_↑"] - summary["mixture"]["top1_accuracy_↑"],
            "isolated_minus_mixture_top5_↑": summary["isolated"]["top5_accuracy_↑"] - summary["mixture"]["top5_accuracy_↑"],
            "isolated_minus_aligned_top1_absolute_↓": abs(summary["isolated"]["top1_accuracy_↑"] - summary["aligned"]["top1_accuracy_↑"]),
        },
        "decision": decision,
        "grouped": {
            "cleanliness_tier": grouped_summaries(items, "cleanliness_tier"),
            "duration_equalized": grouped_summaries(items, "duration_equalized"),
            "recipe_kind": grouped_summaries(items, "recipe_kind"),
            "requested_overlap_fraction": grouped_summaries(items, "requested_overlap_fraction"),
            "maximum_concurrency": grouped_summaries(items, "maximum_concurrency"),
        },
        "worst_isolated_classes": worst_classes,
        "per_class": class_details,
        "top_confusions": {
            condition: top_confusions(items, condition)
            for condition in ("isolated", "aligned", "mixture")
        },
        "metric_direction_legend": {"↑": "higher is better", "↓": "lower is better"},
    }
    atomic_text(
        output_dir / "items.jsonl",
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in items),
    )
    atomic_text(
        output_dir / "report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {"summary": summary, "decision": decision, "report": str(output_dir / "report.json")},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
