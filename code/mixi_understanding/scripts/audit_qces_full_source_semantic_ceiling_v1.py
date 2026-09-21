#!/usr/bin/env python3
"""Paired semantic audit of short rendered components and their full sources.

This diagnostic answers one question before any further training: is the
1.6-second render crop discarding semantic context that the frozen R1 BEATs
detector needs?  Each original source occurs once.  The same detector sees:

* ``component``: the best rendered component used by overlap-v3;
* ``full_source``: the corresponding clean source stem, up to ten seconds.

Gold labels are used only after inference for metrics.  Audio selection and
temporal pooling are label independent.  Consequently this report can decide
whether a long-view teacher is justified without introducing a QA oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    load_ontology,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import _sha256_file


FORMAT = "qces_full_source_semantic_ceiling_v1"
SAMPLE_RATE = 16_000
FIXED_SAMPLES = 10 * SAMPLE_RATE
NUM_FRAMES = 250
TOP_KS = (1, 5, 8, 20)


@dataclass(frozen=True)
class SourcePair:
    source_id: str
    source_sha256: str
    label: str
    label_id: int
    component_path: Path
    source_path: Path


def parse_args() -> argparse.Namespace:
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_gold_natural_v3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", type=Path, default=data / "event_components_dev.jsonl")
    parser.add_argument("--scene-manifest", type=Path, default=data / "detector_scene_manifest_overlap_dev.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--output-dir", type=Path, default=base / "full_source_semantic_ceiling_v1")
    parser.add_argument("--pretrained-checkpoint", default="BEATs_strong_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.resolve().open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_pairs(
    component_manifest: Path,
    scene_manifest: Path,
    labels: Sequence[str],
    *,
    max_sources: int = 0,
) -> list[SourcePair]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    source_meta: dict[str, tuple[str, Path, str]] = {}
    for scene in read_jsonl(scene_manifest):
        for event in scene.get("events") or []:
            source_id = str(event.get("source_id") or "")
            label = str(event.get("label") or "")
            source_path = Path(str(event.get("source_path") or "")).resolve()
            source_sha = str(event.get("source_sha256") or "")
            if not source_id or label not in label_to_id or not source_path.is_file():
                raise ValueError(f"invalid source event: {source_id=} {label=} {source_path=}")
            value = (label, source_path, source_sha)
            if source_id in source_meta and source_meta[source_id] != value:
                raise ValueError(f"inconsistent source metadata for {source_id}")
            source_meta[source_id] = value

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(component_manifest):
        grouped[str(row.get("source_id") or "")].append(row)

    result: list[SourcePair] = []
    for source_id in sorted(grouped):
        if source_id not in source_meta:
            raise ValueError(f"component source missing from scene manifest: {source_id}")
        candidates = grouped[source_id]
        best = max(
            candidates,
            key=lambda row: (
                int(row.get("num_component_samples") or 0)
                - int(row.get("tail_zero_padding_samples") or 0),
                int(row.get("num_component_samples") or 0),
                str(row.get("component_sha256") or ""),
            ),
        )
        label, source_path, source_sha = source_meta[source_id]
        if str(best.get("label") or "") != label:
            raise ValueError(f"component/source label mismatch for {source_id}")
        component_path = Path(str(best.get("component_path") or "")).resolve()
        if not component_path.is_file():
            raise FileNotFoundError(component_path)
        result.append(
            SourcePair(
                source_id=source_id,
                source_sha256=source_sha,
                label=label,
                label_id=label_to_id[label],
                component_path=component_path,
                source_path=source_path,
            )
        )
        if max_sources > 0 and len(result) >= max_sources:
            break
    if not result:
        raise ValueError("no paired sources")
    return result


def _load_mono(path: Path) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.float().mean(dim=0)
    if int(sample_rate) != SAMPLE_RATE:
        waveform = AF.resample(waveform, int(sample_rate), SAMPLE_RATE)
    return waveform


def _energy_dense_window(waveform: torch.Tensor) -> torch.Tensor:
    """Select a ten-second window without consulting event identity."""
    if waveform.numel() <= FIXED_SAMPLES:
        return waveform
    hop = SAMPLE_RATE // 2
    energies = []
    starts = range(0, waveform.numel() - FIXED_SAMPLES + 1, hop)
    starts = list(starts)
    final = waveform.numel() - FIXED_SAMPLES
    if not starts or starts[-1] != final:
        starts.append(final)
    for start in starts:
        energies.append(float(waveform[start : start + FIXED_SAMPLES].square().mean()))
    return waveform[starts[max(range(len(starts)), key=energies.__getitem__)] :][:FIXED_SAMPLES]


def _pad(waveform: torch.Tensor) -> tuple[torch.Tensor, int]:
    waveform = _energy_dense_window(waveform)
    valid = min(int(waveform.numel()), FIXED_SAMPLES)
    if waveform.numel() < FIXED_SAMPLES:
        waveform = F.pad(waveform, (0, FIXED_SAMPLES - waveform.numel()))
    return waveform[:FIXED_SAMPLES], valid


class PairDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[SourcePair]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        component, component_valid = _pad(_load_mono(row.component_path))
        source, source_valid = _pad(_load_mono(row.source_path))
        return {
            "component": component,
            "source": source,
            "component_valid": component_valid,
            "source_valid": source_valid,
            "row": row,
        }


def collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "waveforms": torch.cat(
            (
                torch.stack([item["component"] for item in items]),
                torch.stack([item["source"] for item in items]),
            ),
            dim=0,
        ),
        "component_valid": [int(item["component_valid"]) for item in items],
        "source_valid": [int(item["source_valid"]) for item in items],
        "rows": [item["row"] for item in items],
    }


def _pool(logits: torch.Tensor, waveform: torch.Tensor, valid_samples: int) -> dict[str, torch.Tensor]:
    valid_frames = max(1, min(NUM_FRAMES, int(math.ceil(valid_samples / FIXED_SAMPLES * NUM_FRAMES))))
    logits = logits[:valid_frames]
    samples = waveform[:valid_samples]
    boundaries = torch.linspace(0, max(1, valid_samples), valid_frames + 1).round().long()
    energy = torch.stack(
        [
            samples[int(boundaries[i]) : max(int(boundaries[i]) + 1, int(boundaries[i + 1]))]
            .square()
            .mean()
            for i in range(valid_frames)
        ]
    )
    # Absolute silence is excluded.  The relative floor is intentionally low
    # to retain quiet event tails and is independent of class/logits.
    threshold = max(float(energy.max()) * 0.0025, 1e-10)
    active = energy >= threshold
    if int(active.sum()) < 2:
        active = torch.ones_like(active, dtype=torch.bool)
    return {
        "valid_mean": logits.mean(dim=0),
        "energy_active_mean": logits[active].mean(dim=0),
    }


def _rank(scores: torch.Tensor, gold: int) -> tuple[int, int]:
    order = scores.argsort(descending=True)
    rank = int(torch.where(order == int(gold))[0].item()) + 1
    return rank, int(order[0].item())


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    labels: Sequence[str],
    device: torch.device,
    *,
    amp: bool,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    processed = 0
    model.eval()
    for batch in loader:
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            logits = forward_logits(model, waveforms).float().cpu()
        count = len(batch["rows"])
        for index, row in enumerate(batch["rows"]):
            item: dict[str, Any] = {
                **asdict(row),
                "component_duration_seconds": batch["component_valid"][index] / SAMPLE_RATE,
                "source_duration_seconds": batch["source_valid"][index] / SAMPLE_RATE,
                "conditions": {},
            }
            for name, logit_index, valid_key in (
                ("component", index, "component_valid"),
                ("full_source", count + index, "source_valid"),
            ):
                pooled = _pool(
                    logits[logit_index],
                    waveforms[logit_index].detach().float().cpu(),
                    batch[valid_key][index],
                )
                item["conditions"][name] = {}
                for policy, scores in pooled.items():
                    rank, predicted = _rank(scores, row.label_id)
                    item["conditions"][name][policy] = {
                        "rank": rank,
                        "predicted_label": labels[predicted],
                    }
            result.append(item)
        processed += count
        if processed == count or processed % 256 < count:
            print(f"semantic_audit {processed}/{len(loader.dataset)}", flush=True)
    return result


def summarize(items: Sequence[Mapping[str, Any]], condition: str, policy: str) -> dict[str, Any]:
    ranks = [int(item["conditions"][condition][policy]["rank"]) for item in items]
    values: dict[str, Any] = {"sources": len(ranks), "mean_rank_↓": sum(ranks) / len(ranks)}
    for k in TOP_KS:
        values[f"top{k}_accuracy_↑"] = sum(rank <= k for rank in ranks) / len(ranks)
    return values


def per_class(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[str(item["label"])].append(item)
    return {
        label: {
            condition: {
                policy: summarize(rows, condition, policy)
                for policy in ("valid_mean", "energy_active_mean")
            }
            for condition in ("component", "full_source")
        }
        for label, rows in sorted(grouped.items())
    }


def main() -> None:
    args = parse_args()
    for path in (args.component_manifest, args.scene_manifest, args.ontology, args.detector_checkpoint):
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_ontology(args.ontology.resolve())
    rows = load_pairs(
        args.component_manifest.resolve(),
        args.scene_manifest.resolve(),
        labels,
        max_sources=args.max_sources,
    )
    device = make_device(args.device)
    model = load_model(len(labels), args.pretrained_checkpoint, device, unfreeze_last_blocks=0)
    checkpoint = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if list(checkpoint.get("labels") or []) != labels:
        raise ValueError("detector/ontology label mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    loader = DataLoader(
        PairDataset(rows),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    items = evaluate(model, loader, labels, device, amp=args.amp)
    summary = {
        condition: {
            policy: summarize(items, condition, policy)
            for policy in ("valid_mean", "energy_active_mean")
        }
        for condition in ("component", "full_source")
    }
    best_component = max(summary["component"], key=lambda policy: summary["component"][policy]["top1_accuracy_↑"])
    best_source = max(summary["full_source"], key=lambda policy: summary["full_source"][policy]["top1_accuracy_↑"])
    source_gain = (
        summary["full_source"][best_source]["top1_accuracy_↑"]
        - summary["component"][best_component]["top1_accuracy_↑"]
    )
    decision = {
        "best_component_policy": best_component,
        "best_full_source_policy": best_source,
        "full_source_minus_component_top1_↑": source_gain,
        "long_view_teacher_justified": bool(source_gain >= 0.05 and summary["full_source"][best_source]["top1_accuracy_↑"] >= 0.75),
        "next_action": (
            "train_long_view_teacher_and_distill_short_view"
            if source_gain >= 0.05 and summary["full_source"][best_source]["top1_accuracy_↑"] >= 0.75
            else "replace_or_ensemble_semantic_encoder_before_slot_distillation"
        ),
    }
    duration_values = sorted(float(item["source_duration_seconds"]) for item in items)
    quantile = lambda p: duration_values[min(len(duration_values) - 1, int(p * (len(duration_values) - 1)))]
    report = {
        "format": FORMAT,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_eligible": False,
        "claim_boundary": "paired dev diagnostic; checkpoint and policy decisions require a future locked test",
        "answer_label_used_as_model_input": False,
        "sources": len(items),
        "classes": len({item["label"] for item in items}),
        "source_duration_seconds": {"min": quantile(0), "p50": quantile(0.5), "p90": quantile(0.9), "max": quantile(1.0)},
        "inputs": {
            "component_manifest": str(args.component_manifest.resolve()),
            "component_manifest_sha256": _sha256_file(args.component_manifest.resolve()),
            "scene_manifest": str(args.scene_manifest.resolve()),
            "scene_manifest_sha256": _sha256_file(args.scene_manifest.resolve()),
            "detector_checkpoint": str(args.detector_checkpoint.resolve()),
            "detector_checkpoint_sha256": _sha256_file(args.detector_checkpoint.resolve()),
        },
        "summary": summary,
        "decision": decision,
        "per_class": per_class(items),
    }
    atomic_json(output_dir / "report.json", report)
    atomic_json(output_dir / "predictions.json", {"format": FORMAT, "items": items})
    print(json.dumps({"summary": summary, "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
