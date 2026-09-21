#!/usr/bin/env python3
"""Measure semantic recognizability as active component duration increases.

The crop for every duration is the deterministic maximum-energy window used
by the overlap-v3 builder.  Crop selection is label independent.  Gold labels
are consulted only after frozen-R1 inference to compute ranks.  The report
selects the smallest duration whose top-1 accuracy is within two absolute
points of the full-active ceiling and whose top-5 is at least 95%.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
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

from mixi_understanding.qces.clean_evidence_scenes import load_source_bank
from mixi_understanding.scripts.audit_qces_full_source_semantic_ceiling_v1 import (
    FIXED_SAMPLES,
    NUM_FRAMES,
    SAMPLE_RATE,
    SourcePair,
    load_pairs,
)
from mixi_understanding.scripts.build_qces_overlap_gold_natural_v3 import (
    HOP_SAMPLES,
    _highest_energy_window,
)
from mixi_understanding.scripts.train_qces_beats_event_rank_v1 import forward_logits
from mixi_understanding.scripts.train_qces_pretrainedsed_detector import (
    load_model,
    load_ontology,
    make_device,
)
from mixi_understanding.scripts.train_qces_qdor_dense import _sha256_file


FORMAT = "qces_semantic_duration_curve_v1"
FULL_KEY = "full_active"


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full188_overlap_gold_natural_v3"
    base = PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-manifest", type=Path, default=data / "event_components_dev.jsonl")
    parser.add_argument("--scene-manifest", type=Path, default=data / "detector_scene_manifest_overlap_dev.jsonl")
    parser.add_argument("--source-bank", type=Path, default=PROJECT_ROOT / "outputs/qces_full191_r1_data_v1/source_bank_accepted.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_188.txt")
    parser.add_argument("--detector-checkpoint", type=Path, default=base / "pretrainedsed_beats_qces_detector.pt")
    parser.add_argument("--output-dir", type=Path, default=base / "semantic_duration_curve_v1")
    parser.add_argument("--duration-seconds", type=float, nargs="+", default=[1.6, 2.4, 3.2, 4.0])
    parser.add_argument("--pretrained-checkpoint", default="BEATs_strong_1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2, help="Source identities per batch; waveforms are multiplied by duration conditions.")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_active(source: Any) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(source.audio_path)
    mono = waveform.float().mean(dim=0)
    if int(sample_rate) != SAMPLE_RATE:
        mono = AF.resample(mono, int(sample_rate), SAMPLE_RATE)
    start = max(0, int(round(float(source.active_onset_seconds) * SAMPLE_RATE)))
    end = min(mono.numel(), int(round(float(source.active_offset_seconds) * SAMPLE_RATE)))
    if end <= start:
        raise ValueError(f"invalid active interval: {source.source_id}")
    return mono[start:end]


def _variant(active: torch.Tensor, frames: int | None) -> tuple[torch.Tensor, int]:
    if frames is not None:
        active, _ = _highest_energy_window(active, int(frames) * HOP_SAMPLES)
    active = active[:FIXED_SAMPLES]
    valid = max(1, int(active.numel()))
    target_frames = min(NUM_FRAMES, int(math.ceil(valid / HOP_SAMPLES)))
    target_samples = target_frames * HOP_SAMPLES
    if active.numel() < target_samples:
        active = F.pad(active, (0, target_samples - active.numel()))
    audible = active[:valid]
    rms = audible.square().mean().sqrt().clamp_min(1e-5)
    active = active / rms * 0.06
    if active.numel() < FIXED_SAMPLES:
        active = F.pad(active, (0, FIXED_SAMPLES - active.numel()))
    return active[:FIXED_SAMPLES], target_frames


class DurationDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[SourcePair], source_by_id: Mapping[str, Any], keys: Sequence[str], frames: Sequence[int | None]) -> None:
        self.rows = list(rows)
        self.source_by_id = source_by_id
        self.keys = list(keys)
        self.frames = list(frames)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source = self.source_by_id[row.source_id]
        active = _load_active(source)
        variants = [_variant(active, frames) for frames in self.frames]
        return {
            "waveforms": torch.stack([item[0] for item in variants]),
            "valid_frames": [item[1] for item in variants],
            "active_seconds": active.numel() / SAMPLE_RATE,
            "row": row,
        }


def collate(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "waveforms": torch.cat([item["waveforms"] for item in items], dim=0),
        "valid_frames": [item["valid_frames"] for item in items],
        "active_seconds": [float(item["active_seconds"]) for item in items],
        "rows": [item["row"] for item in items],
    }


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, keys: Sequence[str], labels: Sequence[str], device: torch.device, *, amp: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    conditions = len(keys)
    processed = 0
    model.eval()
    for batch in loader:
        waveforms = batch["waveforms"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=bool(amp and device.type == "cuda")):
            logits = forward_logits(model, waveforms).float().cpu()
        for source_index, row in enumerate(batch["rows"]):
            item = {
                "source_id": row.source_id,
                "label": row.label,
                "label_id": row.label_id,
                "active_seconds": batch["active_seconds"][source_index],
                "conditions": {},
            }
            for condition_index, key in enumerate(keys):
                flat_index = source_index * conditions + condition_index
                length = int(batch["valid_frames"][source_index][condition_index])
                scores = logits[flat_index, :length].mean(dim=0)
                order = scores.argsort(descending=True)
                rank = int(torch.where(order == row.label_id)[0].item()) + 1
                item["conditions"][key] = {
                    "rank": rank,
                    "predicted_label": labels[int(order[0])],
                }
            result.append(item)
        processed += len(batch["rows"])
        if processed == len(batch["rows"]) or processed % 256 < len(batch["rows"]):
            print(f"duration_curve {processed}/{len(loader.dataset)}", flush=True)
    return result


def summarize(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    ranks = [int(item["conditions"][key]["rank"]) for item in items]
    grouped: dict[str, list[int]] = defaultdict(list)
    for item, rank in zip(items, ranks):
        grouped[str(item["label"])].append(rank)
    per_class = {label: sum(rank == 1 for rank in values) / len(values) for label, values in grouped.items()}
    return {
        "sources": len(ranks),
        "top1_accuracy_↑": sum(rank == 1 for rank in ranks) / len(ranks),
        "top5_accuracy_↑": sum(rank <= 5 for rank in ranks) / len(ranks),
        "top20_accuracy_↑": sum(rank <= 20 for rank in ranks) / len(ranks),
        "macro_top1_accuracy_↑": sum(per_class.values()) / len(per_class),
        "mean_rank_↓": sum(ranks) / len(ranks),
        "per_class_top1": per_class,
    }


def main() -> None:
    args = parse_args()
    if any(value <= 0 or value > 10 for value in args.duration_seconds):
        raise SystemExit("duration candidates must be in (0,10]")
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
    sources = load_source_bank(args.source_bank.resolve(), require_audio_file=True)
    source_by_id = {source.source_id: source for source in sources}
    missing = sorted({row.source_id for row in rows} - set(source_by_id))
    if missing:
        raise ValueError(f"source bank misses dev identities: {missing[:5]}")
    keys = [f"cap_{value:g}s" for value in args.duration_seconds] + [FULL_KEY]
    frames: list[int | None] = [int(round(value / 0.04)) for value in args.duration_seconds] + [None]
    device = make_device(args.device)
    model = load_model(len(labels), args.pretrained_checkpoint, device, unfreeze_last_blocks=0)
    checkpoint = torch.load(args.detector_checkpoint.resolve(), map_location="cpu", weights_only=True)
    if list(checkpoint.get("labels") or []) != labels:
        raise ValueError("detector ontology mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.requires_grad_(False)
    loader = DataLoader(
        DurationDataset(rows, source_by_id, keys, frames),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )
    items = evaluate(model, loader, keys, labels, device, amp=args.amp)
    summary = {key: summarize(items, key) for key in keys}
    full_top1 = float(summary[FULL_KEY]["top1_accuracy_↑"])
    eligible = [
        key
        for key in keys[:-1]
        if float(summary[key]["top1_accuracy_↑"]) >= full_top1 - 0.02
        and float(summary[key]["top5_accuracy_↑"]) >= 0.95
    ]
    selected = eligible[0] if eligible else None
    active = sorted(float(item["active_seconds"]) for item in items)
    quantile = lambda p: active[min(len(active) - 1, int(p * (len(active) - 1)))]
    report = {
        "format": FORMAT,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_eligible": False,
        "claim_boundary": "development duration selection; final model requires a locked identity-disjoint test",
        "answer_label_used_as_model_input": False,
        "selection_policy": "smallest cap with top1 within 0.02 of full-active and top5 >= 0.95",
        "selected_condition": selected,
        "gate_passed": selected is not None,
        "sources": len(items),
        "classes": len({item["label"] for item in items}),
        "active_duration_seconds": {"min": quantile(0), "p50": quantile(0.5), "p90": quantile(0.9), "max": quantile(1)},
        "summary": summary,
        "inputs": {
            "source_bank": str(args.source_bank.resolve()),
            "source_bank_sha256": _sha256_file(args.source_bank.resolve()),
            "component_manifest": str(args.component_manifest.resolve()),
            "component_manifest_sha256": _sha256_file(args.component_manifest.resolve()),
            "detector_checkpoint": str(args.detector_checkpoint.resolve()),
            "detector_checkpoint_sha256": _sha256_file(args.detector_checkpoint.resolve()),
        },
    }
    atomic_json(output_dir / "report.json", report)
    atomic_json(output_dir / "predictions.json", {"format": FORMAT, "items": items})
    print(json.dumps({"summary": {key: {metric: value for metric, value in metrics.items() if metric != "per_class_top1"} for key, metrics in summary.items()}, "selected_condition": selected, "gate_passed": selected is not None}, indent=2), flush=True)


if __name__ == "__main__":
    main()
