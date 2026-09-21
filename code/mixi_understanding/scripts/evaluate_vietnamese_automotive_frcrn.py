#!/usr/bin/env python3
"""Run the official ClearerVoice FRCRN_SE_16K model on VieNeu car mixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from clearvoice import ClearVoice

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import (
    PROJECT_ROOT,
    _atomic_text,
    _portable,
    _resolve,
)


SAMPLE_RATE = 16_000


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(path: Path) -> torch.Tensor:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1)).float()
    return AF.resample(wave, source_rate, SAMPLE_RATE) if source_rate != SAMPLE_RATE else wave


def _metrics(prediction: torch.Tensor, mixture: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    length = min(prediction.numel(), mixture.numel(), target.numel())
    prediction, mixture, target = prediction[:length], mixture[:length], target[:length]
    si = float(scale_invariant_sdr(prediction[None], target[None])[0])
    mix_si = float(scale_invariant_sdr(mixture[None], target[None])[0])
    sd = float(scale_dependent_sdr(prediction[None], target[None])[0])
    mix_sd = float(scale_dependent_sdr(mixture[None], target[None])[0])
    return {
        "si_sdr_db_↑": si,
        "mixture_si_sdr_db_↑": mix_si,
        "si_sdri_db_↑": si - mix_si,
        "sd_sdr_db_↑": sd,
        "mixture_sd_sdr_db_↑": mix_sd,
        "sd_sdri_db_↑": sd - mix_sd,
        "l1_↓": float((prediction - target).abs().mean()),
        "output_to_target_energy_ratio_db_↔": 10.0 * math.log10(
            (float(prediction.square().mean()) + 1e-12) / (float(target.square().mean()) + 1e-12)
        ),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_realistic_v1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_frcrn_zero_shot_v1",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/FRCRN_SE_16K/last_best_checkpoint.pt",
    )
    parser.add_argument("--max-scenes", type=int, default=0)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    enhancer = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
    rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        mixture = _load(_resolve(scene["mixture_path"]))
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        target = _load(_resolve(speech["stem_path"]))
        with torch.inference_mode():
            output = enhancer(mixture[None].numpy())
        prediction = torch.from_numpy(np.asarray(output)[0]).float()
        length = min(prediction.numel(), mixture.numel(), target.numel())
        prediction, mixture, target = prediction[:length], mixture[:length], target[:length]
        path = output_dir / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, prediction.numpy(), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        metrics = _metrics(prediction, mixture, target)
        rows.append({
            "scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
            "enhanced_path": _portable(path), **metrics,
        })
        print(json.dumps({"done": index, "total": len(scenes), "scene": scene["scene_id"], "si_sdri": metrics["si_sdri_db_↑"]}), flush=True)

    summary: dict[str, Any] = {}
    for split in ("val", "test"):
        chosen = [row for row in rows if row["split"] == split]
        summary[split] = {
            "scenes": len(chosen),
            "mean_si_sdr_db_↑": _mean(chosen, "si_sdr_db_↑"),
            "mean_si_sdri_db_↑": _mean(chosen, "si_sdri_db_↑"),
            "mean_sd_sdri_db_↑": _mean(chosen, "sd_sdri_db_↑"),
            "mean_l1_↓": _mean(chosen, "l1_↓"),
            "mean_output_to_target_energy_ratio_db_↔": _mean(chosen, "output_to_target_energy_ratio_db_↔"),
        }
    item_path = output_dir / "enhancement_items.jsonl"
    _atomic_text(item_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    checkpoint = args.checkpoint.resolve()
    receipt = {
        "format": "qces_vietnamese_automotive_frcrn_receipt_v1",
        "complete": True,
        "protocol": "official frozen ClearerVoice FRCRN_SE_16K; zero-shot; no QCES fitting or selection",
        "sample_rate": SAMPLE_RATE,
        "checkpoint": "alibabasglab/FRCRN_SE_16K last_best_checkpoint.pt",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "summary": summary,
        "enhancement_items": _portable(item_path),
    }
    _atomic_text(output_dir / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
