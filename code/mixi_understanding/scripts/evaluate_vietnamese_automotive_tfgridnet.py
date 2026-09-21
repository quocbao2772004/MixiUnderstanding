#!/usr/bin/env python3
"""Evaluate the official MS-SNSD TF-GridNet checkpoint on VieNeu car mixtures.

The released model operates at 8 kHz.  Predictions are written at 16 kHz so
that downstream Vietnamese ASR can consume them without special handling.
"""

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
from espnet2.enh.separator.tfgridnet_separator import TFGridNet

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import (
    PROJECT_ROOT,
    _atomic_text,
    _portable,
    _resolve,
)


MODEL_SAMPLE_RATE = 8_000
OUTPUT_SAMPLE_RATE = 16_000


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(path: Path, rate: int) -> torch.Tensor:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1)).float()
    return AF.resample(wave, source_rate, rate) if source_rate != rate else wave


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


def _model(checkpoint: Path, device: torch.device) -> TFGridNet:
    model = TFGridNet(
        input_dim=257,
        n_srcs=1,
        n_fft=512,
        stride=256,
        window="hann",
        n_imics=1,
        n_layers=4,
        lstm_hidden_units=128,
        attn_n_head=4,
        attn_approx_qk_dim=512,
        emb_dim=32,
        emb_ks=4,
        emb_hs=4,
        activation="prelu",
        eps=1e-5,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    separator_state = {
        key.removeprefix("separator."): value
        for key, value in state.items()
        if key.startswith("separator.")
    }
    if not separator_state:
        separator_state = state
    model.load_state_dict(separator_state, strict=True)
    return model.to(device).eval()


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_stress_v2",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path.home()
        / ".cache/huggingface/hub/models--espnet--ms_snsd_tfgridnet/snapshots/"
        "a203c32f0e5a0df3942e52d69d329ece391953ec/valid.loss.best.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_tfgridnet_zero_shot_v1",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-scenes", type=int, default=0)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scenes = _jsonl(dataset_dir / "scenes.jsonl")
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    model = _model(checkpoint, device)
    rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, 1):
        mixture_8k = _load(_resolve(scene["mixture_path"]), MODEL_SAMPLE_RATE)
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        target_16k = _load(_resolve(speech["stem_path"]), OUTPUT_SAMPLE_RATE)
        mixture_16k = _load(_resolve(scene["mixture_path"]), OUTPUT_SAMPLE_RATE)
        lengths = torch.tensor([mixture_8k.numel()], dtype=torch.long, device=device)
        with torch.inference_mode():
            outputs, _, _ = model(mixture_8k[None].to(device), lengths)
        prediction_8k = outputs[0][0].detach().cpu()
        prediction_16k = AF.resample(prediction_8k, MODEL_SAMPLE_RATE, OUTPUT_SAMPLE_RATE)
        length = min(prediction_16k.numel(), mixture_16k.numel(), target_16k.numel())
        prediction_16k = prediction_16k[:length]
        mixture_16k, target_16k = mixture_16k[:length], target_16k[:length]
        path = output_dir / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, prediction_16k.numpy(), OUTPUT_SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        metrics = _metrics(prediction_16k, mixture_16k, target_16k)
        row = {
            "scene_id": scene["scene_id"],
            "split": scene["split"],
            "difficulty": scene["difficulty"],
            "enhanced_path": _portable(path),
            **metrics,
        }
        rows.append(row)
        print(
            json.dumps(
                {"done": index, "total": len(scenes), "scene": scene["scene_id"], "si_sdri": metrics["si_sdri_db_↑"]}
            ),
            flush=True,
        )

    summary: dict[str, Any] = {}
    for split in ("val", "test"):
        chosen = [row for row in rows if row["split"] == split]
        summary[split] = {
            "scenes": len(chosen),
            "mean_si_sdr_db_↑": _mean(chosen, "si_sdr_db_↑"),
            "mean_si_sdri_db_↑": _mean(chosen, "si_sdri_db_↑"),
            "mean_sd_sdri_db_↑": _mean(chosen, "sd_sdri_db_↑"),
            "mean_l1_↓": _mean(chosen, "l1_↓"),
            "mean_output_to_target_energy_ratio_db_↔": _mean(
                chosen, "output_to_target_energy_ratio_db_↔"
            ),
        }
    item_path = output_dir / "enhancement_items.jsonl"
    _atomic_text(item_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    receipt = {
        "format": "qces_vietnamese_automotive_tfgridnet_receipt_v1",
        "complete": True,
        "protocol": "official frozen MS-SNSD TF-GridNet checkpoint; no QCES samples used for fitting or selection",
        "model_sample_rate": MODEL_SAMPLE_RATE,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "checkpoint": "espnet/ms_snsd_tfgridnet valid.loss.best.pth",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "summary": summary,
        "enhancement_items": _portable(item_path),
    }
    _atomic_text(output_dir / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
