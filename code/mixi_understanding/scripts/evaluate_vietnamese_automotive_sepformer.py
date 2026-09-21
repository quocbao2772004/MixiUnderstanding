#!/usr/bin/env python3
"""Evaluate a frozen SpeechBrain SepFormer speech enhancer on automotive noise."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from speechbrain.inference.separation import SepformerSeparation
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import (
    PROJECT_ROOT,
    _atomic_text,
    _portable,
    _resolve,
)


SAMPLE_RATE = 16_000


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load(path: Path, rate: int = SAMPLE_RATE) -> torch.Tensor:
    value, source_rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(value.mean(axis=1)).float()
    return AF.resample(wave, source_rate, rate) if source_rate != rate else wave


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def _wer(reference: str, hypothesis: str) -> float:
    left, right = _words(reference), _words(hypothesis)
    previous = list(range(len(right) + 1))
    for i, source in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target)))
        previous = current
    return previous[-1] / max(len(left), 1)


def _asr_audio(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-6:
        return value
    gain = min(10.0, (10.0 ** (-24.0 / 20.0)) / rms)
    return np.clip(value * gain, -1.0, 1.0)


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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_sepformer_v1")
    parser.add_argument("--source", default="speechbrain/sepformer-wham16k-enhancement")
    parser.add_argument("--savedir", type=Path, default=PROJECT_ROOT / "outputs/_models/sepformer-wham16k-enhancement")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--asr-model", default=str(Path.home() / ".cache/huggingface/hub/models--vinai--PhoWhisper-tiny/snapshots/cc51d32be916efebde04ff549854fa1741cb5c02"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-scenes", type=int, default=0)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]

    device = args.device if torch.cuda.is_available() else "cpu"
    enhancer = SepformerSeparation.from_hparams(
        source=args.source,
        savedir=str(args.savedir.resolve()),
        run_opts={"device": device},
    )
    checkpoint_epoch = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
        enhancer.mods.load_state_dict(checkpoint["mods"], strict=True)
        checkpoint_epoch = int(checkpoint["epoch"])
    rows: list[dict[str, Any]] = []
    for scene in scenes:
        mixture = _load(_resolve(scene["mixture_path"]))
        speech = next(x for x in scene["events"] if x["label"] == "Speech")
        target = _load(_resolve(speech["stem_path"]))
        with torch.inference_mode():
            prediction = enhancer.separate_batch(mixture[None].to(device))[0, :, 0].detach().cpu()
        path = output / "audio" / scene["split"] / f"{scene['scene_id']}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, prediction.numpy(), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        metrics = _metrics(prediction, mixture, target)
        row = {
            "scene_id": scene["scene_id"],
            "split": scene["split"],
            "difficulty": scene["difficulty"],
            "enhanced_path": _portable(path),
            **metrics,
        }
        rows.append(row)
        print(json.dumps({"scene": scene["scene_id"], "si_sdri": metrics["si_sdri_db_↑"]}), flush=True)

    del enhancer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(args.asr_model, local_files_only=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.asr_model, local_files_only=True, dtype=dtype)
    if torch.cuda.is_available():
        asr_model = asr_model.cuda()
    asr = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if torch.cuda.is_available() else -1,
    )
    audio, metadata = [], []
    for scene in scenes:
        speech = next(x for x in scene["events"] if x["label"] == "Speech")
        path = output / "audio" / scene["split"] / f"{scene['scene_id']}.flac"
        wave, rate = sf.read(path, dtype="float32")
        onset, offset = float(speech["onset_seconds"]), float(speech["offset_seconds"])
        for mode, crop in (("sepformer_full", False), ("sepformer_oracle_span", True)):
            value = wave
            if crop:
                start = max(0, int(math.floor((onset - 0.15) * rate)))
                end = min(len(value), int(math.ceil((offset + 0.15) * rate)))
                value = value[start:end]
            audio.append({"array": _asr_audio(value), "sampling_rate": rate})
            metadata.append({
                "scene_id": scene["scene_id"], "split": scene["split"], "mode": mode,
                "reference": speech["transcript"], "audio_path": _portable(path),
            })
    hypotheses = asr(audio, batch_size=4, generate_kwargs={"language": "vi", "task": "transcribe"})
    asr_rows = []
    for meta, hypothesis in zip(metadata, hypotheses):
        text = str(hypothesis["text"]).strip()
        error = _wer(meta["reference"], text)
        asr_rows.append({**meta, "hypothesis": text, "wer_↓": error, "correct_at_wer_0.25": error <= 0.25})

    summary: dict[str, Any] = {}
    for split in ("val", "test"):
        split_rows = [x for x in rows if x["split"] == split]
        summary[split] = {
            "separator": {
                "scenes": len(split_rows),
                "mean_si_sdr_db_↑": float(np.mean([x["si_sdr_db_↑"] for x in split_rows])),
                "mean_si_sdri_db_↑": float(np.mean([x["si_sdri_db_↑"] for x in split_rows])),
                "mean_sd_sdri_db_↑": float(np.mean([x["sd_sdri_db_↑"] for x in split_rows])),
            },
            "asr": {},
        }
        for mode in ("sepformer_full", "sepformer_oracle_span"):
            chosen = [x for x in asr_rows if x["split"] == split and x["mode"] == mode]
            summary[split]["asr"][mode] = {
                "scenes": len(chosen),
                "accuracy_at_wer_0.25_↑": float(np.mean([x["correct_at_wer_0.25"] for x in chosen])),
                "mean_wer_↓": float(np.mean([x["wer_↓"] for x in chosen])),
            }

    item_path, asr_path = output / "enhancement_items.jsonl", output / "asr_items.jsonl"
    _atomic_text(item_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows))
    _atomic_text(asr_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in asr_rows))
    receipt = {
        "format": "qces_vietnamese_automotive_sepformer_receipt_v1",
        "complete": True,
        "enhancer": args.source,
        "enhancer_frozen": True,
        "domain_adapted_checkpoint": _portable(args.checkpoint.resolve()) if args.checkpoint else None,
        "checkpoint_epoch": checkpoint_epoch,
        "summary": summary,
        "enhancement_items": _portable(item_path),
        "asr_items": _portable(asr_path),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    selected = output / "audio/selected"
    for scene in scenes:
        source = output / "audio" / scene["split"] / f"{scene['scene_id']}.flac"
        target = selected / scene["split"] / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
