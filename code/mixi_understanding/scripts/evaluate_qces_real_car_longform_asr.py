#!/usr/bin/env python3
"""Compare ASR on the full long-form car audio across enhancement methods."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def _wer(reference: str, hypothesis: str) -> float:
    left, right = _words(reference), _words(hypothesis)
    previous = list(range(len(right) + 1))
    for index, source in enumerate(left, 1):
        current = [index]
        for column, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (source != target)))
        previous = current
    return previous[-1] / max(len(left), 1)


def _load(path: Path) -> tuple[np.ndarray, int]:
    waveform, rate = sf.read(path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    return np.asarray(waveform, dtype=np.float32), int(rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/qces_real_car_longform_v1")
    parser.add_argument("--audiosep-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_real_car_longform_audiosep_v1")
    parser.add_argument("--frcrn-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_real_car_longform_frcrn_v1")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_real_car_longform_asr_v1")
    parser.add_argument("--asr-model", type=Path, default=Path.home() / ".cache/huggingface/hub/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")
    modes = ("mixture", "audiosep", "frcrn", "clean")
    audio_items: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for scene in scenes:
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        paths = {
            "mixture": _resolve(str(scene["mixture_path"])),
            "audiosep": args.audiosep_dir.resolve() / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac",
            "frcrn": args.frcrn_dir.resolve() / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac",
            "clean": _resolve(str(speech["stem_path"])),
        }
        for mode in modes:
            waveform, rate = _load(paths[mode])
            audio_items.append({"array": waveform, "sampling_rate": rate})
            metadata.append({"scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"], "mode": mode, "reference": speech["transcript"], "audio_path": _portable(paths[mode])})
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(str(args.asr_model), local_files_only=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(str(args.asr_model), local_files_only=True, dtype=dtype)
    if device.type == "cuda":
        model = model.cuda()
    transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device=0 if device.type == "cuda" else -1)
    hypotheses = transcriber(audio_items, batch_size=1, generate_kwargs={"language": "vi", "task": "transcribe", "max_new_tokens": 256})
    rows: list[dict[str, Any]] = []
    for meta, hypothesis in zip(metadata, hypotheses):
        text = str(hypothesis["text"]).strip()
        error = _wer(meta["reference"], text)
        rows.append({**meta, "hypothesis": text, "wer_↓": error, "correct_at_wer_0.25": error <= 0.25})
    summary: dict[str, Any] = {}
    for split in ("val", "test"):
        summary[split] = {}
        for mode in modes:
            chosen = [row for row in rows if row["split"] == split and row["mode"] == mode]
            summary[split][mode] = {"scenes": len(chosen), "mean_wer_↓": float(np.mean([row["wer_↓"] for row in chosen])), "accuracy_at_wer_0.25_↑": float(np.mean([row["correct_at_wer_0.25"] for row in chosen]))}
    item_path = output / "asr_items.jsonl"
    _atomic_text(item_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    receipt = {"format": "qces_real_car_longform_asr_receipt_v1", "complete": True, "asr_model": str(args.asr_model), "full_audio": True, "summary": summary, "asr_items": _portable(item_path)}
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
