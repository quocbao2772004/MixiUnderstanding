#!/usr/bin/env python3
"""Evaluate deployable speech enhancement outputs with constrained Whisper-medium."""

from __future__ import annotations

import argparse
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
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


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


def _normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-6:
        return value
    return np.clip(value * min(10.0, (10 ** (-24 / 20)) / rms), -1.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--enhancement-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_sepformer_finetuned_eval_v1")
    parser.add_argument("--audiosep-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_audiosep_v1")
    parser.add_argument("--model", default=str(Path.home() / ".cache/huggingface/hub/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_enhancement_whisper_medium_v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = [x for x in _jsonl(args.dataset_dir.resolve() / "scenes.jsonl") if x["split"] == "test"]

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, local_files_only=True, dtype=torch.float16).cuda()
    transcriber = pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, device=0,
    )
    modes = ("domain_enhanced_full", "domain_enhanced_oracle_span", "audiosep_full", "clean_upper_bound")
    audio, metadata = [], []
    for scene in scenes:
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        enhanced = args.enhancement_dir.resolve() / "audio/selected/test" / f"{scene['scene_id']}.flac"
        audiosep = args.audiosep_dir.resolve() / "audio/selected/test" / f"{scene['scene_id']}.flac"
        sources = {
            "domain_enhanced_full": (enhanced, False),
            "domain_enhanced_oracle_span": (enhanced, True),
            "audiosep_full": (audiosep, False),
            "clean_upper_bound": (_resolve(speech["stem_path"]), True),
        }
        for mode, (path, crop) in sources.items():
            wave, rate = sf.read(path, dtype="float32")
            if crop:
                start = max(0, int((float(speech["onset_seconds"]) - 0.15) * rate))
                end = min(len(wave), int((float(speech["offset_seconds"]) + 0.15) * rate))
                wave = wave[start:end]
            audio.append({"array": _normalize(wave), "sampling_rate": rate})
            metadata.append({
                "scene_id": scene["scene_id"], "mode": mode, "reference": speech["transcript"],
                "audio_path": _portable(path),
            })
    hypotheses = transcriber(
        audio, batch_size=1,
        generate_kwargs={
            "language": "vi", "task": "transcribe", "max_new_tokens": 96,
            "no_repeat_ngram_size": 3, "repetition_penalty": 1.05,
        },
    )
    rows = []
    for meta, hypothesis in zip(metadata, hypotheses):
        text = str(hypothesis["text"]).strip()
        error = _wer(meta["reference"], text)
        row = {**meta, "hypothesis": text, "wer_↓": error, "correct_at_wer_0.25": error <= 0.25}
        rows.append(row)
        print(json.dumps({"scene": meta["scene_id"], "mode": meta["mode"], "wer": error}, ensure_ascii=False), flush=True)
    summary = {}
    for mode in modes:
        chosen = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            "scenes": len(chosen),
            "accuracy_at_wer_0.25_↑": float(np.mean([row["correct_at_wer_0.25"] for row in chosen])),
            "mean_wer_↓": float(np.mean([row["wer_↓"] for row in chosen])),
        }
    item_path = output / "items.jsonl"
    _atomic_text(item_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    receipt = {
        "format": "qces_vietnamese_enhancement_whisper_medium_receipt_v1",
        "complete": True,
        "model": "openai/whisper-medium multilingual",
        "decoding": "greedy, max 96 tokens, no-repeat trigram, repetition penalty 1.05",
        "test": summary,
        "items": _portable(item_path),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
