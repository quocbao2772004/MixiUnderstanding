#!/usr/bin/env python3
"""Focused Whisper-medium evaluation of the two validation-selected AudioSep prompts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
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
    parser.add_argument("--audiosep-si-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_audiosep_v1")
    parser.add_argument("--audiosep-asr-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_audiosep_human_v1")
    parser.add_argument("--model", default=str(Path.home() / ".cache/huggingface/hub/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_audiosep_whisper_medium_v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, local_files_only=True, dtype=torch.float16).cuda()
    transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                           feature_extractor=processor.feature_extractor, device=0)
    modes = ("audiosep_si_prompt", "audiosep_asr_prompt", "clean_speech_upper_bound")
    audio, meta = [], []
    for scene in scenes:
        speech = next(x for x in scene["events"] if x["label"] == "Speech")
        paths = {
            "audiosep_si_prompt": args.audiosep_si_dir.resolve() / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac",
            "audiosep_asr_prompt": args.audiosep_asr_dir.resolve() / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac",
            "clean_speech_upper_bound": _resolve(speech["stem_path"]),
        }
        for mode, path in paths.items():
            wave, rate = sf.read(path, dtype="float32")
            if mode == "clean_speech_upper_bound":
                start = max(0, int((float(speech["onset_seconds"]) - .15) * rate))
                end = min(len(wave), int((float(speech["offset_seconds"]) + .15) * rate))
                wave = wave[start:end]
            audio.append({"array": _normalize(wave), "sampling_rate": rate})
            meta.append({"scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
                         "mode": mode, "reference": speech["transcript"], "audio_path": _portable(path)})
    hypotheses = transcriber(
        audio, batch_size=1,
        generate_kwargs={"language": "vi", "task": "transcribe", "max_new_tokens": 96,
                         "no_repeat_ngram_size": 3, "repetition_penalty": 1.05},
    )
    rows = []
    for values, hypothesis in zip(meta, hypotheses):
        text = str(hypothesis["text"]).strip()
        error = _wer(values["reference"], text)
        rows.append({**values, "hypothesis": text, "wer_↓": error, "correct_at_wer_0.25": error <= .25})
    summary: dict[str, Any] = {}
    for split in ("val", "test"):
        summary[split] = {}
        for mode in modes:
            selected = [x for x in rows if x["split"] == split and x["mode"] == mode]
            summary[split][mode] = {
                "scenes": len(selected), "accuracy_at_wer_0.25_↑": float(np.mean([x["correct_at_wer_0.25"] for x in selected])),
                "mean_wer_↓": float(np.mean([x["wer_↓"] for x in selected])),
            }
    deployable = modes[:2]
    selected_mode = min(deployable, key=lambda x: (summary["val"][x]["mean_wer_↓"], -summary["val"][x]["accuracy_at_wer_0.25_↑"]))
    for scene in scenes:
        row = next(x for x in rows if x["scene_id"] == scene["scene_id"] and x["mode"] == selected_mode)
        target = output / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_resolve(row["audio_path"]), target)
    item_path = output / "items.jsonl"
    _atomic_text(item_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows))
    receipt = {
        "format": "qces_vietnamese_audiosep_whisper_medium_receipt_v1", "complete": True,
        "model": "openai/whisper-medium multilingual", "decoding": "greedy, max 96 tokens, no-repeat trigram",
        "selection_protocol": "select one global AudioSep prompt by validation WER, then lock for test",
        "selected_mode": selected_mode, "asr": summary, "items": _portable(item_path),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected_mode": selected_mode, "asr": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
