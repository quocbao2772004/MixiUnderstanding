#!/usr/bin/env python3
"""ASR and signal-quality audit for a VieNeu speech source manifest."""

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

from mixi_understanding.scripts.build_qces_speech_event_v1 import (
    PROJECT_ROOT,
    _atomic_text,
    _jsonl,
    _portable,
    _resolve,
)


DEFAULT_ASR = (
    Path.home()
    / ".cache/huggingface/hub/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e"
)


_DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")


def _number_vi(value: int) -> str:
    """Canonical Vietnamese reading for the small numbers used in scenarios."""
    if value < 10:
        return _DIGITS[value]
    if value < 100:
        tens, ones = divmod(value, 10)
        words = ["mười"] if tens == 1 else [_DIGITS[tens], "mươi"]
        if ones:
            words.append("mốt" if ones == 1 and tens > 1 else "lăm" if ones == 5 else _DIGITS[ones])
        return " ".join(words)
    if value < 1000:
        hundreds, remainder = divmod(value, 100)
        words = [_DIGITS[hundreds], "trăm"]
        if remainder:
            if remainder < 10:
                words.append("lẻ")
            words.append(_number_vi(remainder))
        return " ".join(words)
    return str(value)


def _words(value: str) -> list[str]:
    # Whisper commonly emits Arabic digits for spoken Vietnamese numbers.
    value = value.lower()
    value = re.sub(
        r"(\d+)\s*m\b",
        lambda match: f" {_number_vi(int(match.group(1)))} mét ",
        value,
    )
    value = re.sub(r"\d+", lambda match: f" {_number_vi(int(match.group()))} ", value)
    return re.findall(r"\w+", value, flags=re.UNICODE)


def _wer(reference: str, hypothesis: str) -> float:
    left, right = _words(reference), _words(hypothesis)
    if not left:
        return float(bool(right))
    previous = list(range(len(right) + 1))
    for i, source in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (source != target),
                )
            )
        previous = current
    return previous[-1] / len(left)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_vietnamese_vieneu_source_v1",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--asr-model", type=Path, default=DEFAULT_ASR)
    parser.add_argument("--max-wer", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    dataset = args.dataset_dir.resolve()
    output = (args.output_dir or dataset / "audit").resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(dataset / "scenes.jsonl")
    if not scenes:
        raise RuntimeError("Empty scene manifest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.asr_model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.asr_model, local_files_only=True, torch_dtype=dtype
    )
    if device.type == "cuda":
        model = model.cuda()
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if device.type == "cuda" else -1,
    )

    inputs: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for scene in scenes:
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        wave, rate = sf.read(_resolve(speech["stem_path"]), dtype="float32", always_2d=True)
        mono = wave.mean(axis=1, dtype=np.float32)
        inputs.append({"array": mono, "sampling_rate": rate})
        metadata.append(
            {
                "scene_id": scene["scene_id"],
                "split": scene["split"],
                "scenario_category": scene["scenario_category"],
                "voice": speech["tts_voice"],
                "reference": speech["transcript"],
                "clean_path": speech["stem_path"],
                "duration_seconds": len(mono) / rate,
                "peak": float(np.max(np.abs(mono))),
                "clipped_sample_fraction": float(np.mean(np.abs(mono) >= 0.999)),
            }
        )

    hypotheses = asr(
        inputs,
        batch_size=args.batch_size,
        generate_kwargs={"language": "vi", "task": "transcribe"},
    )
    rows = []
    for meta, hypothesis in zip(metadata, hypotheses):
        text = str(hypothesis["text"]).strip()
        error = _wer(meta["reference"], text)
        rows.append(
            {
                **meta,
                "hypothesis": text,
                "wer_↓": error,
                "asr_gate_pass": error <= args.max_wer,
                "signal_gate_pass": meta["clipped_sample_fraction"] == 0.0,
                "accepted": error <= args.max_wer and meta["clipped_sample_fraction"] == 0.0,
            }
        )
    item_path = output / "audit_items.jsonl"
    _atomic_text(item_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows))
    accepted = [row for row in rows if row["accepted"]]
    receipt = {
        "format": "qces_vieneu_asr_audit_receipt_v1",
        "complete": len(rows) == len(scenes),
        "paper_eligible": False,
        "paper_eligibility_reason": "TTS speech is controlled stress-test data, not a real-speech test set",
        "dataset": _portable(dataset),
        "asr_model": str(args.asr_model),
        "asr_model_role": "automatic TTS quality gate; not ground truth",
        "items": len(rows),
        "accepted": len(accepted),
        "acceptance_rate_↑": len(accepted) / len(rows),
        "mean_wer_↓": float(np.mean([row["wer_↓"] for row in rows])),
        "median_wer_↓": float(np.median([row["wer_↓"] for row in rows])),
        "wer_by_split_↓": {
            split: float(np.mean([row["wer_↓"] for row in rows if row["split"] == split]))
            for split in sorted({row["split"] for row in rows})
        },
        "acceptance_by_voice": {
            voice: {
                "items": len(group),
                "accepted": sum(row["accepted"] for row in group),
                "mean_wer_↓": float(np.mean([row["wer_↓"] for row in group])),
            }
            for voice in sorted({row["voice"] for row in rows})
            for group in [[row for row in rows if row["voice"] == voice]]
        },
        "rejected_scene_ids": [row["scene_id"] for row in rows if not row["accepted"]],
        "items_manifest": _portable(item_path),
    }
    _atomic_text(output / "audit_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
