#!/usr/bin/env python3
"""Evaluate PhoWhisper-large on the already locked FRCRN+OA speech evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def words(value: str) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def edit_distance(reference: str, hypothesis: str) -> tuple[int, int]:
    left, right = words(reference), words(hypothesis)
    previous = list(range(len(right) + 1))
    for index, source in enumerate(left, 1):
        current = [index]
        for column, target in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (source != target),
                )
            )
        previous = current
    return previous[-1], max(len(left), 1)


def normalize_audio(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-6:
        return value
    return np.clip(value * min(10.0, (10.0 ** (-24.0 / 20.0)) / rms), -1.0, 1.0)


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edits = sum(int(row["edit_distance"]) for row in rows)
    reference_words = sum(int(row["reference_words"]) for row in rows)
    return {
        "utterances": len(rows),
        "corpus_wer_↓": edits / max(reference_words, 1),
        "mean_utterance_wer_↓": float(np.mean([row["wer_↓"] for row in rows])),
        "accuracy_at_wer_0.25_↑": float(np.mean([row["wer_↓"] <= 0.25 for row in rows])),
        "exact_transcript_accuracy_↑": float(np.mean([row["wer_↓"] == 0.0 for row in rows])),
        "total_edits": edits,
        "reference_words": reference_words,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="vinai/PhoWhisper-large",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_realistic_v1",
    )
    parser.add_argument(
        "--source-items",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_frcrn_oa_v1/asr_items.jsonl",
    )
    parser.add_argument("--mode", default="frcrn_oa_beta_0_25")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_phowhisper_large_v1",
    )
    args = parser.parse_args()

    scenes = {row["scene_id"]: row for row in read_jsonl(args.dataset_dir.resolve() / "scenes.jsonl")}
    source_rows = [row for row in read_jsonl(args.source_items.resolve()) if row["mode"] == args.mode]
    audio_items: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for row in source_rows:
        scene = scenes[row["scene_id"]]
        speech = next(event for event in scene["events"] if event["event_kind"] == "speech")
        waveform, sample_rate = sf.read(resolve(row["audio_path"]), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        start = max(0, int(math.floor((float(speech["onset_seconds"]) - 0.15) * sample_rate)))
        end = min(len(waveform), int(math.ceil((float(speech["offset_seconds"]) + 0.15) * sample_rate)))
        audio_items.append(
            {"array": normalize_audio(waveform[start:end]), "sampling_rate": int(sample_rate)}
        )
        metadata.append(
            {
                "scene_id": row["scene_id"],
                "split": row["split"],
                "difficulty": row["difficulty"],
                "mode": row["mode"],
                "audio_path": row["audio_path"],
                "reference": row["reference"],
                "source_whisper_medium_hypothesis": row["hypothesis"],
                "source_whisper_medium_wer_↓": row["wer_↓"],
            }
        )

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model,
        local_files_only=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0 if torch.cuda.is_available() else -1,
    )
    print(f"PhoWhisper-large items={len(audio_items)}", flush=True)
    hypotheses = transcriber(
        audio_items,
        batch_size=args.batch_size,
        generate_kwargs={
            "language": "vi",
            "task": "transcribe",
            "max_new_tokens": 96,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.05,
        },
    )
    rows: list[dict[str, Any]] = []
    for meta, hypothesis in zip(metadata, hypotheses):
        text = str(hypothesis["text"]).strip()
        edits, reference_words = edit_distance(meta["reference"], text)
        row = {
            **meta,
            "hypothesis": text,
            "edit_distance": edits,
            "reference_words": reference_words,
            "wer_↓": edits / reference_words,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "scene": row["scene_id"],
                    "split": row["split"],
                    "wer": row["wer_↓"],
                    "hypothesis": text,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    report = {split: summary([row for row in rows if row["split"] == split]) for split in ("val", "test")}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "asr_items.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = {
        "format": "qces_phowhisper_large_fixed_evidence_eval_v1",
        "complete": True,
        "model": "vinai/PhoWhisper-large",
        "evidence_mode": args.mode,
        "protocol": "same locked FRCRN+OA beta=0.25 waveform and oracle speech window as Whisper-medium",
        "summary": report,
        "items": "asr_items.jsonl",
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
