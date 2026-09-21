#!/usr/bin/env python3
"""Generate absolute word timestamps for locked PhoWhisper-large evidence."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from mixi_understanding.scripts.evaluate_phowhisper_large_fixed_evidence import (
    PROJECT_ROOT,
    normalize_audio,
    read_jsonl,
    resolve,
)


def main() -> int:
    dataset = PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_realistic_v1/scenes.jsonl"
    asr_items = PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_phowhisper_large_v1/asr_items.jsonl"
    output_path = PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_phowhisper_large_v1/word_timestamps.jsonl"
    scenes = {row["scene_id"]: row for row in read_jsonl(dataset)}
    rows = [row for row in read_jsonl(asr_items) if row["split"] == "test"]
    audio_items = []
    metadata = []
    for row in rows:
        scene = scenes[row["scene_id"]]
        speech = next(event for event in scene["events"] if event["event_kind"] == "speech")
        waveform, sample_rate = sf.read(resolve(row["audio_path"]), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        start = max(0, int(math.floor((float(speech["onset_seconds"]) - 0.15) * sample_rate)))
        end = min(len(waveform), int(math.ceil((float(speech["offset_seconds"]) + 0.15) * sample_rate)))
        audio_items.append({"array": normalize_audio(waveform[start:end]), "sampling_rate": int(sample_rate)})
        metadata.append({"row": row, "crop_start_seconds": start / sample_rate})

    model_name = "vinai/PhoWhisper-large"
    processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, local_files_only=True, dtype=torch.float16, low_cpu_mem_usage=True
    ).cuda()
    transcriber = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=0,
    )
    results = []
    for index, (audio, meta) in enumerate(zip(audio_items, metadata), 1):
        output = transcriber(
            audio,
            return_timestamps="word",
            generate_kwargs={
                "language": "vi",
                "task": "transcribe",
                "max_new_tokens": 96,
            },
        )
        shift = float(meta["crop_start_seconds"])
        chunks = []
        for chunk in output.get("chunks", []):
            timestamp = chunk.get("timestamp") or (None, None)
            if timestamp[0] is None:
                continue
            chunks.append(
                {
                    "text": str(chunk.get("text", "")).strip(),
                    "start_seconds": shift + float(timestamp[0]),
                    "end_seconds": shift + float(timestamp[1] if timestamp[1] is not None else timestamp[0]),
                }
            )
        row = {
            "scene_id": meta["row"]["scene_id"],
            "split": "test",
            "audio_path": meta["row"]["audio_path"],
            "reference": meta["row"]["reference"],
            "evaluated_hypothesis": meta["row"]["hypothesis"],
            "timestamp_hypothesis": str(output["text"]).strip(),
            "crop_start_seconds": shift,
            "words": chunks,
        }
        results.append(row)
        print(
            json.dumps(
                {"index": index, "scene_id": row["scene_id"], "words": len(chunks), "text": row["timestamp_hypothesis"]},
                ensure_ascii=False,
            ),
            flush=True,
        )
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    print(f"wrote={output_path} rows={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
