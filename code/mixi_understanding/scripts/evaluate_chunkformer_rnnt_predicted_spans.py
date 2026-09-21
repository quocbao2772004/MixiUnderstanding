#!/usr/bin/env python3
"""Evaluate ChunkFormer-RNNT on the deployed BEATs-predicted speech spans."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from chunkformer import ChunkFormerModel

from mixi_understanding.scripts.evaluate_chunkformer_rnnt_fixed_evidence import (
    edit_distance,
    read_jsonl,
    resolve,
    summarize,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="khanhld/chunkformer-rnnt-large-vie")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_realistic_v1",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_combined_honest_zero_shot_v1/predicted_span_asr_predictions.jsonl",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_combined_honest_chunkformer_rnnt_v1",
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    crop_dir = output / "predicted_speech_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    scenes = {row["scene_id"]: row for row in read_jsonl(args.dataset_dir.resolve() / "scenes.jsonl")}
    phowhisper_rows = read_jsonl(args.predictions.resolve())
    metadata: list[dict[str, Any]] = []
    paths: list[str] = []
    for prediction in phowhisper_rows:
        scene = scenes[prediction["scene_id"]]
        speech = next(event for event in scene["events"] if event["event_kind"] == "speech")
        waveform, sample_rate = sf.read(resolve(prediction["mixture_path"]), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        span = prediction.get("predicted_speech_span_seconds")
        if span is None:
            crop = waveform[: max(1, int(0.02 * sample_rate))] * 0.0
        else:
            start = max(0, int(float(span[0]) * sample_rate))
            end = min(len(waveform), int(float(span[1]) * sample_rate))
            crop = waveform[start : max(start + 1, end)]
        crop_path = crop_dir / f"{prediction['scene_id']}.wav"
        sf.write(crop_path, crop, int(sample_rate), subtype="PCM_16")
        paths.append(str(crop_path))
        metadata.append(
            {
                "scene_id": prediction["scene_id"],
                "split": prediction["split"],
                "mixture_path": prediction["mixture_path"],
                "predicted_speech_span_seconds": span,
                "reference": speech["transcript"],
                "phowhisper_hypothesis": prediction["hypothesis"],
            }
        )

    print(f"[DATA] predicted speech crops={len(paths)}", flush=True)
    load_started = time.perf_counter()
    model = ChunkFormerModel.from_pretrained(args.model).to(args.device).eval()
    load_seconds = time.perf_counter() - load_started
    decode_started = time.perf_counter()
    hypotheses = model.batch_decode(
        paths,
        chunk_size=64,
        left_context_size=128,
        right_context_size=128,
        total_batch_duration=300,
    )
    decode_seconds = time.perf_counter() - decode_started
    rows: list[dict[str, Any]] = []
    for meta, hypothesis in zip(metadata, hypotheses):
        text = str(hypothesis).strip()
        edits, reference_words = edit_distance(meta["reference"], text)
        old_edits, _ = edit_distance(meta["reference"], meta["phowhisper_hypothesis"])
        row = {
            **meta,
            "hypothesis": text,
            "edit_distance": edits,
            "reference_words": reference_words,
            "wer_↓": edits / reference_words,
            "phowhisper_wer_↓": old_edits / reference_words,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "scene": row["scene_id"],
                    "split": row["split"],
                    "phowhisper_wer": row["phowhisper_wer_↓"],
                    "chunkformer_wer": row["wer_↓"],
                    "hypothesis": text,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {split: summarize([row for row in rows if row["split"] == split]) for split in ("val", "test")}
    phowhisper_summary: dict[str, Any] = {}
    for split in ("val", "test"):
        subset = [row for row in rows if row["split"] == split]
        edits = sum(round(float(row["phowhisper_wer_↓"]) * int(row["reference_words"])) for row in subset)
        total_words = sum(int(row["reference_words"]) for row in subset)
        phowhisper_summary[split] = {
            "utterances": len(subset),
            "corpus_wer_↓": edits / max(total_words, 1),
            "accuracy_at_wer_0.25_↑": sum(float(row["phowhisper_wer_↓"]) <= 0.25 for row in subset) / max(len(subset), 1),
        }
    comparison = {
        split: {
            "phowhisper_large": phowhisper_summary[split],
            "chunkformer_rnnt": summary[split],
            "absolute_wer_change_↓": summary[split]["corpus_wer_↓"] - phowhisper_summary[split]["corpus_wer_↓"],
        }
        for split in ("val", "test")
    }
    (output / "asr_items.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = {
        "format": "qces_chunkformer_rnnt_predicted_span_eval_v1",
        "complete": True,
        "model": args.model,
        "protocol": "same raw mixture and BEATs val-locked predicted speech spans as deployed PhoWhisper-large branch",
        "runtime": {"device": args.device, "model_load_seconds": load_seconds, "decode_seconds": decode_seconds},
        "summary": summary,
        "comparison": comparison,
        "items": "asr_items.jsonl",
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
