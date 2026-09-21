#!/usr/bin/env python3
"""Compare ChunkFormer-RNNT with PhoWhisper on identical locked speech evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from chunkformer import ChunkFormerModel


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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    parser.add_argument("--model", default="khanhld/chunkformer-rnnt-large-vie")
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
    parser.add_argument(
        "--phowhisper-receipt",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_phowhisper_large_v1/receipt.json",
    )
    parser.add_argument("--mode", default="frcrn_oa_beta_0_25")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--left-context-size", type=int, default=128)
    parser.add_argument("--right-context-size", type=int, default=128)
    parser.add_argument("--total-batch-duration", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_chunkformer_rnnt_v1",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    crop_dir = output / "locked_speech_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    scenes = {row["scene_id"]: row for row in read_jsonl(args.dataset_dir.resolve() / "scenes.jsonl")}
    source_rows = [row for row in read_jsonl(args.source_items.resolve()) if row["mode"] == args.mode]
    if args.limit is not None:
        source_rows = source_rows[: max(0, args.limit)]

    metadata: list[dict[str, Any]] = []
    crop_paths: list[str] = []
    for row in source_rows:
        scene = scenes[row["scene_id"]]
        speech = next(event for event in scene["events"] if event["event_kind"] == "speech")
        waveform, sample_rate = sf.read(resolve(row["audio_path"]), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1)
        start = max(0, int(math.floor((float(speech["onset_seconds"]) - 0.15) * sample_rate)))
        end = min(len(waveform), int(math.ceil((float(speech["offset_seconds"]) + 0.15) * sample_rate)))
        crop_path = crop_dir / f"{row['scene_id']}.wav"
        if not crop_path.is_file():
            sf.write(crop_path, normalize_audio(waveform[start:end]), int(sample_rate), subtype="PCM_16")
        crop_paths.append(str(crop_path))
        metadata.append(
            {
                "scene_id": row["scene_id"],
                "split": row["split"],
                "difficulty": row["difficulty"],
                "mode": row["mode"],
                "audio_path": row["audio_path"],
                "crop_path": str(crop_path.relative_to(PROJECT_ROOT)),
                "reference": row["reference"],
            }
        )

    print(f"[DATA] locked evidence crops={len(crop_paths)} mode={args.mode}", flush=True)
    load_started = time.perf_counter()
    model = ChunkFormerModel.from_pretrained(args.model).to(args.device).eval()
    load_seconds = time.perf_counter() - load_started
    print(f"[MODEL] loaded={args.model} device={args.device} seconds={load_seconds:.2f}", flush=True)
    decode_started = time.perf_counter()
    hypotheses = model.batch_decode(
        crop_paths,
        chunk_size=args.chunk_size,
        left_context_size=args.left_context_size,
        right_context_size=args.right_context_size,
        total_batch_duration=args.total_batch_duration,
    )
    decode_seconds = time.perf_counter() - decode_started
    if len(hypotheses) != len(metadata):
        raise RuntimeError(f"decode count mismatch: {len(hypotheses)} != {len(metadata)}")

    rows: list[dict[str, Any]] = []
    for meta, hypothesis in zip(metadata, hypotheses):
        text = str(hypothesis).strip()
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

    summary = {
        split: summarize([row for row in rows if row["split"] == split])
        for split in ("val", "test")
    }
    phowhisper = json.loads(args.phowhisper_receipt.resolve().read_text(encoding="utf-8"))["summary"]
    comparison = {
        split: {
            "phowhisper_large_corpus_wer_↓": float(phowhisper[split]["corpus_wer_↓"]),
            "chunkformer_rnnt_corpus_wer_↓": float(summary[split]["corpus_wer_↓"]),
            "absolute_wer_change_↓": float(summary[split]["corpus_wer_↓"])
            - float(phowhisper[split]["corpus_wer_↓"]),
        }
        for split in ("val", "test")
    }
    (output / "asr_items.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = {
        "format": "qces_chunkformer_rnnt_fixed_evidence_eval_v1",
        "complete": True,
        "model": args.model,
        "evidence_mode": args.mode,
        "protocol": "identical locked FRCRN+OA beta=0.25 waveform and oracle speech window as PhoWhisper-large",
        "decoder": {
            "chunk_size": args.chunk_size,
            "left_context_size": args.left_context_size,
            "right_context_size": args.right_context_size,
            "total_batch_duration": args.total_batch_duration,
        },
        "runtime": {
            "device": args.device,
            "model_load_seconds": load_seconds,
            "decode_seconds": decode_seconds,
            "audio_seconds_per_wall_second": sum(
                sf.info(path).duration for path in crop_paths
            ) / max(decode_seconds, 1e-9),
        },
        "summary": summary,
        "comparison": comparison,
        "items": "asr_items.jsonl",
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "comparison": comparison}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
