#!/usr/bin/env python3
"""Evaluate observation adding (OA) on AudioSep speech outputs.

All ASR candidates use the same oracle speech span.  OA weights are evaluated
on validation, and the single selected weight is then reported on test.  The
test oracle is analysis-only and is never used for selection.
"""

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
import torchaudio.functional as AF
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import (
    PROJECT_ROOT,
    _atomic_text,
    _portable,
    _resolve,
)


BETAS = (0.0, 0.1, 0.25, 0.5, 1.0)
SAMPLE_RATE = 32_000


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(path: Path, rate: int = SAMPLE_RATE) -> torch.Tensor:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1)).float()
    return AF.resample(wave, source_rate, rate) if source_rate != rate else wave


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def _edit_distance(reference: str, hypothesis: str) -> tuple[int, int]:
    left, right = _words(reference), _words(hypothesis)
    previous = list(range(len(right) + 1))
    for i, source in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target)))
        previous = current
    return previous[-1], max(len(left), 1)


def _normalize_for_asr(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-6:
        return value
    return np.clip(value * min(10.0, (10.0 ** (-24.0 / 20.0)) / rms), -1.0, 1.0)


def _speech_bounds(scene: dict[str, Any], rate: int, length: int) -> tuple[int, int]:
    speech = next(event for event in scene["events"] if event["label"] == "Speech")
    start = max(0, int(math.floor((float(speech["onset_seconds"]) - 0.15) * rate)))
    end = min(length, int(math.ceil((float(speech["offset_seconds"]) + 0.15) * rate)))
    return start, end


def _oa(enhanced: torch.Tensor, mixture: torch.Tensor, start: int, end: int, beta: float) -> torch.Tensor:
    length = min(enhanced.numel(), mixture.numel())
    enhanced, mixture = enhanced[:length], mixture[:length]
    start, end = min(start, length), min(end, length)
    enhanced_rms = enhanced[start:end].square().mean().sqrt().clamp_min(1e-8)
    mixture_rms = mixture[start:end].square().mean().sqrt().clamp_min(1e-8)
    aligned_mixture = mixture * (enhanced_rms / mixture_rms)
    return (1.0 - beta) * enhanced + beta * aligned_mixture


def _wave_metrics(prediction: torch.Tensor, mixture: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    length = min(prediction.numel(), mixture.numel(), target.numel())
    prediction, mixture, target = prediction[:length], mixture[:length], target[:length]
    si = float(scale_invariant_sdr(prediction[None], target[None])[0])
    mix_si = float(scale_invariant_sdr(mixture[None], target[None])[0])
    sd = float(scale_dependent_sdr(prediction[None], target[None])[0])
    mix_sd = float(scale_dependent_sdr(mixture[None], target[None])[0])
    return {
        "si_sdr_db_↑": si,
        "si_sdri_db_↑": si - mix_si,
        "sd_sdr_db_↑": sd,
        "sd_sdri_db_↑": sd - mix_sd,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edits = sum(int(row["edit_distance"]) for row in rows)
    words = sum(int(row["reference_words"]) for row in rows)
    return {
        "utterances": len(rows),
        "corpus_wer_↓": edits / max(words, 1),
        "mean_utterance_wer_↓": float(np.mean([row["wer_↓"] for row in rows])),
        "median_utterance_wer_↓": float(np.median([row["wer_↓"] for row in rows])),
        "accuracy_at_wer_0.25_↑": float(np.mean([row["wer_↓"] <= 0.25 for row in rows])),
        "total_edits": edits,
        "reference_words": words,
    }


def _mode(beta: float, enhancer_name: str) -> str:
    if beta == 0.0:
        return f"{enhancer_name}_beta_0"
    if beta == 1.0:
        return "mixture_beta_1"
    return f"{enhancer_name}_oa_beta_{str(beta).replace('.', '_')}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_realistic_v1",
    )
    parser.add_argument(
        "--audiosep-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_realistic_audiosep_v1",
    )
    parser.add_argument(
        "--enhancement-dir",
        type=Path,
        default=None,
        help="Generic enhancement directory with audio/selected/{split}; overrides --audiosep-dir.",
    )
    parser.add_argument("--enhancer-name", default="audiosep")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_audiosep_oa_v1",
    )
    parser.add_argument(
        "--asr-model",
        default=str(
            Path.home()
            / ".cache/huggingface/hub/models--openai--whisper-medium/snapshots/"
            "abdf7c39ab9d0397620ccaea8974cc764cd0953e"
        ),
    )
    parser.add_argument("--asr-batch-size", type=int, default=4)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    enhancement_dir = (args.enhancement_dir or args.audiosep_dir).resolve()
    enhancer_name = re.sub(r"[^a-z0-9]+", "_", args.enhancer_name.lower()).strip("_")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(dataset_dir / "scenes.jsonl")

    audio_items: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    wave_rows: list[dict[str, Any]] = []
    for scene in scenes:
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        mixture = _load(_resolve(scene["mixture_path"]))
        target = _load(_resolve(speech["stem_path"]))
        enhanced = _load(enhancement_dir / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac")
        length = min(mixture.numel(), target.numel(), enhanced.numel())
        mixture, target, enhanced = mixture[:length], target[:length], enhanced[:length]
        start, end = _speech_bounds(scene, SAMPLE_RATE, length)
        for beta in BETAS:
            candidate = _oa(enhanced, mixture, start, end, beta)
            mode = _mode(beta, enhancer_name)
            path = output_dir / "audio" / mode / scene["split"] / f"{scene['scene_id']}.flac"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, candidate.numpy(), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
            metrics = _wave_metrics(candidate, mixture, target)
            wave_rows.append({
                "scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
                "beta": beta, "mode": mode, "audio_path": _portable(path), **metrics,
            })
            cropped = candidate[start:end].numpy()
            audio_items.append({"array": _normalize_for_asr(cropped), "sampling_rate": SAMPLE_RATE})
            metadata.append({
                "scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
                "measured_snr_db": scene["measured_speech_to_all_interference_snr_db"],
                "beta": beta, "mode": mode, "reference": speech["transcript"], "audio_path": _portable(path),
            })
        clean_crop = target[start:end].numpy()
        audio_items.append({"array": _normalize_for_asr(clean_crop), "sampling_rate": SAMPLE_RATE})
        metadata.append({
            "scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
            "measured_snr_db": scene["measured_speech_to_all_interference_snr_db"],
            "beta": None, "mode": "clean_upper_bound", "reference": speech["transcript"],
            "audio_path": speech["stem_path"],
        })

    processor = AutoProcessor.from_pretrained(args.asr_model, local_files_only=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.asr_model, local_files_only=True, dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    transcriber = pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, device=0 if torch.cuda.is_available() else -1,
    )
    print(f"Whisper-medium items={len(audio_items)}", flush=True)
    hypotheses = transcriber(
        audio_items,
        batch_size=args.asr_batch_size,
        generate_kwargs={
            "language": "vi", "task": "transcribe", "max_new_tokens": 96,
            "no_repeat_ngram_size": 3, "repetition_penalty": 1.05,
        },
    )
    asr_rows: list[dict[str, Any]] = []
    for meta, hypothesis in zip(metadata, hypotheses):
        text = str(hypothesis["text"]).strip()
        edits, words = _edit_distance(meta["reference"], text)
        row = {
            **meta, "hypothesis": text, "edit_distance": edits, "reference_words": words,
            "wer_↓": edits / words,
        }
        asr_rows.append(row)
        print(json.dumps({"scene": meta["scene_id"], "mode": meta["mode"], "wer": row["wer_↓"]}), flush=True)

    by_split_mode: dict[str, dict[str, Any]] = {}
    all_modes = [_mode(beta, enhancer_name) for beta in BETAS] + ["clean_upper_bound"]
    for split in ("val", "test"):
        by_split_mode[split] = {}
        for mode in all_modes:
            chosen = [row for row in asr_rows if row["split"] == split and row["mode"] == mode]
            by_split_mode[split][mode] = _summary(chosen)

    candidate_modes = [_mode(beta, enhancer_name) for beta in BETAS]
    selected_mode = min(candidate_modes, key=lambda mode: by_split_mode["val"][mode]["corpus_wer_↓"])
    selected_beta = next(beta for beta in BETAS if _mode(beta, enhancer_name) == selected_mode)

    test_scene_ids = sorted({row["scene_id"] for row in asr_rows if row["split"] == "test"})
    oracle_rows = []
    transitions = {"better": 0, "same": 0, "worse": 0}
    for scene_id in test_scene_ids:
        candidates = [
            row for row in asr_rows
            if row["split"] == "test" and row["scene_id"] == scene_id and row["mode"] in candidate_modes
        ]
        oracle = min(candidates, key=lambda row: (row["edit_distance"], row["beta"]))
        oracle_rows.append(oracle)
        mixture = next(row for row in candidates if row["mode"] == "mixture_beta_1")
        selected = next(row for row in candidates if row["mode"] == selected_mode)
        delta = selected["edit_distance"] - mixture["edit_distance"]
        transitions["better" if delta < 0 else "worse" if delta > 0 else "same"] += 1

    per_difficulty: dict[str, Any] = {}
    for difficulty in sorted({row["difficulty"] for row in asr_rows if row["split"] == "test"}):
        per_difficulty[difficulty] = {}
        for mode in ("mixture_beta_1", selected_mode):
            chosen = [
                row for row in asr_rows
                if row["split"] == "test" and row["difficulty"] == difficulty and row["mode"] == mode
            ]
            per_difficulty[difficulty][mode] = _summary(chosen)

    asr_path = output_dir / "asr_items.jsonl"
    wave_path = output_dir / "waveform_items.jsonl"
    _atomic_text(asr_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in asr_rows))
    _atomic_text(wave_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in wave_rows))
    receipt = {
        "format": "qces_vietnamese_audiosep_observation_adding_receipt_v1",
        "complete": True,
        "enhancer": enhancer_name,
        "enhancement_dir": _portable(enhancement_dir),
        "protocol": "all candidates use identical oracle speech span and Whisper decoding; beta selected only on val corpus WER",
        "betas": list(BETAS),
        "oa_definition": "RMS-align mixture to enhanced inside speech span, then (1-beta)*enhanced + beta*mixture",
        "asr_model": "openai/whisper-medium multilingual",
        "summary": by_split_mode,
        "selected_on_validation": {"mode": selected_mode, "beta": selected_beta},
        "locked_test_result": by_split_mode["test"][selected_mode],
        "test_mixture_result": by_split_mode["test"]["mixture_beta_1"],
        "test_oracle_candidate_result_analysis_only": _summary(oracle_rows),
        "selected_vs_mixture_test_transitions": transitions,
        "test_by_difficulty": per_difficulty,
        "asr_items": _portable(asr_path),
        "waveform_items": _portable(wave_path),
    }
    _atomic_text(output_dir / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
