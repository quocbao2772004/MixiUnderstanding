#!/usr/bin/env python3
"""Generate and evaluate two-pass speech-enhancement cascades.

The three stages intentionally run in separate environments:

* ``frcrn``: ClearerVoice environment; creates FRCRN→FRCRN and AudioSep→FRCRN.
* ``audiosep``: project environment; creates FRCRN→AudioSep.
* ``evaluate``: project environment; evaluates every cascade with one Whisper run.

For each two-pass family, observation adding uses the first-pass output rather
than the original mixture.  Beta is selected on validation corpus WER and then
locked for test.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import (
    PROJECT_ROOT,
    _atomic_text,
    _portable,
    _resolve,
)


SAMPLE_RATE = 32_000
CASCADE_BETAS = (0.0, 0.1, 0.25, 0.5)
FAMILIES = {
    "frcrn_to_frcrn": "frcrn1",
    "audiosep_to_frcrn": "audiosep1",
    "frcrn_to_audiosep": "frcrn1",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(path: Path, rate: int = SAMPLE_RATE) -> torch.Tensor:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1)).float()
    return AF.resample(wave, source_rate, rate) if source_rate != rate else wave


def _write(path: Path, wave: torch.Tensor, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave.detach().cpu().numpy(), rate, format="FLAC", subtype="PCM_16")


def _speech_bounds(scene: dict[str, Any], rate: int, length: int) -> tuple[int, int]:
    speech = next(event for event in scene["events"] if event["label"] == "Speech")
    start = max(0, int(math.floor((float(speech["onset_seconds"]) - 0.15) * rate)))
    end = min(length, int(math.ceil((float(speech["offset_seconds"]) + 0.15) * rate)))
    return start, end


def _oa(second: torch.Tensor, first: torch.Tensor, start: int, end: int, beta: float) -> torch.Tensor:
    length = min(second.numel(), first.numel())
    second, first = second[:length], first[:length]
    start, end = min(start, length), min(end, length)
    second_rms = second[start:end].square().mean().sqrt().clamp_min(1e-8)
    first_rms = first[start:end].square().mean().sqrt().clamp_min(1e-8)
    aligned_first = first * (second_rms / first_rms)
    return (1.0 - beta) * second + beta * aligned_first


def _mode(family: str, beta: float) -> str:
    if beta == 0:
        return f"{family}_beta_0"
    return f"{family}_oa_firstpass_beta_{str(beta).replace('.', '_')}"


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


def _normalize_for_asr(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-6:
        return value
    return np.clip(value * min(10.0, (10.0 ** (-24.0 / 20.0)) / rms), -1.0, 1.0)


def _words(value: str) -> list[str]:
    import re

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


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    edits = sum(int(row["edit_distance"]) for row in rows)
    words = sum(int(row["reference_words"]) for row in rows)
    return {
        "utterances": len(rows),
        "corpus_wer_↓": edits / max(words, 1),
        "mean_utterance_wer_↓": float(np.mean([row["wer_↓"] for row in rows])),
        "accuracy_at_wer_0.25_↑": float(np.mean([row["wer_↓"] <= 0.25 for row in rows])),
        "total_edits": edits,
        "reference_words": words,
    }


def _input_path(directory: Path, scene: dict[str, Any]) -> Path:
    return directory / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac"


def generate_frcrn(args: argparse.Namespace, scenes: list[dict[str, Any]]) -> None:
    from clearvoice import ClearVoice

    enhancer = ClearVoice(task="speech_enhancement", model_names=["FRCRN_SE_16K"])
    inputs = {
        "frcrn_to_frcrn": args.frcrn_dir.resolve(),
        "audiosep_to_frcrn": args.audiosep_dir.resolve(),
    }
    total = len(scenes) * len(inputs)
    done = 0
    for family, input_dir in inputs.items():
        for scene in scenes:
            done += 1
            source = _load(_input_path(input_dir, scene), rate=16_000)
            with torch.inference_mode():
                result = enhancer(source[None].numpy())
            prediction = torch.from_numpy(np.asarray(result)[0]).float()
            path = args.output_dir.resolve() / "audio/raw" / family / scene["split"] / f"{scene['scene_id']}.flac"
            _write(path, prediction, 16_000)
            print(json.dumps({"stage": "frcrn", "done": done, "total": total, "family": family, "scene": scene["scene_id"]}), flush=True)
    _atomic_text(args.output_dir.resolve() / "frcrn_stage.complete", "complete\n")


def generate_audiosep(args: argparse.Namespace, scenes: list[dict[str, Any]]) -> None:
    from mixi_understanding.scripts.evaluate_audiosep_baselines import encode_prompts, _load_separator

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    prompt = "a person speaking"
    embedding = encode_prompts(
        args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(), [prompt], batch_size=1
    )[prompt][None].to(device)
    separator = _load_separator(args, device)
    for index, scene in enumerate(scenes, 1):
        source = _load(_input_path(args.frcrn_dir.resolve(), scene), rate=SAMPLE_RATE).to(device)
        with torch.inference_mode():
            prediction = separator({"mixture": source[None, None], "condition": embedding})["waveform"][0, 0]
        path = args.output_dir.resolve() / "audio/raw/frcrn_to_audiosep" / scene["split"] / f"{scene['scene_id']}.flac"
        _write(path, prediction, SAMPLE_RATE)
        print(json.dumps({"stage": "audiosep", "done": index, "total": len(scenes), "scene": scene["scene_id"]}), flush=True)
    del separator, embedding
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _atomic_text(args.output_dir.resolve() / "audiosep_stage.complete", "complete\n")


def evaluate(args: argparse.Namespace, scenes: list[dict[str, Any]]) -> None:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    output_dir = args.output_dir.resolve()
    audio_items: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    wave_rows: list[dict[str, Any]] = []
    baseline_names = (
        "mixture",
        "audiosep1",
        "audiosep1_oa_mixture_beta_0_25",
        "frcrn1",
        "frcrn1_oa_mixture_beta_0_25",
        "clean_upper_bound",
    )
    all_modes = list(baseline_names)
    for family in FAMILIES:
        all_modes.extend(_mode(family, beta) for beta in CASCADE_BETAS)

    for scene_index, scene in enumerate(scenes, 1):
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        mixture = _load(_resolve(scene["mixture_path"]))
        target = _load(_resolve(speech["stem_path"]))
        audiosep1 = _load(_input_path(args.audiosep_dir.resolve(), scene))
        frcrn1 = _load(_input_path(args.frcrn_dir.resolve(), scene))
        common_length = min(mixture.numel(), target.numel(), audiosep1.numel(), frcrn1.numel())
        mixture, target = mixture[:common_length], target[:common_length]
        audiosep1, frcrn1 = audiosep1[:common_length], frcrn1[:common_length]
        start, end = _speech_bounds(scene, SAMPLE_RATE, common_length)
        candidates: dict[str, torch.Tensor] = {
            "mixture": mixture,
            "audiosep1": audiosep1,
            "audiosep1_oa_mixture_beta_0_25": _oa(audiosep1, mixture, start, end, 0.25),
            "frcrn1": frcrn1,
            "frcrn1_oa_mixture_beta_0_25": _oa(frcrn1, mixture, start, end, 0.25),
            "clean_upper_bound": target,
        }
        first_passes = {"audiosep1": audiosep1, "frcrn1": frcrn1}
        for family, first_name in FAMILIES.items():
            second_path = output_dir / "audio/raw" / family / scene["split"] / f"{scene['scene_id']}.flac"
            second = _load(second_path)[:common_length]
            first = first_passes[first_name]
            length = min(second.numel(), first.numel(), common_length)
            second, first = second[:length], first[:length]
            for beta in CASCADE_BETAS:
                candidate = _oa(second, first, start, end, beta)
                candidates[_mode(family, beta)] = candidate

        for mode in all_modes:
            candidate = candidates[mode]
            path = output_dir / "audio/evaluated" / mode / scene["split"] / f"{scene['scene_id']}.flac"
            _write(path, candidate, SAMPLE_RATE)
            metrics = _wave_metrics(candidate, mixture, target)
            wave_rows.append({
                "scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
                "mode": mode, "audio_path": _portable(path), **metrics,
            })
            crop = candidate[start : min(end, candidate.numel())].numpy()
            audio_items.append({"array": _normalize_for_asr(crop), "sampling_rate": SAMPLE_RATE})
            metadata.append({
                "scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
                "mode": mode, "reference": speech["transcript"], "audio_path": _portable(path),
            })
        print(json.dumps({"stage": "prepare_eval", "done": scene_index, "total": len(scenes)}), flush=True)

    processor = AutoProcessor.from_pretrained(args.asr_model, local_files_only=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.asr_model, local_files_only=True, dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    transcriber = pipeline(
        "automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor, device=0 if torch.cuda.is_available() else -1,
    )
    print(json.dumps({"stage": "whisper", "items": len(audio_items)}), flush=True)
    hypotheses = transcriber(
        audio_items,
        batch_size=args.asr_batch_size,
        generate_kwargs={
            "language": "vi", "task": "transcribe", "max_new_tokens": 96,
            "no_repeat_ngram_size": 3, "repetition_penalty": 1.05,
        },
    )
    asr_rows: list[dict[str, Any]] = []
    for index, (meta, hypothesis) in enumerate(zip(metadata, hypotheses), 1):
        text = str(hypothesis["text"]).strip()
        edits, words = _edit_distance(meta["reference"], text)
        row = {**meta, "hypothesis": text, "edit_distance": edits, "reference_words": words, "wer_↓": edits / words}
        asr_rows.append(row)
        if index % 24 == 0 or index == len(metadata):
            print(json.dumps({"stage": "whisper", "done": index, "total": len(metadata)}), flush=True)

    summary: dict[str, dict[str, Any]] = {}
    for split in ("val", "test"):
        summary[split] = {}
        for mode in all_modes:
            summary[split][mode] = _summary([row for row in asr_rows if row["split"] == split and row["mode"] == mode])

    selected: dict[str, Any] = {}
    for family in FAMILIES:
        family_modes = [_mode(family, beta) for beta in CASCADE_BETAS]
        selected_mode = min(family_modes, key=lambda mode: summary["val"][mode]["corpus_wer_↓"])
        selected[family] = {
            "mode": selected_mode,
            "beta": next(beta for beta in CASCADE_BETAS if _mode(family, beta) == selected_mode),
            "validation": summary["val"][selected_mode],
            "locked_test": summary["test"][selected_mode],
        }

    test_modes = ["mixture", "audiosep1_oa_mixture_beta_0_25", "frcrn1_oa_mixture_beta_0_25"]
    test_modes.extend(value["mode"] for value in selected.values())
    best_mode = min(test_modes, key=lambda mode: summary["test"][mode]["corpus_wer_↓"])
    transitions: dict[str, dict[str, int]] = {}
    baseline_mode = "frcrn1_oa_mixture_beta_0_25"
    for family, selection in selected.items():
        counts = {"better": 0, "same": 0, "worse": 0}
        for scene in (row for row in scenes if row["split"] == "test"):
            base = next(row for row in asr_rows if row["scene_id"] == scene["scene_id"] and row["mode"] == baseline_mode)
            candidate = next(row for row in asr_rows if row["scene_id"] == scene["scene_id"] and row["mode"] == selection["mode"])
            delta = candidate["edit_distance"] - base["edit_distance"]
            counts["better" if delta < 0 else "worse" if delta > 0 else "same"] += 1
        transitions[family] = counts

    asr_path = output_dir / "asr_items.jsonl"
    wave_path = output_dir / "waveform_items.jsonl"
    _atomic_text(asr_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in asr_rows))
    _atomic_text(wave_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in wave_rows))
    receipt = {
        "format": "qces_vietnamese_automotive_cascade_receipt_v1",
        "complete": True,
        "protocol": "cascade beta selected on validation corpus WER; locked once on test; identical oracle speech spans and Whisper decoding",
        "cascade_betas": list(CASCADE_BETAS),
        "cascade_oa_observation": "RMS-aligned first-pass output, never the original mixture",
        "summary": summary,
        "selected_on_validation": selected,
        "selected_vs_best_single_frcrn_test_transitions": transitions,
        "best_test_mode_analysis_only": {"mode": best_mode, **summary["test"][best_mode]},
        "asr_items": _portable(asr_path),
        "waveform_items": _portable(wave_path),
    }
    _atomic_text(output_dir / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "selected_on_validation": selected,
        "test_baselines": {mode: summary["test"][mode] for mode in test_modes[:3]},
        "transitions": transitions,
        "best_test_analysis_only": receipt["best_test_mode_analysis_only"],
    }, ensure_ascii=False, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("frcrn", "audiosep", "evaluate"), required=True)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=PROJECT_ROOT / "data/qces_vietnamese_automotive_vieneu_realistic_v1",
    )
    parser.add_argument(
        "--audiosep-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_realistic_audiosep_v1",
    )
    parser.add_argument(
        "--frcrn-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_frcrn_zero_shot_v1",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_vieneu_cascade_v1",
    )
    parser.add_argument("--audiosep-root", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep")
    parser.add_argument(
        "--audiosep-config", type=Path,
        default=PROJECT_ROOT / "code/baseline/audiosep/config/audiosep_base.yaml",
    )
    parser.add_argument(
        "--audiosep-checkpoint", type=Path,
        default=PROJECT_ROOT / "code/baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--asr-model",
        default=str(
            Path.home() / ".cache/huggingface/hub/models--openai--whisper-medium/snapshots/"
            "abdf7c39ab9d0397620ccaea8974cc764cd0953e"
        ),
    )
    parser.add_argument("--asr-batch-size", type=int, default=4)
    parser.add_argument("--max-scenes", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")
    if args.max_scenes:
        scenes = scenes[: args.max_scenes]
    if args.stage == "frcrn":
        generate_frcrn(args, scenes)
    elif args.stage == "audiosep":
        generate_audiosep(args, scenes)
    else:
        evaluate(args, scenes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
