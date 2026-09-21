#!/usr/bin/env python3
"""Extract Vietnamese speech from severe automotive mixtures with frozen AudioSep."""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from mixi_understanding.qces.metrics import scale_dependent_sdr, scale_invariant_sdr
from mixi_understanding.scripts.evaluate_audiosep_baselines import encode_prompts, _load_separator
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve


PROMPTS = (
    "speech",
    "human speech",
    "a person speaking",
    "a person speaking Vietnamese",
    "a Vietnamese person talking inside a noisy car",
)
SAMPLE_RATE = 32_000


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load(path: Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    value = torch.from_numpy(wave.mean(axis=1)).float()
    if rate != sample_rate:
        value = AF.resample(value, rate, sample_rate)
    return value


def _align(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    length = min(left.numel(), right.numel())
    return left[:length], right[:length]


def _metric(prediction: torch.Tensor, mixture: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction, target = _align(prediction.float(), target.float())
    mixture, target = _align(mixture.float(), target.float())
    prediction, mixture = _align(prediction, mixture)
    target = target[: prediction.numel()]
    si = float(scale_invariant_sdr(prediction[None], target[None])[0])
    mix_si = float(scale_invariant_sdr(mixture[None], target[None])[0])
    sd = float(scale_dependent_sdr(prediction[None], target[None])[0])
    mix_sd = float(scale_dependent_sdr(mixture[None], target[None])[0])
    return {
        "si_sdr_db_↑": si, "mixture_si_sdr_db_↑": mix_si, "si_sdri_db_↑": si - mix_si,
        "sd_sdr_db_↑": sd, "mixture_sd_sdr_db_↑": mix_sd, "sd_sdri_db_↑": sd - mix_sd,
        "l1_↓": float((prediction - target).abs().mean()),
        "output_to_target_energy_ratio_db_↔": 10.0 * math.log10((float(prediction.square().mean()) + 1e-12) / (float(target.square().mean()) + 1e-12)),
    }


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _words(value: str) -> list[str]:
    return re.findall(r"\w+", str(value).lower(), flags=re.UNICODE)


def _wer(reference: str, hypothesis: str) -> float:
    left, right = _words(reference), _words(hypothesis)
    if not left:
        return 0.0 if not right else 1.0
    previous = list(range(len(right) + 1))
    for i, source in enumerate(left, 1):
        current = [i]
        for j, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (source != target)))
        previous = current
    return previous[-1] / len(left)


def _asr_audio(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-6:
        return value
    gain = min(10.0 ** (20.0 / 20.0), (10.0 ** (-24.0 / 20.0)) / rms)
    return np.clip(value * gain, -1.0, 1.0)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--audiosep-root", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep")
    parser.add_argument("--audiosep-config", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep/config/audiosep_base.yaml")
    parser.add_argument("--audiosep-checkpoint", type=Path, default=PROJECT_ROOT / "code/baseline/audiosep/checkpoint/hf_audiosep/pytorch_model.bin")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_audiosep_v1")
    parser.add_argument("--asr-model", default=str(Path.home() / ".cache/huggingface/hub/models--vinai--PhoWhisper-tiny/snapshots/cc51d32be916efebde04ff549854fa1741cb5c02"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-val-scenes", type=int, default=0)
    parser.add_argument("--max-test-scenes", type=int, default=0)
    parser.add_argument("--fixed-prompt", choices=PROMPTS, default="")
    parser.add_argument("--asr-batch-size", type=int, default=8)
    parser.add_argument("--asr-max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")
    val = [x for x in scenes if x["split"] == "val"]
    test = [x for x in scenes if x["split"] == "test"]
    if args.max_val_scenes:
        val = val[:args.max_val_scenes]
    if args.max_test_scenes:
        test = test[:args.max_test_scenes]

    active_prompts = (args.fixed_prompt,) if args.fixed_prompt else PROMPTS
    prompt_embeddings = encode_prompts(args.audiosep_root.resolve(), args.audiosep_checkpoint.resolve(), active_prompts, batch_size=5)
    separator = _load_separator(args, device)
    separator_rows: list[dict[str, Any]] = []

    def separate(scene: dict[str, Any], prompt: str, path: Path) -> dict[str, Any]:
        mixture = _load(_resolve(scene["mixture_path"])).to(device)
        speech_event = next(x for x in scene["events"] if x["label"] == "Speech")
        target = _load(_resolve(speech_event["stem_path"]))
        condition = prompt_embeddings[prompt][None].to(device)
        with torch.inference_mode():
            prediction = separator({"mixture": mixture[None, None], "condition": condition})["waveform"][0, 0].detach().cpu()
        metrics = _metric(prediction, mixture.cpu(), target)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, prediction.numpy(), SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        print(json.dumps({"scene": scene["scene_id"], "split": scene["split"], "prompt": prompt, "si_sdri": metrics["si_sdri_db_↑"]}), flush=True)
        return {
            "scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
            "prompt": prompt, "separated_path": _portable(path), **metrics,
        }

    for scene in val:
        for prompt in active_prompts:
            path = output / "audio/validation_prompt_bank" / scene["scene_id"] / f"{_slug(prompt)}.flac"
            separator_rows.append(separate(scene, prompt, path))
    prompt_summary = {}
    for prompt in active_prompts:
        rows = [x for x in separator_rows if x["prompt"] == prompt]
        prompt_summary[prompt] = {
            "scenes": len(rows), "mean_si_sdr_db_↑": _mean(rows, "si_sdr_db_↑"),
            "mean_si_sdri_db_↑": _mean(rows, "si_sdri_db_↑"), "mean_sd_sdri_db_↑": _mean(rows, "sd_sdri_db_↑"),
            "mean_l1_↓": _mean(rows, "l1_↓"),
        }
    selected_prompt = args.fixed_prompt or max(active_prompts, key=lambda x: (prompt_summary[x]["mean_si_sdri_db_↑"], -prompt_summary[x]["mean_l1_↓"]))
    print(f"SELECTED_PROMPT={selected_prompt}", flush=True)
    for scene in val:
        source = output / "audio/validation_prompt_bank" / scene["scene_id"] / f"{_slug(selected_prompt)}.flac"
        target = output / "audio/selected" / "val" / f"{scene['scene_id']}.flac"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for scene in test:
        path = output / "audio/selected" / "test" / f"{scene['scene_id']}.flac"
        separator_rows.append(separate(scene, selected_prompt, path))

    del separator, prompt_embeddings
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    processor = AutoProcessor.from_pretrained(args.asr_model, local_files_only=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(args.asr_model, local_files_only=True, dtype=dtype)
    if device.type == "cuda":
        asr_model = asr_model.cuda()
    asr = pipeline("automatic-speech-recognition", model=asr_model, tokenizer=processor.tokenizer,
                   feature_extractor=processor.feature_extractor, device=0 if device.type == "cuda" else -1)
    asr_audio, asr_meta = [], []
    evaluated_scenes = val + test
    for scene in evaluated_scenes:
        speech = next(x for x in scene["events"] if x["label"] == "Speech")
        mixture, mix_rate = sf.read(_resolve(scene["mixture_path"]), dtype="float32")
        clean, clean_rate = sf.read(_resolve(speech["stem_path"]), dtype="float32")
        enhanced_path = output / "audio/selected" / scene["split"] / f"{scene['scene_id']}.flac"
        enhanced, enhanced_rate = sf.read(enhanced_path, dtype="float32")
        onset, offset = float(speech["onset_seconds"]), float(speech["offset_seconds"])
        for mode, wave, rate, crop in (
            ("mixture_oracle_span", mixture, mix_rate, True),
            ("audiosep_full", enhanced, enhanced_rate, False),
            ("audiosep_oracle_span", enhanced, enhanced_rate, True),
            ("clean_speech_upper_bound", clean, clean_rate, True),
        ):
            if crop:
                start = max(0, int(math.floor((onset - .15) * rate)))
                end = min(len(wave), int(math.ceil((offset + .15) * rate)))
                wave = wave[start:end]
            asr_audio.append({"array": _asr_audio(wave), "sampling_rate": rate})
            asr_meta.append({"scene_id": scene["scene_id"], "split": scene["split"], "difficulty": scene["difficulty"],
                             "mode": mode, "reference": speech["transcript"], "enhanced_path": _portable(enhanced_path)})
    print(f"PhoWhisper items={len(asr_audio)}", flush=True)
    hypotheses = asr(
        asr_audio,
        batch_size=args.asr_batch_size,
        generate_kwargs={
            "language": "vi",
            "task": "transcribe",
            "max_new_tokens": args.asr_max_new_tokens,
        },
    )
    asr_rows = []
    for meta, hypothesis in zip(asr_meta, hypotheses):
        text = str(hypothesis["text"]).strip()
        error = _wer(meta["reference"], text)
        asr_rows.append({**meta, "hypothesis": text, "wer_↓": error, "correct_at_wer_0.25": error <= .25})

    asr_summary: dict[str, Any] = {}
    for split in ("val", "test"):
        asr_summary[split] = {}
        for mode in ("mixture_oracle_span", "audiosep_full", "audiosep_oracle_span", "clean_speech_upper_bound"):
            rows = [x for x in asr_rows if x["split"] == split and x["mode"] == mode]
            asr_summary[split][mode] = {
                "scenes": len(rows), "accuracy_at_wer_0.25_↑": float(np.mean([x["correct_at_wer_0.25"] for x in rows])),
                "mean_wer_↓": _mean(rows, "wer_↓"),
            }
    test_sep = [x for x in separator_rows if x["split"] == "test" and x["prompt"] == selected_prompt]
    test_sep_summary = {
        "scenes": len(test_sep), "mean_si_sdr_db_↑": _mean(test_sep, "si_sdr_db_↑"),
        "mean_si_sdri_db_↑": _mean(test_sep, "si_sdri_db_↑"), "mean_sd_sdri_db_↑": _mean(test_sep, "sd_sdri_db_↑"),
        "mean_l1_↓": _mean(test_sep, "l1_↓"),
    }
    separator_path, asr_path = output / "separator_items.jsonl", output / "asr_items.jsonl"
    _atomic_text(separator_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in separator_rows))
    _atomic_text(asr_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in asr_rows))
    receipt = {
        "format": "qces_vietnamese_automotive_audiosep_receipt_v1", "complete": True,
        "selection_protocol": ("fixed prompt supplied from a separate validation-ASR audit; test labels/transcripts never used" if args.fixed_prompt else "global prompt chosen on validation mean SI-SDRi; test labels/transcripts never used for prompt selection"),
        "prompt_bank": list(active_prompts), "validation_prompt_summary": prompt_summary, "selected_prompt": selected_prompt,
        "test_separator": test_sep_summary, "asr": asr_summary,
        "audiosep_frozen": True, "audiosep_checkpoint_sha256": hashlib.sha256(args.audiosep_checkpoint.resolve().read_bytes()).hexdigest(),
        "separator_items": _portable(separator_path), "asr_items": _portable(asr_path),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected_prompt": selected_prompt, "test_separator": test_sep_summary, "asr": asr_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
