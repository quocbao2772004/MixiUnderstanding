#!/usr/bin/env python3
"""Validation-select an ASR-preserving fusion of speech enhancement outputs."""

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

from mixi_understanding.qces.metrics import scale_invariant_sdr
from mixi_understanding.scripts.evaluate_qces_speech_event_pipeline import PROJECT_ROOT, _atomic_text, _portable, _resolve


RATE = 16_000


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _load(path: Path) -> np.ndarray:
    value, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(value.mean(axis=1)).float()
    if rate != RATE:
        wave = AF.resample(wave, rate, RATE)
    return wave.numpy()


def _align(*values: np.ndarray) -> list[np.ndarray]:
    length = min(len(value) for value in values)
    return [value[:length] for value in values]


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
    rms = float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12))
    if rms < 1e-6:
        return value.astype(np.float32)
    return np.clip(value * min(10.0, (10 ** (-24 / 20)) / rms), -1.0, 1.0).astype(np.float32)


def _candidate(name: str, mixture: np.ndarray, audiosep: np.ndarray, domain: np.ndarray) -> np.ndarray:
    if name == "audiosep":
        return audiosep
    if name == "domain":
        return domain
    if name.startswith("audiosep_domain_"):
        alpha = float(name.rsplit("_", 1)[1])
        return alpha * audiosep + (1.0 - alpha) * domain
    if name.startswith("audiosep_mix_"):
        beta = float(name.rsplit("_", 1)[1])
        return (1.0 - beta) * audiosep + beta * mixture
    if name.startswith("domain_mix_"):
        beta = float(name.rsplit("_", 1)[1])
        return (1.0 - beta) * domain + beta * mixture
    if name.startswith("ensemble_mix_"):
        beta = float(name.rsplit("_", 1)[1])
        return (1.0 - beta) * (0.5 * audiosep + 0.5 * domain) + beta * mixture
    raise ValueError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--audiosep-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_audiosep_v1")
    parser.add_argument("--domain-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_sepformer_finetuned_eval_v1")
    parser.add_argument("--model", default=str(Path.home() / ".cache/huggingface/hub/models--openai--whisper-medium/snapshots/abdf7c39ab9d0397620ccaea8974cc764cd0953e"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_vietnamese_automotive_fusion_v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scenes = _jsonl(args.dataset_dir.resolve() / "scenes.jsonl")
    candidates = (
        "audiosep", "domain",
        "audiosep_domain_0.25", "audiosep_domain_0.50", "audiosep_domain_0.75",
        "audiosep_mix_0.10", "audiosep_mix_0.20", "audiosep_mix_0.30",
        "domain_mix_0.10", "domain_mix_0.20", "domain_mix_0.30",
        "ensemble_mix_0.10", "ensemble_mix_0.20",
    )

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, local_files_only=True, dtype=torch.float16).cuda()
    transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                           feature_extractor=processor.feature_extractor, device=0)

    cache: dict[str, dict[str, Any]] = {}
    def scene_audio(scene: dict[str, Any]) -> dict[str, Any]:
        scene_id, split = scene["scene_id"], scene["split"]
        speech = next(event for event in scene["events"] if event["label"] == "Speech")
        mixture = _load(_resolve(scene["mixture_path"]))
        audiosep = _load(args.audiosep_dir.resolve() / "audio/selected" / split / f"{scene_id}.flac")
        domain = _load(args.domain_dir.resolve() / "audio/selected" / split / f"{scene_id}.flac")
        target = _load(_resolve(speech["stem_path"]))
        mixture, audiosep, domain, target = _align(mixture, audiosep, domain, target)
        return {"mixture": mixture, "audiosep": audiosep, "domain": domain, "target": target, "speech": speech}

    val_audio, val_meta = [], []
    for scene in [row for row in scenes if row["split"] == "val"]:
        values = cache.setdefault(scene["scene_id"], scene_audio(scene))
        speech = values["speech"]
        start = max(0, int((float(speech["onset_seconds"]) - 0.15) * RATE))
        end = min(len(values["mixture"]), int((float(speech["offset_seconds"]) + 0.15) * RATE))
        for name in candidates:
            wave = _candidate(name, values["mixture"], values["audiosep"], values["domain"])
            val_audio.append({"array": _normalize(wave[start:end]), "sampling_rate": RATE})
            val_meta.append({"scene_id": scene["scene_id"], "split": "val", "mode": name, "reference": speech["transcript"]})
    hypotheses = transcriber(
        val_audio, batch_size=1,
        generate_kwargs={"language": "vi", "task": "transcribe", "max_new_tokens": 96,
                         "no_repeat_ngram_size": 3, "repetition_penalty": 1.05},
    )
    rows = []
    for meta, result in zip(val_meta, hypotheses):
        hypothesis = str(result["text"]).strip()
        rows.append({**meta, "hypothesis": hypothesis, "wer_↓": _wer(meta["reference"], hypothesis)})
    validation = {}
    for name in candidates:
        chosen = [row for row in rows if row["mode"] == name]
        sisdri = []
        for scene in [row for row in scenes if row["split"] == "val"]:
            values = cache[scene["scene_id"]]
            prediction = torch.from_numpy(_candidate(name, values["mixture"], values["audiosep"], values["domain"]))
            mixture, target = torch.from_numpy(values["mixture"]), torch.from_numpy(values["target"])
            enhanced = float(scale_invariant_sdr(prediction[None], target[None])[0])
            base = float(scale_invariant_sdr(mixture[None], target[None])[0])
            sisdri.append(enhanced - base)
        validation[name] = {"mean_wer_↓": float(np.mean([row["wer_↓"] for row in chosen])),
                            "mean_si_sdri_db_↑": float(np.mean(sisdri))}
    selected = min(candidates, key=lambda name: (validation[name]["mean_wer_↓"], -validation[name]["mean_si_sdri_db_↑"]))
    print(json.dumps({"selected": selected, "validation": validation}, ensure_ascii=False, indent=2), flush=True)

    test_modes = ("mixture", "audiosep", "domain", selected, "clean_upper_bound")
    test_audio, test_meta = [], []
    for scene in [row for row in scenes if row["split"] == "test"]:
        values = cache.setdefault(scene["scene_id"], scene_audio(scene))
        speech = values["speech"]
        start = max(0, int((float(speech["onset_seconds"]) - 0.15) * RATE))
        end = min(len(values["mixture"]), int((float(speech["offset_seconds"]) + 0.15) * RATE))
        mode_values = {
            "mixture": values["mixture"], "audiosep": values["audiosep"], "domain": values["domain"],
            selected: _candidate(selected, values["mixture"], values["audiosep"], values["domain"]),
            "clean_upper_bound": values["target"],
        }
        for name in dict.fromkeys(test_modes):
            wave = mode_values[name]
            test_audio.append({"array": _normalize(wave[start:end]), "sampling_rate": RATE})
            test_meta.append({"scene_id": scene["scene_id"], "split": "test", "mode": name, "reference": speech["transcript"]})
        chosen_path = output / "audio/selected/test" / f"{scene['scene_id']}.flac"
        chosen_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(chosen_path, mode_values[selected], RATE, format="FLAC", subtype="PCM_16")
    hypotheses = transcriber(
        test_audio, batch_size=1,
        generate_kwargs={"language": "vi", "task": "transcribe", "max_new_tokens": 96,
                         "no_repeat_ngram_size": 3, "repetition_penalty": 1.05},
    )
    for meta, result in zip(test_meta, hypotheses):
        hypothesis = str(result["text"]).strip()
        rows.append({**meta, "hypothesis": hypothesis, "wer_↓": _wer(meta["reference"], hypothesis)})
    test = {}
    for name in dict.fromkeys(test_modes):
        chosen = [row for row in rows if row["split"] == "test" and row["mode"] == name]
        test[name] = {"scenes": len(chosen), "mean_wer_↓": float(np.mean([row["wer_↓"] for row in chosen])),
                      "accuracy_at_wer_0.25_↑": float(np.mean([row["wer_↓"] <= 0.25 for row in chosen]))}
    item_path = output / "items.jsonl"
    _atomic_text(item_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    receipt = {"format": "qces_vietnamese_automotive_fusion_receipt_v1", "complete": True,
               "selection_protocol": "one global fusion selected on validation Whisper-medium WER; test locked",
               "selected_mode": selected, "validation": validation, "test": test, "items": _portable(item_path)}
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"selected": selected, "test": test}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
