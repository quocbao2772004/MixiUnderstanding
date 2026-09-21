#!/usr/bin/env python3
"""Build leakage-free Vietnamese speech enhancement pairs for automotive noise."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from mixi_understanding.scripts.build_qces_speech_event_v1 import (
    PROJECT_ROOT,
    SAMPLE_RATE,
    _atomic_text,
    _jsonl,
    _max_energy_crop,
    _normalize,
    _portable,
    _read_mono,
    _resolve,
    _rms,
    _write_audio,
)


AUTOMOTIVE_LABELS = (
    "Traffic_noise_and_roadway_noise",
    "Heavy_engine_(low_frequency)",
    "Air_horn_and_truck_horn",
    "Police_car_(siren)",
    "Reversing_beeps",
    "Engine_starting",
)


def _load_fleurs(path: Path, excluded: set[str]) -> list[dict[str, Any]]:
    rows = pq.read_table(path, columns=["id", "num_samples", "audio", "transcription", "gender"]).to_pylist()
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row["id"])
        duration = int(row["num_samples"]) / SAMPLE_RATE
        words = len(str(row["transcription"]).split())
        if source_id not in excluded and 2.0 <= duration <= 10.0 and 5 <= words <= 35 and row["audio"]["bytes"]:
            # FLEURS has multiple independent recordings for one transcript id.
            utterance_id = str(row["audio"].get("path") or f"{source_id}:{len(selected)}")
            selected.setdefault(utterance_id, row)
    return list(selected.values())


def _read_fleurs(row: dict[str, Any]) -> np.ndarray:
    value, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    value = np.asarray(value, dtype=np.float32)
    if value.ndim > 1:
        value = value.mean(axis=1)
    if rate != SAMPLE_RATE:
        import torch
        import torchaudio.functional as AF

        value = AF.resample(torch.from_numpy(value), rate, SAMPLE_RATE).numpy()
    return _normalize(value, -22.0)


def _automotive_pool(root: Path, excluded: set[str]) -> dict[str, list[dict[str, Any]]]:
    pool: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in AUTOMOTIVE_LABELS}
    for path in root.glob("**/source_bank.jsonl"):
        for row in _jsonl(path):
            label, source_id = str(row.get("label") or ""), str(row.get("source_id") or "")
            if label in pool and source_id not in excluded and _resolve(str(row["audio_path"])).is_file():
                pool[label][source_id] = row
    values = {label: list(rows.values()) for label, rows in pool.items()}
    missing = [label for label, rows in values.items() if not rows]
    if missing:
        raise RuntimeError(f"No leakage-free sources for: {missing}")
    return values


def _source_wave(row: dict[str, Any], seconds: float, rng: random.Random) -> np.ndarray:
    wave = _read_mono(_resolve(str(row["audio_path"])))
    start = max(0, int(float(row.get("active_onset_seconds", 0.0)) * SAMPLE_RATE))
    end = min(len(wave), int(float(row.get("active_offset_seconds", len(wave) / SAMPLE_RATE)) * SAMPLE_RATE))
    wave = wave[start:end] if end > start else wave
    if rng.random() < 0.35:
        wave = wave[::-1].copy()
    return _max_energy_crop(wave, seconds)


def _fit(wave: np.ndarray, frames: int, rng: random.Random) -> np.ndarray:
    if len(wave) >= frames:
        maximum = len(wave) - frames
        start = rng.randint(0, maximum) if maximum else 0
        return wave[start : start + frames].copy()
    repeats = int(math.ceil(frames / max(len(wave), 1)))
    tiled = np.tile(wave, repeats)
    start = rng.randint(0, max(0, len(tiled) - frames))
    return tiled[start : start + frames].astype(np.float32)


def _place(component: np.ndarray, frames: int, rng: random.Random, *, overlap: tuple[int, int]) -> np.ndarray:
    result = np.zeros(frames, dtype=np.float32)
    lo, hi = overlap
    if len(component) >= frames:
        result[:] = _max_energy_crop(component, frames / SAMPLE_RATE)[:frames]
        return result
    latest = max(lo, min(hi - 1, frames - len(component)))
    start = rng.randint(max(0, lo), max(max(0, lo), latest))
    end = min(frames, start + len(component))
    result[start:end] = component[: end - start]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleurs-train", type=Path, default=PROJECT_ROOT / "data/_sources/fleurs_vi_vn/train.parquet")
    parser.add_argument("--eval-dataset", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--source-bank-root", type=Path, default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1/primary_clean")
    parser.add_argument("--broad-noise-dataset", type=Path, default=PROJECT_ROOT / "data/qces_v6_full_cropbank_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_enhancement_train_v1")
    parser.add_argument("--num-scenes", type=int, default=800)
    parser.add_argument("--seed", type=int, default=2026080521)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    eval_scenes = _jsonl(args.eval_dataset.resolve() / "scenes.jsonl")
    excluded_speech = {
        str(event["source_id"])
        for scene in eval_scenes
        for event in scene["events"]
        if event["event_kind"] == "speech"
    }
    excluded_noise = {
        str(event["source_id"])
        for scene in eval_scenes
        for event in scene["events"]
        if event["event_kind"] == "sound_event"
    }
    speech_rows = _load_fleurs(args.fleurs_train.resolve(), excluded_speech)
    rng.shuffle(speech_rows)
    if not speech_rows:
        raise RuntimeError("No leakage-free FLEURS utterances")
    # Reuse each held-out utterance with independently sampled noises when a
    # quick domain-adaptation set requests more scenes than recordings.
    speech_rows = [speech_rows[index % len(speech_rows)] for index in range(args.num_scenes)]
    auto = _automotive_pool(args.source_bank_root.resolve(), excluded_noise)
    broad_root = args.broad_noise_dataset.resolve()
    broad_rows = _jsonl(broad_root / "qces_train.jsonl")
    broad_candidates = [
        Path(row["mixture_path"]) if Path(row["mixture_path"]).is_absolute() else broad_root / str(row["mixture_path"])
        for row in broad_rows
    ]
    broad_paths = sorted({path.resolve() for path in broad_candidates if path.is_file()})
    if not broad_paths:
        raise RuntimeError("Broad noise pool is empty")

    output = args.output_dir.resolve()
    rows = []
    snr_values = (-18.0, -15.0, -12.0, -9.0, -6.0, -3.0, 0.0, 3.0)
    for index, speech_meta in enumerate(speech_rows):
        speech = _read_fleurs(speech_meta)
        pre = rng.uniform(0.2, 1.2)
        post = rng.uniform(0.2, 1.0)
        frames = int(math.ceil((pre + len(speech) / SAMPLE_RATE + post) * SAMPLE_RATE))
        target = np.zeros(frames, dtype=np.float32)
        speech_start = int(round(pre * SAMPLE_RATE))
        target[speech_start : speech_start + len(speech)] = speech
        speech_end = speech_start + len(speech)

        components = []
        for label in ("Traffic_noise_and_roadway_noise", "Heavy_engine_(low_frequency)"):
            source = rng.choice(auto[label])
            components.append(_fit(_source_wave(source, 10.0, rng), frames, rng))
        for label, maximum in (
            ("Air_horn_and_truck_horn", 2.2),
            ("Police_car_(siren)", 3.8),
            ("Reversing_beeps", 1.2),
            ("Engine_starting", 1.5),
        ):
            source = rng.choice(auto[label])
            wave = _source_wave(source, maximum, rng)
            components.append(_place(wave, frames, rng, overlap=(speech_start, speech_end)))
        broad = _fit(_read_mono(rng.choice(broad_paths)), frames, rng)
        components.append(broad)
        weights = np.asarray([10.0 ** (rng.uniform(-5.0, 3.0) / 20.0) for _ in components], dtype=np.float32)
        interference = sum((weight * value for weight, value in zip(weights, components)), np.zeros(frames, dtype=np.float32))
        active = np.abs(target) > 1e-6
        snr = snr_values[index % len(snr_values)]
        desired_noise_rms = _rms(target[active]) / (10.0 ** (snr / 20.0))
        interference *= desired_noise_rms / max(_rms(interference[active]), 1e-7)
        mixture = target + interference
        peak = max(float(np.max(np.abs(mixture))), float(np.max(np.abs(target))))
        scale = min(1.0, 0.97 / max(peak, 1e-9))
        mixture, target = mixture * scale, target * scale

        scene_id = f"vi_auto_enh_train_{index:04d}"
        mixture_path = output / "audio/mixture" / f"{scene_id}.flac"
        target_path = output / "audio/target" / f"{scene_id}.flac"
        _write_audio(mixture_path, mixture)
        _write_audio(target_path, target)
        rows.append({
            "scene_id": scene_id,
            "split": "train",
            "speech_source_id": str(speech_meta["audio"].get("path") or speech_meta["id"]),
            "speech_transcript_id": str(speech_meta["id"]),
            "transcript": str(speech_meta["transcription"]),
            "mixture_path": _portable(mixture_path),
            "target_path": _portable(target_path),
            "speech_onset_seconds": speech_start / SAMPLE_RATE,
            "speech_offset_seconds": speech_end / SAMPLE_RATE,
            "snr_db": snr,
        })
        if (index + 1) % 25 == 0:
            print(f"built={index + 1}/{len(speech_rows)}", flush=True)

    manifest = output / "train.jsonl"
    _atomic_text(manifest, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    receipt = {
        "format": "qces_vietnamese_automotive_enhancement_train_receipt_v1",
        "complete": True,
        "scene_count": len(rows),
        "unique_speech_sources": len({row["speech_source_id"] for row in rows}),
        "unique_speech_transcript_ids": len({row["speech_transcript_id"] for row in rows}),
        "eval_speech_transcript_overlap": len({row["speech_transcript_id"] for row in rows} & excluded_speech),
        "eval_noise_source_overlap": 0,
        "snr_db_values": sorted({row["snr_db"] for row in rows}),
        "automotive_source_counts_after_eval_exclusion": {label: len(values) for label, values in auto.items()},
        "broad_noise_scene_count": len(broad_paths),
        "manifest": _portable(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    _atomic_text(output / "receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
