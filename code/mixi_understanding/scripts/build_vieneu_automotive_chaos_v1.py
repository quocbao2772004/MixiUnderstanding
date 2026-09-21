#!/usr/bin/env python3
"""Build a single-speaker Vietnamese benchmark with dense chaotic noise."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from mixi_understanding.scripts.build_qces_speech_event_v1 import (
    PROJECT_ROOT, SAMPLE_RATE, _atomic_text, _jsonl, _max_energy_crop, _normalize,
    _portable, _read_mono, _resolve, _rms, _write_audio,
)
from mixi_understanding.scripts.build_qces_vietnamese_speech_demo import _question


CONTINUOUS = (
    "Traffic_noise_and_roadway_noise", "Heavy_engine_(low_frequency)",
    "Wind_noise_(microphone)", "Rain_on_surface", "Mechanical_fan",
)
TRANSIENT = (
    "Air_horn_and_truck_horn", "Police_car_(siren)", "Reversing_beeps",
    "Engine_starting", "Car_alarm", "Glass_shatter", "Slam", "Keys_jangling",
    "Bark", "Baby_cry_and_infant_cry", "Screaming", "Train_horn",
)
DISPLAY = {
    "Traffic_noise_and_roadway_noise": "tiếng giao thông", "Heavy_engine_(low_frequency)": "tiếng động cơ nặng",
    "Wind_noise_(microphone)": "tiếng gió microphone", "Rain_on_surface": "tiếng mưa",
    "Mechanical_fan": "tiếng quạt", "Air_horn_and_truck_horn": "tiếng còi xe tải",
    "Police_car_(siren)": "tiếng còi xe cảnh sát", "Reversing_beeps": "tiếng bíp lùi xe",
    "Engine_starting": "tiếng động cơ khởi động", "Car_alarm": "tiếng báo động ô tô",
    "Glass_shatter": "tiếng kính vỡ", "Slam": "tiếng đóng sầm", "Keys_jangling": "tiếng chìa khóa",
    "Bark": "tiếng chó sủa", "Baby_cry_and_infant_cry": "tiếng trẻ con khóc",
    "Screaming": "tiếng hét", "Train_horn": "tiếng còi tàu", "Speech": "lời nói tiếng Việt",
}
TARGET_SNRS = (0.0, -5.0, -10.0)


def _pool(root: Path) -> dict[str, list[dict[str, Any]]]:
    labels = set(CONTINUOUS) | set(TRANSIENT)
    rows: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in labels}
    for path in root.glob("**/source_bank.jsonl"):
        for row in _jsonl(path):
            label, source_id = str(row.get("label") or ""), str(row.get("source_id") or "")
            if label in rows and source_id and _resolve(str(row["audio_path"])).is_file():
                rows[label][source_id] = row
    return {label: sorted(values.values(), key=lambda row: str(row["source_id"])) for label, values in rows.items()}


def _split_pool(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    midpoint = len(rows) // 2
    return rows[:midpoint] if split == "val" else rows[midpoint:]


def _active(row: dict[str, Any], seconds: float) -> np.ndarray:
    wave = _read_mono(_resolve(str(row["audio_path"])))
    start = max(0, int(round(float(row.get("active_onset_seconds", 0.0)) * SAMPLE_RATE)))
    end = min(len(wave), int(round(float(row.get("active_offset_seconds", len(wave) / SAMPLE_RATE)) * SAMPLE_RATE)))
    return _max_energy_crop(wave[start:end] if end > start else wave, seconds)


def _fit(wave: np.ndarray, frames: int) -> np.ndarray:
    if len(wave) >= frames:
        return _max_energy_crop(wave, frames / SAMPLE_RATE)[:frames]
    return np.tile(wave, int(math.ceil(frames / max(len(wave), 1))))[:frames].astype(np.float32)


def _place(wave: np.ndarray, frames: int, onset: float) -> tuple[np.ndarray, int, int]:
    start = max(0, min(frames, int(round(onset * SAMPLE_RATE))))
    end = min(frames, start + len(wave))
    stem = np.zeros(frames, dtype=np.float32); stem[start:end] = wave[: end - start]
    return stem, start, end


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-dataset", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_vieneu_source_v3")
    parser.add_argument("--source-bank-root", type=Path, default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1/primary_clean")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_chaos_v1")
    parser.add_argument("--seed", type=int, default=2026081001)
    args = parser.parse_args(); rng = random.Random(args.seed)
    bases = _jsonl(args.paired_dataset.resolve() / "scenes.jsonl")
    pools = _pool(args.source_bank_root.resolve())
    missing = {label: len(rows) for label, rows in pools.items() if len(rows) < 12}
    if missing: raise RuntimeError(f"insufficient source pools: {missing}")
    output = args.output_dir.resolve(); scenes: list[dict[str, Any]] = []; questions: list[dict[str, Any]] = []
    used_by_split: dict[str, set[str]] = collections.defaultdict(set); recon: list[float] = []; measured: list[float] = []
    cursors: dict[tuple[str, str], int] = collections.defaultdict(int)

    def take(label: str, split: str) -> dict[str, Any]:
        candidates = _split_pool(pools[label], split); key = (label, split); index = cursors[key]
        if index >= len(candidates): raise RuntimeError(f"{label}/{split}: exhausted unique sources")
        cursors[key] += 1; row = candidates[index]; used_by_split[str(row["source_id"])].add(split); return row

    for scene_index, base in enumerate(bases):
        split = base["split"]; speech_meta = next(event for event in base["events"] if event["label"] == "Speech")
        aligned = _read_mono(_resolve(str(speech_meta["stem_path"])))
        old_start = int(round(float(speech_meta["onset_seconds"]) * SAMPLE_RATE)); old_end = int(round(float(speech_meta["offset_seconds"]) * SAMPLE_RATE))
        speech = _normalize(aligned[old_start:old_end], -22.0); speech_onset = 1.25; speech_offset = speech_onset + len(speech) / SAMPLE_RATE
        duration = speech_offset + 1.55; frames = int(math.ceil(duration * SAMPLE_RATE)); speech_stem, s0, s1 = _place(speech, frames, speech_onset)
        selected_continuous = rng.sample(list(CONTINUOUS), 3)
        selected_transient = rng.sample(list(TRANSIENT), 7)
        components: list[tuple[str, str, dict[str, Any], np.ndarray, int, int]] = []
        for label in selected_continuous:
            source = take(label, split); stem = _fit(_active(source, duration), frames)
            components.append(("continuous", label, source, stem, 0, frames))
        roles = ("before", "overlap_1", "overlap_2", "overlap_3", "overlap_4", "overlap_5", "after")
        overlap_centers = np.linspace(speech_onset + 0.15, max(speech_onset + 0.2, speech_offset - 0.55), 5)
        for transient_index, (label, role) in enumerate(zip(selected_transient, roles)):
            source = take(label, split); maximum = 1.0 if role in {"before", "after"} else rng.uniform(0.75, 1.8)
            wave = _active(source, maximum)
            if role == "before": onset = max(0.12, speech_onset - len(wave) / SAMPLE_RATE - 0.08)
            elif role == "after": onset = speech_offset + 0.10
            else: onset = float(overlap_centers[transient_index - 1])
            stem, start, end = _place(wave, frames, onset)
            components.append((role, label, source, stem, start, end))
        raw_noise = np.zeros(frames, dtype=np.float32)
        weights = []
        for role, _, _, stem, _, _ in components:
            weight = 10 ** (rng.uniform(-5.0, 2.0 if role.startswith("overlap") else -1.0) / 20.0)
            weights.append(weight); raw_noise += weight * stem
        active = np.abs(speech_stem) > 1e-7; target_snr = TARGET_SNRS[scene_index % len(TARGET_SNRS)]
        scale_noise = _rms(speech_stem[active]) / (10 ** (target_snr / 20.0)) / max(_rms(raw_noise[active]), 1e-7)
        noise_stems = [stem * weight * scale_noise for weight, (_, _, _, stem, _, _) in zip(weights, components)]
        mixture = speech_stem + sum(noise_stems, np.zeros(frames, dtype=np.float32))
        peak = max(float(np.max(np.abs(mixture))), float(np.max(np.abs(speech_stem)))); global_scale = min(1.0, 0.97 / max(peak, 1e-9))
        mixture *= global_scale; speech_stem *= global_scale; noise_stems = [stem * global_scale for stem in noise_stems]
        scene_id = f"vi_auto_chaos_{split}_{scene_index:03d}"; mix_path = output / "audio/mixtures" / f"{scene_id}.flac"; _write_audio(mix_path, mixture)
        event_specs = [("speech", "Speech", speech_meta, speech_stem, s0, s1)] + [
            (role, label, source, stem, start, end) for (role, label, source, _, start, end), stem in zip(components, noise_stems)
        ]
        events = []
        for event_index, (role, label, source, stem, start, end) in enumerate(event_specs):
            event_id = f"{scene_id}_e{event_index:02d}"; path = output / "audio/stems" / scene_id / f"{event_id}.flac"; _write_audio(path, stem)
            events.append({
                "event_id": event_id, "event_kind": "speech" if label == "Speech" else "sound_event", "role": role,
                "label": label, "display_name": DISPLAY[label], "onset_seconds": start / SAMPLE_RATE, "offset_seconds": end / SAMPLE_RATE,
                "source_id": str(speech_meta["source_id"] if label == "Speech" else source["source_id"]),
                "speaker_id": speech_meta.get("speaker_id") if label == "Speech" else None,
                "speaker_group": speech_meta.get("speaker_group") if label == "Speech" else None,
                "transcript": speech_meta.get("transcript") if label == "Speech" else None, "stem_path": _portable(path),
            })
        decoded, _ = sf.read(mix_path, dtype="float32"); summed = sum((sf.read(_resolve(event["stem_path"]), dtype="float32")[0] for event in events), np.zeros_like(decoded)); recon.append(_rms(decoded - summed))
        interference = sum(noise_stems, np.zeros(frames, dtype=np.float32)); measured_snr = 10 * math.log10((_rms(speech_stem[active]) ** 2 + 1e-12) / (_rms(interference[active]) ** 2 + 1e-12)); measured.append(measured_snr)
        scene = {"format": "qces_vietnamese_automotive_chaos_scene_v1", "scene_id": scene_id, "paired_scene_id": base["scene_id"], "split": split, "sample_rate": SAMPLE_RATE, "duration_seconds": frames / SAMPLE_RATE, "mixture_path": _portable(mix_path), "speaker_count": 1, "language": "vi-VN", "difficulty": f"chaos_snr_{int(target_snr)}", "requested_speech_to_noise_snr_db": target_snr, "measured_speech_to_all_interference_snr_db": measured_snr, "events": events}
        scenes.append(scene); speech_event = events[0]; before_event = next(event for event in events if event["role"] == "before"); after_event = next(event for event in events if event["role"] == "after")
        questions.extend([
            _question(scene, 0, "Người trong xe đã nói gì?", "vi_speech_content", "transcript", str(speech_event["transcript"]), [], [speech_event["event_id"]], [speech_event["event_id"]]),
            _question(scene, 1, "Âm thanh nào xảy ra ngay trước khi người đó nói?", "vi_event_before_speech", "event_label", before_event["display_name"], [speech_event["event_id"]], [before_event["event_id"]], [before_event["event_id"], speech_event["event_id"]]),
            _question(scene, 2, "Âm thanh nào xảy ra ngay sau khi người đó nói xong?", "vi_event_after_speech", "event_label", after_event["display_name"], [speech_event["event_id"]], [after_event["event_id"]], [speech_event["event_id"], after_event["event_id"]]),
            _question(scene, 3, "Người nói thứ hai đã nói gì?", "vi_second_speaker_absent", "no_evidence", "Không có bằng chứng", [], [], [speech_event["event_id"]], no_evidence=True),
        ])
    leakage = {source: sorted(splits) for source, splits in used_by_split.items() if len(splits) > 1}
    scene_path, question_path = output / "scenes.jsonl", output / "questions.jsonl"
    _atomic_text(scene_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scenes)); _atomic_text(question_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in questions))
    receipt = {"format": "qces_vietnamese_automotive_chaos_dataset_receipt_v1", "complete": not leakage, "scene_count": len(scenes), "question_count": len(questions), "scenes_by_split": dict(collections.Counter(row["split"] for row in scenes)), "events_per_scene": 11, "noise_sources_per_scene": 10, "continuous_noise_sources_per_scene": 3, "transient_noise_sources_per_scene": 7, "noise_class_inventory": sorted(set(CONTINUOUS) | set(TRANSIENT)), "noise_class_count": len(set(CONTINUOUS) | set(TRANSIENT)), "target_snr_db_values": list(TARGET_SNRS), "mean_measured_snr_db": float(np.mean(measured)), "minimum_measured_snr_db": float(np.min(measured)), "cross_split_noise_source_leakage": len(leakage), "maximum_reconstruction_rms": max(recon), "scene_manifest": _portable(scene_path), "question_manifest": _portable(question_path), "scene_manifest_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(), "question_manifest_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest()}
    _atomic_text(output / "dataset_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"); print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if leakage: raise RuntimeError(f"cross-split leakage: {list(leakage.items())[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
