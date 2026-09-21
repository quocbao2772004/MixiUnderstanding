#!/usr/bin/env python3
"""Build a small real-Vietnamese speech + automotive-noise demo set."""

from __future__ import annotations

import argparse
import collections
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


SCENES_PER_SPLIT = {"val": 12, "test": 12}
DIFFICULTIES = (("moderate_overlap", 3.0), ("hard_overlap", 0.0), ("extreme_overlap", -5.0))
VI_LABEL = {
    "Bark": "tiếng chó sủa",
    "Meow": "tiếng mèo kêu",
    "Reversing_beeps": "tiếng bíp lùi xe",
    "Crumpling_and_crinkling": "tiếng vò giấy",
    "Single-lens_reflex_camera": "tiếng máy ảnh",
    "Chirp_and_tweet": "tiếng chim hót",
    "Engine_starting": "tiếng động cơ khởi động",
    "Toilet_flush": "tiếng xả nước",
    "Finger_snapping": "tiếng búng tay",
    "Glass_shatter": "tiếng kính vỡ",
    "Air_horn_and_truck_horn": "tiếng còi xe tải",
    "Speech": "lời nói tiếng Việt",
}


def _scale_rms(waveform: np.ndarray, target: float) -> np.ndarray:
    return np.asarray(waveform, dtype=np.float32) * (target / max(_rms(waveform), 1e-7))


def _load_fleurs(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path, columns=["id", "num_samples", "audio", "transcription", "gender"]).to_pylist()
    selected = []
    seen_ids: set[str] = set()
    for row in rows:
        duration = int(row["num_samples"]) / SAMPLE_RATE
        words = str(row["transcription"]).split()
        source_id = str(row["id"])
        if source_id in seen_ids:
            continue
        if 2.2 <= duration <= 9.0 and 6 <= len(words) <= 30 and row["audio"]["bytes"]:
            selected.append(row)
            seen_ids.add(source_id)
    return selected


def _active_from_event(event: dict[str, Any]) -> np.ndarray:
    waveform = _read_mono(_resolve(str(event["stem_path"])))
    start = max(0, int(round(float(event["onset_seconds"]) * SAMPLE_RATE)))
    end = min(len(waveform), int(round(float(event["offset_seconds"]) * SAMPLE_RATE)))
    return _max_energy_crop(waveform[start:end] if end > start else waveform, 1.4)


def _question(scene: dict[str, Any], index: int, text: str, operation: str,
              answer_type: str, answer: str, anchors: list[str], answers: list[str],
              evidence: list[str], no_evidence: bool = False) -> dict[str, Any]:
    return {
        "format": "qces_vietnamese_speech_question_v1",
        "question_id": f"{scene['scene_id']}_q{index:02d}",
        "scene_id": scene["scene_id"],
        "split": scene["split"],
        "question": text,
        "operation": operation,
        "answer_type": answer_type,
        "answer": answer,
        "anchor_event_ids": anchors,
        "answer_event_ids": answers,
        "evidence_event_ids": evidence,
        "no_evidence": no_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleurs-parquet", type=Path, default=PROJECT_ROOT / "data/_sources/fleurs_vi_vn/validation.parquet")
    parser.add_argument("--hard-dataset", type=Path, default=PROJECT_ROOT / "data/qces_speech_event_hard_v2")
    parser.add_argument("--source-bank-root", type=Path, default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1/primary_clean")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_speech_event_demo_v1")
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output = args.output_dir.resolve()

    fleurs = _load_fleurs(args.fleurs_parquet.resolve())
    rng.shuffle(fleurs)
    required = sum(SCENES_PER_SPLIT.values())
    if len(fleurs) < required:
        raise RuntimeError(f"Need {required} FLEURS rows, got {len(fleurs)}")
    fleurs = fleurs[:required]

    donor_scenes = _jsonl(args.hard_dataset.resolve() / "scenes.jsonl")
    donors = {split: [row for row in donor_scenes if row["split"] == split] for split in SCENES_PER_SPLIT}
    horn_rows: list[dict[str, Any]] = []
    for path in args.source_bank_root.resolve().glob("**/source_bank.jsonl"):
        horn_rows.extend(row for row in _jsonl(path) if row.get("label") == "Air_horn_and_truck_horn" and _resolve(str(row["audio_path"])).is_file())
    unique_horns = {str(row["source_id"]): row for row in horn_rows}
    horn_rows = list(unique_horns.values())
    rng.shuffle(horn_rows)
    if len(horn_rows) < required:
        raise RuntimeError(f"Need {required} unique horns, got {len(horn_rows)}")

    scenes: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    reconstruction = []
    fleurs_cursor = horn_cursor = 0
    for split, count in SCENES_PER_SPLIT.items():
        rng.shuffle(donors[split])
        for local_index in range(count):
            source = fleurs[fleurs_cursor]
            horn = horn_rows[horn_cursor]
            fleurs_cursor += 1
            horn_cursor += 1
            donor = donors[split][local_index % len(donors[split])]
            donor_noise = [event for event in donor["events"] if event["event_kind"] == "sound_event"]
            before_source, overlap_source, after_source = donor_noise[0], donor_noise[3], donor_noise[2]

            speech, rate = sf.read(io.BytesIO(source["audio"]["bytes"]), dtype="float32")
            speech = np.asarray(speech, dtype=np.float32)
            if speech.ndim > 1:
                speech = speech.mean(axis=1)
            if rate != SAMPLE_RATE:
                import torchaudio.functional as AF
                import torch
                speech = AF.resample(torch.from_numpy(speech), rate, SAMPLE_RATE).numpy()
            speech = _normalize(speech, -22.0)
            before = _normalize(_active_from_event(before_source), -23.0)
            after = _normalize(_active_from_event(after_source), -23.0)
            overlap_raw = _active_from_event(overlap_source)
            horn_wave = _read_mono(_resolve(str(horn["audio_path"])))
            h0 = max(0, int(round(float(horn.get("active_onset_seconds", 0.0)) * SAMPLE_RATE)))
            h1 = min(len(horn_wave), int(round(float(horn.get("active_offset_seconds", len(horn_wave) / SAMPLE_RATE)) * SAMPLE_RATE)))
            horn_wave = _max_energy_crop(horn_wave[h0:h1] if h1 > h0 else horn_wave, 1.8)

            difficulty, snr = DIFFICULTIES[(len(scenes)) % len(DIFFICULTIES)]
            speech_rms = _rms(speech)
            combined_noise_rms = speech_rms / (10.0 ** (snr / 20.0))
            overlap = _scale_rms(overlap_raw, combined_noise_rms / math.sqrt(2.0))
            horn_wave = _scale_rms(horn_wave, combined_noise_rms / math.sqrt(2.0))

            scene_id = f"vi_speech_{split}_{local_index:03d}"
            before_onset = 0.20
            speech_onset = before_onset + len(before) / SAMPLE_RATE + 0.12
            speech_offset = speech_onset + len(speech) / SAMPLE_RATE
            after_onset = speech_offset + 0.12
            total_seconds = after_onset + len(after) / SAMPLE_RATE + 0.20
            placements = [
                ("noise_before", before_onset, before, before_source, "sound_event"),
                ("speech_1", speech_onset, speech, source, "speech"),
                ("noise_overlap", speech_onset + 0.30 * len(speech) / SAMPLE_RATE, overlap, overlap_source, "sound_event"),
                ("vehicle_horn_overlap", speech_onset + 0.55 * len(speech) / SAMPLE_RATE, horn_wave, horn, "sound_event"),
                ("noise_after", after_onset, after, after_source, "sound_event"),
            ]
            frames = int(math.ceil(total_seconds * SAMPLE_RATE))
            stems: list[np.ndarray] = []
            events: list[dict[str, Any]] = []
            for event_index, (role, onset, wave, meta, kind) in enumerate(placements):
                start = int(round(onset * SAMPLE_RATE))
                end = min(frames, start + len(wave))
                stem = np.zeros(frames, dtype=np.float32)
                stem[start:end] = wave[:end - start]
                stems.append(stem)
                label = "Speech" if kind == "speech" else str(meta["label"])
                event_id = f"{scene_id}_e{event_index:02d}"
                events.append({
                    "event_id": event_id, "event_kind": kind, "role": role, "label": label,
                    "display_name": VI_LABEL.get(label, label.replace("_", " ")),
                    "onset_seconds": start / SAMPLE_RATE, "offset_seconds": end / SAMPLE_RATE,
                    "source_id": str(source["id"] if kind == "speech" else meta["source_id"]),
                    "speaker_id": f"fleurs_vi_{source['id']}" if kind == "speech" else None,
                    "speaker_group": ("male" if int(source["gender"]) == 0 else "female") if kind == "speech" else None,
                    "transcript": str(source["transcription"]) if kind == "speech" else None,
                })
            mixture = np.sum(np.stack(stems), axis=0, dtype=np.float32)
            peak = max(float(np.max(np.abs(mixture))), *(float(np.max(np.abs(x))) for x in stems))
            scale = min(1.0, 0.97 / max(peak, 1e-9))
            mixture *= scale
            stems = [value * scale for value in stems]
            mix_path = output / "audio/mixtures" / f"{scene_id}.flac"
            _write_audio(mix_path, mixture)
            for event, stem in zip(events, stems):
                stem_path = output / "audio/stems" / scene_id / f"{event['event_id']}.flac"
                _write_audio(stem_path, stem)
                event["stem_path"] = _portable(stem_path)
            decoded, _ = sf.read(mix_path, dtype="float32")
            summed = sum((sf.read(_resolve(event["stem_path"]), dtype="float32")[0] for event in events), np.zeros_like(decoded))
            reconstruction.append(_rms(decoded - summed))
            scene = {
                "format": "qces_vietnamese_speech_scene_v1", "scene_id": scene_id, "split": split,
                "mixture_path": _portable(mix_path), "sample_rate": SAMPLE_RATE,
                "duration_seconds": len(mixture) / SAMPLE_RATE, "speaker_count": 1,
                "speaker_ids": [f"fleurs_vi_{source['id']}"],
                "speaker_groups": [events[1]["speaker_group"]], "language": "vi-VN",
                "difficulty": difficulty, "requested_speech_to_overlap_noise_snr_db": snr,
                "events": events,
            }
            scenes.append(scene)
            n0, s0, o0, horn0, n1 = events
            transcript = str(s0["transcript"])
            questions.extend([
                _question(scene, 0, "Người nói đã nói gì?", "vi_speech_content", "transcript", transcript, [], [s0["event_id"]], [s0["event_id"]]),
                _question(scene, 1, "Người đó nói gì trong khi tiếng còi xe tải đang phát?", "vi_speech_content_during_horn", "transcript", transcript, [horn0["event_id"]], [s0["event_id"]], [horn0["event_id"], s0["event_id"]]),
                _question(scene, 2, "Âm thanh nào xảy ra ngay trước khi người đó bắt đầu nói?", "vi_event_before_speech", "event_label", n0["display_name"], [s0["event_id"]], [n0["event_id"]], [n0["event_id"], s0["event_id"]]),
                _question(scene, 3, "Âm thanh nào xảy ra ngay sau khi người đó nói xong?", "vi_event_after_speech", "event_label", n1["display_name"], [s0["event_id"]], [n1["event_id"]], [s0["event_id"], n1["event_id"]]),
                _question(scene, 4, f"Người đó nói gì trong khi {o0['display_name']} chồng lên lời nói?", "vi_speech_content_during_event", "transcript", transcript, [o0["event_id"]], [s0["event_id"]], [o0["event_id"], s0["event_id"]]),
                _question(scene, 5, "Người nói thứ hai đã nói gì?", "vi_second_speaker_absent", "no_evidence", "Không có bằng chứng", [], [], [s0["event_id"]], no_evidence=True),
            ])

    scene_path, question_path = output / "scenes.jsonl", output / "questions.jsonl"
    _atomic_text(scene_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scenes))
    _atomic_text(question_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in questions))
    all_sources: dict[str, set[str]] = collections.defaultdict(set)
    for scene in scenes:
        for event in scene["events"]:
            all_sources[event["source_id"]].add(scene["split"])
    leakage = [key for key, splits in all_sources.items() if len(splits) > 1]
    receipt = {
        "format": "qces_vietnamese_speech_dataset_receipt_v1", "complete": not leakage,
        "source_dataset": "Google FLEURS vi_vn validation (CC-BY-4.0)", "seed": args.seed,
        "scene_count": len(scenes), "question_count": len(questions),
        "questions_per_scene": 6, "scenes_by_split": dict(collections.Counter(x["split"] for x in scenes)),
        "difficulty_counts": dict(collections.Counter(x["difficulty"] for x in scenes)),
        "requested_snr_db_values": sorted({x["requested_speech_to_overlap_noise_snr_db"] for x in scenes}),
        "single_speaker": True, "language": "vi-VN", "noise_class_count": 11,
        "noise_classes": sorted({e["label"] for s in scenes for e in s["events"] if e["event_kind"] == "sound_event"}),
        "unique_speech_utterances": required,
        "unique_noise_sources": len({e["source_id"] for s in scenes for e in s["events"] if e["event_kind"] == "sound_event"}),
        "cross_split_source_leakage_count": len(leakage), "maximum_reconstruction_rms": max(reconstruction),
        "scene_manifest": _portable(scene_path), "question_manifest": _portable(question_path),
        "scene_manifest_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "question_manifest_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output / "dataset_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if leakage:
        raise RuntimeError(f"source leakage: {leakage[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
