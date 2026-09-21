#!/usr/bin/env python3
"""Build a small long-form real-speech automotive-noise demo.

Speech comes from the local FLEURS Vietnamese validation recording set.  The
road/engine/horn/siren layers come from the existing AudioSet-derived bank.
This is a controlled demo, not a real cabin recording benchmark.
"""

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


SCENES = 6
SPLITS = ("val", "test")
DIFFICULTIES = (("easy", 3.0), ("medium", -3.0), ("hard", -10.0))


def _scale(waveform: np.ndarray, target_rms: float) -> np.ndarray:
    return np.asarray(waveform, dtype=np.float32) * (target_rms / max(_rms(waveform), 1e-7))


def _loop(waveform: np.ndarray, length: int) -> np.ndarray:
    if len(waveform) == 0:
        return np.zeros(length, dtype=np.float32)
    repeats = int(math.ceil(length / len(waveform)))
    return np.tile(waveform, repeats)[:length].astype(np.float32)


def _fleurs(path: Path, rng: random.Random, count: int) -> list[dict[str, Any]]:
    rows = pq.read_table(path, columns=["id", "num_samples", "audio", "transcription", "gender"]).to_pylist()
    rows = [
        row for row in rows
        if 2.2 <= int(row["num_samples"]) / SAMPLE_RATE <= 7.0
        and 6 <= len(str(row["transcription"]).split()) <= 40
        and row["audio"]["bytes"]
    ]
    rng.shuffle(rows)
    if len(rows) < count:
        raise RuntimeError(f"need {count} FLEURS utterances, got {len(rows)}")
    return rows[:count]


def _speech_wave(row: dict[str, Any]) -> np.ndarray:
    waveform, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1, dtype=np.float32)
    if rate != SAMPLE_RATE:
        import torchaudio.functional as AF
        import torch

        waveform = AF.resample(torch.from_numpy(waveform), rate, SAMPLE_RATE).numpy()
    return _normalize(waveform, -22.0)


def _active(event: dict[str, Any], maximum_seconds: float = 2.0) -> np.ndarray:
    waveform = _read_mono(_resolve(str(event["stem_path"])))
    start = max(0, int(float(event["onset_seconds"]) * SAMPLE_RATE))
    end = min(len(waveform), int(float(event["offset_seconds"]) * SAMPLE_RATE))
    return _max_energy_crop(waveform[start:end] if end > start else waveform, maximum_seconds)


def _question(
    scene: dict[str, Any],
    index: int,
    text: str,
    operation: str,
    answer_type: str,
    answer: str,
    answer_ids: list[str],
    evidence_ids: list[str],
    no_evidence: bool = False,
) -> dict[str, Any]:
    return {
        "format": "qces_real_car_longform_question_v1",
        "question_id": f"{scene['scene_id']}_q{index:02d}",
        "scene_id": scene["scene_id"],
        "split": scene["split"],
        "mixture_path": scene["mixture_path"],
        "question": text,
        "operation": operation,
        "answer_type": answer_type,
        "answer": answer,
        "answer_event_ids": answer_ids,
        "evidence_event_ids": evidence_ids,
        "no_evidence": no_evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleurs-parquet", type=Path, default=PROJECT_ROOT / "data/_sources/fleurs_vi_vn/validation.parquet")
    parser.add_argument("--noise-dataset", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/qces_real_car_longform_v1")
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output = args.output_dir.resolve()
    speech_rows = _fleurs(args.fleurs_parquet.resolve(), rng, SCENES * 4)
    donor_scenes = _jsonl((args.noise_dataset / "scenes.jsonl").resolve())
    donors = [scene for scene in donor_scenes if scene["split"] in SPLITS]
    if len(donors) < SCENES:
        raise RuntimeError("not enough automotive donor scenes")

    scenes: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    cursor = 0
    reconstruction_errors: list[float] = []
    measured_snrs: list[float] = []
    for index in range(SCENES):
        split = "val" if index < SCENES // 2 else "test"
        difficulty, requested_snr = DIFFICULTIES[index % len(DIFFICULTIES)]
        scene_id = f"real_car_longform_{split}_{index:02d}"
        utterances = speech_rows[cursor : cursor + 4]
        cursor += 4
        speech_parts = [_speech_wave(row) for row in utterances]
        speech = np.concatenate(
            [part if part_index == 0 else np.concatenate([np.zeros(int(0.18 * SAMPLE_RATE), dtype=np.float32), part]) for part_index, part in enumerate(speech_parts)]
        )
        donor = donors[index]
        donor_events = {str(event["label"]): event for event in donor["events"] if event["event_kind"] == "sound_event"}
        road_event = next((event for label, event in donor_events.items() if "Traffic" in label), None)
        engine_event = next((event for label, event in donor_events.items() if "Heavy_engine" in label), None)
        horn_event = next((event for label, event in donor_events.items() if "horn" in label.lower()), None)
        siren_event = next((event for label, event in donor_events.items() if "siren" in label.lower()), None)
        fallback = [event for event in donor["events"] if event["event_kind"] == "sound_event"]
        road_event = road_event or fallback[0]
        engine_event = engine_event or fallback[1 % len(fallback)]
        horn_event = horn_event or fallback[2 % len(fallback)]
        siren_event = siren_event or fallback[3 % len(fallback)]
        speech_onset = int(0.7 * SAMPLE_RATE)
        total_frames = speech_onset + len(speech) + int(0.8 * SAMPLE_RATE)
        speech_full = np.zeros(total_frames, dtype=np.float32)
        speech_full[speech_onset : speech_onset + len(speech)] = speech
        speech_rms = _rms(speech)
        noise_target = speech_rms / (10.0 ** (requested_snr / 20.0))
        road = _scale(_loop(_active(road_event, 5.0), total_frames), noise_target * 0.65)
        engine = _scale(_loop(_active(engine_event, 5.0), total_frames), noise_target * 0.45)
        horn = _scale(_loop(_active(horn_event, 1.5), int(1.2 * SAMPLE_RATE)), noise_target * 0.9)
        siren = _scale(_loop(_active(siren_event, 1.5), int(1.8 * SAMPLE_RATE)), noise_target * 0.7)
        placements = [
            ("speech_1", speech_onset, speech_full, None),
            ("continuous_traffic", 0, road, road_event),
            ("continuous_engine", 0, engine, engine_event),
            ("truck_horn_overlap", speech_onset + int(0.38 * len(speech)), horn, horn_event),
            ("police_siren_overlap", speech_onset + int(0.68 * len(speech)), siren, siren_event),
        ]
        stems: list[np.ndarray] = []
        events: list[dict[str, Any]] = []
        for event_index, (role, onset, waveform, source) in enumerate(placements):
            start = int(onset)
            end = min(total_frames, start + len(waveform))
            stem = np.zeros(total_frames, dtype=np.float32)
            stem[start:end] = waveform[: end - start]
            stems.append(stem)
            if role == "speech_1":
                label, kind, display = "Speech", "speech", "human speech"
                source_id, speaker_group, speaker_id, transcript = f"fleurs_{utterances[0]['id']}", "real_speech", f"fleurs_{utterances[0]['id']}", " ".join(str(row["transcription"]) for row in utterances)
            else:
                label, kind, display = str(source["label"]), "sound_event", str(source.get("display_name") or source["label"])
                source_id, speaker_group, speaker_id, transcript = str(source["source_id"]), None, None, None
            event_id = f"{scene_id}_e{event_index:02d}"
            events.append({
                "event_id": event_id, "event_kind": kind, "role": role, "label": label,
                "display_name": display, "onset_seconds": start / SAMPLE_RATE,
                "offset_seconds": end / SAMPLE_RATE, "source_id": source_id,
                "speaker_group": speaker_group, "speaker_id": speaker_id, "transcript": transcript,
            })
        mixture = np.sum(np.stack(stems), axis=0, dtype=np.float32)
        peak = max(float(np.max(np.abs(mixture))), *(float(np.max(np.abs(stem))) for stem in stems))
        scale = min(1.0, 0.97 / max(peak, 1e-9))
        mixture *= scale
        stems = [stem * scale for stem in stems]
        mixture_path = output / "audio/mixtures" / f"{scene_id}.flac"
        _write_audio(mixture_path, mixture)
        for event, stem in zip(events, stems):
            stem_path = output / "audio/stems" / scene_id / f"{event['event_id']}.flac"
            _write_audio(stem_path, stem)
            event["stem_path"] = _portable(stem_path)
        decoded = _read_mono(mixture_path)
        summed = np.sum(np.stack([_read_mono(_resolve(event["stem_path"])) for event in events]), axis=0)
        reconstruction_errors.append(_rms(decoded - summed))
        overlap_mask = np.abs(stems[3] + stems[4]) > 1e-7
        measured_snrs.append(10.0 * math.log10((float(np.mean(np.square(stems[0][overlap_mask], dtype=np.float64))) + 1e-12) / (float(np.mean(np.square((stems[3] + stems[4])[overlap_mask], dtype=np.float64))) + 1e-12)))
        scene = {
            "format": "qces_real_car_longform_scene_v1", "scene_id": scene_id, "split": split,
            "mixture_path": _portable(mixture_path), "sample_rate": SAMPLE_RATE,
            "duration_seconds": len(mixture) / SAMPLE_RATE, "speaker_count": 1,
            "difficulty": difficulty, "requested_speech_to_noise_snr_db": requested_snr,
            "events": events, "source_utterance_ids": [str(row["id"]) for row in utterances],
        }
        scenes.append(scene)
        speech_id = events[0]["event_id"]
        traffic_id, engine_id, horn_id, siren_id = (event["event_id"] for event in events[1:])
        transcript = " ".join(str(row["transcription"]) for row in utterances)
        questions.extend(
            [
                _question(scene, 0, "What did the person say in the car?", "speech_transcription", "transcript", transcript, [speech_id], [speech_id]),
                _question(scene, 1, "What vehicle sound overlaps the person's speech?", "event_overlap_speech", "event_label", events[3]["display_name"], [horn_id], [speech_id, horn_id]),
                _question(scene, 2, "What emergency sound overlaps the person's speech?", "event_overlap_speech", "event_label", events[4]["display_name"], [siren_id], [speech_id, siren_id]),
                _question(scene, 3, "What sound is already present before the person starts speaking?", "event_before_speech", "event_label", events[1]["display_name"], [traffic_id], [traffic_id, speech_id]),
                _question(scene, 4, "What sound continues after the person stops speaking?", "event_after_speech", "event_label", events[2]["display_name"], [engine_id], [speech_id, engine_id]),
                _question(scene, 5, "Did a second person speak in the car?", "speaker_absent", "no_evidence", "No evidence", [], [], True),
            ]
        )

    scene_path, question_path = output / "scenes.jsonl", output / "questions.jsonl"
    _atomic_text(scene_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scenes))
    _atomic_text(question_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in questions))
    receipt = {
        "format": "qces_real_car_longform_dataset_receipt_v1", "complete": True, "source_dataset": "Google FLEURS vi_vn validation recordings",
        "real_speech": True, "scene_count": len(scenes), "question_count": len(questions),
        "questions_per_scene": len(questions) // max(len(scenes), 1),
        "scenes_by_split": dict(collections.Counter(scene["split"] for scene in scenes)),
        "difficulty_counts": dict(collections.Counter(scene["difficulty"] for scene in scenes)),
        "requested_snr_db_values": sorted({scene["requested_speech_to_noise_snr_db"] for scene in scenes}),
        "duration_seconds": {scene["scene_id"]: scene["duration_seconds"] for scene in scenes},
        "mean_measured_speech_to_noise_snr_db": float(np.mean(measured_snrs)),
        "maximum_reconstruction_rms": max(reconstruction_errors),
        "scene_manifest": _portable(scene_path), "question_manifest": _portable(question_path),
        "scene_manifest_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "question_manifest_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output / "dataset_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
