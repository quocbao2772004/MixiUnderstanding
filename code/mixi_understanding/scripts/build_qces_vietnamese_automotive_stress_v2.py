#!/usr/bin/env python3
"""Build paired Vietnamese speech scenes under severe automotive interference."""

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
from mixi_understanding.scripts.build_qces_vietnamese_speech_demo import _question, _scale_rms


STRESS_TIERS = (
    {"name": "loud", "horn": -5.0, "siren": 0.0, "traffic": 0.0, "engine": 3.0},
    {"name": "severe", "horn": -10.0, "siren": -5.0, "traffic": -3.0, "engine": 0.0},
    {"name": "extreme", "horn": -15.0, "siren": -10.0, "traffic": -6.0, "engine": -3.0},
)
LABELS = (
    "Traffic_noise_and_roadway_noise", "Heavy_engine_(low_frequency)",
    "Air_horn_and_truck_horn", "Police_car_(siren)",
    "Reversing_beeps", "Engine_starting",
)
VI_LABEL = {
    "Traffic_noise_and_roadway_noise": "tiếng giao thông trên đường",
    "Heavy_engine_(low_frequency)": "tiếng động cơ hạng nặng",
    "Air_horn_and_truck_horn": "tiếng còi xe tải",
    "Police_car_(siren)": "tiếng còi xe cảnh sát",
    "Reversing_beeps": "tiếng bíp lùi xe",
    "Engine_starting": "tiếng động cơ khởi động",
    "Speech": "lời nói tiếng Việt",
}


def _pool(root: Path) -> dict[str, list[dict[str, Any]]]:
    values: dict[str, dict[str, dict[str, Any]]] = {label: {} for label in LABELS}
    for path in root.glob("**/source_bank.jsonl"):
        for row in _jsonl(path):
            label = str(row.get("label") or "")
            if label in values and _resolve(str(row["audio_path"])).is_file():
                values[label][str(row["source_id"])] = row
    return {label: list(rows.values()) for label, rows in values.items()}


def _active(row: dict[str, Any], maximum: float) -> np.ndarray:
    wave = _read_mono(_resolve(str(row["audio_path"])))
    start = max(0, int(round(float(row.get("active_onset_seconds", 0.0)) * SAMPLE_RATE)))
    end = min(len(wave), int(round(float(row.get("active_offset_seconds", len(wave) / SAMPLE_RATE)) * SAMPLE_RATE)))
    return _max_energy_crop(wave[start:end] if end > start else wave, maximum)


def _fit_continuous(wave: np.ndarray, frames: int) -> np.ndarray:
    if len(wave) >= frames:
        return _max_energy_crop(wave, frames / SAMPLE_RATE)[:frames]
    repeats = int(math.ceil(frames / max(len(wave), 1)))
    return np.tile(wave, repeats)[:frames].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-dataset", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_speech_event_demo_v1")
    parser.add_argument("--source-bank-root", type=Path, default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1/primary_clean")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/qces_vietnamese_automotive_stress_v2")
    parser.add_argument("--seed", type=int, default=2026080517)
    parser.add_argument(
        "--speech-snr-shift-db",
        type=float,
        default=0.0,
        help="Raise speech relative to every noise component without changing source/layout selection.",
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output = args.output_dir.resolve()
    paired_root = args.paired_dataset.resolve()
    paired = _jsonl(paired_root / "scenes.jsonl")
    paired_receipt_path = paired_root / "dataset_receipt.json"
    paired_receipt = json.loads(paired_receipt_path.read_text(encoding="utf-8")) if paired_receipt_path.is_file() else {}
    pools = _pool(args.source_bank_root.resolve())
    count = len(paired)
    selected: dict[str, list[dict[str, Any]]] = {}
    for label in LABELS:
        rng.shuffle(pools[label])
        if len(pools[label]) < count:
            raise RuntimeError(f"{label}: need {count}, got {len(pools[label])}")
        selected[label] = pools[label][:count]

    scenes, questions, reconstruction, measured_snrs = [], [], [], []
    for index, base in enumerate(paired):
        base_tier = STRESS_TIERS[index % len(STRESS_TIERS)]
        tier = {
            key: (value + args.speech_snr_shift_db if key != "name" else value)
            for key, value in base_tier.items()
        }
        base_speech = next(event for event in base["events"] if event["label"] == "Speech")
        aligned_speech = _read_mono(_resolve(str(base_speech["stem_path"])))
        s0 = int(round(float(base_speech["onset_seconds"]) * SAMPLE_RATE))
        s1 = int(round(float(base_speech["offset_seconds"]) * SAMPLE_RATE))
        speech = _normalize(aligned_speech[s0:s1], -22.0)
        speech_rms = _rms(speech)
        sources = {label: selected[label][index] for label in LABELS}

        reverse = _scale_rms(_active(sources["Reversing_beeps"], .85), speech_rms / (10 ** (2.0 / 20)))
        after = _scale_rms(_active(sources["Engine_starting"], 1.2), speech_rms / (10 ** (0.0 / 20)))
        speech_onset = 1.20
        speech_offset = speech_onset + len(speech) / SAMPLE_RATE
        after_onset = speech_offset + .10
        duration = after_onset + len(after) / SAMPLE_RATE + .20
        frames = int(math.ceil(duration * SAMPLE_RATE))

        traffic = _fit_continuous(_active(sources["Traffic_noise_and_roadway_noise"], 10.0), frames)
        engine = _fit_continuous(_active(sources["Heavy_engine_(low_frequency)"], 10.0), frames)
        traffic = _scale_rms(traffic, speech_rms / (10 ** (tier["traffic"] / 20)))
        engine = _scale_rms(engine, speech_rms / (10 ** (tier["engine"] / 20)))
        horn = _active(sources["Air_horn_and_truck_horn"], 2.0)
        siren = _active(sources["Police_car_(siren)"], min(3.5, len(speech) / SAMPLE_RATE * .65))
        horn = _scale_rms(horn, speech_rms / (10 ** (tier["horn"] / 20)))
        siren = _scale_rms(siren, speech_rms / (10 ** (tier["siren"] / 20)))
        horn_onset = speech_onset + .20 * len(speech) / SAMPLE_RATE
        siren_onset = speech_onset + .38 * len(speech) / SAMPLE_RATE
        reverse_onset = max(.12, speech_onset - len(reverse) / SAMPLE_RATE - .08)
        placements = [
            ("continuous_traffic", 0.0, traffic, sources["Traffic_noise_and_roadway_noise"], "sound_event"),
            ("continuous_heavy_engine", 0.0, engine, sources["Heavy_engine_(low_frequency)"], "sound_event"),
            ("noise_before", reverse_onset, reverse, sources["Reversing_beeps"], "sound_event"),
            ("speech_1", speech_onset, speech, base_speech, "speech"),
            ("truck_horn_overlap", horn_onset, horn, sources["Air_horn_and_truck_horn"], "sound_event"),
            ("police_siren_overlap", siren_onset, siren, sources["Police_car_(siren)"], "sound_event"),
            ("noise_after", after_onset, after, sources["Engine_starting"], "sound_event"),
        ]
        scene_id = f"vi_auto_stress_{base['split']}_{int(base['scene_id'].split('_')[-1]):03d}"
        stems, events = [], []
        for event_index, (role, onset, wave, meta, kind) in enumerate(placements):
            start = int(round(onset * SAMPLE_RATE))
            end = min(frames, start + len(wave))
            stem = np.zeros(frames, dtype=np.float32)
            stem[start:end] = wave[:end-start]
            stems.append(stem)
            label = "Speech" if kind == "speech" else str(meta["label"])
            source_id = str(base_speech["source_id"] if kind == "speech" else meta["source_id"])
            event_id = f"{scene_id}_e{event_index:02d}"
            events.append({
                "event_id": event_id, "event_kind": kind, "role": role, "label": label,
                "display_name": VI_LABEL[label], "onset_seconds": start / SAMPLE_RATE,
                "offset_seconds": end / SAMPLE_RATE, "source_id": source_id,
                "speaker_id": base_speech.get("speaker_id") if kind == "speech" else None,
                "speaker_group": base_speech.get("speaker_group") if kind == "speech" else None,
                "transcript": base_speech.get("transcript") if kind == "speech" else None,
            })
        mixture = np.sum(np.stack(stems), axis=0, dtype=np.float32)
        peak = max(float(np.max(np.abs(mixture))), *(float(np.max(np.abs(x))) for x in stems))
        scale = min(1.0, .97 / max(peak, 1e-9))
        mixture *= scale
        stems = [x * scale for x in stems]
        mix_path = output / "audio/mixtures" / f"{scene_id}.flac"
        _write_audio(mix_path, mixture)
        for event, stem in zip(events, stems):
            path = output / "audio/stems" / scene_id / f"{event['event_id']}.flac"
            _write_audio(path, stem)
            event["stem_path"] = _portable(path)
        speech_stem = stems[3]
        interference = sum((x for i, x in enumerate(stems) if i != 3), np.zeros(frames, dtype=np.float32))
        active = np.abs(speech_stem) > 1e-7
        measured = 10 * math.log10((float(np.mean(speech_stem[active] ** 2)) + 1e-12) / (float(np.mean(interference[active] ** 2)) + 1e-12))
        measured_snrs.append(measured)
        decoded, _ = sf.read(mix_path, dtype="float32")
        summed = sum((sf.read(_resolve(e["stem_path"]), dtype="float32")[0] for e in events), np.zeros_like(decoded))
        reconstruction.append(_rms(decoded - summed))
        scene = {
            "format": "qces_vietnamese_automotive_stress_scene_v2", "scene_id": scene_id,
            "paired_scene_id": base["scene_id"], "split": base["split"], "mixture_path": _portable(mix_path),
            "sample_rate": SAMPLE_RATE, "duration_seconds": frames / SAMPLE_RATE, "speaker_count": 1,
            "speaker_ids": base["speaker_ids"], "speaker_groups": base["speaker_groups"], "language": "vi-VN",
            "difficulty": f"automotive_{tier['name']}", "requested_speech_to_horn_snr_db": tier["horn"],
            "requested_speech_to_overlap_noise_snr_db": tier["horn"],
            "measured_speech_to_all_interference_snr_db": measured, "events": events,
        }
        scenes.append(scene)
        traffic_e, engine_e, reverse_e, speech_e, horn_e, siren_e, after_e = events
        transcript = str(speech_e["transcript"])
        questions.extend([
            _question(scene, 0, "Người nói đã nói gì?", "vi_speech_content", "transcript", transcript, [], [speech_e["event_id"]], [speech_e["event_id"]]),
            _question(scene, 1, "Người đó nói gì trong khi tiếng còi xe tải đang phát?", "vi_speech_content_during_horn", "transcript", transcript, [horn_e["event_id"]], [speech_e["event_id"]], [horn_e["event_id"], speech_e["event_id"]]),
            _question(scene, 2, "Người đó nói gì trong khi tiếng còi xe tải và còi xe cảnh sát cùng phát?", "vi_speech_content_during_horn", "transcript", transcript, [horn_e["event_id"], siren_e["event_id"]], [speech_e["event_id"]], [horn_e["event_id"], siren_e["event_id"], speech_e["event_id"]]),
            _question(scene, 3, "Âm thanh nào xảy ra ngay trước khi người đó bắt đầu nói?", "vi_event_before_speech", "event_label", reverse_e["display_name"], [speech_e["event_id"]], [reverse_e["event_id"]], [reverse_e["event_id"], speech_e["event_id"]]),
            _question(scene, 4, "Âm thanh nào xảy ra ngay sau khi người đó nói xong?", "vi_event_after_speech", "event_label", after_e["display_name"], [speech_e["event_id"]], [after_e["event_id"]], [speech_e["event_id"], after_e["event_id"]]),
            _question(scene, 5, "Người nói thứ hai đã nói gì?", "vi_second_speaker_absent", "no_evidence", "Không có bằng chứng", [], [], [speech_e["event_id"]], no_evidence=True),
        ])

    scene_path, question_path = output / "scenes.jsonl", output / "questions.jsonl"
    _atomic_text(scene_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in scenes))
    _atomic_text(question_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in questions))
    split_sources: dict[str, set[str]] = collections.defaultdict(set)
    for scene in scenes:
        for event in scene["events"]:
            split_sources[event["source_id"]].add(scene["split"])
    leakage = [x for x, splits in split_sources.items() if len(splits) > 1]
    receipt = {
        "format": "qces_vietnamese_automotive_stress_dataset_receipt_v2", "complete": not leakage,
        "source_dataset": (
            f"paired speech ({paired_receipt.get('source_dataset') or paired_receipt.get('tts_model_id') or paired_root.name}) "
            "+ AudioSet-derived automotive source bank"
        ),
        "speech_is_synthetic": bool(paired_receipt.get("synthetic_speech", False)),
        "speech_tts_model_id": paired_receipt.get("tts_model_id"),
        "paired_dataset": _portable(args.paired_dataset.resolve()), "scene_count": len(scenes),
        "question_count": len(questions), "questions_per_scene": 6,
        "scenes_by_split": dict(collections.Counter(x["split"] for x in scenes)),
        "difficulty_counts": dict(collections.Counter(x["difficulty"] for x in scenes)),
        "speech_snr_shift_db": args.speech_snr_shift_db,
        "requested_snr_db_values": sorted({x["requested_speech_to_horn_snr_db"] for x in scenes}),
        "mean_measured_speech_to_all_interference_snr_db": float(np.mean(measured_snrs)),
        "min_measured_speech_to_all_interference_snr_db": float(np.min(measured_snrs)),
        "single_speaker": True, "language": "vi-VN", "noise_class_count": len(LABELS),
        "noise_classes": list(LABELS), "unique_speech_utterances": len(scenes),
        "unique_noise_sources": len({e["source_id"] for s in scenes for e in s["events"] if e["event_kind"] == "sound_event"}),
        "cross_split_source_leakage_count": len(leakage), "maximum_reconstruction_rms": max(reconstruction),
        "scene_manifest": _portable(scene_path), "question_manifest": _portable(question_path),
        "scene_manifest_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "question_manifest_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output / "dataset_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if leakage:
        raise RuntimeError(f"cross-split leakage: {leakage[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
