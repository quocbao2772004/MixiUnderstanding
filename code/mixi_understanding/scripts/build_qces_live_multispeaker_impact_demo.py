#!/usr/bin/env python3
"""Build one reproducible two-speaker + impact-sound live-demo recording."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = PROJECT_ROOT / "outputs/qces_live_demo_multispeaker_impacts_v2"
SAMPLE_RATE = 16_000
DURATION_SECONDS = 10.0

SOURCES = {
    "female": PROJECT_ROOT
    / "data/qces_vietnamese_vieneu_source_v3/audio/clean/vi_vieneu_test_021.flac",
    "male": PROJECT_ROOT
    / "data/qces_vietnamese_vieneu_source_v3/audio/clean/vi_vieneu_test_018.flac",
    "clapping": PROJECT_ROOT
    / "outputs/qces_full200_adaptive_v1/primary_clean/train/"
    "train-enyoukai_audioset_strong_pinned_parquet_v1-0010/audio/train/Clapping/"
    "55bfb852b73a9a5f73140d6df59910ff.stem.flac",
    "table_knock": PROJECT_ROOT
    / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1/audio/train/Knock/256512.wav",
    "chair_thump": PROJECT_ROOT
    / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1/audio/train/Thump_and_thud/349267.wav",
}


def load(path: Path, start: float = 0.0, end: float | None = None) -> np.ndarray:
    value, rate = sf.read(path, dtype="float32", always_2d=True)
    value = value.mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(int(rate), SAMPLE_RATE)
        value = resample_poly(value, SAMPLE_RATE // divisor, int(rate) // divisor)
    left = max(0, int(round(start * SAMPLE_RATE)))
    right = len(value) if end is None else min(len(value), int(round(end * SAMPLE_RATE)))
    return np.asarray(value[left:right], dtype=np.float32)


def rms_db(value: np.ndarray) -> float:
    return 20.0 * math.log10(float(np.sqrt(np.mean(value.astype(np.float64) ** 2) + 1e-12)))


def gain_to(value: np.ndarray, target_db: float) -> np.ndarray:
    if not len(value):
        return value
    gain = 10.0 ** ((target_db - rms_db(value)) / 20.0)
    output = np.asarray(value * min(gain, 12.0), dtype=np.float32)
    # Preserve impact dynamics while preventing one source transient from
    # turning down every speech stem at the final master stage.
    peak = float(np.max(np.abs(output)))
    if peak > 0.72:
        output *= 0.72 / peak
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing sources: " + ", ".join(missing))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem_root = OUTPUT / "stems"
    stem_root.mkdir(parents=True, exist_ok=True)
    total_samples = int(DURATION_SECONDS * SAMPLE_RATE)
    mixture = np.zeros(total_samples, dtype=np.float32)
    timeline: list[dict[str, Any]] = []

    recipe = [
        {
            "name": "female_speech",
            "label": "Speech_female",
            "display": "người phụ nữ nói",
            "source": SOURCES["female"],
            "source_start": 0.0,
            "source_end": None,
            "onset": 0.25,
            "target_rms_dbfs": -21.0,
            "transcript": "Có người đi bộ phía trước, chú ý phanh xe nhé.",
        },
        {
            "name": "clapping",
            "label": "Clapping",
            "display": "tiếng vỗ tay",
            "source": SOURCES["clapping"],
            "source_start": 0.0,
            "source_end": None,
            "onset": 2.30,
            "target_rms_dbfs": -15.0,
            "transcript": None,
        },
        {
            "name": "male_speech",
            "label": "Speech_male",
            "display": "người đàn ông nói",
            "source": SOURCES["male"],
            "source_start": 0.0,
            "source_end": None,
            "onset": 3.55,
            "target_rms_dbfs": -21.0,
            "transcript": "Tôi đang đứng cạnh cột số năm ở ga quốc nội.",
        },
        {
            "name": "table_knock",
            "label": "Knock",
            "display": "tiếng gõ/đập bàn",
            "source": SOURCES["table_knock"],
            "source_start": 0.50,
            "source_end": 1.75,
            "onset": 4.80,
            "target_rms_dbfs": -14.0,
            "transcript": None,
        },
        {
            "name": "chair_thump",
            "label": "Thump_and_thud",
            "display": "tiếng va đập trầm/đập ghế",
            "source": SOURCES["chair_thump"],
            "source_start": 0.50,
            "source_end": 1.75,
            "onset": 7.50,
            "target_rms_dbfs": -13.0,
            "transcript": None,
        },
    ]
    for event_id, item in enumerate(recipe):
        audio = load(item["source"], item["source_start"], item["source_end"])
        audio = gain_to(audio, float(item["target_rms_dbfs"]))
        onset_sample = int(round(float(item["onset"]) * SAMPLE_RATE))
        usable = min(len(audio), total_samples - onset_sample)
        audio = audio[:usable]
        stem = np.zeros(total_samples, dtype=np.float32)
        stem[onset_sample : onset_sample + usable] = audio
        mixture += stem
        stem_path = stem_root / f"{event_id:02d}_{item['name']}.flac"
        sf.write(stem_path, stem, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
        timeline.append(
            {
                "event_id": f"live_demo_e{event_id:02d}",
                "label": item["label"],
                "display_name": item["display"],
                "onset_seconds": float(item["onset"]),
                "offset_seconds": float(item["onset"]) + usable / SAMPLE_RATE,
                "transcript": item["transcript"],
                "target_rms_dbfs": item["target_rms_dbfs"],
                "source_path": str(item["source"].relative_to(PROJECT_ROOT)),
                "source_crop_seconds": [item["source_start"], item["source_end"]],
                "stem_path": str(stem_path.relative_to(PROJECT_ROOT)),
            }
        )
    peak_before = float(np.max(np.abs(mixture)))
    master_gain = min(1.0, 0.92 / max(peak_before, 1e-8))
    mixture *= master_gain
    mixture_path = OUTPUT / "multispeaker_clap_overlap_table_chair.wav"
    sf.write(mixture_path, mixture, SAMPLE_RATE, subtype="PCM_16")
    receipt = {
        "format": "qces_live_multispeaker_impact_demo_v2",
        "complete": True,
        "synthetic_mixture": True,
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": DURATION_SECONDS,
        "mixture_path": str(mixture_path.relative_to(PROJECT_ROOT)),
        "mixture_sha256": sha256(mixture_path),
        "peak_before_master": peak_before,
        "master_gain": master_gain,
        "events": timeline,
        "recommended_questions": [
            "Người phụ nữ đã nói gì?",
            "Người đàn ông đã nói gì?",
            "Tiếng vỗ tay chồng lên lời của ai?",
            "Âm thanh nào chồng lên lời người đàn ông?",
            "Tiếng đập bàn xuất hiện lúc nào?",
            "Âm thanh cuối cùng là gì?",
        ],
        "limitations": [
            "Knock and thump recordings are acoustic proxies for a table hit and chair impact.",
            "Live inference segments non-overlapping speech and predicts coarse speaker gender; "
            "it is not a full speaker-diarization or simultaneous-speaker separation system.",
        ],
    }
    receipt_path = OUTPUT / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
