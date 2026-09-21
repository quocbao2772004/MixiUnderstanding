#!/usr/bin/env python3
"""Build a short Vietnamese three-speaker movie conversation with sound events.

The output keeps every speech and background source as a time-aligned stem so
the final mixture can be audited by listening, not only from its labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from mixi_understanding.scripts.build_vieneu_automotive_speech_source import (
    MODEL_ID,
    MODEL_VERSION,
    _load_tts,
    _synthesise,
    _to_16k,
)


SAMPLE_RATE = 16_000
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DIALOGUE = (
    {
        "speaker": "Người nói 1",
        "voice": "Mai Anh",
        "gender": "female",
        "text": "Phim tối qua hay thật, cảnh cuối quá bất ngờ.",
        "onset": 0.25,
    },
    {
        "speaker": "Người nói 2",
        "voice": "Xuân Vĩnh",
        "gender": "male",
        "text": "Mình thích nhất phần âm nhạc và diễn xuất.",
        "onset": 2.65,
    },
    {
        "speaker": "Người nói 3",
        "voice": "Phạm Tuyên",
        "gender": "male",
        "text": "Đoạn kết hơi vội, nhưng phim vẫn đáng xem.",
        "onset": 5.05,
    },
)

EVENTS = (
    {
        "label": "Clapping",
        "display_name": "Tiếng vỗ tay",
        "source": PROJECT_ROOT
        / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1/audio/eval/Clapping/83509.wav",
        "source_onset": 0.50,
        "source_offset": 1.75,
        "onset": 0.75,
        "rms_db": -22.0,
        "attribution": (
            'Freesound ID 83509, "Clapping.WAV", uploaded by bsumusictech; '
            "https://freesound.org/s/83509/"
        ),
    },
    {
        "label": "Bark",
        "display_name": "Tiếng chó sủa",
        "source": PROJECT_ROOT
        / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1/audio/eval/Bark/89210.wav",
        "source_onset": 0.50,
        "source_offset": 1.75,
        "onset": 3.55,
        "rms_db": -20.0,
        "attribution": (
            'Freesound ID 89210, "BARKING 4 -S.WAV", uploaded by smokum; '
            "https://freesound.org/s/89210/"
        ),
    },
    {
        "label": "Meow",
        "display_name": "Tiếng mèo kêu",
        "source": PROJECT_ROOT
        / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1/audio/eval/Meow/85238.wav",
        "source_onset": 0.50,
        "source_offset": 1.75,
        "onset": 6.35,
        "rms_db": -19.0,
        "attribution": (
            'Freesound ID 85238, "cat pleads.wav", uploaded by cognito perceptu; '
            "https://freesound.org/s/85238/"
        ),
    },
)


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate == SAMPLE_RATE:
        return audio
    divisor = int(np.gcd(source_rate, SAMPLE_RATE))
    return np.asarray(
        resample_poly(audio, SAMPLE_RATE // divisor, source_rate // divisor),
        dtype=np.float32,
    )


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def _set_rms(audio: np.ndarray, dbfs: float) -> np.ndarray:
    current = _rms(audio)
    target = 10.0 ** (dbfs / 20.0)
    if current <= 1e-7:
        raise RuntimeError("Cannot normalize a silent source")
    return np.asarray(audio * (target / current), dtype=np.float32)


def _fade(audio: np.ndarray, seconds: float = 0.025) -> np.ndarray:
    result = np.asarray(audio, dtype=np.float32).copy()
    size = min(int(seconds * SAMPLE_RATE), len(result) // 2)
    if size > 0:
        ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)
        result[:size] *= ramp
        result[-size:] *= ramp[::-1]
    return result


def _aligned(source: np.ndarray, onset: float, samples: int) -> np.ndarray:
    stem = np.zeros(samples, dtype=np.float32)
    start = max(0, int(round(onset * SAMPLE_RATE)))
    end = min(samples, start + len(source))
    if end > start:
        stem[start:end] = source[: end - start]
    return stem


def _write(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")


def _jsonable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_three_speaker_movie_demo_v2",
    )
    parser.add_argument("--backend", choices=("onnx", "pytorch"), default="onnx")
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--seed", type=int, default=2026082501)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    stems_dir = output_dir / "stems"
    output_dir.mkdir(parents=True, exist_ok=True)

    tts = _load_tts(args.backend, args.precision)
    available = {voice_id for _, voice_id in tts.list_preset_voices()}
    requested = {item["voice"] for item in DIALOGUE}
    missing = sorted(requested - available)
    if missing:
        raise RuntimeError(f"VieNeu preset voices unavailable: {missing}")

    utterances: list[np.ndarray] = []
    for index, item in enumerate(DIALOGUE):
        raw, rate = _synthesise(
            tts,
            str(item["text"]),
            str(item["voice"]),
            args.seed + index,
        )
        utterances.append(_fade(_to_16k(raw, rate)))

    dialogue_end = max(
        float(item["onset"]) + len(audio) / SAMPLE_RATE
        for item, audio in zip(DIALOGUE, utterances, strict=True)
    )
    event_end = max(float(item["onset"]) + 1.25 for item in EVENTS)
    duration = max(9.5, dialogue_end + 0.35, event_end + 0.35)
    if duration > 10.0:
        raise RuntimeError(
            f"Synthesised scene is {duration:.2f}s; shorten the dialogue to keep it <=10s"
        )
    samples = int(round(duration * SAMPLE_RATE))

    stem_arrays: list[np.ndarray] = []
    receipt: dict[str, Any] = {
        "scene_id": "three_speaker_movie_conversation_v2",
        "description": (
            "Ba người nói tiếng Việt trò chuyện về một bộ phim, với tiếng vỗ tay, "
            "chó sủa và mèo kêu ở nền."
        ),
        "sample_rate": SAMPLE_RATE,
        "duration_seconds": duration,
        "tts": {"model_id": MODEL_ID, "sdk_version": MODEL_VERSION},
        "speakers": [],
        "events": [],
    }

    for index, (item, utterance) in enumerate(zip(DIALOGUE, utterances, strict=True), start=1):
        stem = _aligned(utterance, float(item["onset"]), samples)
        stem_path = stems_dir / f"speaker_{index:02d}.wav"
        _write(stem_path, stem)
        stem_arrays.append(stem)
        receipt["speakers"].append(
            {
                **item,
                "offset": float(item["onset"]) + len(utterance) / SAMPLE_RATE,
                "stem_path": _jsonable_path(stem_path),
                "stem_rms_dbfs": 20.0 * np.log10(max(_rms(utterance), 1e-12)),
            }
        )

    for item in EVENTS:
        source_path = Path(item["source"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source, source_rate = sf.read(source_path, dtype="float32")
        source = _resample(_mono(source), source_rate)
        begin = int(round(float(item["source_onset"]) * SAMPLE_RATE))
        finish = int(round(float(item["source_offset"]) * SAMPLE_RATE))
        event_audio = _fade(_set_rms(source[begin:finish], float(item["rms_db"])))
        stem = _aligned(event_audio, float(item["onset"]), samples)
        stem_path = stems_dir / f"{str(item['label']).lower()}.wav"
        _write(stem_path, stem)
        stem_arrays.append(stem)
        receipt["events"].append(
            {
                "label": item["label"],
                "display_name": item["display_name"],
                "onset": item["onset"],
                "offset": float(item["onset"]) + len(event_audio) / SAMPLE_RATE,
                "target_rms_dbfs": item["rms_db"],
                "source_path": _jsonable_path(source_path),
                "source_interval_seconds": [item["source_onset"], item["source_offset"]],
                "attribution": item["attribution"],
                "stem_path": _jsonable_path(stem_path),
            }
        )

    mixture = np.sum(np.stack(stem_arrays, axis=0), axis=0, dtype=np.float32)
    peak = float(np.max(np.abs(mixture)))
    scale = min(1.0, 0.95 / max(peak, 1e-12))
    if scale < 1.0:
        mixture *= scale
        for path in stems_dir.glob("*.wav"):
            stem, rate = sf.read(path, dtype="float32")
            if rate != SAMPLE_RATE:
                raise RuntimeError(f"Unexpected stem sample rate: {path}")
            _write(path, np.asarray(stem, dtype=np.float32) * scale)

    mixture_path = output_dir / "mixture.wav"
    _write(mixture_path, mixture)
    receipt["mixture_path"] = _jsonable_path(mixture_path)
    receipt["pre_normalization_peak"] = peak
    receipt["global_peak_scale"] = scale
    receipt["mixture_peak"] = float(np.max(np.abs(mixture)))
    receipt["mixture_rms_dbfs"] = 20.0 * np.log10(max(_rms(mixture), 1e-12))

    receipt_path = output_dir / "scene.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
