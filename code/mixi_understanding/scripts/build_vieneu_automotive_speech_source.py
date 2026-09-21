#!/usr/bin/env python3
"""Generate realistic Vietnamese in-car utterances with VieNeu-TTS v3 Turbo.

This builds a speech-only, QCES-compatible source dataset.  The existing
automotive stress builder can then mix these clean stems with held-out horns,
sirens, traffic, engines, and reversing beeps without changing old datasets.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from mixi_understanding.scripts.build_qces_speech_event_v1 import (
    PROJECT_ROOT,
    SAMPLE_RATE,
    _atomic_text,
    _normalize,
    _portable,
    _rms,
    _write_audio,
)
from mixi_understanding.scripts.build_qces_vietnamese_speech_demo import _question


MODEL_ID = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
MODEL_VERSION = "vieneu-sdk-3.2.4"

# Natural, short commands and conversations that plausibly occur inside a car.
# Avoid punctuation-heavy written prose and retain exact transcripts for ASR QA.
SCENARIOS: tuple[dict[str, str], ...] = (
    {"category": "navigation", "text": "Đi thẳng thêm hai trăm mét rồi rẽ phải ở ngã tư tiếp theo."},
    {"category": "climate", "text": "Em hạ điều hòa xuống hai mươi hai độ giúp anh nhé."},
    {"category": "dropoff", "text": "Cho tôi xuống ở cổng bệnh viện phía bên trái."},
    {"category": "traffic", "text": "Phía trước đang tắc đường, mình đi đường vành đai nhé."},
    {"category": "phone_call", "text": "Anh đang lái xe, khoảng mười phút nữa anh gọi lại nhé."},
    {"category": "fuel", "text": "Cây xăng gần nhất còn cách đây khoảng ba cây số."},
    {"category": "pickup", "text": "Em đứng chờ ở sảnh B, xe màu trắng biển ba tám."},
    {"category": "delivery", "text": "Đơn hàng này giao tới số mười hai phố Trần Phú."},
    {"category": "assistant", "text": "Hãy gọi cho chị Lan và bật loa ngoài giúp tôi."},
    {"category": "safety", "text": "Có xe máy đang vượt bên phải, anh đi chậm lại."},
    {"category": "weather", "text": "Trời mưa to rồi, bật gạt mưa nhanh hơn đi."},
    {"category": "parking", "text": "Chỗ đỗ xe nằm dưới tầng hầm số hai."},
    {"category": "navigation", "text": "Qua cây cầu này thì nhập vào làn bên trái giúp em."},
    {"category": "climate", "text": "Trong xe hơi nóng, anh mở thêm cửa gió phía sau nhé."},
    {"category": "dropoff", "text": "Dừng giúp chị trước cửa trường tiểu học Nguyễn Du."},
    {"category": "traffic", "text": "Đoạn phía trước có tai nạn, mình đổi sang đường Lê Lợi."},
    {"category": "phone_call", "text": "Con đang trên đường về, mẹ cứ ăn cơm trước đi nhé."},
    {"category": "fuel", "text": "Xe báo sắp hết xăng rồi, tìm trạm gần đây giúp tôi."},
    {"category": "pickup", "text": "Tôi đang đứng cạnh cột số năm ở ga quốc nội."},
    {"category": "delivery", "text": "Anh để gói hàng ở quầy lễ tân rồi gọi cho em."},
    {"category": "assistant", "text": "Bật bản tin giao thông và giảm âm lượng xuống một chút."},
    {"category": "safety", "text": "Có người đi bộ phía trước, chú ý phanh xe nhé."},
    {"category": "weather", "text": "Sương mù dày quá, bật đèn chiếu gần giúp anh."},
    {"category": "parking", "text": "Lùi chậm thôi, phía sau còn khoảng nửa mét."},
)

SPLIT_VOICES: dict[str, tuple[tuple[str, str], ...]] = {
    "val": (
        ("Ngọc Linh", "female"),
        ("Trúc Ly", "female"),
        ("Xuân Vĩnh", "male"),
        ("Thái Sơn", "male"),
    ),
    "test": (
        ("Đoan Trang", "female"),
        ("Mai Anh", "female"),
        ("Minh Đức", "male"),
        ("Phạm Tuyên", "male"),
    ),
}

# Determined by an independent Whisper-medium clean-speech audit on v1.  These
# substitutions repair TTS pronunciation failures, not noisy-mixture outcomes.
QUALITY_REPAIR_VOICES: dict[int, tuple[str, str]] = {
    3: ("Ngọc Linh", "female"),
    5: ("Ngọc Linh", "female"),
    6: ("Ngọc Linh", "female"),
    10: ("Ngọc Linh", "female"),
}


def _load_tts(backend: str, precision: str):
    try:
        from vieneu import Vieneu
    except ImportError as error:
        raise RuntimeError(
            "VieNeu is not installed. Run with the isolated .venv-vieneu environment."
        ) from error
    return Vieneu(backend=backend, precision=precision)


def _synthesise(tts: Any, text: str, voice: str, seed: int) -> tuple[np.ndarray, int]:
    # VieNeu uses both Python and NumPy randomness in the current SDK path.
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    audio = np.asarray(
        tts.infer(text, voice=voice, style="tu_nhien", temperature=0.8),
        dtype=np.float32,
    ).reshape(-1)
    return audio, 48_000


def _to_16k(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if source_rate != SAMPLE_RATE:
        divisor = int(np.gcd(source_rate, SAMPLE_RATE))
        audio = resample_poly(audio, SAMPLE_RATE // divisor, source_rate // divisor)
    # Trim only long digital silence; preserve natural breath and brief pauses.
    active = np.flatnonzero(np.abs(audio) >= 2e-4)
    if active.size:
        pad = int(0.12 * SAMPLE_RATE)
        start = max(0, int(active[0]) - pad)
        end = min(len(audio), int(active[-1]) + pad + 1)
        audio = audio[start:end]
    return _normalize(np.asarray(audio, dtype=np.float32), -22.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_vietnamese_vieneu_source_v1",
    )
    parser.add_argument("--backend", choices=("onnx", "pytorch"), default="onnx")
    parser.add_argument("--precision", choices=("int8", "fp32"), default="int8")
    parser.add_argument("--seed", type=int, default=2026080501)
    parser.add_argument("--limit", type=int, default=len(SCENARIOS))
    parser.add_argument(
        "--reuse-from",
        type=Path,
        default=None,
        help="Copy cached clean clips from a previous compatible build.",
    )
    parser.add_argument(
        "--force-index",
        type=int,
        action="append",
        default=[],
        help="Re-synthesise this scenario index even when --reuse-from is set.",
    )
    parser.add_argument(
        "--quality-repair-profile",
        choices=("none", "whisper-medium-v1"),
        default="none",
        help="Apply voice substitutions learned only from the clean TTS audit.",
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    selected = SCENARIOS[: max(0, min(args.limit, len(SCENARIOS)))]
    if not selected:
        raise RuntimeError("No scenarios selected")
    tts = _load_tts(args.backend, args.precision)
    reuse_root = args.reuse_from.resolve() if args.reuse_from else None
    force_indices = set(args.force_index)
    # SDK returns (human-readable description, short voice id); infer() expects id.
    available_voices = {voice_id for _, voice_id in tts.list_preset_voices()}
    requested_voices = {
        voice for split_voices in SPLIT_VOICES.values() for voice, _ in split_voices
    }
    missing_voices = sorted(requested_voices - available_voices)
    if missing_voices:
        raise RuntimeError(
            f"Preset voices unavailable in the installed SDK: {missing_voices}; "
            f"available={sorted(available_voices)}"
        )

    scenes: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    voice_counts: collections.Counter[str] = collections.Counter()
    durations: list[float] = []
    for index, scenario in enumerate(selected):
        split = "val" if index < (len(SCENARIOS) // 2) else "test"
        voice, gender = SPLIT_VOICES[split][index % len(SPLIT_VOICES[split])]
        repaired = args.quality_repair_profile != "none" and index in QUALITY_REPAIR_VOICES
        if repaired:
            voice, gender = QUALITY_REPAIR_VOICES[index]
        scene_id = f"vi_vieneu_{split}_{index:03d}"
        clean_path = output / "audio/clean" / f"{scene_id}.flac"
        reusable_path = reuse_root / "audio/clean" / f"{scene_id}.flac" if reuse_root else None
        if not clean_path.is_file() and reusable_path and reusable_path.is_file() and index not in force_indices:
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(reusable_path, clean_path)
        if clean_path.is_file():
            speech, rate = sf.read(clean_path, dtype="float32")
            if rate != SAMPLE_RATE:
                raise RuntimeError(f"Unexpected cached sample rate for {clean_path}: {rate}")
            speech = np.asarray(speech, dtype=np.float32)
        else:
            synthesis_seed = args.seed + index + (10_000 if repaired else 0)
            raw, rate = _synthesise(tts, scenario["text"], voice, synthesis_seed)
            speech = _to_16k(raw, rate)
            _write_audio(clean_path, speech)
        duration = len(speech) / SAMPLE_RATE
        durations.append(duration)
        voice_counts[voice] += 1
        source_id = f"vieneu_v3_{hashlib.sha1((voice + '|' + scenario['text']).encode()).hexdigest()[:16]}"
        event_id = f"{scene_id}_e00"
        event = {
            "event_id": event_id,
            "event_kind": "speech",
            "role": "speech_1",
            "label": "Speech",
            "display_name": "lời nói tiếng Việt",
            "onset_seconds": 0.0,
            "offset_seconds": duration,
            "source_id": source_id,
            "speaker_id": f"vieneu_preset_{voice.replace(' ', '_').lower()}",
            "speaker_group": gender,
            "speaker_role": "người ở trong xe",
            "transcript": scenario["text"],
            "scenario_category": scenario["category"],
            "tts_model_id": MODEL_ID,
            "tts_sdk_version": MODEL_VERSION,
            "tts_voice": voice,
            "tts_style": "tu_nhien",
            "tts_temperature": 0.8,
            "tts_quality_repair_profile": args.quality_repair_profile if repaired else None,
            "stem_path": _portable(clean_path),
        }
        scene = {
            "format": "qces_vietnamese_vieneu_speech_scene_v1",
            "scene_id": scene_id,
            "split": split,
            "mixture_path": _portable(clean_path),
            "sample_rate": SAMPLE_RATE,
            "duration_seconds": duration,
            "speaker_count": 1,
            "speaker_ids": [event["speaker_id"]],
            "speaker_groups": [gender],
            "language": "vi-VN",
            "scenario_category": scenario["category"],
            "events": [event],
        }
        scenes.append(scene)
        questions.append(
            _question(
                scene,
                0,
                "Người trong xe đã nói gì?",
                "vi_speech_content",
                "transcript",
                scenario["text"],
                [],
                [event_id],
                [event_id],
            )
        )

    scene_path = output / "scenes.jsonl"
    question_path = output / "questions.jsonl"
    _atomic_text(scene_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in scenes))
    _atomic_text(question_path, "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in questions))
    split_voices = {
        split: sorted({scene["speaker_ids"][0] for scene in scenes if scene["split"] == split})
        for split in ("val", "test")
    }
    overlap = sorted(set(split_voices["val"]) & set(split_voices["test"]))
    receipt = {
        "format": "qces_vietnamese_vieneu_source_receipt_v1",
        "complete": len(scenes) == len(selected) and not overlap,
        "purpose": "realistic in-car Vietnamese speech source for controlled automotive mixtures",
        "synthetic_speech": True,
        "tts_model_id": MODEL_ID,
        "tts_sdk_version": MODEL_VERSION,
        "tts_backend": args.backend,
        "tts_precision": args.precision,
        "quality_repair_profile": args.quality_repair_profile,
        "reused_source_dataset": _portable(reuse_root) if reuse_root else None,
        "forced_resynthesis_indices": sorted(force_indices),
        "quality_repaired_scene_count": sum(
            index in QUALITY_REPAIR_VOICES
            for index in range(len(selected))
            if args.quality_repair_profile != "none"
        ),
        "tts_license": "Apache-2.0",
        "sample_rate": SAMPLE_RATE,
        "scene_count": len(scenes),
        "question_count": len(questions),
        "scenario_categories": dict(collections.Counter(x["scenario_category"] for x in scenes)),
        "voice_counts": dict(voice_counts),
        "speaker_cross_split_overlap_count": len(overlap),
        "mean_duration_seconds": float(np.mean(durations)),
        "minimum_duration_seconds": float(np.min(durations)),
        "maximum_duration_seconds": float(np.max(durations)),
        "mean_clean_rms": float(np.mean([_rms(sf.read(Path(x["mixture_path"]), dtype="float32")[0]) for x in scenes])),
        "requires_tts_asr_audit": True,
        "scene_manifest": _portable(scene_path),
        "question_manifest": _portable(question_path),
        "scene_manifest_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "question_manifest_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output / "dataset_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
