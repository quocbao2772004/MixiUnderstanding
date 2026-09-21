#!/usr/bin/env python3
"""Build a compact single-speaker speech + 10-event QCES branch."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_RATE = 16_000
SELECTED_LABELS = (
    "Bark",
    "Meow",
    "Reversing_beeps",
    "Crumpling_and_crinkling",
    "Single-lens_reflex_camera",
    "Chirp_and_tweet",
    "Engine_starting",
    "Toilet_flush",
    "Finger_snapping",
    "Glass_shatter",
)
SPLIT_SCENES = {"train": 60, "val": 20, "test": 20}


def _portable(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(value, encoding="utf-8")
    os.replace(tmp, path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_mono(path: Path, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = np.mean(samples, axis=1, dtype=np.float32)
    if rate != target_rate:
        divisor = math.gcd(int(rate), target_rate)
        waveform = resample_poly(
            waveform, target_rate // divisor, int(rate) // divisor
        ).astype(np.float32)
    return waveform


def _rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)) + 1e-12))


def _normalize(waveform: np.ndarray, target_dbfs: float) -> np.ndarray:
    value = np.asarray(waveform, dtype=np.float32)
    value = value - float(np.mean(value))
    gain = (10.0 ** (target_dbfs / 20.0)) / max(_rms(value), 1e-7)
    return (value * min(gain, 30.0)).astype(np.float32)


def _max_energy_crop(waveform: np.ndarray, maximum_seconds: float) -> np.ndarray:
    maximum = int(round(maximum_seconds * SAMPLE_RATE))
    if len(waveform) <= maximum:
        return waveform
    hop = max(1, SAMPLE_RATE // 20)
    best_start, best_energy = 0, -1.0
    for start in range(0, len(waveform) - maximum + 1, hop):
        energy = float(np.mean(np.square(waveform[start : start + maximum], dtype=np.float64)))
        if energy > best_energy:
            best_start, best_energy = start, energy
    return waveform[best_start : best_start + maximum]


def _load_noise_sources(paths: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    by_label: dict[str, dict[str, dict[str, Any]]] = {
        label: {} for label in SELECTED_LABELS
    }
    for path in paths:
        for row in _jsonl(path):
            label = str(row.get("label") or "")
            if label not in by_label:
                continue
            source_video = str(row.get("source_video_id") or row.get("video_id") or "")
            audio_path = _resolve(str(row.get("audio_path") or row.get("stem_path") or ""))
            if not source_video or not audio_path.is_file():
                continue
            existing = by_label[label].get(source_video)
            if existing is None or str(row.get("acceptance_tier")) == "gold":
                by_label[label][source_video] = row
    output = {label: list(rows.values()) for label, rows in by_label.items()}
    missing = {label: len(rows) for label, rows in output.items() if len(rows) < 40}
    if missing:
        raise RuntimeError(f"need at least 40 unique clean sources/class: {missing}")
    return output


def _write_audio(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, SAMPLE_RATE, format="FLAC", subtype="PCM_16")


def _question(
    scene: dict[str, Any], index: int, question: str, operation: str,
    answer_type: str, answer: str, anchors: list[str], answers: list[str],
    evidence: list[str], *, no_evidence: bool = False,
) -> dict[str, Any]:
    return {
        "format": "qces_speech_event_question_v1",
        "question_id": f"{scene['scene_id']}_q{index:02d}",
        "scene_id": scene["scene_id"],
        "split": scene["split"],
        "mixture_path": scene["mixture_path"],
        "question": question,
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
    parser.add_argument(
        "--speech-manifest", type=Path,
        default=PROJECT_ROOT / "upstream/ljspeech_qces_single_speaker_v1/ljspeech_manifest.jsonl",
    )
    parser.add_argument(
        "--noise-root", type=Path,
        default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1/primary_clean/train",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data/qces_speech_event_v1_smoke100",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir.resolve()

    speech_rows = [
        row for row in _jsonl(args.speech_manifest.resolve())
        if 1.0 <= float(row["duration_seconds"]) <= 12.0
        and 3 <= len(str(row["normalized_text"]).split()) <= 40
        and _resolve(str(row["audio_path"])).is_file()
    ]
    required_speech = 2 * sum(SPLIT_SCENES.values())
    rng.shuffle(speech_rows)
    if len(speech_rows) < required_speech:
        raise RuntimeError(
            f"need {required_speech} usable LJSpeech utterances, got {len(speech_rows)}"
        )
    speech_rows = speech_rows[:required_speech]

    source_paths = sorted(args.noise_root.resolve().glob("*/source_bank.jsonl"))
    if not source_paths:
        raise RuntimeError(f"no completed source banks under {args.noise_root}")
    noise_by_label = _load_noise_sources(source_paths)
    split_noise: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {} for split in SPLIT_SCENES
    }
    for label in SELECTED_LABELS:
        rows = noise_by_label[label]
        rng.shuffle(rows)
        rows = rows[:40]
        split_noise["train"][label] = rows[:24]
        split_noise["val"][label] = rows[24:32]
        split_noise["test"][label] = rows[32:40]

    scenes: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    speech_cursor = 0
    noise_cursor = {split: collections.Counter() for split in SPLIT_SCENES}
    reconstruction_rms: list[float] = []
    global_scene_index = 0
    for split, scene_count in SPLIT_SCENES.items():
        for local_index in range(scene_count):
            scene_id = f"speech_event_{split}_{local_index:04d}"
            speech_pair = speech_rows[speech_cursor : speech_cursor + 2]
            speech_cursor += 2
            chosen_labels = [
                SELECTED_LABELS[(4 * local_index + slot) % len(SELECTED_LABELS)]
                for slot in range(4)
            ]
            noise_rows: list[dict[str, Any]] = []
            for label in chosen_labels:
                candidates = split_noise[split][label]
                cursor = noise_cursor[split][label]
                noise_cursor[split][label] += 1
                noise_rows.append(candidates[cursor])

            speech_waves = [
                _normalize(_read_mono(_resolve(str(row["audio_path"]))), -22.0)
                for row in speech_pair
            ]
            noise_waves: list[np.ndarray] = []
            for slot, row in enumerate(noise_rows):
                waveform = _read_mono(_resolve(str(row["audio_path"])))
                onset = max(0, int(round(float(row.get("active_onset_seconds", 0.0)) * SAMPLE_RATE)))
                offset = min(len(waveform), int(round(float(row.get("active_offset_seconds", len(waveform) / SAMPLE_RATE)) * SAMPLE_RATE)))
                waveform = waveform[onset:offset] if offset > onset else waveform
                maximum = 1.5 if slot != 3 else min(1.2, 0.65 * len(speech_waves[0]) / SAMPLE_RATE)
                waveform = _max_energy_crop(waveform, max(0.35, maximum))
                noise_waves.append(_normalize(waveform, -24.0 if slot != 3 else -30.0))

            gap = lambda: rng.uniform(0.15, 0.35)
            cursor_seconds = 0.25
            placements: list[tuple[str, float, np.ndarray, dict[str, Any]]] = []
            placements.append(("noise_before", cursor_seconds, noise_waves[0], noise_rows[0]))
            cursor_seconds += len(noise_waves[0]) / SAMPLE_RATE + gap()
            speech0_onset = cursor_seconds
            placements.append(("speech_1", speech0_onset, speech_waves[0], speech_pair[0]))
            cursor_seconds += len(speech_waves[0]) / SAMPLE_RATE + gap()
            placements.append(("noise_between", cursor_seconds, noise_waves[1], noise_rows[1]))
            cursor_seconds += len(noise_waves[1]) / SAMPLE_RATE + gap()
            placements.append(("speech_2", cursor_seconds, speech_waves[1], speech_pair[1]))
            cursor_seconds += len(speech_waves[1]) / SAMPLE_RATE + gap()
            placements.append(("noise_after", cursor_seconds, noise_waves[2], noise_rows[2]))
            cursor_seconds += len(noise_waves[2]) / SAMPLE_RATE + 0.25
            overlap_onset = speech0_onset + max(
                0.05,
                (len(speech_waves[0]) - len(noise_waves[3])) / (2 * SAMPLE_RATE),
            )
            placements.append(("noise_overlap", overlap_onset, noise_waves[3], noise_rows[3]))

            total_frames = int(math.ceil(cursor_seconds * SAMPLE_RATE))
            aligned: list[np.ndarray] = []
            events: list[dict[str, Any]] = []
            for event_index, (role, onset_seconds, waveform, source) in enumerate(placements):
                start = int(round(onset_seconds * SAMPLE_RATE))
                end = min(total_frames, start + len(waveform))
                stem = np.zeros(total_frames, dtype=np.float32)
                stem[start:end] = waveform[: end - start]
                aligned.append(stem)
                is_speech = role.startswith("speech")
                event_id = f"{scene_id}_e{event_index:02d}"
                event = {
                    "event_id": event_id,
                    "event_kind": "speech" if is_speech else "sound_event",
                    "role": role,
                    "label": "Speech" if is_speech else str(source["label"]),
                    "display_name": "female speech" if is_speech else str(source.get("canonical_display_name") or source["label"].replace("_", " ")),
                    "onset_seconds": start / SAMPLE_RATE,
                    "offset_seconds": end / SAMPLE_RATE,
                    "source_id": str(source.get("utterance_id") if is_speech else source.get("source_id")),
                    "source_video_id": "" if is_speech else str(source.get("source_video_id") or ""),
                    "speaker_id": "LJ" if is_speech else None,
                    "speaker_group": "female" if is_speech else None,
                    "transcript": str(source.get("normalized_text") or "") if is_speech else None,
                }
                events.append(event)

            mixture = np.sum(np.stack(aligned), axis=0, dtype=np.float32)
            peak = float(np.max(np.abs(mixture), initial=0.0))
            scale = min(1.0, 0.97 / max(peak, 1e-12))
            mixture *= scale
            aligned = [stem * scale for stem in aligned]
            mixture_path = output_dir / "audio/mixtures" / f"{scene_id}.flac"
            _write_audio(mixture_path, mixture)
            for event, stem in zip(events, aligned):
                stem_path = output_dir / "audio/stems" / scene_id / f"{event['event_id']}.flac"
                _write_audio(stem_path, stem)
                event["stem_path"] = _portable(stem_path)

            decoded_mix, _ = sf.read(mixture_path, dtype="float32")
            decoded_sum = np.zeros_like(decoded_mix)
            for event in events:
                decoded, _ = sf.read(_resolve(str(event["stem_path"])), dtype="float32")
                decoded_sum += decoded
            reconstruction_rms.append(_rms(decoded_mix - decoded_sum))
            scene = {
                "format": "qces_speech_event_scene_v1",
                "scene_id": scene_id,
                "split": split,
                "mixture_path": _portable(mixture_path),
                "sample_rate": SAMPLE_RATE,
                "duration_seconds": len(mixture) / SAMPLE_RATE,
                "speaker_count": 1,
                "speaker_ids": ["LJ"],
                "speaker_groups": ["female"],
                "events": events,
            }
            scenes.append(scene)
            by_role = {event["role"]: event for event in events}
            n0, n1, n2, no = (
                by_role["noise_before"], by_role["noise_between"],
                by_role["noise_after"], by_role["noise_overlap"],
            )
            s0, s1 = by_role["speech_1"], by_role["speech_2"]
            quote_words = str(s0["transcript"]).split()[:8]
            quote = " ".join(quote_words).rstrip(".,;:!?")
            qbase = len(questions)
            local_questions = [
                _question(scene, 0, "What sound occurs immediately before the woman starts speaking?", "event_before_speech", "event_label", n0["display_name"], [s0["event_id"]], [n0["event_id"]], [n0["event_id"], s0["event_id"]]),
                _question(scene, 1, "What sound occurs immediately after the woman finishes her first utterance?", "event_after_speech", "event_label", n1["display_name"], [s0["event_id"]], [n1["event_id"]], [s0["event_id"], n1["event_id"]]),
                _question(scene, 2, "What sound occurs after the woman finishes her second utterance?", "event_after_speech", "event_label", n2["display_name"], [s1["event_id"]], [n2["event_id"]], [s1["event_id"], n2["event_id"]]),
                _question(scene, 3, "What sound overlaps the woman's first utterance?", "event_during_speech", "event_label", no["display_name"], [s0["event_id"]], [no["event_id"]], [s0["event_id"], no["event_id"]]),
                _question(scene, 4, "What sound occurs between the woman's first and second utterances?", "event_between_speech", "event_label", n1["display_name"], [s0["event_id"], s1["event_id"]], [n1["event_id"]], [s0["event_id"], n1["event_id"], s1["event_id"]]),
                _question(scene, 5, "What did the woman say first?", "speech_content_ordinal", "transcript", s0["transcript"], [], [s0["event_id"]], [s0["event_id"]]),
                _question(scene, 6, "What did the woman say second?", "speech_content_ordinal", "transcript", s1["transcript"], [], [s1["event_id"]], [s1["event_id"]]),
                _question(scene, 7, f"What did the woman say after the utterance beginning \"{quote}\"?", "speech_after_quote", "transcript", s1["transcript"], [s0["event_id"]], [s1["event_id"]], [s0["event_id"], s1["event_id"]]),
                _question(scene, 8, "What did the man say?", "speaker_absent", "no_evidence", "No evidence", [], [], [s0["event_id"], s1["event_id"]], no_evidence=True),
                _question(scene, 9, "What is the first sound event in the audio?", "first_event", "event_label", n0["display_name"], [], [n0["event_id"]], [n0["event_id"]]),
            ]
            assert len(questions) == qbase
            questions.extend(local_questions)
            global_scene_index += 1

    scene_path = output_dir / "scenes.jsonl"
    question_path = output_dir / "questions.jsonl"
    _atomic_text(scene_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in scenes))
    _atomic_text(question_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in questions))

    source_splits: dict[str, set[str]] = collections.defaultdict(set)
    for scene in scenes:
        for event in scene["events"]:
            source_splits[str(event["source_id"])].add(str(scene["split"]))
    leakage = sorted(source for source, splits in source_splits.items() if len(splits) > 1)
    operation_counts = collections.Counter(row["operation"] for row in questions)
    split_counts = collections.Counter(row["split"] for row in scenes)
    receipt = {
        "format": "qces_speech_event_dataset_receipt_v1",
        "complete": not leakage,
        "seed": args.seed,
        "single_speaker": True,
        "speaker_id": "LJ",
        "speaker_group": "female",
        "noise_class_count": len(SELECTED_LABELS),
        "noise_classes": list(SELECTED_LABELS),
        "scene_count": len(scenes),
        "question_count": len(questions),
        "scenes_by_split": dict(split_counts),
        "questions_per_scene": 10,
        "operation_counts": dict(operation_counts),
        "unique_speech_utterances": required_speech,
        "unique_noise_sources": len({event["source_id"] for scene in scenes for event in scene["events"] if event["event_kind"] == "sound_event"}),
        "cross_split_source_leakage_count": len(leakage),
        "maximum_reconstruction_rms": max(reconstruction_rms),
        "scene_manifest": _portable(scene_path),
        "question_manifest": _portable(question_path),
        "scene_manifest_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "question_manifest_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output_dir / "dataset_receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if leakage:
        raise RuntimeError(f"cross-split source leakage: {leakage[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

