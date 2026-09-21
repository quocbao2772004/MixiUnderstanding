#!/usr/bin/env python3
"""Build a diagnostic multi-speaker speech/event benchmark.

The existing speech branch has one generic ``Speech`` label.  This builder
creates two temporally ordered speakers with gender-specific labels and three
controlled overlap levels.  Speech is synthetic VieNeu speech; the benchmark
is for detector feasibility, not a real-conversation claim.
"""

from __future__ import annotations

import collections
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

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


SPLIT_SCENES = {"train": 12, "val": 6, "test": 6}
DIFFICULTIES = (
    ("easy_overlap", 6.0),
    ("medium_overlap", 0.0),
    ("hard_overlap", -6.0),
)
SPEAKER_LABELS = {"male": "Speech_male", "female": "Speech_female"}


def _scale_rms(waveform: np.ndarray, target_rms: float) -> np.ndarray:
    return np.asarray(waveform, dtype=np.float32) * (
        float(target_rms) / max(_rms(waveform), 1e-7)
    )


def _question(
    scene: dict[str, Any],
    index: int,
    text: str,
    operation: str,
    answer: str,
    answers: list[str],
    evidence: list[str],
    *,
    no_evidence: bool = False,
) -> dict[str, Any]:
    return {
        "format": "qces_multispeaker_speech_event_question_v1",
        "question_id": f"{scene['scene_id']}_q{index:02d}",
        "scene_id": scene["scene_id"],
        "split": scene["split"],
        "mixture_path": scene["mixture_path"],
        "question": text,
        "operation": operation,
        "answer": answer,
        "answer_event_ids": answers,
        "evidence_event_ids": evidence,
        "no_evidence": no_evidence,
    }


def _speech_pairs(source_scenes: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_gender: dict[str, list[dict[str, Any]]] = {"male": [], "female": []}
    for scene in source_scenes:
        event = scene["events"][0]
        by_gender[str(event["speaker_group"])].append(scene)
    if not by_gender["male"] or not by_gender["female"]:
        raise RuntimeError("multi-speaker source needs both male and female clips")
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index in range(max(len(by_gender["male"]), len(by_gender["female"]))):
        female = by_gender["female"][index % len(by_gender["female"])]
        male = by_gender["male"][index % len(by_gender["male"])]
        pairs.append((female, male))
    return pairs


def _speech_wave(scene: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    event = scene["events"][0]
    return _normalize(_read_mono(_resolve(str(event["stem_path"]))), -22.0), event


def _noise_wave(event: dict[str, Any], maximum_seconds: float = 1.3) -> np.ndarray:
    waveform = _read_mono(_resolve(str(event["stem_path"])))
    start = max(0, int(round(float(event["onset_seconds"]) * SAMPLE_RATE)))
    end = min(len(waveform), int(round(float(event["offset_seconds"]) * SAMPLE_RATE)))
    active = waveform[start:end] if end > start else waveform
    return _normalize(_max_energy_crop(active, maximum_seconds), -26.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speech-source",
        type=Path,
        default=PROJECT_ROOT / "data/qces_vietnamese_vieneu_source_v3",
    )
    parser.add_argument(
        "--noise-dataset",
        type=Path,
        default=PROJECT_ROOT / "data/qces_speech_event_hard_v2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_multispeaker_speech_event_v1",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output = args.output_dir.resolve()

    source_scenes = _jsonl((args.speech_source / "scenes.jsonl").resolve())
    source_by_split = {
        split: [scene for scene in source_scenes if scene["split"] == split]
        for split in ("val", "test")
    }
    # Train/validation use disjoint source IDs from the source validation pool;
    # test uses the source test pool and therefore has disjoint voices too.
    train_source = source_by_split["val"][:8]
    val_source = source_by_split["val"][8:]
    test_source = source_by_split["test"]
    source_pools = {"train": train_source, "val": val_source, "test": test_source}
    for split, pool in source_pools.items():
        if not pool:
            raise RuntimeError(f"empty speech source pool: {split}")

    noise_scenes = _jsonl((args.noise_dataset / "scenes.jsonl").resolve())
    noise_by_split = {
        split: [scene for scene in noise_scenes if scene["split"] == split]
        for split in SPLIT_SCENES
    }
    if any(not values for values in noise_by_split.values()):
        raise RuntimeError("noise dataset must contain train/val/test scenes")

    scenes: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    measured_snrs: list[float] = []
    source_usage: dict[str, set[str]] = collections.defaultdict(set)

    for split, count in SPLIT_SCENES.items():
        pairs = _speech_pairs(source_pools[split])
        donors = list(noise_by_split[split])
        rng.shuffle(donors)
        for local_index in range(count):
            difficulty, requested_snr = DIFFICULTIES[(local_index + (0 if split == "train" else 1)) % len(DIFFICULTIES)]
            first_scene, second_scene = pairs[local_index % len(pairs)]
            if local_index % 2:
                first_scene, second_scene = second_scene, first_scene
            speech0, meta0 = _speech_wave(first_scene)
            speech1, meta1 = _speech_wave(second_scene)
            donor = donors[local_index % len(donors)]
            noise_events = [
                event for event in donor["events"] if event["event_kind"] == "sound_event"
            ]
            if len(noise_events) < 4:
                raise RuntimeError(f"noise donor has fewer than four events: {donor['scene_id']}")
            noise0, noise1, noise2, noise3 = (_noise_wave(event) for event in noise_events[:4])
            speech0_rms = _rms(speech0)
            speech1_rms = _rms(speech1)
            overlap0 = _scale_rms(noise1, speech0_rms / (10.0 ** (requested_snr / 20.0)))
            overlap1 = _scale_rms(noise2, speech1_rms / (10.0 ** (requested_snr / 20.0)))

            gap = lambda: rng.uniform(0.10, 0.22)
            cursor = 0.20
            placements: list[tuple[str, float, np.ndarray, dict[str, Any] | None]] = []
            placements.append(("noise_before", cursor, noise0, noise_events[0]))
            cursor += len(noise0) / SAMPLE_RATE + gap()
            speech0_onset = cursor
            placements.append(("speech_1", speech0_onset, speech0, meta0))
            speech0_offset = speech0_onset + len(speech0) / SAMPLE_RATE
            placements.append(("noise_overlap_first", speech0_onset + 0.35 * len(speech0) / SAMPLE_RATE, overlap0, noise_events[1]))
            cursor = speech0_offset + gap()
            placements.append(("noise_between", cursor, noise3, noise_events[3]))
            cursor += len(noise3) / SAMPLE_RATE + gap()
            speech1_onset = cursor
            placements.append(("speech_2", speech1_onset, speech1, meta1))
            speech1_offset = speech1_onset + len(speech1) / SAMPLE_RATE
            placements.append(("noise_overlap_second", speech1_onset + 0.35 * len(speech1) / SAMPLE_RATE, overlap1, noise_events[2]))
            cursor = speech1_offset + gap()
            placements.append(("noise_after", cursor, noise0, noise_events[0]))
            total_frames = int(math.ceil((cursor + len(noise0) / SAMPLE_RATE + 0.20) * SAMPLE_RATE))

            aligned: list[np.ndarray] = []
            events: list[dict[str, Any]] = []
            scene_id = f"multispeaker_{split}_{local_index:04d}"
            for event_index, (role, onset, waveform, source) in enumerate(placements):
                start = int(round(onset * SAMPLE_RATE))
                end = min(total_frames, start + len(waveform))
                stem = np.zeros(total_frames, dtype=np.float32)
                stem[start:end] = waveform[: end - start]
                aligned.append(stem)
                is_speech = role.startswith("speech")
                speaker_group = str(source["speaker_group"]) if is_speech and source else None
                label = SPEAKER_LABELS[speaker_group] if speaker_group else str(source["label"])
                event_id = f"{scene_id}_e{event_index:02d}"
                event = {
                    "event_id": event_id,
                    "event_kind": "speech" if is_speech else "sound_event",
                    "role": role,
                    "label": label,
                    "display_name": (
                        f"{speaker_group} speech" if speaker_group else str(source.get("display_name") or source["label"])
                    ),
                    "onset_seconds": start / SAMPLE_RATE,
                    "offset_seconds": end / SAMPLE_RATE,
                    "source_id": str(source["source_id"] if is_speech else source["source_id"]),
                    "speaker_id": str(source.get("speaker_id")) if is_speech else None,
                    "speaker_group": speaker_group,
                    "transcript": str(source.get("transcript") or "") if is_speech else None,
                }
                events.append(event)
                source_usage[event["source_id"]].add(split)

            mixture = np.sum(np.stack(aligned), axis=0, dtype=np.float32)
            peak = max(float(np.max(np.abs(mixture), initial=0.0)), *(float(np.max(np.abs(stem), initial=0.0)) for stem in aligned))
            scale = min(1.0, 0.97 / max(peak, 1e-12))
            mixture *= scale
            aligned = [stem * scale for stem in aligned]
            mixture_path = output / "audio/mixtures" / f"{scene_id}.flac"
            _write_audio(mixture_path, mixture)
            for event, stem in zip(events, aligned):
                stem_path = output / "audio/stems" / scene_id / f"{event['event_id']}.flac"
                _write_audio(stem_path, stem)
                event["stem_path"] = _portable(stem_path)

            first, second = events[1], events[4]
            before, between, after = events[0], events[3], events[6]
            overlap_first, overlap_second = events[2], events[5]
            first_gender, second_gender = first["speaker_group"], second["speaker_group"]
            scene = {
                "format": "qces_multispeaker_speech_event_scene_v1",
                "scene_id": scene_id,
                "split": split,
                "mixture_path": _portable(mixture_path),
                "sample_rate": SAMPLE_RATE,
                "duration_seconds": len(mixture) / SAMPLE_RATE,
                "speaker_count": 2,
                "speaker_groups": [first_gender, second_gender],
                "speaker_ids": [first["speaker_id"], second["speaker_id"]],
                "difficulty": difficulty,
                "requested_speech_to_overlap_noise_snr_db": requested_snr,
                "events": events,
            }
            scenes.append(scene)
            questions.extend(
                [
                    _question(scene, 0, "Who speaks first, the man or the woman?", "speaker_first", first_gender, [first["event_id"]], [first["event_id"], second["event_id"]]),
                    _question(scene, 1, "Who speaks second, the man or the woman?", "speaker_second", second_gender, [second["event_id"]], [first["event_id"], second["event_id"]]),
                    _question(scene, 2, "What sound occurs immediately before the first speaker starts talking?", "event_before_first_speech", before["label"], [before["event_id"]], [before["event_id"], first["event_id"]]),
                    _question(scene, 3, "What sound occurs between the two speakers?", "event_between_speakers", between["label"], [between["event_id"]], [first["event_id"], between["event_id"], second["event_id"]]),
                    _question(scene, 4, "What sound overlaps the second speaker?", "event_overlap_second_speech", overlap_second["label"], [overlap_second["event_id"]], [second["event_id"], overlap_second["event_id"]]),
                    _question(scene, 5, "What sound occurs immediately after the second speaker finishes?", "event_after_second_speech", after["label"], [after["event_id"]], [second["event_id"], after["event_id"]]),
                    _question(scene, 6, "Did a third person speak?", "third_speaker_absent", "No evidence", [], [first["event_id"], second["event_id"]], no_evidence=True),
                ]
            )
            overlap_mask = np.abs(aligned[2] + aligned[5]) > 1e-7
            speech_mask = aligned[1] if first_gender == "male" else aligned[4]
            measured_snrs.append(10.0 * math.log10((float(np.mean(np.square(speech_mask[overlap_mask], dtype=np.float64))) + 1e-12) / (float(np.mean(np.square((aligned[2] + aligned[5])[overlap_mask], dtype=np.float64))) + 1e-12)))
            decoded = _read_mono(mixture_path)
            summed = np.sum(np.stack([_read_mono(_resolve(event["stem_path"])) for event in events]), axis=0)
            reconstruction_errors.append(_rms(decoded - summed))

    leakage = sorted(source_id for source_id, splits in source_usage.items() if len(splits) > 1)
    scene_path, question_path = output / "scenes.jsonl", output / "questions.jsonl"
    _atomic_text(scene_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in scenes))
    _atomic_text(question_path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in questions))
    receipt = {
        "format": "qces_multispeaker_speech_event_dataset_receipt_v1",
        "complete": not leakage,
        "seed": args.seed,
        "scene_count": len(scenes),
        "question_count": len(questions),
        "scenes_by_split": dict(collections.Counter(scene["split"] for scene in scenes)),
        "questions_per_scene": 7,
        "difficulty_counts": dict(collections.Counter(scene["difficulty"] for scene in scenes)),
        "speaker_labels": ["male", "female"],
        "speech_source": _portable(args.speech_source.resolve()),
        "noise_source": _portable(args.noise_dataset.resolve()),
        "synthetic_speech": True,
        "cross_split_source_leakage_count": len(leakage),
        "maximum_reconstruction_rms": max(reconstruction_errors),
        "mean_measured_speech_to_overlap_noise_snr_db": float(np.mean(measured_snrs)),
        "scene_manifest": _portable(scene_path),
        "question_manifest": _portable(question_path),
        "scene_manifest_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "question_manifest_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output / "dataset_receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if leakage:
        raise RuntimeError(f"cross-split source leakage: {leakage[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
