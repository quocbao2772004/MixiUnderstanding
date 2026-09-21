#!/usr/bin/env python3
"""Build noisy, overlapping single-speaker QCES scenes at controlled SNRs."""

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
    PROJECT_ROOT,
    SAMPLE_RATE,
    SELECTED_LABELS,
    _atomic_text,
    _jsonl,
    _load_noise_sources,
    _max_energy_crop,
    _normalize,
    _portable,
    _question,
    _read_mono,
    _resolve,
    _rms,
    _write_audio,
)


SPLIT_SCENES = {"train": 36, "val": 12, "test": 12}
DIFFICULTIES = (
    ("moderate_overlap", 3.0),
    ("hard_overlap", 0.0),
    ("extreme_overlap", -5.0),
)


def _scale_rms(waveform: np.ndarray, target_rms: float) -> np.ndarray:
    return np.asarray(waveform, dtype=np.float32) * (
        float(target_rms) / max(_rms(waveform), 1e-7)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--speech-manifest",
        type=Path,
        default=PROJECT_ROOT / "upstream/ljspeech_qces_single_speaker_v1/ljspeech_manifest.jsonl",
    )
    parser.add_argument(
        "--noise-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_full200_adaptive_v1/primary_clean/train",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data/qces_speech_event_hard_v2",
    )
    parser.add_argument(
        "--split-reference",
        type=Path,
        default=PROJECT_ROOT / "data/qces_speech_event_v1_smoke100/scenes.jsonl",
        help="Easy-branch scenes whose source-to-split assignment remains authoritative.",
    )
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir.resolve()

    reference_scenes = _jsonl(args.split_reference.resolve())
    reference_split: dict[str, str] = {}
    for scene in reference_scenes:
        for event in scene["events"]:
            source_id = str(event["source_id"])
            previous = reference_split.setdefault(source_id, str(scene["split"]))
            if previous != str(scene["split"]):
                raise RuntimeError(f"split reference leaks {source_id}: {previous}/{scene['split']}")

    speech_rows = [
        row
        for row in _jsonl(args.speech_manifest.resolve())
        if 1.0 <= float(row["duration_seconds"]) <= 12.0
        and 3 <= len(str(row["normalized_text"]).split()) <= 40
        and _resolve(str(row["audio_path"])).is_file()
    ]
    required_speech = 2 * sum(SPLIT_SCENES.values())
    speech_by_split: dict[str, list[dict[str, Any]]] = {}
    for split, count in SPLIT_SCENES.items():
        candidates = [
            row
            for row in speech_rows
            if reference_split.get(str(row["utterance_id"])) == split
        ]
        rng.shuffle(candidates)
        if len(candidates) < 2 * count:
            raise RuntimeError(
                f"speech split {split}: need {2 * count}, got {len(candidates)}"
            )
        speech_by_split[split] = candidates[: 2 * count]

    source_paths = sorted(args.noise_root.resolve().glob("*/source_bank.jsonl"))
    noise_by_label = _load_noise_sources(source_paths)
    schedules: dict[str, list[list[str]]] = {}
    needed: dict[str, collections.Counter[str]] = {}
    for split, count in SPLIT_SCENES.items():
        schedules[split] = [
            [SELECTED_LABELS[(6 * index + slot) % len(SELECTED_LABELS)] for slot in range(6)]
            for index in range(count)
        ]
        needed[split] = collections.Counter(
            label for scene_labels in schedules[split] for label in scene_labels
        )
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {} for split in SPLIT_SCENES
    }
    for label in SELECTED_LABELS:
        for split in SPLIT_SCENES:
            count = needed[split][label]
            rows = [
                row
                for row in noise_by_label[label]
                if reference_split.get(str(row["source_id"])) == split
            ]
            rng.shuffle(rows)
            if len(rows) < count:
                raise RuntimeError(
                    f"{label}/{split}: need {count} split-locked sources, got {len(rows)}"
                )
            pools[split][label] = rows[:count]

    source_cursor = {split: collections.Counter() for split in SPLIT_SCENES}
    speech_cursor = collections.Counter()
    scenes: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    reconstruction_rms: list[float] = []
    actual_snrs: list[float] = []
    global_index = 0
    for split, scene_count in SPLIT_SCENES.items():
        for local_index in range(scene_count):
            difficulty, requested_snr = DIFFICULTIES[global_index % len(DIFFICULTIES)]
            scene_id = f"speech_hard_{split}_{local_index:04d}"
            speech_pair = speech_by_split[split][
                speech_cursor[split] : speech_cursor[split] + 2
            ]
            speech_cursor[split] += 2
            noise_rows: list[dict[str, Any]] = []
            for label in schedules[split][local_index]:
                index = source_cursor[split][label]
                source_cursor[split][label] += 1
                noise_rows.append(pools[split][label][index])

            speech_waves = [
                _normalize(_read_mono(_resolve(str(row["audio_path"]))), -22.0)
                for row in speech_pair
            ]
            raw_noises: list[np.ndarray] = []
            for slot, row in enumerate(noise_rows):
                waveform = _read_mono(_resolve(str(row["audio_path"])))
                onset = max(
                    0,
                    int(round(float(row.get("active_onset_seconds", 0.0)) * SAMPLE_RATE)),
                )
                offset = min(
                    len(waveform),
                    int(
                        round(
                            float(
                                row.get(
                                    "active_offset_seconds", len(waveform) / SAMPLE_RATE
                                )
                            )
                            * SAMPLE_RATE
                        )
                    ),
                )
                waveform = waveform[onset:offset] if offset > onset else waveform
                maximum = 1.20 if slot < 3 else 2.20
                raw_noises.append(_max_energy_crop(waveform, maximum))
            sequential_noises = [_normalize(value, -23.0) for value in raw_noises[:3]]
            speech0_rms, speech1_rms = _rms(speech_waves[0]), _rms(speech_waves[1])
            combined_target0 = speech0_rms / (10.0 ** (requested_snr / 20.0))
            per_noise0 = combined_target0 / math.sqrt(2.0)
            overlap_noises = [
                _scale_rms(raw_noises[3], per_noise0),
                _scale_rms(raw_noises[4], per_noise0),
                _scale_rms(
                    raw_noises[5],
                    speech1_rms / (10.0 ** (requested_snr / 20.0)),
                ),
            ]

            gap = lambda: rng.uniform(0.07, 0.17)
            cursor_seconds = 0.20
            placements: list[tuple[str, float, np.ndarray, dict[str, Any]]] = []
            placements.append(("noise_before", cursor_seconds, sequential_noises[0], noise_rows[0]))
            cursor_seconds += len(sequential_noises[0]) / SAMPLE_RATE + gap()
            speech0_onset = cursor_seconds
            placements.append(("speech_1", speech0_onset, speech_waves[0], speech_pair[0]))
            speech0_offset = speech0_onset + len(speech_waves[0]) / SAMPLE_RATE
            cursor_seconds = speech0_offset + gap()
            placements.append(("noise_between", cursor_seconds, sequential_noises[1], noise_rows[1]))
            cursor_seconds += len(sequential_noises[1]) / SAMPLE_RATE + gap()
            speech1_onset = cursor_seconds
            placements.append(("speech_2", speech1_onset, speech_waves[1], speech_pair[1]))
            speech1_offset = speech1_onset + len(speech_waves[1]) / SAMPLE_RATE
            cursor_seconds = speech1_offset + gap()
            placements.append(("noise_after", cursor_seconds, sequential_noises[2], noise_rows[2]))
            cursor_seconds += len(sequential_noises[2]) / SAMPLE_RATE + 0.20

            speech0_duration = len(speech_waves[0]) / SAMPLE_RATE
            speech1_duration = len(speech_waves[1]) / SAMPLE_RATE
            overlap0a_onset = speech0_onset + 0.22 * speech0_duration
            overlap0b_onset = speech0_onset + 0.52 * speech0_duration
            overlap1_onset = speech1_onset + 0.32 * speech1_duration
            placements.extend(
                [
                    ("noise_overlap_first_1", overlap0a_onset, overlap_noises[0], noise_rows[3]),
                    ("noise_overlap_first_2", overlap0b_onset, overlap_noises[1], noise_rows[4]),
                    ("noise_overlap_second", overlap1_onset, overlap_noises[2], noise_rows[5]),
                ]
            )

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
                events.append(
                    {
                        "event_id": event_id,
                        "event_kind": "speech" if is_speech else "sound_event",
                        "role": role,
                        "label": "Speech" if is_speech else str(source["label"]),
                        "display_name": (
                            "female speech"
                            if is_speech
                            else str(
                                source.get("canonical_display_name")
                                or source["label"].replace("_", " ")
                            )
                        ),
                        "onset_seconds": start / SAMPLE_RATE,
                        "offset_seconds": end / SAMPLE_RATE,
                        "source_id": str(
                            source.get("utterance_id") if is_speech else source.get("source_id")
                        ),
                        "source_video_id": "" if is_speech else str(source.get("source_video_id") or ""),
                        "speaker_id": "LJ" if is_speech else None,
                        "speaker_group": "female" if is_speech else None,
                        "transcript": (
                            str(source.get("normalized_text") or "") if is_speech else None
                        ),
                    }
                )

            mixture = np.sum(np.stack(aligned), axis=0, dtype=np.float32)
            peak = max(
                float(np.max(np.abs(mixture), initial=0.0)),
                max(float(np.max(np.abs(stem), initial=0.0)) for stem in aligned),
            )
            scale = min(1.0, 0.97 / max(peak, 1e-12))
            mixture *= scale
            aligned = [stem * scale for stem in aligned]
            mixture_path = output_dir / "audio/mixtures" / f"{scene_id}.flac"
            _write_audio(mixture_path, mixture)
            for event, stem in zip(events, aligned):
                stem_path = output_dir / "audio/stems" / scene_id / f"{event['event_id']}.flac"
                _write_audio(stem_path, stem)
                event["stem_path"] = _portable(stem_path)

            speech0_aligned = aligned[1]
            overlap0_sum = aligned[5] + aligned[6]
            overlap_mask = np.abs(overlap0_sum) > 1e-7
            actual_snr = 10.0 * math.log10(
                (float(np.mean(np.square(speech0_aligned[overlap_mask], dtype=np.float64))) + 1e-12)
                / (float(np.mean(np.square(overlap0_sum[overlap_mask], dtype=np.float64))) + 1e-12)
            )
            actual_snrs.append(actual_snr)
            decoded_mix, _ = sf.read(mixture_path, dtype="float32")
            decoded_sum = np.zeros_like(decoded_mix)
            for event in events:
                decoded, _ = sf.read(_resolve(event["stem_path"]), dtype="float32")
                decoded_sum += decoded
            reconstruction_rms.append(_rms(decoded_mix - decoded_sum))

            scene = {
                "format": "qces_speech_event_hard_scene_v2",
                "scene_id": scene_id,
                "split": split,
                "mixture_path": _portable(mixture_path),
                "sample_rate": SAMPLE_RATE,
                "duration_seconds": len(mixture) / SAMPLE_RATE,
                "speaker_count": 1,
                "speaker_ids": ["LJ"],
                "speaker_groups": ["female"],
                "difficulty": difficulty,
                "requested_speech_to_overlap_noise_snr_db": requested_snr,
                "measured_first_speech_to_overlap_noise_snr_db": actual_snr,
                "events": events,
            }
            scenes.append(scene)
            by_role = {event["role"]: event for event in events}
            n0, n1, n2 = by_role["noise_before"], by_role["noise_between"], by_role["noise_after"]
            o0, o1, o2 = (
                by_role["noise_overlap_first_1"],
                by_role["noise_overlap_first_2"],
                by_role["noise_overlap_second"],
            )
            s0, s1 = by_role["speech_1"], by_role["speech_2"]
            quote = " ".join(str(s0["transcript"]).split()[:8]).rstrip(".,;:!?")
            questions.extend(
                [
                    _question(scene, 0, "What sound occurs immediately before the woman starts speaking?", "event_before_speech", "event_label", n0["display_name"], [s0["event_id"]], [n0["event_id"]], [n0["event_id"], s0["event_id"]]),
                    _question(scene, 1, "What sound occurs immediately after the woman finishes her first utterance?", "event_after_speech", "event_label", n1["display_name"], [s0["event_id"]], [n1["event_id"]], [s0["event_id"], n1["event_id"]]),
                    _question(scene, 2, "What sound occurs after the woman finishes her second utterance?", "event_after_speech", "event_label", n2["display_name"], [s1["event_id"]], [n2["event_id"]], [s1["event_id"], n2["event_id"]]),
                    _question(scene, 3, "What sound starts first while the woman is speaking her first utterance?", "event_during_speech_ordinal", "event_label", o0["display_name"], [s0["event_id"]], [o0["event_id"]], [s0["event_id"], o0["event_id"]]),
                    _question(scene, 4, "What sound starts second while the woman is speaking her first utterance?", "event_during_speech_ordinal", "event_label", o1["display_name"], [s0["event_id"]], [o1["event_id"]], [s0["event_id"], o1["event_id"]]),
                    _question(scene, 5, "What sound overlaps the woman's second utterance?", "event_during_second_speech", "event_label", o2["display_name"], [s1["event_id"]], [o2["event_id"]], [s1["event_id"], o2["event_id"]]),
                    _question(scene, 6, "What did the woman say first?", "speech_content_ordinal", "transcript", s0["transcript"], [], [s0["event_id"]], [s0["event_id"]]),
                    _question(scene, 7, "What did the woman say second?", "speech_content_ordinal", "transcript", s1["transcript"], [], [s1["event_id"]], [s1["event_id"]]),
                    _question(scene, 8, f"What did the woman say after the utterance beginning \"{quote}\"?", "speech_after_quote", "transcript", s1["transcript"], [s0["event_id"]], [s1["event_id"]], [s0["event_id"], s1["event_id"]]),
                    _question(scene, 9, "What did the man say?", "speaker_absent", "no_evidence", "No evidence", [], [], [s0["event_id"], s1["event_id"]], no_evidence=True),
                ]
            )
            global_index += 1

    scene_path, question_path = output_dir / "scenes.jsonl", output_dir / "questions.jsonl"
    _atomic_text(scene_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in scenes))
    _atomic_text(question_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in questions))
    source_splits: dict[str, set[str]] = collections.defaultdict(set)
    for scene in scenes:
        for event in scene["events"]:
            source_splits[event["source_id"]].add(scene["split"])
    leakage = [source for source, splits in source_splits.items() if len(splits) > 1]
    reference_mismatches = [
        source
        for source, splits in source_splits.items()
        if source in reference_split and splits != {reference_split[source]}
    ]
    receipt = {
        "format": "qces_speech_event_hard_dataset_receipt_v2",
        "complete": not leakage,
        "seed": args.seed,
        "scene_count": len(scenes),
        "question_count": len(questions),
        "scenes_by_split": dict(collections.Counter(scene["split"] for scene in scenes)),
        "difficulty_counts": dict(collections.Counter(scene["difficulty"] for scene in scenes)),
        "requested_snr_db_values": sorted({scene["requested_speech_to_overlap_noise_snr_db"] for scene in scenes}),
        "mean_measured_first_speech_to_overlap_noise_snr_db": float(np.mean(actual_snrs)),
        "single_speaker": True,
        "noise_class_count": len(SELECTED_LABELS),
        "noise_classes": list(SELECTED_LABELS),
        "unique_speech_utterances": required_speech,
        "unique_noise_sources": len({event["source_id"] for scene in scenes for event in scene["events"] if event["event_kind"] == "sound_event"}),
        "cross_split_source_leakage_count": len(leakage),
        "cross_branch_split_mismatch_count": len(reference_mismatches),
        "split_reference": _portable(args.split_reference.resolve()),
        "maximum_reconstruction_rms": max(reconstruction_rms),
        "scene_manifest": _portable(scene_path),
        "question_manifest": _portable(question_path),
        "scene_manifest_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        "question_manifest_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
    }
    _atomic_text(output_dir / "dataset_receipt.json", json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if leakage or reference_mismatches:
        raise RuntimeError(
            f"split failure leakage={leakage[:5]} reference={reference_mismatches[:5]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
