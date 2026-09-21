#!/usr/bin/env python3
"""Materialize source-disjoint pair/triple-overlap data for slot-query tuning.

The number of overlap scenes mirrors the historical non-overlap train/dev
scene counts.  Combining both stores therefore gives exactly 50% sequential,
25% pair-overlap, and 25% triple-overlap exposure (up to one odd scene).
Official-train sources are used for train, the pre-existing source-disjoint
development partition for dev, and official-eval/test sources are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as AF

from mixi_understanding.qces.clean_evidence_scenes import (
    CleanSource,
    _atomic_write_text,
    load_source_bank,
    partition_sources,
    sha256_file,
)


FORMAT = "qces_overlap_query_train_dev_v1"
DEFAULT_SEED = 2093
SAMPLE_RATE = 16_000
SCENE_SECONDS = 10.0
NUM_SAMPLES = int(SAMPLE_RATE * SCENE_SECONDS)
FRAME_HOP_SECONDS = 0.04
NUM_FRAMES = int(SCENE_SECONDS / FRAME_HOP_SECONDS)
MINIMUM_EVENT_FRAMES = 4
MAXIMUM_EVENT_FRAMES = 30
OVERLAP_TIERS = (0.25, 0.50, 0.75, 0.90)


HARD_GROUPS = (
    ("Artillery_fire", "Cap_gun", "Firecracker", "Machine_gun", "Slam", "Thunk", "Whack_and_thwack", "Clang", "Glass_shatter"),
    ("Alarm_clock", "Beep_and_bleep", "Busy_signal", "Car_alarm", "Fire_alarm", "Reversing_beeps", "Ringtone", "Dial_tone"),
    ("Air_horn_and_truck_horn", "Civil_defense_siren", "Fire_engine_and_fire_truck_(siren)", "Police_car_(siren)", "Train_horn", "Train_whistle"),
    ("Single-lens_reflex_camera", "Keys_jangling", "Coin_(dropping)", "Computer_keyboard", "Typewriter", "Tick", "Scissors"),
    ("Meow", "Purr", "Caterwaul", "Bark", "Growling", "Howl", "Whimper_(dog)"),
    ("Female_speech_and_woman_speaking", "Male_speech_and_man_speaking", "Child_speech_and_kid_speaking", "Babbling", "Whispering"),
)


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=data / "source_bank_accepted.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_191.txt")
    parser.add_argument("--nonoverlap-train-scenes", type=Path, default=data / "multievent/scene_ids_train.txt")
    parser.add_argument("--nonoverlap-dev-scenes", type=Path, default=data / "multievent/scene_ids_dev.txt")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v1",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _stable_seed(seed: int, *values: Any) -> int:
    payload = "\0".join(str(value) for value in (FORMAT, seed, *values))
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _load_scene_count(path: Path) -> int:
    count = sum(bool(line.strip()) for line in path.resolve().read_text(encoding="utf-8").splitlines())
    if count < 1:
        raise ValueError(f"empty scene list: {path}")
    return count


def _labels_for(
    labels: Sequence[str], anchor: str, count: int, rng: random.Random
) -> list[str]:
    selected = [anchor]
    matching = [group for group in HARD_GROUPS if anchor in group]
    if matching:
        hard = [label for label in rng.choice(matching) if label != anchor and label in labels]
        rng.shuffle(hard)
        selected.extend(hard[: min(2, count - 1)])
    remaining = [label for label in labels if label not in selected]
    rng.shuffle(remaining)
    selected.extend(remaining[: count - len(selected)])
    return selected


def _load_crop(source: CleanSource, rng: random.Random) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(source.audio_path)
    mono = waveform.float().mean(dim=0)
    if int(sample_rate) != SAMPLE_RATE:
        mono = AF.resample(mono, int(sample_rate), SAMPLE_RATE)
    start = max(0, int(round(source.active_onset_seconds * SAMPLE_RATE)))
    end = min(mono.numel(), int(round(source.active_offset_seconds * SAMPLE_RATE)))
    if end <= start:
        raise ValueError(f"invalid active crop: {source.source_id}")
    crop = mono[start:end]
    maximum = MAXIMUM_EVENT_FRAMES * int(round(FRAME_HOP_SECONDS * SAMPLE_RATE))
    if crop.numel() > maximum:
        crop_start = rng.randint(0, crop.numel() - maximum)
        crop = crop[crop_start: crop_start + maximum]
    minimum = MINIMUM_EVENT_FRAMES * int(round(FRAME_HOP_SECONDS * SAMPLE_RATE))
    if crop.numel() < minimum:
        crop = F.interpolate(
            crop.view(1, 1, -1), size=minimum, mode="linear", align_corners=False
        ).reshape(-1)
    frames = max(
        MINIMUM_EVENT_FRAMES,
        min(MAXIMUM_EVENT_FRAMES, int(math.ceil(crop.numel() / SAMPLE_RATE / FRAME_HOP_SECONDS))),
    )
    target_samples = frames * int(round(FRAME_HOP_SECONDS * SAMPLE_RATE))
    if crop.numel() != target_samples:
        crop = F.interpolate(
            crop.view(1, 1, -1), size=target_samples, mode="linear", align_corners=False
        ).reshape(-1)
    rms = crop.square().mean().sqrt().clamp_min(1e-5)
    gain_db = rng.choice((-9.0, -6.0, -3.0, 0.0, 3.0))
    return crop / rms * (0.06 * (10.0 ** (gain_db / 20.0)))


def _cluster_starts(
    lengths: Sequence[int], *, kind: str, overlap: float, rng: random.Random
) -> list[int]:
    cluster_size = 2 if kind == "pair_overlap" else 3
    starts: list[int] = []
    cursor = rng.randint(5, 15)
    for cluster_begin in range(0, len(lengths), cluster_size):
        cluster_lengths = list(lengths[cluster_begin: cluster_begin + cluster_size])
        minimum = min(cluster_lengths)
        if len(cluster_lengths) == 1:
            local = [cursor]
        elif cluster_size == 2 or len(cluster_lengths) == 2:
            step = max(1, int(round((1.0 - overlap) * minimum)))
            step = min(step, minimum - 1)
            local = [cursor, cursor + step]
        else:
            # The intersection shared by all three events is approximately
            # ``overlap`` of the shortest event, with unique onset frames.
            step = max(1, int(round((1.0 - overlap) * minimum / 2.0)))
            step = min(step, max(1, (minimum - 1) // 2))
            local = [cursor, cursor + step, cursor + 2 * step]
        starts.extend(local)
        cluster_end = max(start + length for start, length in zip(local, cluster_lengths, strict=True))
        cursor = cluster_end + rng.randint(5, 15)
    if max(start + length for start, length in zip(starts, lengths, strict=True)) > NUM_FRAMES:
        # With max 1.2-s events this is rare; deterministic left packing keeps
        # the same overlaps without truncating any event.
        shift = min(starts) - 2
        starts = [start - shift for start in starts]
    if max(start + length for start, length in zip(starts, lengths, strict=True)) > NUM_FRAMES:
        raise RuntimeError("overlap layout exceeds the fixed scene")
    return starts


def _maximum_concurrency(events: Sequence[Mapping[str, Any]]) -> int:
    points = []
    for event in events:
        points.extend(((event["onset_seconds"], 1), (event["offset_seconds"], -1)))
    active = maximum = 0
    for _, delta in sorted(points, key=lambda row: (row[0], row[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _render_split(
    *,
    split: str,
    scene_count: int,
    sources: Sequence[CleanSource],
    labels: Sequence[str],
    label_to_id: Mapping[str, int],
    staging: Path,
    final_root: Path,
    seed: int,
) -> list[dict[str, Any]]:
    by_label: dict[str, list[CleanSource]] = defaultdict(list)
    for source in sources:
        if source.label in label_to_id:
            by_label[source.label].append(source)
    missing = [label for label in labels if not by_label[label]]
    if missing:
        raise ValueError(f"{split} source pool misses labels: {missing[:10]}")
    scenes = []
    audio_dir = staging / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    for scene_index in range(scene_count):
        kind = "pair_overlap" if scene_index % 2 == 0 else "triple_overlap"
        overlap = OVERLAP_TIERS[(scene_index // 2) % len(OVERLAP_TIERS)]
        rng = random.Random(_stable_seed(seed, split, scene_index))
        count = rng.randint(3, 6)
        anchor = labels[scene_index % len(labels)]
        scene_labels = _labels_for(labels, anchor, count, rng)
        selected = [rng.choice(by_label[label]) for label in scene_labels]
        crops = [_load_crop(source, rng) for source in selected]
        hop_samples = int(round(FRAME_HOP_SECONDS * SAMPLE_RATE))
        lengths = [crop.numel() // hop_samples for crop in crops]
        starts = _cluster_starts(lengths, kind=kind, overlap=overlap, rng=rng)
        mixture = torch.zeros(NUM_SAMPLES, dtype=torch.float32)
        events = []
        scene_id = f"overlap_query_{split}_{scene_index:07d}"
        for event_index, (label, source, crop, start, length) in enumerate(
            zip(scene_labels, selected, crops, starts, lengths, strict=True)
        ):
            begin = start * hop_samples
            mixture[begin: begin + crop.numel()] += crop
            events.append(
                {
                    "event_id": f"{scene_id}:e{event_index:02d}",
                    "event_kind": "semantic",
                    "label": label,
                    "label_id": int(label_to_id[label]),
                    "onset_seconds": start * FRAME_HOP_SECONDS,
                    "offset_seconds": (start + length) * FRAME_HOP_SECONDS,
                    "onset_frame": start,
                    "offset_frame": start + length,
                    "source_id": source.source_id,
                    "source_video_id": source.video_id,
                    "source_sha256": source.source_sha256,
                    "source_path": source.audio_path,
                    "cleanliness_tier": source.cleanliness_tier,
                }
            )
        peak = float(mixture.abs().max())
        if peak > 0.95:
            mixture *= 0.95 / peak
        relative_audio = Path("audio") / split / f"{scene_id}.wav"
        torchaudio.save(
            str(staging / relative_audio),
            mixture.unsqueeze(0),
            SAMPLE_RATE,
            encoding="PCM_S",
            bits_per_sample=16,
        )
        events.sort(key=lambda event: (event["onset_seconds"], event["event_id"]))
        maximum_concurrency = _maximum_concurrency(events)
        expected = 2 if kind == "pair_overlap" else min(3, count)
        if maximum_concurrency != expected:
            raise RuntimeError(
                f"{scene_id}: concurrency={maximum_concurrency}, expected={expected}"
            )
        scenes.append(
            {
                "format": "qces_clean_evidence_scene_v1",
                "scene_id": scene_id,
                "scene_family_id": scene_id,
                "split": split,
                "source_route": "synthetic_clean_single_event_bank_overlap_query_training",
                "mixture_path": str((final_root / relative_audio).resolve()),
                "duration_seconds": SCENE_SECONDS,
                "sample_rate": SAMPLE_RATE,
                "audio_num_frames": NUM_SAMPLES,
                "audio_num_channels": 1,
                "audio_sha256": sha256_file(staging / relative_audio),
                "events": events,
                "recipe_kind": kind,
                "requested_overlap_fraction": overlap,
                "maximum_concurrency": maximum_concurrency,
                "rendered": True,
            }
        )
        if (scene_index + 1) % 250 == 0:
            print(f"rendered {split}: {scene_index + 1}/{scene_count}", flush=True)
    return scenes


def _identities(sources: Sequence[CleanSource]) -> set[tuple[str, str]]:
    return {identity for source in sources for identity in source.hard_identities}


def main() -> None:
    args = parse_args()
    labels = [line.strip() for line in args.ontology.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 191 or len(labels) != len(set(labels)):
        raise ValueError(f"requires frozen 191-label ontology, got {len(labels)}")
    train_count = _load_scene_count(args.nonoverlap_train_scenes)
    dev_count = _load_scene_count(args.nonoverlap_dev_scenes)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    sources = load_source_bank(args.source_bank.resolve(), require_audio_file=True)
    partitioned, partition_receipt = partition_sources(sources, labels, seed=2041, dev_fraction=0.20)
    train_sources = list(partitioned["train"])
    dev_sources = list(partitioned["dev"])
    test_sources = list(partitioned["test"])
    overlaps = {
        "train_dev": len(_identities(train_sources) & _identities(dev_sources)),
        "train_test": len(_identities(train_sources) & _identities(test_sources)),
        "dev_test": len(_identities(dev_sources) & _identities(test_sources)),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"source partition identity leakage: {overlaps}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        label_to_id = {label: index for index, label in enumerate(labels)}
        scenes = {
            "train": _render_split(
                split="train", scene_count=train_count, sources=train_sources,
                labels=labels, label_to_id=label_to_id, staging=staging,
                final_root=output_dir, seed=args.seed,
            ),
            "dev": _render_split(
                split="dev", scene_count=dev_count, sources=dev_sources,
                labels=labels, label_to_id=label_to_id, staging=staging,
                final_root=output_dir, seed=args.seed,
            ),
        }
        artifacts = {}
        for split in ("train", "dev"):
            manifest = staging / f"detector_scene_manifest_overlap_{split}.jsonl"
            ids = staging / f"scene_ids_overlap_{split}.txt"
            _atomic_write_text(manifest, _jsonl(scenes[split]))
            _atomic_write_text(ids, "".join(f"{scene['scene_id']}\n" for scene in scenes[split]))
            artifacts[manifest.name] = sha256_file(manifest)
            artifacts[ids.name] = sha256_file(ids)
        ontology = staging / "ontology_191.txt"
        _atomic_write_text(ontology, "".join(f"{label}\n" for label in labels))
        artifacts[ontology.name] = sha256_file(ontology)

        split_stats = {}
        for split in ("train", "dev"):
            rows = scenes[split]
            label_counts = Counter(event["label"] for scene in rows for event in scene["events"])
            split_stats[split] = {
                "scenes": len(rows),
                "events": sum(len(scene["events"]) for scene in rows),
                "recipe_distribution": dict(sorted(Counter(scene["recipe_kind"] for scene in rows).items())),
                "concurrency_distribution": dict(sorted(Counter(scene["maximum_concurrency"] for scene in rows).items())),
                "overlap_tier_distribution": dict(sorted(Counter(f"{scene['requested_overlap_fraction']:.2f}" for scene in rows).items())),
                "classes_present": len(label_counts),
                "minimum_events_per_class": min(label_counts.values()),
                "maximum_events_per_class": max(label_counts.values()),
                "unique_source_rows_used": len({event["source_id"] for scene in rows for event in scene["events"]}),
            }
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "policy": {
                "combined_training_exposure": {
                    "historical_nonoverlap": 0.50,
                    "pair_overlap": 0.25,
                    "triple_overlap": 0.25,
                },
                "answer_label_used": False,
                "official_eval_sources_used": False,
                "source_reuse_within_training_split": "allowed augmentation",
                "train_dev_source_reuse": False,
                "event_count_range": [3, 6],
                "overlap_tiers": list(OVERLAP_TIERS),
            },
            "nonoverlap_reference_scene_counts": {"train": train_count, "dev": dev_count},
            "overlap_scene_counts": {"train": train_count, "dev": dev_count},
            "split_stats": split_stats,
            "hard_identity_overlap": overlaps,
            "partition_policy": partition_receipt["policy"],
            "artifacts": artifacts,
        }
        _atomic_write_text(
            staging / "build_receipt.json",
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
        staging = None
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
