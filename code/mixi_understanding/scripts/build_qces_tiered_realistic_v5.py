#!/usr/bin/env python3
"""Build the realistic-main tier paired with the frozen V4 stress tier.

V4 deliberately places long events on one dense timeline.  That is useful as
an adversarial stress test, but it leaves almost no lightly-overlapped examples.
This sidecar builder keeps the same Gold-only 188-class source contract and the
pre-registered 4.48-second crop cap while controlling scene density explicitly:

* sparse: 3--4 events, low overlap, maximum concurrency 2;
* moderate: 4--5 events, partial overlap, maximum concurrency 2;
* hard: 5--6 events, heavy overlap, maximum concurrency 3.

Every layout is selected by deterministic random search against an event-level
overlap target.  Active crops are RMS-normalized before per-event gain sampling.
The manifest stores measured overlap, interference SIR, and exact component
stems so the intended curriculum can be audited rather than inferred.
"""

from __future__ import annotations

import argparse
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

import soundfile as sf
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.qces.clean_evidence_scenes import (
    CleanSource,
    _atomic_write_text,
    load_source_bank,
    partition_sources,
    sha256_file,
)
from mixi_understanding.scripts import build_qces_overlap_gold_natural_v3 as v3
from mixi_understanding.scripts import build_qces_overlap_query_train_dev_v1 as v1
from mixi_understanding.scripts import build_qces_overlap_semantic_sufficient_v4 as v4


FORMAT = "qces_tiered_realistic_v5"
COMPONENT_FORMAT = "qces_tiered_realistic_component_v5"
DEFAULT_SEED = 2505
SAMPLE_RATE = v4.SAMPLE_RATE
FRAME_HOP_SECONDS = v4.FRAME_HOP_SECONDS
HOP_SAMPLES = v4.HOP_SAMPLES
NUM_FRAMES = v4.NUM_FRAMES
NUM_SAMPLES = v4.NUM_SAMPLES
MAX_EVENT_FRAMES = v4.MAX_EVENT_FRAMES

PROFILES: dict[str, dict[str, Any]] = {
    "sparse": {
        "fraction": 0.55,
        "counts": (3,),
        "target_overlap": 0.12,
        "maximum_concurrency": 2,
        "gain_db": (-3.0, 0.0, 3.0),
    },
    "moderate": {
        "fraction": 0.35,
        "counts": (4,),
        "target_overlap": 0.35,
        "maximum_concurrency": 2,
        "gain_db": (-6.0, -3.0, 0.0, 3.0, 6.0),
    },
    "hard": {
        "fraction": 0.10,
        "counts": (5, 6),
        "target_overlap": 0.78,
        "maximum_concurrency": 3,
        "gain_db": (-6.0, -3.0, 0.0, 3.0, 6.0),
    },
}


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=data / "source_bank_accepted.jsonl")
    parser.add_argument("--source-ontology", type=Path, default=data / "ontology_191.txt")
    parser.add_argument("--reference-single-train", type=Path, default=data / "detector_scene_manifest_single_train.jsonl")
    parser.add_argument("--reference-test", type=Path, default=data / "detector_scene_manifest_multi_test.jsonl")
    parser.add_argument("--reference-train-scenes", type=Path, default=data / "multievent/scene_ids_train.txt")
    parser.add_argument("--reference-dev-scenes", type=Path, default=data / "multievent/scene_ids_dev.txt")
    parser.add_argument("--duration-audit", type=Path, default=PROJECT_ROOT / "outputs/qces_pretrainedsed_detector_full188/gold_natural_v3_headtransfer_r1/semantic_duration_curve_5s_v1/report.json")
    parser.add_argument("--stress-tier", type=Path, default=PROJECT_ROOT / "outputs/qces_full188_overlap_semantic_sufficient_v4")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_full188_tiered_realistic_v5")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-gold-train-sources", type=int, default=v3.MIN_GOLD_TRAIN_SOURCES)
    parser.add_argument("--min-gold-dev-sources", type=int, default=v3.MIN_GOLD_DEV_SOURCES)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--layout-trials", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _identities(sources: Sequence[CleanSource]) -> set[tuple[str, str]]:
    return {identity for source in sources for identity in source.hard_identities}


def _profile_for(index: int) -> str:
    # A fixed 20-scene cycle realizes 55/35/10 exactly without random drift.
    position = index % 20
    if position < 11:
        return "sparse"
    if position < 18:
        return "moderate"
    return "hard"


def _occupancy(starts: Sequence[int], lengths: Sequence[int]) -> torch.Tensor:
    occupancy = torch.zeros(NUM_FRAMES, dtype=torch.int16)
    for start, length in zip(starts, lengths, strict=True):
        occupancy[int(start) : int(start) + int(length)] += 1
    return occupancy


def event_overlap_fractions(starts: Sequence[int], lengths: Sequence[int]) -> list[float]:
    occupancy = _occupancy(starts, lengths)
    return [
        float((occupancy[int(start) : int(start) + int(length)] > 1).float().mean())
        for start, length in zip(starts, lengths, strict=True)
    ]


def _layout_score(
    starts: Sequence[int],
    lengths: Sequence[int],
    *,
    target_overlap: float,
    maximum_concurrency: int,
) -> tuple[float, dict[str, Any]]:
    occupancy = _occupancy(starts, lengths)
    overlaps = event_overlap_fractions(starts, lengths)
    observed_maximum = int(occupancy.max())
    mean_overlap = sum(overlaps) / len(overlaps)
    overflow = max(0, observed_maximum - maximum_concurrency)
    # Prefer layouts near the requested event-level overlap, strongly reject
    # excess concurrency, and weakly discourage unused tails of the scene.
    span = max(int(s) + int(n) for s, n in zip(starts, lengths, strict=True)) - min(starts)
    score = (
        10.0 * abs(mean_overlap - target_overlap)
        + 50.0 * overflow
        + 0.15 * abs(span / NUM_FRAMES - min(0.92, 0.55 + target_overlap / 2.0))
    )
    return score, {
        "overlap_fractions": overlaps,
        "mean_overlap": mean_overlap,
        "maximum_concurrency": observed_maximum,
    }


def _structured_candidate(lengths: Sequence[int], rng: random.Random) -> list[int]:
    """Create a near-sequential candidate; compression introduces mild overlap."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    gaps = [rng.randint(1, 8) for _ in range(max(0, len(lengths) - 1))]
    raw = [0] * len(lengths)
    cursor = rng.randint(2, 8)
    for rank, event_index in enumerate(order):
        raw[event_index] = cursor
        cursor += int(lengths[event_index])
        if rank < len(gaps):
            cursor += gaps[rank]
    maximum_end = max(s + int(n) for s, n in zip(raw, lengths, strict=True))
    if maximum_end >= NUM_FRAMES - 2:
        origin = min(raw)
        scale = (NUM_FRAMES - 5 - origin) / max(maximum_end - origin, 1)
        raw = [origin + int(round((start - origin) * scale)) for start in raw]
    return v4._deduplicate_starts(raw, lengths)


def _laned_candidate(
    lengths: Sequence[int], *, lane_count: int, rng: random.Random
) -> list[int]:
    """Schedule non-overlapping events per lane, bounding global concurrency."""
    lanes: list[list[int]] = [[] for _ in range(lane_count)]
    lane_loads = [0] * lane_count
    order = sorted(range(len(lengths)), key=lambda index: (-int(lengths[index]), rng.random()))
    for event_index in order:
        lane = min(range(lane_count), key=lambda index: lane_loads[index])
        lanes[lane].append(event_index)
        lane_loads[lane] += int(lengths[event_index])
    starts = [0] * len(lengths)
    used: set[int] = set()
    for lane_index, lane_events in enumerate(lanes):
        rng.shuffle(lane_events)
        content = sum(int(lengths[index]) for index in lane_events)
        room = NUM_FRAMES - content - 4
        if room < 0:
            raise RuntimeError("events cannot fit within the requested lane bound")
        gaps = len(lane_events) - 1
        gap_budget = min(room, gaps * rng.randint(1, 8)) if gaps else 0
        base = 2 + (rng.randint(0, max(0, room - gap_budget)) if room > gap_budget else 0)
        cursor = base
        remaining_gap = gap_budget
        for rank, event_index in enumerate(lane_events):
            # Avoid tied onsets across lanes without changing within-lane order.
            while cursor in used and cursor + int(lengths[event_index]) < NUM_FRAMES - 1:
                cursor += 1
            starts[event_index] = cursor
            used.add(cursor)
            cursor += int(lengths[event_index])
            if rank < gaps:
                slots = gaps - rank
                gap = remaining_gap // slots
                cursor += gap
                remaining_gap -= gap
    return starts


def choose_layout(
    lengths: Sequence[int], *, profile: str, rng: random.Random, trials: int
) -> tuple[list[int], dict[str, Any]]:
    config = PROFILES[profile]
    best: tuple[float, list[int], dict[str, Any]] | None = None
    for trial in range(max(32, trials)):
        if trial < max(16, trials // 4):
            starts = _laned_candidate(
                lengths,
                lane_count=int(config["maximum_concurrency"]),
                rng=rng,
            )
        elif trial < max(24, trials // 2):
            starts = _structured_candidate(lengths, rng)
        else:
            starts = [rng.randint(2, NUM_FRAMES - int(length) - 2) for length in lengths]
            starts = v4._deduplicate_starts(starts, lengths)
        score, measurements = _layout_score(
            starts,
            lengths,
            target_overlap=float(config["target_overlap"]),
            maximum_concurrency=int(config["maximum_concurrency"]),
        )
        candidate = (score, starts, measurements)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    score, starts, measurements = best
    if measurements["maximum_concurrency"] > int(config["maximum_concurrency"]):
        raise RuntimeError(f"could not satisfy {profile} concurrency after {trials} trials")
    measurements["layout_score"] = score
    return starts, measurements


def _save_component(path: Path, waveform: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy(), SAMPLE_RATE, subtype="FLOAT")
    return sha256_file(path)


def _sir_db(
    target: torch.Tensor,
    target_start: int,
    all_crops: Sequence[torch.Tensor],
    all_starts: Sequence[int],
    target_index: int,
) -> float:
    begin = int(target_start) * HOP_SAMPLES
    end = begin + target.numel()
    interference = torch.zeros_like(target)
    for index, (crop, start_frame) in enumerate(zip(all_crops, all_starts, strict=True)):
        if index == target_index:
            continue
        other_begin = int(start_frame) * HOP_SAMPLES
        left = max(begin, other_begin)
        right = min(end, other_begin + crop.numel())
        if right > left:
            interference[left - begin : right - begin] += crop[left - other_begin : right - other_begin]
    target_energy = float(target.square().sum())
    interference_energy = float(interference.square().sum())
    return min(40.0, 10.0 * math.log10((target_energy + 1e-10) / (interference_energy + 1e-10)))


def _distribution(values: Sequence[float], bins: Sequence[tuple[str, float, float]]) -> dict[str, Any]:
    counts = Counter()
    for value in values:
        for name, lower, upper in bins:
            if lower <= value < upper:
                counts[name] += 1
                break
    return {
        name: {"count": counts[name], "fraction": counts[name] / max(len(values), 1)}
        for name, _, _ in bins
    }


def render_split(
    *,
    split: str,
    scene_count: int,
    sources: Sequence[CleanSource],
    labels: Sequence[str],
    label_to_id: Mapping[str, int],
    staging: Path,
    final_root: Path,
    seed: int,
    layout_trials: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_label: dict[str, list[CleanSource]] = defaultdict(list)
    for source in sources:
        if source.label in label_to_id and v3._gold(source):
            by_label[source.label].append(source)
    if missing := [label for label in labels if not by_label[label]]:
        raise ValueError(f"{split} Gold pool misses eligible labels: {missing}")

    scenes: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    all_overlaps: list[float] = []
    all_sirs: list[float] = []
    for scene_index in range(scene_count):
        profile = _profile_for(scene_index)
        config = PROFILES[profile]
        rng = random.Random(v3._stable_seed(seed, split, scene_index, FORMAT))
        event_count = rng.choice(config["counts"])
        anchor = labels[scene_index % len(labels)]
        scene_labels = v3._labels_for(labels, anchor, event_count, rng)
        selected = [rng.choice(by_label[label]) for label in scene_labels]
        gains = [rng.choice(config["gain_db"]) for _ in selected]
        loaded = [
            v3.load_natural_crop(source, max_event_frames=MAX_EVENT_FRAMES, gain_db=gain)
            for source, gain in zip(selected, gains, strict=True)
        ]
        crops = [item[0] for item in loaded]
        metadata = [item[1] for item in loaded]
        lengths = [int(item["frames"]) for item in metadata]
        starts, layout = choose_layout(lengths, profile=profile, rng=rng, trials=layout_trials)
        overlaps = list(layout["overlap_fractions"])
        sirs = [
            _sir_db(crop, start, crops, starts, event_index)
            for event_index, (crop, start) in enumerate(zip(crops, starts, strict=True))
        ]
        all_overlaps.extend(overlaps)
        all_sirs.extend(sirs)

        scene_id = f"tiered_realistic_v5_{split}_{scene_index:07d}"
        mixture = torch.zeros(NUM_SAMPLES, dtype=torch.float32)
        events: list[dict[str, Any]] = []
        for event_index, (label, source, crop, item_metadata, gain, start, length, overlap, sir) in enumerate(
            zip(scene_labels, selected, crops, metadata, gains, starts, lengths, overlaps, sirs, strict=True)
        ):
            begin = int(start) * HOP_SAMPLES
            mixture[begin : begin + crop.numel()] += crop
            events.append(
                {
                    "event_id": f"{scene_id}:e{event_index:02d}",
                    "event_kind": "semantic",
                    "label": label,
                    "label_id": int(label_to_id[label]),
                    "onset_seconds": start * FRAME_HOP_SECONDS,
                    "offset_seconds": (start + length) * FRAME_HOP_SECONDS,
                    "onset_frame": int(start),
                    "offset_frame": int(start + length),
                    "source_id": source.source_id,
                    "source_video_id": source.video_id,
                    "source_sha256": source.source_sha256,
                    "source_path": source.audio_path,
                    "cleanliness_tier": "gold",
                    "source_gain_db": float(gain),
                    "overlap_fraction": float(overlap),
                    "active_sir_db": float(sir),
                    **item_metadata,
                }
            )
        peak = float(mixture.abs().max())
        global_scale = 0.95 / peak if peak > 0.95 else 1.0
        mixture *= global_scale
        relative_audio = Path("audio") / split / f"{scene_id}.wav"
        audio_path = staging / relative_audio
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(audio_path), mixture.unsqueeze(0), SAMPLE_RATE, encoding="PCM_S", bits_per_sample=16)

        reconstructed = torch.zeros_like(mixture)
        for event_index, (event, crop) in enumerate(zip(events, crops, strict=True)):
            scaled = crop * global_scale
            begin = int(event["onset_frame"]) * HOP_SAMPLES
            reconstructed[begin : begin + scaled.numel()] += scaled
            relative = Path("components") / split / scene_id / f"e{event_index:02d}.wav"
            component_path = staging / relative
            component_sha = _save_component(component_path, scaled)
            event["component_path"] = str((final_root / relative).resolve())
            event["component_sha256"] = component_sha
            event["global_scene_scale"] = global_scale
            components.append(
                {
                    "format": COMPONENT_FORMAT,
                    "split": split,
                    "scene_id": scene_id,
                    "event_id": event["event_id"],
                    "label": event["label"],
                    "label_id": event["label_id"],
                    "component_path": event["component_path"],
                    "component_sha256": component_sha,
                    "num_component_samples": scaled.numel(),
                    "sample_rate": SAMPLE_RATE,
                    "onset_frame": event["onset_frame"],
                    "offset_frame": event["offset_frame"],
                    "onset_sample": begin,
                    "offset_sample": begin + scaled.numel(),
                    "source_id": event["source_id"],
                    "source_sha256": event["source_sha256"],
                    "cleanliness_tier": "gold",
                    "source_gain_db": event["source_gain_db"],
                    "overlap_fraction": event["overlap_fraction"],
                    "active_sir_db": event["active_sir_db"],
                    "duration_equalized": False,
                    "time_stretched": False,
                    "crop_was_capped": event["crop_was_capped"],
                    "tail_zero_padding_samples": event["tail_zero_padding_samples"],
                    "layout_profile": profile,
                    "global_scene_scale": global_scale,
                }
            )
        decoded, rate = torchaudio.load(audio_path)
        if int(rate) != SAMPLE_RATE:
            raise RuntimeError("stored sample rate changed")
        error = float((decoded.mean(dim=0) - reconstructed).abs().max())
        if error > 2.0 / 32768.0:
            raise RuntimeError(f"{scene_id}: component reconstruction error={error}")
        reconstruction_errors.append(error)
        events.sort(key=lambda item: (item["onset_seconds"], item["event_id"]))
        tied = len(events) - len({int(event["onset_frame"]) for event in events})
        scenes.append(
            {
                "format": "qces_clean_evidence_scene_v1",
                "scene_id": scene_id,
                "scene_family_id": scene_id,
                "split": split,
                "source_route": "synthetic_gold_tiered_realistic_v5",
                "mixture_path": str((final_root / relative_audio).resolve()),
                "duration_seconds": 10.0,
                "sample_rate": SAMPLE_RATE,
                "audio_num_frames": NUM_SAMPLES,
                "audio_num_channels": 1,
                "audio_sha256": sha256_file(audio_path),
                "events": events,
                "layout_profile": profile,
                "target_event_overlap_fraction": config["target_overlap"],
                "actual_mean_event_overlap_fraction": layout["mean_overlap"],
                "maximum_concurrency": layout["maximum_concurrency"],
                "layout_score": layout["layout_score"],
                "tied_onset_events": tied,
                "duration_policy": "individual active duration retained up to 4.48s; no time stretching",
                "layout_policy": "deterministic tiered random search with profile-specific overlap and concurrency targets",
                "rendered": True,
            }
        )
        if (scene_index + 1) % 250 == 0:
            print(f"rendered {split}: {scene_index + 1}/{scene_count}", flush=True)

    overlap_bins = (("zero", -1e-9, 1e-9), ("light_(0,.25]", 1e-9, 0.2500001), ("moderate_(.25,.75]", 0.2500001, 0.7500001), ("heavy_(.75,1]", 0.7500001, 1.0000001))
    sir_bins = (("below_-10", -1e9, -10.0), ("[-10,0)", -10.0, 0.0), ("[0,10)", 0.0, 10.0), ("at_least_10", 10.0, 1e9))
    label_counts = Counter(str(row["label"]) for row in components)
    return scenes, components, {
        "scenes": len(scenes),
        "events": len(components),
        "unique_sources": len({row["source_id"] for row in components}),
        "profile_scenes": dict(Counter(row["layout_profile"] for row in scenes)),
        "class_coverage": {
            "classes": len(label_counts),
            "minimum_events_per_class": min(label_counts.values()),
            "maximum_events_per_class": max(label_counts.values()),
        },
        "maximum_reconstruction_error": max(reconstruction_errors),
        "capped_events": sum(bool(row["crop_was_capped"]) for row in components),
        "tail_padded_events": sum(int(row["tail_zero_padding_samples"]) > 0 for row in components),
        "scenes_with_tied_onsets": sum(int(row["tied_onset_events"]) > 0 for row in scenes),
        "maximum_concurrency": {
            "minimum": min(int(row["maximum_concurrency"]) for row in scenes),
            "maximum": max(int(row["maximum_concurrency"]) for row in scenes),
            "mean": sum(int(row["maximum_concurrency"]) for row in scenes) / len(scenes),
        },
        "event_overlap_fraction": {
            "minimum": min(all_overlaps),
            "maximum": max(all_overlaps),
            "mean": sum(all_overlaps) / len(all_overlaps),
            "bins": _distribution(all_overlaps, overlap_bins),
        },
        "active_sir_db": {
            "minimum": min(all_sirs),
            "maximum": max(all_sirs),
            "mean": sum(all_sirs) / len(all_sirs),
            "bins": _distribution(all_sirs, sir_bins),
        },
    }


def main() -> None:
    args = parse_args()
    duration_audit = json.loads(args.duration_audit.resolve().read_text(encoding="utf-8"))
    if duration_audit.get("selected_condition") != "cap_4.5s" or not duration_audit.get("gate_passed"):
        raise ValueError("the frozen duration-selection audit did not pass")
    stress_receipt_path = args.stress_tier.resolve() / "build_receipt.json"
    stress_receipt = json.loads(stress_receipt_path.read_text(encoding="utf-8"))
    if stress_receipt.get("format") != v4.FORMAT or stress_receipt.get("ontology", {}).get("classes") != 188:
        raise ValueError("V4 stress tier contract is unavailable")

    source_labels = [line.strip() for line in args.source_ontology.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    sources = load_source_bank(args.source_bank.resolve(), require_audio_file=True)
    partitioned, partition_receipt = partition_sources(sources, source_labels, seed=2041, dev_fraction=0.20)
    labels, coverage = v3.select_eligible_labels(
        source_labels,
        partitioned["train"],
        partitioned["dev"],
        min_train=args.min_gold_train_sources,
        min_dev=args.min_gold_dev_sources,
    )
    if labels != stress_receipt["ontology"]["labels"]:
        raise RuntimeError("V5 ontology differs from the frozen V4 stress tier")
    eligible = set(labels)
    split_sources = {
        split: [source for source in partitioned[split] if source.label in eligible and v3._gold(source)]
        for split in ("train", "dev", "test")
    }
    identity_overlap = {
        "train_dev": len(_identities(split_sources["train"]) & _identities(split_sources["dev"])),
        "train_test": len(_identities(split_sources["train"]) & _identities(split_sources["test"])),
        "dev_test": len(_identities(split_sources["dev"]) & _identities(split_sources["test"])),
    }
    if any(identity_overlap.values()):
        raise RuntimeError(f"identity leakage: {identity_overlap}")
    counts = {
        "train": v3._scene_count(args.reference_train_scenes.resolve(), args.max_train_scenes),
        "dev": v3._scene_count(args.reference_dev_scenes.resolve(), args.max_dev_scenes),
    }

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        label_to_id = {label: index for index, label in enumerate(labels)}
        outputs: dict[str, Any] = {}
        for split in ("train", "dev"):
            scenes, components, stats = render_split(
                split=split,
                scene_count=counts[split],
                sources=split_sources[split],
                labels=labels,
                label_to_id=label_to_id,
                staging=staging,
                final_root=output_dir,
                seed=args.seed,
                layout_trials=args.layout_trials,
            )
            _atomic_write_text(staging / f"detector_scene_manifest_tiered_{split}.jsonl", _jsonl(scenes))
            _atomic_write_text(staging / f"event_components_{split}.jsonl", _jsonl(components))
            _atomic_write_text(staging / f"scene_ids_tiered_{split}.txt", "".join(f"{row['scene_id']}\n" for row in scenes))
            outputs[split] = {"stats": stats}

        quality_gates: dict[str, Any] = {}
        for split in ("train", "dev"):
            stats = outputs[split]["stats"]
            overlap_bins = stats["event_overlap_fraction"]["bins"]
            light_fraction = overlap_bins["zero"]["fraction"] + overlap_bins["light_(0,.25]"]["fraction"]
            checks = {
                "all_188_classes_present_when_scene_count_sufficient": (
                    stats["class_coverage"]["classes"] == 188 if counts[split] >= 188 else True
                ),
                "maximum_concurrency_at_most_3": stats["maximum_concurrency"]["maximum"] <= 3,
                "heavy_overlap_fraction_at_most_0.30": overlap_bins["heavy_(.75,1]"]["fraction"] <= 0.30,
                "zero_or_light_overlap_fraction_at_least_0.40": light_fraction >= 0.40,
                "sir_below_minus_10_fraction_at_most_0.10": stats["active_sir_db"]["bins"]["below_-10"]["fraction"] <= 0.10,
                "no_tied_onsets": stats["scenes_with_tied_onsets"] == 0,
                "component_reconstruction_within_pcm_tolerance": stats["maximum_reconstruction_error"] <= 2.0 / 32768.0,
            }
            quality_gates[split] = {"passed": all(checks.values()), "checks": checks}
        if not all(item["passed"] for item in quality_gates.values()):
            raise RuntimeError(f"V5 quality gate failed: {quality_gates}")

        single = v3.filter_reference_manifest(args.reference_single_train.resolve(), eligible, gold_only=True)
        locked = v3.filter_reference_manifest(args.reference_test.resolve(), eligible, gold_only=False)
        _atomic_write_text(staging / "detector_scene_manifest_single_train_gold.jsonl", _jsonl(single))
        _atomic_write_text(staging / "detector_scene_manifest_locked_test_filtered.jsonl", _jsonl(locked))
        ontology_name = f"ontology_{len(labels)}.txt"
        _atomic_write_text(staging / ontology_name, "".join(f"{label}\n" for label in labels))
        names = [f"detector_scene_manifest_tiered_{split}.jsonl" for split in ("train", "dev")] + [f"event_components_{split}.jsonl" for split in ("train", "dev")] + [f"scene_ids_tiered_{split}.txt" for split in ("train", "dev")] + ["detector_scene_manifest_single_train_gold.jsonl", "detector_scene_manifest_locked_test_filtered.jsonl", ontology_name]
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "ontology": {"classes": len(labels), "labels": labels, "coverage_gate": coverage},
            "policy": {
                "role": "realistic main tier; V4 remains the frozen stress tier",
                "profile_mix": {name: config["fraction"] for name, config in PROFILES.items()},
                "profiles": PROFILES,
                "layout_trials": args.layout_trials,
                "model_predictions_used_for_source_or_label_selection": False,
                "answer_labels_used": False,
                "source_quality": "Gold only",
                "official_eval_sources_used_for_train_or_dev": False,
                "time_stretching": False,
                "maximum_active_crop_seconds": MAX_EVENT_FRAMES * FRAME_HOP_SECONDS,
                "rms_policy": "each active crop normalized by V4 loader, then profile-specific gain sampled",
            },
            "frozen_stress_tier": {
                "path": str(args.stress_tier.resolve()),
                "receipt_sha256": sha256_file(stress_receipt_path),
                "format": stress_receipt["format"],
            },
            "duration_audit": {
                "path": str(args.duration_audit.resolve()),
                "sha256": sha256_file(args.duration_audit.resolve()),
                "selected_condition": duration_audit["selected_condition"],
            },
            "partition_policy": partition_receipt["policy"],
            "hard_identity_overlap": identity_overlap,
            "quality_gates": quality_gates,
            "data": {
                "single_train_gold": len(single),
                "locked_test_filtered": len(locked),
                "train": outputs["train"]["stats"],
                "dev": outputs["dev"]["stats"],
            },
            "artifacts": {name: sha256_file(staging / name) for name in names},
        }
        _atomic_write_text(staging / "build_receipt.json", json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
        staging = None
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
