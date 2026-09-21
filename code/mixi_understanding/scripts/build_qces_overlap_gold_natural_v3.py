#!/usr/bin/env python3
"""Build Gold-only overlap scenes without shortest-member truncation.

V2 equalized every pair/triple cluster to its shortest event.  The locked
oracle-component audit showed a large recognizability loss for those shortened
events.  V3 fixes the data contract before changing the detector:

* ontology eligibility uses source counts only (never model predictions);
* every retained class has >=8 independent Gold train sources and >=3 Gold
  development sources;
* Silver sources are excluded from both train and development;
* pair/triple members retain their own duration -- no cluster equalization;
* active crops are never time-stretched; sub-frame tails are zero padded;
* long active regions use a deterministic highest-energy window up to 1.6 s;
* exact scaled component waveforms are materialized and verified to reconstruct
  each stored PCM16 mixture.

This is a sidecar builder and does not modify v1/v2 data.
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

import soundfile as sf
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
from mixi_understanding.scripts import build_qces_overlap_query_train_dev_v1 as v1


FORMAT = "qces_overlap_gold_natural_v3"
COMPONENT_FORMAT = "qces_overlap_gold_natural_component_v3"
DEFAULT_SEED = 2203
SAMPLE_RATE = 16_000
FRAME_HOP_SECONDS = 0.04
HOP_SAMPLES = int(round(SAMPLE_RATE * FRAME_HOP_SECONDS))
NUM_FRAMES = 250
NUM_SAMPLES = SAMPLE_RATE * 10
MIN_GOLD_TRAIN_SOURCES = 8
MIN_GOLD_DEV_SOURCES = 3
# Three pair-overlap clusters must fit in a fixed 10-second scene even at the
# lowest overlap tier.  Forty frames increases semantic context from v1/v2's
# 1.2 s to 1.6 s without introducing layout-dependent truncation.
MAX_EVENT_FRAMES = 40


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=data / "source_bank_accepted.jsonl")
    parser.add_argument("--source-ontology", type=Path, default=data / "ontology_191.txt")
    parser.add_argument("--reference-single-train", type=Path, default=data / "detector_scene_manifest_single_train.jsonl")
    parser.add_argument("--reference-test", type=Path, default=data / "detector_scene_manifest_multi_test.jsonl")
    parser.add_argument("--reference-train-scenes", type=Path, default=data / "multievent/scene_ids_train.txt")
    parser.add_argument("--reference-dev-scenes", type=Path, default=data / "multievent/scene_ids_dev.txt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_full188_overlap_gold_natural_v3",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-gold-train-sources", type=int, default=MIN_GOLD_TRAIN_SOURCES)
    parser.add_argument("--min-gold-dev-sources", type=int, default=MIN_GOLD_DEV_SOURCES)
    parser.add_argument("--max-event-frames", type=int, default=MAX_EVENT_FRAMES)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _stable_seed(seed: int, *values: Any) -> int:
    body = "\0".join(str(value) for value in (FORMAT, seed, *values))
    return int(hashlib.sha256(body.encode()).hexdigest()[:16], 16)


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _scene_count(path: Path, limit: int) -> int:
    count = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
    if count <= 0:
        raise ValueError(f"empty scene list: {path}")
    return min(count, limit) if limit > 0 else count


def _gold(source: CleanSource) -> bool:
    return (
        source.cleanliness_passed
        and source.audibility_passed
        and source.cleanliness_tier.strip().lower() == "gold"
    )


def select_eligible_labels(
    source_labels: Sequence[str],
    train_sources: Sequence[CleanSource],
    dev_sources: Sequence[CleanSource],
    *,
    min_train: int,
    min_dev: int,
) -> tuple[list[str], dict[str, Any]]:
    train_counts = Counter(source.label for source in train_sources if _gold(source))
    dev_counts = Counter(source.label for source in dev_sources if _gold(source))
    labels = [
        label
        for label in source_labels
        if train_counts[label] >= min_train and dev_counts[label] >= min_dev
    ]
    excluded = [
        {
            "label": label,
            "gold_train_sources": train_counts[label],
            "gold_dev_sources": dev_counts[label],
        }
        for label in source_labels
        if label not in set(labels)
    ]
    return labels, {
        "policy": "source-count-only; no model predictions; Gold sources only",
        "minimum_gold_train_sources": min_train,
        "minimum_gold_dev_sources": min_dev,
        "source_labels": len(source_labels),
        "eligible_labels": len(labels),
        "excluded": excluded,
        "eligible_train_count_range": [min(train_counts[x] for x in labels), max(train_counts[x] for x in labels)],
        "eligible_dev_count_range": [min(dev_counts[x] for x in labels), max(dev_counts[x] for x in labels)],
    }


def _highest_energy_window(crop: torch.Tensor, samples: int) -> tuple[torch.Tensor, int]:
    if samples >= crop.numel():
        return crop, 0
    squared = crop.square()
    cumulative = torch.cat((squared.new_zeros(1), squared.cumsum(dim=0)))
    energy = cumulative[samples:] - cumulative[:-samples]
    start = int(energy.argmax().item())
    return crop[start : start + samples], start


def load_natural_crop(
    source: CleanSource,
    *,
    max_event_frames: int,
    gain_db: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load an active event without time stretching and pad only its tail."""

    waveform, sample_rate = torchaudio.load(source.audio_path)
    mono = waveform.float().mean(dim=0)
    if int(sample_rate) != SAMPLE_RATE:
        mono = AF.resample(mono, int(sample_rate), SAMPLE_RATE)
    active_start = max(0, int(round(source.active_onset_seconds * SAMPLE_RATE)))
    active_end = min(mono.numel(), int(round(source.active_offset_seconds * SAMPLE_RATE)))
    if active_end <= active_start:
        raise ValueError(f"invalid active crop: {source.source_id}")
    crop = mono[active_start:active_end]
    raw_samples = crop.numel()
    maximum = max_event_frames * HOP_SAMPLES
    crop, window_start = _highest_energy_window(crop, maximum)
    frames = max(1, min(max_event_frames, int(math.ceil(crop.numel() / HOP_SAMPLES))))
    target_samples = frames * HOP_SAMPLES
    padded_samples = target_samples - crop.numel()
    if padded_samples:
        crop = F.pad(crop, (0, padded_samples))
    active = crop[: target_samples - padded_samples if padded_samples else target_samples]
    rms = active.square().mean().sqrt().clamp_min(1e-5)
    crop = crop / rms * (0.06 * (10.0 ** (gain_db / 20.0)))
    return crop, {
        "source_active_samples": raw_samples,
        "crop_window_start_samples": window_start,
        "crop_was_capped": raw_samples > maximum,
        "tail_zero_padding_samples": padded_samples,
        "time_stretched": False,
        "frames": frames,
    }


def _identities(sources: Sequence[CleanSource]) -> set[tuple[str, str]]:
    return {identity for source in sources for identity in source.hard_identities}


def _labels_for(labels: Sequence[str], anchor: str, count: int, rng: random.Random) -> list[str]:
    return v1._labels_for(labels, anchor, count, rng)


def natural_cluster_starts(
    lengths: Sequence[int], *, kind: str, overlap: float, rng: random.Random
) -> list[int]:
    """Lay out natural-duration events without assuming a minimum length.

    The v1 scheduler could assume every source had first been stretched to at
    least eight frames. V3 deliberately removes that distortion, so a true
    transient may occupy only one or two frames. Such an event cannot have a
    unique onset *and* share a frame with two later onsets. We preserve the
    requested concurrency by tying the minimum number of onsets in that edge
    case; downstream relational QA must exclude tied-onset pairs.
    """

    if not lengths or any(int(length) < 1 for length in lengths):
        raise ValueError("natural event lengths must be positive")
    if kind not in {"pair_overlap", "triple_overlap"}:
        raise ValueError(f"unknown overlap kind: {kind}")
    cluster_size = 2 if kind == "pair_overlap" else 3
    starts: list[int] = []
    cursor = rng.randint(5, 15)
    for cluster_begin in range(0, len(lengths), cluster_size):
        cluster_lengths = [
            int(value) for value in lengths[cluster_begin : cluster_begin + cluster_size]
        ]
        minimum = min(cluster_lengths)
        if len(cluster_lengths) == 1:
            local = [cursor]
        elif cluster_size == 2 or len(cluster_lengths) == 2:
            if minimum == 1:
                local = [cursor, cursor]
            else:
                step = max(1, int(round((1.0 - overlap) * minimum)))
                step = min(step, minimum - 1)
                local = [cursor, cursor + step]
        elif minimum == 1:
            local = [cursor, cursor, cursor]
        elif minimum == 2:
            local = [cursor, cursor, cursor + 1]
        else:
            step = max(1, int(round((1.0 - overlap) * minimum / 2.0)))
            step = min(step, (minimum - 1) // 2)
            local = [cursor, cursor + step, cursor + 2 * step]
        starts.extend(local)
        cluster_end = max(
            start + length for start, length in zip(local, cluster_lengths, strict=True)
        )
        cursor = cluster_end + rng.randint(5, 15)
    final_frame = max(
        start + length for start, length in zip(starts, lengths, strict=True)
    )
    if final_frame > NUM_FRAMES:
        shift = min(starts) - 2
        starts = [start - shift for start in starts]
        final_frame = max(
            start + length for start, length in zip(starts, lengths, strict=True)
        )
    if min(starts) < 0 or final_frame > NUM_FRAMES:
        raise RuntimeError("natural overlap layout exceeds the fixed scene")
    return starts


def _save_component(path: Path, waveform: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy(), SAMPLE_RATE, subtype="FLOAT")
    return sha256_file(path)


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
    max_event_frames: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_label: dict[str, list[CleanSource]] = defaultdict(list)
    for source in sources:
        if source.label in label_to_id and _gold(source):
            by_label[source.label].append(source)
    missing = [label for label in labels if not by_label[label]]
    if missing:
        raise ValueError(f"{split} Gold pool misses eligible labels: {missing}")

    scenes: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    for scene_index in range(scene_count):
        kind = "pair_overlap" if scene_index % 2 == 0 else "triple_overlap"
        overlap = v1.OVERLAP_TIERS[(scene_index // 2) % len(v1.OVERLAP_TIERS)]
        rng = random.Random(_stable_seed(seed, split, scene_index))
        event_count = rng.randint(3, 6)
        anchor = labels[scene_index % len(labels)]
        scene_labels = _labels_for(labels, anchor, event_count, rng)
        selected = [rng.choice(by_label[label]) for label in scene_labels]
        loaded = [
            load_natural_crop(
                source,
                max_event_frames=max_event_frames,
                gain_db=rng.choice((-9.0, -6.0, -3.0, 0.0, 3.0)),
            )
            for source in selected
        ]
        crops = [item[0] for item in loaded]
        crop_metadata = [item[1] for item in loaded]
        lengths = [int(item["frames"]) for item in crop_metadata]
        starts = natural_cluster_starts(lengths, kind=kind, overlap=overlap, rng=rng)
        mixture = torch.zeros(NUM_SAMPLES, dtype=torch.float32)
        unscaled_components: list[torch.Tensor] = []
        scene_id = f"overlap_gold_v3_{split}_{scene_index:07d}"
        events: list[dict[str, Any]] = []
        for event_index, (label, source, crop, crop_meta, start, length) in enumerate(
            zip(scene_labels, selected, crops, crop_metadata, starts, lengths, strict=True)
        ):
            begin = start * HOP_SAMPLES
            mixture[begin : begin + crop.numel()] += crop
            unscaled_components.append(crop)
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
                    "cleanliness_tier": "gold",
                    **crop_meta,
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
        for event, crop in zip(events, unscaled_components, strict=True):
            scaled = crop * global_scale
            begin = int(event["onset_frame"]) * HOP_SAMPLES
            reconstructed[begin : begin + scaled.numel()] += scaled
            component_relative = Path("components") / split / scene_id / f"e{int(event['event_id'].split('e')[-1]):02d}.wav"
            component_path = staging / component_relative
            component_sha256 = _save_component(component_path, scaled)
            event["component_path"] = str((final_root / component_relative).resolve())
            event["component_sha256"] = component_sha256
            event["global_scene_scale"] = global_scale
            component_rows.append(
                {
                    "format": COMPONENT_FORMAT,
                    "split": split,
                    "scene_id": scene_id,
                    "event_id": event["event_id"],
                    "label": event["label"],
                    "label_id": event["label_id"],
                    "component_path": event["component_path"],
                    "component_sha256": component_sha256,
                    "num_component_samples": scaled.numel(),
                    "sample_rate": SAMPLE_RATE,
                    "onset_frame": event["onset_frame"],
                    "offset_frame": event["offset_frame"],
                    "onset_sample": begin,
                    "offset_sample": begin + scaled.numel(),
                    "source_id": event["source_id"],
                    "source_sha256": event["source_sha256"],
                    "cleanliness_tier": "gold",
                    "duration_equalized": False,
                    "time_stretched": False,
                    "crop_was_capped": event["crop_was_capped"],
                    "tail_zero_padding_samples": event["tail_zero_padding_samples"],
                    "recipe_kind": kind,
                    "requested_overlap_fraction": overlap,
                    "global_scene_scale": global_scale,
                }
            )
        decoded, decoded_rate = torchaudio.load(audio_path)
        if int(decoded_rate) != SAMPLE_RATE:
            raise RuntimeError("stored mixture sample rate changed")
        error = float((decoded.mean(dim=0) - reconstructed).abs().max())
        reconstruction_errors.append(error)
        if error > 2.0 / 32768.0:
            raise RuntimeError(f"{scene_id}: component reconstruction error={error}")

        events.sort(key=lambda item: (item["onset_seconds"], item["event_id"]))
        maximum_concurrency = v1._maximum_concurrency(events)
        expected = 2 if kind == "pair_overlap" else min(3, event_count)
        if maximum_concurrency != expected:
            raise RuntimeError(f"{scene_id}: concurrency={maximum_concurrency}, expected={expected}")
        tied_onset_events = len(events) - len({int(event["onset_frame"]) for event in events})
        scenes.append(
            {
                "format": "qces_clean_evidence_scene_v1",
                "scene_id": scene_id,
                "scene_family_id": scene_id,
                "split": split,
                "source_route": "synthetic_gold_natural_duration_overlap_v3",
                "mixture_path": str((final_root / relative_audio).resolve()),
                "duration_seconds": 10.0,
                "sample_rate": SAMPLE_RATE,
                "audio_num_frames": NUM_SAMPLES,
                "audio_num_channels": 1,
                "audio_sha256": sha256_file(audio_path),
                "events": events,
                "recipe_kind": kind,
                "requested_overlap_fraction": overlap,
                "maximum_concurrency": maximum_concurrency,
                "tied_onset_events": tied_onset_events,
                "duration_policy": f"individual active duration retained up to {max_event_frames * FRAME_HOP_SECONDS:.2f}s; no cluster equalization",
                "rendered": True,
            }
        )
        if (scene_index + 1) % 250 == 0:
            print(f"rendered {split}: {scene_index + 1}/{scene_count}", flush=True)
    return scenes, component_rows, {
        "scenes": len(scenes),
        "events": len(component_rows),
        "unique_sources": len({row["source_id"] for row in component_rows}),
        "maximum_reconstruction_error": max(reconstruction_errors),
        "capped_events": sum(bool(row["crop_was_capped"]) for row in component_rows),
        "tail_padded_events": sum(int(row["tail_zero_padding_samples"]) > 0 for row in component_rows),
        "scenes_with_tied_onsets": sum(int(row["tied_onset_events"]) > 0 for row in scenes),
        "tied_onset_events": sum(int(row["tied_onset_events"]) for row in scenes),
        "duration_frames": {
            "minimum": min(int(row["offset_frame"]) - int(row["onset_frame"]) for row in component_rows),
            "maximum": max(int(row["offset_frame"]) - int(row["onset_frame"]) for row in component_rows),
            "mean": sum(int(row["offset_frame"]) - int(row["onset_frame"]) for row in component_rows) / len(component_rows),
        },
    }


def filter_reference_manifest(
    path: Path,
    eligible: set[str],
    *,
    gold_only: bool,
) -> list[dict[str, Any]]:
    rows = []
    for row in _read_jsonl(path):
        events = list(row.get("events") or [])
        if not events:
            continue
        if any(str(event.get("label")) not in eligible for event in events):
            continue
        if gold_only and any(str(event.get("cleanliness_tier") or "").lower() != "gold" for event in events):
            continue
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.min_gold_train_sources < 1 or args.min_gold_dev_sources < 1:
        raise SystemExit("Gold source minima must be positive")
    if not 1 <= args.max_event_frames <= 60:
        raise SystemExit("--max-event-frames must be in [1,60]")
    source_labels = [line.strip() for line in args.source_ontology.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(source_labels) != 191 or len(source_labels) != len(set(source_labels)):
        raise ValueError("source ontology must contain the frozen 191 unique labels")
    sources = load_source_bank(args.source_bank.resolve(), require_audio_file=True)
    partitioned, partition_receipt = partition_sources(sources, source_labels, seed=2041, dev_fraction=0.20)
    labels, coverage = select_eligible_labels(
        source_labels,
        partitioned["train"],
        partitioned["dev"],
        min_train=args.min_gold_train_sources,
        min_dev=args.min_gold_dev_sources,
    )
    if len(labels) < 180:
        raise RuntimeError(f"coverage gate retained only {len(labels)} labels")
    eligible = set(labels)
    train_sources = [source for source in partitioned["train"] if source.label in eligible and _gold(source)]
    dev_sources = [source for source in partitioned["dev"] if source.label in eligible and _gold(source)]
    test_sources = [source for source in partitioned["test"] if source.label in eligible and _gold(source)]
    identity_overlap = {
        "train_dev": len(_identities(train_sources) & _identities(dev_sources)),
        "train_test": len(_identities(train_sources) & _identities(test_sources)),
        "dev_test": len(_identities(dev_sources) & _identities(test_sources)),
    }
    if any(identity_overlap.values()):
        raise RuntimeError(f"identity leakage: {identity_overlap}")

    train_count = _scene_count(args.reference_train_scenes.resolve(), args.max_train_scenes)
    dev_count = _scene_count(args.reference_dev_scenes.resolve(), args.max_dev_scenes)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        label_to_id = {label: index for index, label in enumerate(labels)}
        split_outputs: dict[str, Any] = {}
        for split, count, pool in (("train", train_count, train_sources), ("dev", dev_count, dev_sources)):
            scenes, components, stats = render_split(
                split=split,
                scene_count=count,
                sources=pool,
                labels=labels,
                label_to_id=label_to_id,
                staging=staging,
                final_root=output_dir,
                seed=args.seed,
                max_event_frames=args.max_event_frames,
            )
            scene_path = staging / f"detector_scene_manifest_overlap_{split}.jsonl"
            component_path = staging / f"event_components_{split}.jsonl"
            ids_path = staging / f"scene_ids_overlap_{split}.txt"
            _atomic_write_text(scene_path, _jsonl(scenes))
            _atomic_write_text(component_path, _jsonl(components))
            _atomic_write_text(ids_path, "".join(f"{scene['scene_id']}\n" for scene in scenes))
            split_outputs[split] = {"stats": stats, "scenes": scenes, "components": components}

        single_train = filter_reference_manifest(
            args.reference_single_train.resolve(), eligible, gold_only=True
        )
        locked_test = filter_reference_manifest(
            args.reference_test.resolve(), eligible, gold_only=False
        )
        _atomic_write_text(staging / "detector_scene_manifest_single_train_gold.jsonl", _jsonl(single_train))
        _atomic_write_text(staging / "detector_scene_manifest_locked_test_filtered.jsonl", _jsonl(locked_test))
        _atomic_write_text(staging / f"ontology_{len(labels)}.txt", "".join(f"{label}\n" for label in labels))

        artifact_names = [
            f"detector_scene_manifest_overlap_{split}.jsonl" for split in ("train", "dev")
        ] + [
            f"event_components_{split}.jsonl" for split in ("train", "dev")
        ] + [
            f"scene_ids_overlap_{split}.txt" for split in ("train", "dev")
        ] + [
            "detector_scene_manifest_single_train_gold.jsonl",
            "detector_scene_manifest_locked_test_filtered.jsonl",
            f"ontology_{len(labels)}.txt",
        ]
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "ontology": {"classes": len(labels), "labels": labels, "coverage_gate": coverage},
            "policy": {
                "model_predictions_used_for_selection": False,
                "answer_labels_used": False,
                "source_quality": "Gold only",
                "official_eval_sources_used_for_train_or_dev": False,
                "cluster_duration_equalization": False,
                "time_stretching": False,
                "long_active_crop": f"deterministic highest-energy window capped at {args.max_event_frames * FRAME_HOP_SECONDS:.2f}s",
                "grid_alignment": "zero-pad tail to 40 ms boundary",
            },
            "partition_policy": partition_receipt["policy"],
            "hard_identity_overlap": identity_overlap,
            "data": {
                "single_train_gold": len(single_train),
                "locked_test_filtered": len(locked_test),
                "train": split_outputs["train"]["stats"],
                "dev": split_outputs["dev"]["stats"],
            },
            "artifacts": {name: sha256_file(staging / name) for name in artifact_names},
        }
        _atomic_write_text(
            staging / "build_receipt.json",
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
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
