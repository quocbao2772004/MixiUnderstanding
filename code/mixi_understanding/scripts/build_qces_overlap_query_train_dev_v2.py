#!/usr/bin/env python3
"""Build duration-balanced polyphonic train/dev scenes for query tuning.

V1 preserved each source crop duration.  Consequently most train/dev events
were 1.2 s long, while a real polyphonic cluster can be governed by its
shortest transient.  V2 keeps the same source-disjoint partitions and overlap
tiers, but equalizes every pair/triple cluster to the shortest member without
time-stretching audio.  The retained window is the highest-energy contiguous
window, which avoids turning a valid transient crop into silence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torchaudio

from mixi_understanding.qces.clean_evidence_scenes import (
    CleanSource,
    _atomic_write_text,
    load_source_bank,
    partition_sources,
    sha256_file,
)
from mixi_understanding.scripts import build_qces_overlap_query_train_dev_v1 as v1


FORMAT = "qces_overlap_query_train_dev_v2"


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=data / "source_bank_accepted.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_191.txt")
    parser.add_argument("--nonoverlap-train-scenes", type=Path, default=data / "multievent/scene_ids_train.txt")
    parser.add_argument("--nonoverlap-dev-scenes", type=Path, default=data / "multievent/scene_ids_dev.txt")
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2",
    )
    parser.add_argument("--seed", type=int, default=v1.DEFAULT_SEED)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _highest_energy_window(crop: torch.Tensor, samples: int) -> torch.Tensor:
    if samples <= 0 or samples > crop.numel():
        raise ValueError("invalid equalization window")
    if samples == crop.numel():
        return crop
    # Sliding energy through a cumulative sum is O(N), including for a 1.2-s
    # window; a direct convolution would be unnecessarily expensive here.
    squared = crop.square()
    cumulative = torch.cat((squared.new_zeros(1), squared.cumsum(dim=0)))
    energy = cumulative[samples:] - cumulative[:-samples]
    start = int(energy.argmax())
    return crop[start: start + samples]


def _equalize_clusters(
    crops: Sequence[torch.Tensor], *, kind: str
) -> tuple[list[torch.Tensor], list[int], list[int]]:
    hop_samples = int(round(v1.FRAME_HOP_SECONDS * v1.SAMPLE_RATE))
    raw_lengths = [crop.numel() // hop_samples for crop in crops]
    cluster_size = 2 if kind == "pair_overlap" else 3
    adjusted = list(crops)
    final_lengths = list(raw_lengths)
    for begin in range(0, len(crops), cluster_size):
        end = min(begin + cluster_size, len(crops))
        if end - begin < 2:
            continue
        common = min(raw_lengths[begin:end])
        target_samples = common * hop_samples
        for index in range(begin, end):
            adjusted[index] = _highest_energy_window(crops[index], target_samples)
            final_lengths[index] = common
    return adjusted, raw_lengths, final_lengths


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
    by_label: dict[str, list[CleanSource]] = {}
    for label in labels:
        by_label[label] = [source for source in sources if source.label == label]
    missing = [label for label, rows in by_label.items() if not rows]
    if missing:
        raise ValueError(f"{split} source pool misses labels: {missing[:10]}")
    scenes: list[dict[str, Any]] = []
    audio_dir = staging / "audio" / split
    audio_dir.mkdir(parents=True, exist_ok=True)
    hop_samples = int(round(v1.FRAME_HOP_SECONDS * v1.SAMPLE_RATE))
    for scene_index in range(scene_count):
        kind = "pair_overlap" if scene_index % 2 == 0 else "triple_overlap"
        overlap = v1.OVERLAP_TIERS[(scene_index // 2) % len(v1.OVERLAP_TIERS)]
        rng = __import__("random").Random(v1._stable_seed(seed, "duration_balanced_v2", split, scene_index))
        count = rng.randint(3, 6)
        anchor = labels[scene_index % len(labels)]
        scene_labels = v1._labels_for(labels, anchor, count, rng)
        selected = [rng.choice(by_label[label]) for label in scene_labels]
        raw_crops = [v1._load_crop(source, rng) for source in selected]
        crops, raw_lengths, lengths = _equalize_clusters(raw_crops, kind=kind)
        starts = v1._cluster_starts(lengths, kind=kind, overlap=overlap, rng=rng)
        mixture = torch.zeros(v1.NUM_SAMPLES, dtype=torch.float32)
        events: list[dict[str, Any]] = []
        scene_id = f"overlap_query_v2_{split}_{scene_index:07d}"
        for event_index, (label, source, crop, start, raw_length, length) in enumerate(
            zip(scene_labels, selected, crops, starts, raw_lengths, lengths, strict=True)
        ):
            begin = start * hop_samples
            mixture[begin: begin + crop.numel()] += crop
            events.append(
                {
                    "event_id": f"{scene_id}:e{event_index:02d}",
                    "event_kind": "semantic",
                    "label": label,
                    "label_id": int(label_to_id[label]),
                    "onset_seconds": start * v1.FRAME_HOP_SECONDS,
                    "offset_seconds": (start + length) * v1.FRAME_HOP_SECONDS,
                    "onset_frame": start,
                    "offset_frame": start + length,
                    "raw_crop_frames": raw_length,
                    "duration_equalized": length != raw_length,
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
            str(staging / relative_audio), mixture.unsqueeze(0), v1.SAMPLE_RATE,
            encoding="PCM_S", bits_per_sample=16,
        )
        events.sort(key=lambda event: (event["onset_seconds"], event["event_id"]))
        maximum_concurrency = v1._maximum_concurrency(events)
        expected = 2 if kind == "pair_overlap" else min(3, count)
        if maximum_concurrency != expected:
            raise RuntimeError(f"{scene_id}: concurrency={maximum_concurrency}, expected={expected}")
        scenes.append(
            {
                "format": "qces_clean_evidence_scene_v1",
                "scene_id": scene_id,
                "scene_family_id": scene_id,
                "split": split,
                "source_route": "synthetic_clean_single_event_bank_overlap_query_duration_balanced_v2",
                "mixture_path": str((final_root / relative_audio).resolve()),
                "duration_seconds": v1.SCENE_SECONDS,
                "sample_rate": v1.SAMPLE_RATE,
                "audio_num_frames": v1.NUM_SAMPLES,
                "audio_num_channels": 1,
                "audio_sha256": sha256_file(staging / relative_audio),
                "events": events,
                "recipe_kind": kind,
                "requested_overlap_fraction": overlap,
                "maximum_concurrency": maximum_concurrency,
                "duration_policy": "equalize_each_polyphonic_cluster_to_shortest_member",
                "rendered": True,
            }
        )
        if (scene_index + 1) % 250 == 0:
            print(f"rendered {split}: {scene_index + 1}/{scene_count}", flush=True)
    return scenes


def _identities(sources: Sequence[CleanSource]) -> set[tuple[str, str]]:
    return {identity for source in sources for identity in source.hard_identities}


def _duration_stats(scenes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    durations = sorted(
        float(event["offset_seconds"]) - float(event["onset_seconds"])
        for scene in scenes for event in scene["events"]
    )
    def quantile(fraction: float) -> float:
        index = round(fraction * (len(durations) - 1))
        return durations[index]
    return {
        "count": len(durations),
        "mean_seconds": sum(durations) / max(len(durations), 1),
        "q10_seconds": quantile(0.10),
        "q25_seconds": quantile(0.25),
        "median_seconds": quantile(0.50),
        "q75_seconds": quantile(0.75),
        "q90_seconds": quantile(0.90),
        "minimum_seconds": durations[0],
        "maximum_seconds": durations[-1],
    }


def main() -> None:
    args = parse_args()
    labels = [line.strip() for line in args.ontology.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 191 or len(labels) != len(set(labels)):
        raise ValueError(f"requires frozen 191-label ontology, got {len(labels)}")
    train_count = v1._load_scene_count(args.nonoverlap_train_scenes)
    dev_count = v1._load_scene_count(args.nonoverlap_dev_scenes)
    if args.max_train_scenes > 0:
        train_count = min(train_count, args.max_train_scenes)
    if args.max_dev_scenes > 0:
        dev_count = min(dev_count, args.max_dev_scenes)
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
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        label_to_id = {label: index for index, label in enumerate(labels)}
        scenes = {
            "train": _render_split(
                split="train", scene_count=train_count, sources=train_sources, labels=labels,
                label_to_id=label_to_id, staging=staging, final_root=output_dir, seed=args.seed,
            ),
            "dev": _render_split(
                split="dev", scene_count=dev_count, sources=dev_sources, labels=labels,
                label_to_id=label_to_id, staging=staging, final_root=output_dir, seed=args.seed,
            ),
        }
        artifacts: dict[str, str] = {}
        split_stats: dict[str, Any] = {}
        for split in ("train", "dev"):
            manifest = staging / f"detector_scene_manifest_overlap_{split}.jsonl"
            ids = staging / f"scene_ids_overlap_{split}.txt"
            _atomic_write_text(manifest, _jsonl(scenes[split]))
            _atomic_write_text(ids, "".join(f"{scene['scene_id']}\n" for scene in scenes[split]))
            artifacts[manifest.name] = sha256_file(manifest)
            artifacts[ids.name] = sha256_file(ids)
            label_counts = Counter(event["label"] for scene in scenes[split] for event in scene["events"])
            split_stats[split] = {
                "scenes": len(scenes[split]),
                "events": sum(len(scene["events"]) for scene in scenes[split]),
                "classes_present": len(label_counts),
                "minimum_events_per_class": min(label_counts.values()),
                "maximum_events_per_class": max(label_counts.values()),
                "equalized_events": sum(
                    bool(event["duration_equalized"]) for scene in scenes[split] for event in scene["events"]
                ),
                "duration": _duration_stats(scenes[split]),
                "recipe_distribution": dict(sorted(Counter(scene["recipe_kind"] for scene in scenes[split]).items())),
            }
        ontology = staging / "ontology_191.txt"
        _atomic_write_text(ontology, "".join(f"{label}\n" for label in labels))
        artifacts[ontology.name] = sha256_file(ontology)
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "policy": {
                "answer_label_used": False,
                "official_eval_sources_used": False,
                "train_dev_source_reuse": False,
                "duration_equalization": "each pair/triple cluster uses its shortest raw member",
                "audio_shortening": "highest-energy contiguous crop; no time stretching",
                "event_count_range": [3, 6],
                "overlap_tiers": list(v1.OVERLAP_TIERS),
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
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
