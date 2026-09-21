#!/usr/bin/env python3
"""Build Gold-only, semantically sufficient dense-overlap scenes.

V3 capped every active component at 1.6 seconds so separate overlap clusters
could be concatenated in a ten-second clip.  A frozen paired audit showed that
this cap loses 8.1 absolute top-1 points relative to full active sources.  The
pre-registered duration curve selected 4.5 seconds as the smallest condition
within two points of full-active top-1 while exceeding 95% top-5.

V4 therefore retains up to 112 frames (4.48 s) and places pair/triple overlap
clusters on one dense timeline.  Cross-cluster overlap is allowed and actual
maximum concurrency is stored rather than forced to two or three.  All source,
quality, identity, waveform, and reconstruction contracts from V3 remain.
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


FORMAT = "qces_overlap_semantic_sufficient_v4"
COMPONENT_FORMAT = "qces_overlap_semantic_sufficient_component_v4"
DEFAULT_SEED = 2404
SAMPLE_RATE = v3.SAMPLE_RATE
FRAME_HOP_SECONDS = v3.FRAME_HOP_SECONDS
HOP_SAMPLES = v3.HOP_SAMPLES
NUM_FRAMES = v3.NUM_FRAMES
NUM_SAMPLES = v3.NUM_SAMPLES
MAX_EVENT_FRAMES = 112


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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_full188_overlap_semantic_sufficient_v4")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-gold-train-sources", type=int, default=v3.MIN_GOLD_TRAIN_SOURCES)
    parser.add_argument("--min-gold-dev-sources", type=int, default=v3.MIN_GOLD_DEV_SOURCES)
    parser.add_argument("--max-event-frames", type=int, default=MAX_EVENT_FRAMES)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _identities(sources: Sequence[CleanSource]) -> set[tuple[str, str]]:
    return {identity for source in sources for identity in source.hard_identities}


def _deduplicate_starts(starts: list[int], lengths: Sequence[int]) -> list[int]:
    """Make onsets unique with minimal shifts while keeping every event in bounds."""
    result: list[int] = []
    used: set[int] = set()
    for start, length in zip(starts, lengths, strict=True):
        maximum = NUM_FRAMES - int(length) - 2
        candidate = min(int(start), maximum)
        while candidate in used and candidate < maximum:
            candidate += 1
        while candidate in used and candidate > 2:
            candidate -= 1
        if candidate in used:
            # Only one-frame transients can make a unique overlapping onset
            # impossible; retain the tie and record it in the manifest.
            candidate = int(start)
        used.add(candidate)
        result.append(candidate)
    return result


def dense_cluster_starts(
    lengths: Sequence[int], *, kind: str, overlap: float, rng: random.Random
) -> list[int]:
    """Place overlap clusters on a shared timeline instead of concatenating them."""
    if not lengths or any(int(value) < 1 or int(value) > MAX_EVENT_FRAMES for value in lengths):
        raise ValueError("invalid V4 event length")
    if kind not in {"pair_overlap", "triple_overlap"}:
        raise ValueError(f"unknown overlap kind: {kind}")
    cluster_size = 2 if kind == "pair_overlap" else 3
    starts: list[int] = []
    # A 7--11 frame shift makes cluster anchors distinct while allowing long
    # components to overlap across clusters.  The seed fixes the exact shift.
    cluster_shift = rng.randint(7, 11)
    initial = rng.randint(4, 8)
    for cluster_index, begin in enumerate(range(0, len(lengths), cluster_size)):
        current = [int(value) for value in lengths[begin : begin + cluster_size]]
        base = initial + cluster_index * cluster_shift
        minimum = min(current)
        if len(current) == 1:
            local = [0]
        elif len(current) == 2:
            step = 0 if minimum == 1 else max(1, int(round((1.0 - overlap) * minimum)))
            step = min(step, max(0, minimum - 1))
            local = [0, step]
        elif minimum == 1:
            local = [0, 0, 0]
        elif minimum == 2:
            local = [0, 0, 1]
        else:
            step = max(1, int(round((1.0 - overlap) * minimum / 2.0)))
            step = min(step, (minimum - 1) // 2)
            local = [0, step, 2 * step]
        starts.extend(base + offset for offset in local)
    starts = _deduplicate_starts(starts, lengths)
    if min(starts) < 0 or max(start + int(length) for start, length in zip(starts, lengths, strict=True)) > NUM_FRAMES:
        raise RuntimeError("V4 dense layout exceeds ten seconds")
    return starts


def _save_component(path: Path, waveform: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform.detach().cpu().numpy(), SAMPLE_RATE, subtype="FLOAT")
    return sha256_file(path)


def _within_cluster_overlap(events: Sequence[Mapping[str, Any]], kind: str) -> list[float]:
    cluster_size = 2 if kind == "pair_overlap" else 3
    values: list[float] = []
    for begin in range(0, len(events), cluster_size):
        current = list(events[begin : begin + cluster_size])
        if len(current) < 2:
            continue
        intersection = max(
            0,
            min(int(event["offset_frame"]) for event in current)
            - max(int(event["onset_frame"]) for event in current),
        )
        denominator = min(
            int(event["offset_frame"]) - int(event["onset_frame"])
            for event in current
        )
        values.append(intersection / max(denominator, 1))
    return values


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
        if source.label in label_to_id and v3._gold(source):
            by_label[source.label].append(source)
    if missing := [label for label in labels if not by_label[label]]:
        raise ValueError(f"{split} Gold pool misses eligible labels: {missing}")

    scenes: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    overlap_values: list[float] = []
    for scene_index in range(scene_count):
        kind = "pair_overlap" if scene_index % 2 == 0 else "triple_overlap"
        requested_overlap = v1.OVERLAP_TIERS[(scene_index // 2) % len(v1.OVERLAP_TIERS)]
        rng = random.Random(v3._stable_seed(seed, split, scene_index, FORMAT))
        event_count = rng.randint(3, 6)
        anchor = labels[scene_index % len(labels)]
        scene_labels = v3._labels_for(labels, anchor, event_count, rng)
        selected = [rng.choice(by_label[label]) for label in scene_labels]
        loaded = [
            v3.load_natural_crop(
                source,
                max_event_frames=max_event_frames,
                gain_db=rng.choice((-9.0, -6.0, -3.0, 0.0, 3.0)),
            )
            for source in selected
        ]
        crops = [item[0] for item in loaded]
        crop_metadata = [item[1] for item in loaded]
        lengths = [int(item["frames"]) for item in crop_metadata]
        starts = dense_cluster_starts(lengths, kind=kind, overlap=requested_overlap, rng=rng)
        scene_id = f"overlap_semantic_v4_{split}_{scene_index:07d}"
        mixture = torch.zeros(NUM_SAMPLES, dtype=torch.float32)
        events: list[dict[str, Any]] = []
        for event_index, (label, source, crop, metadata, start, length) in enumerate(
            zip(scene_labels, selected, crops, crop_metadata, starts, lengths, strict=True)
        ):
            begin = start * HOP_SAMPLES
            mixture[begin : begin + crop.numel()] += crop
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
                    **metadata,
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
            path = staging / relative
            component_sha = _save_component(path, scaled)
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
                    "duration_equalized": False,
                    "time_stretched": False,
                    "crop_was_capped": event["crop_was_capped"],
                    "tail_zero_padding_samples": event["tail_zero_padding_samples"],
                    "recipe_kind": kind,
                    "requested_overlap_fraction": requested_overlap,
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
        actual_overlap = _within_cluster_overlap(events, kind)
        overlap_values.extend(actual_overlap)
        events.sort(key=lambda item: (item["onset_seconds"], item["event_id"]))
        concurrency = v1._maximum_concurrency(events)
        tied = len(events) - len({int(event["onset_frame"]) for event in events})
        scenes.append(
            {
                "format": "qces_clean_evidence_scene_v1",
                "scene_id": scene_id,
                "scene_family_id": scene_id,
                "split": split,
                "source_route": "synthetic_gold_semantically_sufficient_dense_overlap_v4",
                "mixture_path": str((final_root / relative_audio).resolve()),
                "duration_seconds": 10.0,
                "sample_rate": SAMPLE_RATE,
                "audio_num_frames": NUM_SAMPLES,
                "audio_num_channels": 1,
                "audio_sha256": sha256_file(audio_path),
                "events": events,
                "recipe_kind": kind,
                "requested_overlap_fraction": requested_overlap,
                "actual_within_cluster_overlap_fractions": actual_overlap,
                "maximum_concurrency": concurrency,
                "tied_onset_events": tied,
                "duration_policy": f"individual active duration retained up to {max_event_frames * FRAME_HOP_SECONDS:.2f}s; duration selected by frozen semantic audit",
                "layout_policy": "pair/triple clusters share a dense ten-second timeline; cross-cluster overlap allowed",
                "rendered": True,
            }
        )
        if (scene_index + 1) % 250 == 0:
            print(f"rendered {split}: {scene_index + 1}/{scene_count}", flush=True)
    return scenes, components, {
        "scenes": len(scenes),
        "events": len(components),
        "unique_sources": len({row["source_id"] for row in components}),
        "maximum_reconstruction_error": max(reconstruction_errors),
        "capped_events": sum(bool(row["crop_was_capped"]) for row in components),
        "tail_padded_events": sum(int(row["tail_zero_padding_samples"]) > 0 for row in components),
        "scenes_with_tied_onsets": sum(int(row["tied_onset_events"]) > 0 for row in scenes),
        "maximum_concurrency": {
            "minimum": min(int(row["maximum_concurrency"]) for row in scenes),
            "maximum": max(int(row["maximum_concurrency"]) for row in scenes),
            "mean": sum(int(row["maximum_concurrency"]) for row in scenes) / len(scenes),
        },
        "actual_within_cluster_overlap": {
            "minimum": min(overlap_values),
            "maximum": max(overlap_values),
            "mean": sum(overlap_values) / len(overlap_values),
        },
        "duration_frames": {
            "minimum": min(int(row["offset_frame"]) - int(row["onset_frame"]) for row in components),
            "maximum": max(int(row["offset_frame"]) - int(row["onset_frame"]) for row in components),
            "mean": sum(int(row["offset_frame"]) - int(row["onset_frame"]) for row in components) / len(components),
        },
    }


def main() -> None:
    args = parse_args()
    if args.max_event_frames != MAX_EVENT_FRAMES:
        raise SystemExit(f"V4 duration was pre-selected at exactly {MAX_EVENT_FRAMES} frames")
    audit = json.loads(args.duration_audit.resolve().read_text(encoding="utf-8"))
    if audit.get("format") != "qces_semantic_duration_curve_v1" or not audit.get("gate_passed"):
        raise ValueError("duration audit did not pass")
    if audit.get("selected_condition") != "cap_4.5s":
        raise ValueError(f"unexpected selected duration: {audit.get('selected_condition')}")
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
    if len(labels) < 180:
        raise RuntimeError(f"coverage gate retained only {len(labels)} labels")
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
                max_event_frames=args.max_event_frames,
            )
            _atomic_write_text(staging / f"detector_scene_manifest_overlap_{split}.jsonl", _jsonl(scenes))
            _atomic_write_text(staging / f"event_components_{split}.jsonl", _jsonl(components))
            _atomic_write_text(staging / f"scene_ids_overlap_{split}.txt", "".join(f"{row['scene_id']}\n" for row in scenes))
            outputs[split] = {"stats": stats}
        single = v3.filter_reference_manifest(args.reference_single_train.resolve(), eligible, gold_only=True)
        locked = v3.filter_reference_manifest(args.reference_test.resolve(), eligible, gold_only=False)
        _atomic_write_text(staging / "detector_scene_manifest_single_train_gold.jsonl", _jsonl(single))
        _atomic_write_text(staging / "detector_scene_manifest_locked_test_filtered.jsonl", _jsonl(locked))
        ontology_name = f"ontology_{len(labels)}.txt"
        _atomic_write_text(staging / ontology_name, "".join(f"{label}\n" for label in labels))
        names = [f"detector_scene_manifest_overlap_{split}.jsonl" for split in ("train", "dev")] + [f"event_components_{split}.jsonl" for split in ("train", "dev")] + [f"scene_ids_overlap_{split}.txt" for split in ("train", "dev")] + ["detector_scene_manifest_single_train_gold.jsonl", "detector_scene_manifest_locked_test_filtered.jsonl", ontology_name]
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "ontology": {"classes": len(labels), "labels": labels, "coverage_gate": coverage},
            "policy": {
                "model_predictions_used_for_source_or_label_selection": False,
                "answer_labels_used": False,
                "source_quality": "Gold only",
                "official_eval_sources_used_for_train_or_dev": False,
                "time_stretching": False,
                "duration_selection_receipt": str(args.duration_audit.resolve()),
                "maximum_active_crop_seconds": args.max_event_frames * FRAME_HOP_SECONDS,
                "layout": "dense shared timeline with pair/triple within-cluster overlap and permitted cross-cluster overlap",
            },
            "duration_audit": {
                "sha256": sha256_file(args.duration_audit.resolve()),
                "selected_condition": audit["selected_condition"],
                "selected_metrics": audit["summary"][audit["selected_condition"]],
                "full_active_metrics": audit["summary"]["full_active"],
            },
            "partition_policy": partition_receipt["policy"],
            "hard_identity_overlap": identity_overlap,
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
