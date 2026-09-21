#!/usr/bin/env python3
"""Replay overlap-v2 recipes and materialize exact per-event component stems.

The existing overlap-v2 dataset stores mixtures and event metadata, but not
the isolated waveforms used to render each event.  A span-conditioned
separator needs those waveforms as supervision.  This sidecar builder replays
the frozen deterministic recipe, verifies it against the already-rendered
mixture, and writes only short event components (it never copies or changes
the original mixtures).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch
import torchaudio
import soundfile as sf

from mixi_understanding.qces.clean_evidence_scenes import (
    CleanSource,
    _atomic_write_text,
    load_source_bank,
    partition_sources,
    sha256_file,
)
from mixi_understanding.scripts import build_qces_overlap_query_train_dev_v1 as v1
from mixi_understanding.scripts import build_qces_overlap_query_train_dev_v2 as v2


FORMAT = "qces_overlap_event_components_v1"


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
    overlap = PROJECT_ROOT / "outputs/qces_full191_overlap_query_train_dev_v2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=data / "source_bank_accepted.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_191.txt")
    parser.add_argument("--overlap-root", type=Path, default=overlap)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_full191_overlap_components_v1",
    )
    parser.add_argument("--seed", type=int, default=v1.DEFAULT_SEED)
    parser.add_argument("--max-train-scenes", type=int, default=0)
    parser.add_argument("--max-dev-scenes", type=int, default=0)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _event_map(scene: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(event["event_id"]): event for event in scene["events"]}


def _assert_replay_metadata(
    scene: Mapping[str, Any], labels: Sequence[str], selected: Sequence[CleanSource],
    starts: Sequence[int], lengths: Sequence[int],
) -> None:
    expected = _event_map(scene)
    scene_id = str(scene["scene_id"])
    if len(expected) != len(labels):
        raise RuntimeError(f"{scene_id}: event count differs during replay")
    for index, (label, source, start, length) in enumerate(
        zip(labels, selected, starts, lengths, strict=True)
    ):
        event_id = f"{scene_id}:e{index:02d}"
        event = expected.get(event_id)
        if event is None:
            raise RuntimeError(f"{scene_id}: missing event id {event_id}")
        observed = (
            str(event["label"]), str(event["source_id"]), int(event["onset_frame"]),
            int(event["offset_frame"]),
        )
        replayed = (label, source.source_id, int(start), int(start + length))
        if observed != replayed:
            raise RuntimeError(
                f"{event_id}: deterministic recipe mismatch: stored={observed}, replayed={replayed}"
            )


def _render_split(
    *, split: str, scenes: Sequence[Mapping[str, Any]], sources: Sequence[CleanSource],
    labels: Sequence[str], seed: int, staging: Path, final_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    by_label = {label: [source for source in sources if source.label == label] for label in labels}
    missing = [label for label, rows in by_label.items() if not rows]
    if missing:
        raise ValueError(f"{split} source pool misses labels: {missing[:10]}")

    hop_samples = int(round(v1.FRAME_HOP_SECONDS * v1.SAMPLE_RATE))
    rows: list[dict[str, Any]] = []
    replay_max_errors: list[float] = []
    component_sum_max_errors: list[float] = []
    total_pcm_samples = 0

    for scene_index, scene in enumerate(scenes):
        expected_scene_id = f"overlap_query_v2_{split}_{scene_index:07d}"
        if str(scene["scene_id"]) != expected_scene_id:
            raise RuntimeError(
                f"scene order mismatch at {split}/{scene_index}: "
                f"{scene['scene_id']} != {expected_scene_id}"
            )
        kind = "pair_overlap" if scene_index % 2 == 0 else "triple_overlap"
        overlap = v1.OVERLAP_TIERS[(scene_index // 2) % len(v1.OVERLAP_TIERS)]
        rng = random.Random(v1._stable_seed(seed, "duration_balanced_v2", split, scene_index))
        count = rng.randint(3, 6)
        anchor = labels[scene_index % len(labels)]
        scene_labels = v1._labels_for(labels, anchor, count, rng)
        selected = [rng.choice(by_label[label]) for label in scene_labels]
        raw_crops = [v1._load_crop(source, rng) for source in selected]
        crops, raw_lengths, lengths = v2._equalize_clusters(raw_crops, kind=kind)
        starts = v1._cluster_starts(lengths, kind=kind, overlap=overlap, rng=rng)
        _assert_replay_metadata(scene, scene_labels, selected, starts, lengths)

        mixture = torch.zeros(v1.NUM_SAMPLES, dtype=torch.float32)
        for crop, start in zip(crops, starts, strict=True):
            begin = int(start) * hop_samples
            mixture[begin: begin + crop.numel()] += crop
        peak_before_scaling = float(mixture.abs().max())
        global_scale = min(1.0, 0.95 / peak_before_scaling) if peak_before_scaling > 0 else 1.0
        mixture *= global_scale

        stored_mixture, stored_sr = torchaudio.load(str(scene["mixture_path"]))
        stored_mixture = stored_mixture.float().mean(dim=0)
        if int(stored_sr) != v1.SAMPLE_RATE or stored_mixture.numel() != v1.NUM_SAMPLES:
            raise RuntimeError(f"{scene['scene_id']}: stored mixture shape/sample-rate mismatch")
        replay_error = float((stored_mixture - mixture).abs().max())
        replay_max_errors.append(replay_error)
        if replay_error > 3.2e-5:
            raise RuntimeError(
                f"{scene['scene_id']}: replay max error {replay_error:.8f} exceeds one PCM16 step"
            )

        component_reconstruction = torch.zeros_like(stored_mixture)
        for event_index, (label, source, crop, start, raw_length, length) in enumerate(
            zip(scene_labels, selected, crops, starts, raw_lengths, lengths, strict=True)
        ):
            scaled_crop = crop * global_scale
            # A component can exceed [-1, 1] even when destructive interference
            # keeps the final mixture below the scene peak limit.  Integer PCM
            # would clip that component and break additivity, so stems are kept
            # as lossless IEEE float32 WAV.
            relative = Path("components") / split / str(scene["scene_id"]) / f"e{event_index:02d}.wav"
            component_path = staging / relative
            component_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(component_path, scaled_crop.detach().cpu().numpy(), v1.SAMPLE_RATE, subtype="FLOAT")
            decoded, decoded_sr = torchaudio.load(component_path)
            if int(decoded_sr) != v1.SAMPLE_RATE:
                raise RuntimeError(f"{component_path}: unexpected sample rate {decoded_sr}")
            decoded = decoded.float().mean(dim=0)
            begin = int(start) * hop_samples
            component_reconstruction[begin: begin + decoded.numel()] += decoded
            total_pcm_samples += int(decoded.numel())
            rows.append(
                {
                    "format": FORMAT,
                    "scene_id": str(scene["scene_id"]),
                    "split": split,
                    "recipe_kind": kind,
                    "requested_overlap_fraction": float(overlap),
                    "mixture_path": str(Path(str(scene["mixture_path"])).resolve()),
                    "component_path": str((final_root / relative).resolve()),
                    "event_id": f"{scene['scene_id']}:e{event_index:02d}",
                    "label": label,
                    "label_id": int(labels.index(label)),
                    "source_id": source.source_id,
                    "source_sha256": source.source_sha256,
                    "onset_frame": int(start),
                    "offset_frame": int(start + length),
                    "onset_sample": int(begin),
                    "offset_sample": int(begin + scaled_crop.numel()),
                    "raw_crop_frames": int(raw_length),
                    "duration_equalized": bool(length != raw_length),
                    "global_scene_scale": float(global_scale),
                    "sample_rate": v1.SAMPLE_RATE,
                    "num_component_samples": int(scaled_crop.numel()),
                    "component_sha256": sha256_file(component_path),
                }
            )
        component_error = float((component_reconstruction - stored_mixture).abs().max())
        component_sum_max_errors.append(component_error)
        # The mixture and each component are quantized independently.  The
        # worst case is bounded by concurrent PCM16 rounding errors.
        if component_error > 1.6e-4:
            raise RuntimeError(
                f"{scene['scene_id']}: decoded component sum error {component_error:.8f} too large"
            )
        if (scene_index + 1) % 250 == 0:
            print(f"materialized {split}: {scene_index + 1}/{len(scenes)}", flush=True)

    return rows, {
        "scenes": len(scenes),
        "components": len(rows),
        "total_component_pcm_samples": total_pcm_samples,
        "replay_max_abs_error": max(replay_max_errors, default=0.0),
        "decoded_component_sum_max_abs_error": max(component_sum_max_errors, default=0.0),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing sidecar: {output_dir}")

    labels = [
        line.strip() for line in args.ontology.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) != 191 or len(labels) != len(set(labels)):
        raise ValueError(f"requires frozen 191-label ontology, got {len(labels)}")

    overlap_root = args.overlap_root.resolve()
    all_scenes = {
        split: _read_jsonl(overlap_root / f"detector_scene_manifest_overlap_{split}.jsonl")
        for split in ("train", "dev")
    }
    if args.max_train_scenes > 0:
        all_scenes["train"] = all_scenes["train"][: args.max_train_scenes]
    if args.max_dev_scenes > 0:
        all_scenes["dev"] = all_scenes["dev"][: args.max_dev_scenes]

    sources = load_source_bank(args.source_bank.resolve(), require_audio_file=True)
    partitioned, partition_receipt = partition_sources(sources, labels, seed=2041, dev_fraction=0.20)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        manifests: dict[str, list[dict[str, Any]]] = {}
        split_stats: dict[str, Mapping[str, float | int]] = {}
        artifacts: dict[str, str] = {}
        for split in ("train", "dev"):
            manifests[split], split_stats[split] = _render_split(
                split=split,
                scenes=all_scenes[split],
                sources=list(partitioned[split]),
                labels=labels,
                seed=args.seed,
                staging=staging,
                final_root=output_dir,
            )
            manifest_path = staging / f"event_components_{split}.jsonl"
            _atomic_write_text(manifest_path, _jsonl(manifests[split]))
            artifacts[manifest_path.name] = sha256_file(manifest_path)

        source_receipt = overlap_root / "build_receipt.json"
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "source_overlap_build_receipt_sha256": sha256_file(source_receipt),
            "source_bank_sha256": sha256_file(args.source_bank.resolve()),
            "ontology_sha256": sha256_file(args.ontology.resolve()),
            "partition_policy": partition_receipt["policy"],
            "policy": {
                "changes_existing_dataset": False,
                "copies_mixture_audio": False,
                "answer_label_used": False,
                "component_storage": "active-duration WAV IEEE float32 after global scene scaling",
                "verification": "deterministic replay against stored PCM16 mixture and decoded component sum",
            },
            "split_stats": split_stats,
            "artifacts": artifacts,
        }
        receipt_path = staging / "receipt.json"
        _atomic_write_text(
            receipt_path,
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        os.replace(staging, output_dir)
        staging = None
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True), flush=True)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
