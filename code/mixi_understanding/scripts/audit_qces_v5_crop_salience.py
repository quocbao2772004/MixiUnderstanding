#!/usr/bin/env python3
"""Compare current random QCES crops with label-free max-variance crops.

The comparison scores isolated, full-scene event stems with a frozen BEATs
AudioSet classifier.  BEATs is only a diagnostic here: this script does not
modify data and its result cannot be presented as an independent evaluator if
the proposed crop policy is later selected using this report.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf
import torch

from mixi_understanding.qces.beats_auditor import BEATsAudioSetAuditor
from mixi_understanding.scripts.build_qa_removal_dataset import (
    SourceClip,
    crop_source_event,
)


FORMAT = "qces_v5_crop_salience_diagnostic_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    baseline = PROJECT_ROOT / "code/baseline/beats-unilm"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--beats-root", type=Path, default=baseline)
    parser.add_argument(
        "--beats-checkpoint",
        type=Path,
        default=baseline / "checkpoint/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    )
    parser.add_argument(
        "--audioset-labels",
        type=Path,
        default=baseline / "checkpoint/class_labels_indices.csv",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def safe_path(root: Path, relative: object, context: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"invalid {context}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{context} escapes root: {relative}") from error
    if not path.is_file():
        raise ValueError(f"missing {context}: {path}")
    return path


def load_base_scenes(manifest: Path) -> list[Mapping[str, Any]]:
    scenes: dict[str, Mapping[str, Any]] = {}
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"non-object at {manifest}:{line_number}")
            if record.get("variant_id") != "base":
                continue
            scene_id = record.get("scene_id")
            if not isinstance(scene_id, str):
                raise ValueError(f"missing scene_id at {manifest}:{line_number}")
            previous = scenes.setdefault(scene_id, record)
            if previous.get("render_recipe_id") != record.get("render_recipe_id"):
                raise ValueError(f"inconsistent repeated scene: {scene_id}")
    if not scenes:
        raise ValueError("manifest has no base scenes")
    return [scenes[key] for key in sorted(scenes)]


def read_mono(path: Path, expected_rate: int, expected_samples: int) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != expected_rate or len(waveform) != expected_samples:
        raise ValueError(f"audio shape/rate mismatch: {path}")
    result = waveform.mean(axis=1, dtype=np.float64).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"non-finite audio: {path}")
    return result


def summarize(values: Sequence[float], direction: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("cannot summarize values")
    return {
        f"mean {direction}": float(array.mean()),
        f"median {direction}": float(np.median(array)),
        f"p05 {direction}": float(np.quantile(array, 0.05)),
        f"p95 {direction}": float(np.quantile(array, 0.95)),
        f"minimum {direction}": float(array.min()),
        f"maximum {direction}": float(array.max()),
    }


def proposed_event(
    event: Mapping[str, Any],
    record: Mapping[str, Any],
    project_root: Path,
) -> tuple[np.ndarray, tuple[float, float]]:
    sample_rate = int(record["sample_rate"])
    num_samples = int(record["num_samples"])
    source = SourceClip(
        source_id=str(event["source_id"]),
        label=str(event["label"]),
        interval_seconds=tuple(float(value) for value in event["source_interval_seconds"]),
        caption=str(event["label"]),
        audio_path=safe_path(project_root, event["source_path"], "source_path"),
    )
    duration = float(event["offset_seconds"]) - float(event["onset_seconds"])
    clip, crop = crop_source_event(
        source,
        duration,
        sample_rate,
        10.0,
        np.random.default_rng(0),
        prefer_maximum_variance_crop=True,
    )
    scale = 10.0 ** (float(event["gain_db"]) / 20.0) * float(record["family_gain"])
    clip = clip * scale
    start = int(round(float(event["onset_seconds"]) * sample_rate))
    stop = start + len(clip)
    if start < 0 or stop > num_samples:
        raise ValueError(f"proposed crop exceeds scene: {record['scene_id']}")
    canvas = np.zeros(num_samples, dtype=np.float32)
    canvas[start:stop] = clip
    return canvas, crop


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.expanduser().resolve()
    manifest_root = manifest.parent
    project_root = args.project_root.expanduser().resolve()
    scenes = load_base_scenes(manifest)
    candidates: list[dict[str, Any]] = []
    labels: set[str] = set()
    for record in scenes:
        events = record.get("events")
        if not isinstance(events, list):
            raise ValueError(f"missing events: {record.get('scene_id')}")
        for event in events:
            if not isinstance(event, dict) or event.get("event_kind") != "semantic":
                continue
            label = str(event["label"])
            labels.add(label)
            current = read_mono(
                safe_path(manifest_root, event["stem_path"], "stem_path"),
                int(record["sample_rate"]),
                int(record["num_samples"]),
            )
            proposed, proposed_crop = proposed_event(event, record, project_root)
            candidates.append(
                {
                    "scene_family_id": record["scene_family_id"],
                    "scene_id": record["scene_id"],
                    "event_id": event["event_id"],
                    "label": label,
                    "source_id": event["source_id"],
                    "current_crop_interval_seconds": event[
                        "source_crop_interval_seconds"
                    ],
                    "proposed_crop_interval_seconds": list(proposed_crop),
                    "current": current,
                    "proposed": proposed,
                }
            )

    device = torch.device(args.device)
    auditor = BEATsAudioSetAuditor.from_assets(
        args.beats_root,
        args.beats_checkpoint,
        args.audioset_labels,
        labels,
        device,
    )
    for start in range(0, len(candidates), args.batch_size):
        chunk = candidates[start : start + args.batch_size]
        streams = np.stack(
            [waveform for item in chunk for waveform in (item["current"], item["proposed"])]
        )
        probabilities = auditor.score(
            torch.from_numpy(streams),
            int(scenes[0]["sample_rate"]),
            args.batch_size * 2,
        )
        for index, item in enumerate(chunk):
            coordinate = auditor.label_indices[str(item["label"])]
            item["current_label_probability ↑"] = float(
                probabilities[index * 2, coordinate]
            )
            item["proposed_label_probability ↑"] = float(
                probabilities[index * 2 + 1, coordinate]
            )
            item["probability_delta ↑"] = (
                item["proposed_label_probability ↑"]
                - item["current_label_probability ↑"]
            )
            del item["current"]
            del item["proposed"]
        print(
            f"scored {min(start + len(chunk), len(candidates))}/{len(candidates)} events",
            flush=True,
        )

    current = [float(item["current_label_probability ↑"]) for item in candidates]
    proposed = [float(item["proposed_label_probability ↑"]) for item in candidates]
    deltas = [float(item["probability_delta ↑"]) for item in candidates]
    return {
        "format": FORMAT,
        "claim_boundary": (
            "Post-hoc crop-policy diagnostic; BEATs influenced policy comparison, "
            "so this report is not independent final evidence."
        ),
        "manifest": identity(manifest),
        "auditor": auditor.provenance,
        "counts": {
            "base_scene_count ↑": len(scenes),
            "semantic_event_count ↑": len(candidates),
            "label_count ↑": len(labels),
        },
        "summary": {
            "current_label_probability ↑": summarize(current, "↑"),
            "proposed_label_probability ↑": summarize(proposed, "↑"),
            "probability_delta ↑": summarize(deltas, "↑"),
            "events_improved_rate ↑": sum(delta > 0.0 for delta in deltas)
            / len(deltas),
            "events_improved_by_at_least_0.05_rate ↑": sum(
                delta >= 0.05 for delta in deltas
            )
            / len(deltas),
            "events_degraded_by_at_least_0.05_rate ↓": sum(
                delta <= -0.05 for delta in deltas
            )
            / len(deltas),
        },
        "events": sorted(
            candidates, key=lambda item: float(item["probability_delta ↑"])
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise ValueError(f"output exists; pass --overwrite: {output}")
    report = build_report(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps({"status": "ok", "output": str(output), **report["counts"]}))


if __name__ == "__main__":
    main()
