#!/usr/bin/env python3
"""Select one label-conditioned semantic center for every used QCES v5 source.

The frozen BEATs AudioSet classifier is a *data curator*, not an independent
evaluation model. Each minimum-duration 0.90-second crop is rendered with the
same normalization used by the QCES builder and placed at a fixed position in
a 10-second canvas. Candidate centers are legal only when the maximum-duration
1.25-second crop also fits. The highest probability for the source's audited
label is stored as an immutable crop bank entry.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
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
    read_mono_audio,
    resample_audio,
    rms,
)


FORMAT = "qces_v5_semantic_crop_bank_v2"
# The builder samples semantic durations in [0.90, 1.25] seconds. Selecting
# with the shortest possible crop guarantees that the selected acoustic core
# remains inside every longer crop centered at the same point. Candidate
# centers are nevertheless constrained by the maximum duration.
SELECTION_WINDOW_SECONDS = 0.90
MAXIMUM_BUILDER_WINDOW_SECONDS = 1.25
SELECTION_HOP_SECONDS = 0.25
RENDER_SAMPLE_RATE = 32_000
RENDER_DURATION_SECONDS = 10.0
RENDER_ONSET_SECONDS = 0.5
FADE_MILLISECONDS = 10.0
TARGET_RMS = 0.08


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
        default=baseline
        / "checkpoint/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    )
    parser.add_argument(
        "--audioset-labels",
        type=Path,
        default=baseline / "checkpoint/class_labels_indices.csv",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--listening-dir",
        type=Path,
        help="Optional output for selected/runner-up previews and a JSONL queue.",
    )
    parser.add_argument("--weakest-per-class", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.top_k < 2 or args.weakest_per_class <= 0:
        parser.error("batch size must be positive, top-k >= 2, weakest/class > 0")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_source_path(project_root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("source_path must be a non-empty string")
    path = (project_root / raw_path).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as error:
        raise ValueError(f"source path escapes project root: {raw_path}") from error
    if not path.is_file():
        raise ValueError(f"missing source audio: {path}")
    return path


def source_key(source_id: str, label: str) -> str:
    return f"{label}::{source_id}"


def collect_sources(manifest: Path, project_root: Path) -> list[dict[str, Any]]:
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
            if not isinstance(scene_id, str) or not scene_id:
                raise ValueError(f"missing scene_id at {manifest}:{line_number}")
            scenes.setdefault(scene_id, record)
    if not scenes:
        raise ValueError("manifest contains no base scenes")

    by_key: dict[str, dict[str, Any]] = {}
    usage_counts: dict[str, int] = defaultdict(int)
    for scene_id in sorted(scenes):
        events = scenes[scene_id].get("events")
        if not isinstance(events, list):
            raise ValueError(f"scene lacks events: {scene_id}")
        for event in events:
            if not isinstance(event, dict) or event.get("event_kind") != "semantic":
                continue
            label = str(event["label"])
            source_id = str(event["source_id"])
            key = source_key(source_id, label)
            candidate = {
                "key": key,
                "source_id": source_id,
                "label": label,
                "source_path": str(event["source_path"]),
                "source_sha256": str(event["source_sha256"]),
                "source_interval_seconds": [
                    float(value) for value in event["source_interval_seconds"]
                ],
            }
            previous = by_key.setdefault(key, candidate)
            if previous != candidate:
                raise ValueError(f"inconsistent source identity for {key}")
            usage_counts[key] += 1

    for key, item in by_key.items():
        path = safe_source_path(project_root, item["source_path"])
        actual_sha = sha256_file(path)
        if actual_sha != item["source_sha256"]:
            raise ValueError(f"source hash mismatch for {key}")
        interval = item["source_interval_seconds"]
        if (
            len(interval) != 2
            or not all(math.isfinite(value) for value in interval)
            or interval[0] < 0.0
            or interval[1] - interval[0] < MAXIMUM_BUILDER_WINDOW_SECONDS
        ):
            raise ValueError(f"source interval is too short or invalid for {key}")
        item["usage_count"] = usage_counts[key]
    return [by_key[key] for key in sorted(by_key)]


def candidate_starts(
    source_start: int,
    source_end: int,
    window_samples: int,
    hop_samples: int,
) -> list[int]:
    last = source_end - window_samples
    if source_start < 0 or last < source_start:
        raise ValueError("invalid candidate bounds")
    starts = list(range(source_start, last + 1, hop_samples))
    if starts[-1] != last:
        starts.append(last)
    return starts


def render_candidate_clip(
    waveform: np.ndarray,
    source_rate: int,
    crop_start: int,
    window_samples: int,
    duration_seconds: float = SELECTION_WINDOW_SECONDS,
) -> np.ndarray:
    clip = resample_audio(
        waveform[crop_start : crop_start + window_samples],
        source_rate,
        RENDER_SAMPLE_RATE,
    )
    target_samples = int(round(duration_seconds * RENDER_SAMPLE_RATE))
    if clip.size < target_samples:
        clip = np.pad(clip, (0, target_samples - clip.size))
    else:
        clip = clip[:target_samples]
    clip = clip - float(np.mean(clip))
    fade_samples = min(
        int(round(FADE_MILLISECONDS * RENDER_SAMPLE_RATE / 1000.0)),
        clip.size // 2,
    )
    if fade_samples:
        phase = np.linspace(0.0, math.pi / 2.0, fade_samples, dtype=np.float32)
        ramp = np.sin(phase) ** 2
        clip[:fade_samples] *= ramp
        clip[-fade_samples:] *= ramp[::-1]
    clip_rms = rms(clip)
    if clip_rms < 1e-5:
        return np.zeros(target_samples, dtype=np.float32)
    return (clip * (TARGET_RMS / clip_rms)).astype(np.float32)


def canvas_for_clip(clip: np.ndarray) -> np.ndarray:
    canvas = np.zeros(
        int(round(RENDER_DURATION_SECONDS * RENDER_SAMPLE_RATE)), dtype=np.float32
    )
    start = int(round(RENDER_ONSET_SECONDS * RENDER_SAMPLE_RATE))
    canvas[start : start + clip.size] = clip
    return canvas


def score_source(
    source: Mapping[str, Any],
    project_root: Path,
    auditor: BEATsAudioSetAuditor,
    batch_size: int,
    top_k: int,
) -> dict[str, Any]:
    path = safe_source_path(project_root, source["source_path"])
    waveform, source_rate = read_mono_audio(path)
    interval = source["source_interval_seconds"]
    source_start = max(0, int(round(float(interval[0]) * source_rate)))
    source_end = min(waveform.size, int(round(float(interval[1]) * source_rate)))
    window_samples = int(math.ceil(SELECTION_WINDOW_SECONDS * source_rate))
    maximum_window_samples = int(
        math.ceil(MAXIMUM_BUILDER_WINDOW_SECONDS * source_rate)
    )
    hop_samples = max(1, int(round(SELECTION_HOP_SECONDS * source_rate)))
    maximum_starts = candidate_starts(
        source_start, source_end, maximum_window_samples, hop_samples
    )
    centers = [
        start + maximum_window_samples // 2 for start in maximum_starts
    ]
    starts = [center - window_samples // 2 for center in centers]
    coordinate = auditor.label_indices[str(source["label"])]
    scores: list[float] = []
    silent_candidates = 0
    for chunk_start in range(0, len(starts), batch_size):
        chunk_starts = starts[chunk_start : chunk_start + batch_size]
        clips = [
            render_candidate_clip(waveform, source_rate, start, window_samples)
            for start in chunk_starts
        ]
        silent_candidates += sum(not bool(np.any(clip)) for clip in clips)
        canvases = np.stack([canvas_for_clip(clip) for clip in clips])
        probabilities = auditor.score(
            torch.from_numpy(canvases), RENDER_SAMPLE_RATE, batch_size
        )
        scores.extend(float(value) for value in probabilities[:, coordinate])
    ranking = sorted(range(len(starts)), key=lambda index: (-scores[index], starts[index]))
    selected = ranking[0]
    runner_up = ranking[1] if len(ranking) > 1 else ranking[0]

    def ranked_item(index: int) -> dict[str, Any]:
        start_seconds = starts[index] / source_rate
        maximum_start_seconds = maximum_starts[index] / source_rate
        return {
            "crop_interval_seconds": [
                start_seconds,
                start_seconds + SELECTION_WINDOW_SECONDS,
            ],
            "maximum_crop_interval_seconds": [
                maximum_start_seconds,
                maximum_start_seconds + MAXIMUM_BUILDER_WINDOW_SECONDS,
            ],
            "center_seconds": centers[index] / source_rate,
            "label_probability ↑": scores[index],
        }

    return {
        **dict(source),
        "source_sample_rate": source_rate,
        "candidate_count ↑": len(starts),
        "silent_candidate_count ↓": silent_candidates,
        "selected": ranked_item(selected),
        "runner_up": ranked_item(runner_up),
        "selection_margin ↑": scores[selected] - scores[runner_up],
        "top_candidates": [ranked_item(index) for index in ranking[:top_k]],
    }


def write_listening_queue(
    entries: Sequence[Mapping[str, Any]],
    listening_dir: Path,
    project_root: Path,
    weakest_per_class: int,
) -> dict[str, Any]:
    if listening_dir.exists():
        raise FileExistsError(f"listening directory exists: {listening_dir}")
    audio_dir = listening_dir / "audio"
    audio_dir.mkdir(parents=True)
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_label[str(entry["label"])].append(entry)
    selected_entries: dict[str, Mapping[str, Any]] = {}
    for label, class_entries in by_label.items():
        weakest = sorted(
            class_entries,
            key=lambda item: (
                float(item["selected"]["label_probability ↑"]),
                float(item["selection_margin ↑"]),
                str(item["source_id"]),
            ),
        )[:weakest_per_class]
        for item in weakest:
            selected_entries[str(item["key"])] = item

    queue_path = listening_dir / "listening_queue.jsonl"
    with queue_path.open("w", encoding="utf-8") as handle:
        for queue_index, key in enumerate(sorted(selected_entries), 1):
            entry = selected_entries[key]
            waveform, source_rate = read_mono_audio(
                safe_source_path(project_root, entry["source_path"])
            )
            window_samples = int(math.ceil(SELECTION_WINDOW_SECONDS * source_rate))
            paths: dict[str, str] = {}
            for role in ("selected", "runner_up"):
                crop_start = int(
                    round(float(entry[role]["crop_interval_seconds"][0]) * source_rate)
                )
                clip = render_candidate_clip(
                    waveform,
                    source_rate,
                    crop_start,
                    window_samples,
                    float(entry[role]["crop_interval_seconds"][1])
                    - float(entry[role]["crop_interval_seconds"][0]),
                )
                preview = np.pad(
                    clip,
                    (
                        int(0.25 * RENDER_SAMPLE_RATE),
                        int(0.25 * RENDER_SAMPLE_RATE),
                    ),
                )
                filename = f"{queue_index:03d}_{entry['label']}_{entry['source_id']}_{role}.wav"
                filename = filename.replace("/", "_").replace(" ", "_")
                path = audio_dir / filename
                sf.write(path, preview, RENDER_SAMPLE_RATE, subtype="PCM_16")
                paths[role] = str(path.relative_to(listening_dir))
            row = {
                "queue_index": queue_index,
                "label": entry["label"],
                "source_id": entry["source_id"],
                "source_path": entry["source_path"],
                "selected_source_interval_seconds": entry["selected"][
                    "crop_interval_seconds"
                ],
                "selected_probability ↑": entry["selected"]["label_probability ↑"],
                "runner_up_source_interval_seconds": entry["runner_up"][
                    "crop_interval_seconds"
                ],
                "runner_up_probability ↑": entry["runner_up"]["label_probability ↑"],
                "selection_margin ↑": entry["selection_margin ↑"],
                "selected_preview_path": paths["selected"],
                "runner_up_preview_path": paths["runner_up"],
                "listen_instruction": (
                    "Does SELECTED clearly contain the displayed label? Compare with "
                    "RUNNER_UP only when selected is ambiguous; mark pass/fail/unsure."
                ),
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "queue_path": str(queue_path.resolve()),
        "items ↑": len(selected_entries),
        "weakest_per_class": weakest_per_class,
        "class_count ↑": len(by_label),
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    project_root = args.project_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; use --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = collect_sources(manifest, project_root)
    labels = sorted({str(source["label"]) for source in sources})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    auditor = BEATsAudioSetAuditor.from_assets(
        args.beats_root,
        args.beats_checkpoint,
        args.audioset_labels,
        labels,
        device,
    )
    entries = []
    for index, source in enumerate(sources, 1):
        entries.append(
            score_source(source, project_root, auditor, args.batch_size, args.top_k)
        )
        print(
            f"scored source {index}/{len(sources)} "
            f"{source['label']}::{source['source_id']}",
            flush=True,
        )
    summary_by_class = {}
    for label in labels:
        class_entries = [entry for entry in entries if entry["label"] == label]
        probabilities = np.asarray(
            [entry["selected"]["label_probability ↑"] for entry in class_entries],
            dtype=np.float64,
        )
        summary_by_class[label] = {
            "source_count ↑": len(class_entries),
            "mean_selected_probability ↑": float(probabilities.mean()),
            "minimum_selected_probability ↑": float(probabilities.min()),
        }
    report: dict[str, Any] = {
        "format": FORMAT,
        "curator_boundary": (
            "BEATs selects source windows using audited labels and is therefore "
            "forbidden as an independent final auditor for this rebuilt artifact."
        ),
        "manifest": {
            "path": str(manifest),
            "sha256": sha256_file(manifest),
        },
        "selection": {
            "window_seconds": SELECTION_WINDOW_SECONDS,
            "maximum_builder_window_seconds": MAXIMUM_BUILDER_WINDOW_SECONDS,
            "hop_seconds": SELECTION_HOP_SECONDS,
            "render_sample_rate": RENDER_SAMPLE_RATE,
            "render_duration_seconds": RENDER_DURATION_SECONDS,
            "fixed_onset_seconds": RENDER_ONSET_SECONDS,
            "target_rms": TARGET_RMS,
            "fade_milliseconds": FADE_MILLISECONDS,
            "ranking": (
                "maximum frozen BEATs probability at the audited label coordinate "
                "on the minimum 0.90-second builder crop"
            ),
            "center_constraint": (
                "every candidate center must also admit the maximum 1.25-second "
                "builder crop inside the audited source interval"
            ),
            "tie_break": "earliest valid source window",
        },
        "counts": {
            "sources ↑": len(entries),
            "classes ↑": len(labels),
            "candidates ↑": sum(int(entry["candidate_count ↑"]) for entry in entries),
            "silent_candidates ↓": sum(
                int(entry["silent_candidate_count ↓"]) for entry in entries
            ),
        },
        "summary_by_class": summary_by_class,
        "beats_provenance": auditor.provenance,
        "entries": entries,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    if args.listening_dir is not None:
        queue = write_listening_queue(
            entries,
            args.listening_dir.expanduser().resolve(),
            project_root,
            args.weakest_per_class,
        )
        print(json.dumps(queue, ensure_ascii=False, indent=2), flush=True)
    print(f"crop bank ready: {output}", flush=True)


if __name__ == "__main__":
    main()
