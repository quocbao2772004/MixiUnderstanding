#!/usr/bin/env python3
"""Build a detector source bank directly from normalized FUSS/FSD50K positives.

This is a detector-training artifact, not a QA-scene builder.  It answers two
practical questions before expanding the SED head:

1. Which labels have real positive audio in the local FUSS/FSD50K route?
2. What is the largest supervised ontology we can actually train from?

For every selected source, the builder renders a deterministic 10-second clip
with one labelled event crop placed at a known timestamp.  The crop is selected
by maximum RMS inside the single-label source to avoid random silent crops.
This is weaker than a human/strong-label timestamp, but it is a real positive
from a labelled isolated source and is much safer than pretending all 200
requested labels exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FUSS_ROOT = PROJECT_ROOT / "upstream/fuss_v1.3"
DEFAULT_FUSS_MANIFEST = PROJECT_ROOT / "upstream/qces_v5_normalized_20260722/fuss_sources.jsonl"
DEFAULT_FSD_MANIFEST = PROJECT_ROOT / "upstream/qces_v5_normalized_20260722/fsd50k_labels_provenance.jsonl"
DEFAULT_LABEL_BANK = PROJECT_ROOT / "outputs/qces_open_event_v2/label_bank_200.txt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1"

FORMAT = "qces_detector_sourcebank_fuss_fsd50k_v1"
SPLITS = ("train", "validation", "eval")
MANIFEST_SPLIT_NAMES = {"train": "train", "validation": "val", "eval": "test"}


@dataclass(frozen=True)
class SourceRow:
    source_id: str
    label: str
    split: str
    audio_path: str
    duration_seconds: float
    sha256: str
    creator_id: str
    uploader_id: str
    attribution: str


@dataclass(frozen=True)
class LabelCoverage:
    label: str
    in_requested_label_bank: bool
    all_sources_ge_event: int
    train_sources_ge_event: int
    validation_sources_ge_event: int
    eval_sources_ge_event: int
    train_active_seconds_ge_event: float
    validation_active_seconds_ge_event: float
    eval_active_seconds_ge_event: float
    total_active_seconds_ge_event: float
    train_positive: bool
    ready_weak: bool
    ready_medium: bool
    ready_strong: bool


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fuss-root", type=Path, default=DEFAULT_FUSS_ROOT)
    parser.add_argument("--fuss-manifest", type=Path, default=DEFAULT_FUSS_MANIFEST)
    parser.add_argument("--fsd-manifest", type=Path, default=DEFAULT_FSD_MANIFEST)
    parser.add_argument("--requested-label-bank", type=Path, default=DEFAULT_LABEL_BANK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--render-ontology",
        choices=(
            "train_positive",
            "ready_weak",
            "ready_medium",
            "ready_strong",
            "requested_train_positive",
            "requested_ready_weak",
            "requested_ready_medium",
            "requested_ready_strong",
        ),
        default="train_positive",
        help=(
            "Which ontology to render into train/val/test manifests. All ontology "
            "lists are written regardless of this choice."
        ),
    )
    parser.add_argument("--event-seconds", type=float, default=1.25)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--event-onset-seconds", type=float, default=0.50)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--target-rms", type=float, default=0.08)
    parser.add_argument("--fade-ms", type=float, default=10.0)
    parser.add_argument("--rms-hop-seconds", type=float, default=0.05)
    parser.add_argument(
        "--max-sources-per-label-per-split",
        type=int,
        default=0,
        help="0 means use every eligible source. Positive values are deterministic caps.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write audit/ontology/manifest rows but do not render WAV files.",
    )
    parser.add_argument(
        "--keep-silent-crops",
        action="store_true",
        help=(
            "Keep rendered crops whose selected RMS is effectively zero. Default "
            "skips them so a labelled but silent source cannot become a positive."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.event_seconds <= 0 or args.clip_seconds <= args.event_seconds:
        parser.error("--clip-seconds must be larger than --event-seconds")
    if args.event_onset_seconds < 0 or args.event_onset_seconds + args.event_seconds > args.clip_seconds:
        parser.error("event crop must fit inside clip")
    if args.sample_rate <= 0 or args.target_rms <= 0 or args.rms_hop_seconds <= 0:
        parser.error("sample rate, target RMS, and RMS hop must be positive")
    if args.max_sources_per_label_per_split < 0:
        parser.error("--max-sources-per-label-per-split must be >= 0")
    return args


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"non-object JSONL row at {path}:{line_number}")
            yield payload


def write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_text_lines(path: Path, lines: Sequence[str], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def load_requested_labels(path: Path) -> list[str]:
    labels: list[str] = []
    if not path.exists():
        return labels
    for raw in path.read_text(encoding="utf-8").splitlines():
        label = raw.strip()
        if label and label not in labels:
            labels.append(label)
    return labels


def load_sources(fuss_manifest: Path, fsd_manifest: Path) -> list[SourceRow]:
    fuss_by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(fuss_manifest):
        source_id = str(row["source_id"])
        if source_id in fuss_by_id:
            raise ValueError(f"duplicate FUSS source_id: {source_id}")
        fuss_by_id[source_id] = row

    rows: list[SourceRow] = []
    for row in read_jsonl(fsd_manifest):
        source_id = str(row["source_id"])
        fuss = fuss_by_id.get(source_id)
        if fuss is None:
            continue
        labels = row.get("labels")
        if not isinstance(labels, list) or len(labels) != 1:
            continue
        if str(fuss["split"]) != str(row["split"]):
            continue
        rows.append(
            SourceRow(
                source_id=source_id,
                label=str(labels[0]),
                split=str(fuss["split"]),
                audio_path=str(fuss["audio_path"]),
                duration_seconds=float(fuss["duration_seconds"]),
                sha256=str(fuss["sha256"]),
                creator_id=str(row.get("creator_id") or ""),
                uploader_id=str(row.get("uploader_id") or ""),
                attribution=str(row.get("attribution") or ""),
            )
        )
    return sorted(rows, key=lambda item: (item.label, item.split, int(item.source_id)))


def stable_score(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def compute_coverage(
    sources: Sequence[SourceRow],
    requested: Sequence[str],
    *,
    event_seconds: float,
) -> list[LabelCoverage]:
    requested_set = set(requested)
    by_label_split: dict[tuple[str, str], list[SourceRow]] = defaultdict(list)
    for source in sources:
        if source.duration_seconds + 1e-9 >= event_seconds:
            by_label_split[(source.label, source.split)].append(source)

    labels = sorted({source.label for source in sources} | requested_set)
    rows: list[LabelCoverage] = []
    for label in labels:
        split_counts = {
            split: len(by_label_split.get((label, split), ())) for split in SPLITS
        }
        split_seconds = {
            split: float(sum(source.duration_seconds for source in by_label_split.get((label, split), ())))
            for split in SPLITS
        }
        all_sources = sum(split_counts.values())
        total_seconds = sum(split_seconds.values())
        train_positive = split_counts["train"] >= 1
        ready_weak = (
            split_counts["train"] >= 4
            and split_counts["validation"] >= 1
            and split_counts["eval"] >= 1
        )
        ready_medium = (
            split_counts["train"] >= 10
            and split_counts["validation"] >= 2
            and split_counts["eval"] >= 2
        )
        ready_strong = (
            split_counts["train"] >= 20
            and split_counts["validation"] >= 2
            and split_counts["eval"] >= 2
            and split_seconds["train"] >= 60.0
        )
        rows.append(
            LabelCoverage(
                label=label,
                in_requested_label_bank=label in requested_set,
                all_sources_ge_event=all_sources,
                train_sources_ge_event=split_counts["train"],
                validation_sources_ge_event=split_counts["validation"],
                eval_sources_ge_event=split_counts["eval"],
                train_active_seconds_ge_event=split_seconds["train"],
                validation_active_seconds_ge_event=split_seconds["validation"],
                eval_active_seconds_ge_event=split_seconds["eval"],
                total_active_seconds_ge_event=total_seconds,
                train_positive=train_positive,
                ready_weak=ready_weak,
                ready_medium=ready_medium,
                ready_strong=ready_strong,
            )
        )
    return sorted(
        rows,
        key=lambda item: (
            -item.train_sources_ge_event,
            -item.validation_sources_ge_event,
            -item.eval_sources_ge_event,
            item.label,
        ),
    )


def labels_for_mode(coverage: Sequence[LabelCoverage], mode: str) -> list[str]:
    requested_only = mode.startswith("requested_")
    key = mode.removeprefix("requested_")
    labels: list[str] = []
    for item in coverage:
        if requested_only and not item.in_requested_label_bank:
            continue
        if key == "train_positive" and item.train_positive:
            labels.append(item.label)
        elif key == "ready_weak" and item.ready_weak:
            labels.append(item.label)
        elif key == "ready_medium" and item.ready_medium:
            labels.append(item.label)
        elif key == "ready_strong" and item.ready_strong:
            labels.append(item.label)
    return sorted(labels)


def read_mono_audio(path: Path) -> tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if waveform.size == 0:
        return np.zeros(0, dtype=np.float32), int(sample_rate)
    mono = waveform.mean(axis=1).astype(np.float32)
    return mono, int(sample_rate)


def resample_linear(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    if waveform.size == 0:
        return np.zeros(0, dtype=np.float32)
    duration = waveform.size / float(source_rate)
    target_size = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, waveform.size, endpoint=False)
    target_x = np.linspace(0.0, duration, target_size, endpoint=False)
    return np.interp(target_x, source_x, waveform).astype(np.float32)


def max_rms_crop_start(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    event_seconds: float,
    hop_seconds: float,
) -> int:
    window = int(round(event_seconds * sample_rate))
    hop = max(1, int(round(hop_seconds * sample_rate)))
    if waveform.size <= window:
        return 0
    best_start = 0
    best_energy = -1.0
    squared = np.square(waveform.astype(np.float64))
    cumsum = np.concatenate([[0.0], np.cumsum(squared)])
    starts = list(range(0, waveform.size - window + 1, hop))
    if starts[-1] != waveform.size - window:
        starts.append(waveform.size - window)
    for start in starts:
        energy = float(cumsum[start + window] - cumsum[start])
        if energy > best_energy:
            best_energy = energy
            best_start = start
    return best_start


def render_source_clip(
    source: SourceRow,
    *,
    fuss_root: Path,
    destination: Path,
    clip_seconds: float,
    event_seconds: float,
    event_onset_seconds: float,
    sample_rate: int,
    target_rms: float,
    fade_ms: float,
    rms_hop_seconds: float,
) -> dict[str, Any]:
    source_path = (fuss_root / source.audio_path).resolve()
    waveform, source_rate = read_mono_audio(source_path)
    if waveform.size == 0:
        raise ValueError(f"empty source audio: {source_path}")
    crop_start = max_rms_crop_start(
        waveform,
        source_rate,
        event_seconds=event_seconds,
        hop_seconds=rms_hop_seconds,
    )
    crop_samples = int(round(event_seconds * source_rate))
    crop = waveform[crop_start : crop_start + crop_samples]
    if crop.size < crop_samples:
        crop = np.pad(crop, (0, crop_samples - crop.size))
    crop = resample_linear(crop, source_rate, sample_rate)
    target_crop_samples = int(round(event_seconds * sample_rate))
    if crop.size < target_crop_samples:
        crop = np.pad(crop, (0, target_crop_samples - crop.size))
    elif crop.size > target_crop_samples:
        crop = crop[:target_crop_samples]
    crop = crop - float(np.mean(crop))
    fade_samples = min(int(round(fade_ms * sample_rate / 1000.0)), crop.size // 2)
    if fade_samples > 0:
        phase = np.linspace(0.0, math.pi / 2.0, fade_samples, dtype=np.float32)
        ramp = np.sin(phase) ** 2
        crop[:fade_samples] *= ramp
        crop[-fade_samples:] *= ramp[::-1]
    crop_rms = float(np.sqrt(np.mean(np.square(crop, dtype=np.float64))))
    silent = crop_rms < 1e-6
    if not silent:
        crop = (crop * (target_rms / crop_rms)).astype(np.float32)
    else:
        crop = np.zeros_like(crop, dtype=np.float32)

    clip_samples = int(round(clip_seconds * sample_rate))
    onset_sample = int(round(event_onset_seconds * sample_rate))
    clip = np.zeros(clip_samples, dtype=np.float32)
    clip[onset_sample : onset_sample + crop.size] = crop[: max(0, clip_samples - onset_sample)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, clip, sample_rate, subtype="PCM_16")
    source_start_seconds = crop_start / float(source_rate)
    return {
        "selected_source_interval_seconds": [
            round(source_start_seconds, 6),
            round(source_start_seconds + event_seconds, 6),
        ],
        "selected_crop_rms": crop_rms,
        "selected_crop_silent": silent,
    }


def manifest_row(
    source: SourceRow,
    *,
    output_audio_path: Path,
    render_info: Mapping[str, Any] | None,
    clip_seconds: float,
    event_seconds: float,
    event_onset_seconds: float,
    sample_rate: int,
) -> dict[str, Any]:
    event = {
        "event_id": f"{source.source_id}_{source.label}",
        "event_kind": "semantic",
        "label": source.label,
        "onset_seconds": round(event_onset_seconds, 6),
        "offset_seconds": round(event_onset_seconds + event_seconds, 6),
        "source_id": source.source_id,
        "source_path": str(source.audio_path),
        "source_sha256": source.sha256,
        "source_interval_seconds": (
            render_info.get("selected_source_interval_seconds")
            if render_info
            else [0.0, round(event_seconds, 6)]
        ),
        "creator_id": source.creator_id,
        "uploader_id": source.uploader_id,
    }
    if render_info:
        event["selected_crop_rms"] = render_info["selected_crop_rms"]
        event["selected_crop_silent"] = render_info["selected_crop_silent"]
    return {
        "format": FORMAT,
        "scene_id": f"det_{source.split}_{source.source_id}_{source.label}".replace("/", "_"),
        "source_id": source.source_id,
        "split": MANIFEST_SPLIT_NAMES[source.split],
        "mixture_path": project_relative(output_audio_path),
        "duration_seconds": clip_seconds,
        "sample_rate": sample_rate,
        "events": [event],
        "labels": [source.label],
        "provenance": {
            "source_route": "FUSS-v1.3/FSD50K-v1.0 single-label positive",
            "attribution": source.attribution,
            "crop_policy": "max-rms fixed-duration crop from labelled isolated source",
        },
    }


def maybe_cap_sources(
    sources: Sequence[SourceRow],
    *,
    max_sources_per_label_per_split: int,
) -> list[SourceRow]:
    grouped: dict[tuple[str, str], list[SourceRow]] = defaultdict(list)
    for source in sources:
        grouped[(source.label, source.split)].append(source)
    kept: list[SourceRow] = []
    for key, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda item: stable_score(key[0], key[1], item.source_id))
        if max_sources_per_label_per_split > 0:
            ordered = ordered[:max_sources_per_label_per_split]
        kept.extend(ordered)
    return sorted(kept, key=lambda item: (item.split, item.label, int(item.source_id)))


def write_coverage_tsv(path: Path, rows: Sequence[LabelCoverage], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            fieldnames = list(asdict(rows[0]).keys()) if rows else ["label"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"output dir not empty: {output_dir}; use --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_labels = load_requested_labels(args.requested_label_bank.resolve())
    sources = load_sources(args.fuss_manifest.resolve(), args.fsd_manifest.resolve())
    eligible_sources = [
        source
        for source in sources
        if source.duration_seconds + 1e-9 >= float(args.event_seconds)
    ]
    coverage = compute_coverage(
        sources,
        requested_labels,
        event_seconds=float(args.event_seconds),
    )

    ontology_modes = (
        "train_positive",
        "ready_weak",
        "ready_medium",
        "ready_strong",
        "requested_train_positive",
        "requested_ready_weak",
        "requested_ready_medium",
        "requested_ready_strong",
    )
    ontologies = {mode: labels_for_mode(coverage, mode) for mode in ontology_modes}
    for mode, labels in ontologies.items():
        write_text_lines(output_dir / f"ontology_{mode}.txt", labels, overwrite=args.overwrite)

    write_coverage_tsv(output_dir / "label_coverage.tsv", coverage, overwrite=args.overwrite)
    write_json(
        output_dir / "label_coverage.json",
        {
            "format": FORMAT,
            "event_seconds": args.event_seconds,
            "requested_label_bank": (
                str(args.requested_label_bank.resolve())
                if args.requested_label_bank.exists()
                else None
            ),
            "counts": {
                "requested_label_bank_labels": len(requested_labels),
                "all_exact_fuss_fsd_labels": len({source.label for source in sources}),
                "all_sources": len(sources),
                "eligible_sources_ge_event_seconds": len(eligible_sources),
                "train_positive_labels": len(ontologies["train_positive"]),
                "ready_weak_labels": len(ontologies["ready_weak"]),
                "ready_medium_labels": len(ontologies["ready_medium"]),
                "ready_strong_labels": len(ontologies["ready_strong"]),
                "requested_train_positive_labels": len(ontologies["requested_train_positive"]),
                "requested_ready_weak_labels": len(ontologies["requested_ready_weak"]),
                "requested_ready_medium_labels": len(ontologies["requested_ready_medium"]),
                "requested_ready_strong_labels": len(ontologies["requested_ready_strong"]),
            },
            "definitions": {
                "train_positive": "train sources >= 1",
                "ready_weak": "train >= 4, validation >= 1, eval >= 1",
                "ready_medium": "train >= 10, validation >= 2, eval >= 2",
                "ready_strong": "train >= 20, validation >= 2, eval >= 2, train active seconds >= 60",
            },
            "rows": [asdict(row) for row in coverage],
        },
        overwrite=args.overwrite,
    )

    render_labels = set(ontologies[args.render_ontology])
    selected_sources = [
        source for source in eligible_sources if source.label in render_labels
    ]
    selected_sources = maybe_cap_sources(
        selected_sources,
        max_sources_per_label_per_split=args.max_sources_per_label_per_split,
    )
    rendered_audio_root = output_dir / "audio"
    manifest_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "eval": [],
    }
    silent_count = 0
    skipped_silent_count = 0
    for index, source in enumerate(selected_sources, 1):
        safe_label = source.label.replace("/", "_").replace(" ", "_")
        destination = rendered_audio_root / source.split / safe_label / f"{source.source_id}.wav"
        render_info = None
        if not args.manifest_only:
            render_info = render_source_clip(
                source,
                fuss_root=args.fuss_root.resolve(),
                destination=destination,
                clip_seconds=float(args.clip_seconds),
                event_seconds=float(args.event_seconds),
                event_onset_seconds=float(args.event_onset_seconds),
                sample_rate=int(args.sample_rate),
                target_rms=float(args.target_rms),
                fade_ms=float(args.fade_ms),
                rms_hop_seconds=float(args.rms_hop_seconds),
            )
            if bool(render_info["selected_crop_silent"]):
                silent_count += 1
                if not args.keep_silent_crops:
                    skipped_silent_count += 1
                    continue
        row = manifest_row(
            source,
            output_audio_path=destination,
            render_info=render_info,
            clip_seconds=float(args.clip_seconds),
            event_seconds=float(args.event_seconds),
            event_onset_seconds=float(args.event_onset_seconds),
            sample_rate=int(args.sample_rate),
        )
        manifest_by_split[source.split].append(row)
        if index % 500 == 0 or index == len(selected_sources):
            print(
                f"processed {index}/{len(selected_sources)} "
                f"label={source.label} split={source.split}",
                flush=True,
            )

    manifest_paths: dict[str, str] = {}
    for upstream_split, rows in manifest_by_split.items():
        split_name = MANIFEST_SPLIT_NAMES[upstream_split]
        path = output_dir / f"detector_source_manifest_{split_name}.jsonl"
        write_jsonl(path, rows, overwrite=args.overwrite)
        manifest_paths[split_name] = str(path.resolve())

    rendered_counts = Counter(source.split for source in selected_sources)
    summary = {
        "format": FORMAT,
        "render_ontology": args.render_ontology,
        "rendered_audio": not args.manifest_only,
        "render_config": {
            "clip_seconds": args.clip_seconds,
            "event_seconds": args.event_seconds,
            "event_onset_seconds": args.event_onset_seconds,
            "sample_rate": args.sample_rate,
            "target_rms": args.target_rms,
            "fade_ms": args.fade_ms,
            "rms_hop_seconds": args.rms_hop_seconds,
            "max_sources_per_label_per_split": args.max_sources_per_label_per_split,
        },
        "counts": {
            "render_labels": len(render_labels),
            "render_sources": len(selected_sources),
            "render_sources_train": rendered_counts.get("train", 0),
            "render_sources_validation": rendered_counts.get("validation", 0),
            "render_sources_eval": rendered_counts.get("eval", 0),
            "silent_selected_crops": silent_count,
            "skipped_silent_crops": skipped_silent_count,
        },
        "coverage_counts": {
            mode: len(labels) for mode, labels in ontologies.items()
        },
        "manifest_paths": manifest_paths,
        "ontology_paths": {
            mode: str((output_dir / f"ontology_{mode}.txt").resolve())
            for mode in ontology_modes
        },
        "source_inputs": {
            "fuss_root": str(args.fuss_root.resolve()),
            "fuss_manifest": str(args.fuss_manifest.resolve()),
            "fuss_manifest_sha256": sha256_file(args.fuss_manifest.resolve()),
            "fsd_manifest": str(args.fsd_manifest.resolve()),
            "fsd_manifest_sha256": sha256_file(args.fsd_manifest.resolve()),
        },
        "important_limit": (
            "Current FUSS/FSD50K normalized source route contains fewer than 200 "
            "exact single-label classes, so a true 200-class supervised detector "
            "cannot be built from this source alone."
        ),
    }
    write_json(output_dir / "sourcebank_summary.json", summary, overwrite=args.overwrite)
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2), flush=True)
    print(f"sourcebank ready: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
