#!/usr/bin/env python3
"""Build the deterministic paired-scene QA removal v2 overfit dataset."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import platform
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy
import soundfile as sf
from scipy.signal import resample_poly


CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.data.qa_schema import (  # noqa: E402
    NON_OVERLAP_SNR_MEASUREMENT,
    NO_EDIT_TARGET,
    ORDINAL_CANDIDATE_ORDER_RULE,
    ORDINAL_SEMANTIC_ORDER_RULE,
    OVERLAP_SNR_MEASUREMENT,
    QUESTION_TYPES,
    RELATIVE_SEMANTIC_ORDER_RULE,
    SCHEMA_VERSION,
    parse_record,
)
from mixi_understanding.scripts.validate_qa_dataset import (  # noqa: E402
    validate_dataset,
)


BUILDER_VERSION = "2.3.0"
DEFAULT_SAMPLE_RATE = 32_000
DEFAULT_DURATION_SECONDS = 10.0
DEFAULT_EVENT_DURATION_SECONDS = 1.25
DEFAULT_OVERLAP_INTERFERENCE_DURATION_SECONDS = 3.75
DEFAULT_NON_OVERLAP_INTERFERENCE_DURATION_SECONDS = 1.5
DEFAULT_OBSTRUCTIVE_SNR_DB = -3.0
DEFAULT_BENIGN_OVERLAP_SNR_DB = 18.0
DEFAULT_NON_OVERLAP_SNR_DB = 6.0
DEFAULT_HEADROOM = 0.95
EPSILON = 1e-12


@dataclass(frozen=True)
class SourceClip:
    """One AudioTime source containing exactly one timestamped event."""

    source_id: str
    label: str
    interval_seconds: Tuple[float, float]
    caption: str
    audio_path: Path

    @property
    def duration_seconds(self) -> float:
        return self.interval_seconds[1] - self.interval_seconds[0]


@dataclass(frozen=True)
class FamilySources:
    """Distinct semantic and shared-label interference clips for four scenes."""

    semantic_labels: Tuple[str, str]
    semantic_clips: Tuple[Tuple[SourceClip, ...], Tuple[SourceClip, ...]]
    interference_clips: Tuple[SourceClip, SourceClip, SourceClip, SourceClip]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def lowercase_sha256(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError("must be a lowercase 64-character SHA256 digest")
    return value


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--audiotime-root",
        type=Path,
        default=project_root / "AudioTime-recovered" / "train5000_timestamp",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "removal_qa_synthetic_v2",
    )
    parser.add_argument("--sample-rate", type=positive_int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--duration-seconds", type=positive_float, default=DEFAULT_DURATION_SECONDS
    )
    parser.add_argument(
        "--event-duration-seconds",
        type=positive_float,
        default=DEFAULT_EVENT_DURATION_SECONDS,
    )
    parser.add_argument(
        "--overlap-interference-duration-seconds",
        type=positive_float,
        default=DEFAULT_OVERLAP_INTERFERENCE_DURATION_SECONDS,
    )
    parser.add_argument(
        "--non-overlap-interference-duration-seconds",
        type=positive_float,
        default=DEFAULT_NON_OVERLAP_INTERFERENCE_DURATION_SECONDS,
    )
    parser.add_argument(
        "--obstructive-snr-db", type=float, default=DEFAULT_OBSTRUCTIVE_SNR_DB
    )
    parser.add_argument(
        "--benign-overlap-snr-db",
        type=float,
        default=DEFAULT_BENIGN_OVERLAP_SNR_DB,
    )
    parser.add_argument(
        "--non-overlap-snr-db", type=float, default=DEFAULT_NON_OVERLAP_SNR_DB
    )
    parser.add_argument("--headroom", type=unit_interval, default=DEFAULT_HEADROOM)
    parser.add_argument("--fade-milliseconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--expected-fingerprint",
        type=lowercase_sha256,
        help=(
            "Reject the validated build before output replacement unless its "
            "artifact fingerprint matches this lowercase SHA256 digest."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing output directory without rebuilding it.",
    )
    args = parser.parse_args()
    if args.fade_milliseconds < 0.0:
        parser.error("--fade-milliseconds must be non-negative")
    if args.duration_seconds < 9.0:
        parser.error("--duration-seconds must be at least 9 seconds")
    if args.event_duration_seconds > 1.5:
        parser.error("--event-duration-seconds must be <= 1.5 seconds")
    minimum_overlap_duration = 2.0 * args.event_duration_seconds + 0.85 + 0.25
    if args.overlap_interference_duration_seconds < minimum_overlap_duration:
        parser.error(
            "overlap interference must cover both jittered evidence events; "
            f"need at least {minimum_overlap_duration:.2f} seconds"
        )
    if args.non_overlap_interference_duration_seconds > 2.0:
        parser.error("non-overlap interference duration must be <= 2 seconds")
    return args


def stable_seed(global_seed: int, *parts: object) -> int:
    payload = ":".join([str(global_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_source_key(source: SourceClip) -> Tuple[str, int]:
    suffix = source.source_id.rsplit("_", 1)[-1]
    return source.source_id.rsplit("_", 1)[0], int(suffix) if suffix.isdigit() else 0


def load_sources(
    audiotime_root: Path, require_audio: bool = True
) -> List[SourceClip]:
    metadata_path = audiotime_root / "timestamp_captions.json"
    audio_root = audiotime_root / "audio"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing AudioTime metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sources: List[SourceClip] = []
    for source_id, item in metadata.items():
        events = item.get("event", {})
        if not isinstance(events, dict) or len(events) != 1:
            continue
        label, intervals = next(iter(events.items()))
        if not isinstance(intervals, list) or len(intervals) != 1:
            continue
        interval = intervals[0]
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        start, end = float(interval[0]), float(interval[1])
        audio_path = audio_root / f"{source_id}.wav"
        label = str(label).strip()
        if (
            start < 0.0
            or end <= start
            or not label
            or (require_audio and not audio_path.exists())
        ):
            continue
        sources.append(
            SourceClip(
                source_id=str(source_id),
                label=label,
                interval_seconds=(start, end),
                caption=str(item.get("caption", "")),
                audio_path=audio_path,
            )
        )
    if not sources:
        raise RuntimeError("No eligible single-event AudioTime clips were found")
    return sorted(sources, key=natural_source_key)


def choose_sources(
    sources: Sequence[SourceClip],
    event_duration: float,
    overlap_duration: float,
    non_overlap_duration: float,
    seed: int,
) -> Tuple[FamilySources, FamilySources]:
    """Choose two shared interference labels with globally distinct scene sources."""

    grouped: Dict[str, List[SourceClip]] = {}
    for source in sources:
        grouped.setdefault(source.label, []).append(source)
    for label, label_sources in grouped.items():
        random.Random(stable_seed(seed, "source-order", label)).shuffle(label_sources)

    labels = sorted(grouped)
    random.Random(stable_seed(seed, "label-order")).shuffle(labels)
    interference_labels = [
        label
        for label in labels
        if sum(
            source.duration_seconds >= overlap_duration
            for source in grouped[label]
        )
        >= 4
    ][:2]
    if len(interference_labels) < 2:
        raise RuntimeError(
            "Need two interference labels with four distinct long clips each"
        )
    interference_clips_by_label = {
        label: tuple(
            source
            for source in grouped[label]
            if source.duration_seconds >= overlap_duration
        )[:4]
        for label in interference_labels
    }

    semantic_labels = [
        label
        for label in labels
        if label not in set(interference_labels)
        and sum(
            source.duration_seconds >= event_duration for source in grouped[label]
        )
        >= 4
    ][:4]
    if len(semantic_labels) < 4:
        raise RuntimeError("Need four semantic labels with four clips each")

    families: List[FamilySources] = []
    for family_index in range(2):
        family_semantic_labels = (
            semantic_labels[family_index * 2],
            semantic_labels[family_index * 2 + 1],
        )
        semantic_clips = tuple(
            tuple(
                source
                for source in grouped[label]
                if source.duration_seconds >= event_duration
            )[:4]
            for label in family_semantic_labels
        )
        source_offset = family_index * 2
        interference_clips = (
            interference_clips_by_label[interference_labels[0]][source_offset],
            interference_clips_by_label[interference_labels[1]][source_offset],
            interference_clips_by_label[interference_labels[0]][source_offset + 1],
            interference_clips_by_label[interference_labels[1]][source_offset + 1],
        )
        if interference_clips[3].duration_seconds < non_overlap_duration:
            raise AssertionError("Selected non-overlap interference clip is too short")
        families.append(
            FamilySources(
                semantic_labels=family_semantic_labels,
                semantic_clips=(semantic_clips[0], semantic_clips[1]),
                interference_clips=interference_clips,
            )
        )
    return families[0], families[1]


def read_mono_audio(path: Path) -> Tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.shape[1] > 1:
        waveform = waveform.mean(axis=1)
    else:
        waveform = waveform[:, 0]
    if not np.all(np.isfinite(waveform)):
        raise ValueError(f"Non-finite waveform: {path}")
    return waveform, int(sample_rate)


def resample_audio(
    waveform: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(
        waveform, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)


def rms(waveform: np.ndarray) -> float:
    if waveform.size == 0:
        raise ValueError("Cannot measure RMS of an empty waveform")
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def maximum_variance_crop_offset(
    waveform: np.ndarray,
    source_start: int,
    source_end: int,
    required_source_samples: int,
) -> int:
    """Return the most energetic valid window offset without label input."""

    if (
        waveform.ndim != 1
        or source_start < 0
        or source_end > waveform.size
        or source_end <= source_start
        or required_source_samples <= 0
        or source_end - source_start < required_source_samples
    ):
        raise ValueError("invalid maximum-variance crop bounds")
    interval = waveform[source_start:source_end].astype(np.float64, copy=False)
    cumulative = np.concatenate(([0.0], np.cumsum(interval)))
    cumulative_square = np.concatenate(([0.0], np.cumsum(np.square(interval))))
    window_sum = (
        cumulative[required_source_samples:] - cumulative[:-required_source_samples]
    )
    window_square_sum = (
        cumulative_square[required_source_samples:]
        - cumulative_square[:-required_source_samples]
    )
    window_variance = (
        window_square_sum / required_source_samples
        - np.square(window_sum / required_source_samples)
    )
    return int(np.argmax(np.maximum(window_variance, 0.0)))


def crop_source_event(
    source: SourceClip,
    duration_seconds: float,
    sample_rate: int,
    fade_milliseconds: float,
    rng: np.random.Generator,
    *,
    recover_silent_random_crop: bool = False,
    prefer_maximum_variance_crop: bool = False,
    preferred_center_seconds: float | None = None,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    waveform, source_rate = read_mono_audio(source.audio_path)
    source_start = max(0, int(round(source.interval_seconds[0] * source_rate)))
    source_end = min(waveform.size, int(round(source.interval_seconds[1] * source_rate)))
    required_source_samples = int(math.ceil(duration_seconds * source_rate))
    available = source_end - source_start
    if available < required_source_samples:
        raise ValueError(
            f"Source {source.source_id} is too short after sample rounding: "
            f"{available} < {required_source_samples}"
        )
    max_offset = available - required_source_samples
    if preferred_center_seconds is not None:
        if not math.isfinite(preferred_center_seconds):
            raise ValueError("preferred_center_seconds must be finite")
        preferred_center = int(round(preferred_center_seconds * source_rate))
        preferred_start = preferred_center - required_source_samples // 2
        crop_offset = min(max_offset, max(0, preferred_start - source_start))
    elif prefer_maximum_variance_crop:
        crop_offset = maximum_variance_crop_offset(
            waveform, source_start, source_end, required_source_samples
        )
    else:
        crop_offset = int(rng.integers(0, max_offset + 1)) if max_offset else 0
    crop_start = source_start + crop_offset

    def render(candidate_start: int) -> np.ndarray:
        candidate_end = candidate_start + required_source_samples
        candidate = resample_audio(
            waveform[candidate_start:candidate_end], source_rate, sample_rate
        )
        target_samples = int(round(duration_seconds * sample_rate))
        if candidate.size < target_samples:
            candidate = np.pad(candidate, (0, target_samples - candidate.size))
        else:
            candidate = candidate[:target_samples]
        candidate = candidate - float(np.mean(candidate))
        fade_samples = min(
            int(round(fade_milliseconds * sample_rate / 1000.0)),
            candidate.size // 2,
        )
        if fade_samples:
            phase = np.linspace(
                0.0, math.pi / 2.0, fade_samples, dtype=np.float32
            )
            ramp = np.sin(phase) ** 2
            candidate[:fade_samples] *= ramp
            candidate[-fade_samples:] *= ramp[::-1]
        return candidate

    clip = render(crop_start)
    clip_rms = rms(clip)
    if clip_rms < 1e-5 and recover_silent_random_crop and max_offset > 0:
        # FUSS source clips can contain a short labelled event surrounded by
        # long digital silence. Preserve the seeded random crop whenever it is
        # usable; only a would-be hard failure falls back to the maximum-
        # variance window. Prefix sums make the exhaustive search deterministic
        # without decoding or resampling every candidate window.
        best_offset = maximum_variance_crop_offset(
            waveform, source_start, source_end, required_source_samples
        )
        crop_start = source_start + best_offset
        clip = render(crop_start)
        clip_rms = rms(clip)
    if clip_rms < 1e-5:
        raise ValueError(f"Source {source.source_id} is effectively silent")
    clip = (clip * (0.08 / clip_rms)).astype(np.float32)
    crop_start_seconds = crop_start / source_rate
    return clip, (crop_start_seconds, crop_start_seconds + duration_seconds)


def render_clip(
    canvas: np.ndarray, clip: np.ndarray, onset_sample: int, sample_rate: int
) -> Tuple[float, float]:
    offset_sample = onset_sample + clip.size
    if onset_sample < 0 or offset_sample > canvas.size:
        raise ValueError("Rendered clip exceeds output duration")
    canvas[onset_sample:offset_sample] += clip
    return onset_sample / sample_rate, offset_sample / sample_rate


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate, subtype="PCM_16")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_question(
    question_type: str,
    anchor_label: str,
    candidate_labels: Tuple[str, str],
) -> str:
    if question_type == "temporal_before":
        return f"What sound occurs immediately before {anchor_label}?"
    if question_type == "temporal_after":
        return f"What sound occurs immediately after {anchor_label}?"
    if question_type == "temporal_first":
        return (
            f"Which sound occurs first, {candidate_labels[0]} "
            f"or {candidate_labels[1]}?"
        )
    if question_type == "temporal_last":
        return (
            f"Which sound occurs last, {candidate_labels[0]} "
            f"or {candidate_labels[1]}?"
        )
    raise ValueError(f"Unsupported question type: {question_type}")


def event_payload(
    event_id: str,
    source: SourceClip,
    source_sha256: str,
    project_root: Path,
    crop_interval: Tuple[float, float],
    rendered_interval: Tuple[float, float],
    role: str,
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "label": source.label,
        "source_dataset": "AudioTime",
        "source_id": source.source_id,
        "source_path": relative_posix(source.audio_path, project_root),
        "source_sha256": source_sha256,
        "source_interval_seconds": list(source.interval_seconds),
        "source_crop_interval_seconds": list(crop_interval),
        "onset_seconds": rendered_interval[0],
        "offset_seconds": rendered_interval[1],
        "role": role,
    }


def write_timeline_svg(path: Path, record: Mapping[str, Any]) -> None:
    width = 1080
    plot_left = 130
    plot_width = 900
    row_height = 42
    duration = float(record["duration_seconds"])
    colors = {
        "anchor": "#2563eb",
        "answer": "#16a34a",
        "interference": "#dc2626" if record["edit_needed"] else "#f59e0b",
    }
    rows = []
    for index, event in enumerate(record["events"]):
        y = 105 + index * row_height
        x = plot_left + float(event["onset_seconds"]) / duration * plot_width
        event_width = (
            (float(event["offset_seconds"]) - float(event["onset_seconds"]))
            / duration
            * plot_width
        )
        role = str(event["role"])
        rows.append(
            f'<text x="10" y="{y + 20}" font-size="13">{html.escape(role)}</text>'
            f'<rect x="{x:.2f}" y="{y}" width="{event_width:.2f}" '
            f'height="26" rx="4" fill="{colors[role]}" opacity="0.85"/>'
            f'<text x="{x + 5:.2f}" y="{y + 18}" font-size="11" fill="white">'
            f'{html.escape(str(event["label"]))}</text>'
        )
    ticks = []
    for second in range(int(duration) + 1):
        x = plot_left + second / duration * plot_width
        ticks.append(
            f'<line x1="{x:.2f}" y1="90" x2="{x:.2f}" y2="245" '
            f'stroke="#d1d5db" stroke-width="1"/>'
            f'<text x="{x - 4:.2f}" y="264" font-size="10">{second}</text>'
        )
    subtitle = (
        f"scene={record['scene_id']} | rationale={record['edit_rationale']} | "
        f"selector={record['selector_target']} | answer={record['answer']}"
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="285">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="10" y="24" font-size="15" font-weight="bold">'
        f'{html.escape(str(record["id"]))}</text>'
        f'<text x="10" y="48" font-size="13">'
        f'{html.escape(str(record["question"]))}</text>'
        f'<text x="10" y="70" font-size="12">{html.escape(subtitle)}</text>'
        + "".join(ticks)
        + "".join(rows)
        + '<text x="1038" y="264" font-size="10">s</text></svg>\n'
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def make_scene_seed(
    args: argparse.Namespace,
    metadata_sha256: str,
    scene_index: int,
    scene_sources: Sequence[SourceClip],
    source_hashes: Mapping[str, str],
) -> int:
    identity_parts: List[object] = [
        "scene",
        scene_index,
        metadata_sha256,
        args.sample_rate,
        args.duration_seconds,
        args.event_duration_seconds,
        args.overlap_interference_duration_seconds,
        args.non_overlap_interference_duration_seconds,
        args.obstructive_snr_db,
        args.benign_overlap_snr_db,
        args.non_overlap_snr_db,
        args.fade_milliseconds,
        args.headroom,
    ]
    for source in scene_sources:
        identity_parts.extend((source.source_id, source_hashes[source.source_id]))
    return stable_seed(args.seed, *identity_parts)


def compose_scene(
    scene_index: int,
    family_index: int,
    scene_slot: int,
    family: FamilySources,
    args: argparse.Namespace,
    metadata_sha256: str,
    source_hashes: Mapping[str, str],
    project_root: Path,
    staging_root: Path,
) -> List[Dict[str, Any]]:
    ordinal = family_index == 1
    question_pair = (
        ("temporal_first", "temporal_last")
        if ordinal
        else ("temporal_before", "temporal_after")
    )
    semantic_sources = (
        family.semantic_clips[0][scene_slot],
        family.semantic_clips[1][scene_slot],
    )
    interference_source = family.interference_clips[scene_slot]
    if scene_slot == 0:
        edit_needed = True
        edit_rationale = "overlap_obstructive_proxy"
        requested_snr = args.obstructive_snr_db
    elif scene_slot == 1:
        edit_needed = True
        edit_rationale = "overlap_obstructive_proxy"
        requested_snr = args.obstructive_snr_db
    elif scene_slot == 2:
        edit_needed = False
        edit_rationale = "overlap_benign_proxy"
        requested_snr = args.benign_overlap_snr_db
    else:
        edit_needed = False
        edit_rationale = "non_overlap"
        requested_snr = args.non_overlap_snr_db

    scene_sources = semantic_sources + (interference_source,)
    scene_seed = make_scene_seed(
        args,
        metadata_sha256,
        scene_index,
        scene_sources,
        source_hashes,
    )
    layout_rng = np.random.default_rng(stable_seed(scene_seed, "layout"))
    target_samples = int(round(args.sample_rate * args.duration_seconds))
    event_samples = int(round(args.sample_rate * args.event_duration_seconds))
    early_onset_sample = int(
        layout_rng.integers(
            int(round(0.65 * args.sample_rate)),
            int(round(1.65 * args.sample_rate)) + 1,
        )
    )
    gap_samples = int(
        layout_rng.integers(
            int(round(0.35 * args.sample_rate)),
            int(round(0.85 * args.sample_rate)) + 1,
        )
    )
    late_onset_sample = early_onset_sample + event_samples + gap_samples
    semantic_reversed = ordinal and scene_slot in {1, 3}
    early_source = semantic_sources[1] if semantic_reversed else semantic_sources[0]
    late_source = semantic_sources[0] if semantic_reversed else semantic_sources[1]

    if edit_rationale == "non_overlap":
        separation_samples = int(
            layout_rng.integers(
                int(round(0.45 * args.sample_rate)),
                int(round(0.90 * args.sample_rate)) + 1,
            )
        )
        interference_onset_sample = (
            late_onset_sample + event_samples + separation_samples
        )
        interference_duration = args.non_overlap_interference_duration_seconds
        snr_measurement = NON_OVERLAP_SNR_MEASUREMENT
    else:
        lead_samples = int(
            layout_rng.integers(
                int(round(0.10 * args.sample_rate)),
                int(round(0.25 * args.sample_rate)) + 1,
            )
        )
        interference_onset_sample = early_onset_sample - lead_samples
        interference_duration = args.overlap_interference_duration_seconds
        snr_measurement = OVERLAP_SNR_MEASUREMENT

    crop_rngs = {
        source.source_id: np.random.default_rng(
            stable_seed(scene_seed, "crop", source.source_id)
        )
        for source in scene_sources
    }
    early_clip, early_crop = crop_source_event(
        early_source,
        args.event_duration_seconds,
        args.sample_rate,
        args.fade_milliseconds,
        crop_rngs[early_source.source_id],
    )
    late_clip, late_crop = crop_source_event(
        late_source,
        args.event_duration_seconds,
        args.sample_rate,
        args.fade_milliseconds,
        crop_rngs[late_source.source_id],
    )
    interference_clip, interference_crop = crop_source_event(
        interference_source,
        interference_duration,
        args.sample_rate,
        args.fade_milliseconds,
        crop_rngs[interference_source.source_id],
    )

    clean = np.zeros(target_samples, dtype=np.float32)
    early_interval = render_clip(
        clean, early_clip, early_onset_sample, args.sample_rate
    )
    late_interval = render_clip(clean, late_clip, late_onset_sample, args.sample_rate)
    aligned_interference = np.zeros(target_samples, dtype=np.float32)
    interference_interval = render_clip(
        aligned_interference,
        interference_clip,
        interference_onset_sample,
        args.sample_rate,
    )

    preserve_mask = np.zeros(target_samples, dtype=bool)
    preserve_mask[early_onset_sample : early_onset_sample + early_clip.size] = True
    preserve_mask[late_onset_sample : late_onset_sample + late_clip.size] = True
    interference_mask = np.zeros(target_samples, dtype=bool)
    interference_mask[
        interference_onset_sample : interference_onset_sample + interference_clip.size
    ] = True
    if edit_rationale == "non_overlap":
        clean_reference = clean[preserve_mask]
        interference_reference = aligned_interference[interference_mask]
    else:
        measurement_mask = preserve_mask & interference_mask
        if not np.any(measurement_mask):
            raise AssertionError("Overlap scene has no SNR measurement intersection")
        clean_reference = clean[measurement_mask]
        interference_reference = aligned_interference[measurement_mask]
    interference_gain = rms(clean_reference) / (
        max(rms(interference_reference), EPSILON)
        * (10.0 ** (requested_snr / 20.0))
    )
    aligned_interference *= interference_gain
    mixture = clean + aligned_interference
    peak = max(
        float(np.max(np.abs(clean))),
        float(np.max(np.abs(aligned_interference))),
        float(np.max(np.abs(mixture))),
    )
    global_gain = min(1.0, args.headroom / peak) if peak > 0.0 else 1.0
    clean = (clean * global_gain).astype(np.float32)
    aligned_interference = (aligned_interference * global_gain).astype(np.float32)
    mixture = (clean + aligned_interference).astype(np.float32)
    if edit_rationale == "non_overlap":
        realized_clean = clean[preserve_mask]
        realized_interference = aligned_interference[interference_mask]
    else:
        measurement_mask = preserve_mask & interference_mask
        realized_clean = clean[measurement_mask]
        realized_interference = aligned_interference[measurement_mask]
    realized_snr = 20.0 * math.log10(
        (rms(realized_clean) + EPSILON)
        / (rms(realized_interference) + EPSILON)
    )

    scene_id = f"scene_{scene_index:06d}"
    scene_filename = f"{scene_id}.wav"
    mixture_path = staging_root / "audio" / "mixture" / "train" / scene_filename
    clean_path = staging_root / "audio" / "clean" / "train" / scene_filename
    interference_path = (
        staging_root / "audio" / "interference" / "train" / scene_filename
    )
    write_wav(mixture_path, mixture, args.sample_rate)
    write_wav(clean_path, clean, args.sample_rate)
    write_wav(interference_path, aligned_interference, args.sample_rate)

    physical_events = {
        "event_early": (early_source, early_crop, early_interval),
        "event_late": (late_source, late_crop, late_interval),
        "event_interference": (
            interference_source,
            interference_crop,
            interference_interval,
        ),
    }
    candidate_labels = tuple(sorted(family.semantic_labels))

    records: List[Dict[str, Any]] = []
    for pair_index, question_type in enumerate(question_pair):
        early_is_answer = pair_index == 0
        roles = {
            "event_early": "answer" if early_is_answer else "anchor",
            "event_late": "anchor" if early_is_answer else "answer",
            "event_interference": "interference",
        }
        events = []
        for event_id in ("event_early", "event_late", "event_interference"):
            source, crop_interval, rendered_interval = physical_events[event_id]
            events.append(
                event_payload(
                    event_id,
                    source,
                    source_hashes[source.source_id],
                    project_root,
                    crop_interval,
                    rendered_interval,
                    roles[event_id],
                )
            )
        events.sort(key=lambda event: (float(event["onset_seconds"]), str(event["event_id"])))
        answer_event = next(event for event in events if event["role"] == "answer")
        anchor_event = next(event for event in events if event["role"] == "anchor")
        sample_id = f"train_{scene_index * 2 + pair_index:06d}"
        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "id": sample_id,
            "scene_id": scene_id,
            "split": "train",
            "sample_rate": args.sample_rate,
            "num_channels": 1,
            "num_samples": target_samples,
            "duration_seconds": args.duration_seconds,
            "mixture_path": relative_posix(mixture_path, staging_root),
            "clean_path": relative_posix(clean_path, staging_root),
            "interference_stem_path": relative_posix(interference_path, staging_root),
            "question": build_question(
                question_type, str(anchor_event["label"]), candidate_labels
            ),
            "answer": str(answer_event["label"]),
            "question_type": question_type,
            "events": events,
            "anchor_event_ids": [str(anchor_event["event_id"])],
            "answer_event_ids": [str(answer_event["event_id"])],
            "anchor_intervals": [
                [anchor_event["onset_seconds"], anchor_event["offset_seconds"]]
            ],
            "answer_intervals": [
                [answer_event["onset_seconds"], answer_event["offset_seconds"]]
            ],
            "interference_event_ids": ["event_interference"],
            "interference_intervals": [list(interference_interval)],
            "event_presence_labels": sorted(
                [early_source.label, late_source.label, interference_source.label]
            ),
            "edit_needed": edit_needed,
            "edit_rationale": edit_rationale,
            "selector_target": (
                interference_source.label if edit_needed else NO_EDIT_TARGET
            ),
            "snr_measurement": snr_measurement,
            "snr_db_requested": requested_snr,
            "snr_db": realized_snr,
            "mixture_peak": float(np.max(np.abs(mixture))),
            "source_group_ids": sorted(
                [source.source_id for source in scene_sources]
            ),
            "generation_seed": scene_seed,
        }
        parse_record(record)
        write_timeline_svg(staging_root / "timelines" / f"{sample_id}.svg", record)
        records.append(record)
    return records


def replace_validated_output(staging_root: Path, output_root: Path) -> None:
    """Replace output only after staging validation, restoring it on swap failure."""

    if not output_root.exists():
        os.replace(staging_root, output_root)
        return
    backup_root = output_root.with_name(output_root.name + ".previous")
    if backup_root.exists():
        shutil.rmtree(backup_root)
    os.replace(output_root, backup_root)
    try:
        os.replace(staging_root, output_root)
    except Exception:
        os.replace(backup_root, output_root)
        raise
    shutil.rmtree(backup_root)


def build_dataset(args: argparse.Namespace) -> None:
    """Compose eight paired scenes, stems, manifest, and timeline artifacts."""

    project_root = args.project_root.resolve()
    audiotime_root = args.audiotime_root.resolve()
    output_root = args.output_root.resolve()
    staging_root = output_root.with_name(output_root.name + ".building")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}. Use --overwrite to rebuild."
        )
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    try:
        metadata_path = audiotime_root / "timestamp_captions.json"
        metadata_sha256 = sha256_file(metadata_path)
        # Select from metadata rather than from the accidental set of local WAVs.
        # A 24-file subset therefore reproduces a full AudioTime checkout.
        sources = load_sources(audiotime_root, require_audio=False)
        families = choose_sources(
            sources,
            args.event_duration_seconds,
            args.overlap_interference_duration_seconds,
            args.non_overlap_interference_duration_seconds,
            args.seed,
        )
        selected_sources: Dict[str, SourceClip] = {}
        for family in families:
            for clips in family.semantic_clips:
                for source in clips:
                    selected_sources[source.source_id] = source
            for source in family.interference_clips:
                selected_sources[source.source_id] = source
        missing_audio = [
            source.audio_path
            for source in selected_sources.values()
            if not source.audio_path.exists()
        ]
        if missing_audio:
            preview = ", ".join(path.name for path in sorted(missing_audio)[:8])
            raise FileNotFoundError(
                f"Missing {len(missing_audio)} deterministically selected AudioTime "
                f"WAV files ({preview}). Run download_audiotime_subset.py first."
            )
        source_hashes = {
            source_id: sha256_file(source.audio_path)
            for source_id, source in sorted(selected_sources.items())
        }

        records: List[Dict[str, Any]] = []
        for family_index, family in enumerate(families):
            for scene_slot in range(4):
                scene_index = family_index * 4 + scene_slot
                scene_records = compose_scene(
                    scene_index,
                    family_index,
                    scene_slot,
                    family,
                    args,
                    metadata_sha256,
                    source_hashes,
                    project_root,
                    staging_root,
                )
                records.extend(scene_records)
                print(
                    f"Built {scene_records[0]['scene_id']}: "
                    f"{scene_records[0]['question_type']}+{scene_records[1]['question_type']}, "
                    f"rationale={scene_records[0]['edit_rationale']}"
                )

        write_jsonl(staging_root / "qa_overfit16.jsonl", records)
        source_code_paths = {
            "builder": Path(__file__).resolve(),
            "schema": CODE_ROOT / "mixi_understanding" / "data" / "qa_schema.py",
            "validator": (
                CODE_ROOT
                / "mixi_understanding"
                / "scripts"
                / "validate_qa_dataset.py"
            ),
        }
        runtime_identity = {
            "python_version": platform.python_version(),
            "numpy_version": str(np.__version__),
            "scipy_version": str(scipy.__version__),
            "soundfile_version": str(sf.__version__),
        }
        build_identity = {
            "source_files": {
                role: {
                    "path": relative_posix(path, project_root),
                    "sha256": sha256_file(path),
                }
                for role, path in sorted(source_code_paths.items())
            }
        }
        acquisition_receipt = audiotime_root / "subset_receipt.json"
        source_config = {
            "dataset": "AudioTime",
            "source_split": "train5000_timestamp",
            "metadata_path": relative_posix(metadata_path, project_root),
            "metadata_sha256": metadata_sha256,
            "selection": "single event with one timestamp interval",
            "license": "See upstream AudioTime terms",
        }
        if acquisition_receipt.exists():
            source_config["acquisition_receipt_path"] = relative_posix(
                acquisition_receipt, project_root
            )
            source_config["acquisition_receipt_sha256"] = sha256_file(
                acquisition_receipt
            )
        config = {
            "schema_version": SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "runtime_identity": runtime_identity,
            "build_identity": build_identity,
            "seed": args.seed,
            "sample_rate": args.sample_rate,
            "num_channels": 1,
            "num_samples": int(round(args.sample_rate * args.duration_seconds)),
            "duration_seconds": args.duration_seconds,
            "audio_format": {
                "container": "WAV",
                "subtype": "PCM_16",
                "channels": 1,
            },
            "source": source_config,
            "splits": {"qa_overfit16": 16},
            "scenes": {
                "count": 8,
                "records_per_scene": 2,
                "shared_audio_with_complementary_roles": True,
            },
            "quota": {
                "question_types": list(QUESTION_TYPES),
                "per_question_type": 4,
                "per_question_type_and_edit_state": 2,
                "exact_question_count": 4,
                "per_exact_question": 4,
                "per_exact_question_and_edit_state": 2,
                "distinct_edit_selector_targets_per_exact_question": 2,
                "interference_label_count": 2,
                "sources_per_interference_label": 4,
                "edit_needed": 8,
                "no_edit": 8,
                "edit_rationale_records": {
                    "overlap_obstructive_proxy": 8,
                    "overlap_benign_proxy": 4,
                    "non_overlap": 4,
                },
            },
            "composition": {
                "event_duration_seconds": args.event_duration_seconds,
                "early_onset_range_seconds": [0.65, 1.65],
                "inter_event_gap_range_seconds": [0.35, 0.85],
                "non_overlap_separation_range_seconds": [0.45, 0.90],
                "overlap_interference_lead_range_seconds": [0.10, 0.25],
                "overlap_interference_duration_seconds": (
                    args.overlap_interference_duration_seconds
                ),
                "non_overlap_interference_duration_seconds": (
                    args.non_overlap_interference_duration_seconds
                ),
                "rationale_snr_db": {
                    "overlap_obstructive_proxy": args.obstructive_snr_db,
                    "overlap_benign_proxy": args.benign_overlap_snr_db,
                    "non_overlap": args.non_overlap_snr_db,
                },
                "fade_milliseconds": args.fade_milliseconds,
                "headroom": args.headroom,
            },
            "snr_measurement": {
                "overlap": OVERLAP_SNR_MEASUREMENT,
                "non_overlap": NON_OVERLAP_SNR_MEASUREMENT,
                "requested_realized_tolerance_db": 0.15,
                "definitions": {
                    OVERLAP_SNR_MEASUREMENT: (
                        "clean versus interference RMS on preserve-mask and "
                        "interference-mask intersection"
                    ),
                    NON_OVERLAP_SNR_MEASUREMENT: (
                        "clean RMS on evidence samples versus interference RMS "
                        "on active interference samples"
                    ),
                },
            },
            "edit_labeling": {
                "synthetic_proxy": True,
                "notice": (
                    "Rationales are controlled synthetic SNR/overlap proxies; "
                    "they are not measurements of causal QA degradation."
                ),
                "selector_rule": (
                    "interference label only for edit_needed; otherwise no_edit"
                ),
            },
            "question_control": {
                "relative_semantic_order_rule": RELATIVE_SEMANTIC_ORDER_RULE,
                "ordinal_semantic_order_rule": ORDINAL_SEMANTIC_ORDER_RULE,
                "ordinal_candidate_order_rule": ORDINAL_CANDIDATE_ORDER_RULE,
            },
            "artifact_fingerprint": {
                "algorithm": "sha256(relative_path NUL file_sha256 LF)",
                "includes": [
                    "dataset_config.json",
                    "qa_overfit16.jsonl",
                    "audio/**/*.wav",
                    "timelines/*.svg",
                ],
                "excludes": ["validation_report.json"],
            },
            "answer_usage": {
                "model_input_fields": ["mixture_path", "question"],
                "supervision_only_fields": ["answer"],
            },
        }
        (staging_root / "dataset_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = validate_dataset(staging_root, write_report=True)
        actual_fingerprint = str(report["artifact_fingerprint_sha256"])
        if (
            args.expected_fingerprint is not None
            and actual_fingerprint != args.expected_fingerprint
        ):
            raise RuntimeError(
                "Validated artifact fingerprint mismatch: "
                f"expected {args.expected_fingerprint}, got {actual_fingerprint}"
            )
        replace_validated_output(staging_root, output_root)
        print(f"Dataset ready: {output_root}")
        print(f"Artifact fingerprint: {actual_fingerprint}")
    except Exception:
        print(f"Build failed; staging data left at {staging_root}", file=sys.stderr)
        raise


def main() -> None:
    args = parse_args()
    if args.validate_only:
        report = validate_dataset(args.output_root, write_report=True)
        actual_fingerprint = str(report["artifact_fingerprint_sha256"])
        if (
            args.expected_fingerprint is not None
            and actual_fingerprint != args.expected_fingerprint
        ):
            raise RuntimeError(
                "Validated artifact fingerprint mismatch: "
                f"expected {args.expected_fingerprint}, got {actual_fingerprint}"
            )
    else:
        build_dataset(args)


if __name__ == "__main__":
    main()
