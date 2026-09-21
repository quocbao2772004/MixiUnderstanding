#!/usr/bin/env python3
"""Build a reproducible synthetic REMOVE dataset from AudioCaps and AudioTime."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly


BUILDER_VERSION = "1.0.0"
AUDIOCAPS_REPO = "OpenSound/AudioCaps"
AUDIOCAPS_REVISION = "b29b3243d6ce49c2cd0d48d4b5f0701ae7969ded"
AUDIOCAPS_LICENSE = "CC-BY-NC-4.0"
AUDIOCAPS_SHARD_COUNT = 412
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_DURATION_SECONDS = 10.0
DEFAULT_HEADROOM = 0.95
EPSILON = 1e-8


@dataclass(frozen=True)
class CleanSource:
    audiocap_id: int
    youtube_id: str
    start_time: int
    caption: str
    audio_bytes: bytes
    audio_path: Optional[str]
    shard_name: str

    @property
    def source_id(self) -> str:
        return f"{self.youtube_id}_{self.start_time}"


@dataclass(frozen=True)
class InterferenceSource:
    source_id: str
    event: str
    interval_seconds: Tuple[float, float]
    caption: str
    audio_path: Path


@dataclass(frozen=True)
class PreparedAudio:
    waveform: np.ndarray
    source_sample_rate: int
    crop_start_seconds: float


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description=(
            "Create paired mixture/target audio for the instruction "
            "'Remove <event>' using AudioCaps clean scenes and single-event "
            "AudioTime timestamp clips."
        )
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--audiotime-root",
        type=Path,
        default=project_root / "AudioTime-recovered" / "train5000_timestamp",
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=project_root / "data" / "raw" / "audiocaps",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "removal_synthetic_v1",
    )
    parser.add_argument("--train-size", type=non_negative_int, default=64)
    parser.add_argument("--val-size", type=non_negative_int, default=16)
    parser.add_argument("--test-size", type=non_negative_int, default=16)
    parser.add_argument("--overfit-size", type=non_negative_int, default=16)
    parser.add_argument("--audiocaps-shards", type=positive_int, default=1)
    parser.add_argument("--sample-rate", type=positive_int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--duration-seconds", type=positive_float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--snr-min-db", type=float, default=-5.0)
    parser.add_argument("--snr-max-db", type=float, default=10.0)
    parser.add_argument("--headroom", type=unit_interval, default=DEFAULT_HEADROOM)
    parser.add_argument("--fade-milliseconds", type=non_negative_float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing output directory without rebuilding it.",
    )
    args = parser.parse_args()
    if args.snr_min_db > args.snr_max_db:
        parser.error("--snr-min-db must be <= --snr-max-db")
    if args.overfit_size > args.train_size:
        parser.error("--overfit-size cannot exceed --train-size")
    if args.audiocaps_shards > AUDIOCAPS_SHARD_COUNT:
        parser.error(
            f"--audiocaps-shards cannot exceed {AUDIOCAPS_SHARD_COUNT}"
        )
    return args


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def unit_interval(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0, 1]")
    return parsed


def stable_seed(global_seed: int, split: str, index: int) -> int:
    payload = f"{global_seed}:{split}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, retries: int = 8) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            pq.read_metadata(destination)
            print(f"Using cached shard: {destination}")
            return
        except Exception:
            destination.unlink()

    partial = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, retries + 1):
        existing_size = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "MixiUnderstanding-dataset-builder/1.0"}
        if existing_size:
            headers["Range"] = f"bytes={existing_size}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            print(
                f"Downloading {destination.name} "
                f"(attempt {attempt}/{retries}, resume={existing_size} bytes)"
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", 200)
                append = existing_size > 0 and status == 206
                mode = "ab" if append else "wb"
                bytes_received = 0
                content_length = response.headers.get("Content-Length")
                expected = int(content_length) if content_length else None
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                        bytes_received += len(chunk)
                if expected is not None and bytes_received != expected:
                    raise IOError(
                        f"Incomplete response: expected {expected}, got {bytes_received} bytes"
                    )
            os.replace(partial, destination)
            pq.read_metadata(destination)
            print(f"Downloaded: {destination}")
            return
        except Exception as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))


def audiocaps_shard_name(index: int) -> str:
    return f"train-{index:05d}-of-{AUDIOCAPS_SHARD_COUNT:05d}.parquet"


def ensure_audiocaps_shards(cache_root: Path, count: int) -> List[Path]:
    paths: List[Path] = []
    for index in range(count):
        name = audiocaps_shard_name(index)
        destination = cache_root / name
        url = (
            f"https://huggingface.co/datasets/{AUDIOCAPS_REPO}/resolve/"
            f"{AUDIOCAPS_REVISION}/data/{name}?download=true"
        )
        download_file(url, destination)
        paths.append(destination)
    return paths


def load_clean_sources(shard_paths: Sequence[Path]) -> List[CleanSource]:
    sources: List[CleanSource] = []
    seen_source_ids = set()
    required_columns = [
        "audiocap_id",
        "youtube_id",
        "start_time",
        "caption",
        "audio",
    ]
    for shard_path in shard_paths:
        table = pq.read_table(shard_path, columns=required_columns)
        for row in table.to_pylist():
            audio = row.get("audio") or {}
            audio_bytes = audio.get("bytes") if isinstance(audio, dict) else None
            if not audio_bytes:
                continue
            source = CleanSource(
                audiocap_id=int(row["audiocap_id"]),
                youtube_id=str(row["youtube_id"]),
                start_time=int(row["start_time"]),
                caption=str(row["caption"]),
                audio_bytes=audio_bytes,
                audio_path=audio.get("path") if isinstance(audio, dict) else None,
                shard_name=shard_path.name,
            )
            if source.source_id in seen_source_ids:
                continue
            seen_source_ids.add(source.source_id)
            sources.append(source)
    if not sources:
        raise RuntimeError("No decodable AudioCaps rows were found in the selected shards")
    return sources


def load_interference_sources(audiotime_root: Path) -> List[InterferenceSource]:
    metadata_path = audiotime_root / "timestamp_captions.json"
    audio_root = audiotime_root / "audio"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing AudioTime metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sources: List[InterferenceSource] = []
    for source_id, item in metadata.items():
        events = item.get("event", {})
        if len(events) != 1:
            continue
        event, intervals = next(iter(events.items()))
        if len(intervals) != 1:
            continue
        interval = intervals[0]
        if not isinstance(interval, list) or len(interval) != 2:
            continue
        start_seconds, end_seconds = float(interval[0]), float(interval[1])
        if end_seconds - start_seconds < 0.25:
            continue
        audio_path = audio_root / f"{source_id}.wav"
        if not audio_path.exists():
            continue
        sources.append(
            InterferenceSource(
                source_id=source_id,
                event=str(event),
                interval_seconds=(start_seconds, end_seconds),
                caption=str(item.get("caption", "")),
                audio_path=audio_path,
            )
        )
    if not sources:
        raise RuntimeError("No single-event AudioTime timestamp clips were found")
    return sources


def read_audio(source: Any) -> Tuple[np.ndarray, int]:
    waveform, sample_rate = sf.read(
        source, dtype="float32", always_2d=True
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    if waveform.shape[1] > 1:
        waveform = waveform.mean(axis=1)
    else:
        waveform = waveform[:, 0]
    if not np.all(np.isfinite(waveform)):
        raise ValueError("Waveform contains NaN or infinite values")
    return waveform, int(sample_rate)


def resample_audio(
    waveform: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    return resample_poly(waveform, up, down).astype(np.float32)


def fit_clean_audio(
    waveform: np.ndarray,
    target_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    if waveform.size >= target_samples:
        max_start = waveform.size - target_samples
        start = int(rng.integers(0, max_start + 1)) if max_start else 0
        return waveform[start : start + target_samples].copy(), start / sample_rate
    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: waveform.size] = waveform
    return padded, 0.0


def prepare_clean_source(
    source: CleanSource,
    target_rate: int,
    target_samples: int,
    rng: np.random.Generator,
) -> PreparedAudio:
    waveform, source_rate = read_audio(io.BytesIO(source.audio_bytes))
    waveform = resample_audio(waveform, source_rate, target_rate)
    waveform, crop_start = fit_clean_audio(
        waveform, target_samples, target_rate, rng
    )
    if rms(waveform) < 1e-5:
        raise ValueError(f"Clean source {source.source_id} is effectively silent")
    return PreparedAudio(waveform, source_rate, crop_start)


def prepare_interference_source(
    source: InterferenceSource,
    target_rate: int,
    target_samples: int,
    fade_milliseconds: float,
) -> Tuple[np.ndarray, int]:
    waveform, source_rate = read_audio(source.audio_path)
    start = max(0, int(round(source.interval_seconds[0] * source_rate)))
    end = min(waveform.size, int(round(source.interval_seconds[1] * source_rate)))
    if end <= start:
        raise ValueError(f"Invalid event interval for {source.source_id}")
    stem = waveform[start:end]
    stem = resample_audio(stem, source_rate, target_rate)
    if stem.size > target_samples:
        stem = stem[:target_samples]
    fade_samples = min(
        int(round(fade_milliseconds * target_rate / 1000.0)), stem.size // 2
    )
    if fade_samples:
        phase = np.linspace(0.0, math.pi / 2.0, fade_samples, dtype=np.float32)
        ramp = np.sin(phase) ** 2
        stem = stem.copy()
        stem[:fade_samples] *= ramp
        stem[-fade_samples:] *= ramp[::-1]
    if rms(stem) < 1e-5:
        raise ValueError(f"Interference source {source.source_id} is effectively silent")
    return stem.astype(np.float32, copy=False), source_rate


def rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)) + EPSILON))


def normalized_words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def event_mentioned_in_caption(event: str, caption: str) -> bool:
    caption_words = normalized_words(caption)
    caption_tokens = set(caption_words)
    variants = [event]
    variants.extend(re.findall(r"\(([^)]+)\)", event))
    variants.append(re.sub(r"\([^)]*\)", "", event))
    ignored = {"sound", "sounds", "noise", "noises", "other"}
    for variant in variants:
        words = [word for word in normalized_words(variant) if word not in ignored]
        if not words:
            continue
        if all(word in caption_tokens for word in words):
            return True
        phrase = " ".join(words)
        if f" {phrase} " in f" {' '.join(caption_words)} ":
            return True
    return False


def mix_pair(
    clean: np.ndarray,
    interference_stem: np.ndarray,
    sample_rate: int,
    snr_db: float,
    headroom: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    max_onset = clean.size - interference_stem.size
    onset = int(rng.integers(0, max_onset + 1)) if max_onset else 0
    clean_rms = rms(clean)
    stem_rms = rms(interference_stem)
    interference_gain = clean_rms / (
        stem_rms * (10.0 ** (snr_db / 20.0)) + EPSILON
    )
    aligned = np.zeros_like(clean)
    aligned[onset : onset + interference_stem.size] = (
        interference_stem * interference_gain
    )
    mixture = clean + aligned
    peak = max(
        float(np.max(np.abs(clean))),
        float(np.max(np.abs(aligned))),
        float(np.max(np.abs(mixture))),
    )
    global_gain = min(1.0, headroom / peak) if peak > 0.0 else 1.0
    target = (clean * global_gain).astype(np.float32)
    aligned = (aligned * global_gain).astype(np.float32)
    mixture = (target + aligned).astype(np.float32)
    realized_snr = 20.0 * math.log10(
        (rms(target) + EPSILON)
        / (rms(aligned[onset : onset + interference_stem.size]) + EPSILON)
    )
    metadata = {
        "interference_gain": float(interference_gain),
        "global_gain": float(global_gain),
        "onset_seconds": onset / sample_rate,
        "offset_seconds": (onset + interference_stem.size) / sample_rate,
        "snr_db_realized": float(realized_snr),
        "peak": float(np.max(np.abs(mixture))),
    }
    return mixture, target, aligned, metadata


def relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sample_rate, subtype="PCM_16")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_dataset(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    staging_root = output_root.with_name(output_root.name + ".building")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Use --overwrite to rebuild."
            )
        shutil.rmtree(output_root)
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)

    try:
        shard_paths = ensure_audiocaps_shards(
            args.raw_cache.resolve(), args.audiocaps_shards
        )
        clean_sources = load_clean_sources(shard_paths)
        interference_sources = load_interference_sources(args.audiotime_root.resolve())

        rng = random.Random(args.seed)
        rng.shuffle(clean_sources)
        rng.shuffle(interference_sources)

        requested_total = args.train_size + args.val_size + args.test_size
        if len(clean_sources) < requested_total:
            raise RuntimeError(
                f"Selected shards contain {len(clean_sources)} unique clean clips, "
                f"but {requested_total} are required. Increase --audiocaps-shards."
            )
        if len(interference_sources) < requested_total:
            raise RuntimeError(
                f"Only {len(interference_sources)} eligible interference clips are available"
            )

        split_sizes = {
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
        }
        target_samples = int(round(args.sample_rate * args.duration_seconds))
        manifests: Dict[str, List[Dict[str, Any]]] = {
            split: [] for split in split_sizes
        }
        clean_cursor = 0
        interference_cursor = 0

        for split, split_size in split_sizes.items():
            while len(manifests[split]) < split_size:
                if clean_cursor >= len(clean_sources):
                    raise RuntimeError(
                        "Ran out of valid AudioCaps clips; increase --audiocaps-shards"
                    )
                clean_source = clean_sources[clean_cursor]
                clean_cursor += 1
                index = len(manifests[split])
                sample_seed = stable_seed(args.seed, split, index)
                sample_rng = np.random.default_rng(sample_seed)
                try:
                    clean_audio = prepare_clean_source(
                        clean_source,
                        args.sample_rate,
                        target_samples,
                        sample_rng,
                    )
                except Exception as exc:
                    print(
                        f"Skipping clean source {clean_source.source_id}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                pair_created = False
                while interference_cursor < len(interference_sources):
                    interference_source = interference_sources[interference_cursor]
                    interference_cursor += 1
                    if event_mentioned_in_caption(
                        interference_source.event, clean_source.caption
                    ):
                        continue
                    try:
                        stem, interference_source_rate = prepare_interference_source(
                            interference_source,
                            args.sample_rate,
                            target_samples,
                            args.fade_milliseconds,
                        )
                        snr_db = float(
                            sample_rng.uniform(args.snr_min_db, args.snr_max_db)
                        )
                        mixture, target, aligned, mix_metadata = mix_pair(
                            clean_audio.waveform,
                            stem,
                            args.sample_rate,
                            snr_db,
                            args.headroom,
                            sample_rng,
                        )
                    except Exception as exc:
                        print(
                            f"Skipping interference {interference_source.source_id}: {exc}",
                            file=sys.stderr,
                        )
                        continue

                    sample_id = f"{split}_{index:06d}"
                    mixture_path = (
                        staging_root / "audio" / "mixture" / split / f"{sample_id}.wav"
                    )
                    target_path = (
                        staging_root / "audio" / "target" / split / f"{sample_id}.wav"
                    )
                    interference_path = (
                        staging_root
                        / "audio"
                        / "interference"
                        / split
                        / f"{sample_id}.wav"
                    )
                    write_wav(mixture_path, mixture, args.sample_rate)
                    write_wav(target_path, target, args.sample_rate)
                    write_wav(interference_path, aligned, args.sample_rate)

                    source_relpath = relative_posix(
                        interference_source.audio_path, project_root
                    )
                    manifests[split].append(
                        {
                            "schema_version": "1.0",
                            "id": sample_id,
                            "split": split,
                            "task": "remove",
                            "instruction": f"Remove {interference_source.event}",
                            "remove_event": interference_source.event,
                            "sample_rate": args.sample_rate,
                            "num_samples": target_samples,
                            "duration_seconds": args.duration_seconds,
                            "mixture_path": relative_posix(mixture_path, staging_root),
                            "target_path": relative_posix(target_path, staging_root),
                            "interference_path": relative_posix(
                                interference_path, staging_root
                            ),
                            "snr_db_requested": snr_db,
                            "snr_db_realized": mix_metadata["snr_db_realized"],
                            "interference_onset_seconds": mix_metadata[
                                "onset_seconds"
                            ],
                            "interference_offset_seconds": mix_metadata[
                                "offset_seconds"
                            ],
                            "clean_source": {
                                "dataset": "AudioCaps",
                                "repository": AUDIOCAPS_REPO,
                                "revision": AUDIOCAPS_REVISION,
                                "source_split": "train",
                                "source_id": clean_source.source_id,
                                "audiocap_id": clean_source.audiocap_id,
                                "youtube_id": clean_source.youtube_id,
                                "start_time": clean_source.start_time,
                                "caption": clean_source.caption,
                                "source_audio_path": clean_source.audio_path,
                                "source_sample_rate": clean_audio.source_sample_rate,
                                "source_crop_start_seconds": clean_audio.crop_start_seconds,
                                "shard": clean_source.shard_name,
                            },
                            "interference_source": {
                                "dataset": "AudioTime",
                                "source_split": "train5000_timestamp",
                                "source_id": interference_source.source_id,
                                "event": interference_source.event,
                                "caption": interference_source.caption,
                                "source_path": source_relpath,
                                "source_interval_seconds": list(
                                    interference_source.interval_seconds
                                ),
                                "source_sample_rate": interference_source_rate,
                            },
                            "mixing": {
                                "seed": sample_seed,
                                "interference_gain": mix_metadata[
                                    "interference_gain"
                                ],
                                "global_gain": mix_metadata["global_gain"],
                                "fade_milliseconds": args.fade_milliseconds,
                                "headroom": args.headroom,
                            },
                        }
                    )
                    pair_created = True
                    print(
                        f"Built {sample_id}: Remove {interference_source.event} "
                        f"at {snr_db:.2f} dB"
                    )
                    break

                if not pair_created:
                    raise RuntimeError("Ran out of eligible AudioTime interference clips")

        manifest_root = staging_root / "manifests"
        for split, rows in manifests.items():
            write_jsonl(manifest_root / f"{split}.jsonl", rows)
        write_jsonl(
            manifest_root / "overfit16.jsonl",
            manifests["train"][: args.overfit_size],
        )

        config = {
            "schema_version": "1.0",
            "builder_version": BUILDER_VERSION,
            "task": "remove",
            "seed": args.seed,
            "sample_rate": args.sample_rate,
            "duration_seconds": args.duration_seconds,
            "num_samples": target_samples,
            "snr_db_range": [args.snr_min_db, args.snr_max_db],
            "headroom": args.headroom,
            "fade_milliseconds": args.fade_milliseconds,
            "splits": {
                **{split: len(rows) for split, rows in manifests.items()},
                "overfit16": args.overfit_size,
            },
            "clean_source": {
                "dataset": "AudioCaps",
                "repository": AUDIOCAPS_REPO,
                "revision": AUDIOCAPS_REVISION,
                "license": AUDIOCAPS_LICENSE,
                "usage": "research/non-commercial",
                "source_split": "train",
                "shards": [
                    {
                        "name": path.name,
                        "sha256": sha256_file(path),
                    }
                    for path in shard_paths
                ],
            },
            "interference_source": {
                "dataset": "AudioTime",
                "source_split": "train5000_timestamp",
                "license": "See upstream AudioTime terms",
                "selection": "exactly one event label and one timestamp interval",
            },
            "audio_format": {
                "container": "WAV",
                "subtype": "PCM_16",
                "channels": 1,
            },
        }
        (staging_root / "dataset_config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        validate_dataset(staging_root, write_report=True)
        os.replace(staging_root, output_root)
        print(f"Dataset ready: {output_root}")
    except Exception:
        print(f"Build failed; staging data left at {staging_root}", file=sys.stderr)
        raise


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_dataset(root: Path, write_report: bool = False) -> Dict[str, Any]:
    root = root.resolve()
    config_path = root / "dataset_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing dataset config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_rate = int(config["sample_rate"])
    expected_samples = int(config["num_samples"])
    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    clean_ids: Dict[str, set] = {}
    interference_ids: Dict[str, set] = {}
    maximum_reconstruction_error = 0.0
    maximum_peak = 0.0
    maximum_snr_error_db = 0.0
    checked = 0

    for split in ("train", "val", "test"):
        manifest_path = root / "manifests" / f"{split}.jsonl"
        rows = read_jsonl(manifest_path)
        expected_count = int(config["splits"][split])
        if len(rows) != expected_count:
            raise AssertionError(
                f"{split} count mismatch: expected {expected_count}, got {len(rows)}"
            )
        split_rows[split] = rows
        clean_ids[split] = set()
        interference_ids[split] = set()

        for row in rows:
            if row["split"] != split or row["task"] != "remove":
                raise AssertionError(f"Invalid task/split in sample {row.get('id')}")
            if row["instruction"] != f"Remove {row['remove_event']}":
                raise AssertionError(f"Invalid instruction in sample {row['id']}")
            arrays: Dict[str, np.ndarray] = {}
            for key in ("mixture_path", "target_path", "interference_path"):
                audio_path = root / row[key]
                if not audio_path.exists():
                    raise FileNotFoundError(f"Missing audio file: {audio_path}")
                waveform, sample_rate = read_audio(audio_path)
                if sample_rate != expected_rate:
                    raise AssertionError(
                        f"Sample rate mismatch for {audio_path}: {sample_rate}"
                    )
                if waveform.size != expected_samples:
                    raise AssertionError(
                        f"Length mismatch for {audio_path}: {waveform.size}"
                    )
                arrays[key] = waveform
                maximum_peak = max(maximum_peak, float(np.max(np.abs(waveform))))

            reconstruction_error = float(
                np.max(
                    np.abs(
                        arrays["mixture_path"]
                        - arrays["target_path"]
                        - arrays["interference_path"]
                    )
                )
            )
            maximum_reconstruction_error = max(
                maximum_reconstruction_error, reconstruction_error
            )
            if reconstruction_error > 1.1e-4:
                raise AssertionError(
                    f"Mixture reconstruction error too high for {row['id']}: "
                    f"{reconstruction_error}"
                )
            if maximum_peak > 1.0:
                raise AssertionError("Detected clipped audio outside [-1, 1]")

            onset = int(round(row["interference_onset_seconds"] * expected_rate))
            offset = int(round(row["interference_offset_seconds"] * expected_rate))
            active_interference = arrays["interference_path"][onset:offset]
            measured_snr = 20.0 * math.log10(
                (rms(arrays["target_path"]) + EPSILON)
                / (rms(active_interference) + EPSILON)
            )
            snr_error = abs(measured_snr - float(row["snr_db_realized"]))
            maximum_snr_error_db = max(maximum_snr_error_db, snr_error)
            if snr_error > 0.15:
                raise AssertionError(
                    f"SNR mismatch for {row['id']}: error={snr_error:.4f} dB"
                )

            clean_ids[split].add(row["clean_source"]["source_id"])
            interference_ids[split].add(
                row["interference_source"]["source_id"]
            )
            checked += 1

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        clean_overlap = clean_ids[left] & clean_ids[right]
        interference_overlap = interference_ids[left] & interference_ids[right]
        if clean_overlap:
            raise AssertionError(
                f"Clean-source leakage between {left}/{right}: {sorted(clean_overlap)[:3]}"
            )
        if interference_overlap:
            raise AssertionError(
                f"Interference-source leakage between {left}/{right}: "
                f"{sorted(interference_overlap)[:3]}"
            )

    overfit_rows = read_jsonl(root / "manifests" / "overfit16.jsonl")
    expected_overfit = int(config["splits"]["overfit16"])
    if len(overfit_rows) != expected_overfit:
        raise AssertionError(
            f"overfit16 count mismatch: expected {expected_overfit}, "
            f"got {len(overfit_rows)}"
        )
    train_ids = {row["id"] for row in split_rows["train"]}
    if not {row["id"] for row in overfit_rows}.issubset(train_ids):
        raise AssertionError("overfit16 must be a subset of train")

    report = {
        "status": "passed",
        "checked_samples": checked,
        "split_counts": {
            split: len(rows) for split, rows in split_rows.items()
        },
        "overfit_count": len(overfit_rows),
        "sample_rate": expected_rate,
        "num_samples": expected_samples,
        "maximum_peak": maximum_peak,
        "maximum_reconstruction_error": maximum_reconstruction_error,
        "maximum_snr_error_db": maximum_snr_error_db,
        "source_leakage": False,
    }
    if write_report:
        (root / "validation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    args = parse_args()
    if args.validate_only:
        validate_dataset(args.output_root, write_report=True)
    else:
        build_dataset(args)


if __name__ == "__main__":
    main()
