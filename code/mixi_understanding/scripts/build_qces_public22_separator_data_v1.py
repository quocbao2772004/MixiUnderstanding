#!/usr/bin/env python3
"""Build a provenance-explicit 22-label predicted-span separator pilot.

The strong full-ontology condition manifest supplies real predicted spans.
Seven public labels use declared acoustic proxies because their exact public
surface names were not selected into the 191-label training ontology.  The
`Thump_and_thud` class is materialized from exact-label, source-disjoint FSD50K
clips and mixed with two distractors; it is never aliased to `Slam`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


FORMAT = "qces_public22_separator_data_v1"
RATE = 16_000
SCENE_SECONDS = 5.12
SCENE_SAMPLES = int(RATE * SCENE_SECONDS)
PUBLIC22_PATH = PROJECT_ROOT / "code/mixi_understanding/resources/qces_public22_ontology.txt"

# Direct labels keep their original semantics.  Proxies are explicit and are
# reported independently so they cannot be mistaken for exact-label data.
SOURCE_TO_PUBLIC = {
    "Bark": "Bark",
    "Howl": "Howl",
    "Whimper_(dog)": "Whimper_(dog)",
    "Meow": "Meow",
    "Purr": "Purr",
    "Caterwaul": "Caterwaul",
    "Clapping": "Clapping",
    "Baby_laughter": "Laughter",
    "Giggle": "Giggle",
    "Hubbub_and_speech_noise_and_speech_babble": "Conversation",
    "Screaming": "Shout",
    "Baby_cry_and_infant_cry": "Crying_and_sobbing",
    "Tap": "Knock",
    "Slam": "Slam",
    "Traffic_noise_and_roadway_noise": "Traffic_noise_and_roadway_noise",
    "Heavy_engine_(low_frequency)": "Heavy_engine_(low_frequency)",
    "Reversing_beeps": "Reversing_beeps",
    "Female_speech_and_woman_speaking": "Speech",
    "Male_speech_and_man_speaking": "Speech",
    "Child_speech_and_kid_speaking": "Speech",
    "Air_horn_and_truck_horn": "Air_horn_and_truck_horn",
    "Police_car_(siren)": "Police_car_(siren)",
    "Engine_starting": "Engine_starting",
}


def parse_args() -> argparse.Namespace:
    conditions = PROJECT_ROOT / "outputs/qces_predicted_span_conditions_v1"
    sourcebank = PROJECT_ROOT / "outputs/qces_detector_sourcebank_fuss_fsd50k_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-conditions", type=Path, default=conditions / "event_components_train.jsonl")
    parser.add_argument("--dev-conditions", type=Path, default=conditions / "event_components_dev.jsonl")
    parser.add_argument("--train-sourcebank", type=Path, default=sourcebank / "detector_source_manifest_train_clean.jsonl")
    parser.add_argument("--dev-sourcebank", type=Path, default=sourcebank / "detector_source_manifest_val_clean.jsonl")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/qces_public22_separator_data_v1")
    parser.add_argument("--seed", type=int, default=2243)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load(path: Path) -> np.ndarray:
    wave, source_rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if int(source_rate) != RATE:
        divisor = math.gcd(int(source_rate), RATE)
        wave = resample_poly(wave, RATE // divisor, int(source_rate) // divisor)
    return np.asarray(wave, dtype=np.float32)


def _interval_iou(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / max(union, 1)


def _transform_conditions(rows: Sequence[dict[str, Any]], labels: Sequence[str]) -> list[dict[str, Any]]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    output: list[dict[str, Any]] = []
    for source in rows:
        public = SOURCE_TO_PUBLIC.get(str(source["label"]))
        if public is None or not bool(source.get("condition_available")):
            continue
        row = dict(source)
        row.update({
            "source_label": str(source["label"]),
            "label": public,
            "label_id": label_to_id[public],
            "semantic_supervision": "exact" if source["label"] == public else "declared_acoustic_proxy",
            "condition_source": "frozen_temporal_v2_prediction",
        })
        output.append(row)
    return output


def _source_segment(row: Mapping[str, Any]) -> np.ndarray:
    wave = _load(_resolve(str(row["mixture_path"])))
    event = row["events"][0]
    begin = max(0, int(round(float(event["onset_seconds"]) * RATE)))
    end = min(len(wave), int(round(float(event["offset_seconds"]) * RATE)))
    segment = wave[begin:end]
    if not len(segment):
        raise RuntimeError(f"empty source segment: {row['scene_id']}")
    peak = float(np.max(np.abs(segment)))
    return segment / max(peak, 1e-4)


def _materialize_thumps(
    *, rows: Sequence[dict[str, Any]], split: str, output: Path,
    labels: Sequence[str], seed: int,
) -> list[dict[str, Any]]:
    target_rows = [row for row in rows if "Thump_and_thud" in row.get("labels", [])]
    distractors = [row for row in rows if "Thump_and_thud" not in row.get("labels", [])]
    rng = random.Random(seed)
    label_id = list(labels).index("Thump_and_thud")
    rendered: list[dict[str, Any]] = []
    for index, target_row in enumerate(target_rows):
        target = _source_segment(target_row)[: int(1.8 * RATE)]
        onset = rng.randint(int(0.65 * RATE), int(1.35 * RATE))
        offset = onset + len(target)
        if offset > SCENE_SAMPLES:
            target = target[: SCENE_SAMPLES - onset]
            offset = onset + len(target)
        mixture = np.zeros(SCENE_SAMPLES, dtype=np.float32)
        target_gain = 10.0 ** (rng.uniform(-2.0, 2.0) / 20.0)
        mixture[onset:offset] += target * target_gain
        chosen = rng.sample(distractors, k=min(2, len(distractors)))
        for distractor_row in chosen:
            distractor = _source_segment(distractor_row)[: int(1.8 * RATE)]
            d_onset = max(0, onset + rng.randint(-int(0.45 * RATE), int(0.45 * RATE)))
            d_end = min(SCENE_SAMPLES, d_onset + len(distractor))
            d_gain = 10.0 ** (rng.uniform(-4.0, 2.0) / 20.0)
            mixture[d_onset:d_end] += distractor[: d_end - d_onset] * d_gain
        peak = float(np.max(np.abs(mixture)))
        global_scale = 0.90 / max(peak, 0.90)
        mixture *= global_scale
        component = target * target_gain * global_scale
        scene_id = f"public22_thump_{split}_{index:05d}"
        audio_dir = output / "audio" / split / scene_id
        audio_dir.mkdir(parents=True, exist_ok=True)
        mixture_path = audio_dir / "mixture.wav"
        component_path = audio_dir / "component.wav"
        sf.write(mixture_path, mixture, RATE, subtype="PCM_16")
        sf.write(component_path, component, RATE, subtype="PCM_16")
        pad_left = rng.randint(int(0.04 * RATE), int(0.20 * RATE))
        pad_right = rng.randint(int(0.04 * RATE), int(0.20 * RATE))
        jitter = rng.randint(-int(0.10 * RATE), int(0.10 * RATE))
        condition_onset = max(0, onset - pad_left + jitter)
        condition_offset = min(SCENE_SAMPLES, offset + pad_right + jitter)
        rendered.append({
            "format": FORMAT,
            "scene_id": scene_id,
            "event_id": f"{scene_id}:e00",
            "split": split,
            "label": "Thump_and_thud",
            "source_label": "Thump_and_thud",
            "label_id": label_id,
            "mixture_path": str(mixture_path.resolve()),
            "component_path": str(component_path.resolve()),
            "sample_rate": RATE,
            "onset_sample": onset,
            "offset_sample": offset,
            "num_component_samples": len(component),
            "condition_available": True,
            "condition_onset_sample": condition_onset,
            "condition_offset_sample": condition_offset,
            "condition_iou": _interval_iou((onset, offset), (condition_onset, condition_offset)),
            "condition_score": None,
            "condition_source": "source_disjoint_synthetic_boundary_noise",
            "semantic_supervision": "exact",
            "source_id": str(target_row["source_id"]),
            "recipe_kind": "target_plus_two_source_disjoint_distractors",
            "requested_overlap_fraction": 1.0,
        })
    return rendered


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    labels = [line.strip() for line in PUBLIC22_PATH.read_text().splitlines() if line.strip()]
    split_specs = {
        "train": (args.train_conditions.resolve(), args.train_sourcebank.resolve(), args.seed),
        "dev": (args.dev_conditions.resolve(), args.dev_sourcebank.resolve(), args.seed + 1),
    }
    summaries: dict[str, Any] = {}
    for split, (conditions_path, sourcebank_path, seed) in split_specs.items():
        transformed = _transform_conditions(_read_jsonl(conditions_path), labels)
        thumps = _materialize_thumps(
            rows=_read_jsonl(sourcebank_path), split=split, output=output,
            labels=labels, seed=seed,
        )
        rows = transformed + thumps
        rows.sort(key=lambda row: (str(row["scene_id"]), str(row["event_id"]), str(row["label"])))
        manifest = output / f"event_components_{split}.jsonl"
        _atomic_jsonl(manifest, rows)
        counts = Counter(str(row["label"]) for row in rows)
        source_modes = Counter(str(row["semantic_supervision"]) for row in rows)
        missing = [label for label in labels if counts[label] == 0]
        if missing:
            raise RuntimeError(f"{split}: labels without data: {missing}")
        summaries[split] = {
            "events": len(rows),
            "labels": len(counts),
            "events_per_label": dict(sorted(counts.items())),
            "semantic_supervision": dict(sorted(source_modes.items())),
            "condition_iou_median": float(np.median([float(row["condition_iou"]) for row in rows])),
            "manifest": str(manifest),
            "manifest_sha256": _sha256(manifest),
        }
        print(json.dumps({"split": split, **summaries[split]}, ensure_ascii=False), flush=True)
    receipt = {
        "format": FORMAT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "labels": labels,
        "source_to_public": SOURCE_TO_PUBLIC,
        "proxy_policy": "declared acoustic proxies only; exact/proxy results must be reported separately",
        "splits": summaries,
        "inputs": {
            "train_conditions_sha256": _sha256(args.train_conditions.resolve()),
            "dev_conditions_sha256": _sha256(args.dev_conditions.resolve()),
            "train_sourcebank_sha256": _sha256(args.train_sourcebank.resolve()),
            "dev_sourcebank_sha256": _sha256(args.dev_sourcebank.resolve()),
        },
    }
    _atomic_json(output / "receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
