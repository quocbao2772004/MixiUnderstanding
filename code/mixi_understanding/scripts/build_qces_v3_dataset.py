#!/usr/bin/env python3
"""Build a question-contrastive QCES v3 overfit dataset from AudioTime sources."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import scipy
import soundfile as sf

from mixi_understanding.data.qces_schema import (
    NO_EVIDENCE_ANSWER,
    SCHEMA_VERSION,
    parse_qces_record,
)
from mixi_understanding.scripts.build_qa_removal_dataset import (
    SourceClip,
    crop_source_event,
    load_sources,
    natural_source_key,
    relative_posix,
    render_clip,
    replace_validated_output,
    rms,
    sha256_file,
    stable_seed,
    write_jsonl,
    write_wav,
)


BUILDER_VERSION = "3.0.0"
SAMPLE_RATE = 32_000
DURATION_SECONDS = 8.0
EVENT_DURATION_SECONDS = 1.0
NUISANCE_DURATION_SECONDS = 5.8
NUISANCE_SNR_DB = -3.0
HEADROOM = 0.95
FADE_MILLISECONDS = 10.0
SEMANTIC_LABELS = ("Buzz", "Croak", "Engine knocking", "Jackhammer")
NUISANCE_LABELS = ("Ambulance (siren)", "Sawing")
ABSENT_LABELS = ("Rain", "Dog barking", "Glass breaking", "Church bell")
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--audiotime-root",
        type=Path,
        default=PROJECT_ROOT / "AudioTime-recovered" / "train5000_timestamp",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "qces_synthetic_v3",
    )
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def _receipt_sources(root: Path) -> tuple[Dict[str, Any], List[SourceClip]]:
    receipt_path = root / "subset_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("format") != "audiotime_subset_v1":
        raise ValueError("unsupported or missing AudioTime subset receipt")
    receipt_sources = receipt.get("sources")
    if not isinstance(receipt_sources, dict):
        raise TypeError("receipt sources must be an object")
    metadata_sources = {
        source.source_id: source
        for source in load_sources(root, require_audio=False)
    }
    selected = []
    for source_id, entry in receipt_sources.items():
        source = metadata_sources.get(source_id)
        if source is None:
            raise ValueError(f"receipt source is absent from metadata: {source_id}")
        if not source.audio_path.is_file():
            raise FileNotFoundError(source.audio_path)
        if sha256_file(source.audio_path) != entry.get("sha256"):
            raise ValueError(f"source hash differs from receipt: {source_id}")
        selected.append(source)
    return receipt, sorted(selected, key=natural_source_key)


def _group_sources(sources: Sequence[SourceClip]) -> Dict[str, List[SourceClip]]:
    grouped: Dict[str, List[SourceClip]] = {}
    for source in sources:
        grouped.setdefault(source.label, []).append(source)
    for label in SEMANTIC_LABELS + NUISANCE_LABELS:
        if len(grouped.get(label, [])) < 4:
            raise ValueError(f"need four downloaded sources for {label}")
        grouped[label] = sorted(grouped[label], key=natural_source_key)[:4]
    return grouped


def _event_payload(
    event_id: str,
    kind: str,
    source: SourceClip,
    source_hash: str,
    crop_interval: Tuple[float, float],
    rendered_interval: Tuple[float, float],
    stem_path: Path,
    project_root: Path,
    staging_root: Path,
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "label": source.label,
        "event_kind": kind,
        "source_dataset": "AudioTime",
        "source_id": source.source_id,
        "source_path": relative_posix(source.audio_path, project_root),
        "source_sha256": source_hash,
        "source_interval_seconds": list(source.interval_seconds),
        "source_crop_interval_seconds": list(crop_interval),
        "onset_seconds": rendered_interval[0],
        "offset_seconds": rendered_interval[1],
        "stem_path": relative_posix(stem_path, staging_root),
    }


def _question_specs(
    events: Sequence[Mapping[str, Any]], scene_index: int
) -> List[Dict[str, Any]]:
    semantic = {
        event["event_id"]: event
        for event in events
        if event["event_kind"] == "semantic"
    }
    first, second, third, fourth = (
        semantic[f"event_semantic_{index}"] for index in range(4)
    )
    first_candidates = [third["label"], second["label"]]
    if scene_index % 2:
        first_candidates.reverse()
    return [
        {
            "question_type": "temporal_after",
            "question": f"What sound occurs immediately after {first['label']}?",
            "answer": second["label"],
            "anchor": first["event_id"],
            "answer_id": second["event_id"],
            "absent_label": None,
        },
        {
            "question_type": "temporal_before",
            "question": f"What sound occurs immediately before {fourth['label']}?",
            "answer": third["label"],
            "anchor": fourth["event_id"],
            "answer_id": third["event_id"],
            "absent_label": None,
        },
        {
            "question_type": "temporal_first",
            "question": (
                f"Which sound occurs first, {first_candidates[0]} "
                f"or {first_candidates[1]}?"
            ),
            "answer": second["label"],
            "anchor": third["event_id"],
            "answer_id": second["event_id"],
            "absent_label": None,
        },
        {
            "question_type": "no_evidence_after",
            "question": (
                f"What sound occurs immediately after {ABSENT_LABELS[scene_index]}?"
            ),
            "answer": NO_EVIDENCE_ANSWER,
            "anchor": None,
            "answer_id": None,
            "absent_label": ABSENT_LABELS[scene_index],
        },
    ]


def _sum_stems(stems: Mapping[str, np.ndarray], ids: Iterable[str]) -> np.ndarray:
    result = np.zeros_like(next(iter(stems.values())))
    for event_id in ids:
        result += stems[event_id]
    return result


def _compose_scene(
    scene_index: int,
    grouped: Mapping[str, Sequence[SourceClip]],
    source_hashes: Mapping[str, str],
    metadata_hash: str,
    args: argparse.Namespace,
    staging_root: Path,
) -> List[Dict[str, Any]]:
    semantic_sources = [grouped[label][scene_index] for label in SEMANTIC_LABELS]
    nuisance_label = NUISANCE_LABELS[scene_index % len(NUISANCE_LABELS)]
    nuisance_source = grouped[nuisance_label][scene_index]
    source_identity = [
        item
        for source in semantic_sources + [nuisance_source]
        for item in (source.source_id, source_hashes[source.source_id])
    ]
    scene_seed = stable_seed(
        args.seed,
        "qces-v3-scene",
        scene_index,
        metadata_hash,
        *source_identity,
    )
    rng = np.random.default_rng(scene_seed)
    order = rng.permutation(len(semantic_sources))
    ordered_sources = [semantic_sources[int(index)] for index in order]
    target_samples = int(round(SAMPLE_RATE * DURATION_SECONDS))
    event_samples = int(round(SAMPLE_RATE * EVENT_DURATION_SECONDS))
    onset = int(rng.integers(int(0.70 * SAMPLE_RATE), int(0.90 * SAMPLE_RATE)))
    onset_samples = []
    for index in range(4):
        onset_samples.append(onset)
        if index < 3:
            gap = int(rng.integers(int(0.30 * SAMPLE_RATE), int(0.50 * SAMPLE_RATE)))
            onset += event_samples + gap
    nuisance_onset = int(round(0.45 * SAMPLE_RATE))
    if onset_samples[-1] + event_samples > target_samples:
        raise AssertionError("semantic layout exceeds the scene")

    scene_sources = ordered_sources + [nuisance_source]
    crop_rngs = {
        source.source_id: np.random.default_rng(
            stable_seed(scene_seed, "crop", source.source_id)
        )
        for source in scene_sources
    }
    crops: Dict[str, Tuple[float, float]] = {}
    intervals: Dict[str, Tuple[float, float]] = {}
    raw_stems: Dict[str, np.ndarray] = {}
    semantic_active = np.zeros(target_samples, dtype=bool)
    for index, (source, event_onset) in enumerate(zip(ordered_sources, onset_samples)):
        event_id = f"event_semantic_{index}"
        clip, crop = crop_source_event(
            source,
            EVENT_DURATION_SECONDS,
            SAMPLE_RATE,
            FADE_MILLISECONDS,
            crop_rngs[source.source_id],
        )
        stem = np.zeros(target_samples, dtype=np.float32)
        interval = render_clip(stem, clip, event_onset, SAMPLE_RATE)
        crops[event_id] = crop
        intervals[event_id] = interval
        raw_stems[event_id] = stem
        semantic_active[event_onset : event_onset + clip.size] = True

    nuisance_id = "event_nuisance"
    nuisance_clip, nuisance_crop = crop_source_event(
        nuisance_source,
        NUISANCE_DURATION_SECONDS,
        SAMPLE_RATE,
        FADE_MILLISECONDS,
        crop_rngs[nuisance_source.source_id],
    )
    nuisance_stem = np.zeros(target_samples, dtype=np.float32)
    nuisance_interval = render_clip(
        nuisance_stem, nuisance_clip, nuisance_onset, SAMPLE_RATE
    )
    semantic_mix = _sum_stems(raw_stems, raw_stems.keys())
    nuisance_gain = rms(semantic_mix[semantic_active]) / (
        max(rms(nuisance_stem[semantic_active]), EPSILON)
        * 10.0 ** (NUISANCE_SNR_DB / 20.0)
    )
    nuisance_stem *= nuisance_gain
    raw_stems[nuisance_id] = nuisance_stem
    crops[nuisance_id] = nuisance_crop
    intervals[nuisance_id] = nuisance_interval
    mixture = _sum_stems(raw_stems, raw_stems.keys())
    peak = float(np.max(np.abs(mixture)))
    gain = min(1.0, HEADROOM / peak) if peak > 0 else 1.0
    stems = {event_id: stem * gain for event_id, stem in raw_stems.items()}
    mixture = _sum_stems(stems, stems.keys()).astype(np.float32)
    realized_snr = 20.0 * math.log10(
        (rms(_sum_stems(stems, stems.keys() - {nuisance_id})[semantic_active]) + EPSILON)
        / (rms(stems[nuisance_id][semantic_active]) + EPSILON)
    )

    scene_id = f"scene_{scene_index:06d}"
    mixture_path = staging_root / "audio" / "mixture" / "train" / f"{scene_id}.wav"
    write_wav(mixture_path, mixture, SAMPLE_RATE)
    events = []
    for index, source in enumerate(ordered_sources):
        event_id = f"event_semantic_{index}"
        event_path = staging_root / "audio" / "events" / "train" / scene_id / f"{event_id}.wav"
        write_wav(event_path, stems[event_id], SAMPLE_RATE)
        events.append(
            _event_payload(
                event_id,
                "semantic",
                source,
                source_hashes[source.source_id],
                crops[event_id],
                intervals[event_id],
                event_path,
                args.project_root.resolve(),
                staging_root,
            )
        )
    nuisance_path = staging_root / "audio" / "events" / "train" / scene_id / f"{nuisance_id}.wav"
    write_wav(nuisance_path, stems[nuisance_id], SAMPLE_RATE)
    events.append(
        _event_payload(
            nuisance_id,
            "nuisance",
            nuisance_source,
            source_hashes[nuisance_source.source_id],
            crops[nuisance_id],
            intervals[nuisance_id],
            nuisance_path,
            args.project_root.resolve(),
            staging_root,
        )
    )
    events.sort(key=lambda event: (event["onset_seconds"], event["event_id"]))
    event_by_id = {event["event_id"]: event for event in events}

    records = []
    for question_index, spec in enumerate(_question_specs(events, scene_index)):
        sample_id = f"train_{scene_index * 4 + question_index:06d}"
        answerable = spec["anchor"] is not None
        anchor_ids = [spec["anchor"]] if answerable else []
        answer_ids = [spec["answer_id"]] if answerable else []
        evidence_ids = anchor_ids + answer_ids
        evidence = _sum_stems(stems, evidence_ids)
        residual_ids = set(stems) - set(evidence_ids)
        residual = _sum_stems(stems, residual_ids)
        anchor = _sum_stems(stems, anchor_ids)
        answer = _sum_stems(stems, answer_ids)
        output_paths = {
            name: staging_root / "audio" / name / "train" / f"{sample_id}.wav"
            for name in ("evidence", "residual", "anchor", "answer")
        }
        for name, waveform in (
            ("evidence", evidence),
            ("residual", residual),
            ("anchor", anchor),
            ("answer", answer),
        ):
            write_wav(output_paths[name], waveform, SAMPLE_RATE)
        record = {
            "schema_version": SCHEMA_VERSION,
            "id": sample_id,
            "scene_id": scene_id,
            "question_family_id": f"{scene_id}:contrastive",
            "question_index": question_index,
            "split": "train",
            "sample_rate": SAMPLE_RATE,
            "num_channels": 1,
            "num_samples": target_samples,
            "duration_seconds": DURATION_SECONDS,
            "mixture_path": relative_posix(mixture_path, staging_root),
            "evidence_stem_path": relative_posix(output_paths["evidence"], staging_root),
            "residual_stem_path": relative_posix(output_paths["residual"], staging_root),
            "anchor_stem_path": relative_posix(output_paths["anchor"], staging_root),
            "answer_stem_path": relative_posix(output_paths["answer"], staging_root),
            "question": spec["question"],
            "answer": spec["answer"],
            "question_type": spec["question_type"],
            "no_evidence": not answerable,
            "absent_label": spec["absent_label"],
            "events": events,
            "anchor_event_ids": anchor_ids,
            "answer_event_ids": answer_ids,
            "evidence_event_ids": evidence_ids,
            "anchor_intervals": [],
            "answer_intervals": [],
            "event_presence_labels": sorted(event["label"] for event in events),
            "source_group_ids": sorted(event["source_id"] for event in events),
            "nuisance_snr_db_requested": NUISANCE_SNR_DB,
            "nuisance_snr_db": realized_snr,
            "mixture_peak": float(np.max(np.abs(mixture))),
            "generation_seed": scene_seed,
        }
        record["anchor_intervals"] = (
            [[event_by_id[anchor_ids[0]]["onset_seconds"], event_by_id[anchor_ids[0]]["offset_seconds"]]]
            if anchor_ids
            else []
        )
        record["answer_intervals"] = (
            [[event_by_id[answer_ids[0]]["onset_seconds"], event_by_id[answer_ids[0]]["offset_seconds"]]]
            if answer_ids
            else []
        )
        parse_qces_record(record)
        records.append(record)
    return records


def build_dataset(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    args.project_root = project_root
    audiotime_root = args.audiotime_root.resolve()
    output_root = args.output_root.resolve()
    staging_root = output_root.with_name(output_root.name + ".building")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_root}; use --overwrite")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    try:
        receipt, sources = _receipt_sources(audiotime_root)
        grouped = _group_sources(sources)
        source_hashes = {
            source.source_id: sha256_file(source.audio_path) for source in sources
        }
        metadata_path = audiotime_root / "timestamp_captions.json"
        metadata_hash = sha256_file(metadata_path)
        records = []
        for scene_index in range(4):
            records.extend(
                _compose_scene(
                    scene_index,
                    grouped,
                    source_hashes,
                    metadata_hash,
                    args,
                    staging_root,
                )
            )
            print(f"built scene_{scene_index:06d}: four contrastive questions")
        manifest_path = staging_root / "qces_overfit16.jsonl"
        write_jsonl(manifest_path, records)
        source_files = {
            "builder": Path(__file__).resolve(),
            "schema": CODE_ROOT / "mixi_understanding" / "data" / "qces_schema.py",
            "validator": CODE_ROOT / "mixi_understanding" / "scripts" / "validate_qces_v3_dataset.py",
        }
        config = {
            "schema_version": SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "seed": args.seed,
            "sample_rate": SAMPLE_RATE,
            "num_channels": 1,
            "num_samples": int(SAMPLE_RATE * DURATION_SECONDS),
            "duration_seconds": DURATION_SECONDS,
            "audio_format": {"container": "WAV", "subtype": "PCM_16"},
            "counts": {"scenes": 4, "questions_per_scene": 4, "records": 16},
            "question_types": [
                "temporal_after",
                "temporal_before",
                "temporal_first",
                "no_evidence_after",
            ],
            "composition": {
                "semantic_labels": list(SEMANTIC_LABELS),
                "nuisance_labels": list(NUISANCE_LABELS),
                "event_duration_seconds": EVENT_DURATION_SECONDS,
                "nuisance_duration_seconds": NUISANCE_DURATION_SECONDS,
                "nuisance_snr_db": NUISANCE_SNR_DB,
                "headroom": HEADROOM,
            },
            "source": {
                "dataset": receipt["dataset"],
                "dataset_revision": receipt["dataset_revision"],
                "receipt_path": relative_posix(audiotime_root / "subset_receipt.json", project_root),
                "receipt_sha256": sha256_file(audiotime_root / "subset_receipt.json"),
                "metadata_path": relative_posix(metadata_path, project_root),
                "metadata_sha256": metadata_hash,
            },
            "runtime_identity": {
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "soundfile_version": sf.__version__,
            },
            "build_identity": {
                role: {
                    "path": relative_posix(path, project_root),
                    "sha256": sha256_file(path),
                }
                for role, path in source_files.items()
            },
        }
        (staging_root / "dataset_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        from mixi_understanding.scripts.validate_qces_v3_dataset import validate_dataset

        report = validate_dataset(staging_root, write_report=True)
        replace_validated_output(staging_root, output_root)
        print(f"dataset ready: {output_root}")
        print(f"artifact fingerprint: {report['artifact_fingerprint_sha256']}")
    except Exception:
        print(f"build failed; staging retained at {staging_root}", file=sys.stderr)
        raise


def main() -> None:
    args = parse_args()
    if args.validate_only:
        from mixi_understanding.scripts.validate_qces_v3_dataset import validate_dataset

        validate_dataset(args.output_root, write_report=True)
    else:
        build_dataset(args)


if __name__ == "__main__":
    main()
