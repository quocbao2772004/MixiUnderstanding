#!/usr/bin/env python3
"""Build the QCES RealDESED real-world development benchmark.

The builder creates one deterministic 10-second crop per eligible reviewed
recording and then derives relational questions from event onset order.  It
keeps label-free inference rows separate from post-hoc scoring rows.  It does
not create oracle waveforms, source stems, or SDR targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.data.qces_realdesed import (  # noqa: E402
    DATASET_NAME,
    DATASET_RECORD_ID,
    DISPLAY_NAME,
    DURATION_SECONDS,
    Event,
    NUM_CHANNELS,
    NUM_SAMPLES,
    REALDESED_CLASSES,
    SAMPLE_RATE,
    SCENE_SCHEMA_VERSION,
    build_question_records,
    canonical_json_sha256,
    choose_ten_second_crop,
    merge_reviewed_events,
    validate_inference_record,
    validate_scoring_record,
)


BUILD_SCHEMA_VERSION = "qces_realdesed_build_report_v1"
class BuildError(RuntimeError):
    """Raised when source data violates the declared derivation contract."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise BuildError(f"CSV is empty: {path}")
    return rows


def _sha256(path: Path, chunk_bytes: int = 4 << 20) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )


def _normalize_license(value: str) -> str:
    """Validate but preserve the publisher's complete attribution string."""

    exact = value.strip()
    normalized = " ".join(exact.casefold().replace("_", " ").split())
    is_cc0 = normalized.startswith("cc0 (") or normalized in {
        "cc0",
        "cc0 1.0",
        "cc0-1.0",
    }
    is_cc_by = normalized.startswith("cc-by 4.0 (attribution license), author:")
    if not (is_cc0 or is_cc_by):
        raise BuildError(f"unsupported or missing per-file license: {value!r}")
    if is_cc_by and not normalized.split("author:", maxsplit=1)[1].strip():
        raise BuildError(f"CC-BY attribution has no author: {value!r}")
    return exact


def _decode_mono_pcm16(path: Path, ffmpeg: str) -> bytes:
    command = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-vn",
        "-ac",
        str(NUM_CHANNELS),
        "-ar",
        str(SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]
    result = subprocess.run(command, check=False, capture_output=True)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise BuildError(f"ffmpeg decode failed for {path}: {message}")
    if len(result.stdout) % 2:
        raise BuildError(f"decoded PCM byte count is odd for {path}")
    return result.stdout


def _write_pcm16_wav(path: Path, pcm: bytes) -> None:
    if len(pcm) != NUM_SAMPLES * 2:
        raise BuildError("canonical PCM must contain exactly 320000 int16 samples")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(NUM_CHANNELS)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm)


def _pcm_values(pcm: bytes) -> tuple[int, ...]:
    return struct.unpack(f"<{len(pcm) // 2}h", pcm)


def _rms_dbfs(values: Sequence[int]) -> float:
    if not values:
        return -120.0
    mean_square = sum(float(value) * float(value) for value in values) / len(values)
    if mean_square <= 0:
        return -120.0
    return 20.0 * math.log10(math.sqrt(mean_square) / 32768.0)


def _event_acoustic_diagnostic(
    values: Sequence[int], event: Event
) -> dict[str, float]:
    start = max(0, min(NUM_SAMPLES, round(event.onset * SAMPLE_RATE)))
    end = max(start + 1, min(NUM_SAMPLES, round(event.offset * SAMPLE_RATE)))
    interval = values[start:end]
    context = (*values[:start], *values[end:])
    interval_db = _rms_dbfs(interval)
    context_db = _rms_dbfs(context)
    peak = max((abs(value) for value in interval), default=0) / 32768.0
    return {
        "interval_rms_dbfs_↑": round(interval_db, 4),
        "interval_to_context_rms_db_↑": round(interval_db - context_db, 4),
        "interval_peak_linear_↑": round(peak, 6),
    }


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def _scene_id(source_split: str, filename: str) -> str:
    digest = hashlib.sha256(
        f"RealDESED:{source_split}:{filename}".encode("utf-8")
    ).hexdigest()[:16]
    prefix = "val" if source_split == "validation" else "test"
    return f"realdesed_{prefix}_{digest}"


def rebalance_answer_option_positions(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Place gold answers uniformly over A--E without changing distractors."""

    balanced = []
    for global_index, original in enumerate(rows):
        row = dict(original)
        answer = row["answer"]
        distractors = [option for option in row["answer_options"] if option != answer]
        if len(distractors) != 4:
            raise BuildError(f"{row.get('id')}: invalid option set before balancing")
        answer_index = global_index % 5
        distractors.insert(answer_index, answer)
        row["answer_options"] = distractors
        row["answer_option_index"] = answer_index
        balanced.append(row)
    return balanced


def _annotation_events(
    rows: Sequence[Mapping[str, str]], *, filename: str
) -> tuple[Event, ...]:
    parsed = []
    for row in rows:
        label = row.get("class", "").strip()
        if label not in REALDESED_CLASSES:
            raise BuildError(f"{filename}: unknown annotation class {label!r}")
        try:
            onset = float(row["onset"])
            offset = float(row["offset"])
        except (KeyError, TypeError, ValueError) as error:
            raise BuildError(f"{filename}: malformed annotation interval") from error
        parsed.append(Event(onset=onset, offset=offset, label=label))
    return merge_reviewed_events(parsed)


def _validate_source_layout(source_dir: Path, source_split: str) -> None:
    required = (
        source_dir / "audio",
        source_dir / "metadata.csv",
        source_dir / "annotations.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise BuildError(f"source split is incomplete: {missing}")
    if source_split == "train":
        raise BuildError("RealDESED train is not a QCES real-transfer evaluation split")


def build(
    *,
    source_dir: Path,
    source_split: str,
    output_dir: Path,
    acquisition_receipt: Path,
    ffmpeg: str,
    max_scenes: int | None,
) -> dict[str, Any]:
    if source_split not in {"validation", "test"}:
        raise ValueError("source_split must be validation or test")
    if source_split == "test":
        raise BuildError(
            "real-test derivation is intentionally locked until the method-freeze "
            "receipt workflow is implemented and activated"
        )
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    _validate_source_layout(source_dir, source_split)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {output_dir}")
    acquisition = json.loads(acquisition_receipt.read_text(encoding="utf-8"))
    if (
        acquisition.get("dataset") != DATASET_NAME
        or acquisition.get("dataset_record_id") != DATASET_RECORD_ID
        or acquisition.get("split") != source_split
        or acquisition.get("archive", {}).get("publisher_md5")
        != acquisition.get("archive", {}).get("verified_md5")
    ):
        raise BuildError("acquisition receipt is missing, mismatched, or unverified")

    metadata_rows = _read_csv(source_dir / "metadata.csv")
    annotation_rows = _read_csv(source_dir / "annotations.csv")
    metadata_by_file: dict[str, dict[str, str]] = {}
    for row in metadata_rows:
        filename = row.get("filename", "").strip()
        if not filename or filename in metadata_by_file:
            raise BuildError(f"missing or duplicate metadata filename: {filename!r}")
        metadata_by_file[filename] = row
    annotations_by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in annotation_rows:
        filename = row.get("filename", "").strip()
        if filename not in metadata_by_file:
            raise BuildError(f"annotation filename absent from metadata: {filename!r}")
        annotations_by_file[filename].append(row)

    stage_parent = output_dir.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=output_dir.name + ".staging.", dir=stage_parent))
    scenes = []
    inference_rows = []
    scoring_rows = []
    exclusions = Counter()
    question_counts = Counter()
    class_counts = Counter()
    license_counts = Counter()
    try:
        selected_filenames = sorted(metadata_by_file)
        if max_scenes is not None:
            selected_filenames = selected_filenames[:max_scenes]
        for filename in selected_filenames:
            metadata = metadata_by_file[filename]
            source_audio = source_dir / "audio" / filename
            if not source_audio.is_file():
                raise BuildError(f"metadata audio is missing: {source_audio}")
            if filename not in annotations_by_file:
                exclusions["no_reviewed_annotations_↓"] += 1
                continue
            source_license = _normalize_license(metadata.get("license", ""))
            events = _annotation_events(annotations_by_file[filename], filename=filename)
            pcm = _decode_mono_pcm16(source_audio, ffmpeg)
            source_num_samples = len(pcm) // 2
            source_duration = source_num_samples / SAMPLE_RATE
            if source_duration < DURATION_SECONDS:
                exclusions["shorter_than_10s_↓"] += 1
                continue
            for event in events:
                if event.offset > source_duration + 1 / SAMPLE_RATE:
                    raise BuildError(
                        f"{filename}: event {event.event_id} exceeds decoded audio duration"
                    )
            crop_start, crop_events = choose_ten_second_crop(
                events,
                source_duration_seconds=source_duration,
            )
            source_event_by_id = {event.event_id: event for event in events}
            unique_crop_labels = {event.label for event in crop_events}
            if len(unique_crop_labels) < 2:
                exclusions["fewer_than_2_distinct_events_in_crop_↓"] += 1
                continue
            crop_start_sample = round(crop_start * SAMPLE_RATE)
            crop_end_sample = crop_start_sample + NUM_SAMPLES
            crop_pcm = pcm[crop_start_sample * 2 : crop_end_sample * 2]
            if len(crop_pcm) != NUM_SAMPLES * 2:
                raise BuildError(f"{filename}: deterministic crop is not exactly 10 seconds")
            scene_id = _scene_id(source_split, filename)
            relative_audio_path = f"audio/{scene_id}.wav"
            canonical_audio = stage_dir / relative_audio_path
            _write_pcm16_wav(canonical_audio, crop_pcm)
            canonical_sha256 = _sha256(canonical_audio)
            values = _pcm_values(crop_pcm)

            inference, scoring = build_question_records(
                scene_id=scene_id,
                upstream_record_id=filename,
                upstream_split=source_split,
                mixture_path=relative_audio_path,
                mixture_sha256=canonical_sha256,
                source_license=source_license,
                events=crop_events,
            )
            if not inference:
                canonical_audio.unlink()
                exclusions["no_unambiguous_relational_questions_↓"] += 1
                continue
            license_counts[
                "CC0"
                if source_license.casefold().startswith("cc0")
                else "CC-BY-4.0"
            ] += 1
            inference_rows.extend(inference)
            scoring_rows.extend(scoring)
            question_counts.update(row["relation"] for row in inference)
            class_counts.update(event.label for event in crop_events)
            scenes.append(
                {
                    "schema_version": SCENE_SCHEMA_VERSION,
                    "scene_id": scene_id,
                    "dataset": DATASET_NAME,
                    "dataset_record_id": DATASET_RECORD_ID,
                    "upstream_split": source_split,
                    "upstream_record_id": filename,
                    "source_audio_path": f"raw/{source_split}/audio/{filename}",
                    "source_audio_sha256": _sha256(source_audio),
                    "source_license": source_license,
                    "source_duration_seconds": round(source_duration, 6),
                    "crop_start_seconds": crop_start,
                    "crop_end_seconds": round(crop_start + DURATION_SECONDS, 6),
                    "canonical_audio_path": relative_audio_path,
                    "canonical_audio_sha256": canonical_sha256,
                    "sample_rate": SAMPLE_RATE,
                    "num_channels": NUM_CHANNELS,
                    "num_samples": NUM_SAMPLES,
                    "metadata": {
                        "target_classes": _split_list(
                            metadata.get("target_classes", "")
                        ),
                        "non_target_classes": _split_list(
                            metadata.get("non_target_classes", "")
                        ),
                        "recording_device": metadata.get(
                            "recording_device", ""
                        ).strip(),
                        "device_placement": metadata.get(
                            "device_placement", ""
                        ).strip(),
                        "recording_environment": _split_list(
                            metadata.get("recording_environment", "")
                        ),
                        "scene_description": metadata.get(
                            "scene_description", ""
                        ).strip(),
                    },
                    "events": [
                        {
                            "event_id": event.event_id,
                            "class": event.label,
                            "display_name": DISPLAY_NAME[event.label],
                            "relative_interval": event.interval,
                            "absolute_interval": [
                                round(event.onset + crop_start, 6),
                                round(event.offset + crop_start, 6),
                            ],
                            "source_annotation_interval": source_event_by_id[
                                event.event_id
                            ].interval,
                            "right_censored_by_crop": (
                                source_event_by_id[event.event_id].offset
                                > crop_start + event.offset + 2e-6
                            ),
                            "acoustic_diagnostic_not_semantic_proof": (
                                _event_acoustic_diagnostic(values, event)
                            ),
                        }
                        for event in crop_events
                    ],
                    "clean_reference_stems_available": False,
                    "waveform_sdr_evaluation_allowed": False,
                }
            )

        if not scenes:
            raise BuildError("no eligible real scenes were produced")
        scoring_rows = rebalance_answer_option_positions(scoring_rows)
        for row in inference_rows:
            validate_inference_record(row)
        for row in scoring_rows:
            validate_scoring_record(row)
        if len({row["id"] for row in inference_rows}) != len(inference_rows):
            raise BuildError("duplicate question IDs")
        if len({scene["scene_id"] for scene in scenes}) != len(scenes):
            raise BuildError("duplicate scene IDs")

        _write_jsonl(stage_dir / "scenes.scoring.jsonl", scenes)
        _write_jsonl(
            stage_dir / "ATTRIBUTION.jsonl",
            (
                {
                    "dataset": DATASET_NAME,
                    "dataset_record_id": DATASET_RECORD_ID,
                    "upstream_record_id": scene["upstream_record_id"],
                    "source_audio_sha256": scene["source_audio_sha256"],
                    "source_license_exact": scene["source_license"],
                }
                for scene in scenes
            ),
        )
        _write_jsonl(stage_dir / "qces_realdesed_inference.jsonl", inference_rows)
        _write_jsonl(stage_dir / "qces_realdesed_scoring.jsonl", scoring_rows)
        ffmpeg_version = subprocess.run(
            [ffmpeg, "-version"], check=True, capture_output=True, text=True
        ).stdout.splitlines()[0]
        report = {
            "schema_version": BUILD_SCHEMA_VERSION,
            "dataset": DATASET_NAME,
            "dataset_record_id": DATASET_RECORD_ID,
            "source_split": source_split,
            "qces_split": "real_dev",
            "source_dir": str(source_dir),
            "acquisition_receipt": str(acquisition_receipt.resolve()),
            "acquisition_receipt_sha256": _sha256(acquisition_receipt),
            "derivation": {
                "canonical_audio": "mono PCM16, 32 kHz, exactly 10.0 seconds",
                "crop_selection": (
                    "label-only deterministic event-rich window; no waveform or "
                    "model-score selection"
                ),
                "event_policy": (
                    "reviewed annotations; exact duplicates removed; touching "
                    "same-class regions merged at <=80 ms; event onset must lie "
                    "inside crop with 50 ms boundary margin; event offset may be "
                    "right-censored at 9.95 s and is declared per event; "
                    "left-censored events are forbidden"
                ),
                "question_policy": (
                    "only labels unique within the crop can anchor relational QA"
                ),
                "ffmpeg_version": ffmpeg_version,
            },
            "counts": {
                "metadata_recordings_inspected_↑": len(selected_filenames),
                "eligible_scenes_↑": len(scenes),
                "questions_↑": len(inference_rows),
                "answerable_questions_↑": sum(
                    not row["no_evidence"] for row in scoring_rows
                ),
                "no_evidence_questions_↑": sum(
                    row["no_evidence"] for row in scoring_rows
                ),
                "questions_by_relation": dict(sorted(question_counts.items())),
                "retained_events_by_class": dict(sorted(class_counts.items())),
                "eligible_scenes_by_license": dict(sorted(license_counts.items())),
                "exclusions": dict(sorted(exclusions.items())),
            },
            "claim_boundary": {
                "real_recordings": True,
                "reviewed_strong_temporal_annotations": True,
                "clean_reference_stems_available": False,
                "waveform_sdr_evaluation_allowed": False,
                "allowed_primary_metrics": [
                    "temporal IoU ↑",
                    "QA sufficiency ↑",
                    "residual answer leakage ↓",
                    "retained duration ↓",
                    "no-evidence AUROC ↑",
                ],
            },
        }
        report["inference_manifest_sha256"] = _sha256(
            stage_dir / "qces_realdesed_inference.jsonl"
        )
        report["scoring_manifest_sha256"] = _sha256(
            stage_dir / "qces_realdesed_scoring.jsonl"
        )
        report["scene_manifest_sha256"] = _sha256(stage_dir / "scenes.scoring.jsonl")
        report["attribution_manifest_sha256"] = _sha256(
            stage_dir / "ATTRIBUTION.jsonl"
        )
        report["dataset_contract_sha256"] = canonical_json_sha256(report)
        (stage_dir / "build_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stage_dir.replace(output_dir)
        return report
    except BaseException:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument(
        "--source-split", choices=("validation", "test"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--max-scenes", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report = build(
            source_dir=args.source_dir,
            source_split=args.source_split,
            output_dir=args.output_dir,
            acquisition_receipt=args.acquisition_receipt,
            ffmpeg=args.ffmpeg,
            max_scenes=args.max_scenes,
        )
    except (
        BuildError,
        FileExistsError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        raise SystemExit(f"QCES RealDESED build failed: {error}") from error
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
