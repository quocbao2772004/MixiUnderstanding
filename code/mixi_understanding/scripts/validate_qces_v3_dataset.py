#!/usr/bin/env python3
"""Validate QCES v3 audio identities, question contrast, stems, and provenance."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf

from mixi_understanding.data.qces_schema import (
    NO_EVIDENCE_ANSWER,
    SCHEMA_VERSION,
    QCESRecord,
    parse_qces_record,
)


PCM_TOLERANCE = 2.5e-4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under(root: Path, relative: str, context: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"{context} escapes root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError(f"row {line_number} must be an object")
            rows.append(payload)
    return rows


def read_wav(path: Path, rate: int, samples: int) -> np.ndarray:
    info = sf.info(path)
    if (
        info.samplerate != rate
        or info.channels != 1
        or info.frames != samples
        or info.subtype != "PCM_16"
    ):
        raise ValueError(f"unexpected WAV format: {path}: {info}")
    waveform, _ = sf.read(path, dtype="float32", always_2d=False)
    if waveform.shape != (samples,) or not np.isfinite(waveform).all():
        raise ValueError(f"invalid waveform: {path}")
    return waveform


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def artifact_identity(root: Path, paths: Iterable[str]) -> tuple[str, Dict[str, str]]:
    digest = hashlib.sha256()
    per_file = {}
    for relative in sorted(set(paths)):
        file_hash = sha256_file(resolve_under(root, relative, "artifact"))
        per_file[relative] = file_hash
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode())
        digest.update(b"\n")
    return digest.hexdigest(), per_file


def _validate_question(record: QCESRecord) -> None:
    if record.no_evidence:
        expected = f"What sound occurs immediately after {record.absent_label}?"
        if record.question != expected or record.answer != NO_EVIDENCE_ANSWER:
            raise AssertionError(f"no-evidence question mismatch: {record.sample_id}")
        return
    anchor = record.event_by_id(record.anchor_event_ids[0])
    answer = record.event_by_id(record.answer_event_ids[0])
    if record.question_type == "temporal_after":
        expected = f"What sound occurs immediately after {anchor.label}?"
        if anchor.offset_seconds >= answer.onset_seconds:
            raise AssertionError("temporal_after events are not strictly ordered")
    elif record.question_type == "temporal_before":
        expected = f"What sound occurs immediately before {anchor.label}?"
        if answer.offset_seconds >= anchor.onset_seconds:
            raise AssertionError("temporal_before events are not strictly ordered")
    elif record.question_type == "temporal_first":
        candidates = {anchor.label, answer.label}
        expected_options = {
            f"Which sound occurs first, {left} or {right}?"
            for left, right in (tuple(candidates), tuple(reversed(tuple(candidates))))
        }
        if record.question not in expected_options:
            raise AssertionError("temporal_first candidate labels mismatch")
        if answer.onset_seconds >= anchor.onset_seconds:
            raise AssertionError("temporal_first answer is not first")
        expected = record.question
    else:
        raise AssertionError(f"unexpected answerable type: {record.question_type}")
    if record.question != expected or record.answer != answer.label:
        raise AssertionError(f"question/answer mismatch: {record.sample_id}")


def validate_dataset(root: Path, write_report: bool = False) -> Dict[str, Any]:
    root = root.resolve()
    project_root = root.parents[1]
    config_path = root / "dataset_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("config schema version mismatch")
    if config.get("counts") != {
        "scenes": 4,
        "questions_per_scene": 4,
        "records": 16,
    }:
        raise AssertionError("config count declaration mismatch")
    rate = int(config["sample_rate"])
    samples = int(config["num_samples"])
    if rate != 32_000 or samples != 256_000:
        raise AssertionError("QCES v3 diagnostic must be 8 seconds at 32 kHz")

    source_config = config["source"]
    receipt_path = resolve_under(
        project_root, source_config["receipt_path"], "source receipt"
    )
    if sha256_file(receipt_path) != source_config["receipt_sha256"]:
        raise AssertionError("source receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("dataset_revision") != source_config["dataset_revision"]:
        raise AssertionError("source revision mismatch")
    receipt_sources = receipt.get("sources", {})
    metadata_path = resolve_under(
        project_root, source_config["metadata_path"], "source metadata"
    )
    if sha256_file(metadata_path) != source_config["metadata_sha256"]:
        raise AssertionError("source metadata hash mismatch")

    for role, identity in config["build_identity"].items():
        source_path = resolve_under(project_root, identity["path"], role)
        if sha256_file(source_path) != identity["sha256"]:
            raise AssertionError(f"build identity mismatch: {role}")

    rows = read_jsonl(root / "qces_overfit16.jsonl")
    if len(rows) != 16:
        raise AssertionError("manifest must contain 16 records")
    records = [parse_qces_record(row) for row in rows]
    by_scene: Dict[str, List[QCESRecord]] = defaultdict(list)
    question_counts: Counter[str] = Counter()
    source_scene: Dict[str, str] = {}
    audio_cache: Dict[str, np.ndarray] = {}
    artifact_paths = {"dataset_config.json", "qces_overfit16.jsonl"}
    maximum_reconstruction_error = 0.0
    maximum_stem_error = 0.0
    maximum_snr_error = 0.0

    def audio(relative: str) -> np.ndarray:
        if relative not in audio_cache:
            path = resolve_under(root, relative, "dataset audio")
            audio_cache[relative] = read_wav(path, rate, samples)
            artifact_paths.add(relative)
        return audio_cache[relative]

    for record in records:
        by_scene[record.scene_id].append(record)
        question_counts[record.question_type] += 1
        _validate_question(record)
        mixture = audio(record.mixture_path)
        evidence = audio(record.evidence_stem_path)
        residual = audio(record.residual_stem_path)
        anchor = audio(record.anchor_stem_path)
        answer = audio(record.answer_stem_path)
        event_stems = {
            event.event_id: audio(event.stem_path) for event in record.events
        }
        expected_mixture = sum(event_stems.values(), np.zeros(samples, np.float32))
        expected_evidence = sum(
            (event_stems[event_id] for event_id in record.evidence_event_ids),
            np.zeros(samples, np.float32),
        )
        expected_anchor = sum(
            (event_stems[event_id] for event_id in record.anchor_event_ids),
            np.zeros(samples, np.float32),
        )
        expected_answer = sum(
            (event_stems[event_id] for event_id in record.answer_event_ids),
            np.zeros(samples, np.float32),
        )
        maximum_stem_error = max(
            maximum_stem_error,
            max_error(mixture, expected_mixture),
            max_error(evidence, expected_evidence),
            max_error(anchor, expected_anchor),
            max_error(answer, expected_answer),
        )
        maximum_reconstruction_error = max(
            maximum_reconstruction_error, max_error(mixture, evidence + residual)
        )
        if record.no_evidence:
            if float(np.max(np.abs(evidence))) != 0.0:
                raise AssertionError("no-evidence target is not silent")
            if max_error(mixture, residual) > PCM_TOLERANCE:
                raise AssertionError("no-evidence residual is not the mixture")
        semantic_active = np.zeros(samples, dtype=bool)
        nuisance = None
        semantic_mix = np.zeros(samples, np.float32)
        for event in record.events:
            start = int(round(event.onset_seconds * rate))
            end = int(round(event.offset_seconds * rate))
            if event.event_kind == "semantic":
                semantic_active[start:end] = True
                semantic_mix += event_stems[event.event_id]
            else:
                nuisance = event_stems[event.event_id]
            receipt_entry = receipt_sources.get(event.source_id)
            source_path = resolve_under(project_root, event.source_path, "event source")
            if receipt_entry is None or sha256_file(source_path) != event.source_sha256:
                raise AssertionError(f"source provenance mismatch: {event.source_id}")
            if receipt_entry["sha256"] != event.source_sha256:
                raise AssertionError(f"receipt provenance mismatch: {event.source_id}")
            owner = source_scene.setdefault(event.source_id, record.scene_id)
            if owner != record.scene_id:
                raise AssertionError(f"cross-scene source reuse: {event.source_id}")
        assert nuisance is not None
        realized_snr = 20.0 * math.log10(
            (rms(semantic_mix[semantic_active]) + 1e-12)
            / (rms(nuisance[semantic_active]) + 1e-12)
        )
        maximum_snr_error = max(
            maximum_snr_error, abs(realized_snr - record.nuisance_snr_db)
        )

    if maximum_stem_error > PCM_TOLERANCE:
        raise AssertionError(f"stem identity error too large: {maximum_stem_error}")
    if maximum_reconstruction_error > PCM_TOLERANCE:
        raise AssertionError(
            f"evidence/residual reconstruction error: {maximum_reconstruction_error}"
        )
    if set(question_counts.values()) != {4} or set(question_counts) != {
        "temporal_after",
        "temporal_before",
        "temporal_first",
        "no_evidence_after",
    }:
        raise AssertionError(f"question balance mismatch: {question_counts}")
    for scene_id, scene_records in by_scene.items():
        if len(scene_records) != 4:
            raise AssertionError(f"{scene_id} does not have four questions")
        if len({record.mixture_path for record in scene_records}) != 1:
            raise AssertionError(f"{scene_id} questions do not share one mixture")
        evidence_hashes = {
            sha256_file(resolve_under(root, record.evidence_stem_path, "evidence"))
            for record in scene_records
        }
        if len(evidence_hashes) != 4:
            raise AssertionError(
                f"{scene_id} does not have four question-specific evidence targets"
            )

    actual_audio_paths = {
        path.relative_to(root).as_posix() for path in (root / "audio").rglob("*.wav")
    }
    if actual_audio_paths != artifact_paths - {"dataset_config.json", "qces_overfit16.jsonl"}:
        raise AssertionError("audio artifact set contains missing or unreferenced files")
    fingerprint, file_hashes = artifact_identity(root, artifact_paths)
    report = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "checked_records": len(records),
        "checked_scenes": len(by_scene),
        "question_type_counts": dict(sorted(question_counts.items())),
        "no_evidence_records": sum(record.no_evidence for record in records),
        "unique_source_count": len(source_scene),
        "cross_scene_source_reuse": False,
        "question_specific_evidence_verified": True,
        "maximum_stem_identity_error": maximum_stem_error,
        "maximum_reconstruction_error": maximum_reconstruction_error,
        "maximum_stored_snr_error_db": maximum_snr_error,
        "source_dataset_revision": source_config["dataset_revision"],
        "artifact_file_count": len(file_hashes),
        "artifact_fingerprint_sha256": fingerprint,
        "artifact_file_sha256": file_hashes,
    }
    if write_report:
        (root / "validation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    validate_dataset(args.dataset_root, write_report=args.write_report)


if __name__ == "__main__":
    main()
