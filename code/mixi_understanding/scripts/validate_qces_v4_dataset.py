#!/usr/bin/env python3
"""Strictly validate QCES v4 structure, audio identities, and provenance."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf

from mixi_understanding.data.qces_v4_schema import (
    NO_EVIDENCE_ANSWER,
    QCESV4Record,
    SCHEMA_VERSION,
    parse_qces_v4_record,
)


PCM_TOLERANCE = 2.5e-4
SNR_TOLERANCE_DB = 0.02
EXPECTED_TYPES = Counter(
    {
        "temporal_after": 3,
        "temporal_before": 3,
        "temporal_first": 6,
        "no_evidence_after": 2,
        "no_evidence_before": 2,
    }
)
SPLITS = ("train", "val", "test")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under(root: Path, relative: str, context: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise ValueError(f"{context} escapes root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(payload, dict):
                raise TypeError(f"{path}:{line_number} must contain an object")
            rows.append(payload)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def read_wav(path: Path, sample_rate: int, num_samples: int) -> np.ndarray:
    info = sf.info(path)
    if (
        info.samplerate != sample_rate
        or info.channels != 1
        or info.frames != num_samples
        or info.subtype != "PCM_16"
    ):
        raise ValueError(f"unexpected WAV format: {path}: {info}")
    waveform, _ = sf.read(path, dtype="float32", always_2d=False)
    if waveform.shape != (num_samples,) or not np.isfinite(waveform).all():
        raise ValueError(f"invalid waveform: {path}")
    return waveform


def max_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
    )


def rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def artifact_identity(root: Path, paths: Iterable[str]) -> tuple[str, Dict[str, str]]:
    digest = hashlib.sha256()
    per_file: Dict[str, str] = {}
    for relative in sorted(set(paths)):
        file_hash = sha256_file(resolve_under(root, relative, "artifact"))
        per_file[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), per_file


def _sum_waveforms(
    waveforms: Mapping[str, np.ndarray], event_ids: Iterable[str], num_samples: int
) -> np.ndarray:
    result = np.zeros(num_samples, dtype=np.float32)
    for event_id in event_ids:
        result += waveforms[event_id]
    return result


def _validate_config(config: Mapping[str, Any]) -> tuple[int, int, Dict[str, int]]:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError("config schema version mismatch")
    if config.get("audio_format") != {"container": "WAV", "subtype": "PCM_16"}:
        raise AssertionError("QCES v4 must declare mono PCM-16 WAV")
    rate = int(config.get("sample_rate", 0))
    samples = int(config.get("num_samples", 0))
    if rate != 32_000 or samples != 256_000 or config.get("num_channels") != 1:
        raise AssertionError("QCES v4 pilot must be mono, 8 seconds at 32 kHz")
    counts = config.get("counts")
    if not isinstance(counts, dict):
        raise AssertionError("missing config counts")
    scenes_by_split = counts.get("scenes_by_split")
    records_by_split = counts.get("records_by_split")
    if not isinstance(scenes_by_split, dict) or not isinstance(records_by_split, dict):
        raise AssertionError("missing split count declarations")
    expected_scenes = {split: int(scenes_by_split.get(split, -1)) for split in SPLITS}
    if expected_scenes != {"train": 6, "val": 3, "test": 3}:
        raise AssertionError(f"unexpected pilot scene counts: {expected_scenes}")
    if counts.get("questions_per_scene") != 16:
        raise AssertionError("QCES v4 pilot requires 16 questions per scene")
    expected_records = {split: expected_scenes[split] * 16 for split in SPLITS}
    if {split: records_by_split.get(split) for split in SPLITS} != expected_records:
        raise AssertionError("record split counts disagree with scene counts")
    if counts.get("scenes") != sum(expected_scenes.values()):
        raise AssertionError("total scene count mismatch")
    if counts.get("records") != sum(expected_records.values()):
        raise AssertionError("total record count mismatch")
    question_design = config.get("question_design", {})
    required_design = {
        "after_per_scene": 3,
        "before_per_scene": 3,
        "first_per_scene": 6,
        "matched_absent_labels_per_scene": 2,
        "matched_absent_questions_per_scene": 4,
        "no_evidence_after_per_scene": 2,
        "no_evidence_before_per_scene": 2,
        "answer_options": 5,
        "first_options_include_both_query_candidates": True,
        "first_answer_query_position": "exactly_3_of_6_per_scene",
        "answer_option_position": "cyclic_balanced_per_split",
        "negative_label_relation_coverage": (
            "every_label_x_after_before_in_train_and_val_plus_test"
        ),
        "counterfactual_question_surface": (
            "identical_for_each_relation_label_across_answerability"
        ),
    }
    for key, value in required_design.items():
        if question_design.get(key) != value:
            raise AssertionError(f"question design mismatch: {key}")
    composition = config.get("composition", {})
    required_composition = {
        "semantic_start_design": "stratified_across_available_timeline_per_split",
        "semantic_edge_margin_seconds": 0.15,
        "nuisance_edge_margin_seconds": 0.05,
    }
    for key, value in required_composition.items():
        if composition.get(key) != value:
            raise AssertionError(f"composition design mismatch: {key}")
    return rate, samples, expected_scenes


def _validate_provenance(
    root: Path, project_root: Path, config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    source = config.get("source")
    if not isinstance(source, dict):
        raise AssertionError("missing source config")
    receipt_path = resolve_under(project_root, source["receipt_path"], "receipt")
    if sha256_file(receipt_path) != source.get("receipt_sha256"):
        raise AssertionError("source receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("format") != "qces_v4_audiotime_subset_v1":
        raise AssertionError("unexpected source receipt format")
    if receipt.get("dataset_revision") != source.get("dataset_revision"):
        raise AssertionError("source revision mismatch")
    if receipt.get("source_count") != 60:
        raise AssertionError("QCES v4 pilot receipt must contain 60 sources")
    receipt_sources = receipt.get("sources")
    if not isinstance(receipt_sources, dict) or len(receipt_sources) != 60:
        raise AssertionError("source receipt payload is incomplete")
    metadata_path = resolve_under(project_root, source["metadata_path"], "metadata")
    if sha256_file(metadata_path) != source.get("metadata_sha256"):
        raise AssertionError("source metadata hash mismatch")
    build_identity = config.get("build_identity")
    if not isinstance(build_identity, dict):
        raise AssertionError("missing build identity")
    if set(build_identity) != {"builder", "schema", "validator", "downloader"}:
        raise AssertionError("build identity roles mismatch")
    for role, identity in build_identity.items():
        path = resolve_under(project_root, identity["path"], f"build {role}")
        if sha256_file(path) != identity.get("sha256"):
            raise AssertionError(f"build identity mismatch: {role}")
    return receipt, receipt_sources


def _validate_question_surface(record: QCESV4Record) -> None:
    normalized = record.question.casefold()
    if any(label.casefold() not in normalized for label in record.query_labels):
        raise AssertionError(f"question omits a query label: {record.sample_id}")
    if record.relation == "first":
        if not set(record.query_labels).issubset(record.answer_options):
            raise AssertionError(
                f"first options omit a named candidate: {record.sample_id}"
            )
    elif not record.no_evidence and record.answer.casefold() in normalized:
        raise AssertionError(
            f"implicit after/before answer appears in question: {record.sample_id}"
        )


def _scene_signature(record: QCESV4Record) -> tuple[Any, ...]:
    return (
        record.split,
        record.sample_rate,
        record.num_samples,
        record.mixture_path,
        record.events,
        record.event_presence_labels,
        record.source_group_ids,
        record.nuisance_snr_db_requested,
        record.nuisance_snr_db,
        record.mixture_peak,
        record.generation_seed,
    )


def validate_dataset(root: Path, write_report: bool = False) -> Dict[str, Any]:
    root = root.resolve()
    project_root = root.parents[1]
    config_path = root / "dataset_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rate, samples, expected_scenes = _validate_config(config)
    _, receipt_sources = _validate_provenance(root, project_root, config)

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    manifest_names = []
    for split in SPLITS:
        name = f"qces_{split}.jsonl"
        manifest_names.append(name)
        rows_by_split[split] = read_jsonl(root / name)
        if len(rows_by_split[split]) != expected_scenes[split] * 16:
            raise AssertionError(f"{split} manifest count mismatch")
        if any(row.get("split") != split for row in rows_by_split[split]):
            raise AssertionError(f"{split} manifest contains another split")
    all_name = "qces_all.jsonl"
    manifest_names.append(all_name)
    all_rows = read_jsonl(root / all_name)
    concatenated = [row for split in SPLITS for row in rows_by_split[split]]
    if all_rows != concatenated:
        raise AssertionError("qces_all.jsonl is not the canonical split concatenation")
    records = [parse_qces_v4_record(row) for row in all_rows]
    if len({record.sample_id for record in records}) != len(records):
        raise AssertionError("duplicate record IDs")

    by_scene: Dict[str, List[QCESV4Record]] = defaultdict(list)
    source_owner: Dict[str, tuple[str, str]] = {}
    source_splits: Dict[str, set[str]] = defaultdict(set)
    source_hash_cache: Dict[str, str] = {}
    audio_cache: Dict[str, np.ndarray] = {}
    artifact_paths = {"dataset_config.json", *manifest_names}
    type_counts: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    answer_position_counts: Counter[int] = Counter()
    first_query_answer_positions: Counter[int] = Counter()
    positive_query_labels: Counter[str] = Counter()
    negative_query_labels: Counter[str] = Counter()
    train_negative_label_relations: set[Tuple[str, str]] = set()
    evaluation_negative_label_relations: set[Tuple[str, str]] = set()
    train_positive_label_relations: set[Tuple[str, str]] = set()
    surface_polarities: Dict[Tuple[str, str], set[bool]] = defaultdict(set)
    label_relation_templates: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    maximum_stem_error = 0.0
    maximum_reconstruction_error = 0.0
    maximum_snr_error = 0.0
    maximum_peak_error = 0.0

    def audio(relative: str) -> np.ndarray:
        if relative not in audio_cache:
            path = resolve_under(root, relative, "dataset audio")
            audio_cache[relative] = read_wav(path, rate, samples)
            artifact_paths.add(relative)
        return audio_cache[relative]

    for record in records:
        by_scene[record.scene_id].append(record)
        type_counts[record.question_type] += 1
        relation_counts[record.relation] += 1
        answer_position_counts[record.answer_option_index] += 1
        _validate_question_surface(record)
        if record.relation in {"after", "before"}:
            surface_polarities[(record.relation, record.question)].add(
                record.no_evidence
            )
            label_relation_templates[
                (record.query_labels[0], record.relation)
            ].add(record.paraphrase_family_id)
        if record.no_evidence:
            negative_query_labels.update(record.query_labels)
            pair = (record.query_labels[0], record.relation)
            if record.split == "train":
                train_negative_label_relations.add(pair)
            else:
                evaluation_negative_label_relations.add(pair)
        else:
            positive_query_labels.update(record.query_labels)
            if record.split == "train" and record.relation in {"after", "before"}:
                train_positive_label_relations.add(
                    (record.query_labels[0], record.relation)
                )
            if record.relation == "first":
                first_query_answer_positions[
                    record.query_labels.index(record.answer)
                ] += 1

        mixture = audio(record.mixture_path)
        evidence = audio(record.evidence_stem_path)
        residual = audio(record.residual_stem_path)
        anchor = audio(record.anchor_stem_path)
        answer = audio(record.answer_stem_path)
        event_stems = {event.event_id: audio(event.stem_path) for event in record.events}
        expected_mixture = _sum_waveforms(event_stems, event_stems, samples)
        expected_evidence = _sum_waveforms(
            event_stems, record.evidence_event_ids, samples
        )
        expected_residual = _sum_waveforms(
            event_stems,
            set(event_stems) - set(record.evidence_event_ids),
            samples,
        )
        expected_anchor = _sum_waveforms(
            event_stems, record.anchor_event_ids, samples
        )
        expected_answer = _sum_waveforms(
            event_stems, record.answer_event_ids, samples
        )
        maximum_stem_error = max(
            maximum_stem_error,
            max_error(mixture, expected_mixture),
            max_error(evidence, expected_evidence),
            max_error(residual, expected_residual),
            max_error(anchor, expected_anchor),
            max_error(answer, expected_answer),
        )
        maximum_reconstruction_error = max(
            maximum_reconstruction_error, max_error(mixture, evidence + residual)
        )
        maximum_peak_error = max(
            maximum_peak_error,
            abs(float(np.max(np.abs(mixture))) - record.mixture_peak),
        )
        if record.no_evidence:
            if float(np.max(np.abs(evidence))) != 0.0:
                raise AssertionError("no-evidence target is not exactly silent")
            if max_error(mixture, residual) > PCM_TOLERANCE:
                raise AssertionError("no-evidence residual is not the mixture")

        semantic_active = np.zeros(samples, dtype=bool)
        semantic_mix = np.zeros(samples, dtype=np.float32)
        nuisance_mix = np.zeros(samples, dtype=np.float32)
        for event in record.events:
            start = int(round(event.onset_seconds * rate))
            end = int(round(event.offset_seconds * rate))
            if event.event_kind == "semantic":
                semantic_active[start:end] = True
                semantic_mix += event_stems[event.event_id]
            else:
                nuisance_mix += event_stems[event.event_id]
            receipt_entry = receipt_sources.get(event.source_id)
            if not isinstance(receipt_entry, dict):
                raise AssertionError(f"source absent from receipt: {event.source_id}")
            if receipt_entry.get("label") != event.label:
                raise AssertionError(f"receipt label mismatch: {event.source_id}")
            source_path = resolve_under(project_root, event.source_path, "event source")
            if event.source_path not in source_hash_cache:
                source_hash_cache[event.source_path] = sha256_file(source_path)
            source_hash = source_hash_cache[event.source_path]
            if source_hash != event.source_sha256:
                raise AssertionError(f"source file hash mismatch: {event.source_id}")
            if receipt_entry.get("sha256") != event.source_sha256:
                raise AssertionError(f"receipt hash mismatch: {event.source_id}")
            source_splits[event.source_id].add(record.split)
            owner = source_owner.setdefault(
                event.source_id, (record.split, record.scene_id)
            )
            if owner != (record.split, record.scene_id):
                raise AssertionError(f"cross-scene source reuse: {event.source_id}")
        realized_snr = 20.0 * math.log10(
            (rms(semantic_mix[semantic_active]) + 1e-12)
            / (rms(nuisance_mix[semantic_active]) + 1e-12)
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
    if maximum_snr_error > SNR_TOLERANCE_DB:
        raise AssertionError(f"stored SNR mismatch: {maximum_snr_error} dB")
    if maximum_peak_error > 1.0 / 32768.0 + 1e-8:
        raise AssertionError(f"stored mixture peak mismatch: {maximum_peak_error}")
    split_overlap = {
        source_id: sorted(splits)
        for source_id, splits in source_splits.items()
        if len(splits) > 1
    }
    if split_overlap:
        raise AssertionError(f"source leakage across splits: {split_overlap}")
    if len(source_owner) != 60:
        raise AssertionError(f"expected 60 unique scene sources, got {len(source_owner)}")

    label_universe = set(config["composition"]["semantic_labels"])
    if set(positive_query_labels) != label_universe:
        raise AssertionError("not every semantic label is positively queried")
    if set(negative_query_labels) != label_universe:
        raise AssertionError("not every semantic label is negatively queried")
    expected_negative_pairs = {
        (label, relation)
        for label in label_universe
        for relation in ("after", "before")
    }
    if train_negative_label_relations != expected_negative_pairs:
        raise AssertionError("train lacks full label x negative-relation coverage")
    if evaluation_negative_label_relations != expected_negative_pairs:
        raise AssertionError("val+test lacks full label x negative-relation coverage")
    if train_positive_label_relations != expected_negative_pairs:
        raise AssertionError("train lacks full label x positive-relation coverage")
    if any(polarities != {False, True} for polarities in surface_polarities.values()):
        raise AssertionError(
            "an after/before question surface does not occur with both polarities"
        )
    if set(label_relation_templates) != expected_negative_pairs or any(
        len(templates) != 1 for templates in label_relation_templates.values()
    ):
        raise AssertionError(
            "relation/label pairs do not use one polarity-invariant paraphrase"
        )

    target_diversity: Dict[str, int] = {}
    for scene_id, scene_records in by_scene.items():
        if len(scene_records) != 16:
            raise AssertionError(f"{scene_id} does not have 16 questions")
        if [record.question_index for record in sorted(
            scene_records, key=lambda item: item.question_index
        )] != list(range(16)):
            raise AssertionError(f"{scene_id} question indices are not 0..15")
        if len({_scene_signature(record) for record in scene_records}) != 1:
            raise AssertionError(f"{scene_id} records do not share one scene")
        per_scene_types = Counter(record.question_type for record in scene_records)
        if per_scene_types != EXPECTED_TYPES:
            raise AssertionError(f"{scene_id} question balance mismatch: {per_scene_types}")
        first_position_counts = Counter(
            record.query_labels.index(record.answer)
            for record in scene_records
            if record.relation == "first"
        )
        if first_position_counts != Counter({0: 3, 1: 3}):
            raise AssertionError(
                f"{scene_id} first candidate order is not 3/3 balanced"
            )
        answerable = [record for record in scene_records if not record.no_evidence]
        evidence_hashes = {
            sha256_file(resolve_under(root, record.evidence_stem_path, "evidence"))
            for record in answerable
        }
        if len(evidence_hashes) != 6:
            raise AssertionError(
                f"{scene_id} needs six pair-specific evidence targets, got {len(evidence_hashes)}"
            )
        target_diversity[scene_id] = len(evidence_hashes)
        for relation, expected in (("after", 3), ("before", 3), ("first", 6)):
            hashes = {
                sha256_file(resolve_under(root, record.evidence_stem_path, "evidence"))
                for record in answerable
                if record.relation == relation
            }
            if len(hashes) != expected:
                raise AssertionError(
                    f"{scene_id} {relation} target diversity is {len(hashes)}, expected {expected}"
                )

    expected_type_counts = Counter(
        {name: count * len(by_scene) for name, count in EXPECTED_TYPES.items()}
    )
    if type_counts != expected_type_counts:
        raise AssertionError(f"global question type mismatch: {type_counts}")
    if {record.split for record in records} != set(SPLITS):
        raise AssertionError("missing split")
    if {split: len({record.scene_id for record in records if record.split == split})
        for split in SPLITS} != expected_scenes:
        raise AssertionError("actual split scene counts disagree with config")
    for split in SPLITS:
        split_positions = Counter(
            record.answer_option_index for record in records if record.split == split
        )
        if max(split_positions.values()) - min(split_positions.values()) > 1:
            raise AssertionError(
                f"{split} answer-option positions are not cyclically balanced: "
                f"{split_positions}"
            )

    actual_audio_paths = {
        path.relative_to(root).as_posix() for path in (root / "audio").rglob("*.wav")
    }
    referenced_audio_paths = artifact_paths - {"dataset_config.json", *manifest_names}
    if actual_audio_paths != referenced_audio_paths:
        missing = sorted(referenced_audio_paths - actual_audio_paths)
        unreferenced = sorted(actual_audio_paths - referenced_audio_paths)
        raise AssertionError(
            f"audio artifact set mismatch: missing={missing}, unreferenced={unreferenced}"
        )
    fingerprint, file_hashes = artifact_identity(root, artifact_paths)
    first_total = sum(first_query_answer_positions.values())
    option_deviation = max(
        abs(answer_position_counts[index] / len(records) - 0.2)
        for index in range(5)
    )
    first_bias = abs(first_query_answer_positions[0] / first_total - 0.5)
    report: Dict[str, Any] = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "checked_records": len(records),
        "checked_scenes": len(by_scene),
        "records_by_split": {
            split: sum(record.split == split for record in records) for split in SPLITS
        },
        "question_type_counts": dict(sorted(type_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "no_evidence_records": sum(record.no_evidence for record in records),
        "unique_source_count_↑": len(source_owner),
        "cross_split_source_overlap_count_↓": len(split_overlap),
        "minimum_positive_queries_per_label_↑": min(positive_query_labels.values()),
        "minimum_negative_queries_per_label_↑": min(negative_query_labels.values()),
        "pair_target_diversity_per_scene_↑": target_diversity,
        "answer_option_position_counts": dict(sorted(answer_position_counts.items())),
        "answer_option_position_max_deviation_from_uniform_↓": option_deviation,
        "first_answer_query_position_counts": dict(
            sorted(first_query_answer_positions.items())
        ),
        "first_answer_query_position_absolute_bias_↓": first_bias,
        "maximum_stem_identity_error_↓": maximum_stem_error,
        "maximum_reconstruction_error_↓": maximum_reconstruction_error,
        "maximum_stored_snr_error_db_↓": maximum_snr_error,
        "maximum_stored_peak_error_↓": maximum_peak_error,
        "source_dataset_revision": config["source"]["dataset_revision"],
        "artifact_file_count": len(file_hashes),
        "artifact_fingerprint_sha256": fingerprint,
        "artifact_file_sha256": file_hashes,
    }
    if write_report:
        (root / "validation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    console_report = dict(report)
    console_report["artifact_file_sha256"] = (
        f"{len(file_hashes)} per-file hashes retained in validation_report.json"
        if write_report
        else f"{len(file_hashes)} per-file hashes omitted from console"
    )
    print(json.dumps(console_report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    validate_dataset(args.dataset_root, write_report=args.write_report)


if __name__ == "__main__":
    main()
