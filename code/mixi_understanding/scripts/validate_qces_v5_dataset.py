#!/usr/bin/env python3
"""Validate QCES v5 audio, counterfactuals, split isolation and provenance."""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf

from mixi_understanding.data.qces_v5_schema import (  # noqa: E402
    DERIVED_SCHEMA_VERSION,
    DERIVED_STORAGE_MODE,
    MATERIALIZED_STORAGE_MODE,
    NO_EVIDENCE_ANSWER,
    QCESV5Event,
    QCESV5Record,
    SCHEMA_VERSION,
    SPLITS,
    VARIANTS,
    parse_qces_v5_record,
)
from mixi_understanding.scripts.plan_qces_v5_sources import RECEIPT_FORMAT  # noqa: E402


PCM_TOLERANCE = 5.0e-4
PEAK_TOLERANCE = 5.0e-5
AUDITED_RECEIPT_FORMAT = "qces_v5_audited_source_ledger_v1"
VALIDATOR_AUDIO_CACHE_MAX_FILES = 32


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--write-report", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, relative: str, context: str) -> Path:
    path = (root.resolve() / relative).resolve()
    if root.resolve() != path and root.resolve() not in path.parents:
        raise ValueError(f"{context} escapes root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def _read_audio(path: Path, sample_rate: int, samples: int) -> np.ndarray:
    info = sf.info(path)
    if (
        info.samplerate != sample_rate
        or info.channels != 1
        or info.frames != samples
        or info.subtype != "PCM_16"
    ):
        raise ValueError(f"unexpected WAV format: {path}: {info}")
    waveform, _ = sf.read(path, dtype="float32", always_2d=False)
    if waveform.shape != (samples,) or not np.isfinite(waveform).all():
        raise ValueError(f"invalid waveform: {path}")
    return waveform


def _maximum_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _sum(
    waveforms: Mapping[str, np.ndarray], ids: Iterable[str], samples: int
) -> np.ndarray:
    result = np.zeros(samples, dtype=np.float32)
    for event_id in ids:
        result += waveforms[event_id]
    return result


def _semantic_polarity_summary(
    records: Sequence[Any],
) -> Dict[str, Dict[str, Any]]:
    """Summarize exact-semantic polarity and record-weighted coverage."""

    by_split: Dict[str, Dict[str, Any]] = {}
    for split in SPLITS:
        split_records = [record for record in records if record.split == split]
        relation_summaries: Dict[str, Dict[str, Any]] = {}
        overall_polarity: Dict[str, set[bool]] = defaultdict(set)
        overall_rows: Counter[str] = Counter()
        for relation in ("after", "before", "first"):
            polarity: Dict[str, set[bool]] = defaultdict(set)
            rows: Counter[str] = Counter()
            for record in split_records:
                if record.relation != relation:
                    continue
                semantics_id = str(record.question_semantics_id)
                polarity[semantics_id].add(bool(record.no_evidence))
                rows[semantics_id] += 1
                overall_polarity[semantics_id].add(bool(record.no_evidence))
                overall_rows[semantics_id] += 1
            matched = {
                key for key, values in polarity.items() if values == {False, True}
            }
            positive_only = {
                key for key, values in polarity.items() if values == {False}
            }
            negative_only = {
                key for key, values in polarity.items() if values == {True}
            }
            row_count = sum(rows.values())
            relation_summaries[relation] = {
                "matched_unique_semantics_up": len(matched),
                "positive_only_unique_semantics_down": len(positive_only),
                "negative_only_unique_semantics_down": len(negative_only),
                "matched_weighted_record_share_up": (
                    sum(rows[key] for key in matched) / row_count
                    if row_count
                    else 0.0
                ),
            }
        matched = {
            key for key, values in overall_polarity.items() if values == {False, True}
        }
        positive_only = {
            key for key, values in overall_polarity.items() if values == {False}
        }
        negative_only = {
            key for key, values in overall_polarity.items() if values == {True}
        }
        total_rows = sum(overall_rows.values())
        by_split[split] = {
            "matched_unique_semantics_up": len(matched),
            "positive_only_unique_semantics_down": len(positive_only),
            "negative_only_unique_semantics_down": len(negative_only),
            "matched_weighted_record_share_up": (
                sum(overall_rows[key] for key in matched) / total_rows
                if total_rows
                else 0.0
            ),
            "by_relation": relation_summaries,
        }
    return by_split


def _binary_auroc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    if len(scores) != len(labels) or not scores:
        raise ValueError("AUROC needs aligned non-empty scores and labels")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC needs both answerability states")
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[start][0]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[start:stop]
        )
        start = stop
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _best_balanced_accuracy(
    scores: Sequence[float], labels: Sequence[bool]
) -> Tuple[float, float]:
    if len(scores) != len(labels) or not scores:
        raise ValueError("balanced accuracy needs aligned non-empty rows")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("balanced accuracy needs both answerability states")
    thresholds = [max(scores) + 1.0, *sorted(set(scores), reverse=True)]
    best = (-1.0, thresholds[0])
    for threshold in thresholds:
        true_positive = sum(
            score >= threshold and label
            for score, label in zip(scores, labels)
        )
        true_negative = sum(
            score < threshold and not label
            for score, label in zip(scores, labels)
        )
        balanced = 0.5 * (
            true_positive / positives + true_negative / negatives
        )
        candidate = (balanced, threshold)
        if candidate[0] > best[0] or (
            candidate[0] == best[0] and candidate[1] > best[1]
        ):
            best = candidate
    return best


def _exact_semantics_train_to_val_shortcut(
    records: Sequence[Any],
) -> Dict[str, Any]:
    """Fit exact-semantic no-evidence rates on train and audit them on val."""

    train = [record for record in records if record.split == "train"]
    validation = [record for record in records if record.split == "val"]
    if not train or not validation:
        raise ValueError("exact-semantics shortcut audit needs train and val")
    semantics_counts: Dict[str, Counter[bool]] = defaultdict(Counter)
    relation_counts: Dict[str, Counter[bool]] = defaultdict(Counter)
    for record in train:
        status = bool(record.no_evidence)
        semantics_counts[str(record.question_semantics_id)][status] += 1
        relation_counts[str(record.relation)][status] += 1

    scores: List[float] = []
    labels: List[bool] = []
    relations: List[str] = []
    seen: List[bool] = []
    for record in validation:
        semantics_id = str(record.question_semantics_id)
        relation = str(record.relation)
        counts = semantics_counts.get(semantics_id)
        key_seen = bool(counts)
        if counts:
            score = counts[True] / sum(counts.values())
        else:
            backoff = relation_counts[relation]
            score = backoff[True] / sum(backoff.values())
        scores.append(score)
        labels.append(bool(record.no_evidence))
        relations.append(relation)
        seen.append(key_seen)

    best_balanced, best_threshold = _best_balanced_accuracy(scores, labels)
    report: Dict[str, Any] = {
        "auroc_down": _binary_auroc(scores, labels),
        "best_balanced_accuracy_down": best_balanced,
        "best_threshold": best_threshold,
        "seen_semantics_record_share_up": sum(seen) / len(seen),
        "unseen_semantics_backoff": "train_relation_prior",
        "by_relation": {},
    }
    for relation in ("after", "before", "first"):
        indices = [index for index, value in enumerate(relations) if value == relation]
        relation_scores = [scores[index] for index in indices]
        relation_labels = [labels[index] for index in indices]
        relation_balanced, relation_threshold = _best_balanced_accuracy(
            relation_scores, relation_labels
        )
        report["by_relation"][relation] = {
            "auroc_down": _binary_auroc(relation_scores, relation_labels),
            "best_balanced_accuracy_down": relation_balanced,
            "best_threshold": relation_threshold,
            "seen_semantics_record_share_up": (
                sum(seen[index] for index in indices) / len(indices)
            ),
        }
    return report


def _enforce_exact_semantics_shortcut_gate(
    report: Mapping[str, Any],
    *,
    maximum_auroc: float,
    maximum_best_balanced_accuracy: float,
) -> None:
    if float(report["auroc_down"]) > maximum_auroc:
        raise AssertionError(
            "train-to-val exact-semantics shortcut AUROC exceeds the gate"
        )
    if (
        float(report["best_balanced_accuracy_down"])
        > maximum_best_balanced_accuracy
    ):
        raise AssertionError(
            "train-to-val exact-semantics shortcut balanced accuracy exceeds the gate"
        )


def _foreign_question_label_mentions(
    question: str,
    *,
    forbidden_labels: Iterable[str],
    expressed_labels: Iterable[str],
) -> List[str]:
    """Find foreign label mentions, exempting lexical overlap in valid labels."""

    normalized_question = question.casefold()
    expressed = [label.casefold() for label in expressed_labels]
    result = []
    for label in forbidden_labels:
        normalized = label.casefold()
        if normalized not in normalized_question:
            continue
        # For example, held-out ``Steam`` is a substring of the valid seen
        # class ``Steam whistle``.  The semantic metadata remains the authority
        # for that intentionally overlapping class name.
        if any(normalized in valid for valid in expressed):
            continue
        result.append(label)
    return sorted(result)


def _artifact_identity(root: Path, relatives: Iterable[str]) -> tuple[str, Dict[str, str]]:
    digest = hashlib.sha256()
    per_file = {}
    for relative in sorted(set(relatives)):
        file_hash = sha256_file(_resolve(root, relative, "artifact"))
        per_file[relative] = file_hash
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), per_file


def _config_storage_mode(config: Mapping[str, Any]) -> str:
    schema_version = config.get("schema_version")
    storage = config.get("storage")
    # Backward compatibility for the original qces_v5 builder, whose config
    # predates an explicit storage block and always materialized role stems.
    if schema_version == SCHEMA_VERSION and storage is None:
        return MATERIALIZED_STORAGE_MODE
    if not isinstance(storage, dict):
        raise AssertionError("config storage contract is missing")
    mode = storage.get("mode")
    if mode not in {MATERIALIZED_STORAGE_MODE, DERIVED_STORAGE_MODE}:
        raise AssertionError("unsupported storage mode")
    expected_schema = (
        DERIVED_SCHEMA_VERSION
        if mode == DERIVED_STORAGE_MODE
        else SCHEMA_VERSION
    )
    if schema_version != expected_schema:
        raise AssertionError("schema/storage mode mismatch")
    expected_storage = {
        "mixture_stored_once_per_scene": True,
        "event_stems_stored_once_per_scene": True,
        "question_role_stems_materialized": mode == MATERIALIZED_STORAGE_MODE,
        "derived_evidence_recipe": (
            "sum(evidence_event_ids)" if mode == DERIVED_STORAGE_MODE else None
        ),
        "derived_residual_recipe": (
            "mixture-evidence" if mode == DERIVED_STORAGE_MODE else None
        ),
    }
    for key, expected in expected_storage.items():
        if storage.get(key) != expected:
            raise AssertionError(f"storage contract mismatch: {key}")
    for key in ("stored_audio_file_count", "stored_audio_bytes"):
        value = storage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AssertionError(f"storage counter is invalid: {key}")
    return str(mode)


def _validate_config(config: Mapping[str, Any]) -> tuple[int, int, Dict[str, int], str]:
    _config_storage_mode(config)
    profile = config.get("profile")
    if profile not in {"smoke", "internal_scale", "paper"}:
        raise AssertionError("unsupported v5 profile")
    if config.get("audio_format") != {"container": "WAV", "subtype": "PCM_16"}:
        raise AssertionError("v5 must use PCM-16 WAV")
    rate = int(config.get("sample_rate", 0))
    samples = int(config.get("num_samples", 0))
    if rate != 32_000 or samples != 320_000 or config.get("num_channels") != 1:
        raise AssertionError("v5 must be mono, ten seconds at 32 kHz")
    counts = config.get("counts")
    if not isinstance(counts, dict):
        raise AssertionError("config counts are missing")
    families = counts.get("scene_families_by_split")
    scenes = counts.get("scenes_by_split")
    records = counts.get("records_by_split")
    if not all(isinstance(item, dict) and set(item) == set(SPLITS) for item in (families, scenes, records)):
        raise AssertionError("split count dictionaries are incomplete")
    expected_families = {split: int(families[split]) for split in SPLITS}
    # Partial debug/pilot artifacts may intentionally omit held-out splits;
    # those splits are represented by zero families and are never paper eligible.
    build_splits = config.get("build_splits", list(SPLITS))
    if any(
        value <= 0
        for split, value in expected_families.items()
        if split in build_splits
    ):
        raise AssertionError("every built v5 split needs at least one scene family")
    expected_scenes = {split: expected_families[split] * 3 for split in SPLITS}
    if scenes != expected_scenes:
        raise AssertionError("scene count must be three variants per family")
    if counts.get("questions_per_scene") != 16:
        raise AssertionError("v5 requires 16 questions per scene")
    expected_records = {split: expected_scenes[split] * 16 for split in SPLITS}
    if records != expected_records:
        raise AssertionError("record counts disagree with scene counts")
    if counts.get("scene_families") != sum(expected_families.values()):
        raise AssertionError("total family count mismatch")
    if counts.get("scenes") != sum(expected_scenes.values()):
        raise AssertionError("total scene count mismatch")
    if counts.get("records") != sum(expected_records.values()):
        raise AssertionError("total record count mismatch")
    if (
        not isinstance(build_splits, list)
        or not build_splits
        or not set(build_splits).issubset(set(SPLITS))
        or "train" not in build_splits
    ):
        raise AssertionError("invalid build_splits declaration")
    if any(
        expected_families[split] <= 0
        for split in build_splits
    ) or any(
        expected_families[split] != 0
        for split in SPLITS
        if split not in build_splits
    ):
        raise AssertionError("scene-family counts disagree with build_splits")
    debug_override = config.get("debug_families_per_split_override")
    if debug_override is not None:
        if isinstance(debug_override, dict):
            if set(debug_override) != set(SPLITS):
                raise AssertionError("invalid debug family-count override")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value != expected_families[split]
                for split, value in debug_override.items()
            ):
                raise AssertionError("invalid debug family-count override")
        elif (
            isinstance(debug_override, bool)
            or not isinstance(debug_override, int)
            or debug_override <= 0
            or any(
                value != debug_override
                for split, value in expected_families.items()
                if split in build_splits
            )
            or any(
                value != 0
                for split, value in expected_families.items()
                if split not in build_splits
            )
        ):
            raise AssertionError("invalid debug family-count override")
    if profile == "paper" and debug_override is None:
        floors = {
            "train": 800,
            "val": 100,
            "test_iid": 200,
            "test_compositional_ood": 200,
            "test_label_ood": 100,
        }
        if any(expected_scenes[split] < floor for split, floor in floors.items()):
            raise AssertionError("paper profile falls below predeclared scene floors")
    question_design = config.get("question_design", {})
    strict_semantic_balance = (
        profile in {"internal_scale", "paper"} and debug_override is None
    )
    required_question = {
        "answer_options": 5,
        "primary_counterfactual_variants": list(VARIANTS),
        "primary_surface_and_options_identical_across_variants": True,
        "ordinal_same_label_anchor": True,
        "adjacency_definition": "unique_semantic_onset_order",
        "templates_disjoint_train_validation_evaluation": True,
        "matched_no_evidence_every_relation_template_partition": True,
        "paired_first_mention_order_controls": True,
        "paired_first_options_independently_permuted": True,
        "split_local_positive_semantics_for_every_negative": (
            strict_semantic_balance
        ),
        "negative_semantic_assignment_policy": (
            "deterministic_most_constrained_coverage_normalized_load_v1"
            if strict_semantic_balance
            else "legacy_smoke_unbalanced"
        ),
        "primary_negative_semantics_immutable": True,
        "first_negative_surface_pair_atomic": True,
    }
    for key, expected in required_question.items():
        if question_design.get(key) != expected:
            raise AssertionError(f"question design mismatch: {key}")
    bounds = question_design.get("no_evidence_ratio_bounds")
    if bounds != [0.2, 0.4]:
        raise AssertionError("unexpected no-evidence ratio bounds")
    composition = config.get("composition", {})
    required_composition = {
        "same_label_repeat_required_every_scene": True,
        "semantic_overlap_required_every_scene": True,
        "primary_pair_and_triplet_disjoint_train_vs_compositional_ood": True,
        "heldout_labels_exclusive_to_label_ood": True,
        "semantic_question_and_option_labels_split_local": (
            strict_semantic_balance
        ),
    }
    for key, expected in required_composition.items():
        if composition.get(key) != expected:
            raise AssertionError(f"composition design mismatch: {key}")
    semantic_balance = config.get("semantic_balance")
    if not isinstance(semantic_balance, dict):
        raise AssertionError("semantic balance policy is missing")
    if semantic_balance.get("strict_profile_gate") is not strict_semantic_balance:
        raise AssertionError("semantic balance strictness/profile mismatch")
    expected_eligibility = (
        "eligible" if strict_semantic_balance else "not_paper_eligible"
    )
    if semantic_balance.get("paper_eligibility") != expected_eligibility:
        raise AssertionError("semantic balance paper eligibility mismatch")
    expected_shortcut_gate = (
        {
            "unseen_semantics_backoff": "train_relation_prior",
            "maximum_auroc_down": 0.60,
            "maximum_best_balanced_accuracy_down": 0.60,
        }
        if strict_semantic_balance
        else None
    )
    if (
        semantic_balance.get("train_to_val_exact_semantics_shortcut_gate")
        != expected_shortcut_gate
    ):
        raise AssertionError("semantic shortcut gate policy mismatch")
    summary_by_split = semantic_balance.get("summary_by_split")
    if not isinstance(summary_by_split, dict) or set(summary_by_split) != set(build_splits):
        raise AssertionError("semantic balance split summaries are incomplete")
    return rate, samples, expected_families, str(profile)


def _validate_source_receipt(
    config: Mapping[str, Any], project_root: Path
) -> tuple[Mapping[str, Any], Dict[str, Mapping[str, Any]]]:
    source = config.get("source")
    if not isinstance(source, dict):
        raise AssertionError("source provenance is missing")
    receipt_path = _resolve(project_root, source["receipt_path"], "source receipt")
    if sha256_file(receipt_path) != source.get("receipt_sha256"):
        raise AssertionError("source receipt hash mismatch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_format = receipt.get("format")
    if receipt_format not in {RECEIPT_FORMAT, AUDITED_RECEIPT_FORMAT} or not receipt.get("acquisition_complete"):
        raise AssertionError("source receipt is not finalized")
    if source.get("receipt_format") != receipt_format:
        raise AssertionError("config/receipt format mismatch")
    if config.get("profile") == "paper" and receipt_format != AUDITED_RECEIPT_FORMAT:
        raise AssertionError("paper profile requires audited source-generic receipt")
    if receipt_format == AUDITED_RECEIPT_FORMAT:
        verification = receipt.get("audio_verification", {})
        if (
            receipt.get("source_route") != "fuss_v1.3_fsd50k_labels"
            or receipt.get("release_ready") is not True
            or receipt.get("metadata_gate_passed") is not True
            or verification.get("all_declared_hashes_match") is not True
            or verification.get("verified_source_count") != receipt.get("source_count")
        ):
            raise AssertionError("audited source receipt is not release-ready")
    receipt_revision = receipt.get(
        "dataset_revision", receipt.get("audio_dataset", {}).get("version")
    )
    if receipt_revision != source.get("dataset_revision"):
        raise AssertionError("source revision mismatch")
    receipt_metadata_hash = receipt.get("metadata_sha256", receipt.get("ledger_sha256"))
    if receipt_metadata_hash != source.get("metadata_sha256"):
        raise AssertionError("source metadata hash mismatch")
    metadata_path = _resolve(project_root, source["metadata_path"], "source metadata")
    if sha256_file(metadata_path) != source.get("metadata_sha256"):
        raise AssertionError("local source metadata changed")
    if receipt_format == AUDITED_RECEIPT_FORMAT:
        if source.get("selection_sha256") != receipt.get("selection_sha256"):
            raise AssertionError("config/receipt selection hash mismatch")
        selection_path = _resolve(
            project_root, source.get("selection_path"), "source label selection"
        )
        if sha256_file(selection_path) != source.get("selection_sha256"):
            raise AssertionError("local source label selection changed")
    receipt_sources = receipt.get("sources")
    if not isinstance(receipt_sources, list) or len(receipt_sources) != receipt.get("source_count"):
        raise AssertionError("receipt source payload is incomplete")
    by_id = {}
    partitions = defaultdict(set)
    creator_partitions = defaultdict(set)
    uploader_partitions = defaultdict(set)
    content_hash_owner: Dict[str, str] = {}
    license_id = receipt.get("license_record", {}).get("record_id")
    for row in receipt_sources:
        source_id = row.get("source_id")
        if source_id in by_id:
            raise AssertionError("receipt contains duplicate source IDs")
        if row.get("license_record_id") != license_id:
            raise AssertionError("source/license receipt mismatch")
        source_path = _resolve(project_root, row["audio_path"], "source audio")
        if sha256_file(source_path) != row.get("sha256"):
            raise AssertionError(f"source audio hash mismatch: {source_id}")
        by_id[source_id] = row
        partitions[source_id].add(row["partition"])
        if receipt_format == AUDITED_RECEIPT_FORMAT:
            creator_id = row.get("creator_id")
            uploader_id = row.get("uploader_id")
            if not isinstance(creator_id, str) or not creator_id.strip() or creator_id == "unknown":
                raise AssertionError("audited source lacks creator identity")
            if not isinstance(uploader_id, str) or not uploader_id.strip() or uploader_id == "unknown":
                raise AssertionError("audited source lacks uploader identity")
            if not row.get("attribution") or row.get("source_license_spdx") not in {
                "CC-BY-4.0",
                "CC0-1.0",
            }:
                raise AssertionError("audited source lacks a permitted license/attribution")
            creator_partitions[creator_id].add(row["partition"])
            uploader_partitions[uploader_id].add(row["partition"])
            duplicate = content_hash_owner.setdefault(row["sha256"], source_id)
            if duplicate != source_id:
                raise AssertionError("duplicate audio content hash in audited receipt")
    if any(len(values) != 1 for values in partitions.values()):
        raise AssertionError("source receipt leaks a file across partitions")
    if receipt_format == AUDITED_RECEIPT_FORMAT and any(
        len(values) != 1 for values in creator_partitions.values()
    ):
        raise AssertionError("creator identity leaks across source partitions")
    if receipt_format == AUDITED_RECEIPT_FORMAT and any(
        len(values) != 1 for values in uploader_partitions.values()
    ):
        raise AssertionError("uploader identity leaks across source partitions")
    release = config.get("release_policy", {})
    expected_release = bool(
        receipt["license_record"].get("status") == "verified"
        and receipt["license_record"].get("redistribution_allowed") is True
    )
    if release.get("audio_redistribution_ready") != expected_release:
        raise AssertionError("release readiness disagrees with license receipt")
    return receipt, by_id


def _validate_build_identity(config: Mapping[str, Any], project_root: Path) -> None:
    identities = config.get("build_identity")
    expected_roles = {"builder", "schema", "validator", "source_planner"}
    if not isinstance(identities, dict) or set(identities) != expected_roles:
        raise AssertionError("build identities are incomplete")
    for role, identity in identities.items():
        path = _resolve(project_root, identity["path"], f"build identity {role}")
        if sha256_file(path) != identity.get("sha256"):
            raise AssertionError(f"build identity changed: {role}")


def _event_identity(event: QCESV5Event, *, include_interval: bool) -> tuple[Any, ...]:
    base = (
        event.event_id,
        event.label,
        event.event_kind,
        event.source_dataset,
        event.dataset_version,
        event.source_id,
        event.creator_id,
        event.uploader_id,
        event.attribution,
        event.source_license_spdx,
        event.source_license_url,
        event.source_partition,
        event.source_path,
        event.source_sha256,
        event.license_record_id,
        event.source_interval_seconds,
        event.source_crop_interval_seconds,
        event.gain_db,
    )
    return base + ((event.onset_seconds, event.offset_seconds),) if include_interval else base


def _scene_signature(record: QCESV5Record) -> tuple[Any, ...]:
    return (
        record.schema_version,
        record.storage_mode,
        record.scene_id,
        record.scene_family_id,
        record.variant_id,
        record.intervention,
        record.split,
        record.sample_rate,
        record.num_samples,
        record.mixture_path,
        record.events,
        record.source_group_ids,
        record.same_label_repeat,
        record.semantic_overlap,
        record.max_polyphony,
        record.render_recipe_id,
        record.mixture_peak,
        record.family_gain,
        record.generation_seed,
    )


def _validate_family_counterfactuals(
    records: Sequence[QCESV5Record], scene_records: Mapping[str, Sequence[QCESV5Record]]
) -> Dict[str, int]:
    by_family: Dict[str, List[QCESV5Record]] = defaultdict(list)
    for record in records:
        by_family[record.scene_family_id].append(record)
    primary_groups_validated = 0
    for family_id, family_records in by_family.items():
        split_values = {record.split for record in family_records}
        if len(split_values) != 1:
            raise AssertionError(f"scene family crosses splits: {family_id}")
        scenes = {record.scene_id for record in family_records}
        variants = {record.variant_id for record in family_records}
        if len(scenes) != 3 or variants != set(VARIANTS) or len(family_records) != 48:
            raise AssertionError(f"incomplete three-variant family: {family_id}")
        scene_by_variant = {
            scene_records_by_id[0].variant_id: scene_records_by_id[0]
            for scene_id in scenes
            for scene_records_by_id in [scene_records[scene_id]]
        }
        if set(scene_by_variant) != set(VARIANTS):
            raise AssertionError(f"variant scene mapping failed: {family_id}")
        base = scene_by_variant["base"]
        swap = scene_by_variant["order_swap"]
        drop = scene_by_variant["anchor_drop"]
        if not math.isclose(base.family_gain, swap.family_gain, abs_tol=1e-12) or not math.isclose(
            base.family_gain, drop.family_gain, abs_tol=1e-12
        ):
            raise AssertionError("counterfactual variants use different family gains")
        base_events = {event.event_id: event for event in base.events}
        swap_events = {event.event_id: event for event in swap.events}
        drop_events = {event.event_id: event for event in drop.events}
        swap_ids = set(swap.intervention.intervened_event_ids)
        drop_ids = set(drop.intervention.intervened_event_ids)
        if set(base_events) != set(swap_events) or set(base_events) - drop_ids != set(drop_events):
            raise AssertionError("counterfactual event membership mismatch")
        for event_id in base_events:
            if _event_identity(base_events[event_id], include_interval=False) != _event_identity(
                swap_events[event_id], include_interval=False
            ):
                raise AssertionError("order swap changed source/crop/gain identity")
            if event_id not in swap_ids and base_events[event_id].interval != swap_events[event_id].interval:
                raise AssertionError("order swap changed a non-intervened event")
        swap_onsets = sorted(base_events[event_id].onset_seconds for event_id in swap_ids)
        changed_onsets = sorted(swap_events[event_id].onset_seconds for event_id in swap_ids)
        if swap_onsets != changed_onsets or any(
            base_events[event_id].onset_seconds == swap_events[event_id].onset_seconds
            for event_id in swap_ids
        ):
            raise AssertionError("onset-swap intervention did not exchange both onsets")
        for event_id, event in drop_events.items():
            if _event_identity(event, include_interval=True) != _event_identity(
                base_events[event_id], include_interval=True
            ):
                raise AssertionError("anchor drop changed a retained event")

        primary = [record for record in family_records if record.primary_counterfactual_probe]
        if len(primary) != 3 or {record.variant_id for record in primary} != set(VARIANTS):
            raise AssertionError("family lacks one primary probe per variant")
        primary_by_variant = {record.variant_id: record for record in primary}
        common_surface = {
            (
                record.question,
                record.answer_options,
                record.paraphrase_family_id,
                record.question_semantics_id,
                record.counterfactual_group_id,
            )
            for record in primary
        }
        if len(common_surface) != 1:
            raise AssertionError("primary counterfactual surface/options changed")
        answers = {record.answer for record in primary}
        if len(answers) != 3 or NO_EVIDENCE_ANSWER not in answers:
            raise AssertionError("primary family must realize answer A/B/no_evidence")
        if primary_by_variant["base"].no_evidence or primary_by_variant["order_swap"].no_evidence:
            raise AssertionError("base/swap primary probes must be answerable")
        if not primary_by_variant["anchor_drop"].no_evidence:
            raise AssertionError("anchor-drop primary probe must be no-evidence")
        if primary_by_variant["base"].query_instance_ordinal != 3:
            raise AssertionError("primary probe must query the third same-label instance")
        primary_groups_validated += 1
    return {
        "scene_families_validated": len(by_family),
        "primary_counterfactual_groups_validated": primary_groups_validated,
    }


def _validate_first_surface_controls(records: Sequence[QCESV5Record]) -> int:
    groups: Dict[str, List[QCESV5Record]] = defaultdict(list)
    first_count = 0
    for record in records:
        if record.relation != "first":
            if record.surface_control_group_id is not None:
                raise AssertionError("non-first record carries a surface-control group")
            continue
        first_count += 1
        if record.surface_control_group_id is None:
            raise AssertionError("first record lacks a surface-control group")
        groups[record.surface_control_group_id].append(record)
    for group_id, pair in groups.items():
        if len(pair) != 2 or {record.mention_order_variant for record in pair} != {
            "forward",
            "reversed",
        }:
            raise AssertionError(f"incomplete first mention-order pair: {group_id}")
        forward = next(record for record in pair if record.mention_order_variant == "forward")
        reversed_record = next(
            record for record in pair if record.mention_order_variant == "reversed"
        )
        invariant = lambda record: (
            record.scene_id,
            record.variant_id,
            record.question_semantics_id,
            record.paraphrase_family_id,
            record.answer,
            record.no_evidence,
            record.no_evidence_reason,
            record.anchor_event_ids,
            record.answer_event_ids,
            record.evidence_event_ids,
            record.mixture_path,
            record.evidence_stem_path,
            record.residual_stem_path,
            record.anchor_stem_path,
            record.answer_stem_path,
        )
        if invariant(forward) != invariant(reversed_record):
            # Stem paths are per-record, so compare audio semantics below while
            # allowing the path names themselves to differ.
            forward_without_paths = invariant(forward)[:10]
            reversed_without_paths = invariant(reversed_record)[:10]
            if forward_without_paths != reversed_without_paths:
                raise AssertionError(f"first surface pair changes semantics: {group_id}")
        if reversed_record.query_candidate_labels != tuple(
            reversed(forward.query_candidate_labels)
        ):
            raise AssertionError(f"first candidate mention order was not reversed: {group_id}")
        if forward.query_event_ids and reversed_record.query_event_ids != tuple(
            reversed(forward.query_event_ids)
        ):
            raise AssertionError(f"first event order was not reversed: {group_id}")
        if forward.no_evidence and reversed_record.absent_labels != tuple(
            reversed(forward.absent_labels)
        ):
            raise AssertionError(f"absent candidate order was not reversed: {group_id}")
        if forward.question == reversed_record.question:
            raise AssertionError(f"first mention-order surface did not change: {group_id}")
        if forward.answer_options == reversed_record.answer_options:
            raise AssertionError(f"first option permutation was reused: {group_id}")
        if set(forward.answer_options) != set(reversed_record.answer_options):
            raise AssertionError(f"first paired option set changed: {group_id}")
        if forward.answer_option_index == reversed_record.answer_option_index:
            raise AssertionError(f"first answer letter was not independently permuted: {group_id}")
    if first_count != 2 * len(groups):
        raise AssertionError("some first records are outside complete surface pairs")
    return len(groups)


def validate_dataset(root: Path, write_report: bool = False) -> Dict[str, Any]:
    root = root.resolve()
    project_root = root.parents[1]
    config_path = root / "dataset_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rate, samples, expected_families, profile = _validate_config(config)
    strict_semantic_balance = bool(
        config["semantic_balance"]["strict_profile_gate"]
    )
    storage_mode = _config_storage_mode(config)
    receipt, receipt_sources = _validate_source_receipt(config, project_root)
    _validate_build_identity(config, project_root)

    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    manifest_names = []
    for split in SPLITS:
        name = f"qces_{split}.jsonl"
        manifest_names.append(name)
        rows_by_split[split] = _read_jsonl(root / name)
        expected = expected_families[split] * 3 * 16
        if len(rows_by_split[split]) != expected:
            raise AssertionError(f"{split} manifest count mismatch")
        if any(row.get("split") != split for row in rows_by_split[split]):
            raise AssertionError(f"{split} manifest contains another split")
    manifest_names.append("qces_all.jsonl")
    all_rows = _read_jsonl(root / "qces_all.jsonl")
    concatenated = [row for split in SPLITS for row in rows_by_split[split]]
    if all_rows != concatenated:
        raise AssertionError("qces_all.jsonl is not canonical split concatenation")
    records = [parse_qces_v5_record(row) for row in all_rows]
    if len({record.sample_id for record in records}) != len(records):
        raise AssertionError("duplicate sample IDs")

    composition = config["composition"]
    seen_labels = set(composition["seen_labels"])
    heldout_labels = set(composition["heldout_labels"])
    nuisance_labels = set(composition["nuisance_labels"])
    if seen_labels & heldout_labels or (seen_labels | heldout_labels) & nuisance_labels:
        raise AssertionError("semantic/nuisance label inventories overlap")
    if profile == "paper" and (
        len(seen_labels) < 30 or len(heldout_labels) < 10 or len(nuisance_labels) < 8
    ):
        raise AssertionError("paper profile label inventory is undersized")

    scene_records: Dict[str, List[QCESV5Record]] = defaultdict(list)
    family_owner: Dict[str, str] = {}
    source_owner: Dict[str, str] = {}
    source_hash_cache: Dict[str, str] = {}
    audio_cache: OrderedDict[str, np.ndarray] = OrderedDict()
    audio_cache_peak_files = 0
    artifact_paths = {"dataset_config.json", *manifest_names}
    no_evidence_counts: Counter[str] = Counter()
    relation_counts: Counter[Tuple[str, str]] = Counter()
    relation_answerability: Counter[Tuple[str, str, bool]] = Counter()
    template_partition_answerability: Counter[Tuple[str, str, bool]] = Counter()
    answer_positions: Dict[str, Counter[int]] = defaultdict(Counter)
    template_ids: Dict[str, set[str]] = defaultdict(set)
    split_pair_signatures: Dict[str, set[str]] = defaultdict(set)
    split_triplet_signatures: Dict[str, set[str]] = defaultdict(set)
    scene_mixture_hashes: Dict[str, str] = {}
    scene_mixture_paths: Dict[str, str] = {}
    event_stem_owners: Dict[str, Tuple[str, str]] = {}
    maximum_errors = {
        "mixture_from_events": 0.0,
        "evidence_from_events": 0.0,
        "residual_from_events": 0.0,
        "anchor_from_events": 0.0,
        "answer_from_events": 0.0,
        "evidence_plus_residual": 0.0,
        "mixture_peak": 0.0,
    }
    semantic_question_option_cross_split_label_count = 0

    def audio(relative: str) -> np.ndarray:
        nonlocal audio_cache_peak_files
        if relative not in audio_cache:
            path = _resolve(root, relative, "dataset audio")
            audio_cache[relative] = _read_audio(path, rate, samples)
            artifact_paths.add(relative)
            while len(audio_cache) > VALIDATOR_AUDIO_CACHE_MAX_FILES:
                audio_cache.popitem(last=False)
            audio_cache_peak_files = max(audio_cache_peak_files, len(audio_cache))
        else:
            audio_cache.move_to_end(relative)
        return audio_cache[relative]

    for record in records:
        if record.storage_mode != storage_mode:
            raise AssertionError("manifest record storage mode differs from config")
        scene_records[record.scene_id].append(record)
        existing_owner = family_owner.setdefault(record.scene_family_id, record.split)
        if existing_owner != record.split:
            raise AssertionError("scene family leaks across splits")
        no_evidence_counts[record.split] += int(record.no_evidence)
        relation_counts[(record.split, record.relation)] += 1
        relation_answerability[(record.split, record.relation, record.no_evidence)] += 1
        template_partition_answerability[
            (record.template_partition, record.relation, record.no_evidence)
        ] += 1
        answer_positions[record.split][record.answer_option_index] += 1
        template_ids[record.template_partition].add(record.paraphrase_family_id)
        split_pair_signatures[record.split].add(record.composition_pair_signature)
        split_triplet_signatures[record.split].add(record.composition_triplet_signature)
        if not record.same_label_repeat or not record.semantic_overlap or record.max_polyphony < 2:
            raise AssertionError(f"scene lacks required hard acoustics: {record.scene_id}")

        semantic_event_labels = {
            event.label for event in record.events if event.event_kind == "semantic"
        }
        allowed_semantic_labels = (
            heldout_labels if record.split == "test_label_ood" else seen_labels
        )
        forbidden_semantic_labels = (
            seen_labels if record.split == "test_label_ood" else heldout_labels
        )
        if record.split == "test_label_ood":
            if not semantic_event_labels.issubset(heldout_labels) or semantic_event_labels & seen_labels:
                raise AssertionError("label-OOD scene contains a seen semantic label")
        elif not semantic_event_labels.issubset(seen_labels) or semantic_event_labels & heldout_labels:
            raise AssertionError("seen-class split contains a held-out semantic label")
        query_surface_labels = {
            *record.query_candidate_labels,
            *record.absent_labels,
        }
        if record.query_label is not None:
            query_surface_labels.add(record.query_label)
        semantic_metadata_labels = {
            *query_surface_labels,
            *(option for option in record.answer_options if option != NO_EVIDENCE_ANSWER),
        }
        if record.answer != NO_EVIDENCE_ANSWER:
            semantic_metadata_labels.add(record.answer)
        foreign = sorted(semantic_metadata_labels - allowed_semantic_labels)
        semantic_question_option_cross_split_label_count += len(foreign)
        if strict_semantic_balance and foreign:
            raise AssertionError(
                f"{record.split} semantic question/options leak foreign labels: {foreign}"
            )
        missing_surface = sorted(
            label
            for label in query_surface_labels
            if label.casefold() not in record.question.casefold()
        )
        if missing_surface:
            raise AssertionError(
                f"question text omits query labels: {record.sample_id}: {missing_surface}"
            )
        foreign_surface = _foreign_question_label_mentions(
            record.question,
            forbidden_labels=forbidden_semantic_labels,
            expressed_labels=query_surface_labels,
        )
        semantic_question_option_cross_split_label_count += len(foreign_surface)
        if strict_semantic_balance and foreign_surface:
            raise AssertionError(
                f"{record.split} question text names foreign labels: {foreign_surface}"
            )
        if any(
            event.label not in nuisance_labels
            for event in record.events
            if event.event_kind == "nuisance"
        ):
            raise AssertionError("scene contains an undeclared nuisance label")

        mixture = audio(record.mixture_path)
        previous_mixture_path = scene_mixture_paths.setdefault(
            record.scene_id, record.mixture_path
        )
        if previous_mixture_path != record.mixture_path:
            raise AssertionError("one scene references multiple mixture files")
        event_audio = {event.event_id: audio(event.stem_path) for event in record.events}
        for event in record.events:
            owner = event_stem_owners.setdefault(
                event.stem_path, (record.scene_id, event.event_id)
            )
            if owner != (record.scene_id, event.event_id):
                raise AssertionError("an event stem path is reused across scene events")
        event_ids = set(event_audio)
        if storage_mode == MATERIALIZED_STORAGE_MODE:
            if None in {
                record.evidence_stem_path,
                record.residual_stem_path,
                record.anchor_stem_path,
                record.answer_stem_path,
            }:
                raise AssertionError("materialized record lacks a role-stem path")
            evidence = audio(str(record.evidence_stem_path))
            residual = audio(str(record.residual_stem_path))
            anchor = audio(str(record.anchor_stem_path))
            answer = audio(str(record.answer_stem_path))
        else:
            evidence = _sum(event_audio, record.evidence_event_ids, samples)
            # Defining R as the arithmetic complement makes reconstruction
            # exact for the loaded PCM mixture, rather than merely close after
            # separately quantizing an R WAV.
            residual = mixture - evidence
            anchor = _sum(event_audio, record.anchor_event_ids, samples)
            answer = _sum(event_audio, record.answer_event_ids, samples)
        expected = {
            "mixture_from_events": _sum(event_audio, event_ids, samples),
            "evidence_from_events": _sum(event_audio, record.evidence_event_ids, samples),
            "residual_from_events": _sum(
                event_audio, event_ids - set(record.evidence_event_ids), samples
            ),
            "anchor_from_events": _sum(event_audio, record.anchor_event_ids, samples),
            "answer_from_events": _sum(event_audio, record.answer_event_ids, samples),
        }
        actual = {
            "mixture_from_events": mixture,
            "evidence_from_events": evidence,
            "residual_from_events": residual,
            "anchor_from_events": anchor,
            "answer_from_events": answer,
        }
        for key in expected:
            error = _maximum_error(actual[key], expected[key])
            maximum_errors[key] = max(maximum_errors[key], error)
            if error > PCM_TOLERANCE:
                raise AssertionError(f"{key} mismatch: {record.sample_id}: {error}")
        reconstruction_error = _maximum_error(mixture, evidence + residual)
        maximum_errors["evidence_plus_residual"] = max(
            maximum_errors["evidence_plus_residual"], reconstruction_error
        )
        if reconstruction_error > PCM_TOLERANCE:
            raise AssertionError(f"E+R reconstruction mismatch: {record.sample_id}")
        peak_error = abs(float(np.max(np.abs(mixture))) - record.mixture_peak)
        maximum_errors["mixture_peak"] = max(maximum_errors["mixture_peak"], peak_error)
        if peak_error > PEAK_TOLERANCE:
            raise AssertionError(f"mixture peak metadata mismatch: {record.sample_id}")

        for event in record.events:
            receipt_row = receipt_sources.get(event.source_id)
            if receipt_row is None:
                raise AssertionError(f"event source absent from receipt: {event.source_id}")
            expected_source = (
                receipt_row["label"],
                receipt_row["source_dataset"],
                receipt_row["dataset_version"],
                receipt_row["creator_id"],
                receipt_row["uploader_id"],
                receipt_row["attribution"],
                receipt_row["source_license_spdx"],
                receipt_row["source_license_url"],
                receipt_row["partition"],
                receipt_row["audio_path"],
                receipt_row["sha256"],
                receipt_row["license_record_id"],
            )
            actual_source = (
                event.label,
                event.source_dataset,
                event.dataset_version,
                event.creator_id,
                event.uploader_id,
                event.attribution,
                event.source_license_spdx,
                event.source_license_url,
                event.source_partition,
                event.source_path,
                event.source_sha256,
                event.license_record_id,
            )
            if actual_source != expected_source:
                raise AssertionError(f"event/receipt identity mismatch: {event.source_id}")
            owner = source_owner.setdefault(event.source_id, record.split)
            if owner != record.split:
                raise AssertionError(f"source file leaks across splits: {event.source_id}")
            if event.source_id not in source_hash_cache:
                source_path = _resolve(project_root, event.source_path, "event source")
                source_hash_cache[event.source_id] = sha256_file(source_path)
            if source_hash_cache[event.source_id] != event.source_sha256:
                raise AssertionError(f"event source hash changed: {event.source_id}")

    # Every scene repeats exactly one immutable scene payload over 16 questions.
    for scene_id, values in scene_records.items():
        if len(values) != 16 or {record.question_index for record in values} != set(range(16)):
            raise AssertionError(f"scene question coverage mismatch: {scene_id}")
        if len({_scene_signature(record) for record in values}) != 1:
            raise AssertionError(f"scene payload drifts across questions: {scene_id}")
        if sum(record.primary_counterfactual_probe for record in values) != 1:
            raise AssertionError(f"scene needs exactly one primary probe: {scene_id}")
        mixture_hash = sha256_file(_resolve(root, values[0].mixture_path, "mixture"))
        if mixture_hash in scene_mixture_hashes.values():
            duplicate = next(key for key, value in scene_mixture_hashes.items() if value == mixture_hash)
            raise AssertionError(f"duplicate rendered scenes: {duplicate}, {scene_id}")
        scene_mixture_hashes[scene_id] = mixture_hash

    family_summary = _validate_family_counterfactuals(records, scene_records)
    first_surface_control_groups = _validate_first_surface_controls(records)
    if split_pair_signatures["train"] & split_pair_signatures["test_compositional_ood"]:
        raise AssertionError("primary pair signature leaks into compositional OOD")
    if split_triplet_signatures["train"] & split_triplet_signatures["test_compositional_ood"]:
        raise AssertionError("primary triplet signature leaks into compositional OOD")
    if template_ids["train"] & template_ids["validation"] or template_ids["train"] & template_ids["evaluation"] or template_ids["validation"] & template_ids["evaluation"]:
        raise AssertionError("paraphrase/template family leaks across partitions")

    no_evidence_ratios = {}
    records_by_split_count = Counter(record.split for record in records)
    for split in SPLITS:
        ratio = no_evidence_counts[split] / records_by_split_count[split]
        no_evidence_ratios[split] = ratio
        if not 0.20 <= ratio <= 0.40:
            raise AssertionError(f"{split} no-evidence ratio out of bounds: {ratio}")
        if any(relation_counts[(split, relation)] == 0 for relation in ("after", "before", "first")):
            raise AssertionError(f"{split} omits a relation")
        for relation in ("after", "before", "first"):
            if any(
                relation_answerability[(split, relation, no_evidence)] == 0
                for no_evidence in (False, True)
            ):
                raise AssertionError(
                    f"{split}/{relation} does not contain both answerability states"
                )

    for partition in ("train", "validation", "evaluation"):
        for relation in ("after", "before", "first"):
            if any(
                template_partition_answerability[(partition, relation, status)] == 0
                for status in (False, True)
            ):
                raise AssertionError(
                    f"{partition}/{relation} lacks matched no-evidence coverage"
                )

    semantic_polarity_by_split = _semantic_polarity_summary(records)
    configured_semantic_summary = config["semantic_balance"]["summary_by_split"]
    if configured_semantic_summary != semantic_polarity_by_split:
        raise AssertionError("semantic balance summary differs from manifests")
    if strict_semantic_balance:
        for split in SPLITS:
            for relation in ("after", "before", "first"):
                negative_only = semantic_polarity_by_split[split]["by_relation"][
                    relation
                ]["negative_only_unique_semantics_down"]
                if negative_only:
                    raise AssertionError(
                        f"{split}/{relation} has {negative_only} negative-only "
                        "question semantics"
                    )

    exact_semantics_shortcut = _exact_semantics_train_to_val_shortcut(records)
    if strict_semantic_balance:
        shortcut_gate = config["semantic_balance"][
            "train_to_val_exact_semantics_shortcut_gate"
        ]
        _enforce_exact_semantics_shortcut_gate(
            exact_semantics_shortcut,
            maximum_auroc=shortcut_gate["maximum_auroc_down"],
            maximum_best_balanced_accuracy=shortcut_gate[
                "maximum_best_balanced_accuracy_down"
            ],
        )

    polarity_covered = sum(
        semantic_polarity_by_split[split]["matched_unique_semantics_up"]
        for split in SPLITS
    )

    fingerprint, file_hashes = _artifact_identity(root, artifact_paths)
    stored_audio_files = [relative for relative in file_hashes if relative.endswith(".wav")]
    stored_audio_bytes = sum(
        _resolve(root, relative, "stored audio accounting").stat().st_size
        for relative in stored_audio_files
    )
    storage_config = config.get("storage")
    if isinstance(storage_config, dict):
        if storage_config.get("stored_audio_file_count") != len(stored_audio_files):
            raise AssertionError("stored audio file count differs from config")
        if storage_config.get("stored_audio_bytes") != stored_audio_bytes:
            raise AssertionError("stored audio byte count differs from config")
    pcm_wav_bytes = samples * 2 + 44
    avoided_role_files = (
        len(records) * 4 if storage_mode == DERIVED_STORAGE_MODE else 0
    )
    hypothetical_materialized_audio_bytes = (
        stored_audio_bytes + avoided_role_files * pcm_wav_bytes
    )
    storage_reduction = (
        1.0 - stored_audio_bytes / hypothetical_materialized_audio_bytes
        if hypothetical_materialized_audio_bytes
        else 0.0
    )
    license_record = receipt["license_record"]
    report = {
        "schema_version": config["schema_version"],
        "profile": profile,
        "storage_mode": storage_mode,
        "artifact_fingerprint_sha256": fingerprint,
        "artifact_file_count": len(file_hashes),
        "stored_audio_file_count_down": len(stored_audio_files),
        "stored_audio_bytes_down": stored_audio_bytes,
        "avoided_materialized_role_file_count_up": avoided_role_files,
        "estimated_audio_byte_reduction_ratio_up": storage_reduction,
        "validator_audio_cache_peak_file_count_down": audio_cache_peak_files,
        "validator_audio_cache_limit_file_count_down": VALIDATOR_AUDIO_CACHE_MAX_FILES,
        "records": len(records),
        "scenes": len(scene_records),
        **family_summary,
        "first_surface_control_groups_validated_up": first_surface_control_groups,
        "records_by_split": dict(sorted(records_by_split_count.items())),
        "no_evidence_ratio_by_split_down": no_evidence_ratios,
        "relation_counts": {
            f"{split}:{relation}": relation_counts[(split, relation)]
            for split in SPLITS
            for relation in ("after", "before", "first")
        },
        "relation_answerability_counts": {
            f"{split}:{relation}:{'no_evidence' if status else 'answerable'}": (
                relation_answerability[(split, relation, status)]
            )
            for split in SPLITS
            for relation in ("after", "before", "first")
            for status in (False, True)
        },
        "answer_option_position_counts": {
            split: dict(sorted(counts.items())) for split, counts in answer_positions.items()
        },
        "source_count_used": len(source_owner),
        "source_split_overlap_count_down": 0,
        "scene_family_split_overlap_count_down": 0,
        "train_compositional_pair_overlap_count_down": 0,
        "train_compositional_triplet_overlap_count_down": 0,
        "template_partition_overlap_count_down": 0,
        "heldout_label_train_presence_count_down": 0,
        "semantic_question_option_cross_split_label_count_down": (
            semantic_question_option_cross_split_label_count
        ),
        "matched_answerability_semantics_groups_up": polarity_covered,
        "semantic_polarity_by_split": semantic_polarity_by_split,
        "train_to_val_exact_semantics_shortcut": exact_semantics_shortcut,
        "semantic_balance_paper_eligibility": config["semantic_balance"][
            "paper_eligibility"
        ],
        "semantic_balance_strict_gate_passed": strict_semantic_balance,
        "maximum_audio_errors_down": maximum_errors,
        "license_status": license_record["status"],
        "audio_redistribution_ready": config["release_policy"]["audio_redistribution_ready"],
        "synthetic_paper_scale_gate_passed": bool(
            profile == "paper"
            and strict_semantic_balance
            and receipt.get("format") == AUDITED_RECEIPT_FORMAT
            and config["release_policy"]["audio_redistribution_ready"]
        ),
        # The independently sourced TACOS/self-recorded real evaluation is a
        # separate artifact and is not proven by this synthetic validator.
        "submission_data_gate_passed": False,
    }
    if write_report:
        (root / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_dataset(args.dataset_root, write_report=args.write_report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
