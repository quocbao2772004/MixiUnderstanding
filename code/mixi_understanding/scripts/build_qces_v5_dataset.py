#!/usr/bin/env python3
"""Build QCES v5 counterfactual scene families from a pinned source receipt.

Each family has three variants: a base scene, an onset swap that changes the
answer to one otherwise identical question, and an anchor-drop variant that
turns the same question into ``no_evidence``.  Semantic overlap and a repeated
same-label anchor make source identity and onset order, rather than keywords,
necessary for solving the primary probe.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import scipy
import soundfile as sf

from mixi_understanding.data.qces_v5_schema import (  # noqa: E402
    DERIVED_SCHEMA_VERSION,
    DERIVED_STORAGE_MODE,
    EVALUATION_AXIS_BY_SPLIT,
    MATERIALIZED_STORAGE_MODE,
    NO_EVIDENCE_ANSWER,
    SCHEMA_VERSION,
    SPLITS,
    TEMPLATE_PARTITION_BY_SPLIT,
    parse_qces_v5_record,
)
from mixi_understanding.scripts.build_qa_removal_dataset import (  # noqa: E402
    SourceClip,
    crop_source_event,
    relative_posix,
    render_clip,
    replace_validated_output,
    sha256_file,
    stable_seed,
    write_jsonl,
    write_wav,
)
from mixi_understanding.scripts.plan_qces_v5_sources import (  # noqa: E402
    RECEIPT_FORMAT,
)


BUILDER_VERSION = "5.4.0-label-conditioned-semantic-crops"
SAMPLE_RATE = 32_000
DURATION_SECONDS = 10.0
SEMANTIC_DURATION_RANGE = (0.90, 1.25)
NUISANCE_DURATION_SECONDS = 5.8
HEADROOM = 0.95
FADE_MILLISECONDS = 10.0
QUESTIONS_PER_SCENE = 16
VARIANTS = ("base", "order_swap", "anchor_drop")
AUDITED_RECEIPT_FORMAT = "qces_v5_audited_source_ledger_v1"
SEMANTIC_CROP_BANK_FORMAT = "qces_v5_semantic_crop_bank_v2"


PROFILE_FAMILY_COUNTS = {
    # Fifteen rendered scenes and 240 grouped questions.  This is a mechanism
    # smoke test, never a paper result.
    "smoke": {split: 1 for split in SPLITS},
    # 1,407 scenes and 22,512 questions: just above the predeclared ICASSP floor.
    "internal_scale": {
        "train": 267,
        "val": 34,
        "test_iid": 67,
        "test_compositional_ood": 67,
        "test_label_ood": 34,
    },
    # Same finite scale as the internal stress test, but this profile is
    # accepted only from the audited, source-generic FUSS/FSD50K ledger route.
    "paper": {
        "train": 267,
        "val": 34,
        "test_iid": 67,
        "test_compositional_ood": 67,
        "test_label_ood": 34,
    },
}


TEMPLATES = {
    "train": {
        "after": (
            ("tr_after_next", "What sound begins next after the {ordinal} occurrence of {anchor}?"),
            ("tr_after_follow", "Which sound starts immediately following the {ordinal} {anchor} event?"),
            ("tr_after_onset", "After the {ordinal} instance of {anchor} starts, which event starts next?"),
        ),
        "before": (
            ("tr_before_previous", "What sound begins directly before the {ordinal} occurrence of {anchor}?"),
            ("tr_before_precede", "Which sound starts immediately preceding the {ordinal} {anchor} event?"),
            ("tr_before_onset", "Before the {ordinal} instance of {anchor} starts, which event starts last?"),
        ),
        "first": (
            ("tr_first_earlier", "Which sound class begins first anywhere, {left} or {right}?"),
            ("tr_first_onset", "Whose earliest onset comes first: {left} or {right}?"),
            ("tr_first_start", "Between {left} and {right}, which is first to start?"),
        ),
    },
    "validation": {
        "after": (
            ("va_after_subsequent", "Which event has the subsequent onset after the {ordinal} {anchor}?"),
            ("va_after_then", "The {ordinal} {anchor} begins; what begins next?"),
        ),
        "before": (
            ("va_before_prior", "Which event has the prior onset before the {ordinal} {anchor}?"),
            ("va_before_then", "What begins just before the {ordinal} {anchor} begins?"),
        ),
        "first": (
            ("va_first_initial", "Of {left} and {right}, which has the initial onset?"),
            ("va_first_sooner", "Which begins sooner in the clip, {left} or {right}?"),
        ),
    },
    "evaluation": {
        "after": (
            ("ev_after_successor", "Name the onset-order successor of the {ordinal} {anchor} occurrence."),
            ("ev_after_next_start", "What starts next once the {ordinal} {anchor} has started?"),
            ("ev_after_sequence", "In onset sequence, what follows the {ordinal} occurrence of {anchor}?"),
        ),
        "before": (
            ("ev_before_predecessor", "Name the onset-order predecessor of the {ordinal} {anchor} occurrence."),
            ("ev_before_last_start", "What starts immediately ahead of the {ordinal} {anchor}?"),
            ("ev_before_sequence", "In onset sequence, what precedes the {ordinal} occurrence of {anchor}?"),
        ),
        "first": (
            ("ev_first_leading", "Which class has the leading onset, {left} or {right}?"),
            ("ev_first_earliest", "Whose first occurrence is earlier: {left} or {right}?"),
            ("ev_first_order", "In start-time order, which comes first, {left} or {right}?"),
        ),
    },
}


@dataclass(frozen=True)
class PlannedSource:
    clip: SourceClip
    role: str
    partition: str
    sha256: str
    license_record_id: str
    source_dataset: str
    dataset_version: str
    creator_id: str
    uploader_id: str
    attribution: str
    source_license_spdx: str
    source_license_url: str


@dataclass(frozen=True)
class FamilyLabels:
    repeat_label: str
    base_answer_label: str
    swapped_answer_label: str
    extra_labels: Tuple[str, ...]

    @property
    def distinct_labels(self) -> Tuple[str, ...]:
        return (
            self.repeat_label,
            self.base_answer_label,
            self.swapped_answer_label,
            *self.extra_labels,
        )

    @property
    def pair_signature(self) -> str:
        answers = sorted((self.base_answer_label, self.swapped_answer_label))
        return f"{self.repeat_label}=>{answers[0]}+{answers[1]}"

    @property
    def triplet_signature(self) -> str:
        answers = sorted((self.base_answer_label, self.swapped_answer_label))
        return f"{self.repeat_label}|{answers[0]}|{answers[1]}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_FAMILY_COUNTS), default="smoke")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--source-receipt",
        type=Path,
        help=(
            "default: AudioTime-recovered/train5000_timestamp/"
            "qces_v5_<profile>_source_receipt.json"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="default: data/qces_v5_<profile>",
    )
    parser.add_argument("--seed", type=int, default=314_159)
    parser.add_argument(
        "--storage-mode",
        choices=(MATERIALIZED_STORAGE_MODE, DERIVED_STORAGE_MODE),
        help=(
            "Audio representation. Defaults to scene_event_derived for the "
            "paper profile and materialized for legacy/debug profiles."
        ),
    )
    parser.add_argument(
        "--families-per-split",
        type=int,
        help="Debug override applied to every split; must be positive.",
    )
    parser.add_argument(
        "--semantic-crop-bank",
        type=Path,
        help=(
            "Optional immutable label-conditioned crop bank. When supplied, "
            "every semantic source used by the build must have an exact entry."
        ),
    )
    parser.add_argument(
        "--debug-family-indices",
        type=Path,
        help=(
            "JSON mapping every split to one original full-profile family index. "
            "This builds a source-faithful mechanism smoke and is never paper eligible."
        ),
    )
    parser.add_argument(
        "--build-splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help=(
            "Build only the requested splits. This is a debug/pilot mechanism; "
            "the resulting partial dataset is never paper eligible."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _load_semantic_crop_bank(
    path: Path | None,
) -> tuple[dict[tuple[str, str], Mapping[str, Any]] | None, dict[str, Any] | None]:
    if path is None:
        return None, None
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"semantic crop bank is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != SEMANTIC_CROP_BANK_FORMAT:
        raise ValueError("semantic crop bank has the wrong format")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("semantic crop bank has no entries")
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("semantic crop bank contains a non-object entry")
        source_id = entry.get("source_id")
        label = entry.get("label")
        selected = entry.get("selected")
        if (
            not isinstance(source_id, str)
            or not source_id
            or not isinstance(label, str)
            or not label
            or not isinstance(selected, dict)
        ):
            raise ValueError("semantic crop bank entry lacks source/label/selection")
        center = selected.get("center_seconds")
        probability = selected.get("label_probability ↑")
        if (
            not isinstance(center, (int, float))
            or not math.isfinite(float(center))
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
        ):
            raise ValueError(f"invalid semantic crop selection for {label}::{source_id}")
        key = (source_id, label)
        if key in result:
            raise ValueError(f"duplicate semantic crop bank entry: {label}::{source_id}")
        result[key] = entry
    identity = {
        "format": SEMANTIC_CROP_BANK_FORMAT,
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "source_count": len(result),
        "curator_boundary": payload.get("curator_boundary"),
        "selection": payload.get("selection"),
        "manifest": payload.get("manifest"),
    }
    return result, identity


def resolve_storage_mode(profile: str, requested: str | None) -> str:
    if requested is not None:
        return requested
    return (
        DERIVED_STORAGE_MODE
        if profile == "paper"
        else MATERIALIZED_STORAGE_MODE
    )


def _split_semantic_vocabulary(
    split: str,
    *,
    seen_labels: Sequence[str],
    heldout_labels: Sequence[str],
) -> List[str]:
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    return list(heldout_labels if split == "test_label_ood" else seen_labels)


def _ordinal(index: int) -> str:
    words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}
    return words.get(index, f"{index}th")


def _sum(stems: Mapping[str, np.ndarray], ids: Iterable[str]) -> np.ndarray:
    if not stems:
        raise ValueError("cannot sum an empty stem dictionary")
    result = np.zeros_like(next(iter(stems.values())))
    for event_id in ids:
        result += stems[event_id]
    return result.astype(np.float32)


def _rms(waveform: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))


def _maximum_pcm_render_peak(
    variants: Mapping[str, Mapping[str, np.ndarray]],
) -> float:
    """Bound both stored mixtures and independently quantized event stems.

    Mixture-only headroom is insufficient when overlapping sources partially
    cancel: an individual stem can exceed full scale even though their sum is
    below the mixture limit. Since scene-event-derived storage writes every
    stem to PCM separately, the family gain must prevent clipping in both
    representations or the decoded stems will no longer sum to the mixture.
    """

    peaks: List[float] = []
    for stems in variants.values():
        if not stems:
            continue
        peaks.append(float(np.max(np.abs(_sum(stems, stems)))))
        peaks.extend(float(np.max(np.abs(waveform))) for waveform in stems.values())
    return max(peaks, default=0.0)


def _load_receipt(
    receipt_path: Path, project_root: Path, profile: str
) -> tuple[Dict[str, Any], Dict[Tuple[str, str, str], List[PlannedSource]]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_format = receipt.get("format")
    if receipt_format not in {RECEIPT_FORMAT, AUDITED_RECEIPT_FORMAT}:
        raise ValueError("source input is not a supported finalized QCES v5 receipt")
    if profile == "paper" and receipt_format != AUDITED_RECEIPT_FORMAT:
        raise ValueError("paper profile requires the audited source-generic receipt")
    if profile != "paper" and receipt_format == AUDITED_RECEIPT_FORMAT:
        raise ValueError("audited paper receipt must be built with --profile paper")
    if receipt.get("profile") != profile or not receipt.get("acquisition_complete"):
        raise ValueError("receipt profile/completion mismatch")
    if receipt_format == AUDITED_RECEIPT_FORMAT:
        verification = receipt.get("audio_verification", {})
        if (
            receipt.get("source_route") != "fuss_v1.3_fsd50k_labels"
            or receipt.get("release_ready") is not True
            or receipt.get("metadata_gate_passed") is not True
            or verification.get("all_declared_hashes_match") is not True
            or verification.get("verified_source_count") != receipt.get("source_count")
        ):
            raise ValueError("audited paper receipt has not passed every release gate")
    license_record = receipt.get("license_record")
    if not isinstance(license_record, dict) or not license_record.get("record_id"):
        raise ValueError("receipt lacks a license audit record")
    if receipt_format == AUDITED_RECEIPT_FORMAT and (
        license_record.get("status") != "verified"
        or license_record.get("redistribution_allowed") is not True
    ):
        raise ValueError("audited paper receipt lacks verified redistribution rights")
    metadata_path = (project_root / str(receipt.get("metadata_path", ""))).resolve()
    if not metadata_path.is_file() or sha256_file(metadata_path) != receipt.get(
        "metadata_sha256"
    ):
        raise ValueError("receipt metadata pin file/hash mismatch")
    if receipt_format == AUDITED_RECEIPT_FORMAT:
        selection_path = (
            project_root / str(receipt.get("selection_path", ""))
        ).resolve()
        if not selection_path.is_file() or sha256_file(selection_path) != receipt.get(
            "selection_sha256"
        ):
            raise ValueError("audited receipt selection file/hash mismatch")
    grouped: Dict[Tuple[str, str, str], List[PlannedSource]] = defaultdict(list)
    seen_ids = set()
    for row in receipt.get("sources", []):
        source_id = str(row["source_id"])
        if source_id in seen_ids:
            raise ValueError(f"duplicate receipt source: {source_id}")
        seen_ids.add(source_id)
        path = (project_root / row["audio_path"]).resolve()
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise ValueError(f"source file/hash mismatch: {source_id}")
        interval = tuple(float(value) for value in row["source_interval_seconds"])
        planned = PlannedSource(
            clip=SourceClip(
                source_id=source_id,
                label=str(row["label"]),
                interval_seconds=(interval[0], interval[1]),
                caption="",
                audio_path=path,
            ),
            role=str(row["role"]),
            partition=str(row["partition"]),
            sha256=str(row["sha256"]),
            license_record_id=str(row["license_record_id"]),
            source_dataset=str(row["source_dataset"]),
            dataset_version=str(row["dataset_version"]),
            creator_id=str(row["creator_id"]),
            uploader_id=str(row["uploader_id"]),
            attribution=str(row["attribution"]),
            source_license_spdx=str(row["source_license_spdx"]),
            source_license_url=str(row["source_license_url"]),
        )
        grouped[(planned.partition, planned.role, planned.clip.label)].append(planned)
    for values in grouped.values():
        values.sort(key=lambda source: source.clip.source_id)
    if len(seen_ids) != receipt.get("source_count"):
        raise ValueError("receipt source count mismatch")
    return receipt, grouped


def _family_candidate(labels: Sequence[str], seed: int, split: str, index: int, attempt: int) -> FamilyLabels:
    rng = np.random.default_rng(stable_seed(seed, "v5-label-family", split, index, attempt))
    choice = [labels[int(i)] for i in rng.permutation(len(labels))[:5]]
    return FamilyLabels(
        repeat_label=choice[0],
        base_answer_label=choice[1],
        swapped_answer_label=choice[2],
        extra_labels=(choice[3], choice[4]),
    )


def make_label_schedules(
    *, seen_labels: Sequence[str], heldout_labels: Sequence[str],
    family_counts: Mapping[str, int], seed: int,
) -> Dict[str, List[FamilyLabels]]:
    if len(seen_labels) < 5 or len(heldout_labels) < 5:
        raise ValueError("v5 needs at least five seen and five held-out labels")
    result: Dict[str, List[FamilyLabels]] = {split: [] for split in SPLITS}
    used_train: set[Tuple[str, ...]] = set()
    train_pairs: set[str] = set()
    train_triplets: set[str] = set()
    for index in range(family_counts["train"]):
        for attempt in range(100_000):
            candidate = _family_candidate(seen_labels, seed, "train", index, attempt)
            identity = candidate.distinct_labels
            if identity not in used_train:
                break
        else:
            raise RuntimeError("could not create unique training label families")
        used_train.add(identity)
        result["train"].append(candidate)
        train_pairs.add(candidate.pair_signature)
        train_triplets.add(candidate.triplet_signature)

    # IID/calibration deliberately reuse training role compositions with new
    # source files, timings, overlap and nuisance realizations.
    for split in ("val", "test_iid"):
        result[split] = [
            result["train"][index % len(result["train"])]
            for index in range(family_counts[split])
        ]

    used_ood: set[Tuple[str, ...]] = set()
    for index in range(family_counts["test_compositional_ood"]):
        for attempt in range(200_000):
            candidate = _family_candidate(
                seen_labels, seed, "test_compositional_ood", index, attempt
            )
            if (
                candidate.pair_signature not in train_pairs
                and candidate.triplet_signature not in train_triplets
                and candidate.distinct_labels not in used_ood
            ):
                break
        else:
            raise RuntimeError("could not reserve enough compositional-OOD families")
        used_ood.add(candidate.distinct_labels)
        result["test_compositional_ood"].append(candidate)

    for index in range(family_counts["test_label_ood"]):
        result["test_label_ood"].append(
            _family_candidate(heldout_labels, seed, "test_label_ood", index, 0)
        )
    return result


def _source_for(
    grouped: Mapping[Tuple[str, str, str], Sequence[PlannedSource]],
    *, split: str, role: str, label: str, family_index: int, occurrence: int,
    require_distinct: bool,
) -> PlannedSource:
    candidates = grouped.get((split, role, label), ())
    if not candidates:
        raise KeyError(f"no {split}/{role}/{label} source")
    if require_distinct and len(candidates) < 2:
        raise ValueError(
            f"paper profile needs >=2 distinct {split}/{label} sources"
        )
    offset = occurrence % len(candidates)
    return candidates[(family_index + offset) % len(candidates)]


def _choose_template(
    *, partition: str, relation: str, seed: int, key: str,
) -> Tuple[str, str]:
    templates = TEMPLATES[partition][relation]
    return templates[stable_seed(seed, "v5-template", partition, relation, key) % len(templates)]


def _answer_options(
    *, answer: str, relation: str, candidates: Sequence[str], vocabulary: Sequence[str],
    sample_id: str, seed: int, forced: Sequence[str] | None = None,
) -> List[str]:
    if forced is not None:
        options = list(forced)
        if len(options) != 5 or len(set(options)) != 5 or answer not in options:
            raise ValueError("forced answer options are invalid")
        return options
    required = [] if answer == NO_EVIDENCE_ANSWER else [answer]
    if relation == "first":
        required = list(dict.fromkeys(candidates))
        if len(required) != 2 or (
            answer != NO_EVIDENCE_ANSWER and answer not in required
        ):
            raise ValueError("first options need both candidates")
    pool = [label for label in vocabulary if label not in set(required)]
    rng = np.random.default_rng(stable_seed(seed, "v5-options", sample_id))
    chosen = [pool[int(i)] for i in rng.permutation(len(pool))[: 4 - len(required)]]
    options = required + chosen + [NO_EVIDENCE_ANSWER]
    wrong = [option for option in options if option != answer]
    wrong = [wrong[int(i)] for i in rng.permutation(len(wrong))]
    position = stable_seed(seed, "v5-answer-position", sample_id) % 5
    wrong.insert(position, answer)
    if len(wrong) != 5 or len(set(wrong)) != 5:
        raise AssertionError("answer-option construction failed")
    return wrong


def _template_text(partition: str, relation: str, template_id: str) -> str:
    for candidate_id, template in TEMPLATES[partition][relation]:
        if candidate_id == template_id:
            return template
    raise KeyError(
        f"unknown {partition}/{relation} paraphrase family: {template_id}"
    )


def _paired_control_options(
    previous: Sequence[str], *, answer: str, control_group: str, seed: int
) -> List[str]:
    """Independently permute a reversed first-control option list."""

    permutation_rng = np.random.default_rng(
        stable_seed(seed, "v5-paired-option-permutation", control_group)
    )
    options = [
        previous[int(index)]
        for index in permutation_rng.permutation(len(previous))
    ]
    if options == list(previous) or options.index(answer) == list(previous).index(answer):
        options = options[1:] + options[:1]
    return options


def _directional_semantic_parts(
    semantics_id: str, relation: str
) -> Tuple[str, int]:
    prefix = f"{relation}:"
    if not semantics_id.startswith(prefix):
        raise ValueError(
            f"directional semantic {semantics_id!r} does not match {relation}"
        )
    label, marker, ordinal_text = semantics_id[len(prefix) :].rpartition(
        ":ordinal="
    )
    if marker != ":ordinal=" or not label or not ordinal_text.isdigit():
        raise ValueError(f"malformed directional semantic: {semantics_id}")
    ordinal = int(ordinal_text)
    if ordinal <= 0:
        raise ValueError(f"non-positive semantic ordinal: {semantics_id}")
    return label, ordinal


def _first_semantic_parts(semantics_id: str) -> Tuple[str, str]:
    if not semantics_id.startswith("first:"):
        raise ValueError(f"malformed first semantic: {semantics_id}")
    labels = semantics_id[len("first:") :].split("|")
    if len(labels) != 2 or not all(labels) or labels != sorted(labels):
        raise ValueError(f"malformed/canonical first semantic: {semantics_id}")
    return labels[0], labels[1]


def _semantic_label_counts(record: Mapping[str, Any]) -> Counter[str]:
    return Counter(
        str(event["label"])
        for event in record["events"]
        if event["event_kind"] == "semantic"
    )


def _negative_semantic_is_feasible(
    record: Mapping[str, Any], semantics_id: str, relation: str
) -> bool:
    present = _semantic_label_counts(record)
    if relation in {"after", "before"}:
        label, ordinal = _directional_semantic_parts(semantics_id, relation)
        return present[label] < ordinal
    left, right = _first_semantic_parts(semantics_id)
    return left not in present and right not in present


def _semantic_balance_summary(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_relation: Dict[str, Dict[str, Any]] = {}
    overall_polarity: Dict[str, set[bool]] = defaultdict(set)
    overall_rows: Counter[str] = Counter()
    for relation in ("after", "before", "first"):
        polarity: Dict[str, set[bool]] = defaultdict(set)
        rows: Counter[str] = Counter()
        for record in records:
            if record["relation"] != relation:
                continue
            semantics_id = str(record["question_semantics_id"])
            status = bool(record["no_evidence"])
            polarity[semantics_id].add(status)
            rows[semantics_id] += 1
            overall_polarity[semantics_id].add(status)
            overall_rows[semantics_id] += 1
        matched = {key for key, values in polarity.items() if values == {False, True}}
        positive_only = {key for key, values in polarity.items() if values == {False}}
        negative_only = {key for key, values in polarity.items() if values == {True}}
        row_count = sum(rows.values())
        by_relation[relation] = {
            "matched_unique_semantics_up": len(matched),
            "positive_only_unique_semantics_down": len(positive_only),
            "negative_only_unique_semantics_down": len(negative_only),
            "matched_weighted_record_share_up": (
                sum(rows[key] for key in matched) / row_count if row_count else 0.0
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
    return {
        "matched_unique_semantics_up": len(matched),
        "positive_only_unique_semantics_down": len(positive_only),
        "negative_only_unique_semantics_down": len(negative_only),
        "matched_weighted_record_share_up": (
            sum(overall_rows[key] for key in matched) / total_rows
            if total_rows
            else 0.0
        ),
        "by_relation": by_relation,
    }


def _rewrite_negative_semantic_group(
    group: Sequence[MutableMapping[str, Any]],
    *,
    semantics_id: str,
    semantic_vocabulary: Sequence[str],
    seed: int,
) -> None:
    relation = str(group[0]["relation"])
    group_id = str(
        group[0]["surface_control_group_id"]
        if relation == "first"
        else group[0]["id"]
    )
    if relation in {"after", "before"}:
        if len(group) != 1:
            raise AssertionError("directional negative assignment must be record-local")
        record = group[0]
        label, ordinal = _directional_semantic_parts(semantics_id, relation)
        template = _template_text(
            str(record["template_partition"]),
            relation,
            str(record["paraphrase_family_id"]),
        )
        record.update(
            {
                "question_semantics_id": semantics_id,
                "question": template.format(
                    ordinal=_ordinal(ordinal), anchor=label
                ),
                "query_label": label,
                "query_instance_ordinal": ordinal,
                "absent_labels": [label],
            }
        )
        options = _answer_options(
            answer=NO_EVIDENCE_ANSWER,
            relation=relation,
            candidates=(),
            vocabulary=semantic_vocabulary,
            sample_id=str(record["id"]),
            seed=seed,
        )
        record["answer_options"] = options
        record["answer_option_index"] = options.index(NO_EVIDENCE_ANSWER)
        return

    if len(group) != 2 or {row["mention_order_variant"] for row in group} != {
        "forward",
        "reversed",
    }:
        raise AssertionError(f"incomplete first negative control group: {group_id}")
    if len({row["scene_id"] for row in group}) != 1:
        raise AssertionError(f"first negative control crosses scenes: {group_id}")
    canonical = list(_first_semantic_parts(semantics_id))
    if stable_seed(seed, "v5-balanced-first-order", group_id, semantics_id) % 2:
        canonical.reverse()
    ordered = {
        "forward": canonical,
        "reversed": list(reversed(canonical)),
    }
    by_variant = {str(row["mention_order_variant"]): row for row in group}
    for mention_variant in ("forward", "reversed"):
        record = by_variant[mention_variant]
        candidates = ordered[mention_variant]
        template = _template_text(
            str(record["template_partition"]),
            "first",
            str(record["paraphrase_family_id"]),
        )
        record.update(
            {
                "question_semantics_id": semantics_id,
                "question": template.format(left=candidates[0], right=candidates[1]),
                "query_candidate_labels": list(candidates),
                "absent_labels": list(candidates),
            }
        )
    forward = by_variant["forward"]
    reversed_record = by_variant["reversed"]
    forward_options = _answer_options(
        answer=NO_EVIDENCE_ANSWER,
        relation="first",
        candidates=forward["query_candidate_labels"],
        vocabulary=semantic_vocabulary,
        sample_id=str(forward["id"]),
        seed=seed,
    )
    reversed_options = _paired_control_options(
        forward_options,
        answer=NO_EVIDENCE_ANSWER,
        control_group=group_id,
        seed=seed,
    )
    forward["answer_options"] = forward_options
    forward["answer_option_index"] = forward_options.index(NO_EVIDENCE_ANSWER)
    reversed_record["answer_options"] = reversed_options
    reversed_record["answer_option_index"] = reversed_options.index(
        NO_EVIDENCE_ANSWER
    )


def _rebalance_negative_semantics(
    records: Sequence[MutableMapping[str, Any]],
    *,
    semantic_vocabulary: Sequence[str],
    seed: int,
) -> Dict[str, Any]:
    """Assign feasible split-local positive semantics to every negative.

    The scheduler operates only on metadata.  First-question forward/reversed
    surface controls are one atomic assignment unit, while the primary
    counterfactual negative is fixed by construction and never rewritten.
    """

    split_values = {str(record["split"]) for record in records}
    if len(split_values) != 1:
        raise ValueError("semantic balancing requires exactly one split")
    split = next(iter(split_values))
    positive_counts: Dict[str, Counter[str]] = {
        relation: Counter(
            str(record["question_semantics_id"])
            for record in records
            if record["relation"] == relation and not record["no_evidence"]
        )
        for relation in ("after", "before", "first")
    }
    negative_row_totals = Counter(
        str(record["relation"])
        for record in records
        if record["no_evidence"]
    )
    current_negative: Dict[str, Counter[str]] = {
        relation: Counter(
            str(record["question_semantics_id"])
            for record in records
            if record["relation"] == relation
            and record["no_evidence"]
            and record["primary_counterfactual_probe"]
        )
        for relation in ("after", "before", "first")
    }

    for relation in ("after", "before", "first"):
        pool = positive_counts[relation]
        if not pool:
            raise ValueError(f"{split}/{relation} has no positive semantic pool")
        if relation == "first":
            grouped: Dict[str, List[MutableMapping[str, Any]]] = defaultdict(list)
            for record in records:
                if (
                    record["relation"] == relation
                    and record["no_evidence"]
                    and not record["primary_counterfactual_probe"]
                ):
                    grouped[str(record["surface_control_group_id"])].append(record)
            groups = [(group_id, rows) for group_id, rows in grouped.items()]
        else:
            groups = [
                (str(record["id"]), [record])
                for record in records
                if record["relation"] == relation
                and record["no_evidence"]
                and not record["primary_counterfactual_probe"]
            ]

        planned: List[
            Tuple[str, List[MutableMapping[str, Any]], Tuple[str, ...]]
        ] = []
        for group_id, group in groups:
            representative = group[0]
            if any(
                row["scene_id"] != representative["scene_id"]
                or row["relation"] != relation
                for row in group
            ):
                raise AssertionError(f"invalid negative semantic group: {group_id}")
            feasible = tuple(
                semantics_id
                for semantics_id in sorted(pool)
                if _negative_semantic_is_feasible(
                    representative, semantics_id, relation
                )
            )
            if not feasible:
                raise ValueError(
                    f"no split-local positive semantic is feasible for "
                    f"{split}/{relation}/{group_id}"
                )
            planned.append((group_id, group, feasible))
        planned.sort(key=lambda item: (len(item[2]), item[0]))

        positive_rows = sum(pool.values())
        negative_rows = negative_row_totals[relation]
        target_odds = negative_rows / positive_rows
        for group_id, group, feasible in planned:
            def candidate_key(semantics_id: str) -> Tuple[Any, ...]:
                assigned = current_negative[relation][semantics_id]
                normalized_load = assigned / (
                    pool[semantics_id] * target_odds
                )
                return (
                    assigned > 0,
                    normalized_load,
                    stable_seed(
                        seed,
                        "v5-negative-semantic-balance",
                        split,
                        relation,
                        group_id,
                        semantics_id,
                    ),
                    semantics_id,
                )

            chosen = min(feasible, key=candidate_key)
            _rewrite_negative_semantic_group(
                group,
                semantics_id=chosen,
                semantic_vocabulary=semantic_vocabulary,
                seed=seed,
            )
            current_negative[relation][chosen] += len(group)

    positive_ids = {
        (relation, semantics_id)
        for relation, counts in positive_counts.items()
        for semantics_id in counts
    }
    unsupported_negative = sorted(
        {
            (str(record["relation"]), str(record["question_semantics_id"]))
            for record in records
            if record["no_evidence"]
        }
        - positive_ids
    )
    if unsupported_negative:
        raise AssertionError(
            f"negative semantics lack split-local positives: {unsupported_negative[:5]}"
        )
    return _semantic_balance_summary(records)


def _max_polyphony(events: Sequence[Mapping[str, Any]]) -> int:
    points = []
    for event in events:
        points.extend(
            (
                (float(event["onset_seconds"]), 1),
                (float(event["offset_seconds"]), -1),
            )
        )
    active = maximum = 0
    for _, delta in sorted(points, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _has_semantic_overlap(events: Sequence[Mapping[str, Any]]) -> bool:
    semantic = [event for event in events if event["event_kind"] == "semantic"]
    return any(
        max(float(a["onset_seconds"]), float(b["onset_seconds"]))
        < min(float(a["offset_seconds"]), float(b["offset_seconds"])) - 1e-9
        for index, a in enumerate(semantic)
        for b in semantic[index + 1 :]
    )


def _event_payload(
    *, event_id: str, source: PlannedSource, event_kind: str,
    crop_interval: Tuple[float, float], onset: float, offset: float,
    gain_db: float, occurrence_index: int, stem_path: Path,
    project_root: Path, staging_root: Path,
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "label": source.clip.label,
        "event_kind": event_kind,
        "source_dataset": source.source_dataset,
        "dataset_version": source.dataset_version,
        "source_id": source.clip.source_id,
        "creator_id": source.creator_id,
        "uploader_id": source.uploader_id,
        "attribution": source.attribution,
        "source_license_spdx": source.source_license_spdx,
        "source_license_url": source.source_license_url,
        "source_partition": source.partition,
        "source_path": relative_posix(source.clip.audio_path, project_root),
        "source_sha256": source.sha256,
        "license_record_id": source.license_record_id,
        "source_interval_seconds": list(source.clip.interval_seconds),
        "source_crop_interval_seconds": list(crop_interval),
        "onset_seconds": onset,
        "offset_seconds": offset,
        "gain_db": gain_db,
        "occurrence_index": occurrence_index,
        "stem_path": relative_posix(stem_path, staging_root),
    }


def _resolve_after_before(
    events: Sequence[Mapping[str, Any]], label: str, ordinal: int, relation: str
) -> tuple[List[str], List[str], str]:
    timeline = sorted(
        (event for event in events if event["event_kind"] == "semantic"),
        key=lambda event: (event["onset_seconds"], event["event_id"]),
    )
    occurrences = [event for event in timeline if event["label"] == label]
    if len(occurrences) < ordinal:
        return [], [], NO_EVIDENCE_ANSWER
    anchor = occurrences[ordinal - 1]
    index = next(i for i, event in enumerate(timeline) if event["event_id"] == anchor["event_id"])
    answer_index = index + (1 if relation == "after" else -1)
    if not 0 <= answer_index < len(timeline):
        # The builder avoids this case; treating a present terminal anchor as an
        # absent-anchor negative would violate the schema.
        raise ValueError("question plan selected a terminal anchor")
    answer = timeline[answer_index]
    return [anchor["event_id"]], [answer["event_id"]], str(answer["label"])


def _question_specs(
    *, events: Sequence[Mapping[str, Any]], labels: FamilyLabels,
    semantic_vocabulary: Sequence[str], template_partition: str,
    family_id: str, variant_id: str, seed: int,
) -> List[Dict[str, Any]]:
    semantic = sorted(
        (event for event in events if event["event_kind"] == "semantic"),
        key=lambda event: (event["onset_seconds"], event["event_id"]),
    )
    specs: List[Dict[str, Any]] = []

    def relation_spec(
        label: str, ordinal: int, relation: str, key: str,
        *, primary: bool = False, force_absent: bool = False,
    ) -> Dict[str, Any]:
        anchor_ids, answer_ids, answer = (
            ([], [], NO_EVIDENCE_ANSWER)
            if force_absent or (primary and variant_id == "anchor_drop")
            else _resolve_after_before(events, label, ordinal, relation)
        )
        # Primary anchor-drop is absent by construction.  For all other cases,
        # resolve normally so schema validation proves adjacency.
        no_evidence = answer == NO_EVIDENCE_ANSWER
        template_id, template = _choose_template(
            partition=template_partition, relation=relation, seed=seed,
            key=f"{family_id}:{key}",
        )
        return {
            "relation": relation,
            "question_type": f"temporal_{relation}",
            "question": template.format(ordinal=_ordinal(ordinal), anchor=label),
            "template_id": template_id,
            "semantics_id": f"{relation}:{label}:ordinal={ordinal}",
            "query_label": label,
            "query_ordinal": ordinal,
            "candidate_labels": [],
            "query_ids": anchor_ids,
            "anchor_ids": anchor_ids,
            "answer_ids": answer_ids,
            "answer": answer,
            "no_evidence": no_evidence,
            "no_evidence_reason": "absent_anchor" if no_evidence else None,
            "absent_labels": [label] if no_evidence else [],
            "primary": primary,
            "surface_control_group_id": None,
            "mention_order_variant": "not_applicable",
            "key": key,
        }

    specs.append(
        relation_spec(labels.repeat_label, 3, "after", "primary", primary=True)
    )

    # Five answerable adjacency probes. Their surfaces may vary across variants
    # and are not the controlled primary counterfactual.
    pairs = list(zip(semantic[:-1], semantic[1:]))
    after_pair_indices = np.linspace(0, len(pairs) - 1, 2, dtype=int)
    for slot, pair_index in enumerate(after_pair_indices):
        left, right = pairs[int(pair_index)]
        specs.append(
            relation_spec(
                str(left["label"]), int(left["occurrence_index"]), "after",
                f"dynamic-after-{slot}-{variant_id}",
            )
        )
    before_pair_indices = np.linspace(0, len(pairs) - 1, 3, dtype=int)
    for slot, pair_index in enumerate(before_pair_indices):
        left, right = pairs[int(pair_index)]
        specs.append(
            relation_spec(
                str(right["label"]), int(right["occurrence_index"]), "before",
                f"dynamic-before-{slot}-{variant_id}",
            )
        )

    present = {str(event["label"]) for event in semantic}
    absent_pool = [label for label in semantic_vocabulary if label not in present]
    if len(absent_pool) < 2:
        raise ValueError("semantic inventory needs two absent labels per scene")
    # Matched absent cases for both directional relations. They share the same
    # template partitions as positives, blocking relation -> answerability.
    specs.append(
        relation_spec(
            absent_pool[0], 1, "after", f"absent-after-{variant_id}",
            force_absent=True,
        )
    )
    specs.append(
        relation_spec(
            absent_pool[1], 1, "before", f"absent-before-{variant_id}",
            force_absent=True,
        )
    )

    # Three answerable first-relation semantics and one absent-candidate
    # semantics. Every semantic item is rendered twice with reversed mention
    # order. The paired records use the same audio/roles/answer but later receive
    # independently permuted answer options.
    earliest: Dict[str, Mapping[str, Any]] = {}
    for event in semantic:
        earliest.setdefault(str(event["label"]), event)
    candidate_pairs = list(itertools.combinations(sorted(earliest), 2))
    rng = np.random.default_rng(stable_seed(seed, "v5-first-pairs", family_id, variant_id))
    for slot, index in enumerate(rng.permutation(len(candidate_pairs))[:3]):
        left_label, right_label = candidate_pairs[int(index)]
        left = earliest[left_label]
        right = earliest[right_label]
        earlier, later = sorted(
            (left, right), key=lambda event: (event["onset_seconds"], event["event_id"])
        )
        forward = [left, right]
        if stable_seed(seed, "v5-first-order", family_id, variant_id, slot) % 2:
            forward.reverse()
        control_group = f"{family_id}:{variant_id}:first-control-{slot}"
        template_id, template = _choose_template(
            partition=template_partition, relation="first", seed=seed,
            key=control_group,
        )
        for mention_variant, candidate_order in (
            ("forward", forward),
            ("reversed", list(reversed(forward))),
        ):
            specs.append(
                {
                    "relation": "first",
                    "question_type": "temporal_first",
                    "question": template.format(
                        left=candidate_order[0]["label"],
                        right=candidate_order[1]["label"],
                    ),
                    "template_id": template_id,
                    "semantics_id": "first:" + "|".join(
                        sorted((left_label, right_label))
                    ),
                    "query_label": None,
                    "query_ordinal": None,
                    "candidate_labels": [event["label"] for event in candidate_order],
                    "query_ids": [event["event_id"] for event in candidate_order],
                    "anchor_ids": [later["event_id"]],
                    "answer_ids": [earlier["event_id"]],
                    "answer": earlier["label"],
                    "no_evidence": False,
                    "no_evidence_reason": None,
                    "absent_labels": [],
                    "primary": False,
                    "surface_control_group_id": control_group,
                    "mention_order_variant": mention_variant,
                    "key": f"first-{slot}-{variant_id}-{mention_variant}",
                }
            )

    absent_candidates = absent_pool[:2]
    absent_control_group = f"{family_id}:{variant_id}:first-control-absent"
    template_id, template = _choose_template(
        partition=template_partition,
        relation="first",
        seed=seed,
        key=absent_control_group,
    )
    for mention_variant, candidate_order in (
        ("forward", absent_candidates),
        ("reversed", list(reversed(absent_candidates))),
    ):
        specs.append(
            {
                "relation": "first",
                "question_type": "temporal_first",
                "question": template.format(
                    left=candidate_order[0], right=candidate_order[1]
                ),
                "template_id": template_id,
                "semantics_id": "first:" + "|".join(sorted(absent_candidates)),
                "query_label": None,
                "query_ordinal": None,
                "candidate_labels": list(candidate_order),
                "query_ids": [],
                "anchor_ids": [],
                "answer_ids": [],
                "answer": NO_EVIDENCE_ANSWER,
                "no_evidence": True,
                "no_evidence_reason": "absent_candidates",
                "absent_labels": list(candidate_order),
                "primary": False,
                "surface_control_group_id": absent_control_group,
                "mention_order_variant": mention_variant,
                "key": f"first-absent-{variant_id}-{mention_variant}",
            }
        )
    if len(specs) != QUESTIONS_PER_SCENE:
        raise AssertionError(f"question design produced {len(specs)} records")
    return specs


def _role_intervals(
    event_map: Mapping[str, Mapping[str, Any]], ids: Sequence[str]
) -> List[List[float]]:
    return [
        [float(event_map[event_id]["onset_seconds"]), float(event_map[event_id]["offset_seconds"])]
        for event_id in ids
    ]


def _compose_family(
    *, split: str, family_index: int, labels: FamilyLabels,
    grouped: Mapping[Tuple[str, str, str], Sequence[PlannedSource]],
    seen_labels: Sequence[str], heldout_labels: Sequence[str],
    nuisance_labels: Sequence[str], profile: str, seed: int,
    project_root: Path, staging_root: Path,
    storage_mode: str,
    semantic_crop_bank: Mapping[Tuple[str, str], Mapping[str, Any]] | None,
) -> Dict[str, List[Dict[str, Any]]]:
    family_id = f"family_{split}_{family_index:06d}"
    family_seed = stable_seed(seed, "qces-v5-family", split, family_index)
    rng = np.random.default_rng(family_seed)
    semantic_role = "semantic_heldout" if split == "test_label_ood" else "semantic_seen"
    require_distinct = profile in {"internal_scale", "paper"}

    # Seven semantic events: three instances of the anchor class plus four
    # distinct classes. Dropping the third instance makes the primary ordinal
    # query unanswerable while two same-class events remain in the scene.
    distinct = list(labels.distinct_labels)
    semantic_labels = [
        labels.repeat_label,
        labels.repeat_label,
        labels.repeat_label,
        *distinct[1:],
    ]
    sources: Dict[str, PlannedSource] = {}
    for event_index, label in enumerate(semantic_labels):
        occurrence = sum(previous == label for previous in semantic_labels[:event_index])
        sources[f"sem_{event_index:02d}"] = _source_for(
            grouped,
            split=split,
            role=semantic_role,
            label=label,
            family_index=family_index,
            occurrence=occurrence,
            require_distinct=require_distinct,
        )

    nuisance_sources: Dict[str, PlannedSource] = {}
    available_nuisance = [
        label for label in nuisance_labels if grouped.get((split, "nuisance", label))
    ]
    nuisance_count = 0 if not available_nuisance else family_index % 4
    for index in range(nuisance_count):
        label = available_nuisance[(family_index + index) % len(available_nuisance)]
        nuisance_sources[f"nui_{index:02d}"] = _source_for(
            grouped,
            split=split,
            role="nuisance",
            label=label,
            family_index=family_index,
            occurrence=0,
            require_distinct=False,
        )

    durations = {
        event_id: float(rng.uniform(*SEMANTIC_DURATION_RANGE))
        for event_id in sources
    }
    # Unique onset order with guaranteed semantic overlap between the two
    # repeated anchor events.
    onsets: Dict[str, float] = {}
    cursor = float(rng.uniform(0.25, 0.45))
    for index, event_id in enumerate(sources):
        onsets[event_id] = cursor
        if index == 0:
            cursor += min(0.58, durations[event_id] * 0.55)
        else:
            cursor += float(rng.uniform(0.68, 0.92))
    if max(onsets[event_id] + durations[event_id] for event_id in sources) > 8.2:
        raise AssertionError("semantic layout exceeded reserved timeline")

    clips: Dict[str, np.ndarray] = {}
    crops: Dict[str, Tuple[float, float]] = {}
    gains_db: Dict[str, float] = {}
    for event_id, source in {**sources, **nuisance_sources}.items():
        duration = durations.get(event_id, NUISANCE_DURATION_SECONDS)
        clip_rng = np.random.default_rng(stable_seed(family_seed, "crop", event_id))
        preferred_center: float | None = None
        if event_id.startswith("sem") and semantic_crop_bank is not None:
            bank_key = (source.clip.source_id, source.clip.label)
            bank_entry = semantic_crop_bank.get(bank_key)
            if bank_entry is None:
                if split in {"train", "val"}:
                    raise ValueError(
                        "semantic crop bank lacks required train/val source: "
                        f"{source.clip.label}::{source.clip.source_id}"
                    )
            else:
                if (
                    bank_entry.get("source_sha256") != source.sha256
                    or [float(value) for value in bank_entry.get("source_interval_seconds", [])]
                    != [float(value) for value in source.clip.interval_seconds]
                ):
                    raise ValueError(
                        "semantic crop bank source identity mismatch: "
                        f"{source.clip.label}::{source.clip.source_id}"
                    )
                preferred_center = float(bank_entry["selected"]["center_seconds"])
        clip, crop = crop_source_event(
            source.clip,
            duration,
            SAMPLE_RATE,
            FADE_MILLISECONDS,
            clip_rng,
            # FUSS source clips can hold a short labelled event surrounded by
            # long digital silence.  Nuisance sources keep a seeded-random crop
            # and a rare bank center can also land on a silent window, so both
            # need the maximum-variance recovery.  The recovery fires only when
            # the chosen window would otherwise be a hard silent failure, so a
            # usable bank center is always preserved exactly.
            recover_silent_random_crop=(profile == "paper"),
            preferred_center_seconds=preferred_center,
        )
        gain_db = float(rng.uniform(-4.0, 4.0)) if event_id.startswith("sem") else float(rng.uniform(-10.0, -2.0))
        clips[event_id] = (clip * 10.0 ** (gain_db / 20.0)).astype(np.float32)
        crops[event_id] = crop
        gains_db[event_id] = gain_db

    nuisance_onsets = {
        event_id: float(rng.uniform(0.05, DURATION_SECONDS - NUISANCE_DURATION_SECONDS - 0.05))
        for event_id in nuisance_sources
    }
    variant_onsets = {
        "base": dict(onsets),
        "order_swap": dict(onsets),
        "anchor_drop": dict(onsets),
    }
    variant_onsets["order_swap"]["sem_03"], variant_onsets["order_swap"]["sem_04"] = (
        variant_onsets["order_swap"]["sem_04"],
        variant_onsets["order_swap"]["sem_03"],
    )

    target_samples = int(round(SAMPLE_RATE * DURATION_SECONDS))
    raw_variants: Dict[str, Dict[str, np.ndarray]] = {}
    interval_variants: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for variant in VARIANTS:
        active_semantic = [event_id for event_id in sources if not (variant == "anchor_drop" and event_id == "sem_02")]
        stem_map: Dict[str, np.ndarray] = {}
        interval_map: Dict[str, Tuple[float, float]] = {}
        for event_id in [*active_semantic, *nuisance_sources]:
            canvas = np.zeros(target_samples, dtype=np.float32)
            onset = variant_onsets[variant].get(event_id, nuisance_onsets.get(event_id))
            if onset is None:
                raise AssertionError("event onset is missing")
            interval_map[event_id] = render_clip(
                canvas, clips[event_id], int(round(onset * SAMPLE_RATE)), SAMPLE_RATE
            )
            stem_map[event_id] = canvas
        raw_variants[variant] = stem_map
        interval_variants[variant] = interval_map

    maximum_peak = _maximum_pcm_render_peak(raw_variants)
    family_gain = min(1.0, HEADROOM / maximum_peak) if maximum_peak else 1.0
    for variant in VARIANTS:
        raw_variants[variant] = {
            event_id: waveform * family_gain
            for event_id, waveform in raw_variants[variant].items()
        }

    semantic_vocabulary = (
        list(dict.fromkeys([*seen_labels, *heldout_labels]))
        if profile == "smoke"
        else _split_semantic_vocabulary(
            split,
            seen_labels=seen_labels,
            heldout_labels=heldout_labels,
        )
    )
    template_partition = TEMPLATE_PARTITION_BY_SPLIT[split]
    primary_possible_answers = [
        labels.base_answer_label,
        labels.swapped_answer_label,
        NO_EVIDENCE_ANSWER,
    ]
    primary_distractors = [
        label for label in semantic_vocabulary if label not in primary_possible_answers
    ][:2]
    primary_options = primary_possible_answers + primary_distractors
    primary_rng = np.random.default_rng(stable_seed(seed, "v5-primary-options", family_id))
    primary_options = [primary_options[int(i)] for i in primary_rng.permutation(5)]

    records_by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for variant_index, variant in enumerate(VARIANTS):
        scene_id = f"scene_{split}_{family_index:06d}_{variant}"
        stems = raw_variants[variant]
        mixture = _sum(stems, stems)
        mixture_path = staging_root / "audio" / "mixture" / split / f"{scene_id}.wav"
        write_wav(mixture_path, mixture, SAMPLE_RATE)

        # Occurrence indices are defined after each intervention.
        order = sorted(
            stems,
            key=lambda event_id: (interval_variants[variant][event_id][0], event_id),
        )
        occurrence_counts: Dict[str, int] = defaultdict(int)
        occurrences: Dict[str, int] = {}
        for event_id in order:
            label = (sources | nuisance_sources)[event_id].clip.label
            occurrence_counts[label] += 1
            occurrences[event_id] = occurrence_counts[label]

        events: List[Dict[str, Any]] = []
        for event_id in order:
            source = (sources | nuisance_sources)[event_id]
            event_kind = "semantic" if event_id.startswith("sem") else "nuisance"
            event_path = staging_root / "audio" / "events" / split / scene_id / f"{event_id}.wav"
            write_wav(event_path, stems[event_id], SAMPLE_RATE)
            onset, offset = interval_variants[variant][event_id]
            events.append(
                _event_payload(
                    event_id=event_id,
                    source=source,
                    event_kind=event_kind,
                    crop_interval=crops[event_id],
                    onset=onset,
                    offset=offset,
                    gain_db=gains_db[event_id],
                    occurrence_index=occurrences[event_id],
                    stem_path=event_path,
                    project_root=project_root,
                    staging_root=staging_root,
                )
            )
        event_map = {event["event_id"]: event for event in events}
        specs = _question_specs(
            events=events,
            labels=labels,
            semantic_vocabulary=semantic_vocabulary,
            template_partition=template_partition,
            family_id=family_id,
            variant_id=variant,
            seed=seed,
        )
        records: List[Dict[str, Any]] = []
        first_control_options: Dict[str, List[str]] = {}
        for question_index, spec in enumerate(specs):
            sample_id = f"{split}_{family_index:06d}_{variant_index}_{question_index:02d}"
            evidence_ids = list(dict.fromkeys(spec["anchor_ids"] + spec["answer_ids"]))
            output_paths: Dict[str, Path] = {}
            if storage_mode == MATERIALIZED_STORAGE_MODE:
                evidence = (
                    _sum(stems, evidence_ids)
                    if evidence_ids
                    else np.zeros(target_samples, dtype=np.float32)
                )
                waveforms = {
                    "evidence": evidence,
                    "residual": mixture - evidence,
                    "anchor": (
                        _sum(stems, spec["anchor_ids"])
                        if spec["anchor_ids"]
                        else np.zeros(target_samples, dtype=np.float32)
                    ),
                    "answer": (
                        _sum(stems, spec["answer_ids"])
                        if spec["answer_ids"]
                        else np.zeros(target_samples, dtype=np.float32)
                    ),
                }
                output_paths = {
                    name: staging_root / "audio" / name / split / f"{sample_id}.wav"
                    for name in waveforms
                }
                for name, waveform in waveforms.items():
                    write_wav(output_paths[name], waveform, SAMPLE_RATE)
            options = _answer_options(
                answer=str(spec["answer"]),
                relation=str(spec["relation"]),
                candidates=spec["candidate_labels"],
                vocabulary=semantic_vocabulary,
                sample_id=sample_id,
                seed=seed,
                forced=primary_options if spec["primary"] else None,
            )
            control_group = spec["surface_control_group_id"]
            if control_group is not None:
                previous = first_control_options.get(control_group)
                if previous is not None:
                    options = _paired_control_options(
                        previous,
                        answer=str(spec["answer"]),
                        control_group=str(control_group),
                        seed=seed,
                    )
                first_control_options[control_group] = list(options)
            intervention = {
                "base": {
                    "kind": "none",
                    "parent_variant_id": None,
                    "intervened_event_ids": [],
                },
                "order_swap": {
                    "kind": "onset_swap",
                    "parent_variant_id": "base",
                    "intervened_event_ids": ["sem_03", "sem_04"],
                },
                "anchor_drop": {
                    "kind": "event_drop",
                    "parent_variant_id": "base",
                    "intervened_event_ids": ["sem_02"],
                },
            }[variant]
            tags = ["same_label_instances", "semantic_overlap", "polyphonic_scene"]
            if spec["primary"]:
                tags.append("hard_temporal_counterfactual")
            if variant != "base":
                tags.append(intervention["kind"])
            if spec["no_evidence"]:
                tags.append(str(spec["no_evidence_reason"]))
            record = {
                "schema_version": (
                    DERIVED_SCHEMA_VERSION
                    if storage_mode == DERIVED_STORAGE_MODE
                    else SCHEMA_VERSION
                ),
                "id": sample_id,
                "scene_id": scene_id,
                "scene_family_id": family_id,
                "variant_id": variant,
                "counterfactual_intervention": intervention,
                "question_semantics_id": str(spec["semantics_id"]),
                "counterfactual_group_id": (
                    f"{family_id}:primary"
                    if spec["primary"]
                    else (
                        str(control_group)
                        if control_group is not None
                        else f"{scene_id}:q{question_index:02d}"
                    )
                ),
                "paraphrase_family_id": str(spec["template_id"]),
                "template_partition": template_partition,
                "question_index": question_index,
                "split": split,
                "evaluation_axis": EVALUATION_AXIS_BY_SPLIT[split],
                "sample_rate": SAMPLE_RATE,
                "num_channels": 1,
                "num_samples": target_samples,
                "duration_seconds": DURATION_SECONDS,
                "mixture_path": relative_posix(mixture_path, staging_root),
                "question": str(spec["question"]),
                "answer": str(spec["answer"]),
                "answer_options": options,
                "answer_option_index": options.index(spec["answer"]),
                "question_type": str(spec["question_type"]),
                "relation": str(spec["relation"]),
                "no_evidence": bool(spec["no_evidence"]),
                "no_evidence_reason": spec["no_evidence_reason"],
                "absent_labels": list(spec["absent_labels"]),
                "query_label": spec["query_label"],
                "query_instance_ordinal": spec["query_ordinal"],
                "query_candidate_labels": list(spec["candidate_labels"]),
                "query_event_ids": list(spec["query_ids"]),
                "surface_control_group_id": control_group,
                "mention_order_variant": spec["mention_order_variant"],
                "events": events,
                "anchor_event_ids": list(spec["anchor_ids"]),
                "answer_event_ids": list(spec["answer_ids"]),
                "evidence_event_ids": evidence_ids,
                "anchor_intervals": _role_intervals(event_map, spec["anchor_ids"]),
                "answer_intervals": _role_intervals(event_map, spec["answer_ids"]),
                "source_group_ids": sorted({event["source_id"] for event in events}),
                "primary_counterfactual_probe": bool(spec["primary"]),
                "composition_pair_signature": labels.pair_signature,
                "composition_triplet_signature": labels.triplet_signature,
                "same_label_repeat": len(
                    [event for event in events if event["event_kind"] == "semantic"]
                ) != len(
                    {event["label"] for event in events if event["event_kind"] == "semantic"}
                ),
                "semantic_overlap": _has_semantic_overlap(events),
                "max_polyphony": _max_polyphony(events),
                "hard_case_tags": tags,
                "render_recipe_id": "qces-v5-overlap-gain-v1",
                "mixture_peak": float(np.max(np.abs(mixture))),
                "family_gain": family_gain,
                "generation_seed": family_seed,
            }
            if storage_mode == DERIVED_STORAGE_MODE:
                record["storage_mode"] = DERIVED_STORAGE_MODE
            else:
                record.update(
                    {
                        "evidence_stem_path": relative_posix(
                            output_paths["evidence"], staging_root
                        ),
                        "residual_stem_path": relative_posix(
                            output_paths["residual"], staging_root
                        ),
                        "anchor_stem_path": relative_posix(
                            output_paths["anchor"], staging_root
                        ),
                        "answer_stem_path": relative_posix(
                            output_paths["answer"], staging_root
                        ),
                    }
                )
            records.append(record)
        records_by_variant[variant] = records
    return records_by_variant


def build_dataset(args: argparse.Namespace) -> None:
    project_root = args.project_root.resolve()
    semantic_crop_bank, semantic_crop_bank_identity = _load_semantic_crop_bank(
        getattr(args, "semantic_crop_bank", None)
    )
    storage_mode = resolve_storage_mode(args.profile, getattr(args, "storage_mode", None))
    receipt_path = args.source_receipt or (
        project_root
        / "AudioTime-recovered"
        / "train5000_timestamp"
        / f"qces_v5_{args.profile}_source_receipt.json"
    )
    output_root = (args.output_root or project_root / "data" / f"qces_v5_{args.profile}").resolve()
    staging_root = output_root.with_name(output_root.name + ".building")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_root}; use --overwrite")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    try:
        receipt, grouped = _load_receipt(receipt_path.resolve(), project_root, args.profile)
        selection = receipt["selection"]
        seen_labels = tuple(selection["seen_labels"])
        heldout_labels = tuple(selection["heldout_labels"])
        nuisance_labels = tuple(selection["nuisance_labels"])
        build_splits = tuple(args.build_splits)
        if "train" not in build_splits:
            raise ValueError("--build-splits must include train")
        family_counts = dict(PROFILE_FAMILY_COUNTS[args.profile])
        debug_family_indices: dict[str, list[int]] | None = None
        debug_family_indices_identity: dict[str, Any] | None = None
        debug_family_indices_arg = getattr(args, "debug_family_indices", None)
        if debug_family_indices_arg is not None:
            if args.families_per_split is not None:
                raise ValueError(
                    "--debug-family-indices and --families-per-split are mutually exclusive"
                )
            debug_path = debug_family_indices_arg.expanduser().resolve()
            payload = json.loads(debug_path.read_text(encoding="utf-8"))
            if (
                not isinstance(payload, dict)
                or payload.get("format")
                not in {
                    "qces_v5_debug_family_indices_v1",
                    "qces_v5_debug_family_index_lists_v1",
                }
                or not isinstance(payload.get("indices"), dict)
                or set(payload["indices"]) != set(SPLITS)
            ):
                raise ValueError("invalid debug family-index selection")
            debug_family_indices = {}
            for split in SPLITS:
                raw_indices = payload["indices"][split]
                if isinstance(raw_indices, list):
                    indices = raw_indices
                else:
                    indices = [raw_indices]
                if not indices:
                    raise ValueError(f"debug family index list is empty for {split}")
                normalized: list[int] = []
                for index in indices:
                    if (
                        isinstance(index, bool)
                        or not isinstance(index, int)
                        or not 0 <= index < PROFILE_FAMILY_COUNTS[args.profile][split]
                    ):
                        raise ValueError(f"debug family index is invalid for {split}")
                    normalized.append(index)
                if len(set(normalized)) != len(normalized):
                    raise ValueError(f"debug family indices are duplicated for {split}")
                debug_family_indices[split] = normalized
            debug_family_indices_identity = {
                "path": str(debug_path),
                "sha256": sha256_file(debug_path),
                "indices": debug_family_indices,
                "purpose": payload.get("purpose"),
            }
            family_counts = {
                split: len(debug_family_indices[split]) for split in SPLITS
            }
        if args.families_per_split is not None:
            if args.families_per_split <= 0:
                raise ValueError("--families-per-split must be positive")
            family_counts = {split: args.families_per_split for split in SPLITS}
        family_counts = {
            split: family_counts[split] if split in build_splits else 0
            for split in SPLITS
        }
        schedule_counts = (
            dict(PROFILE_FAMILY_COUNTS[args.profile])
            if debug_family_indices is not None
            else family_counts
        )
        schedules = make_label_schedules(
            seen_labels=seen_labels,
            heldout_labels=heldout_labels,
            family_counts=schedule_counts,
            seed=args.seed,
        )
        records_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for split in build_splits:
            selected_schedule = (
                [
                    (index, schedules[split][index])
                    for index in debug_family_indices[split]
                ]
                if debug_family_indices is not None
                else list(enumerate(schedules[split]))
            )
            for display_index, (family_index, labels) in enumerate(
                selected_schedule, 1
            ):
                variants = _compose_family(
                    split=split,
                    family_index=family_index,
                    labels=labels,
                    grouped=grouped,
                    seen_labels=seen_labels,
                    heldout_labels=heldout_labels,
                    nuisance_labels=nuisance_labels,
                    profile=args.profile,
                    seed=args.seed,
                    project_root=project_root,
                    staging_root=staging_root,
                    storage_mode=storage_mode,
                    semantic_crop_bank=semantic_crop_bank,
                )
                for variant in VARIANTS:
                    records_by_split[split].extend(variants[variant])
                print(
                    f"built {split} family {display_index}/{len(selected_schedule)} "
                    f"(original index {family_index})"
                )
        # A reduced ``--families-per-split`` build is a mechanism smoke, even
        # when it reuses the paper source route. It cannot satisfy the full
        # split-local semantic coverage contract and must never be marked paper
        # eligible. The complete fixed profile retains the strict gate.
        strict_semantic_balance = (
            args.profile in {"internal_scale", "paper"}
            and args.families_per_split is None
            and debug_family_indices is None
        )
        semantic_balance_by_split: Dict[str, Dict[str, Any]] = {}
        for split in build_splits:
            split_semantic_vocabulary = _split_semantic_vocabulary(
                split,
                seen_labels=seen_labels,
                heldout_labels=heldout_labels,
            )
            if strict_semantic_balance:
                semantic_balance_by_split[split] = _rebalance_negative_semantics(
                    records_by_split[split],
                    semantic_vocabulary=split_semantic_vocabulary,
                    seed=args.seed,
                )
            else:
                semantic_balance_by_split[split] = _semantic_balance_summary(
                    records_by_split[split]
                )
            for record in records_by_split[split]:
                parse_qces_v5_record(record)
            write_jsonl(staging_root / f"qces_{split}.jsonl", records_by_split[split])
        all_records = [
            record for split in build_splits for record in records_by_split[split]
        ]
        write_jsonl(staging_root / "qces_all.jsonl", all_records)

        source_planner = (
            CODE_ROOT / "mixi_understanding" / "data" / "qces_v5_source_ledger.py"
            if receipt["format"] == AUDITED_RECEIPT_FORMAT
            else CODE_ROOT
            / "mixi_understanding"
            / "scripts"
            / "plan_qces_v5_sources.py"
        )
        source_files = {
            "builder": Path(__file__).resolve(),
            "schema": CODE_ROOT / "mixi_understanding" / "data" / "qces_v5_schema.py",
            "validator": CODE_ROOT / "mixi_understanding" / "scripts" / "validate_qces_v5_dataset.py",
            "source_planner": source_planner,
        }
        scenes_by_split = {
            split: family_counts[split] * len(VARIANTS) for split in SPLITS
        }
        stored_wavs = sorted((staging_root / "audio").rglob("*.wav"))
        stored_audio_bytes = sum(path.stat().st_size for path in stored_wavs)
        debug_family_counts = (
            {
                split: family_counts[split]
                for split in SPLITS
            }
            if debug_family_indices is not None
            else None
        )
        debug_override: int | dict[str, int] | None = (
            (
                next(iter(debug_family_counts.values()))
                if len(set(debug_family_counts.values())) == 1
                else debug_family_counts
            )
            if debug_family_counts is not None
            else args.families_per_split
        )
        config = {
            "schema_version": (
                DERIVED_SCHEMA_VERSION
                if storage_mode == DERIVED_STORAGE_MODE
                else SCHEMA_VERSION
            ),
            "builder_version": BUILDER_VERSION,
            "profile": args.profile,
            "build_splits": list(build_splits),
            "debug_families_per_split_override": debug_override,
            "debug_family_indices": debug_family_indices_identity,
            "seed": args.seed,
            "sample_rate": SAMPLE_RATE,
            "num_channels": 1,
            "num_samples": int(SAMPLE_RATE * DURATION_SECONDS),
            "duration_seconds": DURATION_SECONDS,
            "audio_format": {"container": "WAV", "subtype": "PCM_16"},
            "storage": {
                "mode": storage_mode,
                "mixture_stored_once_per_scene": True,
                "event_stems_stored_once_per_scene": True,
                "question_role_stems_materialized": (
                    storage_mode == MATERIALIZED_STORAGE_MODE
                ),
                "derived_evidence_recipe": (
                    "sum(evidence_event_ids)"
                    if storage_mode == DERIVED_STORAGE_MODE
                    else None
                ),
                "derived_residual_recipe": (
                    "mixture-evidence"
                    if storage_mode == DERIVED_STORAGE_MODE
                    else None
                ),
                "stored_audio_file_count": len(stored_wavs),
                "stored_audio_bytes": stored_audio_bytes,
            },
            "counts": {
                "scene_families": sum(family_counts.values()),
                "scene_families_by_split": family_counts,
                "scenes": sum(scenes_by_split.values()),
                "scenes_by_split": scenes_by_split,
                "questions_per_scene": QUESTIONS_PER_SCENE,
                "records": len(all_records),
                "records_by_split": {
                    split: len(records_by_split[split]) for split in SPLITS
                },
            },
            "question_design": {
                "relations": ["after", "before", "first"],
                "answer_options": 5,
                "no_evidence_ratio_bounds": [0.20, 0.40],
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
            },
            "semantic_balance": {
                "strict_profile_gate": strict_semantic_balance,
                "paper_eligibility": (
                    "eligible"
                    if strict_semantic_balance
                    else "not_paper_eligible"
                ),
                "train_to_val_exact_semantics_shortcut_gate": (
                    {
                        "unseen_semantics_backoff": "train_relation_prior",
                        "maximum_auroc_down": 0.60,
                        "maximum_best_balanced_accuracy_down": 0.60,
                    }
                    if strict_semantic_balance
                    else None
                ),
                "summary_by_split": semantic_balance_by_split,
            },
            "composition": {
                "seen_labels": list(seen_labels),
                "heldout_labels": list(heldout_labels),
                "nuisance_labels": list(nuisance_labels),
                "semantic_events_per_base_scene": [7, 7],
                "semantic_events_after_anchor_drop": [6, 6],
                "nuisance_events_per_scene": [0, 3],
                "semantic_duration_seconds": list(SEMANTIC_DURATION_RANGE),
                "nuisance_duration_seconds": NUISANCE_DURATION_SECONDS,
                "source_crop_policy": (
                    (
                        "semantic events use the immutable label-conditioned crop "
                        "bank center; nuisance events remain seeded-random"
                    )
                    if semantic_crop_bank is not None
                    else (
                        "seeded_random; paper profile deterministically falls back "
                        "to the maximum-variance valid window only when the random "
                        "crop would be rejected as effectively silent"
                    )
                ),
                "semantic_crop_bank": semantic_crop_bank_identity,
                "semantic_crop_curator_is_independent_evaluator": False,
                "same_label_repeat_required_every_scene": True,
                "semantic_overlap_required_every_scene": True,
                "primary_pair_and_triplet_disjoint_train_vs_compositional_ood": True,
                "heldout_labels_exclusive_to_label_ood": True,
                "semantic_question_and_option_labels_split_local": (
                    strict_semantic_balance
                ),
                "paper_requires_at_least_two_sources_for_repeated_instances": args.profile in {"internal_scale", "paper"},
                "render_recipe_id": "qces-v5-overlap-gain-v1",
                "headroom": HEADROOM,
            },
            "split_protocol": {
                "splits": list(build_splits),
                "source_file_disjoint": True,
                "scene_family_disjoint": True,
                "template_partitions": {
                    split: TEMPLATE_PARTITION_BY_SPLIT[split] for split in SPLITS
                },
                "evaluation_axis": {
                    split: EVALUATION_AXIS_BY_SPLIT[split] for split in SPLITS
                },
            },
            "source": {
                "receipt_path": relative_posix(receipt_path.resolve(), project_root),
                "receipt_sha256": sha256_file(receipt_path.resolve()),
                "receipt_format": receipt["format"],
                "source_route": receipt.get("source_route", "audiotime_internal"),
                "dataset": receipt.get("dataset", receipt.get("audio_dataset", {}).get("name")),
                "dataset_revision": receipt.get(
                    "dataset_revision", receipt.get("audio_dataset", {}).get("version")
                ),
                "metadata_path": receipt.get("metadata_path", receipt.get("ledger_path")),
                "metadata_sha256": receipt.get(
                    "metadata_sha256", receipt.get("ledger_sha256")
                ),
                "selection_path": receipt.get("selection_path"),
                "selection_sha256": receipt.get("selection_sha256"),
                "license_record": receipt["license_record"],
            },
            "release_policy": {
                "audio_redistribution_ready": bool(
                    receipt["license_record"]["status"] == "verified"
                    and receipt["license_record"]["redistribution_allowed"]
                ),
                "unverified_license_blocks_audio_release": True,
                "paper_route_requires_audited_generic_receipt": True,
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
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        from mixi_understanding.scripts.validate_qces_v5_dataset import validate_dataset

        report = validate_dataset(staging_root, write_report=True)
        replace_validated_output(staging_root, output_root)
        print(f"dataset ready: {output_root}")
        print(f"artifact fingerprint: {report['artifact_fingerprint_sha256']}")
    except Exception:
        print(f"build failed; staging retained at {staging_root}", file=sys.stderr)
        raise


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    build_dataset(args)


if __name__ == "__main__":
    main()
