"""Leakage-safe synthetic evidence scenes for the final Q-DOR benchmark.

This module is intentionally a new sidecar.  It does not import or mutate any
historical QCES scene builder.  Its input contract is a *clean single-event
source bank*: every row identifies one audible labelled event, its exact active
source interval, immutable source identities, and its official AudioSet split.

The builder has three unusual but deliberate properties.

* AudioSet ``train`` source groups are partitioned globally into train/dev,
  while official ``eval`` groups are locked to test.  Video, audio hash, path,
  and source id are unioned before partitioning, so aliases cannot leak.
* Core scheduling places every ontology label once at the beginning, middle,
  and end of a scene in each round.  This makes boundary NONE queries and
  internal positive queries available for every label without using model
  predictions to select examples.
* Headline QA rows are paired within the exact key
  ``(relation, anchor_label, anchor_ordinal)``.  Equal numbers of positive and
  NONE examples are retained for every observed key.  Therefore a text-only
  majority classifier using any subset of those fields has balanced accuracy
  exactly 0.5 (up to floating-point arithmetic).

The scene manifest uses the fields consumed by
``export_qces_detector_dense_features.py``: ``scene_id``, ``split``,
``mixture_path``, ``duration_seconds``, ``sample_rate`` and ``events``.  An
explicit QA manifest is emitted alongside it because regenerating every
possible adjacency question would destroy the conditional-balance guarantee.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import tempfile
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


FORMAT = "qces_qdor_clean_evidence_scenes_v1"
SCENE_FORMAT = "qces_qdor_clean_evidence_scene_v1"
QA_FORMAT = "qces_qdor_clean_evidence_qa_v1"
SOURCE_FORMAT = "qces_clean_single_event_source_v1"
FRAME_HOP_SECONDS = 0.04
SCENE_SECONDS = 10.0
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_SEED = 2037
SPLITS = ("train", "dev", "test")
RELATIONS = ("before", "after")


def _stable_digest(seed: int, *values: Any) -> str:
    body = "\0".join([FORMAT, str(seed), *(str(value) for value in values)])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _stable_order(values: Iterable[Any], *, seed: int, namespace: str) -> list[Any]:
    return sorted(values, key=lambda value: _stable_digest(seed, namespace, value))


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return value
    return default


def _finite_float(value: Any, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a finite number") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{context} must be a finite number")
    return parsed


def _bool_value(value: Any, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {
        "true", "false", "yes", "no", "pass", "fail", "passed", "failed",
    }:
        return value.strip().lower() in {"true", "yes", "pass", "passed"}
    raise ValueError(f"{context} must be an explicit boolean")


def _normalize_official_split(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"train", "audioset_train", "balanced_train", "unbalanced_train"}:
        return "train"
    if normalized in {"eval", "evaluation", "test", "audioset_eval", "audioset_test"}:
        return "eval"
    raise ValueError(f"source split must be official AudioSet train/eval, got {value!r}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CleanSource:
    """Canonical single-event source row.

    ``active_*`` coordinates address the source audio file, not a pre-rendered
    scene.  Long events may be deterministically sub-cropped at render time;
    the original strong interval always remains in provenance metadata.
    """

    source_id: str
    label: str
    official_split: str
    video_id: str
    source_sha256: str
    audio_path: str
    active_onset_seconds: float
    active_offset_seconds: float
    cleanliness_passed: bool
    audibility_passed: bool
    cleanliness_tier: str
    audibility_score: float | None
    provenance: Mapping[str, Any]
    assigned_split: str = ""

    @property
    def duration_seconds(self) -> float:
        return self.active_offset_seconds - self.active_onset_seconds

    @property
    def hard_identities(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (name, value)
            for name, value in (
                ("source_id", self.source_id),
                ("video_id", self.video_id),
                ("source_sha256", self.source_sha256),
                ("audio_path", self.audio_path),
            )
            if value
        )


@dataclass(frozen=True)
class SceneRecipe:
    scene_id: str
    split: str
    sources: tuple[CleanSource, ...]
    recipe_kind: str
    round_index: int


@dataclass(frozen=True)
class QARow:
    item_id: str
    scene_id: str
    split: str
    relation: str
    question: str
    answer: str
    no_evidence: bool
    no_evidence_reason: str | None
    anchor_label: str
    anchor_ordinal: int
    answer_label: str | None
    anchor_event_id: str
    answer_event_id: str | None
    evidence_event_ids: tuple[str, ...]
    gold_anchor_interval: tuple[float, float]
    gold_answer_interval: tuple[float, float] | None
    gold_verification_interval: tuple[float, float] | None
    gold_evidence_intervals: tuple[tuple[float, float], ...]
    mixture_path: str
    source_route: str = "synthetic_clean_single_event_bank"
    format: str = QA_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_source_row(
    row: Mapping[str, Any],
    *,
    source_bank_root: Path,
    require_audio_file: bool,
) -> CleanSource:
    """Normalize the final source-bank contract and common planning aliases."""

    label = str(_first(row, ("label", "coverage_label", "event_label"), "")).strip()
    source_id = str(
        _first(row, ("source_id", "source_event_uid", "event_id", "selection_key"), "")
    ).strip()
    video_id = str(_first(row, ("video_id", "source_video_id", "ytid"), "")).strip()
    audio_value = str(
        _first(row, ("audio_path", "materialized_audio_path", "source_path"), "")
    ).strip()
    if not label or not source_id or not video_id or not audio_value:
        raise ValueError("source row requires label, source_id, video_id and audio_path")
    raw_path = Path(audio_value)
    audio_path = raw_path if raw_path.is_absolute() else (source_bank_root / raw_path)
    audio_path = audio_path.resolve()
    if require_audio_file and not audio_path.is_file():
        raise FileNotFoundError(audio_path)

    onset = _finite_float(
        _first(
            row,
            (
                "active_onset_seconds",
                "event_onset_seconds",
                "source_active_onset_seconds",
                "source_onset_seconds",
            ),
        ),
        f"{source_id}.active_onset_seconds",
    )
    offset = _finite_float(
        _first(
            row,
            (
                "active_offset_seconds",
                "event_offset_seconds",
                "source_active_offset_seconds",
                "source_offset_seconds",
            ),
        ),
        f"{source_id}.active_offset_seconds",
    )
    if onset < 0 or offset <= onset:
        raise ValueError(f"{source_id} has an invalid exact active interval")

    clean_raw = _first(
        row,
        ("cleanliness_passed", "clean_source_eligible", "clean", "is_clean"),
    )
    audible_raw = _first(
        row,
        ("audibility_passed", "audible", "human_audible", "audibility_verified"),
    )
    if clean_raw is None or audible_raw is None:
        raise ValueError(
            f"{source_id} lacks explicit cleanliness_passed/audibility_passed gates"
        )
    cleanliness_passed = _bool_value(clean_raw, f"{source_id}.cleanliness_passed")
    audibility_passed = _bool_value(audible_raw, f"{source_id}.audibility_passed")
    if not cleanliness_passed or not audibility_passed:
        raise ValueError(f"{source_id} failed cleanliness or audibility")

    source_sha = str(
        _first(row, ("source_sha256", "audio_sha256", "sha256"), "")
    ).strip().lower()
    if not source_sha and audio_path.is_file():
        source_sha = sha256_file(audio_path)
    if len(source_sha) != 64 or any(char not in "0123456789abcdef" for char in source_sha):
        raise ValueError(f"{source_id} requires a 64-character source SHA-256")

    score_raw = _first(row, ("audibility_score", "audibility_confidence"))
    score = None if score_raw is None else _finite_float(score_raw, f"{source_id}.audibility_score")
    return CleanSource(
        source_id=source_id,
        label=label,
        official_split=_normalize_official_split(
            _first(row, ("official_split", "metadata_split", "source_split", "split"))
        ),
        video_id=video_id,
        source_sha256=source_sha,
        audio_path=audio_path.as_posix(),
        active_onset_seconds=onset,
        active_offset_seconds=offset,
        cleanliness_passed=True,
        audibility_passed=True,
        cleanliness_tier=str(
            _first(row, ("cleanliness_tier", "ambiguity_tier_name", "cleanliness"), "verified")
        ),
        audibility_score=score,
        provenance=dict(row),
    )


def load_source_bank(
    path: Path,
    *,
    require_audio_file: bool,
) -> list[CleanSource]:
    root = path.resolve().parent
    rows: list[CleanSource] = []
    seen_source_ids: set[str] = set()
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"non-object source row at {path}:{line_number}")
            source = normalize_source_row(
                payload,
                source_bank_root=root,
                require_audio_file=require_audio_file,
            )
            if source.source_id in seen_source_ids:
                raise ValueError(f"duplicate source_id: {source.source_id}")
            seen_source_ids.add(source.source_id)
            rows.append(source)
    if not rows:
        raise ValueError(f"empty source bank: {path}")
    return rows


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def source_identity_groups(sources: Sequence[CleanSource]) -> list[tuple[CleanSource, ...]]:
    """Union every alias sharing a source id, video id, hash, or file path."""

    union = _UnionFind(len(sources))
    identity_owner: dict[tuple[str, str], int] = {}
    for index, source in enumerate(sources):
        for identity in source.hard_identities:
            previous = identity_owner.setdefault(identity, index)
            union.union(index, previous)
    groups: dict[int, list[CleanSource]] = defaultdict(list)
    for index, source in enumerate(sources):
        groups[union.find(index)].append(source)
    output = [tuple(group) for group in groups.values()]
    for group in output:
        official = {source.official_split for source in group}
        if len(official) != 1:
            examples = sorted(source.source_id for source in group)[:5]
            raise ValueError(f"hard-linked sources cross official train/eval: {examples}")
    return output


def _partition_objective(
    dev_counts: Mapping[str, int],
    targets: Mapping[str, int],
    totals: Mapping[str, int],
) -> float:
    return sum(
        ((float(dev_counts.get(label, 0)) - float(targets[label])) / max(totals[label], 1)) ** 2
        for label in totals
    )


def partition_sources(
    sources: Sequence[CleanSource],
    ontology: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
    dev_fraction: float = 0.20,
) -> tuple[dict[str, list[CleanSource]], dict[str, Any]]:
    """Create a deterministic hard-identity-disjoint train/dev/test split.

    A stable 80/20 initialization is followed by deterministic single-group
    flips that minimize per-label squared deviation.  This matters when one
    source video contributes multiple clean event rows or labels.
    """

    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be in (0, 1)")
    labels = list(ontology)
    if len(labels) != len(set(labels)) or not labels:
        raise ValueError("ontology must be non-empty and duplicate-free")
    label_set = set(labels)
    unavailable = sorted({source.label for source in sources} - label_set)
    if unavailable:
        raise ValueError(f"source labels are outside the ontology: {unavailable[:10]}")

    groups = source_identity_groups(sources)
    train_groups = [group for group in groups if group[0].official_split == "train"]
    eval_groups = [group for group in groups if group[0].official_split == "eval"]
    totals = Counter(source.label for group in train_groups for source in group)
    targets = {
        label: min(
            totals[label],
            max(1 if totals[label] >= 2 else 0, int(round(totals[label] * dev_fraction))),
        )
        for label in labels
    }

    def group_key(group: Sequence[CleanSource]) -> str:
        return min(
            _stable_digest(seed, "partition-group", name, value)
            for source in group
            for name, value in source.hard_identities
        )

    ordered_groups = sorted(train_groups, key=group_key)
    group_weights = {
        id(group): Counter(source.label for source in group) for group in ordered_groups
    }
    single_label_groups: dict[str, list[tuple[CleanSource, ...]]] = defaultdict(list)
    multi_label_groups: list[tuple[CleanSource, ...]] = []
    for group in ordered_groups:
        labels_in_group = set(group_weights[id(group)])
        if len(labels_in_group) == 1:
            single_label_groups[next(iter(labels_in_group))].append(group)
        else:
            multi_label_groups.append(group)

    assignment: dict[int, bool] = {}
    current_counts: Counter[str] = Counter()
    # Shared-video multi-label groups receive one globally consistent fold.
    # Most clean-bank groups are single-label; those are then assigned against
    # their exact rounded class target in O(number_of_groups), avoiding a slow
    # coordinate-descent pass over every 15k-source candidate for every flip.
    for group in multi_label_groups:
        is_dev = int(group_key(group)[:16], 16) % 5 == 0
        assignment[id(group)] = is_dev
        if is_dev:
            current_counts.update(group_weights[id(group)])
    for label in labels:
        for group in single_label_groups.get(label, []):
            weight = group_weights[id(group)][label]
            current_error = abs(current_counts[label] - targets[label])
            dev_error = abs(current_counts[label] + weight - targets[label])
            is_dev = dev_error < current_error
            assignment[id(group)] = is_dev
            if is_dev:
                current_counts[label] += weight

    current_objective = _partition_objective(current_counts, targets, totals)

    def flip_improvement(group: tuple[CleanSource, ...]) -> float:
        direction = -1 if assignment[id(group)] else 1
        improvement = 0.0
        for label, weight in group_weights[id(group)].items():
            old = current_counts[label]
            new = old + direction * weight
            denominator = max(totals[label], 1) ** 2
            improvement += (
                (old - targets[label]) ** 2 - (new - targets[label]) ** 2
            ) / denominator
        return improvement

    # A short deterministic local improvement handles the rare multi-label or
    # multi-event source group.  Delta evaluation touches only labels present
    # in a group, so the full 200-class source plan remains cheap.
    while True:
        best: tuple[float, str, tuple[CleanSource, ...]] | None = None
        for group in ordered_groups:
            improvement = flip_improvement(group)
            if improvement <= 1e-15:
                continue
            candidate = (improvement, group_key(group), group)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            break
        improvement, _, group = best
        direction = -1 if assignment[id(group)] else 1
        current_counts.update(
            {
                label: direction * weight
                for label, weight in group_weights[id(group)].items()
            }
        )
        assignment[id(group)] = not assignment[id(group)]
        current_objective = max(0.0, current_objective - improvement)

    split_sources: dict[str, list[CleanSource]] = {split: [] for split in SPLITS}
    for group in train_groups:
        split = "dev" if assignment[id(group)] else "train"
        split_sources[split].extend(replace(source, assigned_split=split) for source in group)
    for group in eval_groups:
        split_sources["test"].extend(replace(source, assigned_split="test") for source in group)
    for split in SPLITS:
        split_sources[split] = _stable_order(
            split_sources[split], seed=seed, namespace=f"source-order:{split}"
        )

    per_label = []
    for label in labels:
        train_count = sum(source.label == label for source in split_sources["train"])
        dev_count = sum(source.label == label for source in split_sources["dev"])
        test_count = sum(source.label == label for source in split_sources["test"])
        official_train = train_count + dev_count
        per_label.append(
            {
                "label": label,
                "official_train_sources": official_train,
                "train_sources": train_count,
                "dev_sources": dev_count,
                "test_sources": test_count,
                "dev_target": targets[label],
                "dev_fraction": dev_count / max(official_train, 1),
                "rounded_target_error": dev_count - targets[label],
            }
        )
    receipt = {
        "policy": "global_hard_identity_groups;official_train_80_20;official_eval_test",
        "dev_fraction_requested": dev_fraction,
        "identity_groups": len(groups),
        "official_train_groups": len(train_groups),
        "official_eval_groups": len(eval_groups),
        "objective": current_objective,
        "per_label": per_label,
    }
    return split_sources, receipt


def _pop_source(queues: Mapping[str, list[CleanSource]], label: str) -> CleanSource:
    queue = queues[label]
    if not queue:
        raise RuntimeError(f"source queue unexpectedly exhausted for {label}")
    return queue.pop()


def schedule_scene_recipes(
    split_sources: Sequence[CleanSource],
    ontology: Sequence[str],
    *,
    split: str,
    seed: int = DEFAULT_SEED,
    core_rounds: int | None = None,
    repeat_rounds: int = 1,
    add_distractors: bool = True,
) -> tuple[list[SceneRecipe], dict[str, Any]]:
    """Schedule balanced 3--6 event recipes without reusing a source row."""

    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if repeat_rounds < 0 or core_rounds is not None and core_rounds < 0:
        raise ValueError("round counts must be non-negative")
    labels = list(ontology)
    if len(labels) < 3:
        raise ValueError("at least three ontology labels are required for scene scheduling")
    queues: dict[str, list[CleanSource]] = {label: [] for label in labels}
    for source in split_sources:
        if source.assigned_split and source.assigned_split != split:
            raise ValueError(f"source {source.source_id} assigned to the wrong split")
        if source.label in queues:
            queues[source.label].append(source)
    for label in labels:
        queues[label] = list(
            reversed(_stable_order(queues[label], seed=seed, namespace=f"queue:{split}:{label}"))
        )

    available_min = min(len(queues[label]) for label in labels)
    repeat_rounds_used = min(repeat_rounds, available_min // 9)
    repeat_reserve = 6 * repeat_rounds_used
    maximum_core = max(0, (available_min - repeat_reserve) // 3)
    core_rounds_used = maximum_core if core_rounds is None else min(core_rounds, maximum_core)
    if core_rounds_used < 1:
        counts = sorted((len(queues[label]), label) for label in labels)
        raise ValueError(
            f"split {split} needs at least 3 usable sources per ontology label; "
            f"minimum is {counts[0][0]} for {counts[0][1]}"
        )

    recipes: list[SceneRecipe] = []
    scene_index = 0

    def add_recipe(sources: Sequence[CleanSource], kind: str, round_index: int) -> None:
        nonlocal scene_index
        if not 3 <= len(sources) <= 6:
            raise RuntimeError("scene recipe must contain 3--6 events")
        recipes.append(
            SceneRecipe(
                scene_id=f"scene_{split}_{scene_index:07d}",
                split=split,
                sources=tuple(sources),
                recipe_kind=kind,
                round_index=round_index,
            )
        )
        scene_index += 1

    # In a core round, label i is first in scene i, middle in scene i-1, and
    # last in scene i-2.  All labels therefore obtain the same boundary roles.
    for round_index in range(core_rounds_used):
        order = _stable_order(
            labels, seed=seed, namespace=f"core:{split}:{round_index}"
        )
        for index in range(len(order)):
            scene_labels = (
                order[index],
                order[(index + 1) % len(order)],
                order[(index + 2) % len(order)],
            )
            add_recipe(
                [_pop_source(queues, label) for label in scene_labels],
                "core_boundary_internal_balance",
                round_index,
            )

    # Two paired scenes make ``after second <label>`` positive once and NONE
    # once.  Same-label adjacent questions are filtered later, so only the
    # acoustically meaningful ordinal-2 relation enters the headline QA set.
    for round_index in range(repeat_rounds_used):
        order = _stable_order(
            labels, seed=seed, namespace=f"repeat:{split}:{round_index}"
        )
        for index, label in enumerate(order):
            filler = order[(index + 1) % len(order)]
            add_recipe(
                [
                    _pop_source(queues, label),
                    _pop_source(queues, label),
                    _pop_source(queues, filler),
                ],
                "repeat_ordinal2_positive",
                round_index,
            )
            add_recipe(
                [
                    _pop_source(queues, filler),
                    _pop_source(queues, label),
                    _pop_source(queues, label),
                ],
                "repeat_ordinal2_none",
                round_index,
            )

    # Remaining clips are optional internal distractors.  Boundary events stay
    # fixed, preserving the explicit NONE semantics.  Label usage remains
    # round-robin rather than following source-bank frequency.
    if add_distractors:
        label_usage = Counter(
            source.label for recipe in recipes for source in recipe.sources
        )
        mutable = [list(recipe.sources) for recipe in recipes]
        for recipe_index, current in enumerate(mutable):
            target_count = 3 + int(
                _stable_digest(seed, split, "event-count", recipe_index)[:8], 16
            ) % 4
            while len(current) < target_count:
                candidates = [
                    label
                    for label in labels
                    if queues[label] and label not in {source.label for source in current}
                ]
                if not candidates:
                    break
                minimum_usage = min(label_usage[label] for label in candidates)
                candidates = [label for label in candidates if label_usage[label] == minimum_usage]
                chosen = _stable_order(
                    candidates,
                    seed=seed,
                    namespace=f"distractor:{split}:{recipe_index}:{len(current)}",
                )[0]
                # Insert before the last event so first/last boundary roles do
                # not change.  Source order is temporal order.
                current.insert(len(current) - 1, _pop_source(queues, chosen))
                label_usage[chosen] += 1
        recipes = [replace(recipe, sources=tuple(mutable[index])) for index, recipe in enumerate(recipes)]

    used_ids = [source.source_id for recipe in recipes for source in recipe.sources]
    if len(used_ids) != len(set(used_ids)):
        raise RuntimeError(f"source reuse detected while scheduling split {split}")
    counts = Counter(source.label for recipe in recipes for source in recipe.sources)
    receipt = {
        "split": split,
        "available_minimum_per_label": available_min,
        "core_rounds": core_rounds_used,
        "repeat_rounds": repeat_rounds_used,
        "scenes": len(recipes),
        "events": sum(len(recipe.sources) for recipe in recipes),
        "used_unique_sources": len(used_ids),
        "unused_sources": sum(len(queue) for queue in queues.values()),
        "event_count_distribution": dict(
            sorted(Counter(len(recipe.sources) for recipe in recipes).items())
        ),
        "per_label_event_min": min(counts.values()),
        "per_label_event_max": max(counts.values()),
        "per_label_events": dict(sorted(counts.items())),
    }
    return recipes, receipt


def _choose_active_subcrop(source: CleanSource, max_event_seconds: float, seed: int) -> tuple[float, float]:
    if max_event_seconds <= 0:
        raise ValueError("max_event_seconds must be positive")
    if source.duration_seconds <= max_event_seconds + 1e-9:
        return source.active_onset_seconds, source.active_offset_seconds
    slack = source.duration_seconds - max_event_seconds
    fraction = int(_stable_digest(seed, "long-event-subcrop", source.source_id)[:16], 16) / float(16**16 - 1)
    start = source.active_onset_seconds + slack * fraction
    return start, start + max_event_seconds


def _event_frame_lengths(
    sources: Sequence[CleanSource],
    *,
    max_event_seconds: float,
    minimum_event_frames: int,
    seed: int,
) -> tuple[list[int], list[tuple[float, float]]]:
    intervals = [_choose_active_subcrop(source, max_event_seconds, seed) for source in sources]
    lengths = [
        max(
            minimum_event_frames,
            int(math.ceil((offset - onset) / FRAME_HOP_SECONDS - 1e-9)),
        )
        for onset, offset in intervals
    ]
    return lengths, intervals


def place_recipe_on_grid(
    recipe: SceneRecipe,
    *,
    seed: int = DEFAULT_SEED,
    max_event_seconds: float = 1.20,
    minimum_event_frames: int = 2,
    minimum_gap_frames: int = 2,
) -> tuple[list[dict[str, Any]], list[tuple[float, float]]]:
    """Return exact 40-ms-aligned component metadata for one recipe."""

    frame_count = int(round(SCENE_SECONDS / FRAME_HOP_SECONDS))
    lengths, source_intervals = _event_frame_lengths(
        recipe.sources,
        max_event_seconds=max_event_seconds,
        minimum_event_frames=minimum_event_frames,
        seed=seed,
    )
    required = sum(lengths) + minimum_gap_frames * (len(lengths) - 1)
    if required > frame_count:
        raise ValueError(
            f"scene {recipe.scene_id} cannot fit {len(lengths)} events on the 10 s grid"
        )
    slack = frame_count - required
    rng = random.Random(int(_stable_digest(seed, "placement", recipe.scene_id)[:16], 16))
    # Positive leading/trailing verification regions are mandatory for NONE.
    edge_minimum = min(2, slack // 2)
    gap_extras = [0] * (len(lengths) + 1)
    gap_extras[0] = edge_minimum
    gap_extras[-1] = edge_minimum
    remaining = slack - 2 * edge_minimum
    for _ in range(remaining):
        gap_extras[rng.randrange(len(gap_extras))] += 1

    cursor = gap_extras[0]
    events: list[dict[str, Any]] = []
    intervals: list[tuple[float, float]] = []
    gains = (-6.0, -3.0, 0.0, 3.0)
    for index, (source, length, source_interval) in enumerate(
        zip(recipe.sources, lengths, source_intervals, strict=True)
    ):
        onset_frame = cursor
        offset_frame = onset_frame + length
        onset = onset_frame * FRAME_HOP_SECONDS
        offset = offset_frame * FRAME_HOP_SECONDS
        gain_db = gains[
            int(_stable_digest(seed, "component-gain", recipe.scene_id, index)[:8], 16)
            % len(gains)
        ]
        event_id = f"{recipe.scene_id}:e{index:02d}"
        events.append(
            {
                "event_id": event_id,
                "event_kind": "semantic",
                "label": source.label,
                "onset_seconds": onset,
                "offset_seconds": offset,
                "onset_frame": onset_frame,
                "offset_frame": offset_frame,
                "source_id": source.source_id,
                "source_video_id": source.video_id,
                "source_sha256": source.source_sha256,
                "source_path": source.audio_path,
                "source_annotation_onset_seconds": source.active_onset_seconds,
                "source_annotation_offset_seconds": source.active_offset_seconds,
                "source_crop_onset_seconds": source_interval[0],
                "source_crop_offset_seconds": source_interval[1],
                "source_crop_was_truncated": source_interval != (
                    source.active_onset_seconds,
                    source.active_offset_seconds,
                ),
                "component_gain_db": gain_db,
                "cleanliness_passed": source.cleanliness_passed,
                "audibility_passed": source.audibility_passed,
                "cleanliness_tier": source.cleanliness_tier,
                "audibility_score": source.audibility_score,
            }
        )
        intervals.append((onset, offset))
        cursor = offset_frame
        if index + 1 < len(lengths):
            cursor += minimum_gap_frames + gap_extras[index + 1]
    if cursor + gap_extras[-1] != frame_count:
        raise RuntimeError("grid placement did not consume exactly 250 frames")
    return events, source_intervals


def _ordinal(events: Sequence[Mapping[str, Any]], index: int) -> int:
    label = str(events[index]["label"])
    return 1 + sum(str(event["label"]) == label for event in events[:index])


def _ordinal_word(value: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth"}.get(value, str(value))


def _label_text(label: str) -> str:
    return label.replace("_and_", " / ").replace("_", " ")


def _qa_candidate(
    scene: Mapping[str, Any],
    *,
    relation: str,
    anchor_index: int,
    answer_index: int | None,
) -> QARow:
    events = list(scene["events"])
    anchor = events[anchor_index]
    ordinal = _ordinal(events, anchor_index)
    anchor_interval = (float(anchor["onset_seconds"]), float(anchor["offset_seconds"]))
    no_evidence = answer_index is None
    answer = None if answer_index is None else events[answer_index]
    if answer is None:
        if relation == "before":
            verification = (0.0, anchor_interval[0])
            evidence_intervals = ((0.0, anchor_interval[1]),)
            reason = "no_event_before_anchor"
        else:
            verification = (anchor_interval[1], SCENE_SECONDS)
            evidence_intervals = ((anchor_interval[0], SCENE_SECONDS),)
            reason = "no_event_after_anchor"
        answer_label = None
        answer_event_id = None
        evidence_ids = (str(anchor["event_id"]),)
        gold_answer = None
        answer_text = "no_evidence"
    else:
        verification = None
        gold_answer = (float(answer["onset_seconds"]), float(answer["offset_seconds"]))
        evidence_intervals = (anchor_interval, gold_answer)
        reason = None
        answer_label = str(answer["label"])
        answer_event_id = str(answer["event_id"])
        evidence_ids = (str(anchor["event_id"]), answer_event_id)
        answer_text = answer_label
    question = (
        f"What sound occurs immediately {relation} the "
        f"{_ordinal_word(ordinal)} {_label_text(str(anchor['label']))}?"
    )
    item_suffix = "none" if no_evidence else str(answer_index)
    return QARow(
        item_id=f"{scene['scene_id']}:{relation}:{anchor_index}:{item_suffix}",
        scene_id=str(scene["scene_id"]),
        split=str(scene["split"]),
        relation=relation,
        question=question,
        answer=answer_text,
        no_evidence=no_evidence,
        no_evidence_reason=reason,
        anchor_label=str(anchor["label"]),
        anchor_ordinal=ordinal,
        answer_label=answer_label,
        anchor_event_id=str(anchor["event_id"]),
        answer_event_id=answer_event_id,
        evidence_event_ids=evidence_ids,
        gold_anchor_interval=anchor_interval,
        gold_answer_interval=gold_answer,
        gold_verification_interval=verification,
        gold_evidence_intervals=evidence_intervals,
        mixture_path=str(scene["mixture_path"]),
    )


def build_conditionally_balanced_qa(
    scenes: Sequence[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    maximum_pairs_per_exact_key: int = 0,
) -> tuple[list[QARow], dict[str, Any]]:
    """Pair positive/NONE rows for every exact text-side conditioning key."""

    candidates: dict[tuple[str, str, int], dict[bool, list[QARow]]] = defaultdict(
        lambda: {False: [], True: []}
    )
    raw_positive = raw_none = 0
    for scene in scenes:
        events = sorted(
            scene.get("events") or [],
            key=lambda event: (float(event["onset_seconds"]), str(event["event_id"])),
        )
        if len(events) < 3:
            continue
        local = {**scene, "events": events}
        for anchor_index, anchor in enumerate(events):
            for relation, answer_index in (
                ("before", anchor_index - 1 if anchor_index > 0 else None),
                ("after", anchor_index + 1 if anchor_index + 1 < len(events) else None),
            ):
                # Adjacent occurrences of one label do not identify a distinct
                # acoustic answer event.  Keep ordinal anchors, but omit this
                # ambiguous candidate from the headline set.
                if answer_index is not None and str(events[answer_index]["label"]) == str(anchor["label"]):
                    continue
                row = _qa_candidate(
                    local,
                    relation=relation,
                    anchor_index=anchor_index,
                    answer_index=answer_index,
                )
                key = (row.relation, row.anchor_label, row.anchor_ordinal)
                candidates[key][row.no_evidence].append(row)
                raw_none += int(row.no_evidence)
                raw_positive += int(not row.no_evidence)

    selected: list[QARow] = []
    keys_without_both_targets = 0
    exact_key_counts: dict[str, dict[str, int]] = {}
    for key in sorted(candidates):
        positives = _stable_order(
            candidates[key][False], seed=seed, namespace=f"qa-positive:{key}"
        )
        negatives = _stable_order(
            candidates[key][True], seed=seed, namespace=f"qa-none:{key}"
        )
        count = min(len(positives), len(negatives))
        if maximum_pairs_per_exact_key > 0:
            count = min(count, maximum_pairs_per_exact_key)
        if count == 0:
            keys_without_both_targets += 1
            continue
        selected.extend(positives[:count])
        selected.extend(negatives[:count])
        exact_key_counts["|".join(map(str, key))] = {
            "answerable": count,
            "no_evidence": count,
        }
    selected = _stable_order(selected, seed=seed, namespace="qa-final-order")
    if not selected:
        raise ValueError("no conditionally balanced QA rows could be formed")
    receipt = {
        "raw_answerable_candidates": raw_positive,
        "raw_no_evidence_candidates": raw_none,
        "exact_keys_observed": len(candidates),
        "exact_keys_retained": len(exact_key_counts),
        "exact_keys_without_both_targets": keys_without_both_targets,
        "selected_answerable": sum(not row.no_evidence for row in selected),
        "selected_no_evidence": sum(row.no_evidence for row in selected),
        "exact_key_counts": exact_key_counts,
        "policy": "equal_positive_and_none_per_relation_anchor_label_ordinal",
    }
    return selected, receipt


def _majority_balanced_accuracy(rows: Sequence[QARow], fields: Sequence[str]) -> dict[str, Any]:
    groups: dict[tuple[str, ...], Counter[bool]] = defaultdict(Counter)
    for row in rows:
        key = tuple(str(getattr(row, field)) for field in fields)
        groups[key][not row.no_evidence] += 1
    predictions = {
        key: counts[True] >= counts[False] for key, counts in groups.items()
    }
    tp = tn = positives = negatives = 0
    for row in rows:
        key = tuple(str(getattr(row, field)) for field in fields)
        gold = not row.no_evidence
        prediction = predictions[key]
        positives += int(gold)
        negatives += int(not gold)
        tp += int(gold and prediction)
        tn += int(not gold and not prediction)
    positive_recall = tp / max(positives, 1)
    negative_recall = tn / max(negatives, 1)
    return {
        "features": list(fields),
        "groups": len(groups),
        "answerable_recall": positive_recall,
        "no_evidence_recall": negative_recall,
        "balanced_accuracy": 0.5 * (positive_recall + negative_recall),
    }


def audit_text_shortcuts(rows: Sequence[QARow], *, maximum_ba: float = 0.55) -> dict[str, Any]:
    feature_sets = {
        "relation": ("relation",),
        "relation_plus_anchor": ("relation", "anchor_label"),
        "relation_plus_ordinal": ("relation", "anchor_ordinal"),
        "relation_plus_anchor_plus_ordinal": (
            "relation", "anchor_label", "anchor_ordinal"
        ),
    }
    baselines = {
        name: _majority_balanced_accuracy(rows, fields)
        for name, fields in feature_sets.items()
    }
    return {
        "maximum_allowed_balanced_accuracy": maximum_ba,
        "baselines": baselines,
        "passes": all(
            metric["balanced_accuracy"] <= maximum_ba + 1e-12
            for metric in baselines.values()
        ),
    }


def audit_scene_metadata(
    scenes_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    qa_by_split: Mapping[str, Sequence[QARow]],
    ontology: Sequence[str],
) -> dict[str, Any]:
    """Audit split identity, grid, reconstruction metadata, and QA policy."""

    identity_sets: dict[str, dict[str, set[str]]] = {
        split: defaultdict(set) for split in SPLITS
    }
    grid_issues: list[str] = []
    source_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    label_counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        for scene in scenes_by_split.get(split, []):
            events = list(scene.get("events") or [])
            if not 3 <= len(events) <= 6:
                grid_issues.append(f"{scene.get('scene_id')}:event_count")
            previous_offset = -1.0
            for event in events:
                onset = float(event["onset_seconds"])
                offset = float(event["offset_seconds"])
                if (
                    not math.isclose(onset / FRAME_HOP_SECONDS, round(onset / FRAME_HOP_SECONDS), abs_tol=1e-8)
                    or not math.isclose(offset / FRAME_HOP_SECONDS, round(offset / FRAME_HOP_SECONDS), abs_tol=1e-8)
                    or onset < previous_offset - 1e-9
                    or offset <= onset
                    or offset > SCENE_SECONDS + 1e-9
                ):
                    grid_issues.append(str(event.get("event_id")))
                previous_offset = offset
                source_id = str(event["source_id"])
                source_counts[split][source_id] += 1
                label_counts[split][str(event["label"])] += 1
                for field in ("source_id", "source_video_id", "source_sha256", "source_path"):
                    value = str(event.get(field) or "")
                    if value:
                        identity_sets[split][field].add(value)

    overlaps: dict[str, Any] = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            overlaps[f"{left}__{right}"] = {
                field: sorted(identity_sets[left][field] & identity_sets[right][field])[:20]
                for field in ("source_id", "source_video_id", "source_sha256", "source_path")
            }
    duplicate_sources = {
        split: sorted(source_id for source_id, count in source_counts[split].items() if count > 1)
        for split in SPLITS
    }
    ontology_set = set(ontology)
    class_support = {
        split: {
            "classes_present": len(set(label_counts[split]) & ontology_set),
            "missing_labels": sorted(ontology_set - set(label_counts[split])),
            "minimum_events_per_label": min(
                (label_counts[split][label] for label in ontology), default=0
            ),
            "maximum_events_per_label": max(
                (label_counts[split][label] for label in ontology), default=0
            ),
        }
        for split in SPLITS
    }
    qa_audits = {split: audit_text_shortcuts(list(qa_by_split.get(split, []))) for split in SPLITS}
    evidence_issues: list[str] = []
    for split in SPLITS:
        for row in qa_by_split.get(split, []):
            if row.no_evidence:
                if (
                    row.answer_event_id is not None
                    or row.gold_answer_interval is not None
                    or row.gold_verification_interval is None
                    or row.evidence_event_ids != (row.anchor_event_id,)
                ):
                    evidence_issues.append(row.item_id)
            elif (
                row.answer_event_id is None
                or row.gold_answer_interval is None
                or row.gold_verification_interval is not None
                or row.evidence_event_ids != (row.anchor_event_id, row.answer_event_id)
            ):
                evidence_issues.append(row.item_id)

    overlap_count = sum(
        len(values)
        for pair in overlaps.values()
        for values in pair.values()
    )
    duplicate_count = sum(len(values) for values in duplicate_sources.values())
    passes = (
        not grid_issues
        and overlap_count == 0
        and duplicate_count == 0
        and not evidence_issues
        and all(audit["passes"] for audit in qa_audits.values())
        and all(not support["missing_labels"] for support in class_support.values())
    )
    return {
        "grid": {
            "frame_hop_seconds": FRAME_HOP_SECONDS,
            "scene_seconds": SCENE_SECONDS,
            "issues": grid_issues[:50],
            "passes": not grid_issues,
        },
        "cross_split_hard_identity_overlaps": overlaps,
        "cross_split_overlap_count": overlap_count,
        "source_reuse_within_split": duplicate_sources,
        "source_reuse_count": duplicate_count,
        "class_support": class_support,
        "qa_text_shortcuts": qa_audits,
        "evidence_policy": {
            "positive": "anchor_plus_answer_events",
            "none": "anchor_plus_complete_boundary_verification_region",
            "issues": evidence_issues[:50],
        },
        "passes": passes,
    }


def _resample_linear(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return waveform.astype(np.float32, copy=False)
    target_size = max(1, int(round(waveform.size * target_rate / source_rate)))
    source_positions = np.arange(waveform.size, dtype=np.float64)
    target_positions = np.linspace(0.0, max(waveform.size - 1, 0), target_size)
    return np.interp(target_positions, source_positions, waveform).astype(np.float32)


def _fit_waveform_to_samples(waveform: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 0 or waveform.size <= 0:
        raise ValueError("component waveform and requested samples must be positive")
    if waveform.size == samples:
        return waveform.astype(np.float32, copy=False)
    positions = np.linspace(0.0, waveform.size - 1, samples)
    return np.interp(positions, np.arange(waveform.size), waveform).astype(np.float32)


def render_scene_audio(
    scene: Mapping[str, Any],
    *,
    staging_root: Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    verify_source_hash: bool = False,
) -> dict[str, Any]:
    """Render short component WAVs and a reconstructable float32 mixture."""

    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError("audio rendering requires the soundfile package") from error

    scene_id = str(scene["scene_id"])
    split = str(scene["split"])
    total_samples = int(round(SCENE_SECONDS * sample_rate))
    component_arrays: list[np.ndarray] = []
    rendered_events: list[dict[str, Any]] = []
    for event in scene["events"]:
        path = Path(str(event["source_path"]))
        if verify_source_hash and sha256_file(path) != str(event["source_sha256"]):
            raise ValueError(f"source hash mismatch: {path}")
        waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
        mono = waveform.mean(axis=1)
        start = int(round(float(event["source_crop_onset_seconds"]) * source_rate))
        end = int(round(float(event["source_crop_offset_seconds"]) * source_rate))
        if start < 0 or end <= start or end > mono.size:
            raise ValueError(f"source crop outside audio for {event['event_id']}: {path}")
        crop = _resample_linear(mono[start:end], int(source_rate), sample_rate)
        target_samples = (int(event["offset_frame"]) - int(event["onset_frame"])) * int(
            round(FRAME_HOP_SECONDS * sample_rate)
        )
        crop = _fit_waveform_to_samples(crop, target_samples)
        fade_samples = min(int(round(0.005 * sample_rate)), crop.size // 2)
        if fade_samples > 0:
            ramp = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
            crop[:fade_samples] *= ramp
            crop[-fade_samples:] *= ramp[::-1]
        crop *= float(10.0 ** (float(event["component_gain_db"]) / 20.0))
        component_arrays.append(crop)
        rendered_events.append(dict(event))

    mixture = np.zeros(total_samples, dtype=np.float32)
    for event, component in zip(rendered_events, component_arrays, strict=True):
        start_sample = int(event["onset_frame"]) * int(round(FRAME_HOP_SECONDS * sample_rate))
        mixture[start_sample : start_sample + component.size] += component
    peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
    global_gain = 1.0 if peak <= 0.95 or peak == 0.0 else 0.95 / peak
    mixture *= global_gain
    global_gain_db = 20.0 * math.log10(global_gain) if global_gain > 0 else float("-inf")

    component_root = staging_root / "components" / split / scene_id
    component_root.mkdir(parents=True, exist_ok=True)
    for index, (event, component) in enumerate(zip(rendered_events, component_arrays, strict=True)):
        component = (component * global_gain).astype(np.float32)
        component_path = component_root / f"e{index:02d}.wav"
        sf.write(component_path, component, sample_rate, subtype="FLOAT")
        event["component_path"] = component_path.relative_to(staging_root).as_posix()
        event["component_sha256"] = sha256_file(component_path)
        event["scene_global_gain_db"] = global_gain_db
        event["placement_start_sample"] = int(event["onset_frame"]) * int(
            round(FRAME_HOP_SECONDS * sample_rate)
        )
        event["component_num_samples"] = int(component.size)

    mixture_path = staging_root / "audio" / split / f"{scene_id}.wav"
    mixture_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(mixture_path, mixture, sample_rate, subtype="FLOAT")
    return {
        **scene,
        "sample_rate": sample_rate,
        "audio_num_frames": int(mixture.size),
        "audio_num_channels": 1,
        "events": rendered_events,
        "mixture_path": mixture_path.relative_to(staging_root).as_posix(),
        "audio_sha256": sha256_file(mixture_path),
        "rendered": True,
        "scene_global_gain_db": global_gain_db,
        "mixture_peak": float(np.max(np.abs(mixture))),
    }


def audit_rendered_reconstruction(
    scenes: Sequence[Mapping[str, Any]],
    *,
    root: Path,
    tolerance: float = 2e-6,
) -> dict[str, Any]:
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("reconstruction audit requires soundfile") from error
    maximum_error = 0.0
    issues: list[str] = []
    for scene in scenes:
        mixture, sample_rate = sf.read(root / str(scene["mixture_path"]), dtype="float32")
        reconstruction = np.zeros_like(mixture, dtype=np.float32)
        for event in scene["events"]:
            component, component_rate = sf.read(
                root / str(event["component_path"]), dtype="float32"
            )
            if int(component_rate) != int(sample_rate):
                issues.append(f"{scene['scene_id']}:sample_rate")
                continue
            start = int(event["placement_start_sample"])
            reconstruction[start : start + component.size] += component
        error = float(np.max(np.abs(reconstruction - mixture)))
        maximum_error = max(maximum_error, error)
        if error > tolerance:
            issues.append(str(scene["scene_id"]))
    return {
        "scenes": len(scenes),
        "tolerance": tolerance,
        "maximum_absolute_error": maximum_error,
        "issues": issues[:50],
        "passes": not issues,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _scene_from_recipe(
    recipe: SceneRecipe,
    ontology: Sequence[str],
    *,
    seed: int,
    max_event_seconds: float,
) -> dict[str, Any]:
    events, _ = place_recipe_on_grid(
        recipe,
        seed=seed,
        max_event_seconds=max_event_seconds,
    )
    label_to_id = {label: index for index, label in enumerate(ontology)}
    for event in events:
        event["label_id"] = label_to_id[str(event["label"])]
    return {
        "format": SCENE_FORMAT,
        "scene_id": recipe.scene_id,
        "scene_family_id": recipe.scene_id,
        "split": recipe.split,
        "source_route": "synthetic_clean_single_event_bank",
        "mixture_path": f"audio/{recipe.split}/{recipe.scene_id}.wav",
        "duration_seconds": SCENE_SECONDS,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "audio_num_frames": int(round(SCENE_SECONDS * DEFAULT_SAMPLE_RATE)),
        "audio_num_channels": 1,
        "events": events,
        "recipe_kind": recipe.recipe_kind,
        "round_index": recipe.round_index,
        "rendered": False,
        "source_video_ids": sorted({source.video_id for source in recipe.sources}),
        "source_sha256s": sorted({source.source_sha256 for source in recipe.sources}),
    }


def build_clean_evidence_dataset(
    sources: Sequence[CleanSource],
    ontology: Sequence[str],
    *,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    dev_fraction: float = 0.20,
    core_rounds: int | None = None,
    repeat_rounds: int = 1,
    add_distractors: bool = True,
    max_event_seconds: float = 1.20,
    render_audio: bool = True,
    verify_source_hash: bool = False,
    overwrite: bool = False,
    require_ontology_size: int = 200,
) -> dict[str, Any]:
    """Build, audit, then atomically commit the final synthetic benchmark."""

    labels = list(ontology)
    if require_ontology_size > 0 and len(labels) != require_ontology_size:
        raise ValueError(
            f"final builder requires exactly {require_ontology_size} labels, got {len(labels)}"
        )
    if len(labels) != len(set(labels)):
        raise ValueError("ontology contains duplicate labels")
    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use overwrite=True")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        partitioned, partition_receipt = partition_sources(
            sources, labels, seed=seed, dev_fraction=dev_fraction
        )
        scenes_by_split: dict[str, list[dict[str, Any]]] = {}
        qa_by_split: dict[str, list[QARow]] = {}
        schedule_receipts: dict[str, Any] = {}
        qa_receipts: dict[str, Any] = {}
        for split in SPLITS:
            recipes, schedule_receipt = schedule_scene_recipes(
                partitioned[split],
                labels,
                split=split,
                seed=seed,
                core_rounds=core_rounds,
                repeat_rounds=repeat_rounds,
                add_distractors=add_distractors,
            )
            scenes = [
                _scene_from_recipe(
                    recipe,
                    labels,
                    seed=seed,
                    max_event_seconds=max_event_seconds,
                )
                for recipe in recipes
            ]
            if render_audio:
                scenes = [
                    render_scene_audio(
                        scene,
                        staging_root=staging,
                        sample_rate=DEFAULT_SAMPLE_RATE,
                        verify_source_hash=verify_source_hash,
                    )
                    for scene in scenes
                ]
            qa_rows, qa_receipt = build_conditionally_balanced_qa(scenes, seed=seed)
            scenes_by_split[split] = scenes
            qa_by_split[split] = qa_rows
            schedule_receipts[split] = schedule_receipt
            qa_receipts[split] = qa_receipt

        metadata_audit = audit_scene_metadata(scenes_by_split, qa_by_split, labels)
        reconstruction = {
            split: (
                audit_rendered_reconstruction(scenes_by_split[split], root=staging)
                if render_audio
                else {
                    "scenes": len(scenes_by_split[split]),
                    "passes": None,
                    "reason": "manifest_only",
                }
            )
            for split in SPLITS
        }
        if not metadata_audit["passes"]:
            raise RuntimeError("dataset failed metadata integrity audit")
        if render_audio and not all(report["passes"] for report in reconstruction.values()):
            raise RuntimeError("dataset failed waveform reconstruction audit")

        _atomic_write_text(staging / "ontology.txt", "".join(f"{label}\n" for label in labels))
        artifact_hashes: dict[str, str] = {}
        for split in SPLITS:
            scene_path = staging / f"scene_manifest_{split}.jsonl"
            qa_path = staging / f"qa_manifest_{split}.jsonl"
            ids_path = staging / f"scene_ids_{split}.txt"
            _atomic_write_text(scene_path, _jsonl_text(scenes_by_split[split]))
            _atomic_write_text(qa_path, _jsonl_text(row.to_dict() for row in qa_by_split[split]))
            _atomic_write_text(
                ids_path,
                "".join(f"{scene['scene_id']}\n" for scene in scenes_by_split[split]),
            )
            artifact_hashes[scene_path.name] = sha256_file(scene_path)
            artifact_hashes[qa_path.name] = sha256_file(qa_path)
            artifact_hashes[ids_path.name] = sha256_file(ids_path)
        artifact_hashes["ontology.txt"] = sha256_file(staging / "ontology.txt")

        receipt = {
            "format": FORMAT,
            "schema_version": 1,
            "seed": seed,
            "ontology_size": len(labels),
            "render_audio": render_audio,
            "frame_hop_seconds": FRAME_HOP_SECONDS,
            "scene_seconds": SCENE_SECONDS,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "evidence_definition": {
                "positive": "anchor_plus_answer_events",
                "no_evidence": "anchor_plus_complete_boundary_verification_region",
            },
            "partition": partition_receipt,
            "schedule": schedule_receipts,
            "qa_selection": qa_receipts,
            "metadata_audit": metadata_audit,
            "reconstruction_audit": reconstruction,
            "artifacts": artifact_hashes,
            "passes": metadata_audit["passes"] and (
                not render_audio or all(report["passes"] for report in reconstruction.values())
            ),
        }
        receipt_path = staging / "build_receipt.json"
        _atomic_write_text(
            receipt_path,
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        # The receipt is the commit marker and is written only after every
        # manifest/audio audit has passed.  Directory rename makes a new build
        # atomic; overwrite keeps the previous directory recoverable until the
        # replacement is in place.
        backup: Path | None = None
        if output_dir.exists():
            backup = output_dir.with_name(f".{output_dir.name}.backup-{uuid.uuid4().hex}")
            os.replace(output_dir, backup)
        try:
            os.replace(staging, output_dir)
        except BaseException:
            if backup is not None and backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup is not None:
            shutil.rmtree(backup)
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)


__all__ = [
    "CleanSource",
    "FRAME_HOP_SECONDS",
    "FORMAT",
    "QARow",
    "SCENE_SECONDS",
    "SceneRecipe",
    "audit_rendered_reconstruction",
    "audit_scene_metadata",
    "audit_text_shortcuts",
    "build_clean_evidence_dataset",
    "build_conditionally_balanced_qa",
    "load_source_bank",
    "normalize_source_row",
    "partition_sources",
    "place_recipe_on_grid",
    "schedule_scene_recipes",
    "source_identity_groups",
]
