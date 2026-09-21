#!/usr/bin/env python3
"""Plan sealed, development-only pseudo-label-OOD folds for QCES v5.

This command accepts exactly one QCES-v5 ``train`` manifest and one ``val``
manifest.  It writes only a JSON index plan: manifests and audio are never
copied, rewritten, or opened through paths stored inside a record.  Any test
path marker or non-development split fails closed before fold construction.

The folds diagnose transfer of the downstream QCES controller to labels that
are absent from that fold's meta-training records.  They do not establish that
the frozen CLAP/AudioSep pretraining stack has never encountered those labels.
"""
# flake8: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from mixi_understanding.data.qces_v5_schema import (  # noqa: E402
    QCESV5Record,
    RELATIONS,
    VARIANTS,
    parse_qces_v5_record,
)


PLAN_FORMAT = "qces_v5_pseudo_label_ood_plan_v1"
ALGORITHM_VERSION = "qces-v5-pseudo-label-ood-planner/1.0.0"
DEFAULT_SEED = 314_159
FOLD_COUNT = 3
DEFAULT_LABELS_PER_FOLD = 3
EXPECTED_RECORDS_PER_FAMILY = 48
EXPECTED_RECORDS_PER_SCENE = 16
_EXPECTED_SPLITS = ("train", "val")
_FORBIDDEN_PATH_TOKENS = {
    "all",
    "compositional",
    "eval",
    "evaluation",
    "iid",
    "ood",
    "test",
}


@dataclass(frozen=True)
class CoverageMinimums:
    """Predeclared per-fold gates; CLI users cannot weaken these values."""

    heldout_probe_records: int = 12
    heldout_probe_records_per_relation: int = 1
    answerable_heldout_probe_records: int = 3
    no_evidence_heldout_probe_records: int = 1
    same_label_heldout_probe_records: int = 1
    overlap_heldout_probe_records: int = 1
    complete_validation_families: int = 1
    retained_meta_train_families: int = 1


DEFAULT_MINIMUMS = CoverageMinimums()


@dataclass(frozen=True)
class ManifestReceipt:
    path: str
    sha256: str
    bytes: int
    records: int
    scene_families: int
    record_ids_sha256: str
    accepted_split: str


@dataclass(frozen=True)
class PlannerIndex:
    records_by_id: Mapping[str, QCESV5Record]
    record_ids: frozenset[str]
    family_records: Mapping[str, Tuple[QCESV5Record, ...]]
    family_record_ids: Mapping[str, frozenset[str]]
    semantic_labels: frozenset[str]
    probe_ids_by_label: Mapping[str, frozenset[str]]
    acoustic_family_ids_by_label: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class ComboEvaluation:
    labels: Tuple[str, ...]
    score: Tuple[Any, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--labels-per-fold",
        type=int,
        default=DEFAULT_LABELS_PER_FOLD,
        help=(
            "Number of disjoint development labels held out in each of the "
            f"{FOLD_COUNT} folds (default: {DEFAULT_LABELS_PER_FOLD})."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_digest(seed: int, *parts: str) -> str:
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return digest.hexdigest()


def _path_tokens(path: Path) -> Set[str]:
    tokens: Set[str] = set()
    for part in path.parts:
        tokens.update(
            token
            for token in re.split(r"[^a-z0-9]+", part.casefold())
            if token
        )
    return tokens


def _validate_manifest_path(path: Path, expected_split: str) -> Path:
    if expected_split not in _EXPECTED_SPLITS:
        raise ValueError(f"unsupported development split: {expected_split}")
    if path.suffix.casefold() != ".jsonl":
        raise ValueError(f"{expected_split} manifest must end with .jsonl: {path}")
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    forbidden = sorted(_path_tokens(resolved) & _FORBIDDEN_PATH_TOKENS)
    if forbidden:
        raise ValueError(
            f"refusing non-development/test-like manifest path {resolved}: "
            f"forbidden tokens={forbidden}"
        )
    filename_tokens = _path_tokens(Path(resolved.name))
    expected_tokens = {expected_split}
    if expected_split == "val":
        expected_tokens.add("validation")
    if not filename_tokens & expected_tokens:
        raise ValueError(
            f"{expected_split} manifest filename must identify its split: {resolved.name}"
        )
    return resolved


def _read_manifest(
    path: Path, expected_split: str
) -> Tuple[List[QCESV5Record], ManifestReceipt]:
    """Read only an explicitly named train/val JSONL and validate every row."""

    resolved = _validate_manifest_path(path, expected_split)
    records: List[QCESV5Record] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {resolved}:{line_number}"
                ) from error
            try:
                record = parse_qces_v5_record(payload)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid QCES-v5 record at {resolved}:{line_number}: {error}"
                ) from error
            if record.split != expected_split:
                raise ValueError(
                    f"{resolved}:{line_number} has split={record.split!r}; "
                    f"only {expected_split!r} is accepted"
                )
            records.append(record)
    if not records:
        raise ValueError(f"empty manifest: {resolved}")
    ids = [record.sample_id for record in records]
    duplicates = sorted(
        sample_id for sample_id, count in Counter(ids).items() if count != 1
    )
    if duplicates:
        raise ValueError(
            f"duplicate record IDs in {resolved}: {duplicates[:5]}"
        )
    receipt = ManifestReceipt(
        path=str(resolved),
        sha256=_sha256_file(resolved),
        bytes=resolved.stat().st_size,
        records=len(records),
        scene_families=len({record.scene_family_id for record in records}),
        record_ids_sha256=_id_digest(ids),
        accepted_split=expected_split,
    )
    return records, receipt


def _role_or_absent_labels(record: QCESV5Record) -> frozenset[str]:
    """Labels defining a held-out probe under the predeclared exact rule."""

    if record.no_evidence:
        return frozenset(record.absent_labels)
    event_map = {event.event_id: event for event in record.events}
    role_ids = record.anchor_event_ids + record.answer_event_ids
    return frozenset(event_map[event_id].label for event_id in role_ids)


def _semantic_event_labels(record: QCESV5Record) -> frozenset[str]:
    return frozenset(
        event.label for event in record.events if event.event_kind == "semantic"
    )


def _validate_complete_families(
    records: Sequence[QCESV5Record], expected_split: str
) -> Mapping[str, Tuple[QCESV5Record, ...]]:
    """Require intact 3x16 families before any IDs can enter a fold plan."""

    grouped: Dict[str, List[QCESV5Record]] = defaultdict(list)
    for record in records:
        if record.split != expected_split:
            raise AssertionError("family validator received a cross-split record")
        grouped[record.scene_family_id].append(record)
    complete: Dict[str, Tuple[QCESV5Record, ...]] = {}
    for family_id, values in sorted(grouped.items()):
        values = sorted(values, key=lambda record: record.sample_id)
        if len(values) != EXPECTED_RECORDS_PER_FAMILY:
            raise ValueError(
                f"incomplete QCES-v5 family {family_id}: expected "
                f"{EXPECTED_RECORDS_PER_FAMILY} records, found {len(values)}"
            )
        variants = {record.variant_id for record in values}
        if variants != set(VARIANTS):
            raise ValueError(
                f"incomplete QCES-v5 variants in {family_id}: {sorted(variants)}"
            )
        scene_ids_by_variant: Dict[str, Set[str]] = defaultdict(set)
        counts_by_scene: Counter[str] = Counter()
        for record in values:
            scene_ids_by_variant[record.variant_id].add(record.scene_id)
            counts_by_scene[record.scene_id] += 1
        if any(len(scene_ids_by_variant[variant]) != 1 for variant in VARIANTS):
            raise ValueError(f"variant-to-scene mapping is incomplete in {family_id}")
        if set(counts_by_scene.values()) != {EXPECTED_RECORDS_PER_SCENE}:
            raise ValueError(
                f"each scene in {family_id} must contain "
                f"{EXPECTED_RECORDS_PER_SCENE} records"
            )
        primary = [record for record in values if record.primary_counterfactual_probe]
        if len(primary) != len(VARIANTS) or {
            record.variant_id for record in primary
        } != set(VARIANTS):
            raise ValueError(
                f"{family_id} lacks one primary CEE probe per variant"
            )
        primary_contracts = {
            (
                record.counterfactual_group_id,
                record.question_semantics_id,
                record.question,
                record.answer_options,
            )
            for record in primary
        }
        if len(primary_contracts) != 1:
            raise ValueError(
                f"primary CEE surface/options change across {family_id}"
            )
        surface_groups: Dict[str, List[QCESV5Record]] = defaultdict(list)
        for record in values:
            if record.surface_control_group_id is not None:
                surface_groups[record.surface_control_group_id].append(record)
        for group_id, pair in surface_groups.items():
            if len(pair) != 2 or {
                record.mention_order_variant for record in pair
            } != {"forward", "reversed"}:
                raise ValueError(
                    f"incomplete first mention-order control {group_id} in {family_id}"
                )
        complete[family_id] = tuple(values)
    return complete


def _build_index(records: Sequence[QCESV5Record], split: str) -> PlannerIndex:
    family_records = _validate_complete_families(records, split)
    records_by_id = {record.sample_id: record for record in records}
    semantic_labels: Set[str] = set()
    probe_ids_by_label: Dict[str, Set[str]] = defaultdict(set)
    acoustic_family_ids_by_label: Dict[str, Set[str]] = defaultdict(set)
    for record in records:
        record_semantic = _semantic_event_labels(record)
        semantic_labels.update(record_semantic)
        for label in record_semantic:
            acoustic_family_ids_by_label[label].add(record.scene_family_id)
        for label in _role_or_absent_labels(record):
            probe_ids_by_label[label].add(record.sample_id)
    return PlannerIndex(
        records_by_id=records_by_id,
        record_ids=frozenset(records_by_id),
        family_records=family_records,
        family_record_ids={
            family_id: frozenset(record.sample_id for record in values)
            for family_id, values in family_records.items()
        },
        semantic_labels=frozenset(semantic_labels),
        probe_ids_by_label={
            label: frozenset(ids) for label, ids in probe_ids_by_label.items()
        },
        acoustic_family_ids_by_label={
            label: frozenset(ids)
            for label, ids in acoustic_family_ids_by_label.items()
        },
    )


def _record_counts(records: Iterable[QCESV5Record]) -> Dict[str, Any]:
    values = list(records)
    relation = Counter(record.relation for record in values)
    same_label = Counter(str(record.same_label_repeat).lower() for record in values)
    overlap = Counter(str(record.semantic_overlap).lower() for record in values)
    no_evidence = Counter(str(record.no_evidence).lower() for record in values)
    variants = Counter(record.variant_id for record in values)
    return {
        "records": len(values),
        "scene_families": len({record.scene_family_id for record in values}),
        "scenes": len({record.scene_id for record in values}),
        "by_relation": {name: relation[name] for name in RELATIONS},
        "by_same_label_repeat": {
            "true": same_label["true"],
            "false": same_label["false"],
        },
        "by_semantic_overlap": {
            "true": overlap["true"],
            "false": overlap["false"],
        },
        "by_no_evidence": {
            "true": no_evidence["true"],
            "false": no_evidence["false"],
        },
        "by_variant": {name: variants[name] for name in VARIANTS},
    }


def _union_lookup(
    labels: Iterable[str], mapping: Mapping[str, frozenset[str]]
) -> Set[str]:
    result: Set[str] = set()
    for label in labels:
        result.update(mapping.get(label, ()))
    return result


def _fold_sets(
    labels: Tuple[str, ...], train: PlannerIndex, val: PlannerIndex
) -> Dict[str, Set[str]]:
    probe_ids = _union_lookup(labels, val.probe_ids_by_label)
    validation_family_ids = {
        val.records_by_id[record_id].scene_family_id for record_id in probe_ids
    }
    validation_ids = _union_lookup(validation_family_ids, val.family_record_ids)

    acoustic_blocked_family_ids = _union_lookup(
        labels, train.acoustic_family_ids_by_label
    )
    acoustic_blocked_ids = _union_lookup(
        acoustic_blocked_family_ids, train.family_record_ids
    )
    controller_exposure_ids = _union_lookup(labels, train.probe_ids_by_label)
    controller_only_blocked_ids = controller_exposure_ids - acoustic_blocked_ids
    meta_train_ids = (
        set(train.record_ids) - acoustic_blocked_ids - controller_only_blocked_ids
    )
    meta_train_family_ids = {
        train.records_by_id[record_id].scene_family_id
        for record_id in meta_train_ids
    }
    return {
        "probe_ids": probe_ids,
        "validation_family_ids": validation_family_ids,
        "validation_ids": validation_ids,
        "acoustic_blocked_family_ids": acoustic_blocked_family_ids,
        "acoustic_blocked_ids": acoustic_blocked_ids,
        "controller_only_blocked_ids": controller_only_blocked_ids,
        "meta_train_ids": meta_train_ids,
        "meta_train_family_ids": meta_train_family_ids,
    }


def _coverage_failures(
    probe_counts: Mapping[str, Any], sets: Mapping[str, Set[str]],
    minimums: CoverageMinimums,
) -> List[str]:
    failures: List[str] = []
    checks = {
        "heldout_probe_records": (
            int(probe_counts["records"]), minimums.heldout_probe_records
        ),
        "answerable_heldout_probe_records": (
            int(probe_counts["by_no_evidence"]["false"]),
            minimums.answerable_heldout_probe_records,
        ),
        "no_evidence_heldout_probe_records": (
            int(probe_counts["by_no_evidence"]["true"]),
            minimums.no_evidence_heldout_probe_records,
        ),
        "same_label_heldout_probe_records": (
            int(probe_counts["by_same_label_repeat"]["true"]),
            minimums.same_label_heldout_probe_records,
        ),
        "overlap_heldout_probe_records": (
            int(probe_counts["by_semantic_overlap"]["true"]),
            minimums.overlap_heldout_probe_records,
        ),
        "complete_validation_families": (
            len(sets["validation_family_ids"]),
            minimums.complete_validation_families,
        ),
        "retained_meta_train_families": (
            len(sets["meta_train_family_ids"]),
            minimums.retained_meta_train_families,
        ),
    }
    for relation in RELATIONS:
        checks[f"heldout_probe_records_relation_{relation}"] = (
            int(probe_counts["by_relation"][relation]),
            minimums.heldout_probe_records_per_relation,
        )
    for name, (actual, required) in checks.items():
        if actual < required:
            failures.append(f"{name}={actual} < {required}")
    return failures


def _evaluate_combo(
    labels: Tuple[str, ...], train: PlannerIndex, val: PlannerIndex,
    minimums: CoverageMinimums, seed: int,
) -> ComboEvaluation | None:
    sets = _fold_sets(labels, train, val)
    probe_counts = _record_counts(
        val.records_by_id[record_id] for record_id in sets["probe_ids"]
    )
    if _coverage_failures(probe_counts, sets, minimums):
        return None
    relation_floor = min(probe_counts["by_relation"].values())
    # Preserve meta-training families first, then prefer balanced/richer probes.
    score = (
        -len(sets["meta_train_family_ids"]),
        -relation_floor,
        -int(probe_counts["by_no_evidence"]["true"]),
        -int(probe_counts["by_no_evidence"]["false"]),
        -int(probe_counts["records"]),
        _stable_digest(seed, *labels),
        labels,
    )
    return ComboEvaluation(labels=labels, score=score)


def _choose_disjoint_combos(
    candidates: Sequence[str], labels_per_fold: int, train: PlannerIndex,
    val: PlannerIndex, minimums: CoverageMinimums, seed: int,
) -> Tuple[Tuple[str, ...], ...]:
    if labels_per_fold <= 0:
        raise ValueError("labels_per_fold must be positive")
    required = FOLD_COUNT * labels_per_fold
    if len(candidates) < required:
        raise ValueError(
            f"need at least {required} eligible development labels for "
            f"{FOLD_COUNT} disjoint folds; found {len(candidates)}"
        )
    valid: List[ComboEvaluation] = []
    for labels in itertools.combinations(sorted(candidates), labels_per_fold):
        evaluated = _evaluate_combo(labels, train, val, minimums, seed)
        if evaluated is not None:
            valid.append(evaluated)
    valid.sort(key=lambda item: item.score)
    if not valid:
        raise ValueError(
            "no label combination satisfies the predeclared coverage minimums"
        )

    # Deterministic depth-first search over a fully deterministic quality order.
    # In normal paper-scale use it succeeds near the front; recursion ensures a
    # greedy first choice cannot make an otherwise feasible 3-fold plan fail.
    def search(
        start: int, chosen: Tuple[Tuple[str, ...], ...], used: frozenset[str]
    ) -> Tuple[Tuple[str, ...], ...] | None:
        if len(chosen) == FOLD_COUNT:
            return chosen
        remaining_slots = FOLD_COUNT - len(chosen)
        if len(candidates) - len(used) < remaining_slots * labels_per_fold:
            return None
        for index in range(start, len(valid)):
            labels = valid[index].labels
            if used.intersection(labels):
                continue
            result = search(
                index + 1, chosen + (labels,), used.union(labels)
            )
            if result is not None:
                return result
        return None

    selected = search(0, (), frozenset())
    if selected is None:
        raise ValueError(
            f"cannot construct {FOLD_COUNT} disjoint folds satisfying the "
            "predeclared coverage minimums"
        )
    return selected


def _assert_no_meta_train_leakage(
    labels: Tuple[str, ...], train: PlannerIndex, sets: Mapping[str, Set[str]]
) -> Dict[str, Any]:
    heldout = set(labels)
    retained_acoustic: Set[str] = set()
    retained_controller: Set[str] = set()
    for record_id in sets["meta_train_ids"]:
        record = train.records_by_id[record_id]
        retained_acoustic.update(_semantic_event_labels(record) & heldout)
        retained_controller.update(_role_or_absent_labels(record) & heldout)
    leaked_families = sorted(
        sets["meta_train_family_ids"] & sets["acoustic_blocked_family_ids"]
    )
    if retained_acoustic or retained_controller or leaked_families:
        raise AssertionError(
            "pseudo-label-OOD leakage proof failed: "
            f"acoustic={sorted(retained_acoustic)}, "
            f"controller={sorted(retained_controller)}, "
            f"families={leaked_families[:5]}"
        )
    return {
        "passed": True,
        "heldout_labels_in_meta_train_semantic_events": [],
        "heldout_labels_in_meta_train_role_or_absent_fields": [],
        "acoustic_blocked_families_retained": [],
        "meta_train_record_ids_sha256": _id_digest(sets["meta_train_ids"]),
        "heldout_probe_ids_sha256": _id_digest(sets["probe_ids"]),
        "complete_validation_ids_sha256": _id_digest(sets["validation_ids"]),
    }


def _fold_payload(
    fold_index: int, labels: Tuple[str, ...], train: PlannerIndex,
    val: PlannerIndex, minimums: CoverageMinimums,
) -> Dict[str, Any]:
    sets = _fold_sets(labels, train, val)
    probe_records = [
        val.records_by_id[record_id] for record_id in sorted(sets["probe_ids"])
    ]
    meta_train_records = [
        train.records_by_id[record_id]
        for record_id in sorted(sets["meta_train_ids"])
    ]
    validation_records = [
        val.records_by_id[record_id]
        for record_id in sorted(sets["validation_ids"])
    ]
    probe_counts = _record_counts(probe_records)
    failures = _coverage_failures(probe_counts, sets, minimums)
    if failures:
        raise AssertionError(f"selected fold failed coverage: {failures}")
    probes_by_label = {
        label: sorted(set(sets["probe_ids"]) & set(val.probe_ids_by_label[label]))
        for label in labels
    }
    return {
        "fold_id": f"fold_{fold_index}",
        "heldout_labels": list(labels),
        "meta_train_family_ids": sorted(sets["meta_train_family_ids"]),
        "meta_train_ids": sorted(sets["meta_train_ids"]),
        "meta_validation_family_ids": sorted(sets["validation_family_ids"]),
        "meta_validation_ids": sorted(sets["validation_ids"]),
        "heldout_probe_ids": sorted(sets["probe_ids"]),
        "heldout_probe_ids_by_label": probes_by_label,
        "excluded_acoustic_family_ids": sorted(
            sets["acoustic_blocked_family_ids"]
        ),
        "excluded_controller_exposure_ids": sorted(
            sets["controller_only_blocked_ids"]
        ),
        "counts": {
            "meta_train": _record_counts(meta_train_records),
            "complete_meta_validation_families": _record_counts(
                validation_records
            ),
            "heldout_probes": probe_counts,
            "excluded_acoustic_families": {
                "scene_families": len(sets["acoustic_blocked_family_ids"]),
                "records": len(sets["acoustic_blocked_ids"]),
            },
            "excluded_controller_only_records": len(
                sets["controller_only_blocked_ids"]
            ),
        },
        "coverage": {
            "passed": True,
            "hard_minimums": asdict(minimums),
            "failures": [],
        },
        "leakage_proof": _assert_no_meta_train_leakage(
            labels, train, sets
        ),
    }


def make_plan(
    *, train_records: Sequence[QCESV5Record],
    val_records: Sequence[QCESV5Record], train_receipt: ManifestReceipt,
    val_receipt: ManifestReceipt, seed: int = DEFAULT_SEED,
    labels_per_fold: int = DEFAULT_LABELS_PER_FOLD,
    minimums: CoverageMinimums = DEFAULT_MINIMUMS,
) -> Dict[str, Any]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_ids = {record.sample_id for record in train_records}
    val_ids = {record.sample_id for record in val_records}
    duplicate_ids = sorted(train_ids & val_ids)
    if duplicate_ids:
        raise ValueError(
            f"record IDs cross train/val manifests: {duplicate_ids[:5]}"
        )
    train = _build_index(train_records, "train")
    val = _build_index(val_records, "val")

    # A candidate must be acoustically present in both development splits and
    # must occur in an answerable validation anchor/answer role.  No test label
    # inventory participates in this intersection.
    answerable_val_role_labels: Set[str] = set()
    for record in val_records:
        if not record.no_evidence:
            answerable_val_role_labels.update(_role_or_absent_labels(record))
    candidates = sorted(
        train.semantic_labels & val.semantic_labels & answerable_val_role_labels
    )
    selected = _choose_disjoint_combos(
        candidates, labels_per_fold, train, val, minimums, seed
    )
    folds = [
        _fold_payload(index, labels, train, val, minimums)
        for index, labels in enumerate(selected)
    ]
    selected_labels = [
        label for fold in folds for label in fold["heldout_labels"]
    ]
    if len(selected_labels) != len(set(selected_labels)):
        raise AssertionError("heldout labels overlap across folds")
    return {
        "format": PLAN_FORMAT,
        "algorithm": {
            "version": ALGORITHM_VERSION,
            "seed": seed,
            "fold_count": FOLD_COUNT,
            "labels_per_fold": labels_per_fold,
            "selection": (
                "deterministic coverage-gated disjoint combination search"
            ),
        },
        "scope": {
            "accepted_input_splits": list(_EXPECTED_SPLITS),
            "test_manifests_accepted": False,
            "copies_or_rewrites_manifests": False,
            "reads_audio": False,
            "claim": "downstream_qces_controller_label_ood",
            "not_claimed": [
                "label_unseen_to_clap_pretraining",
                "label_unseen_to_audiosep_pretraining",
                "equivalence_to_the_sealed_qces_test_label_ood_split",
            ],
            "heldout_probe_rule": (
                "answerable: anchor/answer role labels; no-evidence: absent_labels"
            ),
        },
        "inputs": {
            "train": asdict(train_receipt),
            "val": asdict(val_receipt),
        },
        "development_label_inventory": {
            "train_semantic_labels": sorted(train.semantic_labels),
            "val_semantic_labels": sorted(val.semantic_labels),
            "eligible_seen_development_labels": candidates,
            "selected_labels": sorted(selected_labels),
            "selected_labels_disjoint_across_folds": True,
        },
        "family_contract": {
            "records_per_family": EXPECTED_RECORDS_PER_FAMILY,
            "records_per_scene": EXPECTED_RECORDS_PER_SCENE,
            "variants": list(VARIANTS),
            "complete_validation_families_are_retained": True,
        },
        "hard_minimums": asdict(minimums),
        "folds": folds,
        "global_proof": {
            "only_train_and_val_records_loaded": True,
            "input_record_splits": ["train", "val"],
            "fold_count_is_three": len(folds) == FOLD_COUNT,
            "heldout_labels_are_pairwise_disjoint": True,
            "every_fold_passes_coverage": all(
                fold["coverage"]["passed"] for fold in folds
            ),
            "every_fold_passes_meta_train_leakage_check": all(
                fold["leakage_proof"]["passed"] for fold in folds
            ),
        },
    }


def plan_manifests(
    *, train_manifest: Path, val_manifest: Path, output: Path,
    seed: int = DEFAULT_SEED,
    labels_per_fold: int = DEFAULT_LABELS_PER_FOLD,
    overwrite: bool = False,
) -> Dict[str, Any]:
    train_resolved = _validate_manifest_path(train_manifest, "train")
    val_resolved = _validate_manifest_path(val_manifest, "val")
    if train_resolved == val_resolved:
        raise ValueError("train and val manifests must be distinct files")
    output_resolved = output.expanduser().resolve()
    if output_resolved in {train_resolved, val_resolved}:
        raise ValueError("output must not overwrite an input manifest")
    if output_resolved.suffix.casefold() != ".json":
        raise ValueError("output plan must use a .json suffix, never .jsonl")

    train_records, train_receipt = _read_manifest(train_resolved, "train")
    val_records, val_receipt = _read_manifest(val_resolved, "val")
    plan = make_plan(
        train_records=train_records,
        val_records=val_records,
        train_receipt=train_receipt,
        val_receipt=val_receipt,
        seed=seed,
        labels_per_fold=labels_per_fold,
    )
    _atomic_json(output_resolved, plan, overwrite=overwrite)
    return plan


def _atomic_json(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = plan_manifests(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        output=args.output,
        seed=args.seed,
        labels_per_fold=args.labels_per_fold,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.output.resolve()}")
    print(
        "fold labels: "
        + "; ".join(
            f"{fold['fold_id']}={','.join(fold['heldout_labels'])}"
            for fold in plan["folds"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
