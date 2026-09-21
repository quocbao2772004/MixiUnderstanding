"""Leakage-safe split construction for detector and dense-reasoner training.

This is deliberately a sidecar to the historical QCES manifest builders.  It
implements the protocol required by new experiments without changing old
receipts:

* AudioSet-Strong's official ``train`` split is divided into train/dev by
  ``video_id`` with deterministic multi-label stratification;
* AudioSet-Strong's official ``test`` split is test-only;
* sources with official train/validation/eval partitions (currently the
  FUSS/FSD50K source bank) keep those assignments; and
* exact audio/source identities may not cross train, dev, and test.

The module is independent of PyTorch so split integrity can be tested before
any expensive model code is imported.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


PROTOCOL_NAME = "qces_clean_detector_protocol_v1"
CANONICAL_SPLITS = ("train", "dev", "test")

# Creator/uploader identities are intentionally not hard identities: official
# FSD50K partitions can contain distinct recordings from the same uploader.
# Everything below identifies an audio item, source recording, or scene.
STRICT_IDENTITY_FIELDS = (
    "scene_id",
    "video_id",
    "audio_sha256",
    "mixture_path",
    "source_id",
    "event.event_id",
    "event.source_id",
    "event.source_sha256",
    "event.source_path",
)

_SPLIT_ALIASES = {
    "train": frozenset({"train", "training"}),
    "dev": frozenset({"dev", "val", "valid", "validation"}),
    "test": frozenset({"test", "eval", "evaluation"}),
}


class SplitProtocolError(ValueError):
    """Raised when inputs cannot satisfy the clean split protocol."""


@dataclass(frozen=True)
class CleanSplitResult:
    """In-memory clean manifests and their auditable build receipt."""

    splits: dict[str, list[dict[str, Any]]]
    receipt: dict[str, Any]


def _stable_integer(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{PROTOCOL_NAME}:{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _canonical_row_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _semantic_labels(row: Mapping[str, Any]) -> set[str]:
    labels: set[str] = set()
    events = row.get("events") if "events" in row else row.get("gold_events")
    for event in events or ():
        if str(event.get("event_kind", "semantic")) != "semantic":
            continue
        label = str(event.get("label") or "").strip()
        if label:
            labels.add(label)
    return labels


def _filter_row(
    row: Mapping[str, Any], ontology: frozenset[str] | None
) -> dict[str, Any] | None:
    copied = copy.deepcopy(dict(row))
    event_field = "events" if "events" in copied else "gold_events"
    raw_events = list(copied.get(event_field) or ())
    if ontology is not None:
        raw_events = [
            event
            for event in raw_events
            if str(event.get("event_kind", "semantic")) != "semantic"
            or str(event.get("label") or "").strip() in ontology
        ]
        copied[event_field] = raw_events
    labels = sorted(_semantic_labels(copied))
    if not labels:
        return None
    copied["labels"] = labels
    return copied


def _upstream_split(row: Mapping[str, Any]) -> str:
    for field in ("hf_split", "official_split", "dataset_split", "split"):
        value = str(row.get(field) or "").strip().lower()
        if value:
            return value
    return ""


def _validate_declared_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected: str,
    source_name: str,
    audioset: bool = False,
) -> None:
    allowed = frozenset({expected}) if audioset else _SPLIT_ALIASES[expected]
    for index, row in enumerate(rows):
        declared = _upstream_split(row)
        if declared not in allowed:
            raise SplitProtocolError(
                f"{source_name} row {index} declares split={declared!r}; "
                f"expected one of {sorted(allowed)}"
            )
        if audioset and not str(row.get("video_id") or "").strip():
            raise SplitProtocolError(
                f"{source_name} row {index} has no video_id; AudioSet grouping is undefined"
            )


def _identity_values(row: Mapping[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for field in (
        "scene_id",
        "video_id",
        "audio_sha256",
        "mixture_path",
        "source_id",
    ):
        value = str(row.get(field) or "").strip()
        if value:
            values[field].add(value)
    events = row.get("events") if "events" in row else row.get("gold_events")
    for event in events or ():
        for field in ("event_id", "source_id", "source_sha256", "source_path"):
            value = str(event.get(field) or "").strip()
            if value:
                values[f"event.{field}"].add(value)
    return dict(values)


def audit_strict_identity_overlap(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]], *, sample_limit: int = 10
) -> dict[str, Any]:
    """Return exact cross-split overlap for every hard identity field."""

    identities: dict[str, dict[str, set[str]]] = {}
    for split, rows in split_rows.items():
        collected: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            for field, values in _identity_values(row).items():
                collected[field].update(values)
        identities[split] = dict(collected)

    pairs: dict[str, Any] = {}
    total = 0
    split_names = [split for split in CANONICAL_SPLITS if split in split_rows]
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            fields: dict[str, Any] = {}
            for field in STRICT_IDENTITY_FIELDS:
                overlap = identities[left].get(field, set()) & identities[right].get(
                    field, set()
                )
                if overlap:
                    total += len(overlap)
                    fields[field] = {
                        "count": len(overlap),
                        "sample": sorted(overlap)[:sample_limit],
                    }
            pairs[f"{left}__{right}"] = {
                "overlap_count": sum(item["count"] for item in fields.values()),
                "fields": fields,
            }
    return {
        "identity_fields": list(STRICT_IDENTITY_FIELDS),
        "pairs": pairs,
        "overlap_count": total,
        "passes": total == 0,
    }


def _group_audioset_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        video_id = str(row.get("video_id") or "").strip()
        groups[video_id].append(dict(row))
    for video_id in groups:
        groups[video_id].sort(key=_canonical_row_hash)
    return dict(groups)


def _group_labels(rows: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    labels: set[str] = set()
    for row in rows:
        labels.update(_semantic_labels(row))
    return frozenset(labels)


def select_stratified_dev_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    dev_fraction: float,
    seed: int,
) -> tuple[set[str], dict[str, Any]]:
    """Select deterministic multi-label dev groups while retaining train support.

    The objective minimizes normalized squared error between selected dev label
    counts and per-label targets.  A label present in only one AudioSet training
    video is protected from dev assignment.  Labels with at least two videos
    target at least one video in each side.
    """

    if not 0.0 <= dev_fraction < 1.0:
        raise SplitProtocolError("dev_fraction must satisfy 0 <= value < 1")
    group_keys = sorted(groups)
    num_groups = len(group_keys)
    if num_groups < 2 or dev_fraction == 0.0:
        target_groups = 0
    else:
        target_groups = min(num_groups - 1, max(1, int(round(num_groups * dev_fraction))))

    labels_by_group = {key: _group_labels(groups[key]) for key in group_keys}
    total_by_label: Counter[str] = Counter()
    for labels in labels_by_group.values():
        total_by_label.update(labels)
    target_by_label: dict[str, float] = {}
    for label, total in total_by_label.items():
        target_by_label[label] = (
            0.0
            if total <= 1
            else min(float(total - 1), max(1.0, float(total) * dev_fraction))
        )

    selected: set[str] = set()
    selected_by_label: Counter[str] = Counter()
    remaining = set(group_keys)
    while len(selected) < target_groups:
        candidates: list[tuple[float, int, str]] = []
        for key in remaining:
            labels = labels_by_group[key]
            # Never remove the last training video for an ontology label.
            if any(selected_by_label[label] + 1 >= total_by_label[label] for label in labels):
                continue
            gain = 0.0
            for label in labels:
                current = float(selected_by_label[label])
                target = target_by_label[label]
                normalizer = float(max(total_by_label[label], 1))
                gain += (
                    (current - target) ** 2 - (current + 1.0 - target) ** 2
                ) / normalizer
            # Prefer broad-label groups only when doing so improves the same
            # stratification objective; seeded hash resolves exact ties.
            candidates.append((gain, -_stable_integer(seed, key), key))
        if not candidates:
            raise SplitProtocolError(
                "cannot reach requested dev size without removing the last "
                "training example of at least one label; lower --dev-fraction "
                "or remove singleton labels from the ontology"
            )
        _, _, chosen = max(candidates)
        selected.add(chosen)
        remaining.remove(chosen)
        selected_by_label.update(labels_by_group[chosen])

    label_rows: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    for label in sorted(total_by_label):
        total = int(total_by_label[label])
        dev = int(selected_by_label[label])
        target = float(target_by_label[label])
        absolute_errors.append(abs(dev - target) / max(total, 1))
        label_rows.append(
            {
                "label": label,
                "total_train_groups": total,
                "target_dev_groups": target,
                "actual_dev_groups": dev,
                "actual_train_groups": total - dev,
            }
        )
    report = {
        "algorithm": "deterministic_greedy_multilabel_group_stratification_v1",
        "group_field": "video_id",
        "seed": seed,
        "dev_fraction": dev_fraction,
        "official_train_groups": num_groups,
        "target_dev_groups": target_groups,
        "actual_dev_groups": len(selected),
        "actual_train_groups": num_groups - len(selected),
        "labels": len(total_by_label),
        "labels_present_in_dev": sum(row["actual_dev_groups"] > 0 for row in label_rows),
        "labels_retained_in_train": sum(row["actual_train_groups"] > 0 for row in label_rows),
        "mean_normalized_label_target_error": (
            sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0
        ),
        "per_label": label_rows,
    }
    return selected, report


def _decorate_row(
    row: Mapping[str, Any],
    *,
    split: str,
    source: str,
    upstream_split: str,
) -> dict[str, Any]:
    copied = copy.deepcopy(dict(row))
    copied["protocol_name"] = PROTOCOL_NAME
    copied["protocol_source"] = source
    copied["protocol_upstream_split"] = upstream_split
    copied["protocol_split"] = split
    copied["split"] = split
    return copied


def _sort_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("protocol_source") or ""),
            str(row.get("video_id") or row.get("source_id") or row.get("scene_id") or ""),
            _canonical_row_hash(row),
        ),
    )


def _split_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_counts = Counter(str(row.get("protocol_source") or "") for row in rows)
    label_scene_counts: Counter[str] = Counter()
    for row in rows:
        label_scene_counts.update(_semantic_labels(row))
    return {
        "rows": len(rows),
        "sources": dict(sorted(source_counts.items())),
        "positive_labels": len(label_scene_counts),
        "label_scene_support_min": min(label_scene_counts.values(), default=0),
        "label_scene_support_median": (
            sorted(label_scene_counts.values())[len(label_scene_counts) // 2]
            if label_scene_counts
            else 0
        ),
        "label_scene_support_max": max(label_scene_counts.values(), default=0),
    }


def build_clean_detector_splits(
    *,
    audioset_train_rows: Sequence[Mapping[str, Any]],
    audioset_test_rows: Sequence[Mapping[str, Any]],
    preserved_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    ontology: Sequence[str] = (),
    dev_fraction: float = 0.15,
    seed: int = 2028,
) -> CleanSplitResult:
    """Build clean train/dev/test manifests and fail on protocol violations."""

    missing_preserved = set(CANONICAL_SPLITS) - set(preserved_rows)
    if missing_preserved:
        raise SplitProtocolError(
            f"preserved_rows is missing canonical splits: {sorted(missing_preserved)}"
        )
    _validate_declared_split(
        audioset_train_rows,
        expected="train",
        source_name="AudioSet official train",
        audioset=True,
    )
    _validate_declared_split(
        audioset_test_rows,
        expected="test",
        source_name="AudioSet official test",
        audioset=True,
    )
    for split in CANONICAL_SPLITS:
        _validate_declared_split(
            preserved_rows[split],
            expected=split,
            source_name=f"preserved {split}",
        )

    raw_train_video_ids = {
        str(row.get("video_id") or "").strip() for row in audioset_train_rows
    }
    raw_test_video_ids = {
        str(row.get("video_id") or "").strip() for row in audioset_test_rows
    }
    video_overlap = raw_train_video_ids & raw_test_video_ids
    if video_overlap:
        raise SplitProtocolError(
            "AudioSet official train/test share video_id values: "
            f"{sorted(video_overlap)[:10]}"
        )

    ontology_labels = list(dict.fromkeys(str(label).strip() for label in ontology if str(label).strip()))
    ontology_set = frozenset(ontology_labels) if ontology_labels else None

    def eligible(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        return [
            filtered
            for row in rows
            if (filtered := _filter_row(row, ontology_set)) is not None
        ]

    audioset_train = eligible(audioset_train_rows)
    audioset_test = eligible(audioset_test_rows)
    eligible_preserved = {
        split: eligible(preserved_rows[split]) for split in CANONICAL_SPLITS
    }

    groups = _group_audioset_rows(audioset_train)
    dev_video_ids, stratification = select_stratified_dev_groups(
        groups, dev_fraction=dev_fraction, seed=seed
    )

    outputs: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "dev": [],
        "test": [],
    }
    for video_id, rows in groups.items():
        split = "dev" if video_id in dev_video_ids else "train"
        outputs[split].extend(
            _decorate_row(
                row,
                split=split,
                source="audioset_strong",
                upstream_split="train",
            )
            for row in rows
        )
    outputs["test"].extend(
        _decorate_row(
            row,
            split="test",
            source="audioset_strong",
            upstream_split="test",
        )
        for row in audioset_test
    )
    for split in CANONICAL_SPLITS:
        outputs[split].extend(
            _decorate_row(
                row,
                split=split,
                source="preserved_official",
                upstream_split=_upstream_split(row),
            )
            for row in eligible_preserved[split]
        )
    outputs = {split: _sort_rows(outputs[split]) for split in CANONICAL_SPLITS}

    overlap = audit_strict_identity_overlap(outputs)
    if not overlap["passes"]:
        raise SplitProtocolError(
            "strict audio/source identity overlap across output splits: "
            f"{json.dumps(overlap['pairs'], ensure_ascii=False, sort_keys=True)}"
        )

    train_labels: set[str] = set()
    for row in outputs["train"]:
        train_labels.update(_semantic_labels(row))
    missing_train_labels = sorted(set(ontology_labels) - train_labels)
    if missing_train_labels:
        raise SplitProtocolError(
            "ontology labels without a positive training scene after clean split: "
            f"{missing_train_labels}"
        )

    assignment_rows: dict[str, list[dict[str, Any]]] = {}
    for split in CANONICAL_SPLITS:
        counts = Counter(
            (str(row.get("protocol_source")), str(row.get("protocol_upstream_split")))
            for row in outputs[split]
        )
        assignment_rows[split] = [
            {"source": source, "upstream_split": upstream, "rows": count}
            for (source, upstream), count in sorted(counts.items())
        ]

    receipt = {
        "format": PROTOCOL_NAME,
        "seed": seed,
        "ontology_labels": len(ontology_labels),
        "dev_fraction": dev_fraction,
        "policy": {
            "audioset_official_train": "deterministic group-stratified train/dev by video_id",
            "audioset_official_test": "test-only",
            "preserved_sources": "official train/validation/eval assignment",
            "cross_split_identity_overlap": "fatal",
        },
        "input_rows": {
            "audioset_train": len(audioset_train_rows),
            "audioset_test": len(audioset_test_rows),
            **{
                f"preserved_{split}": len(preserved_rows[split])
                for split in CANONICAL_SPLITS
            },
        },
        "ontology_eligible_rows": {
            "audioset_train": len(audioset_train),
            "audioset_test": len(audioset_test),
            **{
                f"preserved_{split}": len(eligible_preserved[split])
                for split in CANONICAL_SPLITS
            },
        },
        "stratification": stratification,
        "split_summary": {
            split: _split_summary(outputs[split]) for split in CANONICAL_SPLITS
        },
        "source_assignments": assignment_rows,
        "identity_audit": overlap,
        "invariants": {
            "official_test_rows_outside_test": sum(
                1
                for split in ("train", "dev")
                for row in outputs[split]
                if row.get("protocol_source") == "audioset_strong"
                and row.get("protocol_upstream_split") == "test"
            ),
            "preserved_assignment_violations": sum(
                1
                for split in CANONICAL_SPLITS
                for row in outputs[split]
                if row.get("protocol_source") == "preserved_official"
                and row.get("protocol_split") != split
            ),
            "cross_split_identity_overlaps": overlap["overlap_count"],
            "ontology_train_positive_labels": len(train_labels & set(ontology_labels)),
            "ontology_missing_train_positive_labels": missing_train_labels,
        },
        "passes": (
            overlap["passes"]
            and not missing_train_labels
            and all(
                row.get("protocol_upstream_split") != "test"
                for split in ("train", "dev")
                for row in outputs[split]
                if row.get("protocol_source") == "audioset_strong"
            )
        ),
    }
    # Keep this assertion close to receipt construction so a future metadata
    # change cannot silently turn a failed invariant into a passing receipt.
    if not receipt["passes"]:
        raise SplitProtocolError("clean split receipt failed one or more invariants")
    return CleanSplitResult(splits=outputs, receipt=receipt)
