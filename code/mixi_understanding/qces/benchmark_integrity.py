"""Leakage-safe utilities for the detector-first QCES benchmark.

This module is a sidecar to the existing experiment code.  It intentionally
does not mutate manifests or replace the historical evaluator: old receipts
must remain reproducible.  New experiments can import
``build_balanced_qa_items`` and the audit helpers below.

The historical temporal-QA builder has two ordering shortcuts when caps are
used: answerable ``after`` examples are appended before ``before`` examples,
and ``before_first`` no-evidence examples are appended before ``after_last``.
The builder here shuffles within a scene using a stable scene-local seed and
round-robins relations before applying caps.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


HARD_IDENTITY_FIELDS = (
    "scene_id",
    "video_id",
    "audio_sha256",
    "mixture_path",
    "event.source_sha256",
    "event.source_path",
)

SOFT_IDENTITY_FIELDS = (
    "event.source_id",
    "event.creator_id",
    "event.uploader_id",
)


@dataclass(frozen=True)
class BalancedQAItem:
    """One temporal QA example with an explicit evidence policy."""

    item_id: str
    scene_id: str
    relation: str
    question: str
    answer: str
    no_evidence: bool
    no_evidence_reason: str | None
    anchor_label: str
    anchor_ordinal: int
    answer_label: str | None
    gold_evidence_intervals: tuple[tuple[float, float], ...]
    gold_anchor_interval: tuple[float, float]
    gold_answer_interval: tuple[float, float] | None
    gold_verification_interval: tuple[float, float] | None
    source_route: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _label_text(label: str) -> str:
    return label.replace("_and_", " / ").replace("_", " ")


def _ordinal_word(index: int) -> str:
    return {
        1: "first",
        2: "second",
        3: "third",
        4: "fourth",
        5: "fifth",
        6: "sixth",
        7: "seventh",
        8: "eighth",
        9: "ninth",
        10: "tenth",
    }.get(index, str(index))


def _event_sort_key(event: Mapping[str, Any]) -> tuple[float, float, str, str]:
    return (
        float(event.get("onset_seconds", 0.0)),
        float(event.get("offset_seconds", 0.0)),
        str(event.get("label", "")),
        str(event.get("event_id", "")),
    )


def semantic_events(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw in row.get("events") or row.get("gold_events") or ():
        if raw.get("event_kind", "semantic") != "semantic":
            continue
        label = str(raw.get("label") or "").strip()
        onset = float(raw.get("onset_seconds", 0.0))
        offset = float(raw.get("offset_seconds", onset))
        if not label or not math.isfinite(onset) or not math.isfinite(offset) or offset <= onset:
            continue
        events.append(
            {
                **raw,
                "label": label,
                "onset_seconds": onset,
                "offset_seconds": offset,
            }
        )
    return sorted(events, key=_event_sort_key)


def _interval(event: Mapping[str, Any]) -> tuple[float, float]:
    return float(event["onset_seconds"]), float(event["offset_seconds"])


def _occurrence_ordinal(events: Sequence[Mapping[str, Any]], index: int) -> int:
    label = str(events[index]["label"])
    return 1 + sum(1 for event in events[:index] if str(event.get("label")) == label)


def _scene_rng(scene_id: str, seed: int) -> random.Random:
    digest = hashlib.sha256(f"qces-balanced-qa:{seed}:{scene_id}".encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _interleave(groups: Sequence[Sequence[BalancedQAItem]], start: int) -> list[BalancedQAItem]:
    output: list[BalancedQAItem] = []
    offsets = [0 for _ in groups]
    if not groups:
        return output
    while True:
        added = False
        for shift in range(len(groups)):
            group_index = (start + shift) % len(groups)
            offset = offsets[group_index]
            if offset < len(groups[group_index]):
                output.append(groups[group_index][offset])
                offsets[group_index] += 1
                added = True
        if not added:
            break
    return output


def build_balanced_qa_items(
    manifest_rows: Sequence[Mapping[str, Any]],
    *,
    max_scenes: int = 0,
    max_answerable_per_scene: int = 4,
    max_no_evidence_per_scene: int = 2,
    min_onset_gap_seconds: float = 0.08,
    require_non_overlapping: bool = True,
    exclude_same_label_pairs: bool = True,
    require_unambiguous_events: bool = True,
    seed: int = 2028,
) -> list[BalancedQAItem]:
    """Build capped QA rows without leaking answerability through relation.

    Evidence is defined consistently as follows:

    * positive: the union of anchor and answer event intervals;
    * no-evidence: anchor plus the complete region in which absence is checked.

    ``max_*_per_scene`` values are literal caps; zero emits no rows of that
    target type.  Scene limiting is applied after removing scenes with fewer
    than two valid semantic events.
    """

    items: list[BalancedQAItem] = []
    accepted_scenes = 0
    for row in manifest_rows:
        events = semantic_events(row)
        if len(events) < 2:
            continue
        if max_scenes and accepted_scenes >= max_scenes:
            break
        accepted_scenes += 1

        scene_id = str(row.get("scene_id") or f"scene_{accepted_scenes:08d}")
        route = str(row.get("source_route") or row.get("split") or "")
        duration = max(
            float(row.get("duration_seconds") or 0.0),
            max(float(event["offset_seconds"]) for event in events),
        )
        rng = _scene_rng(scene_id, seed)

        # A temporal question has no unique acoustic answer if its anchor or
        # answer overlaps another differently labelled event from the active
        # ontology.  This filter is annotation-only and never reads model
        # scores, so it removes ill-posed supervision rather than cherry-pick
        # model successes.
        ambiguous_indices = {
            left_index
            for left_index, left in enumerate(events)
            if any(
                left_index != right_index
                and str(left["label"]) != str(right["label"])
                and min(float(left["offset_seconds"]), float(right["offset_seconds"]))
                > max(float(left["onset_seconds"]), float(right["onset_seconds"]))
                for right_index, right in enumerate(events)
            )
        }

        by_relation: dict[str, list[BalancedQAItem]] = {"after": [], "before": []}
        for relation in ("after", "before"):
            if relation == "after":
                pairs = [(index, index + 1) for index in range(len(events) - 1)]
            else:
                pairs = [(index, index - 1) for index in range(1, len(events))]
            for anchor_index, answer_index in pairs:
                if require_unambiguous_events and (
                    anchor_index in ambiguous_indices or answer_index in ambiguous_indices
                ):
                    continue
                anchor = events[anchor_index]
                answer = events[answer_index]
                onset_gap = abs(
                    float(answer["onset_seconds"]) - float(anchor["onset_seconds"])
                )
                if onset_gap <= min_onset_gap_seconds:
                    continue
                if exclude_same_label_pairs and str(anchor["label"]) == str(answer["label"]):
                    continue
                if require_non_overlapping:
                    earlier, later = (
                        (anchor, answer)
                        if float(anchor["onset_seconds"]) < float(answer["onset_seconds"])
                        else (answer, anchor)
                    )
                    if float(later["onset_seconds"]) < float(earlier["offset_seconds"]):
                        continue
                ordinal = _occurrence_ordinal(events, anchor_index)
                anchor_label = str(anchor["label"])
                answer_label = str(answer["label"])
                question = (
                    f"What sound occurs immediately {relation} the "
                    f"{_ordinal_word(ordinal)} {_label_text(anchor_label)}?"
                )
                by_relation[relation].append(
                    BalancedQAItem(
                        item_id=f"{scene_id}:{relation}:{anchor_index}",
                        scene_id=scene_id,
                        relation=relation,
                        question=question,
                        answer=answer_label,
                        no_evidence=False,
                        no_evidence_reason=None,
                        anchor_label=anchor_label,
                        anchor_ordinal=ordinal,
                        answer_label=answer_label,
                        gold_evidence_intervals=(_interval(anchor), _interval(answer)),
                        gold_anchor_interval=_interval(anchor),
                        gold_answer_interval=_interval(answer),
                        gold_verification_interval=None,
                        source_route=route,
                    )
                )
            rng.shuffle(by_relation[relation])

        start_relation = rng.randrange(2)
        answerable = _interleave(
            (by_relation["after"], by_relation["before"]), start_relation
        )
        items.extend(answerable[: max(0, max_answerable_per_scene)])

        first = events[0]
        first_ordinal = _occurrence_ordinal(events, 0)
        first_anchor = _interval(first)
        before_verification = (0.0, first_anchor[0])
        before_evidence = (0.0, first_anchor[1])
        last_index = len(events) - 1
        last = events[last_index]
        last_ordinal = _occurrence_ordinal(events, last_index)
        last_anchor = _interval(last)
        after_verification = (last_anchor[1], duration)
        after_evidence = (last_anchor[0], duration)

        no_evidence: list[BalancedQAItem] = []
        if not require_unambiguous_events or 0 not in ambiguous_indices:
            no_evidence.append(BalancedQAItem(
                item_id=f"{scene_id}:before_first",
                scene_id=scene_id,
                relation="before",
                question=(
                    "What sound occurs immediately before the "
                    f"{_ordinal_word(first_ordinal)} {_label_text(str(first['label']))}?"
                ),
                answer="no_evidence",
                no_evidence=True,
                no_evidence_reason="no_event_before_anchor",
                anchor_label=str(first["label"]),
                anchor_ordinal=first_ordinal,
                answer_label=None,
                gold_evidence_intervals=(before_evidence,),
                gold_anchor_interval=first_anchor,
                gold_answer_interval=None,
                gold_verification_interval=before_verification,
                source_route=route,
            ))
        if not require_unambiguous_events or last_index not in ambiguous_indices:
            no_evidence.append(BalancedQAItem(
                item_id=f"{scene_id}:after_last",
                scene_id=scene_id,
                relation="after",
                question=(
                    "What sound occurs immediately after the "
                    f"{_ordinal_word(last_ordinal)} {_label_text(str(last['label']))}?"
                ),
                answer="no_evidence",
                no_evidence=True,
                no_evidence_reason="no_event_after_anchor",
                anchor_label=str(last["label"]),
                anchor_ordinal=last_ordinal,
                answer_label=None,
                gold_evidence_intervals=(after_evidence,),
                gold_anchor_interval=last_anchor,
                gold_answer_interval=None,
                gold_verification_interval=after_verification,
                source_route=route,
            ))
        rng.shuffle(no_evidence)
        items.extend(no_evidence[: max(0, max_no_evidence_per_scene)])
    return items


def audit_qa_event_ambiguity(
    items: Sequence[Mapping[str, Any] | BalancedQAItem],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify that headline anchor/answer intervals are acoustically unique.

    Ambiguity is defined within the active semantic ontology: a gold anchor or
    answer interval must not overlap an event carrying a different label.  The
    audit consumes annotations only and therefore cannot select examples based
    on model success.
    """

    scenes = {
        str(row.get("scene_id") or ""): semantic_events(row)
        for row in manifest_rows
        if str(row.get("scene_id") or "")
    }
    ambiguous_items: list[str] = []
    missing_gold_items: list[str] = []

    def interval_matches_event(
        interval: tuple[float, float],
        label: str,
        events: Sequence[Mapping[str, Any]],
    ) -> bool:
        return any(
            str(event["label"]) == label
            and math.isclose(float(event["onset_seconds"]), interval[0], abs_tol=1e-7)
            and math.isclose(float(event["offset_seconds"]), interval[1], abs_tol=1e-7)
            for event in events
        )

    def interval_is_ambiguous(
        interval: tuple[float, float],
        label: str,
        events: Sequence[Mapping[str, Any]],
    ) -> bool:
        return any(
            str(event["label"]) != label
            and min(interval[1], float(event["offset_seconds"]))
            > max(interval[0], float(event["onset_seconds"]))
            for event in events
        )

    for item in items:
        item_id = str(_item_value(item, "item_id", ""))
        scene_id = str(_item_value(item, "scene_id", ""))
        events = scenes.get(scene_id)
        if events is None:
            missing_gold_items.append(item_id)
            continue
        checks = [
            (
                tuple(_item_value(item, "gold_anchor_interval")),
                str(_item_value(item, "anchor_label", "")),
            )
        ]
        answer_interval = _item_value(item, "gold_answer_interval")
        answer_label = _item_value(item, "answer_label")
        if answer_interval is not None and answer_label is not None:
            checks.append((tuple(answer_interval), str(answer_label)))
        if any(
            not interval_matches_event(interval, label, events)
            for interval, label in checks
        ):
            missing_gold_items.append(item_id)
            continue
        if any(
            interval_is_ambiguous(interval, label, events)
            for interval, label in checks
        ):
            ambiguous_items.append(item_id)

    return {
        "policy": "anchor_and_answer_do_not_overlap_different_active_ontology_label",
        "items": len(items),
        "ambiguous_items": len(ambiguous_items),
        "missing_gold_items": len(missing_gold_items),
        "ambiguous_item_examples": ambiguous_items[:20],
        "missing_gold_item_examples": missing_gold_items[:20],
        "passes": not ambiguous_items and not missing_gold_items,
    }


def _item_value(item: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(key, default)
    return getattr(item, key, default)


def _majority_baseline(
    items: Sequence[Mapping[str, Any] | Any], feature_names: Sequence[str]
) -> dict[str, Any]:
    groups: dict[tuple[str, ...], Counter[bool]] = defaultdict(Counter)
    for item in items:
        key = tuple(str(_item_value(item, name, "")) for name in feature_names)
        answerable = not bool(_item_value(item, "no_evidence", False))
        groups[key][answerable] += 1
    predictions = {
        key: counts[True] >= counts[False] for key, counts in groups.items()
    }
    confusion = Counter()
    for item in items:
        key = tuple(str(_item_value(item, name, "")) for name in feature_names)
        gold = not bool(_item_value(item, "no_evidence", False))
        pred = predictions[key]
        confusion[(gold, pred)] += 1
    answerable_recall = confusion[(True, True)] / max(
        confusion[(True, True)] + confusion[(True, False)], 1
    )
    no_evidence_recall = confusion[(False, False)] / max(
        confusion[(False, False)] + confusion[(False, True)], 1
    )
    accuracy = (confusion[(True, True)] + confusion[(False, False)]) / max(len(items), 1)
    return {
        "features": list(feature_names),
        "groups": len(groups),
        "accuracy": accuracy,
        "balanced_accuracy": 0.5 * (answerable_recall + no_evidence_recall),
        "answerable_recall": answerable_recall,
        "no_evidence_recall": no_evidence_recall,
    }


def _scene_fold(scene_id: str, folds: int) -> int:
    digest = hashlib.sha256(f"qces-shortcut-fold:{scene_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _cross_validated_answer_baseline(
    items: Sequence[Mapping[str, Any] | Any],
    feature_names: Sequence[str],
    *,
    folds: int = 5,
) -> dict[str, Any]:
    """Predict the exact answer using question metadata and no audio.

    Folds are grouped by scene so questions from one recording never occur on
    both sides of the shortcut baseline.
    """

    if not items:
        return {
            "features": list(feature_names),
            "folds": folds,
            "accuracy": 0.0,
            "answerable_accuracy": 0.0,
            "no_evidence_accuracy": 0.0,
        }
    correct = answerable_correct = no_evidence_correct = 0
    answerable_total = no_evidence_total = 0
    for fold in range(folds):
        training = [
            item
            for item in items
            if _scene_fold(str(_item_value(item, "scene_id", "")), folds) != fold
        ]
        testing = [
            item
            for item in items
            if _scene_fold(str(_item_value(item, "scene_id", "")), folds) == fold
        ]
        group_targets: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        global_targets: Counter[str] = Counter()
        for item in training:
            key = tuple(str(_item_value(item, name, "")) for name in feature_names)
            target = str(_item_value(item, "answer", "no_evidence"))
            group_targets[key][target] += 1
            global_targets[target] += 1
        fallback = (
            min(
                global_targets,
                key=lambda target: (-global_targets[target], target),
            )
            if global_targets
            else "__unseen_answer__"
        )
        for item in testing:
            key = tuple(str(_item_value(item, name, "")) for name in feature_names)
            counter = group_targets.get(key)
            prediction = (
                min(counter, key=lambda target: (-counter[target], target))
                if counter
                else fallback
            )
            target = str(_item_value(item, "answer", "no_evidence"))
            is_no_evidence = bool(_item_value(item, "no_evidence", False))
            matched = prediction == target
            correct += int(matched)
            if is_no_evidence:
                no_evidence_total += 1
                no_evidence_correct += int(matched)
            else:
                answerable_total += 1
                answerable_correct += int(matched)
    return {
        "features": list(feature_names),
        "folds": folds,
        "accuracy": correct / max(len(items), 1),
        "answerable_accuracy": answerable_correct / max(answerable_total, 1),
        "no_evidence_accuracy": no_evidence_correct / max(no_evidence_total, 1),
    }


def _temporal_pair_audit(
    items: Sequence[Mapping[str, Any] | Any],
) -> dict[str, Any]:
    answerable = [
        item for item in items if not bool(_item_value(item, "no_evidence", False))
    ]
    onset_ties = relation_invalid = overlapping = same_label = 0
    for item in answerable:
        anchor = _item_value(item, "gold_anchor_interval")
        answer = _item_value(item, "gold_answer_interval")
        if not anchor or not answer:
            relation_invalid += 1
            continue
        anchor_onset, anchor_offset = float(anchor[0]), float(anchor[1])
        answer_onset, answer_offset = float(answer[0]), float(answer[1])
        relation = str(_item_value(item, "relation", ""))
        onset_ties += int(abs(anchor_onset - answer_onset) <= 1e-9)
        if relation == "after":
            relation_invalid += int(answer_onset <= anchor_onset)
        elif relation == "before":
            relation_invalid += int(answer_onset >= anchor_onset)
        else:
            relation_invalid += 1
        overlapping += int(min(anchor_offset, answer_offset) > max(anchor_onset, answer_onset))
        same_label += int(
            str(_item_value(item, "anchor_label", ""))
            == str(_item_value(item, "answer_label", ""))
        )
    return {
        "answerable_items": len(answerable),
        "onset_ties": onset_ties,
        "relation_invalid": relation_invalid,
        "overlapping_anchor_answer": overlapping,
        "same_label_anchor_answer": same_label,
        "relation_invalid_rate": relation_invalid / max(len(answerable), 1),
        "overlap_rate": overlapping / max(len(answerable), 1),
        "same_label_rate": same_label / max(len(answerable), 1),
    }


def audit_qa_shortcuts(items: Sequence[Mapping[str, Any] | Any]) -> dict[str, Any]:
    """Measure how well text-side metadata predicts answerability without audio."""

    cross_tab: dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        relation = str(_item_value(item, "relation", "unknown"))
        target = "no_evidence" if bool(_item_value(item, "no_evidence", False)) else "answerable"
        cross_tab[relation][target] += 1
    baselines = {
        "relation": _majority_baseline(items, ("relation",)),
        "ordinal": _majority_baseline(items, ("anchor_ordinal",)),
        "relation_plus_ordinal": _majority_baseline(
            items, ("relation", "anchor_ordinal")
        ),
        "relation_plus_anchor_label": _majority_baseline(
            items, ("relation", "anchor_label")
        ),
    }
    answer_baselines = {
        "relation": _cross_validated_answer_baseline(items, ("relation",)),
        "relation_plus_anchor_label": _cross_validated_answer_baseline(
            items, ("relation", "anchor_label")
        ),
        "relation_plus_anchor_label_plus_ordinal": _cross_validated_answer_baseline(
            items, ("relation", "anchor_label", "anchor_ordinal")
        ),
    }
    return {
        "items": len(items),
        "answerable": sum(not bool(_item_value(item, "no_evidence", False)) for item in items),
        "no_evidence": sum(bool(_item_value(item, "no_evidence", False)) for item in items),
        "relation_target_counts": {
            relation: dict(sorted(counts.items()))
            for relation, counts in sorted(cross_tab.items())
        },
        "text_only_majority_baselines": baselines,
        "text_only_exact_answer_baselines": answer_baselines,
        "temporal_pair_audit": _temporal_pair_audit(items),
        "relation_shortcut": baselines["relation"]["balanced_accuracy"] > 0.60,
    }


def _identities(row: Mapping[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for field in ("scene_id", "video_id", "audio_sha256", "mixture_path"):
        value = str(row.get(field) or "").strip()
        if value:
            values[field].add(value)
    for event in row.get("events") or row.get("gold_events") or ():
        for field in (
            "source_id",
            "source_sha256",
            "source_path",
            "creator_id",
            "uploader_id",
        ):
            value = str(event.get(field) or "").strip()
            if value:
                values[f"event.{field}"].add(value)
    return values


def collect_split_identities(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for field, values in _identities(row).items():
            output[field].update(values)
    return dict(output)


def audit_split_overlaps(
    split_rows: Mapping[str, Sequence[Mapping[str, Any]]], *, sample_limit: int = 12
) -> dict[str, Any]:
    """Find exact source/audio leakage across every pair of splits."""

    identities = {
        split: collect_split_identities(rows) for split, rows in split_rows.items()
    }
    pair_reports: dict[str, Any] = {}
    split_names = sorted(split_rows)
    hard_overlap_total = 0
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            field_reports: dict[str, Any] = {}
            for field in sorted(set(identities[left]) | set(identities[right])):
                overlap = identities[left].get(field, set()) & identities[right].get(field, set())
                if not overlap:
                    continue
                is_hard = field in HARD_IDENTITY_FIELDS
                if is_hard:
                    hard_overlap_total += len(overlap)
                field_reports[field] = {
                    "count": len(overlap),
                    "severity": "error" if is_hard else "warning",
                    "sample": sorted(overlap)[:sample_limit],
                }
            pair_reports[f"{left}__{right}"] = {
                "hard_overlap_count": sum(
                    report["count"]
                    for report in field_reports.values()
                    if report["severity"] == "error"
                ),
                "fields": field_reports,
            }
    return {
        "split_rows": {split: len(rows) for split, rows in split_rows.items()},
        "pairs": pair_reports,
        "hard_overlap_count": hard_overlap_total,
        "passes": hard_overlap_total == 0,
    }


def audit_class_support(
    rows: Sequence[Mapping[str, Any]],
    *,
    ontology: Sequence[str] = (),
    minimum_scenes: int = 20,
    minimum_active_seconds: float = 60.0,
) -> dict[str, Any]:
    """Count independent scene support instead of counting QA questions."""

    scene_ids: dict[str, set[str]] = defaultdict(set)
    source_ids: dict[str, set[str]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    active_seconds: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        scene_id = str(row.get("scene_id") or f"row:{row_index}")
        for event in semantic_events(row):
            label = str(event["label"])
            scene_ids[label].add(scene_id)
            occurrences[label] += 1
            active_seconds[label] += max(
                0.0,
                float(event["offset_seconds"]) - float(event["onset_seconds"]),
            )
            source_id = str(
                event.get("source_sha256")
                or event.get("source_id")
                or event.get("source_path")
                or row.get("video_id")
                or row.get("audio_sha256")
                or scene_id
                or ""
            )
            if source_id:
                source_ids[label].add(source_id)
    labels = list(dict.fromkeys(str(label) for label in ontology if str(label)))
    if not labels:
        labels = sorted(scene_ids)
    per_label: list[dict[str, Any]] = []
    for label in labels:
        num_scenes = len(scene_ids[label])
        seconds = float(active_seconds[label])
        per_label.append(
            {
                "label": label,
                "scenes": num_scenes,
                "sources": len(source_ids[label]),
                "occurrences": int(occurrences[label]),
                "active_seconds": seconds,
                "ready": num_scenes >= minimum_scenes
                and seconds >= minimum_active_seconds,
            }
        )
    per_label.sort(key=lambda row: (row["scenes"], row["active_seconds"], row["label"]))
    thresholds = (1, 5, 10, 20, 40, 100)
    return {
        "labels": len(labels),
        "positive_labels": sum(row["scenes"] > 0 for row in per_label),
        "ready_labels": sum(bool(row["ready"]) for row in per_label),
        "minimum_scenes": minimum_scenes,
        "minimum_active_seconds": minimum_active_seconds,
        "labels_by_minimum_scene_support": {
            str(threshold): sum(row["scenes"] >= threshold for row in per_label)
            for threshold in thresholds
        },
        "scene_support_min": min((row["scenes"] for row in per_label), default=0),
        "scene_support_median": (
            sorted(row["scenes"] for row in per_label)[len(per_label) // 2]
            if per_label
            else 0
        ),
        "under_supported": [row for row in per_label if not bool(row["ready"])],
        "per_label": per_label,
    }


def resolve_manifest_audio_path(manifest_path: Path, mixture_path: str) -> Path:
    """Resolve a mixture path for optional downstream audio hashing."""

    candidate = Path(mixture_path)
    if candidate.is_absolute():
        return candidate
    for base in (Path.cwd(), manifest_path.parent, manifest_path.parent.parent):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (Path.cwd() / candidate).resolve()
