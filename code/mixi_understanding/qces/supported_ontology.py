"""Deterministic, support-gated ontology selection for QCES detectors.

The selector works from local AudioSet-Strong timestamp metadata.  It never
downloads audio.  Selection and materialization are deliberately separate:
metadata proves that a class *can* meet the requested support, while local
manifest audits prove how much audio is already available now.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FORMAT = "qces_supported_ontology_v1"


def safe_label(value: str) -> str:
    text = str(value).strip().replace("&", "and")
    text = re.sub(r",\s*", "_and_", text)
    text = re.sub(r"\s+", "_", text)
    return text.replace("/", "_").replace('"', "")


# This fallback is used only when the official AudioSet ontology JSON is not
# available locally.  It is versioned and surfaced in every receipt.  Entries
# are broad parents or generic catchalls whose children are more useful as
# independently verifiable acoustic events.
CURATED_PARENT_OR_GENERIC_V1: dict[str, str] = {
    safe_label(name): reason
    for name, reason in (
        ("Generic impact sounds", "generic_catchall"),
        ("Background noise", "generic_catchall"),
        ("Human sounds", "broad_parent"),
        ("Human voice", "broad_parent"),
        ("Music", "broad_parent"),
        ("Mechanisms", "broad_parent"),
        ("Noise", "generic_catchall"),
        ("Sound effect", "generic_catchall"),
        ("Surface contact", "broad_parent"),
        ("Source-ambiguous sounds", "generic_catchall"),
        ("Channel, environment and background", "generic_catchall"),
        ("Environmental noise", "generic_catchall"),
        ("Unknown sound", "generic_catchall"),
        ("Silence", "generic_catchall"),
        ("Speech", "broad_parent"),
        ("Singing", "broad_parent"),
        ("Crowd", "broad_parent"),
        ("Laughter", "broad_parent"),
        ("Crying, sobbing", "broad_parent"),
        ("Animal", "broad_parent"),
        ("Wild animals", "broad_parent"),
        ("Bird", "broad_parent"),
        ("Bird vocalization, bird call, bird song", "broad_parent"),
        ("Dog", "broad_parent"),
        ("Cat", "broad_parent"),
        ("Frog", "broad_parent"),
        ("Horse", "broad_parent"),
        ("Chicken, rooster", "broad_parent"),
        ("Insect", "broad_parent"),
        ("Vehicle", "broad_parent"),
        ("Motor vehicle (road)", "broad_parent"),
        ("Car", "broad_parent"),
        ("Aircraft", "broad_parent"),
        ("Boat, Water vehicle", "broad_parent"),
        ("Train", "broad_parent"),
        ("Rail transport", "broad_parent"),
        ("Emergency vehicle", "broad_parent"),
        ("Engine", "broad_parent"),
        ("Alarm", "broad_parent"),
        ("Siren", "broad_parent"),
        ("Bell", "broad_parent"),
        ("Whistle", "broad_parent"),
        ("Door", "broad_parent"),
        ("Glass", "broad_parent"),
        ("Water", "broad_parent"),
        ("Liquid", "broad_parent"),
        ("Fire", "broad_parent"),
        ("Rain", "broad_parent"),
        ("Ocean", "broad_parent"),
        ("Explosion", "broad_parent"),
        ("Tools", "broad_parent"),
        ("Power tool", "broad_parent"),
        ("Camera", "broad_parent"),
        ("Clock", "broad_parent"),
        ("Breaking", "broad_parent"),
        ("Bouncing", "broad_parent"),
        ("Effects unit", "broad_parent"),
        ("Hands", "broad_parent"),
        ("Dishes, pots, and pans", "broad_parent"),
        ("Radio", "broad_source_without_fixed_event"),
        ("Television", "broad_source_without_fixed_event"),
    )
}


@dataclass(frozen=True)
class StrongEvent:
    segment_id: str
    video_id: str
    mid: str
    label: str
    display_name: str
    onset_seconds: float
    offset_seconds: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.offset_seconds - self.onset_seconds)


@dataclass(frozen=True)
class MetadataCoverage:
    mid: str
    label: str
    display_name: str
    train_videos: int
    eval_videos: int
    train_events: int
    eval_events: int
    train_seconds: float
    eval_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionResult:
    selected: tuple[MetadataCoverage, ...]
    audit_rows: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]


@dataclass(frozen=True)
class HierarchyInfo:
    descendants: dict[str, set[str]]
    restrictions: dict[str, set[str]]


def _video_id(segment_id: str) -> str:
    return str(segment_id).rsplit("_", 1)[0]


def _stable_key(seed: int, split: str, value: str) -> str:
    return hashlib.sha256(f"{FORMAT}:{seed}:{split}:{value}".encode("utf-8")).hexdigest()


def load_strong_metadata(
    metadata_dir: Path,
) -> tuple[dict[str, MetadataCoverage], dict[str, dict[str, list[StrongEvent]]]]:
    """Load class coverage and full per-video events from local CSV metadata."""

    metadata_dir = metadata_dir.resolve()
    mid_to_display: dict[str, str] = {}
    with (metadata_dir / "class_labels_indices_strong.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.reader(handle):
            if len(row) != 2:
                raise ValueError(f"invalid strong class row: {row!r}")
            mid_to_display[str(row[0])] = str(row[1])

    events_by_split: dict[str, dict[str, list[StrongEvent]]] = {
        "train": defaultdict(list),
        "eval": defaultdict(list),
    }
    videos_by_split_mid: dict[str, dict[str, set[str]]] = {
        "train": defaultdict(set),
        "eval": defaultdict(set),
    }
    event_counts: dict[str, Counter[str]] = {
        "train": Counter(),
        "eval": Counter(),
    }
    active_seconds: dict[str, Counter[str]] = {
        "train": Counter(),
        "eval": Counter(),
    }
    for split, filename in (
        ("train", "audioset_train_strong.csv"),
        ("eval", "audioset_eval_strong.csv"),
    ):
        with (metadata_dir / filename).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"segment_id", "start_time_seconds", "end_time_seconds", "label"}
            if set(reader.fieldnames or ()) != required:
                raise ValueError(f"invalid {filename} header: {reader.fieldnames}")
            for row in reader:
                mid = str(row["label"])
                display = mid_to_display.get(mid)
                if display is None:
                    raise ValueError(f"strong timestamp MID missing from class map: {mid}")
                onset = float(row["start_time_seconds"])
                offset = float(row["end_time_seconds"])
                if offset <= onset:
                    continue
                segment_id = str(row["segment_id"])
                video_id = _video_id(segment_id)
                events_by_split[split][video_id].append(
                    StrongEvent(
                        segment_id=segment_id,
                        video_id=video_id,
                        mid=mid,
                        label=safe_label(display),
                        display_name=display,
                        onset_seconds=onset,
                        offset_seconds=offset,
                    )
                )
                videos_by_split_mid[split][mid].add(video_id)
                event_counts[split][mid] += 1
                active_seconds[split][mid] += offset - onset

    coverage: dict[str, MetadataCoverage] = {}
    for mid, display in mid_to_display.items():
        label = safe_label(display)
        if not event_counts["train"][mid] and not event_counts["eval"][mid]:
            continue
        coverage[label] = MetadataCoverage(
            mid=mid,
            label=label,
            display_name=display,
            train_videos=len(videos_by_split_mid["train"][mid]),
            eval_videos=len(videos_by_split_mid["eval"][mid]),
            train_events=int(event_counts["train"][mid]),
            eval_events=int(event_counts["eval"][mid]),
            train_seconds=float(active_seconds["train"][mid]),
            eval_seconds=float(active_seconds["eval"][mid]),
        )
    compact_events = {
        split: {video_id: list(events) for video_id, events in rows.items()}
        for split, rows in events_by_split.items()
    }
    return coverage, compact_events


def load_hierarchy(path: Path | None) -> HierarchyInfo:
    """Load official AudioSet ``ontology.json`` as MID -> descendants."""

    if path is None:
        return HierarchyInfo(descendants={}, restrictions={})
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("AudioSet ontology JSON must contain a list")
    children: dict[str, set[str]] = {}
    restrictions: dict[str, set[str]] = {}
    for row in payload:
        mid = str(row.get("id") or "")
        if mid:
            children[mid] = {str(value) for value in row.get("child_ids") or () if value}
            restrictions[mid] = {
                str(value).strip().lower()
                for value in row.get("restrictions") or ()
                if str(value).strip()
            }

    descendants: dict[str, set[str]] = {}

    def visit(mid: str, trail: frozenset[str] = frozenset()) -> set[str]:
        if mid in descendants:
            return descendants[mid]
        if mid in trail:
            raise ValueError(f"cycle in AudioSet ontology at {mid}")
        result: set[str] = set()
        for child in children.get(mid, set()):
            result.add(child)
            result.update(visit(child, trail | {mid}))
        descendants[mid] = result
        return result

    for mid in children:
        visit(mid)
    return HierarchyInfo(descendants=descendants, restrictions=restrictions)


def select_supported_ontology(
    coverage: Mapping[str, MetadataCoverage],
    *,
    target_labels: int = 200,
    minimum_train_videos: int = 100,
    minimum_eval_videos: int = 20,
    hierarchy_descendants: Mapping[str, set[str]] | None = None,
    hierarchy_restrictions: Mapping[str, set[str]] | None = None,
    curated_exclusions: Mapping[str, str] = CURATED_PARENT_OR_GENERIC_V1,
) -> SelectionResult:
    """Select specific, support-qualified labels without relaxing gates."""

    if target_labels <= 0:
        raise ValueError("target_labels must be positive")
    hierarchy_descendants = hierarchy_descendants or {}
    hierarchy_restrictions = hierarchy_restrictions or {}
    eligible = {
        label: row
        for label, row in coverage.items()
        if row.train_videos >= minimum_train_videos
        and row.eval_videos >= minimum_eval_videos
    }
    hierarchy_nonleaf = {
        row.label
        for row in eligible.values()
        if hierarchy_descendants.get(row.mid, set())
    }
    hierarchy_restricted = {
        row.label
        for row in eligible.values()
        if hierarchy_restrictions.get(row.mid, set()) & {"abstract", "blacklist"}
    }
    hierarchy_missing = {
        row.label
        for row in eligible.values()
        if hierarchy_descendants and row.mid not in hierarchy_descendants
    }

    def quality(row: MetadataCoverage) -> tuple[float, int, int, str]:
        bottleneck = min(
            row.train_videos / max(minimum_train_videos, 1),
            row.eval_videos / max(minimum_eval_videos, 1),
        )
        return (-bottleneck, -row.train_videos, -row.eval_videos, row.label)

    specific = [
        row
        for label, row in eligible.items()
        if label not in curated_exclusions
        and label not in hierarchy_nonleaf
        and label not in hierarchy_restricted
        and label not in hierarchy_missing
    ]
    specific.sort(key=quality)
    selected: list[MetadataCoverage] = specific[:target_labels]
    selected_mids = {row.mid for row in selected}

    # If an official hierarchy is supplied and fewer than target leaves exist,
    # add only non-leaf nodes that have no selected ancestor/descendant.  The
    # fallback curated policy is hard: known broad/generic labels never fill a
    # quota merely to make the number look complete.
    if hierarchy_descendants and len(selected) < target_labels:
        inverse_ancestors: dict[str, set[str]] = defaultdict(set)
        for ancestor, descendants in hierarchy_descendants.items():
            for descendant in descendants:
                inverse_ancestors[descendant].add(ancestor)
        fallback = [
            row
            for label, row in eligible.items()
            if label not in curated_exclusions
            and label not in hierarchy_restricted
            and label not in hierarchy_missing
            and label not in {item.label for item in selected}
        ]
        fallback.sort(key=quality)
        for row in fallback:
            related = hierarchy_descendants.get(row.mid, set()) | inverse_ancestors.get(
                row.mid, set()
            )
            if related & selected_mids:
                continue
            selected.append(row)
            selected_mids.add(row.mid)
            if len(selected) == target_labels:
                break

    selected_labels = {row.label for row in selected}
    audit_rows: list[dict[str, Any]] = []
    for label, row in sorted(coverage.items()):
        if row.train_videos < minimum_train_videos or row.eval_videos < minimum_eval_videos:
            status = "below_metadata_support_gate"
            exclusion_reason = None
        elif label in curated_exclusions:
            status = "excluded_curated_parent_or_generic"
            exclusion_reason = curated_exclusions[label]
        elif label in hierarchy_restricted:
            status = "excluded_hierarchy_restriction"
            exclusion_reason = ",".join(sorted(hierarchy_restrictions.get(row.mid, set())))
        elif label in hierarchy_missing:
            status = "excluded_missing_from_official_hierarchy"
            exclusion_reason = "mid_not_present_in_official_ontology_snapshot"
        elif label in hierarchy_nonleaf and label not in selected_labels:
            status = "excluded_hierarchy_nonleaf"
            exclusion_reason = "eligible_descendant_exists"
        elif label in selected_labels:
            status = "selected"
            exclusion_reason = None
        else:
            status = "not_selected_after_rank"
            exclusion_reason = None
        audit_rows.append(
            {
                **row.to_dict(),
                "status": status,
                "exclusion_reason": exclusion_reason,
            }
        )
    selected_parent_child_pairs = sorted(
        (ancestor.label, descendant.label)
        for ancestor in selected
        for descendant in selected
        if descendant.mid in hierarchy_descendants.get(ancestor.mid, set())
    )
    receipt = {
        "format": FORMAT,
        "target_labels": target_labels,
        "minimum_train_videos": minimum_train_videos,
        "minimum_eval_videos": minimum_eval_videos,
        "metadata_labels": len(coverage),
        "metadata_eligible_labels_before_specificity_filter": len(eligible),
        "curated_or_generic_eligible_exclusions": sum(
            label in eligible for label in curated_exclusions
        ),
        "hierarchy_source": (
            "official_ontology_json" if hierarchy_descendants else "curated_fallback_v1"
        ),
        "hierarchy_nonleaf_eligible_labels": len(hierarchy_nonleaf),
        "hierarchy_restricted_eligible_labels": len(hierarchy_restricted),
        "hierarchy_missing_eligible_labels": len(hierarchy_missing),
        "specific_eligible_labels": len(specific),
        "selected_labels": len(selected),
        "selection_feasible": len(selected) == target_labels,
        "selection_shortfall": max(0, target_labels - len(selected)),
        "parent_child_audit_complete": bool(hierarchy_descendants)
        and not any(row.label in hierarchy_missing for row in selected),
        "parent_child_pairs_in_selected": len(selected_parent_child_pairs),
        "parent_child_pair_samples": selected_parent_child_pairs[:20],
    }
    return SelectionResult(
        selected=tuple(selected), audit_rows=tuple(audit_rows), receipt=receipt
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.resolve().open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def materialized_video_assignments(
    clean_split_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, set[str]]:
    """Extract already downloaded AudioSet video IDs from clean split rows."""

    result = {"train": set(), "dev": set(), "test": set()}
    for split in result:
        for row in clean_split_rows.get(split, ()):
            source = str(row.get("protocol_source") or row.get("source_route") or "")
            if "audioset" not in source.lower():
                continue
            video_id = str(row.get("video_id") or "").strip()
            if video_id:
                result[split].add(video_id)
    return result


def audit_materialized_support(
    selected: Sequence[MetadataCoverage],
    *,
    strong_events: Mapping[str, Mapping[str, Sequence[StrongEvent]]],
    audioset_video_ids: Mapping[str, set[str]],
    preserved_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Measure current train/dev/test scene and active-second support.

    AudioSet labels are restored from full strong metadata for already-downloaded
    videos, so labels omitted by a historical ontology filter are not mistaken
    for missing audio.
    """

    selected_labels = {row.label for row in selected}
    scenes: dict[str, dict[str, set[str]]] = {
        label: {split: set() for split in ("train", "dev", "test")}
        for label in selected_labels
    }
    seconds: dict[str, Counter[str]] = {
        label: Counter() for label in selected_labels
    }
    audioset_scenes: dict[str, Counter[str]] = {
        label: Counter() for label in selected_labels
    }
    preserved_scenes: dict[str, Counter[str]] = {
        label: Counter() for label in selected_labels
    }
    for split in ("train", "dev", "test"):
        metadata_split = "eval" if split == "test" else "train"
        for video_id in audioset_video_ids.get(split, set()):
            per_label_seconds: Counter[str] = Counter()
            for event in strong_events.get(metadata_split, {}).get(video_id, ()):
                if event.label in selected_labels:
                    per_label_seconds[event.label] += event.duration_seconds
            for label, active_seconds in per_label_seconds.items():
                scene_id = f"audioset:{video_id}"
                scenes[label][split].add(scene_id)
                seconds[label][split] += active_seconds
                audioset_scenes[label][split] += 1

        for row_index, row in enumerate(preserved_rows.get(split, ())):
            scene_id = str(row.get("scene_id") or f"preserved:{split}:{row_index}")
            per_label_seconds: Counter[str] = Counter()
            events = row.get("events") if "events" in row else row.get("gold_events")
            for event in events or ():
                label = str(event.get("label") or "")
                if label not in selected_labels:
                    continue
                per_label_seconds[label] += max(
                    0.0,
                    float(event.get("offset_seconds", 0.0))
                    - float(event.get("onset_seconds", 0.0)),
                )
            for label, active_seconds in per_label_seconds.items():
                scenes[label][split].add(f"preserved:{scene_id}")
                seconds[label][split] += active_seconds
                preserved_scenes[label][split] += 1

    rows: list[dict[str, Any]] = []
    for metadata in selected:
        label = metadata.label
        row: dict[str, Any] = {
            "label": label,
            "display_name": metadata.display_name,
            "mid": metadata.mid,
        }
        for split in ("train", "dev", "test"):
            row[f"materialized_{split}_scenes"] = len(scenes[label][split])
            row[f"materialized_{split}_active_seconds"] = float(seconds[label][split])
            row[f"materialized_{split}_audioset_scenes"] = int(
                audioset_scenes[label][split]
            )
            row[f"materialized_{split}_preserved_scenes"] = int(
                preserved_scenes[label][split]
            )
        rows.append(row)

    def split_summary(split: str) -> dict[str, Any]:
        counts = [int(row[f"materialized_{split}_scenes"]) for row in rows]
        active = [float(row[f"materialized_{split}_active_seconds"]) for row in rows]
        return {
            "classes_with_positive": sum(value > 0 for value in counts),
            "scene_support_min": min(counts, default=0),
            "scene_support_median": sorted(counts)[len(counts) // 2] if counts else 0,
            "scene_support_max": max(counts, default=0),
            "active_seconds_min": min(active, default=0.0),
            "active_seconds_median": sorted(active)[len(active) // 2] if active else 0.0,
            "active_seconds_max": max(active, default=0.0),
        }

    return {
        "per_label": rows,
        "split_summary": {
            split: split_summary(split) for split in ("train", "dev", "test")
        },
        "classes_positive_all_splits": sum(
            all(int(row[f"materialized_{split}_scenes"]) > 0 for split in ("train", "dev", "test"))
            for row in rows
        ),
    }


def build_missing_materialization_plan(
    selected: Sequence[MetadataCoverage],
    *,
    strong_events: Mapping[str, Mapping[str, Sequence[StrongEvent]]],
    existing_train_video_ids: set[str],
    existing_eval_video_ids: set[str],
    target_train_videos: int = 100,
    target_eval_videos: int = 20,
    seed: int = 2028,
) -> dict[str, Any]:
    """Create exact unseen-video plans that fill AudioSet support deficits."""

    selected_labels = {row.label for row in selected}
    outputs: dict[str, list[dict[str, Any]]] = {}
    per_split_counts: dict[str, dict[str, Any]] = {}
    for split, existing_ids, target in (
        ("train", existing_train_video_ids, target_train_videos),
        ("eval", existing_eval_video_ids, target_eval_videos),
    ):
        current: Counter[str] = Counter()
        for video_id in existing_ids:
            current.update(
                {
                    event.label
                    for event in strong_events.get(split, {}).get(video_id, ())
                    if event.label in selected_labels
                }
            )
        deficits = {
            label: max(0, target - int(current[label])) for label in selected_labels
        }
        initial_deficits = dict(deficits)
        plan: list[dict[str, Any]] = []
        candidates = [
            video_id
            for video_id in strong_events.get(split, {})
            if video_id not in existing_ids
        ]
        candidates.sort(key=lambda value: _stable_key(seed, split, value))
        for video_id in candidates:
            events = [
                event
                for event in strong_events[split][video_id]
                if event.label in selected_labels
            ]
            labels = sorted({event.label for event in events})
            helpful = [label for label in labels if deficits[label] > 0]
            if not helpful:
                continue
            for label in helpful:
                deficits[label] -= 1
            plan.append(
                {
                    "format": FORMAT,
                    "hf_dataset": "enyoukai/AudioSet-Strong",
                    "metadata_split": split,
                    "hf_split": "train" if split == "train" else "test",
                    "video_id": video_id,
                    "segment_ids": sorted({event.segment_id for event in events}),
                    "labels": labels,
                    "covers_deficit_labels": helpful,
                    "events": [asdict(event) for event in events],
                }
            )
            if not any(deficits.values()):
                break
        unresolved = {label: value for label, value in sorted(deficits.items()) if value > 0}
        outputs[split] = plan
        per_split_counts[split] = {
            "target_videos_per_label": target,
            "existing_videos": len(existing_ids),
            "planned_new_videos": len(plan),
            "classes_already_at_target": sum(value == 0 for value in initial_deficits.values()),
            "classes_requiring_materialization": sum(value > 0 for value in initial_deficits.values()),
            "total_initial_class_video_deficit": sum(initial_deficits.values()),
            "total_unresolved_class_video_deficit": sum(unresolved.values()),
            "unresolved": unresolved,
        }
        per_split_counts[split]["per_label"] = {
            label: {
                "existing": int(current[label]),
                "target": target,
                "missing_before_plan": int(initial_deficits[label]),
                "planned_coverage": int(initial_deficits[label] - deficits[label]),
                "missing_after_plan": int(deficits[label]),
            }
            for label in sorted(selected_labels)
        }
    return {"plans": outputs, "summary": per_split_counts}
