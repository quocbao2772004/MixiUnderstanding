#!/usr/bin/env python3
"""Build a source-disjoint polyphonic onset-order stress test.

This sidecar leaves the historical locked and repeated-ordinal tests intact.
For every frozen ontology label it emits one answerable and one NONE scene.
The question is defined by onset order (not by non-overlapping succession):
``Which sound has the next onset after the first <anchor>?``.

The answer label is never used to select a source for the anchor or by the
downstream executor.  Positive and NONE scenes share the same text-side key,
requested overlap tier, concurrency tier, and relative-gain tier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import soundfile as sf

from mixi_understanding.qces.clean_evidence_scenes import (
    DEFAULT_SAMPLE_RATE,
    FRAME_HOP_SECONDS,
    QA_FORMAT,
    SCENE_FORMAT,
    SCENE_SECONDS,
    CleanSource,
    QARow,
    _atomic_write_text,
    _choose_active_subcrop,
    audit_rendered_reconstruction,
    audit_text_shortcuts,
    load_source_bank,
    partition_sources,
    render_scene_audio,
    sha256_file,
)


FORMAT = "qces_overlap_onset_stress_test_v1"
DEFAULT_SEED = 2087
MINIMUM_EVENT_FRAMES = 4
OVERLAP_TIERS = (0.25, 0.50, 0.75, 0.90)
GAIN_TIERS_DB = (-6.0, 0.0, 6.0)


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=data / "source_bank_accepted.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_191.txt")
    parser.add_argument(
        "--locked-test-scenes", type=Path,
        default=data / "multievent/scene_manifest_test.jsonl",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/qces_full191_overlap_onset_stress_test_v1",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-event-seconds", type=float, default=1.20)
    parser.add_argument("--verify-source-hash", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _digest(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{FORMAT}:{seed}:{namespace}:{value}".encode()).hexdigest()


def _jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def _identity_set(source: CleanSource) -> set[tuple[str, str]]:
    return set(source.hard_identities)


def _manifest_identities(path: Path) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for event in row.get("events") or []:
                for field, canonical in (
                    ("source_id", "source_id"),
                    ("source_video_id", "video_id"),
                    ("source_sha256", "source_sha256"),
                    ("source_path", "audio_path"),
                ):
                    value = str(event.get(field) or "").strip()
                    if value:
                        identities.add((canonical, value))
    return identities


def _choose_source(
    queue: list[CleanSource], used_identities: set[tuple[str, str]]
) -> CleanSource:
    while queue:
        source = queue.pop()
        identities = _identity_set(source)
        if identities & used_identities:
            continue
        used_identities.update(identities)
        return source
    raise RuntimeError("source queue exhausted after hard-identity filtering")


def _crop_and_length(
    source: CleanSource, *, max_event_seconds: float, seed: int
) -> tuple[tuple[float, float], int]:
    crop = _choose_active_subcrop(source, max_event_seconds, seed)
    frames = max(
        MINIMUM_EVENT_FRAMES,
        int(math.ceil((crop[1] - crop[0]) / FRAME_HOP_SECONDS - 1e-9)),
    )
    return crop, frames


def _event(
    source: CleanSource,
    *,
    scene_id: str,
    index: int,
    role: str,
    onset_frame: int,
    length_frames: int,
    crop: tuple[float, float],
    gain_db: float,
    label_to_id: Mapping[str, int],
    requested_overlap: float,
    concurrency: int,
) -> dict[str, Any]:
    # Avoid compressing a long crop into a deliberately shorter common
    # polyphonic interval.  Very short transients are only stretched to the
    # documented 160-ms minimum used by this stress protocol.
    requested_seconds = length_frames * FRAME_HOP_SECONDS
    crop_end = min(crop[1], crop[0] + requested_seconds)
    if crop_end <= crop[0]:
        raise ValueError(f"empty source crop: {source.source_id}")
    offset_frame = onset_frame + length_frames
    if onset_frame < 0 or offset_frame > round(SCENE_SECONDS / FRAME_HOP_SECONDS):
        raise ValueError(f"event outside scene grid: {scene_id}:{role}")
    return {
        "event_id": f"{scene_id}:e{index:02d}",
        "event_kind": "semantic",
        "label": source.label,
        "label_id": int(label_to_id[source.label]),
        "stress_role": role,
        "onset_seconds": onset_frame * FRAME_HOP_SECONDS,
        "offset_seconds": offset_frame * FRAME_HOP_SECONDS,
        "onset_frame": onset_frame,
        "offset_frame": offset_frame,
        "source_id": source.source_id,
        "source_video_id": source.video_id,
        "source_sha256": source.source_sha256,
        "source_path": source.audio_path,
        "source_annotation_onset_seconds": source.active_onset_seconds,
        "source_annotation_offset_seconds": source.active_offset_seconds,
        "source_crop_onset_seconds": crop[0],
        "source_crop_offset_seconds": crop_end,
        "source_crop_was_truncated": crop != (
            source.active_onset_seconds, source.active_offset_seconds
        ) or crop_end < crop[1],
        "component_gain_db": float(gain_db),
        "cleanliness_passed": source.cleanliness_passed,
        "audibility_passed": source.audibility_passed,
        "cleanliness_tier": source.cleanliness_tier,
        "audibility_score": source.audibility_score,
        "requested_overlap_fraction": float(requested_overlap),
        "target_concurrency": int(concurrency),
    }


def _make_scene(
    *,
    scene_id: str,
    sources: Sequence[CleanSource],
    roles: Sequence[str],
    answerable: bool,
    requested_overlap: float,
    concurrency: int,
    gain_delta_db: float,
    label_to_id: Mapping[str, int],
    max_event_seconds: float,
    seed: int,
) -> dict[str, Any]:
    if len(sources) != len(roles):
        raise ValueError("source/role length mismatch")
    crops_and_lengths = [
        _crop_and_length(source, max_event_seconds=max_event_seconds, seed=seed)
        for source in sources
    ]
    crop_by_role = {role: value[0] for role, value in zip(roles, crops_and_lengths)}
    raw_length = {role: value[1] for role, value in zip(roles, crops_and_lengths)}
    overlap_roles = {"anchor", "answer", "overlap_partner", "overlap_partner_2"}
    common = min(raw_length[role] for role in roles if role in overlap_roles)
    common = max(MINIMUM_EVENT_FRAMES, common)
    length_by_role = {
        role: common if role in overlap_roles else raw_length[role] for role in roles
    }
    desired = max(1, int(round(requested_overlap * common)))
    desired = min(desired, common - 1)
    starts: dict[str, int] = {"prefix": 10}
    if answerable:
        starts["anchor"] = 75
        starts["answer"] = starts["anchor"] + common - desired
        if concurrency == 3:
            starts["overlap_partner_2"] = starts["answer"] + 1
    else:
        starts["overlap_partner"] = 75
        if concurrency == 2:
            starts["anchor"] = starts["overlap_partner"] + common - desired
        else:
            starts["overlap_partner_2"] = starts["overlap_partner"] + 1
            starts["anchor"] = starts["overlap_partner_2"] + common - desired

    gains = {
        "prefix": -3.0,
        "anchor": 0.0,
        "answer": float(gain_delta_db),
        "overlap_partner": float(gain_delta_db),
        "overlap_partner_2": -3.0,
    }
    events = [
        _event(
            source,
            scene_id=scene_id,
            index=index,
            role=role,
            onset_frame=starts[role],
            length_frames=length_by_role[role],
            crop=crop_by_role[role],
            gain_db=gains[role],
            label_to_id=label_to_id,
            requested_overlap=requested_overlap,
            concurrency=concurrency,
        )
        for index, (source, role) in enumerate(zip(sources, roles, strict=True))
    ]
    events.sort(key=lambda event: (event["onset_frame"], event["event_id"]))
    for index, event in enumerate(events):
        event["event_id"] = f"{scene_id}:e{index:02d}"
    target_left = next(
        event for event in events
        if event["stress_role"] == ("anchor" if answerable else "overlap_partner" if concurrency == 2 else "overlap_partner_2")
    )
    target_right = next(
        event for event in events
        if event["stress_role"] == ("answer" if answerable else "anchor")
    )
    intersection = max(
        0.0,
        min(target_left["offset_seconds"], target_right["offset_seconds"])
        - max(target_left["onset_seconds"], target_right["onset_seconds"]),
    )
    minimum_duration = min(
        target_left["offset_seconds"] - target_left["onset_seconds"],
        target_right["offset_seconds"] - target_right["onset_seconds"],
    )
    return {
        "format": SCENE_FORMAT,
        "scene_id": scene_id,
        "scene_family_id": scene_id,
        "split": "test",
        "source_route": "synthetic_clean_single_event_bank_overlap_onset_stress",
        "mixture_path": f"audio/test/{scene_id}.wav",
        "duration_seconds": SCENE_SECONDS,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "audio_num_frames": int(round(SCENE_SECONDS * DEFAULT_SAMPLE_RATE)),
        "audio_num_channels": 1,
        "events": events,
        "recipe_kind": "overlap_onset_positive" if answerable else "overlap_onset_none",
        "round_index": 0,
        "rendered": False,
        "source_video_ids": sorted({source.video_id for source in sources}),
        "source_sha256s": sorted({source.source_sha256 for source in sources}),
        "overlap_protocol": {
            "relation_semantics": "immediate_next_event_by_strict_onset_order",
            "requested_overlap_fraction": requested_overlap,
            "realized_target_pair_overlap_fraction": intersection / max(minimum_duration, 1e-8),
            "target_concurrency": concurrency,
            "requested_answer_or_partner_gain_delta_db": gain_delta_db,
            "answerable": answerable,
        },
    }


def _qa(scene: Mapping[str, Any]) -> QARow:
    events = list(scene["events"])
    anchor_index = next(
        index for index, event in enumerate(events) if event["stress_role"] == "anchor"
    )
    anchor = events[anchor_index]
    answerable = bool(scene["overlap_protocol"]["answerable"])
    answer = events[anchor_index + 1] if answerable else None
    if answerable and answer is not next(
        event for event in events if event["stress_role"] == "answer"
    ):
        raise RuntimeError("positive answer is not the immediate next onset")
    if not answerable and anchor_index != len(events) - 1:
        raise RuntimeError("NONE anchor is not the final onset")
    anchor_interval = (float(anchor["onset_seconds"]), float(anchor["offset_seconds"]))
    answer_interval = None if answer is None else (
        float(answer["onset_seconds"]), float(answer["offset_seconds"])
    )
    label_text = str(anchor["label"]).replace("_and_", " / ").replace("_", " ")
    return QARow(
        item_id=f"{scene['scene_id']}:next_onset_after_anchor",
        scene_id=str(scene["scene_id"]),
        split="test",
        relation="after",
        question=f"Which sound has the next onset after the first {label_text}?",
        answer="no_evidence" if answer is None else str(answer["label"]),
        no_evidence=answer is None,
        no_evidence_reason="no_event_onset_after_anchor" if answer is None else None,
        anchor_label=str(anchor["label"]),
        anchor_ordinal=1,
        answer_label=None if answer is None else str(answer["label"]),
        anchor_event_id=str(anchor["event_id"]),
        answer_event_id=None if answer is None else str(answer["event_id"]),
        evidence_event_ids=(str(anchor["event_id"]),) if answer is None else (
            str(anchor["event_id"]), str(answer["event_id"])
        ),
        gold_anchor_interval=anchor_interval,
        gold_answer_interval=answer_interval,
        gold_verification_interval=(anchor_interval[1], SCENE_SECONDS) if answer is None else None,
        gold_evidence_intervals=((anchor_interval[0], SCENE_SECONDS),) if answer is None else (
            anchor_interval, answer_interval
        ),
        mixture_path=str(scene["mixture_path"]),
        source_route="synthetic_clean_single_event_bank_overlap_onset_stress",
        format=QA_FORMAT,
    )


def _maximum_concurrency(events: Sequence[Mapping[str, Any]]) -> int:
    points = []
    for event in events:
        points.extend(
            ((float(event["onset_seconds"]), 1), (float(event["offset_seconds"]), -1))
        )
    active = maximum = 0
    for _, delta in sorted(points, key=lambda value: (value[0], value[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _measured_target_snr(scene: Mapping[str, Any], root: Path) -> float | None:
    events = list(scene["events"])
    anchor = next(event for event in events if event["stress_role"] == "anchor")
    other_role = "answer" if scene["overlap_protocol"]["answerable"] else (
        "overlap_partner" if scene["overlap_protocol"]["target_concurrency"] == 2
        else "overlap_partner_2"
    )
    other = next(event for event in events if event["stress_role"] == other_role)
    left, _ = sf.read(root / str(anchor["component_path"]), dtype="float32")
    right, _ = sf.read(root / str(other["component_path"]), dtype="float32")
    start = max(int(anchor["placement_start_sample"]), int(other["placement_start_sample"]))
    end = min(
        int(anchor["placement_start_sample"]) + len(left),
        int(other["placement_start_sample"]) + len(right),
    )
    if end <= start:
        return None
    left_slice = left[start - int(anchor["placement_start_sample"]): end - int(anchor["placement_start_sample"])]
    right_slice = right[start - int(other["placement_start_sample"]): end - int(other["placement_start_sample"])]
    left_energy = float(np.mean(np.square(left_slice, dtype=np.float64)))
    right_energy = float(np.mean(np.square(right_slice, dtype=np.float64)))
    return 10.0 * math.log10((right_energy + 1e-12) / (left_energy + 1e-12))


def main() -> None:
    args = parse_args()
    labels = [
        line.strip() for line in args.ontology.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(labels) != 191 or len(labels) != len(set(labels)):
        raise ValueError(f"overlap stress test requires 191 frozen labels, got {len(labels)}")
    if args.max_event_seconds <= 0:
        raise ValueError("--max-event-seconds must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}; use --overwrite")

    sources = load_source_bank(args.source_bank.resolve(), require_audio_file=True)
    partitioned, partition_receipt = partition_sources(
        sources, labels, seed=2041, dev_fraction=0.20
    )
    test_sources = list(partitioned["test"])
    train_dev_identities = {
        identity
        for split in ("train", "dev")
        for source in partitioned[split]
        for identity in source.hard_identities
    }
    if any(_identity_set(source) & train_dev_identities for source in test_sources):
        raise RuntimeError("official-eval source overlaps train/dev identities")
    locked_identities = _manifest_identities(args.locked_test_scenes)

    by_label: dict[str, list[CleanSource]] = defaultdict(list)
    for source in test_sources:
        by_label[source.label].append(source)
    support = {label: len(by_label[label]) for label in labels}
    if min(support.values()) < 2:
        raise RuntimeError("each anchor class requires two official-eval sources")
    queues: dict[str, list[CleanSource]] = {}
    for label in labels:
        ordered = sorted(
            by_label[label],
            key=lambda source: (
                bool(_identity_set(source) & locked_identities),
                _digest(args.seed, f"source:{label}", source.source_id),
            ),
        )
        queues[label] = list(reversed(ordered))
    used_identities: set[tuple[str, str]] = set()
    anchor_sources: dict[str, tuple[CleanSource, CleanSource]] = {}
    for label in sorted(labels, key=lambda value: (support[value], value)):
        anchor_sources[label] = (
            _choose_source(queues[label], used_identities),
            _choose_source(queues[label], used_identities),
        )

    filler_usage: Counter[str] = Counter()

    def filler(excluded: set[str]) -> CleanSource:
        candidates = sorted(
            (label for label in labels if label not in excluded and queues[label]),
            key=lambda label: (
                filler_usage[label],
                _digest(args.seed, f"filler:{sum(filler_usage.values())}", label),
            ),
        )
        for label in candidates:
            try:
                source = _choose_source(queues[label], used_identities)
            except RuntimeError:
                continue
            filler_usage[label] += 1
            return source
        raise RuntimeError(f"no identity-safe filler remains outside {sorted(excluded)}")

    label_to_id = {label: index for index, label in enumerate(labels)}
    raw_scenes = []
    for label_index, label in enumerate(labels):
        requested_overlap = OVERLAP_TIERS[label_index % len(OVERLAP_TIERS)]
        concurrency = 3 if requested_overlap >= 0.75 else 2
        gain_delta = GAIN_TIERS_DB[(label_index // len(OVERLAP_TIERS)) % len(GAIN_TIERS_DB)]
        positive_roles = ["prefix", "anchor", "answer"]
        if concurrency == 3:
            positive_roles.append("overlap_partner_2")
        positive_sources = [filler({label}), anchor_sources[label][0]]
        positive_sources.append(
            filler({label, *(source.label for source in positive_sources)})
        )
        if concurrency == 3:
            positive_sources.append(
                filler({label, *(source.label for source in positive_sources)})
            )
        negative_roles = ["prefix", "overlap_partner"]
        if concurrency == 3:
            negative_roles.append("overlap_partner_2")
        negative_roles.append("anchor")
        negative_sources = [filler({label})]
        negative_sources.append(
            filler({label, *(source.label for source in negative_sources)})
        )
        if concurrency == 3:
            negative_sources.append(
                filler({label, *(source.label for source in negative_sources)})
            )
        negative_sources.append(anchor_sources[label][1])
        raw_scenes.extend(
            (
                _make_scene(
                    scene_id=f"overlap_onset_test_{label_index:03d}_positive",
                    sources=positive_sources,
                    roles=positive_roles,
                    answerable=True,
                    requested_overlap=requested_overlap,
                    concurrency=concurrency,
                    gain_delta_db=gain_delta,
                    label_to_id=label_to_id,
                    max_event_seconds=args.max_event_seconds,
                    seed=args.seed,
                ),
                _make_scene(
                    scene_id=f"overlap_onset_test_{label_index:03d}_none",
                    sources=negative_sources,
                    roles=negative_roles,
                    answerable=False,
                    requested_overlap=requested_overlap,
                    concurrency=concurrency,
                    gain_delta_db=gain_delta,
                    label_to_id=label_to_id,
                    max_event_seconds=args.max_event_seconds,
                    seed=args.seed,
                ),
            )
        )

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        scenes = [
            render_scene_audio(
                scene,
                staging_root=staging,
                verify_source_hash=args.verify_source_hash,
            )
            for scene in raw_scenes
        ]
        qa = [_qa(scene) for scene in scenes]
        shortcut = audit_text_shortcuts(qa)
        if not shortcut["passes"]:
            raise RuntimeError("overlap QA contains a text-only answerability shortcut")
        reconstruction = audit_rendered_reconstruction(scenes, root=staging)
        if not reconstruction["passes"]:
            raise RuntimeError("overlap mixtures fail reconstruction audit")
        if any(_maximum_concurrency(scene["events"]) != scene["overlap_protocol"]["target_concurrency"] for scene in scenes):
            raise RuntimeError("rendered concurrency differs from protocol target")
        pair_snr = [_measured_target_snr(scene, staging) for scene in scenes]
        for scene, value in zip(scenes, pair_snr, strict=True):
            scene["overlap_protocol"]["measured_other_over_anchor_snr_db"] = value

        selected_sources = [event for scene in scenes for event in scene["events"]]
        selected_hard = {
            identity
            for source in test_sources
            if source.source_id in {event["source_id"] for event in selected_sources}
            for identity in source.hard_identities
        }
        if selected_hard & train_dev_identities:
            raise RuntimeError("selected overlap test sources leak train/dev")
        locked_overlap = selected_hard & locked_identities

        scene_manifest = staging / "scene_manifest_test.jsonl"
        detector_manifest = staging / "detector_scene_manifest_overlap_test.jsonl"
        qa_manifest = staging / "qa_manifest_test.jsonl"
        scene_ids = staging / "scene_ids_test.txt"
        ontology_path = staging / "ontology_191.txt"
        _atomic_write_text(scene_manifest, _jsonl(scenes))
        detector_rows = []
        for scene in scenes:
            row = dict(scene)
            row["mixture_path"] = str((output_dir / str(scene["mixture_path"])).resolve())
            detector_rows.append(row)
        _atomic_write_text(detector_manifest, _jsonl(detector_rows))
        _atomic_write_text(qa_manifest, _jsonl(row.to_dict() for row in qa))
        _atomic_write_text(scene_ids, "".join(f"{scene['scene_id']}\n" for scene in scenes))
        _atomic_write_text(ontology_path, "".join(f"{label}\n" for label in labels))

        realized_by_tier: dict[str, list[float]] = defaultdict(list)
        for scene in scenes:
            protocol = scene["overlap_protocol"]
            realized_by_tier[f"{protocol['requested_overlap_fraction']:.2f}"].append(
                float(protocol["realized_target_pair_overlap_fraction"])
            )
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "policy": {
                "relation": "immediate_next_event_by_strict_onset_order",
                "question_template": "Which sound has the next onset after the first <anchor>?",
                "answer_label_used_for_source_or_example_selection": False,
                "answer_label_used_by_executor": False,
                "positive_and_none_per_anchor_key": [1, 1],
                "requested_overlap_tiers": list(OVERLAP_TIERS),
                "concurrency": "2 for 0.25/0.50; 3 for 0.75/0.90",
                "requested_gain_delta_db_tiers": list(GAIN_TIERS_DB),
                "minimum_event_frames": MINIMUM_EVENT_FRAMES,
            },
            "scenes": len(scenes),
            "events": len(selected_sources),
            "qa": len(qa),
            "qa_answerable": sum(not row.no_evidence for row in qa),
            "qa_no_evidence": sum(row.no_evidence for row in qa),
            "anchor_classes": len({row.anchor_label for row in qa}),
            "event_count_distribution": dict(sorted(Counter(len(scene["events"]) for scene in scenes).items())),
            "concurrency_distribution": dict(sorted(Counter(_maximum_concurrency(scene["events"]) for scene in scenes).items())),
            "requested_overlap_distribution": dict(sorted(Counter(f"{scene['overlap_protocol']['requested_overlap_fraction']:.2f}" for scene in scenes).items())),
            "realized_overlap_by_requested_tier": {
                key: {
                    "count": len(values),
                    "minimum": min(values),
                    "mean": sum(values) / len(values),
                    "maximum": max(values),
                }
                for key, values in sorted(realized_by_tier.items())
            },
            "measured_other_over_anchor_snr_db": {
                "minimum": min(value for value in pair_snr if value is not None),
                "mean": sum(value for value in pair_snr if value is not None) / len(pair_snr),
                "maximum": max(value for value in pair_snr if value is not None),
            },
            "official_eval_source_pool": len(test_sources),
            "selected_unique_source_rows": len({event["source_id"] for event in selected_sources}),
            "selected_unique_hard_identities": len(selected_hard),
            "train_dev_hard_identity_overlap": 0,
            "locked_test_hard_identity_overlap": len(locked_overlap),
            "locked_test_reused_source_rows": len(
                {value for field, value in selected_hard if field == "source_id"}
                & {value for field, value in locked_identities if field == "source_id"}
            ),
            "locked_test_overlap_is_allowed_and_reported": True,
            "partition_policy": partition_receipt["policy"],
            "text_shortcut_audit": shortcut,
            "reconstruction_audit": reconstruction,
            "artifacts": {},
        }
        for path in (scene_manifest, detector_manifest, qa_manifest, scene_ids, ontology_path):
            receipt["artifacts"][path.name] = sha256_file(path)
        _atomic_write_text(
            staging / "build_receipt.json",
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
        staging = None
        print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
