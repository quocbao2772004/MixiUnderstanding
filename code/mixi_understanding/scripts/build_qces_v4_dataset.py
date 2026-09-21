#!/usr/bin/env python3
"""Build the shortcut-resistant QCES v4 pilot from pinned AudioTime sources."""
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import scipy
import soundfile as sf

from mixi_understanding.data.qces_schema import NO_EVIDENCE_ANSWER
from mixi_understanding.data.qces_v4_schema import (
    SCHEMA_VERSION,
    parse_qces_v4_record,
)
from mixi_understanding.scripts.build_qa_removal_dataset import (
    SourceClip,
    crop_source_event,
    load_sources,
    natural_source_key,
    relative_posix,
    render_clip,
    replace_validated_output,
    rms,
    sha256_file,
    stable_seed,
    write_jsonl,
    write_wav,
)


BUILDER_VERSION = "4.0.0-pilot"
RECEIPT_NAME = "qces_v4_subset_receipt.json"
SAMPLE_RATE = 32_000
DURATION_SECONDS = 8.0
NUISANCE_DURATION_SECONDS = 5.8
SEMANTIC_EDGE_MARGIN_SECONDS = 0.15
NUISANCE_EDGE_MARGIN_SECONDS = 0.05
HEADROOM = 0.95
FADE_MILLISECONDS = 10.0
EPSILON = 1e-12
SEMANTIC_LABELS = (
    "Croak",
    "Engine knocking",
    "Jackhammer",
    "Chainsaw",
    "Vacuum cleaner",
    "Fire alarm",
    "Mechanical bell",
    "Steam whistle",
    "Helicopter",
    "Printer",
    "Wind chime",
    "Rain",
)
NUISANCE_LABELS = ("Ambulance (siren)", "Sawing", "Mechanical fan")
SPLIT_SCENES = {"train": 6, "val": 3, "test": 3}
SPLIT_SOURCE_INDICES = {"train": (0, 1), "val": (2,), "test": (3,)}
QUESTIONS_PER_SCENE = 16
SNR_LEVELS_DB = (-6.0, -3.0, 0.0, 3.0)
AFTER_TEMPLATES = (
    ("after_direct", "What sound occurs immediately after {anchor}?"),
    ("after_follows", "Which sound follows {anchor}?"),
    ("after_right_after", "What do you hear right after {anchor}?"),
)
BEFORE_TEMPLATES = (
    ("before_direct", "What sound occurs immediately before {anchor}?"),
    ("before_precedes", "Which sound precedes {anchor}?"),
    ("before_right_before", "What do you hear right before {anchor}?"),
)
FIRST_TEMPLATES = (
    ("first_which", "Which sound occurs first, {left} or {right}?"),
    ("first_heard", "Which is heard earlier, {left} or {right}?"),
    ("first_order", "Between {left} and {right}, which comes first?"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--audiotime-root",
        type=Path,
        default=PROJECT_ROOT / "AudioTime-recovered" / "train5000_timestamp",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "qces_v4_pilot",
    )
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_receipt(root: Path) -> tuple[Dict[str, Any], Dict[str, SourceClip]]:
    receipt_path = root / RECEIPT_NAME
    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"missing {receipt_path}; run download_qces_v4_sources.py first"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("format") != "qces_v4_audiotime_subset_v1":
        raise ValueError("unsupported QCES v4 source receipt")
    selected_payload = receipt.get("sources")
    if not isinstance(selected_payload, dict):
        raise TypeError("receipt sources must be an object")
    metadata_sources = {
        source.source_id: source for source in load_sources(root, require_audio=False)
    }
    selected: Dict[str, SourceClip] = {}
    for source_id, identity in selected_payload.items():
        source = metadata_sources.get(source_id)
        if source is None:
            raise ValueError(f"receipt source missing from metadata: {source_id}")
        if source.label != identity.get("label"):
            raise ValueError(f"receipt label mismatch: {source_id}")
        if not source.audio_path.is_file():
            raise FileNotFoundError(source.audio_path)
        if sha256_file(source.audio_path) != identity.get("sha256"):
            raise ValueError(f"receipt hash mismatch: {source_id}")
        selected[source_id] = source
    return receipt, selected


def _sources_by_label(
    selected: Mapping[str, SourceClip],
) -> Dict[str, Tuple[SourceClip, ...]]:
    grouped: Dict[str, List[SourceClip]] = defaultdict(list)
    for source in selected.values():
        grouped[source.label].append(source)
    result = {}
    for label in SEMANTIC_LABELS + NUISANCE_LABELS:
        sources = tuple(sorted(grouped[label], key=natural_source_key))
        if len(sources) != 4:
            raise ValueError(f"expected exactly four pinned sources for {label}")
        result[label] = sources
    return result


def _label_blocks(seed: int) -> Dict[str, List[Tuple[str, ...]]]:
    blocks: Dict[str, List[Tuple[str, ...]]] = {}
    for split, source_indices in SPLIT_SOURCE_INDICES.items():
        split_blocks: List[Tuple[str, ...]] = []
        for cycle, _ in enumerate(source_indices):
            rng = np.random.default_rng(stable_seed(seed, "label-block", split, cycle))
            permutation = [SEMANTIC_LABELS[int(i)] for i in rng.permutation(12)]
            split_blocks.extend(
                tuple(permutation[start : start + 4])
                for start in range(0, len(permutation), 4)
            )
        if len(split_blocks) != SPLIT_SCENES[split]:
            raise AssertionError(f"incorrect scene block count for {split}")
        blocks[split] = split_blocks
    return blocks


def _negative_labels(
    blocks: Mapping[str, Sequence[Tuple[str, ...]]], seed: int
) -> Dict[Tuple[str, int], Tuple[str, str]]:
    """Assign every label once in train and once across val+test.

    Each selected absent label is queried with both ``after`` and ``before``.
    Consequently every label/relation pair has a negative training example,
    preventing lexical identity from revealing no-evidence status at test time.
    """

    def solve(
        scene_keys: Sequence[Tuple[str, int]], phase: str
    ) -> Dict[Tuple[str, int], Tuple[str, str]]:
        present = {
            key: set(blocks[key[0]][key[1]]) for key in scene_keys
        }
        assigned: Dict[Tuple[str, int], List[str]] = {
            key: [] for key in scene_keys
        }

        def search(remaining: Tuple[str, ...]) -> bool:
            if not remaining:
                return all(len(values) == 2 for values in assigned.values())
            ranked = []
            for label in remaining:
                candidates = [
                    key
                    for key in scene_keys
                    if label not in present[key] and len(assigned[key]) < 2
                ]
                ranked.append((len(candidates), stable_seed(seed, phase, label), label, candidates))
            _, _, label, candidates = min(ranked, key=lambda item: item[:3])
            candidates.sort(
                key=lambda key: (
                    len(assigned[key]),
                    stable_seed(seed, phase, label, key[0], key[1]),
                )
            )
            next_remaining = tuple(item for item in remaining if item != label)
            for key in candidates:
                assigned[key].append(label)
                if search(next_remaining):
                    return True
                assigned[key].pop()
            return False

        label_order = tuple(
            sorted(
                SEMANTIC_LABELS,
                key=lambda label: stable_seed(seed, phase, "order", label),
            )
        )
        if not search(label_order):
            raise AssertionError(f"could not solve absent-label matching for {phase}")
        return {
            key: (values[0], values[1])
            for key, values in assigned.items()
        }

    train_keys = [("train", index) for index in range(len(blocks["train"]))]
    evaluation_keys = [
        (split, index)
        for split in ("val", "test")
        for index in range(len(blocks[split]))
    ]
    assignments = solve(train_keys, "negative-train")
    assignments.update(solve(evaluation_keys, "negative-evaluation"))
    counts = Counter(label for pair in assignments.values() for label in pair)
    if counts != Counter({label: 2 for label in SEMANTIC_LABELS}):
        raise AssertionError(f"unbalanced absent-label matching: {counts}")
    return assignments


def _event_payload(
    event_id: str,
    kind: str,
    source: SourceClip,
    source_hash: str,
    crop_interval: Tuple[float, float],
    rendered_interval: Tuple[float, float],
    stem_path: Path,
    project_root: Path,
    staging_root: Path,
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "label": source.label,
        "event_kind": kind,
        "source_dataset": "AudioTime",
        "source_id": source.source_id,
        "source_path": relative_posix(source.audio_path, project_root),
        "source_sha256": source_hash,
        "source_interval_seconds": list(source.interval_seconds),
        "source_crop_interval_seconds": list(crop_interval),
        "onset_seconds": rendered_interval[0],
        "offset_seconds": rendered_interval[1],
        "stem_path": relative_posix(stem_path, staging_root),
    }


def _sum_stems(stems: Mapping[str, np.ndarray], ids: Iterable[str]) -> np.ndarray:
    result = np.zeros_like(next(iter(stems.values())))
    for event_id in ids:
        result += stems[event_id]
    return result


def _pick_template(
    templates: Sequence[Tuple[str, str]], scene_number: int, item_number: int
) -> Tuple[str, str]:
    return templates[(scene_number + item_number) % len(templates)]


def _pick_label_template(
    templates: Sequence[Tuple[str, str]], relation: str, label: str, seed: int
) -> Tuple[str, str]:
    """Use identical wording for positive/negative instances of one query."""

    index = stable_seed(seed, "relation-label-paraphrase", relation, label) % len(
        templates
    )
    return templates[index]


def _question_specs(
    semantic_events: Sequence[Mapping[str, Any]],
    absent_labels: Sequence[str],
    scene_number: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if len(semantic_events) != 4:
        raise ValueError("the v4 pilot requires four ordered semantic events")
    specs: List[Dict[str, Any]] = []
    item_number = 0
    for left, right in zip(semantic_events[:-1], semantic_events[1:]):
        template_id, template = _pick_label_template(
            AFTER_TEMPLATES, "after", left["label"], seed
        )
        specs.append(
            {
                "question_type": "temporal_after",
                "relation": "after",
                "question": template.format(anchor=left["label"]),
                "template_id": template_id,
                "answer": right["label"],
                "anchor": left["event_id"],
                "answer_id": right["event_id"],
                "query_labels": [left["label"]],
                "query_event_ids": [left["event_id"]],
                "no_evidence_reason": None,
            }
        )
        item_number += 1
    for left, right in zip(semantic_events[:-1], semantic_events[1:]):
        template_id, template = _pick_label_template(
            BEFORE_TEMPLATES, "before", right["label"], seed
        )
        specs.append(
            {
                "question_type": "temporal_before",
                "relation": "before",
                "question": template.format(anchor=right["label"]),
                "template_id": template_id,
                "answer": left["label"],
                "anchor": right["event_id"],
                "answer_id": left["event_id"],
                "query_labels": [right["label"]],
                "query_event_ids": [right["event_id"]],
                "no_evidence_reason": None,
            }
        )
        item_number += 1
    for pair_number, (earlier_index, later_index) in enumerate(
        itertools.combinations(range(4), 2)
    ):
        earlier = semantic_events[earlier_index]
        later = semantic_events[later_index]
        candidate_order = [earlier, later]
        # Six pair questions per scene permit exact 3/3 balancing.  Alternating
        # by pair and scene prevents a question-only model from exploiting
        # which named candidate appears first in the sentence.
        if (scene_number + pair_number) % 2:
            candidate_order.reverse()
        template_id, template = _pick_template(
            FIRST_TEMPLATES, scene_number, item_number
        )
        specs.append(
            {
                "question_type": "temporal_first",
                "relation": "first",
                "question": template.format(
                    left=candidate_order[0]["label"],
                    right=candidate_order[1]["label"],
                ),
                "template_id": template_id,
                "answer": earlier["label"],
                "anchor": later["event_id"],
                "answer_id": earlier["event_id"],
                "query_labels": [event["label"] for event in candidate_order],
                "query_event_ids": [event["event_id"] for event in candidate_order],
                "no_evidence_reason": None,
            }
        )
        item_number += 1
    negative_relations = (
        ("after", "no_evidence_after", AFTER_TEMPLATES),
        ("before", "no_evidence_before", BEFORE_TEMPLATES),
    )
    for absent_label in absent_labels:
        for negative_design in negative_relations:
            relation, question_type, templates = negative_design
            template_id, template = _pick_label_template(
                templates,
                relation,
                absent_label,
                seed,
            )
            specs.append(
                {
                    "question_type": question_type,
                    "relation": relation,
                    "question": template.format(anchor=absent_label),
                    "template_id": template_id,
                    "answer": NO_EVIDENCE_ANSWER,
                    "anchor": None,
                    "answer_id": None,
                    "query_labels": [absent_label],
                    "query_event_ids": [],
                    "no_evidence_reason": "absent_anchor",
                }
            )
    rng = np.random.default_rng(stable_seed(seed, "question-order", scene_number))
    return [specs[int(index)] for index in rng.permutation(len(specs))]


def _answer_options(spec: Mapping[str, Any], sample_id: str, seed: int) -> List[str]:
    answer = str(spec["answer"])
    rng = np.random.default_rng(stable_seed(seed, "qa-options", sample_id))
    required = [] if answer == NO_EVIDENCE_ANSWER else [answer]
    if spec["relation"] == "first":
        # Both alternatives named by the question must be visible.  Omitting
        # the incorrect candidate makes the correct choice recoverable without
        # listening to the waveform.
        required = list(dict.fromkeys(str(label) for label in spec["query_labels"]))
        if answer not in required or len(required) != 2:
            raise AssertionError("first questions require two candidates including the answer")
    excluded = set(required)
    distractors = [label for label in SEMANTIC_LABELS if label not in excluded]
    semantic_slots = 4 - len(required)
    chosen = [
        distractors[int(i)]
        for i in rng.permutation(len(distractors))[:semantic_slots]
    ]
    options = required + chosen + [NO_EVIDENCE_ANSWER]
    wrong_options = [option for option in options if option != answer]
    wrong_options = [
        wrong_options[int(i)] for i in rng.permutation(len(wrong_options))
    ]
    split, scene_index, question_index = sample_id.split("_")
    split_offset = {"train": 0, "val": 2, "test": 4}[split]
    answer_position = (
        int(scene_index) * QUESTIONS_PER_SCENE + int(question_index) + split_offset
    ) % 5
    options = list(wrong_options)
    options.insert(answer_position, answer)
    if len(options) != 5 or len(set(options)) != 5:
        raise AssertionError("QA options must contain five unique answers")
    return options


def _compose_scene(
    split: str,
    split_scene_index: int,
    global_scene_index: int,
    labels: Sequence[str],
    source_occurrences: Mapping[str, int],
    nuisance_occurrence: int,
    absent_labels: Sequence[str],
    grouped: Mapping[str, Sequence[SourceClip]],
    source_hashes: Mapping[str, str],
    metadata_hash: str,
    args: argparse.Namespace,
    staging_root: Path,
) -> List[Dict[str, Any]]:
    semantic_sources = [
        grouped[label][SPLIT_SOURCE_INDICES[split][source_occurrences[label]]]
        for label in labels
    ]
    nuisance_label = NUISANCE_LABELS[split_scene_index % len(NUISANCE_LABELS)]
    nuisance_source = grouped[nuisance_label][SPLIT_SOURCE_INDICES[split][nuisance_occurrence]]
    source_identity = [
        item
        for source in semantic_sources + [nuisance_source]
        for item in (source.source_id, source_hashes[source.source_id])
    ]
    scene_seed = stable_seed(
        args.seed,
        "qces-v4-scene",
        split,
        split_scene_index,
        metadata_hash,
        *source_identity,
    )
    rng = np.random.default_rng(scene_seed)
    ordered_sources = [semantic_sources[int(i)] for i in rng.permutation(4)]
    target_samples = int(round(SAMPLE_RATE * DURATION_SECONDS))
    durations = rng.uniform(0.65, 1.15, size=4)
    gaps = rng.integers(
        int(0.15 * SAMPLE_RATE), int(0.45 * SAMPLE_RATE), size=3
    )
    duration_samples = [
        int(round(float(duration) * SAMPLE_RATE)) for duration in durations
    ]
    semantic_span_samples = sum(duration_samples) + int(gaps.sum())
    margin_samples = int(round(SEMANTIC_EDGE_MARGIN_SECONDS * SAMPLE_RATE))
    available_shift = target_samples - semantic_span_samples - 2 * margin_samples
    if available_shift < 0:
        raise AssertionError("semantic layout exceeds scene duration")
    split_fractions = np.array(
        [
            (index + 0.5) / SPLIT_SCENES[split]
            for index in range(SPLIT_SCENES[split])
        ]
    )
    start_rng = np.random.default_rng(
        stable_seed(args.seed, "semantic-start-order", split)
    )
    split_fractions = split_fractions[
        start_rng.permutation(SPLIT_SCENES[split])
    ]
    onset = margin_samples + int(
        round(float(split_fractions[split_scene_index]) * available_shift)
    )
    onset_samples: List[int] = []
    for index, event_samples in enumerate(duration_samples):
        onset_samples.append(onset)
        if index < 3:
            onset += event_samples + int(gaps[index])
    if onset_samples[-1] + duration_samples[-1] > target_samples:
        raise AssertionError("semantic layout exceeds scene duration")
    nuisance_samples = int(round(NUISANCE_DURATION_SECONDS * SAMPLE_RATE))
    nuisance_margin = int(round(NUISANCE_EDGE_MARGIN_SECONDS * SAMPLE_RATE))
    nuisance_onset = int(
        rng.integers(
            nuisance_margin,
            target_samples - nuisance_samples - nuisance_margin + 1,
        )
    )

    crop_rngs = {
        source.source_id: np.random.default_rng(stable_seed(scene_seed, "crop", source.source_id))
        for source in ordered_sources + [nuisance_source]
    }
    crops: Dict[str, Tuple[float, float]] = {}
    intervals: Dict[str, Tuple[float, float]] = {}
    raw_stems: Dict[str, np.ndarray] = {}
    semantic_active = np.zeros(target_samples, dtype=bool)
    for index, (source, event_onset, event_samples) in enumerate(
        zip(ordered_sources, onset_samples, duration_samples)
    ):
        event_id = f"event_semantic_{index}"
        duration = event_samples / SAMPLE_RATE
        clip, crop = crop_source_event(
            source, duration, SAMPLE_RATE, FADE_MILLISECONDS, crop_rngs[source.source_id]
        )
        stem = np.zeros(target_samples, dtype=np.float32)
        interval = render_clip(stem, clip, event_onset, SAMPLE_RATE)
        crops[event_id] = crop
        intervals[event_id] = interval
        raw_stems[event_id] = stem
        semantic_active[event_onset : event_onset + clip.size] = True

    nuisance_id = "event_nuisance"
    nuisance_clip, nuisance_crop = crop_source_event(
        nuisance_source,
        NUISANCE_DURATION_SECONDS,
        SAMPLE_RATE,
        FADE_MILLISECONDS,
        crop_rngs[nuisance_source.source_id],
    )
    nuisance_stem = np.zeros(target_samples, dtype=np.float32)
    nuisance_interval = render_clip(
        nuisance_stem, nuisance_clip, nuisance_onset, SAMPLE_RATE
    )
    semantic_mix = _sum_stems(raw_stems, raw_stems)
    requested_snr = SNR_LEVELS_DB[global_scene_index % len(SNR_LEVELS_DB)]
    nuisance_gain = rms(semantic_mix[semantic_active]) / (
        max(rms(nuisance_stem[semantic_active]), EPSILON)
        * 10.0 ** (requested_snr / 20.0)
    )
    nuisance_stem *= nuisance_gain
    raw_stems[nuisance_id] = nuisance_stem
    crops[nuisance_id] = nuisance_crop
    intervals[nuisance_id] = nuisance_interval
    mixture = _sum_stems(raw_stems, raw_stems)
    peak = float(np.max(np.abs(mixture)))
    gain = min(1.0, HEADROOM / peak) if peak > 0 else 1.0
    stems = {event_id: stem * gain for event_id, stem in raw_stems.items()}
    mixture = _sum_stems(stems, stems).astype(np.float32)
    semantic_mix = _sum_stems(stems, set(stems) - {nuisance_id})
    realized_snr = 20.0 * math.log10(
        (rms(semantic_mix[semantic_active]) + EPSILON)
        / (rms(stems[nuisance_id][semantic_active]) + EPSILON)
    )

    # Keep the split visible while satisfying the schema's globally stable
    # ``scene_`` namespace.  Sample IDs retain the split prefix separately.
    scene_id = f"scene_{split}_{split_scene_index:06d}"
    mixture_path = staging_root / "audio" / "mixture" / split / f"{scene_id}.wav"
    write_wav(mixture_path, mixture, SAMPLE_RATE)
    events: List[Dict[str, Any]] = []
    for index, source in enumerate(ordered_sources):
        event_id = f"event_semantic_{index}"
        event_path = staging_root / "audio" / "events" / split / scene_id / f"{event_id}.wav"
        write_wav(event_path, stems[event_id], SAMPLE_RATE)
        events.append(
            _event_payload(
                event_id,
                "semantic",
                source,
                source_hashes[source.source_id],
                crops[event_id],
                intervals[event_id],
                event_path,
                args.project_root,
                staging_root,
            )
        )
    nuisance_path = staging_root / "audio" / "events" / split / scene_id / f"{nuisance_id}.wav"
    write_wav(nuisance_path, stems[nuisance_id], SAMPLE_RATE)
    events.append(
        _event_payload(
            nuisance_id,
            "nuisance",
            nuisance_source,
            source_hashes[nuisance_source.source_id],
            nuisance_crop,
            nuisance_interval,
            nuisance_path,
            args.project_root,
            staging_root,
        )
    )
    events.sort(key=lambda event: (event["onset_seconds"], event["event_id"]))
    event_by_id = {event["event_id"]: event for event in events}
    semantic_events = [event for event in events if event["event_kind"] == "semantic"]
    specs = _question_specs(
        semantic_events, absent_labels, global_scene_index, args.seed
    )

    records: List[Dict[str, Any]] = []
    for question_index, spec in enumerate(specs):
        sample_id = f"{split}_{split_scene_index:06d}_{question_index:02d}"
        answerable = spec["answer_id"] is not None
        anchor_ids = [spec["anchor"]] if answerable else []
        answer_ids = [spec["answer_id"]] if answerable else []
        evidence_ids = anchor_ids + answer_ids
        evidence = _sum_stems(stems, evidence_ids)
        residual = _sum_stems(stems, set(stems) - set(evidence_ids))
        anchor = _sum_stems(stems, anchor_ids)
        answer = _sum_stems(stems, answer_ids)
        output_paths = {
            name: staging_root / "audio" / name / split / f"{sample_id}.wav"
            for name in ("evidence", "residual", "anchor", "answer")
        }
        for name, waveform in (
            ("evidence", evidence),
            ("residual", residual),
            ("anchor", anchor),
            ("answer", answer),
        ):
            write_wav(output_paths[name], waveform, SAMPLE_RATE)
        options = _answer_options(spec, sample_id, args.seed)
        query_signature = "|".join(sorted(spec["query_labels"]))
        record = {
            "schema_version": SCHEMA_VERSION,
            "id": sample_id,
            "scene_id": scene_id,
            "question_family_id": f"{scene_id}:all-pairs",
            "counterfactual_group_id": f"{spec['relation']}:{query_signature}",
            "paraphrase_family_id": spec["template_id"],
            "question_index": question_index,
            "split": split,
            "sample_rate": SAMPLE_RATE,
            "num_channels": 1,
            "num_samples": target_samples,
            "duration_seconds": DURATION_SECONDS,
            "mixture_path": relative_posix(mixture_path, staging_root),
            "evidence_stem_path": relative_posix(output_paths["evidence"], staging_root),
            "residual_stem_path": relative_posix(output_paths["residual"], staging_root),
            "anchor_stem_path": relative_posix(output_paths["anchor"], staging_root),
            "answer_stem_path": relative_posix(output_paths["answer"], staging_root),
            "question": spec["question"],
            "answer": spec["answer"],
            "answer_options": options,
            "answer_option_index": options.index(spec["answer"]),
            "question_type": spec["question_type"],
            "relation": spec["relation"],
            "query_labels": spec["query_labels"],
            "query_event_ids": spec["query_event_ids"],
            "no_evidence": not answerable,
            "no_evidence_reason": spec["no_evidence_reason"],
            "absent_label": spec["query_labels"][0] if not answerable else None,
            "events": events,
            "anchor_event_ids": anchor_ids,
            "answer_event_ids": answer_ids,
            "evidence_event_ids": evidence_ids,
            "anchor_intervals": (
                [[event_by_id[anchor_ids[0]]["onset_seconds"], event_by_id[anchor_ids[0]]["offset_seconds"]]]
                if anchor_ids
                else []
            ),
            "answer_intervals": (
                [[event_by_id[answer_ids[0]]["onset_seconds"], event_by_id[answer_ids[0]]["offset_seconds"]]]
                if answer_ids
                else []
            ),
            "event_presence_labels": sorted(event["label"] for event in events),
            "source_group_ids": sorted(event["source_id"] for event in events),
            "nuisance_snr_db_requested": requested_snr,
            "nuisance_snr_db": realized_snr,
            "mixture_peak": float(np.max(np.abs(mixture))),
            "generation_seed": scene_seed,
        }
        parse_qces_v4_record(record)
        records.append(record)
    return records


def build_dataset(args: argparse.Namespace) -> None:
    args.project_root = args.project_root.resolve()
    audiotime_root = args.audiotime_root.resolve()
    output_root = args.output_root.resolve()
    staging_root = output_root.with_name(output_root.name + ".building")
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_root}; use --overwrite")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    staging_root.mkdir(parents=True)
    try:
        receipt, selected = _read_receipt(audiotime_root)
        grouped = _sources_by_label(selected)
        source_hashes = {
            source.source_id: sha256_file(source.audio_path)
            for source in selected.values()
        }
        metadata_path = audiotime_root / "timestamp_captions.json"
        metadata_hash = sha256_file(metadata_path)
        blocks = _label_blocks(args.seed)
        absent_assignments = _negative_labels(blocks, args.seed)
        records_by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        semantic_occurrences = {
            split: Counter({label: 0 for label in SEMANTIC_LABELS})
            for split in SPLIT_SCENES
        }
        nuisance_occurrences = {
            split: Counter({label: 0 for label in NUISANCE_LABELS})
            for split in SPLIT_SCENES
        }
        global_scene_index = 0
        for split in ("train", "val", "test"):
            for split_scene_index, labels in enumerate(blocks[split]):
                per_label_occurrence = {
                    label: semantic_occurrences[split][label] for label in labels
                }
                nuisance_label = NUISANCE_LABELS[
                    split_scene_index % len(NUISANCE_LABELS)
                ]
                nuisance_occurrence = nuisance_occurrences[split][nuisance_label]
                records = _compose_scene(
                    split,
                    split_scene_index,
                    global_scene_index,
                    labels,
                    per_label_occurrence,
                    nuisance_occurrence,
                    absent_assignments[(split, split_scene_index)],
                    grouped,
                    source_hashes,
                    metadata_hash,
                    args,
                    staging_root,
                )
                records_by_split[split].extend(records)
                semantic_occurrences[split].update(labels)
                nuisance_occurrences[split][nuisance_label] += 1
                print(
                    f"built scene_{split}_{split_scene_index:06d}: "
                    f"{len(records)} shortcut-resistant questions"
                )
                global_scene_index += 1

        for split, records in records_by_split.items():
            write_jsonl(staging_root / f"qces_{split}.jsonl", records)
        all_records = [
            record
            for split in ("train", "val", "test")
            for record in records_by_split[split]
        ]
        write_jsonl(staging_root / "qces_all.jsonl", all_records)
        source_files = {
            "builder": Path(__file__).resolve(),
            "schema": CODE_ROOT / "mixi_understanding" / "data" / "qces_v4_schema.py",
            "validator": CODE_ROOT / "mixi_understanding" / "scripts" / "validate_qces_v4_dataset.py",
            "downloader": CODE_ROOT / "mixi_understanding" / "scripts" / "download_qces_v4_sources.py",
        }
        config = {
            "schema_version": SCHEMA_VERSION,
            "builder_version": BUILDER_VERSION,
            "seed": args.seed,
            "sample_rate": SAMPLE_RATE,
            "num_channels": 1,
            "num_samples": int(SAMPLE_RATE * DURATION_SECONDS),
            "duration_seconds": DURATION_SECONDS,
            "audio_format": {"container": "WAV", "subtype": "PCM_16"},
            "counts": {
                "scenes": sum(SPLIT_SCENES.values()),
                "scenes_by_split": SPLIT_SCENES,
                "questions_per_scene": QUESTIONS_PER_SCENE,
                "records": len(all_records),
                "records_by_split": {
                    split: len(records) for split, records in records_by_split.items()
                },
            },
            "question_design": {
                "after_per_scene": 3,
                "before_per_scene": 3,
                "first_per_scene": 6,
                "matched_absent_labels_per_scene": 2,
                "matched_absent_questions_per_scene": 4,
                "no_evidence_after_per_scene": 2,
                "no_evidence_before_per_scene": 2,
                "question_order": "deterministically shuffled",
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
            },
            "composition": {
                "semantic_labels": list(SEMANTIC_LABELS),
                "nuisance_labels": list(NUISANCE_LABELS),
                "semantic_events_per_scene": 4,
                "nuisance_events_per_scene": 1,
                "semantic_duration_seconds": [0.65, 1.15],
                "semantic_gap_seconds": [0.15, 0.45],
                "semantic_start_design": "stratified_across_available_timeline_per_split",
                "semantic_edge_margin_seconds": SEMANTIC_EDGE_MARGIN_SECONDS,
                "nuisance_duration_seconds": NUISANCE_DURATION_SECONDS,
                "nuisance_edge_margin_seconds": NUISANCE_EDGE_MARGIN_SECONDS,
                "nuisance_snr_db_levels": list(SNR_LEVELS_DB),
                "headroom": HEADROOM,
            },
            "source": {
                "dataset": receipt["dataset"],
                "dataset_revision": receipt["dataset_revision"],
                "receipt_path": relative_posix(audiotime_root / RECEIPT_NAME, args.project_root),
                "receipt_sha256": sha256_file(audiotime_root / RECEIPT_NAME),
                "metadata_path": relative_posix(metadata_path, args.project_root),
                "metadata_sha256": metadata_hash,
            },
            "runtime_identity": {
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "soundfile_version": sf.__version__,
            },
            "build_identity": {
                role: {
                    "path": relative_posix(path, args.project_root),
                    "sha256": sha256_file(path),
                }
                for role, path in source_files.items()
            },
        }
        (staging_root / "dataset_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        from mixi_understanding.scripts.validate_qces_v4_dataset import validate_dataset

        report = validate_dataset(staging_root, write_report=True)
        replace_validated_output(staging_root, output_root)
        print(f"dataset ready: {output_root}")
        print(f"artifact fingerprint: {report['artifact_fingerprint_sha256']}")
    except Exception:
        print(f"build failed; staging retained at {staging_root}", file=sys.stderr)
        raise


def main() -> None:
    args = parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
