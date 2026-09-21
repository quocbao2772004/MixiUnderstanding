#!/usr/bin/env python3
"""Build a frozen-checkpoint stress test with repeated ordinal anchors.

The historical locked test is left untouched.  This sidecar uses official-eval
sources only, covers every one of the 191 anchor labels, and emits one matched
answerable/NONE pair per exact ``(after, anchor_label, ordinal)`` key.  Labels
with the most source support receive ordinal 3; the remainder receive ordinal
2, so rare classes are not silently removed from evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from mixi_understanding.qces.clean_evidence_scenes import (
    CleanSource,
    SceneRecipe,
    _atomic_write_text,
    _scene_from_recipe,
    audit_rendered_reconstruction,
    audit_text_shortcuts,
    build_conditionally_balanced_qa,
    load_source_bank,
    partition_sources,
    render_scene_audio,
    sha256_file,
)


FORMAT = "qces_repeated_ordinal_stress_test_v1"
DEFAULT_SEED = 2081


def parse_args() -> argparse.Namespace:
    data = PROJECT_ROOT / "outputs/qces_full191_r1_data_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bank", type=Path, default=data / "source_bank_accepted.jsonl")
    parser.add_argument("--ontology", type=Path, default=data / "ontology_191.txt")
    parser.add_argument("--locked-test-scenes", type=Path, default=data / "multievent/scene_manifest_test.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/qces_full191_repeated_ordinal_stress_test_v1",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ordinal3-labels", type=int, default=95)
    parser.add_argument("--max-event-seconds", type=float, default=1.20)
    parser.add_argument("--verify-source-hash", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _digest(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{FORMAT}:{seed}:{namespace}:{value}".encode()).hexdigest()


def _ordered(values: Iterable[Any], *, seed: int, namespace: str) -> list[Any]:
    return sorted(values, key=lambda value: _digest(seed, namespace, str(value)))


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


def main() -> None:
    args = parse_args()
    labels = [line.strip() for line in args.ontology.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != 191 or len(labels) != len(set(labels)):
        raise ValueError(f"stress test requires the frozen 191-label ontology, got {len(labels)}")
    if not 1 <= args.ordinal3_labels < len(labels):
        raise ValueError("--ordinal3-labels must leave non-empty ordinal-2 and ordinal-3 groups")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output_dir}")

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
        raise RuntimeError("official-eval source overlaps train/dev hard identities")
    locked_identities = _manifest_identities(args.locked_test_scenes)

    by_label: dict[str, list[CleanSource]] = defaultdict(list)
    for source in test_sources:
        by_label[source.label].append(source)
    support = {label: len(by_label[label]) for label in labels}
    if min(support.values()) < 6:
        raise ValueError("every stress anchor label needs at least six official-eval sources")
    # Assign ordinal 3 to high-support classes.  Ties use a stable seeded key.
    ranked_high = sorted(
        labels,
        key=lambda label: (
            -support[label],
            _digest(args.seed, "ordinal-support-tie", label),
        ),
    )
    ordinal3 = set(ranked_high[: args.ordinal3_labels])
    ordinal_by_label = {label: 3 if label in ordinal3 else 2 for label in labels}
    queues = {}
    for label in labels:
        ordered = sorted(
            by_label[label],
            key=lambda source: (
                bool(_identity_set(source) & locked_identities),
                _digest(args.seed, f"source:{label}", source.source_id),
            ),
        )
        # _choose_source pops from the end; reverse so unseen locked-test
        # identities are consumed first and reused clips are fallback only.
        queues[label] = list(reversed(ordered))
    used_identities: set[tuple[str, str]] = set()
    anchor_sources: dict[str, list[CleanSource]] = {}
    # Rare labels reserve their anchors first.  This prevents abundant classes
    # from consuming a shared hard-identity group needed by a rare class.
    for label in sorted(labels, key=lambda value: (support[value], value)):
        required = 2 * ordinal_by_label[label]
        anchor_sources[label] = [
            _choose_source(queues[label], used_identities) for _ in range(required)
        ]

    filler_usage: Counter[str] = Counter()

    def filler(excluded_labels: set[str]) -> CleanSource:
        candidate_labels = sorted(
            (
                label
                for label in labels
                if label not in excluded_labels and queues[label]
            ),
            key=lambda label: (
                filler_usage[label],
                _digest(args.seed, f"filler:{sum(filler_usage.values())}", label),
            ),
        )
        for label in candidate_labels:
            try:
                source = _choose_source(queues[label], used_identities)
            except RuntimeError:
                continue
            filler_usage[label] += 1
            return source
        raise RuntimeError(f"no identity-safe filler remains outside {sorted(excluded_labels)}")

    recipes: list[SceneRecipe] = []
    for label_index, label in enumerate(labels):
        ordinal = ordinal_by_label[label]
        anchors = anchor_sources[label]
        if ordinal == 2:
            positive_fillers: list[CleanSource] = []
            for _ in range(2):
                positive_fillers.append(
                    filler({label, *(source.label for source in positive_fillers)})
                )
            negative_fillers: list[CleanSource] = []
            for _ in range(2):
                negative_fillers.append(
                    filler({label, *(source.label for source in negative_fillers)})
                )
            positive = (anchors[0], positive_fillers[0], anchors[1], positive_fillers[1])
            negative = (negative_fillers[0], anchors[2], negative_fillers[1], anchors[3])
        else:
            positive_fillers = []
            for _ in range(3):
                positive_fillers.append(
                    filler({label, *(source.label for source in positive_fillers)})
                )
            negative_fillers = []
            for _ in range(3):
                negative_fillers.append(
                    filler({label, *(source.label for source in negative_fillers)})
                )
            positive = (
                anchors[0], positive_fillers[0], anchors[1], positive_fillers[1],
                anchors[2], positive_fillers[2],
            )
            negative = (
                negative_fillers[0], anchors[3], negative_fillers[1], anchors[4],
                negative_fillers[2], anchors[5],
            )
        recipes.extend(
            (
                SceneRecipe(
                    scene_id=f"stress_repeat_test_{label_index:03d}_positive",
                    split="test",
                    sources=tuple(positive),
                    recipe_kind=f"repeated_ordinal{ordinal}_positive",
                    round_index=0,
                ),
                SceneRecipe(
                    scene_id=f"stress_repeat_test_{label_index:03d}_none",
                    split="test",
                    sources=tuple(negative),
                    recipe_kind=f"repeated_ordinal{ordinal}_none",
                    round_index=0,
                ),
            )
        )

    selected_sources = [source for recipe in recipes for source in recipe.sources]
    selected_ids = [source.source_id for source in selected_sources]
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("a source row was reused inside the stress test")
    all_hard = [identity for source in selected_sources for identity in source.hard_identities]
    if len(all_hard) != len(set(all_hard)):
        raise RuntimeError("a hard source identity was reused inside the stress test")
    if set(all_hard) & train_dev_identities:
        raise RuntimeError("stress test leaks train/dev source identity")

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        scenes = []
        for recipe in recipes:
            scene = _scene_from_recipe(
                recipe,
                labels,
                seed=args.seed,
                max_event_seconds=args.max_event_seconds,
            )
            scenes.append(
                render_scene_audio(
                    scene,
                    staging_root=staging,
                    verify_source_hash=args.verify_source_hash,
                )
            )
        all_qa, raw_qa_receipt = build_conditionally_balanced_qa(
            scenes, seed=args.seed, maximum_pairs_per_exact_key=1
        )
        qa = [
            row
            for row in all_qa
            if row.relation == "after"
            and row.anchor_ordinal == ordinal_by_label.get(row.anchor_label)
        ]
        counts: dict[tuple[str, int], Counter[bool]] = defaultdict(Counter)
        for row in qa:
            counts[(row.anchor_label, row.anchor_ordinal)][row.no_evidence] += 1
        expected_keys = {(label, ordinal_by_label[label]) for label in labels}
        if set(counts) != expected_keys or any(
            value[False] != 1 or value[True] != 1 for value in counts.values()
        ):
            raise RuntimeError("stress QA is not one positive/NONE pair per anchor key")
        reconstruction = audit_rendered_reconstruction(scenes, root=staging)
        if not reconstruction["passes"]:
            raise RuntimeError("stress mixtures fail evidence reconstruction audit")
        shortcut_audit = audit_text_shortcuts(qa)
        if not shortcut_audit["passes"]:
            raise RuntimeError("stress QA contains a text-only answerability shortcut")

        overlap_locked = set(all_hard) & locked_identities
        overlap_locked_by_field = Counter(field for field, _ in overlap_locked)
        selected_source_ids = {
            value for field, value in all_hard if field == "source_id"
        }
        locked_source_ids = {
            value for field, value in locked_identities if field == "source_id"
        }
        scene_manifest = staging / "scene_manifest_test.jsonl"
        detector_manifest = staging / "detector_scene_manifest_stress_test.jsonl"
        qa_manifest = staging / "qa_manifest_test.jsonl"
        ids_path = staging / "scene_ids_test.txt"
        ontology_path = staging / "ontology_191.txt"
        _atomic_write_text(scene_manifest, _jsonl(scenes))
        detector_rows = []
        for scene in scenes:
            row = dict(scene)
            row["mixture_path"] = str((output_dir / str(scene["mixture_path"])).resolve())
            detector_rows.append(row)
        _atomic_write_text(detector_manifest, _jsonl(detector_rows))
        _atomic_write_text(qa_manifest, _jsonl(row.to_dict() for row in qa))
        _atomic_write_text(ids_path, "".join(f"{scene['scene_id']}\n" for scene in scenes))
        _atomic_write_text(ontology_path, "".join(f"{label}\n" for label in labels))
        receipt = {
            "format": FORMAT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "policy": {
                "anchor_classes": len(labels),
                "ordinal2_classes": sum(value == 2 for value in ordinal_by_label.values()),
                "ordinal3_classes": sum(value == 3 for value in ordinal_by_label.values()),
                "relation": "after",
                "per_exact_key": {"answerable": 1, "no_evidence": 1},
                "answer_label_used_for_selection": False,
            },
            "scenes": len(scenes),
            "events": len(selected_sources),
            "qa": len(qa),
            "qa_answerable": sum(not row.no_evidence for row in qa),
            "qa_no_evidence": sum(row.no_evidence for row in qa),
            "ordinal_qa_distribution": dict(sorted(Counter(row.anchor_ordinal for row in qa).items())),
            "event_count_distribution": dict(sorted(Counter(len(scene["events"]) for scene in scenes).items())),
            "official_eval_source_pool": len(test_sources),
            "selected_unique_sources": len(selected_ids),
            "train_dev_hard_identity_overlap": 0,
            "locked_test_hard_identity_overlap": len(overlap_locked),
            "locked_test_hard_identity_overlap_by_field": dict(
                sorted(overlap_locked_by_field.items())
            ),
            "locked_test_reused_source_rows": len(
                selected_source_ids & locked_source_ids
            ),
            "locked_test_overlap_is_allowed_and_reported": True,
            "source_support": {
                "minimum": min(support.values()),
                "median": sorted(support.values())[len(support) // 2],
                "maximum": max(support.values()),
            },
            "partition_policy": partition_receipt["policy"],
            "raw_qa_builder": raw_qa_receipt,
            "text_shortcut_audit": shortcut_audit,
            "reconstruction_audit": reconstruction,
            "artifacts": {},
        }
        for path in (
            scene_manifest,
            detector_manifest,
            qa_manifest,
            ids_path,
            ontology_path,
        ):
            receipt["artifacts"][path.name] = sha256_file(path)
        _atomic_write_text(
            staging / "build_receipt.json",
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
        staging = None
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
