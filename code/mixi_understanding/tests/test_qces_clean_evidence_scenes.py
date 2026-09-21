from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

import mixi_understanding.qces.clean_evidence_scenes as builder
from mixi_understanding.qces.clean_evidence_scenes import (
    CleanSource,
    SceneRecipe,
    audit_rendered_reconstruction,
    audit_text_shortcuts,
    build_clean_evidence_dataset,
    build_conditionally_balanced_qa,
    partition_sources,
    place_recipe_on_grid,
    schedule_scene_recipes,
)
from mixi_understanding.scripts.train_qces_qdor_dense import load_explicit_qa_manifest


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source(
    label: str,
    index: int,
    *,
    official_split: str = "train",
    assigned_split: str = "",
    path: str | None = None,
    source_sha: str | None = None,
    duration: float = 0.24,
) -> CleanSource:
    identity = f"{official_split}:{label}:{index}"
    return CleanSource(
        source_id=f"source:{identity}",
        label=label,
        official_split=official_split,
        video_id=f"video:{identity}",
        source_sha256=source_sha or _sha(identity),
        audio_path=path or f"/not/materialized/{identity}.wav",
        active_onset_seconds=0.0,
        active_offset_seconds=duration,
        cleanliness_passed=True,
        audibility_passed=True,
        cleanliness_tier="toy_verified",
        audibility_score=1.0,
        provenance={"toy": identity},
        assigned_split=assigned_split,
    )


def test_partition_is_hard_identity_disjoint_and_approximately_80_20_per_label() -> None:
    labels = ["Alpha", "Beta", "Gamma"]
    sources = [
        _source(label, index, official_split="train")
        for label in labels
        for index in range(10)
    ] + [
        _source(label, index, official_split="eval")
        for label in labels
        for index in range(3)
    ]
    partitioned, receipt = partition_sources(sources, labels, seed=19)

    assert receipt["objective"] == 0.0
    for label in labels:
        assert sum(source.label == label for source in partitioned["train"]) == 8
        assert sum(source.label == label for source in partitioned["dev"]) == 2
        assert sum(source.label == label for source in partitioned["test"]) == 3
    for field in ("source_id", "video_id", "source_sha256", "audio_path"):
        values = {
            split: {
                dict(source.hard_identities)[field]
                for source in partitioned[split]
            }
            for split in ("train", "dev", "test")
        }
        assert not values["train"] & values["dev"]
        assert not values["train"] & values["test"]
        assert not values["dev"] & values["test"]


def test_fixed_grid_and_float_wave_reconstruction() -> None:
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - only base env lacks renderer
        raise unittest.SkipTest("soundfile is required for the toy-wave test") from error

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rate = 16_000
        sources = []
        for index, (label, duration, frequency) in enumerate(
            (("Alpha", 0.21, 220.0), ("Beta", 0.33, 330.0), ("Gamma", 0.47, 440.0))
        ):
            samples = int(round(duration * rate))
            time = np.arange(samples, dtype=np.float32) / rate
            waveform = (0.05 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)
            path = root / f"source-{index}.wav"
            sf.write(path, waveform, rate, subtype="FLOAT")
            sources.append(
                _source(
                    label,
                    index,
                    assigned_split="train",
                    path=path.as_posix(),
                    source_sha=builder.sha256_file(path),
                    duration=duration,
                )
            )
        recipe = SceneRecipe(
            scene_id="scene_train_toy",
            split="train",
            sources=tuple(sources),
            recipe_kind="toy",
            round_index=0,
        )
        events, _ = place_recipe_on_grid(recipe, seed=5, max_event_seconds=1.2)
        assert all(
            abs(event["onset_seconds"] / 0.04 - round(event["onset_seconds"] / 0.04)) < 1e-8
            and abs(event["offset_seconds"] / 0.04 - round(event["offset_seconds"] / 0.04)) < 1e-8
            for event in events
        )
        assert all(
            events[index]["offset_seconds"] <= events[index + 1]["onset_seconds"]
            for index in range(len(events) - 1)
        )
        scene = {
            "scene_id": recipe.scene_id,
            "split": "train",
            "mixture_path": "audio/train/scene_train_toy.wav",
            "duration_seconds": 10.0,
            "sample_rate": rate,
            "events": events,
        }
        rendered = builder.render_scene_audio(
            scene,
            staging_root=root / "rendered",
            sample_rate=rate,
            verify_source_hash=True,
        )
        reconstruction = audit_rendered_reconstruction(
            [rendered], root=root / "rendered"
        )
        assert rendered["audio_num_frames"] == int(round(10.0 * rate))
        assert rendered["audio_num_channels"] == 1
        assert reconstruction["passes"]
        assert reconstruction["maximum_absolute_error"] <= 2e-6


def test_200_label_core_schedule_has_full_support_and_conditional_ba_half() -> None:
    labels = [f"Class_{index:03d}" for index in range(200)]
    assigned = [
        _source(label, source_index, assigned_split="dev")
        for label in labels
        for source_index in range(3)
    ]
    recipes, schedule = schedule_scene_recipes(
        assigned,
        labels,
        split="dev",
        seed=31,
        core_rounds=1,
        repeat_rounds=0,
        add_distractors=False,
    )
    scenes = [
        builder._scene_from_recipe(
            recipe,
            labels,
            seed=31,
            max_event_seconds=1.2,
        )
        for recipe in recipes
    ]
    assert all(scene["audio_num_frames"] == 160_000 for scene in scenes)
    assert all(scene["audio_num_channels"] == 1 for scene in scenes)
    qa, qa_receipt = build_conditionally_balanced_qa(scenes, seed=31)
    shortcut = audit_text_shortcuts(qa)
    label_counts = Counter(
        event["label"] for scene in scenes for event in scene["events"]
    )

    assert schedule["scenes"] == 200
    assert schedule["event_count_distribution"] == {3: 200}
    assert set(label_counts) == set(labels)
    assert set(label_counts.values()) == {3}
    assert qa_receipt["selected_answerable"] == qa_receipt["selected_no_evidence"]
    assert qa_receipt["exact_keys_retained"] == 400
    assert shortcut["passes"]
    assert {
        metric["balanced_accuracy"]
        for metric in shortcut["baselines"].values()
    } == {0.5}
    exact_counts = Counter(
        (row.relation, row.anchor_label, row.anchor_ordinal, row.no_evidence)
        for row in qa
    )
    for relation in ("before", "after"):
        for label in labels:
            assert exact_counts[(relation, label, 1, False)] == 1
            assert exact_counts[(relation, label, 1, True)] == 1


def test_repeat_schedule_emits_balanced_second_occurrence_after_queries() -> None:
    labels = ["Alpha", "Beta", "Gamma"]
    assigned = [
        _source(label, source_index, assigned_split="train")
        for label in labels
        for source_index in range(9)
    ]
    recipes, _ = schedule_scene_recipes(
        assigned,
        labels,
        split="train",
        seed=43,
        core_rounds=1,
        repeat_rounds=1,
        add_distractors=False,
    )
    scenes = [
        builder._scene_from_recipe(recipe, labels, seed=43, max_event_seconds=1.2)
        for recipe in recipes
    ]
    qa, _ = build_conditionally_balanced_qa(scenes, seed=43)
    counts = Counter(
        (row.relation, row.anchor_label, row.anchor_ordinal, row.no_evidence)
        for row in qa
    )
    for label in labels:
        assert counts[("after", label, 2, False)] == 1
        assert counts[("after", label, 2, True)] == 1


def test_manifest_only_build_commits_receipt_and_explicit_qa_atomically() -> None:
    labels = ["Alpha", "Beta", "Gamma"]
    sources = [
        _source(label, index, official_split="train")
        for label in labels
        for index in range(15)
    ] + [
        _source(label, index, official_split="eval")
        for label in labels
        for index in range(3)
    ]
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "dataset"
        receipt = build_clean_evidence_dataset(
            sources,
            labels,
            output_dir=output,
            seed=53,
            core_rounds=1,
            repeat_rounds=0,
            add_distractors=False,
            render_audio=False,
            require_ontology_size=0,
        )
        committed = json.loads((output / "build_receipt.json").read_text(encoding="utf-8"))
        assert receipt["passes"]
        assert committed["passes"]
        for split in ("train", "dev", "test"):
            assert (output / f"scene_manifest_{split}.jsonl").is_file()
            assert (output / f"qa_manifest_{split}.jsonl").is_file()
            assert (output / f"scene_ids_{split}.txt").is_file()
            baselines = committed["metadata_audit"]["qa_text_shortcuts"][split]["baselines"]
            assert max(value["balanced_accuracy"] for value in baselines.values()) <= 0.55


def test_qdor_loader_consumes_explicit_subset_and_rejects_bad_evidence_policy() -> None:
    positive_scene = {
        "scene_id": "scene_train_explicit_positive",
        "split": "train",
        "mixture_path": "audio/train/scene_train_explicit.wav",
        "events": [
            {
                "event_id": "e0",
                "label": "Alpha",
                "onset_seconds": 1.0,
                "offset_seconds": 1.4,
            },
            {
                "event_id": "e1",
                "label": "Beta",
                "onset_seconds": 2.0,
                "offset_seconds": 2.4,
            },
            {
                "event_id": "e2",
                "label": "Gamma",
                "onset_seconds": 3.0,
                "offset_seconds": 3.4,
            },
        ],
    }
    negative_scene = {
        "scene_id": "scene_train_explicit_negative",
        "split": "train",
        "mixture_path": "audio/train/scene_train_explicit_negative.wav",
        "events": [
            {
                "event_id": "n0",
                "label": "Beta",
                "onset_seconds": 1.0,
                "offset_seconds": 1.4,
            },
            {
                "event_id": "n1",
                "label": "Gamma",
                "onset_seconds": 2.0,
                "offset_seconds": 2.4,
            },
            {
                "event_id": "n2",
                "label": "Alpha",
                "onset_seconds": 3.0,
                "offset_seconds": 3.4,
            },
        ],
    }
    positive = builder._qa_candidate(
        positive_scene, relation="after", anchor_index=0, answer_index=1
    ).to_dict()
    negative = builder._qa_candidate(
        negative_scene, relation="after", anchor_index=2, answer_index=None
    ).to_dict()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = root / "qa.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in (positive, negative)),
            encoding="utf-8",
        )
        loaded = load_explicit_qa_manifest(
            manifest,
            allowed_scene_ids=[positive_scene["scene_id"], negative_scene["scene_id"]],
        )
        assert len(loaded) == 2
        assert sum(item.no_evidence for item in loaded) == 1

        bad = dict(positive)
        bad["gold_evidence_intervals"] = [positive["gold_anchor_interval"]]
        (root / "bad.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
        try:
            load_explicit_qa_manifest(
                root / "bad.jsonl", allowed_scene_ids=[positive_scene["scene_id"]]
            )
        except ValueError as error:
            assert "anchor+answer" in str(error)
        else:  # pragma: no cover - assertion branch
            raise AssertionError("invalid explicit evidence policy was accepted")


def load_tests(
    loader: unittest.TestLoader,
    tests: unittest.TestSuite,
    pattern: str | None,
) -> unittest.TestSuite:
    del loader, tests, pattern
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite
