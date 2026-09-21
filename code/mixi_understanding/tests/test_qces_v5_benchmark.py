"""Focused source-free/unit-scale tests for the QCES v5 protocol."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from mixi_understanding.data.qces_v5_schema import parse_qces_v5_record
from mixi_understanding.scripts.build_qa_removal_dataset import (
    SourceClip,
    crop_source_event,
    maximum_variance_crop_offset,
)
from mixi_understanding.scripts.build_qces_v5_dataset import (
    FamilyLabels,
    PROFILE_FAMILY_COUNTS,
    TEMPLATES,
    _has_semantic_overlap,
    _load_semantic_crop_bank,
    _maximum_pcm_render_peak,
    _max_polyphony,
    _question_specs,
    _rebalance_negative_semantics,
    _semantic_balance_summary,
    _split_semantic_vocabulary,
    _load_receipt,
    make_label_schedules,
)
from mixi_understanding.scripts.plan_qces_v5_sources import (
    INTERNAL_SCALE_HELDOUT_LABELS,
    INTERNAL_SCALE_NUISANCE_LABELS,
    INTERNAL_SCALE_SEEN_LABELS,
    PROJECT_ROOT,
    _default_license_record,
    _license_record,
    make_plan,
)
from mixi_understanding.scripts.validate_qces_v5_dataset import (
    _validate_config,
    _enforce_exact_semantics_shortcut_gate,
    _exact_semantics_train_to_val_shortcut,
    _validate_family_counterfactuals,
    _validate_first_surface_controls,
)


class QCESV5SourcePlanTest(unittest.TestCase):
    audiotime_root = PROJECT_ROOT / "AudioTime-recovered" / "train5000_timestamp"

    def test_internal_scale_source_plan_is_feasible_without_downloading(self) -> None:
        plan = make_plan(
            audiotime_root=self.audiotime_root,
            profile="internal_scale",
            seed=314159,
            dataset="enyoukai/audiotime-timestamps",
            revision="50e1a871c03eee734af88ee2b61e13a372a38a8d",
            config="default",
            hub_split="train",
            license_record=_default_license_record(),
        )
        self.assertEqual(plan["source_count"], 388)
        self.assertEqual(
            plan["source_counts_by_partition"],
            {
                "train": 136,
                "val": 68,
                "test_iid": 68,
                "test_compositional_ood": 68,
                "test_label_ood": 48,
            },
        )
        source_ids = [row["source_id"] for row in plan["sources"]]
        self.assertEqual(len(source_ids), len(set(source_ids)))
        self.assertFalse(plan["license_record"]["redistribution_allowed"])
        self.assertEqual(plan["license_record"]["status"], "unverified")

    def test_paper_crop_recovers_sparse_event_but_legacy_crop_still_fails(self) -> None:
        rate = 16_000
        waveform = np.zeros(4 * rate, dtype=np.float32)
        event_start = int(2.5 * rate)
        event_stop = int(3.0 * rate)
        phase = np.arange(event_stop - event_start, dtype=np.float32) / rate
        waveform[event_start:event_stop] = 0.2 * np.sin(2.0 * np.pi * 440.0 * phase)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sparse.wav"
            sf.write(path, waveform, rate, subtype="PCM_16")
            source = SourceClip(
                source_id="sparse",
                label="fixture",
                interval_seconds=(0.0, 4.0),
                caption="fixture",
                audio_path=path,
            )
            with self.assertRaisesRegex(ValueError, "effectively silent"):
                crop_source_event(
                    source,
                    1.0,
                    rate,
                    5.0,
                    np.random.default_rng(1),
                )
            clip, crop = crop_source_event(
                source,
                1.0,
                rate,
                5.0,
                np.random.default_rng(1),
                recover_silent_random_crop=True,
            )
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.square(clip, dtype=np.float64)))),
            0.08,
            places=5,
        )
        self.assertLess(crop[0], 3.0)
        self.assertGreater(crop[1], 2.5)

    def test_label_free_salience_crop_chooses_maximum_variance_window(self) -> None:
        rate = 100
        waveform = np.zeros(400, dtype=np.float32)
        waveform[250:300] = np.tile(np.array([-0.4, 0.4], dtype=np.float32), 25)
        offset = maximum_variance_crop_offset(waveform, 0, 400, 100)
        self.assertLessEqual(offset, 250)
        self.assertGreaterEqual(offset + 100, 300)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event.wav"
            sf.write(path, waveform, rate, subtype="FLOAT")
            source = SourceClip(
                source_id="event",
                label="fixture",
                interval_seconds=(0.0, 4.0),
                caption="fixture",
                audio_path=path,
            )
            clip, crop = crop_source_event(
                source,
                1.0,
                rate,
                0.0,
                np.random.default_rng(999),
                prefer_maximum_variance_crop=True,
            )
        self.assertLessEqual(crop[0], 2.5)
        self.assertGreaterEqual(crop[1], 3.0)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(np.square(clip, dtype=np.float64)))),
            0.08,
            places=5,
        )

    def test_preferred_semantic_center_controls_crop_and_clamps_safely(self) -> None:
        rate = 100
        waveform = np.linspace(-0.5, 0.5, 400, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event.wav"
            sf.write(path, waveform, rate, subtype="FLOAT")
            source = SourceClip(
                source_id="event",
                label="fixture",
                interval_seconds=(0.5, 3.5),
                caption="fixture",
                audio_path=path,
            )
            _, centered_crop = crop_source_event(
                source,
                1.0,
                rate,
                0.0,
                np.random.default_rng(999),
                preferred_center_seconds=2.0,
            )
            _, left_clamped_crop = crop_source_event(
                source,
                1.0,
                rate,
                0.0,
                np.random.default_rng(1),
                preferred_center_seconds=0.5,
            )
            _, right_clamped_crop = crop_source_event(
                source,
                1.0,
                rate,
                0.0,
                np.random.default_rng(2),
                preferred_center_seconds=3.5,
            )
        self.assertEqual(centered_crop, (1.5, 2.5))
        self.assertEqual(left_clamped_crop, (0.5, 1.5))
        self.assertEqual(right_clamped_crop, (2.5, 3.5))

    def test_semantic_crop_bank_is_fail_closed_and_fingerprint_bound(self) -> None:
        payload = {
            "format": "qces_v5_semantic_crop_bank_v2",
            "curator_boundary": "fixture curator is not an evaluator",
            "selection": {"window_seconds": 1.25},
            "manifest": {"sha256": "0" * 64},
            "entries": [
                {
                    "source_id": "source-1",
                    "label": "Meow",
                    "source_sha256": "1" * 64,
                    "source_interval_seconds": [0.0, 4.0],
                    "selected": {
                        "center_seconds": 2.0,
                        "label_probability ↑": 0.8,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            entries, identity = _load_semantic_crop_bank(path)
            assert entries is not None and identity is not None
            self.assertEqual(entries[("source-1", "Meow")]["selected"]["center_seconds"], 2.0)
            self.assertEqual(identity["source_count"], 1)
            self.assertEqual(len(identity["sha256"]), 64)

            payload["entries"].append(copy.deepcopy(payload["entries"][0]))
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _load_semantic_crop_bank(path)

    def test_family_headroom_bounds_stems_even_when_mixture_cancels(self) -> None:
        positive = np.array([1.4, 0.0, 0.2], dtype=np.float32)
        cancelling = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        variants = {
            "base": {"a": positive, "b": cancelling},
            "order_swap": {"a": positive.copy(), "b": cancelling.copy()},
        }
        # The mixture peak is only 0.4, but the independently stored first stem
        # needs 1.4 of headroom to remain additive after PCM encoding.
        mixture_peak = float(
            np.max(np.abs(positive + cancelling))
        )
        self.assertAlmostEqual(mixture_peak, 0.4, places=6)
        maximum = _maximum_pcm_render_peak(variants)
        self.assertAlmostEqual(maximum, 1.4, places=6)
        gain = 0.95 / maximum
        self.assertLessEqual(float(np.max(np.abs(positive * gain))), 0.950001)
        self.assertLessEqual(
            float(np.max(np.abs((positive + cancelling) * gain))), 0.950001
        )

    def test_unverified_license_cannot_claim_redistribution(self) -> None:
        # Exercise the validation rule without writing a persistent file.
        import json
        import tempfile

        payload = _default_license_record()
        payload["redistribution_allowed"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "license.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "redistribution requires verified"):
                _license_record(path)


class QCESV5SplitDesignTest(unittest.TestCase):
    def test_composition_and_label_ood_are_disjoint_by_construction(self) -> None:
        schedules = make_label_schedules(
            seen_labels=INTERNAL_SCALE_SEEN_LABELS,
            heldout_labels=INTERNAL_SCALE_HELDOUT_LABELS,
            family_counts={
                "train": 40,
                "val": 5,
                "test_iid": 8,
                "test_compositional_ood": 10,
                "test_label_ood": 5,
            },
            seed=314159,
        )
        train_pairs = {family.pair_signature for family in schedules["train"]}
        train_triplets = {family.triplet_signature for family in schedules["train"]}
        self.assertTrue(
            {family.pair_signature for family in schedules["val"]}.issubset(
                train_pairs
            )
        )
        self.assertTrue(
            {family.pair_signature for family in schedules["test_iid"]}.issubset(
                train_pairs
            )
        )
        self.assertFalse(
            train_pairs
            & {
                family.pair_signature
                for family in schedules["test_compositional_ood"]
            }
        )
        self.assertFalse(
            train_triplets
            & {
                family.triplet_signature
                for family in schedules["test_compositional_ood"]
            }
        )
        heldout = set(INTERNAL_SCALE_HELDOUT_LABELS)
        self.assertTrue(
            all(
                set(family.distinct_labels).issubset(heldout)
                for family in schedules["test_label_ood"]
            )
        )

    def test_template_family_ids_are_partition_disjoint(self) -> None:
        ids = {
            partition: {
                template_id
                for relation_templates in relation_map.values()
                for template_id, _ in relation_templates
            }
            for partition, relation_map in TEMPLATES.items()
        }
        self.assertFalse(ids["train"] & ids["validation"])
        self.assertFalse(ids["train"] & ids["evaluation"])
        self.assertFalse(ids["validation"] & ids["evaluation"])

    def test_paper_scale_exceeds_predeclared_scene_floor(self) -> None:
        scenes = {
            split: families * 3
            for split, families in PROFILE_FAMILY_COUNTS["paper"].items()
        }
        self.assertEqual(sum(scenes.values()), 1407)
        self.assertEqual(sum(scenes.values()) * 16, 22512)
        self.assertGreaterEqual(scenes["train"], 800)
        self.assertGreaterEqual(scenes["val"], 100)
        self.assertGreaterEqual(scenes["test_iid"], 200)
        self.assertGreaterEqual(scenes["test_compositional_ood"], 200)
        self.assertGreaterEqual(scenes["test_label_ood"], 100)

    def test_internal_scale_profile_is_a_valid_nonpaper_stress_profile(self) -> None:
        family_counts = PROFILE_FAMILY_COUNTS["internal_scale"]
        scenes = {split: count * 3 for split, count in family_counts.items()}
        config = {
            "schema_version": "qces_v5",
            "profile": "internal_scale",
            "audio_format": {"container": "WAV", "subtype": "PCM_16"},
            "sample_rate": 32_000,
            "num_samples": 320_000,
            "num_channels": 1,
            "counts": {
                "scene_families": sum(family_counts.values()),
                "scene_families_by_split": family_counts,
                "scenes": sum(scenes.values()),
                "scenes_by_split": scenes,
                "questions_per_scene": 16,
                "records": sum(scenes.values()) * 16,
                "records_by_split": {
                    split: count * 16 for split, count in scenes.items()
                },
            },
            "question_design": {
                "answer_options": 5,
                "primary_counterfactual_variants": [
                    "base",
                    "order_swap",
                    "anchor_drop",
                ],
                "primary_surface_and_options_identical_across_variants": True,
                "ordinal_same_label_anchor": True,
                "adjacency_definition": "unique_semantic_onset_order",
                "templates_disjoint_train_validation_evaluation": True,
                "matched_no_evidence_every_relation_template_partition": True,
                "paired_first_mention_order_controls": True,
                "paired_first_options_independently_permuted": True,
                "split_local_positive_semantics_for_every_negative": True,
                "negative_semantic_assignment_policy": (
                    "deterministic_most_constrained_coverage_normalized_load_v1"
                ),
                "primary_negative_semantics_immutable": True,
                "first_negative_surface_pair_atomic": True,
                "no_evidence_ratio_bounds": [0.2, 0.4],
            },
            "semantic_balance": {
                "strict_profile_gate": True,
                "paper_eligibility": "eligible",
                "train_to_val_exact_semantics_shortcut_gate": {
                    "unseen_semantics_backoff": "train_relation_prior",
                    "maximum_auroc_down": 0.60,
                    "maximum_best_balanced_accuracy_down": 0.60,
                },
                "summary_by_split": {split: {} for split in PROFILE_FAMILY_COUNTS["internal_scale"]},
            },
            "composition": {
                "same_label_repeat_required_every_scene": True,
                "semantic_overlap_required_every_scene": True,
                "primary_pair_and_triplet_disjoint_train_vs_compositional_ood": True,
                "heldout_labels_exclusive_to_label_ood": True,
                "semantic_question_and_option_labels_split_local": True,
            },
        }
        _, _, validated, profile = _validate_config(config)
        self.assertEqual(profile, "internal_scale")
        self.assertEqual(validated, family_counts)

    def test_paper_builder_rejects_nonfinalized_audited_receipt(self) -> None:
        payload = {
            "format": "qces_v5_audited_source_ledger_v1",
            "profile": "paper",
            "acquisition_complete": True,
            "release_ready": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "release gate"):
                _load_receipt(receipt, root, "paper")


class QCESV5SemanticBalanceTest(unittest.TestCase):
    vocabulary = ("bell", "cat", "dog", "frog", "owl", "rain")

    @staticmethod
    def _events(*labels: str) -> list[dict]:
        return [
            {"event_kind": "semantic", "label": label}
            for label in labels
        ]

    @classmethod
    def _fixture(cls) -> list[dict]:
        common = {
            "split": "train",
            "template_partition": "train",
            "primary_counterfactual_probe": False,
        }
        records = [
            {
                **common,
                "id": "after-positive-primary",
                "scene_id": "scene-positive-primary",
                "relation": "after",
                "no_evidence": False,
                "question_semantics_id": "after:frog:ordinal=3",
                "events": cls._events("frog", "frog", "frog", "cat"),
            },
            {
                **common,
                "id": "after-positive",
                "scene_id": "scene-after-positive",
                "relation": "after",
                "no_evidence": False,
                "question_semantics_id": "after:owl:ordinal=1",
                "events": cls._events("owl", "cat"),
            },
            {
                **common,
                "id": "before-positive",
                "scene_id": "scene-before-positive",
                "relation": "before",
                "no_evidence": False,
                "question_semantics_id": "before:dog:ordinal=1",
                "events": cls._events("dog", "cat"),
            },
            {
                **common,
                "id": "first-positive-forward",
                "scene_id": "scene-first-positive",
                "relation": "first",
                "no_evidence": False,
                "question_semantics_id": "first:bell|rain",
                "events": cls._events("bell", "rain", "cat"),
            },
            {
                **common,
                "id": "first-positive-reversed",
                "scene_id": "scene-first-positive",
                "relation": "first",
                "no_evidence": False,
                "question_semantics_id": "first:bell|rain",
                "events": cls._events("bell", "rain", "cat"),
            },
            {
                **common,
                "id": "after-primary-negative",
                "scene_id": "scene-primary-negative",
                "relation": "after",
                "no_evidence": True,
                "primary_counterfactual_probe": True,
                "question_semantics_id": "after:frog:ordinal=3",
                "events": cls._events("frog", "frog", "cat"),
                "question": "What begins after the third frog?",
                "query_label": "frog",
                "query_instance_ordinal": 3,
                "absent_labels": ["frog"],
            },
            {
                **common,
                "id": "after-negative",
                "scene_id": "scene-after-negative",
                "relation": "after",
                "no_evidence": True,
                "question_semantics_id": "after:bell:ordinal=1",
                "events": cls._events("cat"),
                "paraphrase_family_id": "tr_after_next",
                "surface_control_group_id": None,
                "question": "legacy",
                "query_label": "bell",
                "query_instance_ordinal": 1,
                "absent_labels": ["bell"],
                "answer_options": [*cls.vocabulary[:4], "no_evidence"],
                "answer_option_index": 4,
            },
            {
                **common,
                "id": "before-negative",
                "scene_id": "scene-before-negative",
                "relation": "before",
                "no_evidence": True,
                "question_semantics_id": "before:bell:ordinal=1",
                "events": cls._events("cat"),
                "paraphrase_family_id": "tr_before_previous",
                "surface_control_group_id": None,
                "question": "legacy",
                "query_label": "bell",
                "query_instance_ordinal": 1,
                "absent_labels": ["bell"],
                "answer_options": [*cls.vocabulary[:4], "no_evidence"],
                "answer_option_index": 4,
            },
        ]
        for mention_order in ("forward", "reversed"):
            candidates = (
                ["dog", "owl"]
                if mention_order == "forward"
                else ["owl", "dog"]
            )
            records.append(
                {
                    **common,
                    "id": f"first-negative-{mention_order}",
                    "scene_id": "scene-first-negative",
                    "relation": "first",
                    "no_evidence": True,
                    "question_semantics_id": "first:dog|owl",
                    "events": cls._events("cat"),
                    "paraphrase_family_id": "tr_first_earlier",
                    "surface_control_group_id": "first-negative-control",
                    "mention_order_variant": mention_order,
                    "question": "legacy",
                    "query_candidate_labels": candidates,
                    "absent_labels": candidates,
                    "answer_options": [*cls.vocabulary[:4], "no_evidence"],
                    "answer_option_index": 4,
                }
            )
        return records

    def test_split_vocabulary_excludes_the_other_semantic_inventory(self) -> None:
        seen = ("seen-a", "seen-b")
        heldout = ("held-a", "held-b")
        for split in ("train", "val", "test_iid", "test_compositional_ood"):
            self.assertEqual(
                _split_semantic_vocabulary(
                    split, seen_labels=seen, heldout_labels=heldout
                ),
                list(seen),
            )
        self.assertEqual(
            _split_semantic_vocabulary(
                "test_label_ood", seen_labels=seen, heldout_labels=heldout
            ),
            list(heldout),
        )

    def test_balancer_is_deterministic_grouped_and_preserves_primary(self) -> None:
        first = copy.deepcopy(self._fixture())
        second = copy.deepcopy(self._fixture())
        primary_before = copy.deepcopy(
            next(row for row in first if row["id"] == "after-primary-negative")
        )
        first_summary = _rebalance_negative_semantics(
            first, semantic_vocabulary=self.vocabulary, seed=17
        )
        second_summary = _rebalance_negative_semantics(
            second, semantic_vocabulary=self.vocabulary, seed=17
        )
        self.assertEqual(first, second)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(
            next(row for row in first if row["id"] == "after-primary-negative"),
            primary_before,
        )
        positive_ids = {
            (row["relation"], row["question_semantics_id"])
            for row in first
            if not row["no_evidence"]
        }
        self.assertTrue(
            all(
                (row["relation"], row["question_semantics_id"])
                in positive_ids
                for row in first
                if row["no_evidence"]
            )
        )
        forward = next(row for row in first if row["id"] == "first-negative-forward")
        reversed_record = next(
            row for row in first if row["id"] == "first-negative-reversed"
        )
        self.assertEqual(
            reversed_record["query_candidate_labels"],
            list(reversed(forward["query_candidate_labels"])),
        )
        self.assertEqual(
            set(forward["query_candidate_labels"]), {"bell", "rain"}
        )
        self.assertTrue(
            set(forward["query_candidate_labels"]).issubset(
                forward["answer_options"]
            )
        )
        self.assertNotEqual(
            forward["answer_option_index"],
            reversed_record["answer_option_index"],
        )
        self.assertEqual(
            _semantic_balance_summary(first)["negative_only_unique_semantics_down"],
            0,
        )

    def test_balancer_rejects_a_negative_without_feasible_positive_semantic(self) -> None:
        records = self._fixture()
        negative = next(row for row in records if row["id"] == "after-negative")
        negative["events"] = self._events("owl", "frog", "frog", "frog")
        with self.assertRaisesRegex(ValueError, "no split-local positive semantic"):
            _rebalance_negative_semantics(
                records, semantic_vocabulary=self.vocabulary, seed=17
            )


class QCESV5SemanticShortcutGateTest(unittest.TestCase):
    @staticmethod
    def _records(*, leaking: bool) -> list[SimpleNamespace]:
        records = []
        for relation in ("after", "before", "first"):
            for semantic_index in range(2):
                semantics_id = f"{relation}:semantic-{semantic_index}"
                for status in (False, True):
                    train_status = bool(semantic_index) if leaking else status
                    records.append(
                        SimpleNamespace(
                            split="train",
                            relation=relation,
                            question_semantics_id=semantics_id,
                            no_evidence=train_status,
                        )
                    )
                    records.append(
                        SimpleNamespace(
                            split="val",
                            relation=relation,
                            question_semantics_id=semantics_id,
                            no_evidence=(bool(semantic_index) if leaking else status),
                        )
                    )
        return records

    def test_balanced_exact_semantics_lookup_is_at_chance(self) -> None:
        report = _exact_semantics_train_to_val_shortcut(
            self._records(leaking=False)
        )
        self.assertAlmostEqual(report["auroc_down"], 0.5)
        self.assertAlmostEqual(report["best_balanced_accuracy_down"], 0.5)
        for relation in ("after", "before", "first"):
            self.assertAlmostEqual(
                report["by_relation"][relation]["auroc_down"], 0.5
            )
        _enforce_exact_semantics_shortcut_gate(
            report,
            maximum_auroc=0.60,
            maximum_best_balanced_accuracy=0.60,
        )

    def test_leaking_exact_semantics_lookup_is_rejected(self) -> None:
        report = _exact_semantics_train_to_val_shortcut(
            self._records(leaking=True)
        )
        self.assertEqual(report["auroc_down"], 1.0)
        with self.assertRaisesRegex(AssertionError, "AUROC"):
            _enforce_exact_semantics_shortcut_gate(
                report,
                maximum_auroc=0.60,
                maximum_best_balanced_accuracy=0.60,
            )


class QCESV5CounterfactualSchemaTest(unittest.TestCase):
    @staticmethod
    def _event(
        event_id: str,
        label: str,
        onset: float,
        occurrence: int,
        split: str = "train",
    ) -> dict:
        return {
            "event_id": event_id,
            "label": label,
            "event_kind": "semantic",
            "source_dataset": "fixture",
            "dataset_version": "fixture-v1",
            "source_id": f"source_{event_id}",
            "creator_id": f"creator_{event_id}",
            "uploader_id": f"uploader_{event_id}",
            "attribution": f"Fixture creator for {event_id}",
            "source_license_spdx": "CC0-1.0",
            "source_license_url": "https://example.invalid/license",
            "source_partition": split,
            "source_path": f"sources/{event_id}.wav",
            "source_sha256": "0" * 64,
            "license_record_id": "fixture-license",
            "source_interval_seconds": [0.0, 8.0],
            "source_crop_interval_seconds": [0.0, 1.0],
            "onset_seconds": onset,
            "offset_seconds": onset + 1.0,
            "gain_db": 0.0,
            "occurrence_index": occurrence,
            "stem_path": f"audio/events/{event_id}.wav",
        }

    @classmethod
    def _events(cls, variant: str) -> list[dict]:
        events = [
            cls._event("sem_00", "frog", 0.10, 1),
            cls._event("sem_01", "frog", 0.50, 2),
            cls._event("sem_02", "frog", 1.20, 3),
            cls._event("sem_03", "bell", 2.00, 1),
            cls._event("sem_04", "rain", 3.00, 1),
            cls._event("sem_05", "horn", 4.00, 1),
            cls._event("sem_06", "dog", 5.00, 1),
        ]
        if variant == "order_swap":
            by_id = {event["event_id"]: event for event in events}
            by_id["sem_03"]["onset_seconds"] = 3.0
            by_id["sem_03"]["offset_seconds"] = 4.0
            by_id["sem_04"]["onset_seconds"] = 2.0
            by_id["sem_04"]["offset_seconds"] = 3.0
        if variant == "anchor_drop":
            events = [event for event in events if event["event_id"] != "sem_02"]
        return sorted(events, key=lambda event: (event["onset_seconds"], event["event_id"]))

    @classmethod
    def _record(cls, variant: str, question_index: int = 0) -> dict:
        events = cls._events(variant)
        by_id = {event["event_id"]: event for event in events}
        if variant == "base":
            answer_id, answer = "sem_03", "bell"
        elif variant == "order_swap":
            answer_id, answer = "sem_04", "rain"
        else:
            answer_id, answer = None, "no_evidence"
        role_ids = [] if answer_id is None else ["sem_02"]
        answer_ids = [] if answer_id is None else [answer_id]
        evidence_ids = role_ids + answer_ids
        split = "train"
        scene_id = f"scene_{split}_000000_{variant}"
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
        return {
            "schema_version": "qces_v5",
            "id": f"{split}_000000_{variant}_{question_index:02d}",
            "scene_id": scene_id,
            "scene_family_id": "family_train_000000",
            "variant_id": variant,
            "counterfactual_intervention": intervention,
            "question_semantics_id": "after:frog:ordinal=3",
            "counterfactual_group_id": "family_train_000000:primary",
            "paraphrase_family_id": "tr_after_next",
            "template_partition": "train",
            "question_index": question_index,
            "split": split,
            "evaluation_axis": "development",
            "sample_rate": 8000,
            "num_channels": 1,
            "num_samples": 64000,
            "duration_seconds": 8.0,
            "mixture_path": f"audio/mixture/{scene_id}.wav",
            "evidence_stem_path": f"audio/evidence/{scene_id}_{question_index}.wav",
            "residual_stem_path": f"audio/residual/{scene_id}_{question_index}.wav",
            "anchor_stem_path": f"audio/anchor/{scene_id}_{question_index}.wav",
            "answer_stem_path": f"audio/answer/{scene_id}_{question_index}.wav",
            "question": "What begins next after the third occurrence of frog?",
            "answer": answer,
            "answer_options": ["bell", "rain", "no_evidence", "horn", "dog"],
            "answer_option_index": ["bell", "rain", "no_evidence", "horn", "dog"].index(answer),
            "question_type": "temporal_after",
            "relation": "after",
            "no_evidence": answer_id is None,
            "no_evidence_reason": "absent_anchor" if answer_id is None else None,
            "absent_labels": ["frog"] if answer_id is None else [],
            "query_label": "frog",
            "query_instance_ordinal": 3,
            "query_candidate_labels": [],
            "query_event_ids": role_ids,
            "surface_control_group_id": None,
            "mention_order_variant": "not_applicable",
            "events": events,
            "anchor_event_ids": role_ids,
            "answer_event_ids": answer_ids,
            "evidence_event_ids": evidence_ids,
            "anchor_intervals": (
                [[by_id["sem_02"]["onset_seconds"], by_id["sem_02"]["offset_seconds"]]]
                if role_ids
                else []
            ),
            "answer_intervals": (
                [[by_id[answer_id]["onset_seconds"], by_id[answer_id]["offset_seconds"]]]
                if answer_id
                else []
            ),
            "source_group_ids": sorted({event["source_id"] for event in events}),
            "primary_counterfactual_probe": question_index == 0,
            "composition_pair_signature": "frog=>bell+rain",
            "composition_triplet_signature": "frog|bell|rain",
            "same_label_repeat": True,
            "semantic_overlap": _has_semantic_overlap(events),
            "max_polyphony": _max_polyphony(events),
            "hard_case_tags": ["same_label_instances", "semantic_overlap"],
            "render_recipe_id": "fixture-v1",
            "mixture_peak": 0.5,
            "family_gain": 1.0,
            "generation_seed": 1,
        }

    def test_overlapping_repeated_instance_relations_are_valid(self) -> None:
        record = parse_qces_v5_record(self._record("base"))
        self.assertEqual(record.query_instance_ordinal, 3)
        self.assertTrue(record.same_label_repeat)
        self.assertTrue(record.semantic_overlap)
        self.assertEqual(record.answer, "bell")

    def test_wrong_ordinal_instance_is_rejected(self) -> None:
        payload = self._record("base")
        payload["query_instance_ordinal"] = 2
        with self.assertRaisesRegex(ValueError, "query selector"):
            parse_qces_v5_record(payload)

    def test_every_first_semantics_has_reversed_surface_pair_and_negatives(self) -> None:
        specs = _question_specs(
            events=self._events("base"),
            labels=FamilyLabels(
                repeat_label="frog",
                base_answer_label="bell",
                swapped_answer_label="rain",
                extra_labels=("horn", "dog"),
            ),
            semantic_vocabulary=(
                "frog",
                "bell",
                "rain",
                "horn",
                "dog",
                "cat",
                "cow",
            ),
            template_partition="train",
            family_id="family_train_000000",
            variant_id="base",
            seed=314159,
        )
        first = [spec for spec in specs if spec["relation"] == "first"]
        self.assertEqual(len(first), 8)
        self.assertEqual(sum(spec["no_evidence"] for spec in first), 2)
        groups = defaultdict(list)
        for spec in first:
            groups[spec["surface_control_group_id"]].append(spec)
        self.assertEqual(len(groups), 4)
        for pair in groups.values():
            self.assertEqual({item["mention_order_variant"] for item in pair}, {"forward", "reversed"})
            forward = next(item for item in pair if item["mention_order_variant"] == "forward")
            reverse = next(item for item in pair if item["mention_order_variant"] == "reversed")
            self.assertEqual(reverse["candidate_labels"], list(reversed(forward["candidate_labels"])))

    @classmethod
    def _first_record(cls, mention_order: str) -> dict:
        payload = cls._record("base", 7 if mention_order == "forward" else 8)
        forward = mention_order == "forward"
        payload.update(
            {
                "question_semantics_id": "first:bell|rain",
                "counterfactual_group_id": "scene_train_000000_base:first-control",
                "paraphrase_family_id": "tr_first_earlier",
                "question": (
                    "Which begins first, bell or rain?"
                    if forward
                    else "Which begins first, rain or bell?"
                ),
                "answer": "bell",
                "answer_options": (
                    ["bell", "rain", "no_evidence", "horn", "dog"]
                    if forward
                    else ["rain", "no_evidence", "horn", "bell", "dog"]
                ),
                "answer_option_index": 0 if forward else 3,
                "question_type": "temporal_first",
                "relation": "first",
                "no_evidence": False,
                "no_evidence_reason": None,
                "absent_labels": [],
                "query_label": None,
                "query_instance_ordinal": None,
                "query_candidate_labels": ["bell", "rain"] if forward else ["rain", "bell"],
                "query_event_ids": ["sem_03", "sem_04"] if forward else ["sem_04", "sem_03"],
                "surface_control_group_id": "scene_train_000000_base:first-control",
                "mention_order_variant": mention_order,
                "anchor_event_ids": ["sem_04"],
                "answer_event_ids": ["sem_03"],
                "evidence_event_ids": ["sem_04", "sem_03"],
                "anchor_intervals": [[3.0, 4.0]],
                "answer_intervals": [[2.0, 3.0]],
                "primary_counterfactual_probe": False,
            }
        )
        return payload

    def test_first_surface_pair_gate_checks_order_and_option_letters(self) -> None:
        pair = [
            parse_qces_v5_record(self._first_record("forward")),
            parse_qces_v5_record(self._first_record("reversed")),
        ]
        self.assertEqual(_validate_first_surface_controls(pair), 1)

    def test_first_no_evidence_requires_two_explicit_absent_candidates(self) -> None:
        payload = self._first_record("forward")
        payload.update(
            {
                "question_semantics_id": "first:cat|cow",
                "question": "Which begins first, cat or cow?",
                "answer": "no_evidence",
                "answer_options": ["cat", "cow", "no_evidence", "horn", "dog"],
                "answer_option_index": 2,
                "no_evidence": True,
                "no_evidence_reason": "absent_candidates",
                "absent_labels": ["cat", "cow"],
                "query_candidate_labels": ["cat", "cow"],
                "query_event_ids": [],
                "anchor_event_ids": [],
                "answer_event_ids": [],
                "evidence_event_ids": [],
                "anchor_intervals": [],
                "answer_intervals": [],
            }
        )
        record = parse_qces_v5_record(payload)
        self.assertTrue(record.no_evidence)
        self.assertEqual(record.absent_labels, ("cat", "cow"))

    def test_three_way_counterfactual_family_is_validated(self) -> None:
        records = []
        scene_records = defaultdict(list)
        for variant in ("base", "order_swap", "anchor_drop"):
            for index in range(16):
                parsed = parse_qces_v5_record(self._record(variant, index))
                records.append(parsed)
                scene_records[parsed.scene_id].append(parsed)
        summary = _validate_family_counterfactuals(records, scene_records)
        self.assertEqual(summary["scene_families_validated"], 1)
        self.assertEqual(summary["primary_counterfactual_groups_validated"], 1)

    def test_primary_surface_change_is_rejected(self) -> None:
        records = []
        scene_records = defaultdict(list)
        for variant in ("base", "order_swap", "anchor_drop"):
            for index in range(16):
                payload = self._record(variant, index)
                if variant == "order_swap" and index == 0:
                    payload["question"] += " altered"
                parsed = parse_qces_v5_record(payload)
                records.append(parsed)
                scene_records[parsed.scene_id].append(parsed)
        with self.assertRaisesRegex(AssertionError, "surface/options changed"):
            _validate_family_counterfactuals(records, scene_records)


if __name__ == "__main__":
    unittest.main()
