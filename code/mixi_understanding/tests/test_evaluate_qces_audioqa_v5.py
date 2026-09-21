"""Checkpoint-free QCES-v5 contracts for the frozen AudioQA audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import soundfile as sf

from mixi_understanding.data.qces_v5_schema import (
    DERIVED_SCHEMA_VERSION,
    DERIVED_STORAGE_MODE,
    QCESV5Record,
)
from mixi_understanding.scripts import evaluate_qces_audioqa as audioqa


RATE = 8_000
SAMPLES = 800


def _record(
    sample_id: str,
    scene_id: str,
    family_id: str,
    *,
    question_index: int = 0,
) -> QCESV5Record:
    record = Mock(spec=QCESV5Record)
    record.schema_version = DERIVED_SCHEMA_VERSION
    record.storage_mode = DERIVED_STORAGE_MODE
    record.sample_id = sample_id
    record.scene_id = scene_id
    record.scene_family_id = family_id
    record.variant_id = "base"
    record.counterfactual_group_id = f"{family_id}:after"
    record.question_semantics_id = "after:frog:ordinal=1"
    record.question_index = question_index
    record.question_type = "temporal_after"
    record.relation = "after"
    record.split = "val"
    record.sample_rate = RATE
    record.num_samples = SAMPLES
    record.mixture_path = f"audio/{scene_id}/mixture.wav"
    record.evidence_stem_path = None
    record.residual_stem_path = None
    record.evidence_event_ids = ("anchor", "answer")
    record.no_evidence = False
    record.question = "What occurs after frog?"
    record.answer = "bell"
    record.answer_options = ("rain", "frog", "bell", "no_evidence", "wind")
    record.answer_option_index = 2
    record.anchor_event_ids = ("anchor",)
    record.answer_event_ids = ("answer",)
    record.primary_counterfactual_probe = True
    record.same_label_repeat = False
    record.semantic_overlap = False
    record.max_polyphony = 1
    record.hard_case_tags = ()
    record.surface_control_group_id = None
    record.mention_order_variant = "not_applicable"
    events = {
        "anchor": SimpleNamespace(
            event_id="anchor",
            label="frog",
            stem_path=f"audio/{scene_id}/anchor.wav",
        ),
        "answer": SimpleNamespace(
            event_id="answer",
            label="bell",
            stem_path=f"audio/{scene_id}/answer.wav",
        ),
    }
    record.events = tuple(events.values())
    record.event_by_id.side_effect = events.__getitem__
    return record


def _write(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, RATE, subtype="FLOAT")


def _identity(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": audioqa._sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _separator_report(
    root: Path,
    manifest: Path,
    checkpoint: Path,
    record: QCESV5Record,
    *,
    foundation_features: dict | None = None,
) -> Path:
    payload = {
        "format": "qces_question_swap_eval_v1",
        "schema_version": DERIVED_SCHEMA_VERSION,
        "checkpoint": str(checkpoint.resolve()),
        "manifest": str(manifest.resolve()),
        "records": 1,
        "summary_with_directions": {
            "evidence_sd_sdri_answerable_↑": 1.25,
            "maximum_mixture_consistency_l1_↓": 0.0,
        },
        "items": [
            {
                "id": record.sample_id,
                "scene_id": record.scene_id,
                "scene_family_id": record.scene_family_id,
            }
        ],
    }
    if foundation_features is not None:
        payload["foundation_features"] = foundation_features
    path = root / "evaluation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class DerivedInputAndControlTest(unittest.TestCase):
    def test_derived_oracle_evidence_and_residual_match_scene_event_recipe(
        self,
    ) -> None:
        record = _record(
            "val_000000_0_00",
            "scene_val_000000_base",
            "family_val_000000",
        )
        anchor = np.linspace(-0.1, 0.1, SAMPLES, dtype=np.float32)
        answer = np.linspace(0.05, -0.05, SAMPLES, dtype=np.float32)
        nuisance = np.full(SAMPLES, 0.025, dtype=np.float32)
        mixture = anchor + answer + nuisance
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(root / record.mixture_path, mixture)
            for event in record.events:
                _write(
                    root / event.stem_path,
                    anchor if event.label == "frog" else answer,
                )

            descriptors = audioqa.build_input_descriptors(
                [record],
                root,
                None,
                ("oracle_evidence", "oracle_residual"),
            )
            evidence_descriptor = descriptors[(record.sample_id, "oracle_evidence")]
            residual_descriptor = descriptors[(record.sample_id, "oracle_residual")]
            self.assertEqual(
                evidence_descriptor.derivation_recipe,
                "sum(evidence_event_stems)",
            )
            self.assertEqual(len(evidence_descriptor.component_sha256s), 2)
            evidence, evidence_rate = audioqa.materialize_input(evidence_descriptor)
            residual, residual_rate = audioqa.materialize_input(residual_descriptor)

        self.assertEqual(evidence_rate, RATE)
        self.assertEqual(residual_rate, RATE)
        np.testing.assert_allclose(evidence, anchor + answer, atol=1e-7)
        np.testing.assert_allclose(residual, nuisance, atol=1e-7)

    def test_v5_shuffle_crosses_scene_family_not_only_scene_id(self) -> None:
        base = _record("base", "scene_base", "family_shared")
        swap = _record("swap", "scene_swap", "family_shared")
        independent = _record("other", "scene_other", "family_other")
        mapping = audioqa.shuffled_record_map([base, swap, independent])
        self.assertEqual(mapping[base.sample_id].scene_family_id, "family_other")
        self.assertEqual(mapping[swap.sample_id].scene_family_id, "family_other")
        self.assertNotEqual(
            mapping[independent.sample_id].scene_family_id,
            independent.scene_family_id,
        )

    def test_shuffled_oracle_evidence_uses_cross_family_oracle_stems(self) -> None:
        first = _record("first", "scene_first", "family_first")
        second = _record("second", "scene_second", "family_second")
        first_stems = (
            np.full(SAMPLES, 0.01, dtype=np.float32),
            np.full(SAMPLES, 0.02, dtype=np.float32),
        )
        second_stems = (
            np.full(SAMPLES, -0.03, dtype=np.float32),
            np.full(SAMPLES, 0.07, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for record, stems in ((first, first_stems), (second, second_stems)):
                _write(root / record.mixture_path, stems[0] + stems[1])
                for event, waveform in zip(record.events, stems):
                    _write(root / event.stem_path, waveform)
            descriptors = audioqa.build_input_descriptors(
                [first, second], root, None, ("shuffled_oracle_evidence",)
            )
            descriptor = descriptors[(first.sample_id, "shuffled_oracle_evidence")]
            waveform, sample_rate = audioqa.materialize_input(descriptor)

        self.assertEqual(descriptor.record_id, first.sample_id)
        self.assertEqual(descriptor.source_record_id, second.sample_id)
        self.assertEqual(descriptor.derivation_recipe, "sum(evidence_event_stems)")
        self.assertEqual(sample_rate, RATE)
        np.testing.assert_allclose(
            waveform, second_stems[0] + second_stems[1], atol=1e-7
        )

    def test_gold_answer_and_index_are_not_passed_to_scorer(self) -> None:
        record = _record("sample", "scene", "family")

        class SpyScorer:
            def __init__(self) -> None:
                self.args = None

            def score(self, *args):
                self.args = args
                return audioqa.OptionScore(
                    (0.0,) * 5,
                    (0.2,) * 5,
                    0,
                    "spy",
                    (1,) * 5,
                )

        scorer = SpyScorer()
        waveform = np.zeros(SAMPLES, dtype=np.float32)
        audioqa.score_record_without_gold_inputs(scorer, record, waveform, RATE)
        self.assertEqual(len(scorer.args), 4)
        self.assertEqual(scorer.args[0], record.question)
        self.assertEqual(scorer.args[1], record.answer_options)
        self.assertIs(scorer.args[2], waveform)
        self.assertEqual(scorer.args[3], RATE)


class StrictProvenanceAndReportTest(unittest.TestCase):
    def test_isolated_option_order_control_is_gold_independent_and_paired(
        self,
    ) -> None:
        parsed = audioqa.parse_args(
            [
                "--manifest",
                "fixture.jsonl",
                "--output-dir",
                "output",
                "--model",
                "official/model",
                "--revision",
                "pinned",
                "--conditions",
                "mixture",
                "--option-order-control-conditions",
                "mixture",
            ]
        )
        self.assertEqual(
            parsed.conditions,
            ("mixture", "mixture" + audioqa.OPTION_ORDER_CONTROL_SUFFIX),
        )
        with self.assertRaises(SystemExit):
            audioqa.parse_args(
                [
                    "--manifest",
                    "fixture.jsonl",
                    "--output-dir",
                    "output",
                    "--model",
                    "official/model",
                    "--revision",
                    "pinned",
                    "--conditions",
                    "question_only",
                    "--option-order-control-conditions",
                    "mixture",
                ]
            )

        record = _record("sample", "scene", "family")
        control_condition = "mixture" + audioqa.OPTION_ORDER_CONTROL_SUFFIX
        permuted = audioqa.presented_answer_options(record, control_condition, 91)
        self.assertEqual(set(permuted), set(record.answer_options))
        self.assertNotEqual(permuted, record.answer_options)
        self.assertTrue(
            all(
                original != changed
                for original, changed in zip(record.answer_options, permuted)
            )
        )
        changed_answer = Mock(spec=QCESV5Record)
        changed_answer.answer_options = record.answer_options
        changed_answer.sample_id = record.sample_id
        changed_answer.answer = "a deliberately different hidden gold"
        self.assertEqual(
            audioqa.presented_answer_options(changed_answer, control_condition, 91),
            permuted,
        )
        gold_index = permuted.index(record.answer)
        scores = tuple(5.0 if index == gold_index else 0.0 for index in range(5))
        descriptor = audioqa.InputDescriptor(
            record.sample_id,
            control_condition,
            record.sample_id,
            "/tmp/fixture.wav",
            "a" * 64,
            RATE,
            SAMPLES,
        )
        scored_item = audioqa.make_item(
            record,
            descriptor,
            audioqa.OptionScore(
                scores,
                audioqa._softmax(scores),
                gold_index,
                "fixture",
                (1,) * 5,
            ),
            "f" * 64,
            permuted,
        )
        self.assertTrue(scored_item["option_order_control"])
        self.assertEqual(scored_item["gold_option_index"], gold_index)
        self.assertEqual(scored_item["predicted_answer"], record.answer)

        def item(
            record_id: str,
            condition: str,
            options: list[str],
            gold_index: int,
            predicted: str,
            correct: bool,
            log_score: float,
            probability: float,
        ) -> dict:
            return {
                "id": record_id,
                "scene_id": f"scene_{record_id}",
                "scene_family_id": f"family_{record_id}",
                "condition": condition,
                "question": "Which sound occurs?",
                "source_audio_sha256": "a" * 64,
                "answer_options": options,
                "gold_answer": "bell",
                "gold_option_index": gold_index,
                "predicted_answer": predicted,
                "correct": correct,
                "gold_log_score": log_score,
                "gold_option_probability": probability,
            }

        registered_options = ["bell", "rain", "wind", "frog", "no_evidence"]
        permuted_options = ["wind", "frog", "no_evidence", "bell", "rain"]
        items = [
            item("a", "mixture", registered_options, 0, "bell", True, -0.1, 0.7),
            item(
                "a",
                control_condition,
                permuted_options,
                3,
                "bell",
                True,
                -0.2,
                0.6,
            ),
            item("b", "mixture", registered_options, 0, "bell", True, -0.3, 0.6),
            item(
                "b",
                control_condition,
                permuted_options,
                3,
                "rain",
                False,
                -0.7,
                0.2,
            ),
        ]
        summaries, coverage = audioqa.option_order_control_summary(items)
        metrics = summaries["mixture"]
        self.assertEqual(metrics["option_order_gold_position_changed_rate_↑"], 1.0)
        self.assertEqual(metrics["option_order_semantic_prediction_invariance_↑"], 0.5)
        self.assertEqual(metrics["option_order_both_correct_rate_↑"], 0.5)
        self.assertEqual(metrics["option_order_accuracy_absolute_gap_↓"], 0.5)
        self.assertAlmostEqual(
            metrics["option_order_gold_log_score_absolute_gap_↓"], 0.25
        )
        self.assertEqual(coverage["mixture"]["complete_pairs"], 2)
        intervals, bootstrap_coverage = audioqa.bootstrap_option_order_metric_summary(
            items, 20, 7
        )
        self.assertIn("option_order_both_correct_rate_↑", intervals["mixture"])
        self.assertEqual(bootstrap_coverage["independent_scene_clusters"], 2)

    def test_surface_control_pairs_measure_combined_order_robustness(self) -> None:
        def item(
            record_id: str,
            group_id: str,
            variant: str,
            predicted_answer: str,
            correct: bool,
            gold_log_score: float,
            gold_probability: float,
        ) -> dict:
            return {
                "id": record_id,
                "scene_id": "scene",
                "scene_family_id": "family",
                "condition": "mixture",
                "no_evidence": False,
                "relation": "first",
                "question": "Which occurs first, bell or rain?",
                "answer_options": [
                    "bell",
                    "rain",
                    "wind",
                    "frog",
                    "no_evidence",
                ],
                "gold_answer": "bell",
                "surface_control_group_id": group_id,
                "mention_order_variant": variant,
                "correct": correct,
                "predicted_answer": predicted_answer,
                "gold_log_score": gold_log_score,
                "gold_option_probability": gold_probability,
            }

        items = [
            item("a_f", "pair_a", "forward", "bell", True, -0.10, 0.70),
            {
                **item("a_r", "pair_a", "reversed", "bell", True, -0.20, 0.60),
                "answer_options": [
                    "wind",
                    "no_evidence",
                    "rain",
                    "bell",
                    "frog",
                ],
            },
            item("b_f", "pair_b", "forward", "bell", True, -0.30, 0.55),
            {
                **item("b_r", "pair_b", "reversed", "rain", False, -0.70, 0.25),
                "answer_options": [
                    "rain",
                    "frog",
                    "bell",
                    "no_evidence",
                    "wind",
                ],
            },
        ]
        metrics = audioqa.condition_metrics(items)
        self.assertEqual(
            metrics["first_question_surface_pair_prediction_invariance_↑"], 0.5
        )
        self.assertEqual(
            metrics["first_question_surface_pair_both_correct_rate_↑"], 0.5
        )
        self.assertEqual(
            metrics["first_question_surface_pair_at_least_one_correct_rate_↑"],
            1.0,
        )
        self.assertAlmostEqual(
            metrics["first_question_surface_pair_gold_log_score_absolute_gap_↓"],
            0.25,
        )
        self.assertAlmostEqual(
            metrics["first_question_surface_pair_gold_probability_absolute_gap_↓"],
            0.20,
        )
        _, coverage = audioqa._surface_control_pair_summary(items[:-1])
        self.assertEqual(
            coverage,
            {"eligible_rows": 3, "complete_pairs": 1, "incomplete_groups": 1},
        )

    def test_strict_v5_binding_hashes_separator_checkpoint_and_foundation_cache(
        self,
    ) -> None:
        record = _record(
            "val_000000_0_00",
            "scene_val_000000_base",
            "family_val_000000",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("fixture manifest bytes\n", encoding="utf-8")
            checkpoint = root / "separator.pt"
            checkpoint.write_bytes(b"separator checkpoint")
            cache_dir = root / "cache"
            cache_dir.mkdir()
            receipt = cache_dir / "cache_receipt.json"
            question = cache_dir / "question_features.pt"
            scene = cache_dir / "scene_audio_features.pt"
            receipt.write_text("{}", encoding="utf-8")
            question.write_bytes(b"question features")
            scene.write_bytes(b"scene features")
            manifest_identity = _identity(manifest)
            cache_identity = {
                "format": "fixture_cache_v1",
                "directory": str(cache_dir),
                "contains_oracle_or_label_inputs": False,
                "manifest_binding": {
                    "cache_declared": manifest_identity,
                    "run_expected": manifest_identity,
                },
                "audiosep_checkpoint_binding": {
                    "cache_declared": {"sha256": "a" * 64, "size_bytes": 7},
                    "run_expected": {"sha256": "a" * 64, "size_bytes": 7},
                },
                "receipt": _identity(receipt),
                "question_feature_artifact": _identity(question),
                "scene_feature_artifact": _identity(scene),
            }
            predictions = root / "predictions"
            _separator_report(
                predictions,
                manifest,
                checkpoint,
                record,
                foundation_features={"cache_identity": cache_identity},
            )

            provenance = audioqa.strict_v5_prediction_provenance(
                predictions, manifest.resolve(), [record]
            )
            self.assertEqual(
                provenance["separator_checkpoint"]["sha256"],
                audioqa._sha256_file(checkpoint),
            )
            self.assertTrue(provenance["foundation_feature_cache"]["present"])
            self.assertEqual(
                provenance["validation_status"], "strict_v5_binding_passed"
            )
            self.assertIn(
                "evidence_sd_sdri_answerable_↑",
                provenance["model_free_proxy_metrics"],
            )

            question.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "does not match current bytes"):
                audioqa.strict_v5_prediction_provenance(
                    predictions, manifest.resolve(), [record]
                )

    def test_external_caption_planner_report_can_bind_mode_wavs(self) -> None:
        record = _record(
            "val_000000_0_00",
            "scene_val_000000_base",
            "family_val_000000",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("fixture manifest bytes\n", encoding="utf-8")
            checkpoint = root / "audiosep.bin"
            checkpoint.write_bytes(b"frozen audiosep")
            expected_checkpoint_sha = audioqa._sha256_file(checkpoint)
            report = root / "planner" / "evaluation_report.json"
            report.parent.mkdir()
            report.write_text(
                json.dumps(
                    {
                        "format": "qces_v5_caption_planner_audiosep_eval_v1",
                        "schema_versions": [DERIVED_SCHEMA_VERSION],
                        "manifest": str(manifest.resolve()),
                        "manifest_sha256": audioqa._sha256_file(manifest),
                        "audiosep_checkpoint": _identity(checkpoint),
                        "summary": {
                            "evidence_sd_sdri_answerable_mean_db_↑": 1.5,
                            "record_count": 1,
                        },
                        "items": [
                            {
                                "id": record.sample_id,
                                "scene_id": record.scene_id,
                                "scene_family_id": record.scene_family_id,
                                "mode": "caption_planner",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            wav_root = report.parent / "caption_planner"
            wav_root.mkdir()

            provenance = audioqa.strict_v5_prediction_provenance(
                wav_root,
                manifest.resolve(),
                [record],
                report_override=report,
                prediction_mode="caption_planner",
            )

        self.assertEqual(
            provenance["evaluation_report_format"],
            "qces_v5_caption_planner_audiosep_eval_v1",
        )
        self.assertEqual(provenance["prediction_mode"], "caption_planner")
        self.assertEqual(
            provenance["separator_checkpoint"]["sha256"],
            expected_checkpoint_sha,
        )

    def test_multimode_temporal_report_requires_and_selects_mode(self) -> None:
        record = _record(
            "val_000000_0_00",
            "scene_val_000000_base",
            "family_val_000000",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text("fixture manifest bytes\n", encoding="utf-8")
            report = root / "temporal" / "evaluation_report.json"
            report.parent.mkdir()
            modes = ("random", "energy")
            report.write_text(
                json.dumps(
                    {
                        "format": "qces_v5_temporal_only_baselines_v1",
                        "schema_versions": [DERIVED_SCHEMA_VERSION],
                        "manifest": str(manifest.resolve()),
                        "manifest_sha256": audioqa._sha256_file(manifest),
                        "summaries": {
                            mode: {"evidence_sd_sdri_answerable_mean_db_↑": -1.0}
                            for mode in modes
                        },
                        "items": [
                            {
                                "id": record.sample_id,
                                "scene_id": record.scene_id,
                                "scene_family_id": record.scene_family_id,
                                "mode": mode,
                            }
                            for mode in modes
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "select --predictions-mode"):
                audioqa.strict_v5_prediction_provenance(
                    report.parent, manifest.resolve(), [record]
                )
            provenance = audioqa.strict_v5_prediction_provenance(
                report.parent / "energy",
                manifest.resolve(),
                [record],
                report_override=report,
                prediction_mode="energy",
            )

        self.assertIsNone(provenance["separator_checkpoint"])
        self.assertEqual(provenance["prediction_mode"], "energy")
        self.assertIn(
            "evidence_sd_sdri_answerable_mean_db_↑",
            provenance["model_free_proxy_metrics"],
        )

    def test_report_separates_actual_qa_metrics_from_model_free_proxies(self) -> None:
        record = _record("sample", "scene", "family")
        item = {
            "id": "sample",
            "scene_id": "scene",
            "scene_family_id": "family",
            "condition": "mixture",
            "no_evidence": False,
            "relation": "after",
            "variant_id": "base",
            "same_role_label": False,
            "same_label_repeat": False,
            "semantic_overlap": False,
            "primary_counterfactual_probe": True,
            "correct": True,
            "predicted_answer": "bell",
            "gold_log_score": -0.1,
            "gold_option_probability": 0.8,
        }
        report = audioqa.build_report(
            records=[record],
            conditions=("mixture",),
            items=[item],
            run_fingerprint="f" * 64,
            metadata={
                "predictions": {
                    "model_free_proxy_metrics": {"evidence_sd_sdri_answerable_↑": 1.0}
                }
            },
            bootstrap_samples=5,
            seed=7,
        )
        families = report["metric_families"]
        self.assertFalse(
            families["actual_frozen_qa_model_metrics"][
                "gold_answer_or_gold_option_index_used_as_model_input"
            ]
        )
        self.assertEqual(
            families["model_free_acoustic_proxy_metrics"]["metrics"],
            {"evidence_sd_sdri_answerable_↑": 1.0},
        )
        self.assertIn("v5_paired_subgroup_metrics", report)
        self.assertEqual(
            report["paired_independent_cluster_bootstrap_coverage"]["cluster_unit"],
            "scene_family_id_if_available_else_scene_id",
        )


if __name__ == "__main__":
    unittest.main()
