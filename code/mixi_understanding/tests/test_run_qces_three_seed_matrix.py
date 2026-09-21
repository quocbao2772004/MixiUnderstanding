"""Tests for the frozen full-validation three-seed QCES registry."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mixi_understanding.scripts.run_qces_ablation_dev_screen import (
    FORMAT as SCREEN_FORMAT,
    VARIANTS,
    option_value,
)
from mixi_understanding.scripts.run_qces_three_seed_matrix import (
    SEEDS,
    aggregate_seeds,
    build_seed_commands,
    parse_args,
    screen_prerequisite,
)
from mixi_understanding.scripts.train_qces import (
    parse_args as parse_train_args,
    validate_foundation_training_args,
    validate_refiner_training_args,
    validate_semantic_adapter_training_args,
)


class ThreeSeedRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parse_args([])

    def test_registry_has_seven_variants_and_never_mentions_test_manifests(
        self,
    ) -> None:
        for seed in SEEDS:
            commands = build_seed_commands(self.args, seed)
            self.assertEqual(set(commands), set(VARIANTS))
            serialized = json.dumps(commands, sort_keys=True)
            self.assertNotIn("qces_test_iid.jsonl", serialized)
            self.assertNotIn("qces_test_compositional_ood.jsonl", serialized)
            self.assertNotIn("qces_test_label_ood.jsonl", serialized)
            for variant, stages in commands.items():
                with self.subTest(seed=seed, variant=variant):
                    train = stages["train"]
                    self.assertEqual(option_value(train, "--seed"), str(seed))
                    self.assertEqual(option_value(train, "--epochs"), "1")
                    self.assertEqual(option_value(train, "--max-steps"), "0")
                    self.assertEqual(option_value(train, "--batch-size"), "3")
                    self.assertEqual(option_value(train, "--precision"), "amp_fp16")
                    self.assertTrue(
                        option_value(train, "--manifest").endswith("qces_train.jsonl")
                    )
                    self.assertTrue(
                        option_value(train, "--val-manifest").endswith("qces_val.jsonl")
                    )
                    self.assertTrue(
                        option_value(stages["evaluate"], "--manifest").endswith(
                            "qces_val.jsonl"
                        )
                    )

    def test_all_registered_training_commands_pass_fail_closed_contracts(self) -> None:
        commands = build_seed_commands(self.args, 2026)
        for variant, stages in commands.items():
            parsed = parse_train_args(stages["train"][2:])
            with self.subTest(variant=variant):
                validate_refiner_training_args(parsed)
                validate_foundation_training_args(parsed)
                validate_semantic_adapter_training_args(parsed)

    def test_screen_receipt_must_include_integrity_decision_and_memory_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screen.json"
            self.assertFalse(screen_prerequisite(path)["passed"])
            payload = {
                "format": SCREEN_FORMAT,
                "integrity": {"all_passed": True},
                "decision": {
                    "screen_complete_↑": True,
                    "three_seed_registry_may_be_frozen_↑": True,
                },
                "dual_role_memory_smoke": {"sha256": "a" * 64},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(screen_prerequisite(path)["passed"])
            del payload["dual_role_memory_smoke"]
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(screen_prerequisite(path)["passed"])

    def test_three_seed_aggregation_is_paired_and_arrowed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(["--results-root", str(root)])
            metric_names = {
                "evidence_sd_sdri_answerable_↑": "maximize",
                "weakest_role_sd_sdr_answerable_↑": "maximize",
                "answerable_temporal_iou_↑": "maximize",
                "mean_no_evidence_retained_ratio_↓": "minimize",
                "maximum_mixture_consistency_l1_↓": "minimize",
            }
            for offset, seed in enumerate(SEEDS):
                path = root / f"seed_{seed}/seed_comparison.json"
                path.parent.mkdir(parents=True)
                metrics = {}
                for display, direction in metric_names.items():
                    full = 3.0 + offset
                    ablated = 2.0 + offset if direction == "maximize" else 4.0 + offset
                    metrics[display] = {
                        "values": {
                            variant: (full if variant == "full_cee_on" else ablated)
                            for variant in VARIANTS
                        }
                    }
                path.write_text(
                    json.dumps(
                        {
                            "format": "qces_full_validation_three_seed_matrix_v1",
                            "seed": seed,
                            "integrity": {"all_passed": True},
                            "metrics": metrics,
                        }
                    ),
                    encoding="utf-8",
                )
            aggregate = aggregate_seeds(args)
            self.assertIsNotNone(aggregate)
            self.assertTrue(aggregate["integrity"]["all_passed"])
            self.assertFalse(aggregate["test_access_authorized"])
            for display in metric_names:
                effect = aggregate["aggregates"][display][
                    "paired_full_minus_variant_effects_↑"
                ]["direct_question_clap"]["mean_full_minus_variant_effect_oriented_↑"]
                self.assertEqual(effect, 1.0)


if __name__ == "__main__":
    unittest.main()
