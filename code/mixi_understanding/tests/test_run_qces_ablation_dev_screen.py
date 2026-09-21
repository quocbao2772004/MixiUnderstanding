"""Tests for the fail-closed QCES contribution-ablation screen."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mixi_understanding.scripts.run_qces_ablation_dev_screen import (
    CEE_FORMAT,
    TRAINED_VARIANTS,
    build_dual_memory_smoke_command,
    build_variant_evaluate_command,
    build_variant_train_command,
    option_value,
    parse_args,
    promotion_state,
)
from mixi_understanding.scripts.train_qces import (
    parse_args as parse_train_args,
    validate_foundation_training_args,
    validate_refiner_training_args,
    validate_semantic_adapter_training_args,
)


class AblationCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parse_args([])

    def _train(self, variant: str) -> list[str]:
        return build_variant_train_command(self.args, variant)

    def test_all_variants_change_only_the_registered_intervention(self) -> None:
        commands = {variant: self._train(variant) for variant in TRAINED_VARIANTS}
        direct = commands["direct_question_clap"]
        self.assertIn("--freeze-semantic-adapter", direct)
        self.assertNotIn("--semantic-targets", direct)
        self.assertNotIn("--val-semantic-targets", direct)
        self.assertEqual(option_value(direct, "--semantic-weight"), "0")

        dual = commands["dual_role_factorization"]
        self.assertEqual(option_value(dual, "--semantic-separation-mode"), "dual_role")
        self.assertNotIn("--semantic-targets", dual)
        self.assertNotIn("--val-semantic-targets", dual)
        self.assertEqual(option_value(dual, "--role-semantic-weight"), "1.0")
        self.assertEqual(option_value(dual, "--same-semantic-weight"), "0.1")
        self.assertTrue(
            option_value(dual, "--role-semantic-targets").endswith(
                "semantic_dual_train.pt"
            )
        )

        no_weakest = commands["no_weakest_role_supervision"]
        self.assertEqual(option_value(no_weakest, "--weakest-role-weight"), "0")
        self.assertEqual(option_value(no_weakest, "--role-relative-weight"), "0.25")

        no_abstention = commands["no_explicit_no_evidence_supervision"]
        self.assertEqual(option_value(no_abstention, "--no-evidence-weight"), "0")
        self.assertEqual(
            option_value(no_abstention, "--surface-no-evidence-invariance-weight"),
            "0",
        )
        self.assertEqual(
            option_value(no_abstention, "--family-no-evidence-transition-weight"),
            "0",
        )
        self.assertEqual(
            option_value(no_abstention, "--surface-evidence-invariance-weight"),
            "0.1",
        )

        no_compactness = commands["no_compactness_penalty"]
        self.assertEqual(option_value(no_compactness, "--minimality-weight"), "0")
        for command in commands.values():
            self.assertEqual(option_value(command, "--batch-size"), "3")
            self.assertEqual(option_value(command, "--precision"), "amp_fp16")
            self.assertEqual(option_value(command, "--seed"), "2026")
            self.assertEqual(option_value(command, "--max-steps"), "512")
            self.assertEqual(
                option_value(command, "--foundation-semantic-mixing-mode"),
                "question_residual",
            )

    def test_generated_train_commands_pass_argument_contracts(self) -> None:
        for variant in TRAINED_VARIANTS:
            command = self._train(variant)
            parsed = parse_train_args(command[2:])
            with self.subTest(variant=variant):
                validate_refiner_training_args(parsed)
                validate_foundation_training_args(parsed)
                validate_semantic_adapter_training_args(parsed)

    def test_evaluation_paths_are_variant_bound_and_audio_rendering_is_off(
        self,
    ) -> None:
        for variant in TRAINED_VARIANTS:
            command = build_variant_evaluate_command(self.args, variant, ())
            with self.subTest(variant=variant):
                self.assertTrue(
                    option_value(command, "--checkpoint").endswith(
                        f"/{variant}_train/checkpoint.pt"
                    )
                )
                self.assertTrue(
                    option_value(command, "--output-dir").endswith(f"/{variant}_eval")
                )
                self.assertIn("--no-render-audio", command)

    def test_dual_memory_smoke_preserves_batch_and_removes_validation(self) -> None:
        command = build_dual_memory_smoke_command(self.args)
        self.assertEqual(option_value(command, "--max-steps"), "1")
        self.assertEqual(option_value(command, "--epochs"), "1")
        self.assertEqual(option_value(command, "--batch-size"), "3")
        self.assertEqual(option_value(command, "--precision"), "amp_fp16")
        self.assertEqual(
            option_value(command, "--semantic-separation-mode"), "dual_role"
        )
        self.assertNotIn("--val-manifest", command)
        self.assertNotIn("--val-foundation-feature-cache", command)
        self.assertNotIn("--val-role-semantic-targets", command)
        self.assertNotIn("--save-every-epoch", command)
        parsed = parse_train_args(command[2:])
        validate_refiner_training_args(parsed)
        validate_foundation_training_args(parsed)
        validate_semantic_adapter_training_args(parsed)


class PromotionGateTest(unittest.TestCase):
    def test_only_an_integrity_clean_positive_cee_receipt_authorizes_execution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            self.assertFalse(promotion_state(path)["passed"])
            payload = {
                "format": CEE_FORMAT,
                "integrity": {"all_passed": True},
                "decision": {"promote_cee_to_full_seeded_run": False},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(promotion_state(path)["passed"])
            payload["decision"]["promote_cee_to_full_seeded_run"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(promotion_state(path)["passed"])
            payload["integrity"]["all_passed"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(promotion_state(path)["passed"])


if __name__ == "__main__":
    unittest.main()
