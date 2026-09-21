"""Tests for the CEE AMP batch-3 memory-smoke runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mixi_understanding.scripts.run_qces_cee_memory_smoke import (
    CEE_WEIGHT_FLAGS,
    build_command,
    require_idle_gpu,
    validate_summary,
)
from mixi_understanding.scripts.train_qces import parse_args as parse_train_args


def option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class CEEMemorySmokeRunnerTest(unittest.TestCase):
    def test_command_is_amp_batch3_train_only_and_activates_all_cee_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(
                output_dir=root / "out",
                manifest=root / "qces_train.jsonl",
                foundation_cache=root / "foundation_train",
                semantic_targets=root / "semantic_union_train.pt",
                audiosep_root=root / "audiosep",
                audiosep_config=root / "audiosep.yaml",
                audiosep_checkpoint=root / "audiosep.pt",
                max_steps=10,
                seed=10,
            )
            command = build_command(args)
        self.assertEqual(option(command, "--batch-size"), "3")
        self.assertEqual(option(command, "--precision"), "amp_fp16")
        self.assertEqual(option(command, "--max-steps"), "10")
        self.assertEqual(
            option(command, "--foundation-semantic-mixing-mode"),
            "question_residual",
        )
        self.assertEqual(option(command, "--temporal-role-mode"), "independent_sigmoid")
        self.assertNotIn("--val-manifest", command)
        self.assertNotIn("--overwrite", command)
        for flag in CEE_WEIGHT_FLAGS:
            self.assertGreater(float(option(command, flag)), 0.0)
        parsed = parse_train_args(command[2:])
        self.assertEqual(parsed.batch_size, 3)
        self.assertEqual(parsed.precision, "amp_fp16")
        self.assertEqual(parsed.max_steps, 10)

    def test_summary_gate_requires_frozen_amp_cee_batch3(self) -> None:
        summary = {
            "global_step": 10,
            "stop_reason": "max_steps",
            "training_config": {
                "batch_size": 3,
                "precision": "amp_fp16",
                "temporal_role_mode": "independent_sigmoid",
                "counterfactual_enabled": True,
            },
            "counterfactual_evidence_equivariance": {
                "train": {
                    "surface_invariance": True,
                    "family_equivariance": True,
                    "question_equivariance": True,
                }
            },
            "precision": {
                "mode": "amp_fp16",
                "autocast": {"enabled": True},
                "audiosep_parameter_dtype_mutated": False,
                "gradient_scaler": {"current_scale": 32768.0},
            },
            "audiosep": {"backbone_trainable_parameter_count": 0},
            "run_resources": {
                "cuda_peak_allocated_bytes_down": 100,
                "cuda_peak_reserved_bytes_down": 120,
            },
        }
        gate = validate_summary(summary, max_steps=10)
        self.assertTrue(gate["all_gates_passed"])
        summary["training_config"]["batch_size"] = 1
        summary["precision"]["audiosep_parameter_dtype_mutated"] = True
        failed = validate_summary(summary, max_steps=10)
        self.assertFalse(failed["all_gates_passed"])
        self.assertGreaterEqual(len(failed["errors"]), 2)

    def test_gpu_preflight_is_fail_closed(self) -> None:
        require_idle_gpu(
            {"free_memory_mib_↑": 10_000, "utilization_percent_↓": 5},
            9_000,
            10,
        )
        with self.assertRaises(RuntimeError):
            require_idle_gpu(
                {"free_memory_mib_↑": 8_999, "utilization_percent_↓": 5},
                9_000,
                10,
            )
        with self.assertRaises(RuntimeError):
            require_idle_gpu(
                {"free_memory_mib_↑": 10_000, "utilization_percent_↓": 11},
                9_000,
                10,
            )


if __name__ == "__main__":
    unittest.main()
