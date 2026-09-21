"""CPU-only tests for the Phi-4 MM CUDA-forward gate contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mixi_understanding.scripts.evaluate_qces_audioqa import (
    parse_args as parse_audioqa_args,
)
from mixi_understanding.scripts.run_qces_phi4mm_forward_gate import (
    MODEL_ID,
    PRIMARY_CONDITIONS,
    REVISION,
    build_command,
    compare_repeatability,
    load_monitor,
    parse_args,
    validate_output,
)


def _item(condition: str, scores: list[float] | None = None) -> dict:
    values = scores or [-1.0, -2.0, -3.0, -4.0, -5.0]
    return {
        "id": "val_000000_0_00",
        "condition": condition,
        "question": "What follows?",
        "answer_options": ["a", "b", "c", "d", "no_evidence"],
        "source_audio_sha256": "a" * 64,
        "predicted_answer": "a",
        "option_log_scores": values,
        "option_probabilities": [0.5, 0.2, 0.15, 0.1, 0.05],
        "candidate_token_lengths": [1, 1, 1, 1, 1],
        "scoring_method": "single_token_next_log_probability",
    }


def _write_output(root: Path, conditions: tuple[str, ...]) -> None:
    metadata = {
        "run_fingerprint": "f" * 64,
        "run_config": {
            "auditor": "phi4mm",
            "model": MODEL_ID,
            "revision": REVISION,
            "quantization": "4bit",
            "dtype": "float16",
            "device": "cuda",
            "max_records": 1,
            "conditions": list(conditions),
        },
        "runtime_model": {
            "auditor_family": "phi4_multimodal",
            "requested_revision": REVISION,
            "resolved_commit_hash": REVISION,
            "trusted_custom_code": True,
            "option_continuation_token_ids": [[32], [33], [34], [35], [36]],
        },
    }
    root.mkdir(parents=True)
    (root / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "evaluation_report.json").write_text(
        json.dumps(
            {
                "format": "qces_audioqa_audit_v7",
                "run_fingerprint": metadata["run_fingerprint"],
            }
        ),
        encoding="utf-8",
    )
    (root / "items.jsonl").write_text(
        "".join(json.dumps(_item(condition)) + "\n" for condition in conditions),
        encoding="utf-8",
    )


class Phi4MMForwardGateTest(unittest.TestCase):
    def test_generated_evaluator_commands_parse_with_exact_runtime_contract(
        self,
    ) -> None:
        args = parse_args([])
        command = build_command(args, Path("output"), PRIMARY_CONDITIONS)
        parsed = parse_audioqa_args(command[2:])
        self.assertEqual(parsed.auditor, "phi4mm")
        self.assertEqual(parsed.model, MODEL_ID)
        self.assertEqual(parsed.revision, REVISION)
        self.assertEqual(parsed.conditions, PRIMARY_CONDITIONS)
        self.assertEqual(parsed.max_records, 1)
        self.assertEqual(parsed.quantization, "4bit")
        self.assertEqual(parsed.dtype, "float16")
        self.assertEqual(parsed.device, "cuda")
        self.assertTrue(parsed.local_files_only)

        validate = build_command(
            args, Path("output"), PRIMARY_CONDITIONS, validate_only=True
        )
        self.assertIn("--validate-only", validate)

    def test_output_validation_and_fresh_process_repeatability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary_dir = root / "primary"
            repeat_dir = root / "repeat"
            _write_output(primary_dir, PRIMARY_CONDITIONS)
            _write_output(repeat_dir, ("mixture",))
            primary = validate_output(primary_dir, PRIMARY_CONDITIONS)
            repeat = validate_output(repeat_dir, ("mixture",))
            comparison = compare_repeatability(primary, repeat, 1e-5)
            self.assertTrue(comparison["all_checks_passed"])
            self.assertEqual(comparison["maximum_log_score_absolute_difference_↓"], 0.0)

            items = [
                json.loads(line)
                for line in (repeat_dir / "items.jsonl").read_text().splitlines()
            ]
            items[0]["option_log_scores"][0] += 0.01
            (repeat_dir / "items.jsonl").write_text(
                json.dumps(items[0]) + "\n", encoding="utf-8"
            )
            drifted = validate_output(repeat_dir, ("mixture",))
            comparison = compare_repeatability(primary, drifted, 1e-5)
            self.assertFalse(comparison["all_checks_passed"])
            self.assertGreater(
                comparison["maximum_log_score_absolute_difference_↓"], 1e-5
            )

    def test_completed_output_requires_runner_owned_positive_gpu_monitor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "logs" / "primary"
            command = ["python", "evaluate.py"]
            with self.assertRaisesRegex(RuntimeError, "lacks runner-owned"):
                load_monitor(prefix, command)
            prefix.parent.mkdir(parents=True)
            prefix.with_suffix(".monitor.json").write_text(
                json.dumps(
                    {
                        "command": command,
                        "returncode": 0,
                        "peak_process_gpu_memory_mib_↓": 6_100,
                        "gpu_memory_samples_↑": 20,
                    }
                ),
                encoding="utf-8",
            )
            monitor = load_monitor(prefix, command)
            self.assertEqual(monitor["peak_process_gpu_memory_mib_↓"], 6_100)


if __name__ == "__main__":
    unittest.main()
