"""CPU-only contracts for the schedule-matched CEE development pilot."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mixi_understanding.qces.counterfactual import COUNTERFACTUAL_METRIC_DIRECTIONS
from mixi_understanding.scripts.run_qces_cee_dev_pilot import (
    CANDIDATES,
    build_evaluate_command,
    build_train_command,
    candidate_execution_state,
    compare_candidates,
)
from mixi_understanding.scripts.train_qces import parse_args as parse_train_args


def _option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        python=Path("/usr/bin/python3"),
        train_manifest=root / "train.jsonl",
        val_manifest=root / "val.jsonl",
        cache_root=root / "cache",
        results_root=root / "results",
        audiosep_root=root / "audiosep",
        audiosep_config=root / "audiosep.yaml",
        audiosep_checkpoint=root / "audiosep.pt",
        seed=2026,
        epochs=2,
        max_steps=512,
    )


class CEEDevPilotTest(unittest.TestCase):
    def test_candidates_share_training_contract_except_paired_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            off = build_train_command(args, "cee_off_matched")
            on = build_train_command(args, "cee_on")
            off_parsed = parse_train_args(off[2:])
            on_parsed = parse_train_args(on[2:])
            for parsed in (off_parsed, on_parsed):
                self.assertEqual(parsed.batch_size, 3)
                self.assertEqual(parsed.precision, "amp_fp16")
                self.assertEqual(parsed.max_steps, 512)
                self.assertEqual(parsed.seed, 2026)
                self.assertTrue(parsed.save_every_epoch)
            self.assertTrue(off_parsed.force_counterfactual_batching)
            self.assertFalse(on_parsed.force_counterfactual_batching)
            self.assertEqual(off_parsed.family_temporal_delta_weight, 0.0)
            self.assertGreater(on_parsed.family_temporal_delta_weight, 0.0)
            evaluation = build_evaluate_command(
                args, "cee_on", ("one", "two", "three", "four")
            )
            self.assertEqual(evaluation.count("--render-item-id"), 4)
            self.assertNotIn("--no-render-audio", evaluation)

    def test_execution_state_resumes_complete_stages_and_rejects_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pending = candidate_execution_state(root, "cee_on")
            self.assertFalse(pending["train_complete"])
            train = root / "cee_on_train"
            train.mkdir()
            (train / "checkpoint.pt").write_bytes(b"checkpoint")
            with self.assertRaisesRegex(RuntimeError, "partial training"):
                candidate_execution_state(root, "cee_on")
            (train / "summary.json").write_text("{}", encoding="utf-8")
            trained = candidate_execution_state(root, "cee_on")
            self.assertTrue(trained["train_complete"])
            self.assertFalse(trained["evaluation_complete"])
            evaluation = root / "cee_on_eval"
            evaluation.mkdir()
            (evaluation / "evaluation_report.json").write_text("{}", encoding="utf-8")
            complete = candidate_execution_state(root, "cee_on")
            self.assertTrue(complete["evaluation_complete"])

    def test_comparison_enforces_integrity_and_predeclared_promotion_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {
                "surface_invariance": True,
                "family_equivariance": True,
                "question_equivariance": True,
            }
            paired_off = {}
            paired_on = {}
            for name, metadata in COUNTERFACTUAL_METRIC_DIRECTIONS.items():
                if metadata["direction"] == "maximize":
                    paired_off[name], paired_on[name] = 0.25, 0.50
                else:
                    paired_off[name], paired_on[name] = 0.50, 0.25
            for candidate in CANDIDATES:
                train = root / f"{candidate}_train"
                evaluation = root / f"{candidate}_eval"
                train.mkdir()
                evaluation.mkdir()
                on = candidate == "cee_on"
                summary = {
                    "from_scratch_composer_initialization": {"sha256": "same"},
                    "training_config": {
                        "batch_size": 3,
                        "precision": "amp_fp16",
                        "seed": 2026,
                    },
                    "counterfactual_evidence_equivariance": {
                        "train": plan,
                        "paired_objective_enabled": on,
                        "forced_schedule_matched_control": not on,
                    },
                    "best_validation_metrics": paired_on if on else paired_off,
                }
                (train / "summary.json").write_text(
                    json.dumps(summary), encoding="utf-8"
                )
                metrics = {
                    "evidence_sd_sdri_answerable": 1.0 if on else 0.5,
                    "evidence_sd_sdr_answerable": 1.1 if on else 1.0,
                    "weakest_role_sd_sdr_answerable": 0.9 if on else 0.8,
                    "answerable_temporal_iou": 0.4 if on else 0.3,
                    "mean_no_evidence_retained_ratio": 0.05 if on else 0.06,
                    "maximum_mixture_consistency_l1": 0.0,
                }
                (evaluation / "evaluation_report.json").write_text(
                    json.dumps({"summary": metrics}), encoding="utf-8"
                )
            result = compare_candidates(root)
            self.assertTrue(result["integrity"]["all_passed"])
            self.assertEqual(
                result["decision"]["primary_cee_metrics_improved_count_↑"], 4
            )
            self.assertTrue(result["decision"]["promote_cee_to_full_seeded_run"])


if __name__ == "__main__":
    unittest.main()
