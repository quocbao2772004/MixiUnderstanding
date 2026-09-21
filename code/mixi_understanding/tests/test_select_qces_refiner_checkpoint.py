"""Synthetic CPU-only tests for the frozen joint epoch selector."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Sequence

import torch

from mixi_understanding.scripts.select_qces_refiner_checkpoint import (
    SD_SDR_MIN_KEY,
    SD_SDR_P10_KEY,
    SelectionInputError,
    Thresholds,
    build_selection_receipt,
    linear_percentile,
    main,
)


def _identity(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


class _Fixture:
    def __init__(self, root: Path, epochs: int = 2) -> None:
        self.root = root
        self.manifest = root / "val.jsonl"
        self.manifest.write_text("{}\n", encoding="utf-8")
        self.manifest_identity = _identity(self.manifest)
        self.base_checkpoint = root / "base.pt"
        self.base_checkpoint.write_bytes(b"base-checkpoint")
        self.base_report = root / "base_report.json"
        self.summary_path = root / "summary.json"
        self.checkpoints: Dict[int, Path] = {}
        self.report_paths: Dict[int, Path] = {}
        self.summary_rows = []
        for epoch in range(1, epochs + 1):
            checkpoint = root / f"epoch_{epoch:04d}.pt"
            checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
            self.checkpoints[epoch] = checkpoint

    def report(
        self,
        checkpoint: Path,
        sd_values: Sequence[float],
        *,
        epoch: int | None,
        ne_retained: float = 0.10,
        false_silence: float = 0.0,
        si_values: Sequence[float] = (-10.0, -9.0, -8.0),
        temporal_iou: float = 0.4,
        weakest_error: float = 0.8,
        include_tail_summary: bool = False,
    ) -> Dict[str, Any]:
        ids = ["answerable_0", "answerable_1", "answerable_2"]
        items = [
            {
                "id": sample_id,
                "no_evidence": False,
                "conditions": {
                    "learned_semantic__learned_soft_temporal": {
                        "evidence_sd_sdr_db_↑": float(sd),
                        "evidence_si_sdr_db_↑": float(si),
                    }
                },
            }
            for sample_id, sd, si in zip(ids, sd_values, si_values)
        ]
        items.append(
            {
                "id": "no_evidence_0",
                "no_evidence": True,
                "conditions": {
                    "learned_semantic__learned_soft_temporal": {
                        "evidence_sd_sdr_db_↑": None,
                        "evidence_si_sdr_db_↑": None,
                    }
                },
            }
        )
        condition_summary: Dict[str, Any] = {
            "no_evidence_retained_ratio_mean_↓": ne_retained,
            "evidence_sd_sdr_answerable_mean_db_↑": sum(sd_values)
            / len(sd_values),
            "evidence_si_sdr_below_minus20_db_rate_↓": sum(
                value < -20.0 for value in si_values
            )
            / len(si_values),
        }
        if include_tail_summary:
            condition_summary.update(
                {
                    SD_SDR_P10_KEY: linear_percentile(sd_values, 0.1),
                    SD_SDR_MIN_KEY: min(sd_values),
                }
            )
        validation_metrics = {
            "answerable_temporal_iou": temporal_iou,
            "evidence_sd_sdr": sum(sd_values) / len(sd_values),
            "no_evidence_retained_ratio": ne_retained,
            "weakest_role_waveform": weakest_error,
        }
        metadata: Dict[str, Any] = {
            "checkpoint_epoch": epoch if epoch is not None else 4,
            "checkpoint_global_step": (epoch or 4) * 10,
            "checkpoint_role": "retained_epoch" if epoch is not None else "best_validation",
        }
        if epoch is not None:
            metadata["epoch_retention"] = {
                "policy": "save_every_epoch",
                "immutable": True,
                "epoch": epoch,
                "global_step": epoch * 10,
                "checkpoint": str(checkpoint.resolve()),
                "validation_metrics_at_checkpoint": validation_metrics,
            }
            torch.save({"model_state": {}, "extra": metadata}, checkpoint)
        checkpoint_identity = _identity(checkpoint)
        return {
            "format": "qces_audiosep_factorization_v1",
            "split": "val",
            "protocol": {
                "separator_training": "frozen; evaluation-only",
                "composer_training": "frozen; evaluation-only",
                "conditions": {
                    "learned_semantic__learned_soft_temporal": {
                        "deployable": True,
                        "uses_oracle_annotation": False,
                    }
                },
            },
            "dataset_statistics": {
                "record_count": 4,
                "answerable_count": 3,
                "no_evidence_count": 1,
            },
            "condition_summaries": {
                "learned_semantic__learned_soft_temporal": condition_summary
            },
            "no_evidence_calibration": {
                "validation_only_threshold_fit": {
                    "status": "fitted_on_validation_only",
                    "selection_split": "val",
                    "metrics": {"answerable_false_silence_rate_↓": false_silence},
                }
            },
            "provenance": {
                "inputs": {
                    "learned_qces_checkpoint": checkpoint_identity,
                    "manifest": self.manifest_identity,
                    "dataset_audio_inputs": {
                        "sha256": "1" * 64,
                        "total_bytes": 1234,
                        "unique_waveform_count": 9,
                    },
                    "audiosep_config": {"sha256": "2" * 64, "size_bytes": 2},
                    "audiosep_checkpoint": {
                        "sha256": "3" * 64,
                        "size_bytes": 3,
                    },
                    "oracle_semantic_cache": {
                        "sha256": "4" * 64,
                        "size_bytes": 4,
                    },
                },
                "software": {
                    "qces_source_files": [
                        {"path": "/source/evaluator.py", "sha256": "5" * 64}
                    ],
                    "audiosep_source_tree": {
                        "sha256": "6" * 64,
                        "hashed_file_count": 10,
                    },
                },
                "checkpoint_metadata": metadata,
            },
            "items": items,
        }

    def write_base(self) -> None:
        payload = self.report(
            self.base_checkpoint,
            [-30.0, -20.0, -10.0],
            epoch=None,
            ne_retained=0.10,
            false_silence=0.0,
            include_tail_summary=False,
        )
        self.base_report.write_text(json.dumps(payload), encoding="utf-8")

    def write_epoch(
        self,
        epoch: int,
        sd_values: Sequence[float],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload = self.report(
            self.checkpoints[epoch], sd_values, epoch=epoch, **kwargs
        )
        path = self.root / f"epoch_{epoch}_factorization_report.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.report_paths[epoch] = path
        metadata = payload["provenance"]["checkpoint_metadata"]
        validation = metadata["epoch_retention"][
            "validation_metrics_at_checkpoint"
        ]
        self.summary_rows.append(
            {
                "epoch": epoch,
                "global_step": epoch * 10,
                "checkpoint": _identity(self.checkpoints[epoch]),
                "validation_metrics_at_checkpoint": validation,
            }
        )
        return payload

    def write_summary(self) -> None:
        payload = {
            "epoch_checkpoint_retention": {
                "enabled": True,
                "immutable": True,
                "policy": "save_every_epoch",
            },
            "retained_epoch_checkpoints": sorted(
                self.summary_rows, key=lambda row: row["epoch"]
            ),
            "manifests": {"validation": self.manifest_identity},
        }
        self.summary_path.write_text(json.dumps(payload), encoding="utf-8")


class JointSelectorTest(unittest.TestCase):
    def test_linear_p10_definition(self) -> None:
        self.assertAlmostEqual(linear_percentile([-30, -20, -10, 0, 10], 0.1), -26)

    def test_selects_retained_epoch_by_declared_tie_break_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), epochs=2)
            fixture.write_base()
            fixture.write_epoch(
                1,
                [-29.0, -19.0, -9.0],
                temporal_iou=0.5,
                weakest_error=0.5,
            )
            fixture.write_epoch(
                2,
                [-28.0, -18.0, -8.0],
                temporal_iou=0.5,
                weakest_error=0.9,
                include_tail_summary=True,
            )
            fixture.write_summary()
            receipt = build_selection_receipt(
                fixture.base_report,
                fixture.summary_path,
                fixture.report_paths,
            )
            self.assertEqual(receipt["status"], "selected")
            self.assertTrue(receipt["selection_authoritative"])
            self.assertEqual(receipt["selected"]["epoch"], 2)
            self.assertIn("↑", SD_SDR_P10_KEY)
            self.assertEqual(
                receipt["base_metric_sources"][SD_SDR_P10_KEY],
                "derived_from_answerable_items_linear_(n-1)*0.1",
            )

    def test_strict_tail_regression_yields_no_feasible_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), epochs=1)
            fixture.write_base()
            fixture.write_epoch(
                1,
                [-31.0, -19.0, -9.0],
                temporal_iou=0.9,
                ne_retained=0.05,
            )
            fixture.write_summary()
            receipt = build_selection_receipt(
                fixture.base_report,
                fixture.summary_path,
                fixture.report_paths,
                Thresholds(),
            )
            self.assertEqual(receipt["status"], "no_feasible_checkpoint")
            self.assertIsNone(receipt["selected"])
            reasons = receipt["candidates"][0]["rejected_reasons"]
            self.assertTrue(any(SD_SDR_MIN_KEY in reason for reason in reasons))

    def test_item_coverage_mismatch_invalidates_the_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), epochs=1)
            fixture.write_base()
            payload = fixture.write_epoch(1, [-29.0, -19.0, -9.0])
            payload["items"][0]["id"] = "different_id"
            fixture.report_paths[1].write_text(json.dumps(payload), encoding="utf-8")
            fixture.write_summary()
            receipt = build_selection_receipt(
                fixture.base_report,
                fixture.summary_path,
                fixture.report_paths,
            )
            self.assertEqual(receipt["status"], "invalid_candidate_input")
            self.assertIsNone(receipt["selected"])
            self.assertIn(
                "coverage",
                receipt["candidates"][0]["rejected_reasons"][0],
            )

    def test_partial_evaluation_can_report_but_never_select(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), epochs=2)
            fixture.write_base()
            fixture.write_epoch(1, [-29.0, -19.0, -9.0], temporal_iou=0.9)
            fixture.write_epoch(2, [-28.0, -18.0, -8.0], temporal_iou=0.8)
            fixture.write_summary()
            only_one = {1: fixture.report_paths[1]}
            with self.assertRaisesRegex(SelectionInputError, "missing factorization"):
                build_selection_receipt(
                    fixture.base_report,
                    fixture.summary_path,
                    only_one,
                )
            receipt = build_selection_receipt(
                fixture.base_report,
                fixture.summary_path,
                only_one,
                allow_partial_evaluation=True,
            )
            self.assertEqual(receipt["status"], "incomplete_candidate_coverage")
            self.assertFalse(receipt["selection_authoritative"])
            self.assertIsNone(receipt["selected"])

    def test_checkpoint_verified_gate1_rejection_completes_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), epochs=2)
            fixture.write_base()
            fixture.write_epoch(1, [-29.0, -19.0, -9.0], temporal_iou=0.9)
            fixture.write_epoch(
                2,
                [-28.0, -18.0, -8.0],
                temporal_iou=1.0,
                ne_retained=0.2,
            )
            fixture.write_summary()
            receipt = build_selection_receipt(
                fixture.base_report,
                fixture.summary_path,
                {1: fixture.report_paths[1]},
            )
            self.assertEqual(receipt["status"], "selected")
            self.assertTrue(receipt["decision_authoritative"])
            self.assertEqual(receipt["selected"]["epoch"], 1)
            self.assertEqual(
                receipt["candidate_coverage"]["pre_rejected_gate1_epochs"], [2]
            )
            epoch_two = next(
                row for row in receipt["candidates"] if row["epoch"] == 2
            )
            self.assertEqual(epoch_two["evaluation_stage"], "pre_rejected_gate1")
            self.assertIsNone(epoch_two["report"])

    def test_checkpoint_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), epochs=1)
            fixture.write_base()
            fixture.write_epoch(1, [-29.0, -19.0, -9.0])
            fixture.write_summary()
            fixture.checkpoints[1].write_bytes(b"mutated-after-evaluation")
            with self.assertRaisesRegex(SelectionInputError, "identity"):
                build_selection_receipt(
                    fixture.base_report,
                    fixture.summary_path,
                    fixture.report_paths,
                )

    def test_summary_metric_must_match_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), epochs=1)
            fixture.write_base()
            fixture.write_epoch(1, [-29.0, -19.0, -9.0])
            fixture.write_summary()
            summary = json.loads(fixture.summary_path.read_text(encoding="utf-8"))
            summary["retained_epoch_checkpoints"][0][
                "validation_metrics_at_checkpoint"
            ]["no_evidence_retained_ratio"] = 0.99
            fixture.summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaisesRegex(SelectionInputError, "summary/checkpoint"):
                build_selection_receipt(
                    fixture.base_report,
                    fixture.summary_path,
                    fixture.report_paths,
                )

    def test_cli_emits_an_invalid_input_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "invalid_receipt.json"
            exit_code = main(
                [
                    "--base-report",
                    str(root / "missing_base.json"),
                    "--run-summary",
                    str(root / "missing_summary.json"),
                    "--epoch-report",
                    f"1={root / 'missing_epoch.json'}",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 2)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["status"], "invalid_input")
            self.assertFalse(receipt["selection_authoritative"])
            self.assertIsNone(receipt["selected"])
            self.assertTrue(receipt["rejected_reasons"])


if __name__ == "__main__":
    unittest.main()
