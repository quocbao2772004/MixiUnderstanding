"""Tests for the frozen QCES-v5 pilot orchestrator."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from mixi_understanding.scripts.run_qces_v5_dev_pilot import (
    BOUND_ARTIFACTS,
    FORMAT,
    BoundArtifact,
    audit_manifest,
    build_comparison_summary,
    build_plan,
    validate_completed_training,
    validate_hashes,
)


def option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def initialization_receipt(candidate: str) -> dict[str, object]:
    common = [f"common_{index:02d}" for index in range(67)]
    candidate_only = (
        []
        if candidate == "union_single"
        else [
            "same_semantic_head.0.weight",
            "same_semantic_head.0.bias",
            "same_semantic_head.2.weight",
            "same_semantic_head.2.bias",
        ]
    )
    return {
        "common_state_tensor_keys": common,
        "candidate_only_state_tensor_keys": candidate_only,
        "aggregates": {"common_state_tensors_sha256": "shared-hash"},
    }


def training_summary(candidate: str, weakest_error: float) -> dict[str, object]:
    return {
        "global_step": 512,
        "stop_reason": "max_steps",
        "manifest_sha256": BOUND_ARTIFACTS[0].sha256,
        "val_manifest_sha256": BOUND_ARTIFACTS[1].sha256,
        "training_config": {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 0.0003,
            "dropout": 0.1,
            "crop_seconds": 10.0,
            "max_steps": 512,
            "num_workers": 0,
            "seed": 2026,
            "precision": "fp32",
            "deterministic": True,
            "selection_metric": "evidence_sd_sdr",
            "foundation_feature_mode": "audiosep_clap",
            "temporal_role_mode": "independent_sigmoid",
            "counterfactual_enabled": False,
            "separator_aware_refiner": False,
        },
        "from_scratch_composer_initialization": initialization_receipt(candidate),
        "best_validation_metrics": {
            "evidence_sd_sdr": 1.0,
            "answerable_temporal_iou": 0.4,
            "no_evidence_retained_ratio": 0.2,
            "weakest_role_waveform": weakest_error,
            "same_semantic_brier": 0.2,
            "same_semantic_accuracy": 0.8,
        },
        "run_resources": {
            "scope": "training and validation",
            "wall_seconds_down": 10.0,
            "startup_seconds_down": 1.0,
            "training_seconds_down": 7.0,
            "validation_seconds_down": 2.0,
            "optimizer_steps": 512,
            "optimizer_steps_per_second_up": 51.2,
            "optimizer_steps_per_training_second_up": 73.1,
            "cuda_peak_allocated_bytes_down": 100,
            "cuda_peak_reserved_bytes_down": 120,
        },
    }


def evaluation_report(
    *, evidence_sd_sdr: float, temporal_iou: float, retention: float
) -> dict[str, object]:
    summary = {
        "evidence_sd_sdr_answerable": evidence_sd_sdr,
        "evidence_sd_sdri_answerable": evidence_sd_sdr - 0.2,
        "evidence_si_sdr_answerable": evidence_sd_sdr - 0.1,
        "evidence_si_sdri_answerable": evidence_sd_sdr - 0.3,
        "weakest_role_sd_sdr_answerable": evidence_sd_sdr - 0.4,
        "weakest_role_si_sdr_answerable": evidence_sd_sdr - 0.5,
        "answerable_temporal_iou": temporal_iou,
        "mean_no_evidence_retained_ratio": retention,
        "maximum_mixture_consistency_l1": 0.0,
    }
    arrows = {
        "evidence_sd_sdr_answerable": "↑",
        "evidence_sd_sdri_answerable": "↑",
        "evidence_si_sdr_answerable": "↑",
        "evidence_si_sdri_answerable": "↑",
        "weakest_role_sd_sdr_answerable": "↑",
        "weakest_role_si_sdr_answerable": "↑",
        "answerable_temporal_iou": "↑",
        "mean_no_evidence_retained_ratio": "↓",
        "maximum_mixture_consistency_l1": "↓",
    }
    directed = {f"{key}_{arrows[key]}": value for key, value in summary.items()}
    return {
        "records": 288,
        "summary": summary,
        "summary_with_directions": directed,
        "items": [
            {
                "same_role_label": True,
                "no_evidence": False,
                "evidence_sd_sdr": evidence_sd_sdr,
                "temporal_iou": temporal_iou,
            }
        ],
    }


class FrozenPilotOrchestratorTest(unittest.TestCase):
    def test_plan_is_fair_fp32_and_never_names_a_test_manifest(self) -> None:
        root = Path("/tmp/qces_project")
        plan = build_plan(root, Path(sys.executable), root / "outputs" / "results", {})
        self.assertEqual(plan["format"], FORMAT)
        self.assertFalse(plan["paper_result_eligible"])
        self.assertEqual(plan["test_manifests_accessed ↓"], 0)

        union = plan["candidates"]["union_single"]["training_command"]
        dual = plan["candidates"]["dual_role"]["training_command"]
        for command in (union, dual):
            self.assertEqual(option(command, "--max-steps"), "512")
            self.assertEqual(option(command, "--precision"), "fp32")
            self.assertEqual(option(command, "--batch-size"), "1")
            self.assertEqual(option(command, "--crop-seconds"), "10")
            self.assertEqual(option(command, "--seed"), "2026")
            self.assertIn("--deterministic", command)
            self.assertIn("--no-separator-aware-refiner", command)
            self.assertNotIn("--overwrite", command)
            self.assertNotIn("test", Path(option(command, "--manifest")).name)
            self.assertNotIn("test", Path(option(command, "--val-manifest")).name)
            for flag in (
                "--surface-semantic-invariance-weight",
                "--family-evidence-delta-weight",
                "--question-evidence-delta-weight",
            ):
                self.assertEqual(option(command, flag), "0")
        self.assertEqual(option(union, "--semantic-weight"), "0.1")
        self.assertEqual(option(dual, "--role-semantic-weight"), "0.05")
        self.assertEqual(option(dual, "--same-semantic-weight"), "0.1")
        with self.assertRaisesRegex(ValueError, "must stay under"):
            build_plan(root, Path(sys.executable), root / "unsafe", {})

        for candidate in ("union_single", "dual_role"):
            evaluation = plan["candidates"][candidate]["evaluation_command"]
            self.assertIn("--no-render-audio", evaluation)
            self.assertNotIn("--overwrite", evaluation)
            self.assertNotIn("test", Path(option(evaluation, "--manifest")).name)

    def test_hash_and_manifest_preflight_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"bound")
            import hashlib

            digest = hashlib.sha256(b"bound").hexdigest()
            receipt = validate_hashes(
                root, [BoundArtifact(Path("artifact.bin"), digest)]
            )
            self.assertEqual(receipt[0]["sha256"], digest)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                validate_hashes(root, [BoundArtifact(Path("artifact.bin"), "0" * 64)])

            manifest = root / "pilot.jsonl"
            rows = [
                {
                    "split": "train",
                    "scene_family_id": "f0",
                    "scene_id": f"s{index}",
                    "variant_id": variant,
                    "source_group_ids": ["g0"],
                }
                for index, variant in enumerate(("base", "order_swap", "anchor_drop"))
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            audit = audit_manifest(
                manifest,
                expected_split="train",
                expected_records=3,
                expected_families=1,
                expected_scenes=3,
            )
            self.assertEqual(audit["test_records_accessed ↓"], 0)
            rows[0]["split"] = "test"
            manifest.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test record is forbidden"):
                audit_manifest(
                    manifest,
                    expected_split="train",
                    expected_records=3,
                    expected_families=1,
                    expected_scenes=3,
                )

    def test_comparison_summary_has_directions_and_applies_frozen_gate(self) -> None:
        summaries = {
            "union_single": training_summary("union_single", 0.20),
            "dual_role": training_summary("dual_role", 0.205),
        }
        fairness = validate_completed_training(summaries)
        reports = {
            "union_single": evaluation_report(
                evidence_sd_sdr=1.0, temporal_iou=0.40, retention=0.20
            ),
            "dual_role": evaluation_report(
                evidence_sd_sdr=1.6, temporal_iou=0.41, retention=0.205
            ),
        }
        result = build_comparison_summary(
            {"format": FORMAT}, summaries, reports, fairness, wav_file_count=0
        )
        gate = result["predeclared_promotion_gate"]
        self.assertTrue(gate["condition_1_non_regression_pass ↑"])
        self.assertTrue(gate["condition_2_material_gain_pass ↑"])
        self.assertTrue(gate["condition_3_same_semantic_non_reversal_pass ↑"])
        self.assertTrue(gate["dual_authorized_for_full_validation ↑"])
        metrics = result["candidates"]["union_single"]["report_only_evaluation_metrics"]
        self.assertTrue(all("↑" in key or "↓" in key for key in metrics))
        resources = result["candidates"]["union_single"][
            "training_resources_with_directions"
        ]
        resource_metric_keys = [
            key
            for key in resources
            if key not in {"scope", "optimizer_steps (fixed by protocol)"}
        ]
        self.assertTrue(all("↑" in key or "↓" in key for key in resource_metric_keys))


if __name__ == "__main__":
    unittest.main()
