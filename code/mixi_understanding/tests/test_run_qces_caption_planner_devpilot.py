"""CPU-only tests for the durable caption/planner development comparator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mixi_understanding.scripts.evaluate_qces_v5_caption_planner_audiosep import (
    FORMAT_VERSION as EVALUATION_FORMAT,
)
from mixi_understanding.scripts.generate_qces_v5_caption_planner import (
    FORMAT_VERSION as PLANNER_FORMAT,
)
from mixi_understanding.scripts.run_qces_caption_planner_devpilot import (
    EXPECTED_RECORDS,
    EXPECTED_SCENES,
    EXPECTED_SHA256,
    ORACLE_TIME_MODE,
    TOP_ENERGY_MODE,
    _validate_planner_report,
    analyze_reports,
    build_commands,
    parse_args,
)


def _reports() -> tuple[dict, dict, dict]:
    planner = {
        "format": PLANNER_FORMAT,
        "run_fingerprint": "f" * 64,
        "manifest_sha256": EXPECTED_SHA256["manifest"],
        "summary": {
            "record_count_↑": EXPECTED_RECORDS,
            "scene_count_↑": EXPECTED_SCENES,
            "test_records_accessed_↓": 0,
            "caption_valid_json_rate_↑": 0.9,
        },
    }
    evaluation = {
        "format": EVALUATION_FORMAT,
        "manifest_sha256": EXPECTED_SHA256["manifest"],
        "selected_record_count": EXPECTED_RECORDS,
        "planner_provenance": {"run_fingerprint": "f" * 64},
        "summary": {
            "evidence_sd_sdri_answerable_mean_db_↑": 2.0,
            "weakest_role_sd_sdr_answerable_mean_db_↑": -1.0,
            "no_evidence_retained_ratio_mean_↓": 0.1,
            "mixture_consistency_l1_sanity_maximum_↓": 0.0,
        },
    }
    temporal = {
        "summaries": {
            TOP_ENERGY_MODE: {
                "evidence_sd_sdri_answerable_mean_db_↑": -0.1,
                "weakest_role_sd_sdr_answerable_mean_db_↑": -3.0,
                "no_evidence_retained_ratio_mean_↓": 0.6,
            },
            ORACLE_TIME_MODE: {
                "evidence_sd_sdri_answerable_mean_db_↑": 10.0,
                "weakest_role_sd_sdr_answerable_mean_db_↑": -2.0,
                "no_evidence_retained_ratio_mean_↓": 0.0,
            },
        }
    }
    return planner, evaluation, temporal


class CaptionPlannerDevPilotRunnerTest(unittest.TestCase):
    def test_commands_keep_caption_planner_non_oracle_and_environments_separate(
        self,
    ) -> None:
        commands = build_commands(parse_args([]))

        self.assertNotEqual(commands["generate"][0], commands["evaluate"][0])
        self.assertIn("--hash-model-weights", commands["generate"])
        self.assertIn("--local-files-only", commands["generate"])
        self.assertNotIn("--allow-test-split", commands["generate"])
        self.assertNotIn("--allow-test-split", commands["evaluate"])
        self.assertIn("--no-render-audio", commands["evaluate"])

    def test_analysis_orients_all_top_energy_improvements_upward(self) -> None:
        planner, evaluation, temporal = _reports()
        receipt = analyze_reports(planner, evaluation, temporal)

        self.assertTrue(receipt["integrity"]["all_passed"])
        comparison = receipt["waveform_comparison"]
        self.assertAlmostEqual(
            comparison["evidence_sd_sdri_answerable_mean_db_↑"][
                "caption_over_top_energy_improvement_↑"
            ],
            2.1,
        )
        self.assertAlmostEqual(
            comparison["no_evidence_retained_ratio_mean_↓"][
                "caption_over_top_energy_improvement_↑"
            ],
            0.5,
        )
        self.assertEqual(
            receipt["development_interpretation"]["metrics_beating_top_energy_count_↑"],
            3,
        )
        self.assertFalse(receipt["paper_result_eligible"])

    def test_analysis_rejects_an_unbound_waveform_report(self) -> None:
        planner, evaluation, temporal = _reports()
        evaluation["planner_provenance"]["run_fingerprint"] = "wrong"

        receipt = analyze_reports(planner, evaluation, temporal)

        self.assertFalse(receipt["integrity"]["all_passed"])
        self.assertIn(
            "waveform report is not bound to the planner run",
            receipt["integrity"]["errors"],
        )

    def test_planner_report_validation_is_manifest_count_and_test_bound(self) -> None:
        planner, _, _ = _reports()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "planner_report.json"
            path.write_text(json.dumps(planner), encoding="utf-8")
            self.assertEqual(_validate_planner_report(path)["format"], PLANNER_FORMAT)

            planner["summary"]["test_records_accessed_↓"] = 1
            path.write_text(json.dumps(planner), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "frozen pilot contract"):
                _validate_planner_report(path)


if __name__ == "__main__":
    unittest.main()
