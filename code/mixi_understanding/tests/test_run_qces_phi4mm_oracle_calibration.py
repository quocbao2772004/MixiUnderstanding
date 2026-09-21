"""CPU-only contracts for the Phi-4 MM oracle-calibration pilot."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from mixi_understanding.scripts.evaluate_qces_audioqa import (
    FORMAT_VERSION as AUDIOQA_FORMAT,
    parse_args as parse_audioqa_args,
)
from mixi_understanding.scripts.run_qces_phi4mm_oracle_calibration import (
    BASE_CONDITIONS,
    CODE_ROOT,
    EXPECTED_CONDITIONS,
    LOG_SCORE_EFFECTS,
    MODEL_ID,
    OPTION_ORDER_CONDITIONS,
    PROJECT_ROOT,
    REVISION,
    _sha256,
    analyze_report,
    build_command,
    parse_args,
    select_subset,
    validate_calibration_output,
    write_subset,
)


def _report() -> dict:
    paired = {name: 0.5 for name in LOG_SCORE_EFFECTS}
    paired["mixture_gold_log_score_gain_over_question_only_↑"] = 0.25
    return {
        "format": AUDIOQA_FORMAT,
        "counts": {
            "records": 24,
            "answerable": 18,
            "no_evidence": 6,
            "conditions": len(EXPECTED_CONDITIONS),
            "completed_record_conditions": 24 * len(EXPECTED_CONDITIONS),
        },
        "paired_metrics": paired,
        "paired_scene_bootstrap_95ci": {name: [0.1, 0.9] for name in LOG_SCORE_EFFECTS},
        "condition_metrics": {
            "oracle_evidence": {
                "answerable_accuracy_↑": 0.75,
                "answerable_candidate_aware_chance_accuracy_↑": 0.30,
                "no_evidence_accuracy_↑": 0.80,
            }
        },
        "option_order_control_metrics_by_condition": {
            condition: {
                "option_order_gold_position_changed_rate_↑": 1.0,
                "option_order_semantic_prediction_invariance_↑": 0.90,
                "option_order_accuracy_absolute_gap_↓": 0.05,
            }
            for condition in OPTION_ORDER_CONDITIONS
        },
        "paired_subset_counts": {
            "audio_dependent_mixture_correct_question_only_wrong_records": 4
        },
    }


class Phi4MMOracleCalibrationTest(unittest.TestCase):
    def test_identifier_only_subset_is_exactly_six_families_by_four_slots(self) -> None:
        source = PROJECT_ROOT / "data/qces_v5_paper/qces_devpilot_val_seed2026.jsonl"
        rows, receipt = select_subset(source)
        self.assertEqual(len(rows), 24)
        self.assertEqual(len({row["scene_family_id"] for row in rows}), 6)
        self.assertEqual({row["question_index"] for row in rows}, {0, 3, 8, 14})
        self.assertTrue(receipt["selection"]["uses_identifiers_only"])
        self.assertFalse(
            receipt["selection"]["uses_audio_labels_answers_targets_or_metrics"]
        )
        with tempfile.TemporaryDirectory() as directory:
            first = write_subset(Path(directory), rows)
            second = write_subset(Path(directory), rows)
            self.assertEqual(first, second)
            self.assertEqual(len(first.read_text().splitlines()), 24)

    def test_command_parses_all_oracle_and_isolated_option_controls(self) -> None:
        args = parse_args([])
        command = build_command(args, Path("subset.jsonl"), Path("output"))
        parsed = parse_audioqa_args(command[2:])
        self.assertEqual(parsed.conditions, EXPECTED_CONDITIONS)
        self.assertEqual(parsed.audio_conditions, BASE_CONDITIONS)
        self.assertEqual(
            parsed.option_order_control_conditions, OPTION_ORDER_CONDITIONS
        )
        self.assertEqual(parsed.dataset_root, args.source_manifest.resolve().parent)
        self.assertIsNone(parsed.max_records)
        self.assertEqual(parsed.auditor, "phi4mm")

    def test_decision_separates_point_effect_pilot_from_confirmatory_ci(self) -> None:
        report = _report()
        passed = analyze_report(report)
        self.assertTrue(passed["pilot_go"])
        self.assertTrue(passed["confirmatory_pass"])
        self.assertEqual(passed["decision"], "phi4mm_predicted_dev_pilot_authorized")

        uncertain = copy.deepcopy(report)
        uncertain["paired_scene_bootstrap_95ci"][LOG_SCORE_EFFECTS[0]][0] = -0.1
        expansion = analyze_report(uncertain)
        self.assertTrue(expansion["pilot_go"])
        self.assertFalse(expansion["confirmatory_pass"])
        self.assertEqual(expansion["decision"], "phi4mm_oracle_expansion_required")

        failed = copy.deepcopy(report)
        failed["paired_metrics"][LOG_SCORE_EFFECTS[0]] = -0.01
        rejected = analyze_report(failed)
        self.assertFalse(rejected["pilot_go"])
        self.assertEqual(rejected["decision"], "phi4mm_auditor_rejected_on_pilot")

    def test_completed_calibration_is_bound_to_current_evaluator_and_subset(
        self,
    ) -> None:
        args = parse_args([])
        evaluator = CODE_ROOT / "mixi_understanding/scripts/evaluate_qces_audioqa.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subset = root / "subset.jsonl"
            subset.write_text("{}\n", encoding="utf-8")
            output = root / "output"
            output.mkdir()
            metadata = {
                "run_fingerprint": "f" * 64,
                "script_sha256": _sha256(evaluator),
                "run_config": {
                    "evaluator_script_sha256": _sha256(evaluator),
                    "manifest": str(subset.resolve()),
                    "manifest_sha256": _sha256(subset),
                    "dataset_root": str(args.source_manifest.resolve().parent),
                    "conditions": list(EXPECTED_CONDITIONS),
                    "auditor": "phi4mm",
                    "model": MODEL_ID,
                    "revision": REVISION,
                    "quantization": "4bit",
                },
                "runtime_model": {
                    "resolved_commit_hash": REVISION,
                    "auditor_family": "phi4_multimodal",
                },
            }
            (output / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            (output / "evaluation_report.json").write_text(
                json.dumps({"run_fingerprint": metadata["run_fingerprint"]}),
                encoding="utf-8",
            )
            (output / "items.jsonl").write_text("{}\n", encoding="utf-8")
            validated, _ = validate_calibration_output(args, subset, output)
            self.assertEqual(validated["run_fingerprint"], "f" * 64)

            metadata["run_config"]["dataset_root"] = str(root)
            (output / "run_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "dataset_root_match"):
                validate_calibration_output(args, subset, output)


if __name__ == "__main__":
    unittest.main()
