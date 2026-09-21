"""CPU-only tests for the interactive QCES demo contracts."""

from __future__ import annotations

import json
import fcntl
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from mixi_understanding.demo_contract import (
    compact_scene_events,
    DemoContractError,
    build_health_receipt,
    exclusive_lock_available,
    gpu_resource_ready,
    load_fingerprint_bound_examples,
    load_and_validate_health_receipt,
    relation_requirement,
    sha256_file,
)
from mixi_understanding.scripts.evaluate_qces_audioqa import format_mc_prompt
from mixi_understanding.scripts.infer_qces import prepare_audio_window
from mixi_understanding.scripts.infer_qces_audioqa import (
    parse_scene_inventory,
    validate_request,
)
from mixi_understanding.scripts.run_qces_microfit_gate import (
    analyze_microfit_result,
    build_commands,
    parse_args as parse_microfit_args,
    require_idle_gpu,
    select_listening_item_ids,
    write_health_outcome,
)
from mixi_understanding.scripts.train_qces import parse_args as parse_train_args


def _passing_summary() -> dict[str, float]:
    return {
        "answerable_temporal_iou": 0.95,
        "evidence_sd_sdri_answerable": 1.25,
        "no_evidence_accuracy": 0.95,
        "no_evidence_balanced_accuracy": 0.95,
        "no_evidence_auroc": 0.99,
        "mean_no_evidence_retained_ratio": 0.05,
        "maximum_mixture_consistency_l1": 1e-8,
    }


class CheckpointHealthTest(unittest.TestCase):
    def test_receipt_binds_checkpoint_report_and_explicit_directions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"controller")
            report = root / "evaluation_report.json"
            report.write_text(
                json.dumps(
                    {
                        "format": "qces_evaluation_v5",
                        "checkpoint": str(checkpoint.resolve()),
                        "manifest": str((root / "val.jsonl").resolve()),
                        "manifest_sha256": "a" * 64,
                        "summary": _passing_summary(),
                    }
                ),
                encoding="utf-8",
            )
            receipt_payload = build_health_receipt(checkpoint, report, "micro_overfit")
            self.assertTrue(receipt_payload["all_passed"])
            self.assertEqual(
                {gate["direction"] for gate in receipt_payload["gates"]},
                {"↑", "↓"},
            )
            receipt = root / "health.json"
            receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
            validated = load_and_validate_health_receipt(checkpoint, receipt)
            self.assertEqual(validated["permission"], "demo_inference_authorized")

            report.write_text(
                report.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(DemoContractError, "report changed"):
                load_and_validate_health_receipt(checkpoint, receipt)

    def test_silent_checkpoint_metrics_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"collapsed")
            summary = _passing_summary()
            summary["answerable_temporal_iou"] = 0.0
            summary["evidence_sd_sdri_answerable"] = -72.0
            summary["no_evidence_balanced_accuracy"] = 0.5
            summary["no_evidence_auroc"] = 0.5
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint.resolve()),
                        "summary": summary,
                    }
                ),
                encoding="utf-8",
            )
            receipt = build_health_receipt(checkpoint, report, "heldout_validation")
            self.assertFalse(receipt["all_passed"])
            failed = {gate["metric"] for gate in receipt["gates"] if not gate["passed"]}
            self.assertEqual(
                failed,
                {
                    "answerable_temporal_iou",
                    "evidence_sd_sdri_answerable",
                    "no_evidence_balanced_accuracy",
                    "no_evidence_auroc",
                },
            )


class DemoInputTest(unittest.TestCase):
    def test_relation_story_explains_both_required_roles(self) -> None:
        after = relation_requirement("after", "What follows the bell?")
        before = relation_requirement("before", "What precedes the bell?")
        first = relation_requirement("first", "Which is first?")
        self.assertIn("anchor", after)
        self.assertIn("cả âm mốc và âm trả lời", after)
        self.assertIn("ngay trước", before)
        self.assertIn("cả hai", first)
        self.assertIn(
            "ngay sau",
            relation_requirement(None, "What begins next after the bell?"),
        )

    def test_scene_inventory_is_chronological_and_display_safe(self) -> None:
        events = compact_scene_events(
            [
                {
                    "event_id": "b",
                    "label": "Water_tap_and_faucet",
                    "event_kind": "semantic",
                    "occurrence_index": 1,
                    "onset_seconds": 2.0,
                    "offset_seconds": 3.0,
                },
                {
                    "event_id": "a",
                    "label": "Dog_bark",
                    "event_kind": "nuisance",
                    "occurrence_index": 2,
                    "onset_seconds": 0.5,
                    "offset_seconds": 1.0,
                },
            ]
        )
        self.assertEqual([event["event_id"] for event in events], ["a", "b"])
        self.assertEqual(events[1]["sound"], "Water tap and faucet")

    def test_af3_scene_inventory_parser_fails_visible(self) -> None:
        valid = parse_scene_inventory(
            'prefix {"events":[{"order":1,"sound":"a bell"},'
            '{"order":2,"sound":"running water"}]} suffix'
        )
        self.assertEqual(valid["status"], "valid")
        self.assertEqual(valid["events"][1]["sound"], "running water")
        invalid = parse_scene_inventory(
            '{"events":[{"order":2,"sound":"wrong order"}]}'
        )
        self.assertEqual(invalid["status"], "invalid_schema")
        self.assertEqual(invalid["events"], [])

    def test_demo_and_experiment_queue_share_one_fail_closed_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "gpu.lock"
            self.assertTrue(exclusive_lock_available(lock_path))
            with lock_path.open("a+") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(exclusive_lock_available(lock_path))
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            self.assertTrue(exclusive_lock_available(lock_path))

    def test_demo_gpu_preflight_is_fail_closed(self) -> None:
        healthy = {
            "free_memory_mib_↑": 8_000,
            "utilization_percent_↓": 5,
            "temperature_celsius_↓": 70,
        }
        self.assertTrue(gpu_resource_ready(healthy, 7_000, 10))
        self.assertFalse(gpu_resource_ready(None, 7_000, 10))
        self.assertFalse(
            gpu_resource_ready({**healthy, "free_memory_mib_↑": 6_999}, 7_000, 10)
        )
        self.assertFalse(
            gpu_resource_ready({**healthy, "utilization_percent_↓": 11}, 7_000, 10)
        )

    def test_audio_window_resamples_pads_and_reports_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            signal = np.ones((4_000, 2), dtype=np.float32)
            signal[:, 1] *= 0.5
            sf.write(path, signal, 8_000, subtype="FLOAT")
            result, metadata = prepare_audio_window(
                path, 16_000, start_seconds=0.25, window_seconds=1.0
            )
            self.assertEqual(result.shape, (16_000,))
            self.assertEqual(metadata["original_num_channels"], 2)
            self.assertEqual(metadata["copied_input_samples"], 4_000)
            self.assertEqual(metadata["right_padding_samples"], 12_000)
            self.assertTrue(np.allclose(result[:3_900], 0.75, atol=1e-3))
            self.assertTrue(np.all(result[4_100:] == 0.0))

    def test_option_request_is_gold_free_and_supports_two_to_five(self) -> None:
        question, options = validate_request("  What follows? ", ["bell", "rain"])
        self.assertEqual(question, "What follows?")
        self.assertEqual(options, ("bell", "rain"))
        prompt = format_mc_prompt(question, options)
        self.assertIn("A. bell", prompt)
        self.assertIn("B. rain", prompt)
        self.assertNotIn("C.", prompt)
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_request("question", ["Bell", "bell"])
        with self.assertRaisesRegex(ValueError, "two and five"):
            validate_request("question", ["only one"])

    def test_frozen_examples_are_report_selected_and_manifest_fingerprint_bound(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "mixture.wav"
            sf.write(audio, np.zeros(800, dtype=np.float32), 8_000)
            manifest = root / "manifest.jsonl"
            rows = [
                {
                    "id": "not_selected",
                    "question": "Ignored?",
                    "answer_options": ["x", "y"],
                    "answer_option_index": 0,
                    "answer": "x",
                    "mixture_path": audio.name,
                },
                {
                    "id": "selected",
                    "question": "What follows?",
                    "answer_options": ["bell", "rain", "no_evidence"],
                    "answer_option_index": 1,
                    "answer": "rain",
                    "mixture_path": audio.name,
                    "relation": "after",
                    "no_evidence": False,
                    "split": "train",
                    "events": [
                        {
                            "event_id": "sem_0",
                            "label": "Dog_bark",
                            "event_kind": "semantic",
                            "occurrence_index": 1,
                            "onset_seconds": 0.25,
                            "offset_seconds": 0.75,
                        }
                    ],
                    "anchor_event_ids": ["sem_0"],
                    "answer_event_ids": ["sem_0"],
                    "anchor_intervals": [[0.25, 0.75]],
                    "answer_intervals": [[0.25, 0.75]],
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = root / "report.json"
            report.write_text(
                json.dumps({"audio_rendering": {"requested_item_ids": ["selected"]}}),
                encoding="utf-8",
            )
            health = {
                "evaluation_binding": {
                    "manifest": str(manifest),
                    "manifest_sha256": sha256_file(manifest),
                },
                "evaluation_report": {"path": str(report)},
            }
            examples = load_fingerprint_bound_examples(health)
            self.assertEqual([example["id"] for example in examples], ["selected"])
            self.assertEqual(examples[0]["answer"], "rain")
            self.assertEqual(examples[0]["mixture_path"], str(audio.resolve()))
            self.assertEqual(examples[0]["scene_events"][0]["sound"], "Dog bark")
            self.assertEqual(examples[0]["anchor_event_ids"], ["sem_0"])

            manifest.write_text(
                manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(DemoContractError, "manifest changed"):
                load_fingerprint_bound_examples(health)


class MicrofitRunnerTest(unittest.TestCase):
    def test_reproducible_command_enables_reachable_mixing_and_epoch_saves(
        self,
    ) -> None:
        args = parse_microfit_args([])
        commands = build_commands(args)
        train = commands["train"]
        self.assertEqual(
            train[train.index("--foundation-semantic-mixing-mode") + 1],
            "question_residual",
        )
        self.assertIn("--save-every-epoch", train)
        self.assertNotIn("--overwrite", train)
        self.assertNotIn("--no-render-audio", commands["evaluate"])
        self.assertEqual(commands["evaluate"].count("--render-item-id"), 4)
        self.assertEqual(args.max_steps, 1_536)
        parsed_train = parse_train_args(train[2:])
        self.assertEqual(parsed_train.selection_metric, "evidence_sd_sdr")
        self.assertEqual(parsed_train.max_steps, 1_536)

    def test_listening_packet_is_preselected_by_relation_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.jsonl"
            rows = [
                {"id": "z_after", "relation": "after", "no_evidence": False},
                {"id": "a_before", "relation": "before", "no_evidence": False},
                {"id": "c_first_pos", "relation": "first", "no_evidence": False},
                {"id": "b_first_neg", "relation": "first", "no_evidence": True},
                {"id": "extra", "relation": "after", "no_evidence": True},
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(
                select_listening_item_ids(manifest, 4),
                ["z_after", "a_before", "c_first_pos", "b_first_neg"],
            )
            self.assertEqual(select_listening_item_ids(manifest, 0), [])

    def test_gpu_preflight_rejects_busy_or_memory_starved_device(self) -> None:
        healthy = {
            "free_memory_mib_↑": 8_000,
            "utilization_percent_↓": 5,
            "temperature_celsius_↓": 70,
        }
        require_idle_gpu(healthy, 7_000, 10)
        with self.assertRaisesRegex(RuntimeError, "insufficient free"):
            require_idle_gpu({**healthy, "free_memory_mib_↑": 6_999}, 7_000, 10)
        with self.assertRaisesRegex(RuntimeError, "actively used"):
            require_idle_gpu({**healthy, "utilization_percent_↓": 11}, 7_000, 10)

    def test_only_a_passing_gate_emits_demo_authorization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            evaluation = root / "evaluation"
            passing = write_health_outcome(
                train, evaluation, {"all_passed": True, "gates": []}
            )
            self.assertIsNotNone(passing["health_receipt"])
            self.assertIsNone(passing["failed_health_audit"])
            self.assertTrue((train / "demo_health_receipt.json").is_file())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            evaluation = root / "evaluation"
            failed = write_health_outcome(
                train, evaluation, {"all_passed": False, "gates": []}
            )
            self.assertIsNone(failed["health_receipt"])
            self.assertIsNotNone(failed["failed_health_audit"])
            self.assertFalse((train / "demo_health_receipt.json").exists())
            self.assertTrue((evaluation / "failed_health_audit.json").is_file())

    def test_failure_analysis_prioritizes_semantic_before_temporal_when_stalled(
        self,
    ) -> None:
        training = {
            "history": [
                {"semantic_alignment": 0.40},
                {"semantic_alignment": 0.39},
            ]
        }
        evaluation = {
            "summary": {
                "answerable_temporal_iou": 0.2,
                "evidence_sd_sdri_answerable": -3.0,
                "no_evidence_balanced_accuracy": 0.95,
                "no_evidence_auroc": 0.98,
                "mean_no_evidence_retained_ratio": 0.04,
                "maximum_mixture_consistency_l1": 0.0,
            }
        }
        health = {
            "gates": [
                {"metric": "answerable_temporal_iou", "passed": False},
                {"metric": "evidence_sd_sdri_answerable", "passed": False},
            ]
        }
        analysis = analyze_microfit_result(training, evaluation, health)
        self.assertTrue(analysis["semantic_stalled_diagnostic"])
        self.assertIn(
            "semantic gradients",
            analysis["recommended_next_actions_in_order"][0],
        )
        self.assertAlmostEqual(analysis["semantic_target_cosine_gain_↑"], 0.01)

    def test_passing_analysis_moves_to_listening_then_heldout(self) -> None:
        analysis = analyze_microfit_result(
            {"history": [{"semantic_alignment": 0.2}]},
            {"summary": _passing_summary()},
            {"gates": [{"metric": "answerable_temporal_iou", "passed": True}]},
        )
        self.assertEqual(analysis["failed_health_metrics"], [])
        self.assertIn(
            "Streamlit listening",
            analysis["recommended_next_actions_in_order"][0],
        )


if __name__ == "__main__":
    unittest.main()
