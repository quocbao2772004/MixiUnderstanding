"""Tests for the durable receipt-driven QCES GPU queue."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mixi_understanding.scripts.run_qces_cee_dev_pilot import FORMAT as CEE_FORMAT
from mixi_understanding.scripts.run_qces_gpu_queue import (
    FORMAT,
    STAGES,
    Queue,
    cee_promotion,
    parse_args,
)


class DurableQueueTest(unittest.TestCase):
    def test_default_queue_stops_before_the_full_three_seed_matrix(self) -> None:
        args = parse_args([])
        self.assertEqual(args.stop_after, "ablation_screen")
        self.assertLessEqual(args.poll_seconds, 60)
        self.assertLess(
            STAGES.index("caption_baseline"), STAGES.index("ablation_screen")
        )

    def test_cee_promotion_requires_format_integrity_and_positive_decision(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.json"
            passed, _ = cee_promotion(path)
            self.assertFalse(passed)
            payload = {
                "format": CEE_FORMAT,
                "integrity": {"all_passed": True},
                "decision": {"promote_cee_to_full_seeded_run": False},
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(cee_promotion(path)[0])
            payload["decision"]["promote_cee_to_full_seeded_run"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(cee_promotion(path)[0])
            payload["integrity"]["all_passed"] = False
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(cee_promotion(path)[0])

    def test_idle_gate_needs_consecutive_samples_and_persists_arrowed_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = parse_args(
                [
                    "--status-path",
                    str(root / "status.json"),
                    "--log-path",
                    str(root / "queue.log"),
                    "--lock-path",
                    str(root / "queue.lock"),
                    "--poll-seconds",
                    "1",
                    "--idle-streak",
                    "2",
                ]
            )
            queue = Queue(args)
            queue.current_stage = "test_stage"
            samples = iter(
                [
                    {
                        "free_memory_mib_↑": 12_100,
                        "utilization_percent_↓": 0,
                        "temperature_celsius_↓": 40,
                    },
                    {
                        "free_memory_mib_↑": 8_000,
                        "utilization_percent_↓": 0,
                        "temperature_celsius_↓": 40,
                    },
                    {
                        "free_memory_mib_↑": 12_200,
                        "utilization_percent_↓": 1,
                        "temperature_celsius_↓": 41,
                    },
                    {
                        "free_memory_mib_↑": 12_300,
                        "utilization_percent_↓": 2,
                        "temperature_celsius_↓": 42,
                    },
                ]
            )

            def next_state() -> dict[str, int]:
                state = next(samples)
                queue.last_gpu = state
                return state

            with patch.object(queue, "gpu_state", side_effect=next_state), patch(
                "mixi_understanding.scripts.run_qces_gpu_queue.time.sleep"
            ) as sleep:
                queue.wait_idle(minimum_free=12_000, maximum_utilization=10)
            self.assertEqual(sleep.call_count, 3)
            status = json.loads(args.status_path.read_text("utf-8"))
            self.assertEqual(status["format"], FORMAT)
            self.assertEqual(status["current_stage"], "test_stage")
            self.assertEqual(status["last_gpu_sample"]["free_memory_mib_↑"], 12_300)
            self.assertEqual(
                status["metric_direction_legend"],
                {"↑": "higher is better", "↓": "lower is better"},
            )


if __name__ == "__main__":
    unittest.main()
