"""Checkpoint-free tests for QCES-v5 temporal-only controls."""

from __future__ import annotations

import unittest

import torch

from mixi_understanding.scripts.evaluate_qces_v5_temporal_baselines import (
    ACTIVITY_MODE,
    ENERGY_MODE,
    ORACLE_MODE,
    PREDICTED_MODE,
    RANDOM_MODE,
    maximum_energy_compact_mask,
    paint_intervals,
    parse_args,
    random_compact_mask,
    temporal_iou,
    top_energy_activity_mask,
)


class TemporalMaskConstructionTest(unittest.TestCase):
    def test_random_mask_is_seeded_and_has_exact_budget(self) -> None:
        first = random_compact_mask(1000, 100, 2.25, 2026)
        second = random_compact_mask(1000, 100, 2.25, 2026)
        other = random_compact_mask(1000, 100, 2.25, 2027)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))
        self.assertEqual(int(first.sum()), 225)

    def test_maximum_energy_span_selects_the_loud_region(self) -> None:
        mixture = torch.zeros(1000)
        mixture[600:850] = 2.0
        mask = maximum_energy_compact_mask(mixture, 100, 2.0)

        self.assertEqual(int(mask.sum()), 200)
        self.assertGreater(float(mask[650:800].mean()), 0.99)

    def test_activity_mask_uses_fixed_noncontiguous_budget(self) -> None:
        mixture = torch.zeros(1000)
        mixture[100:200] = 1.0
        mixture[700:800] = 2.0
        mask = top_energy_activity_mask(
            mixture, sample_rate=100, budget_seconds=1.0, frame_seconds=0.1
        )

        self.assertEqual(int(mask.sum()), 100)
        self.assertGreater(float(mask[700:800].mean()), 0.99)

    def test_interval_painting_and_iou(self) -> None:
        mask = paint_intervals([[1.0, 2.0], [3.0, 4.0]], 100, 500)
        target = paint_intervals([[1.5, 2.0], [3.0, 3.5]], 100, 500)

        self.assertEqual(int(mask.sum()), 200)
        self.assertAlmostEqual(temporal_iou(mask, target), 0.5, places=6)


class TemporalBaselineCLIContractTest(unittest.TestCase):
    @staticmethod
    def required() -> list[str]:
        return ["--manifest", "val.jsonl", "--output-dir", "output"]

    def test_default_modes_and_frozen_budget(self) -> None:
        args = parse_args(self.required())
        self.assertEqual(args.budget_seconds, 2.25)
        self.assertEqual(
            args.modes, (RANDOM_MODE, ENERGY_MODE, ACTIVITY_MODE, ORACLE_MODE)
        )

    def test_predicted_mode_requires_an_external_report(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args([*self.required(), "--modes", PREDICTED_MODE])
        args = parse_args(
            [
                *self.required(),
                "--predicted-spans-report",
                "predictions.json",
            ]
        )
        self.assertIn(PREDICTED_MODE, args.modes)


if __name__ == "__main__":
    unittest.main()
