"""CPU-only tests for the frozen BEATs event-presence audit."""

from __future__ import annotations

import unittest

import torch

from mixi_understanding.qces.beats_auditor import (
    EXPECTED_CLASS_COUNT,
    normalize_event_label,
    resolve_qces_label_indices,
    score_qces_streams,
    summarize_beats_items,
)
from mixi_understanding.scripts.evaluate_qces_beats_oracle import (
    build_gates,
    cluster_bootstrap_interval,
    summarize_oracle_items,
)


class BEATsAuditorTest(unittest.TestCase):
    def _labels(self) -> tuple[dict[int, str], dict[str, str]]:
        indices = {index: f"/m/{index:04d}" for index in range(EXPECTED_CLASS_COUNT)}
        names = {mid: f"unused class {index}" for index, mid in indices.items()}
        names[indices[4]] = "Coin (dropping)"
        names[indices[9]] = "Sawing"
        names[indices[17]] = "Cymbal"
        return indices, names

    def test_normalization_and_explicit_crash_cymbal_alias(self) -> None:
        self.assertEqual(normalize_event_label("Coin_(dropping)"), "coin dropping")
        self.assertEqual(
            normalize_event_label("Chewing_and_mastication"),
            normalize_event_label("Chewing, mastication"),
        )
        indices, names = self._labels()
        result = resolve_qces_label_indices(
            {"Coin_(dropping)", "Sawing", "Crash_cymbal"}, indices, names
        )
        self.assertEqual(
            result,
            {"Coin_(dropping)": 4, "Crash_cymbal": 17, "Sawing": 9},
        )

    def test_mapping_fails_on_unknown_or_ambiguous_label(self) -> None:
        indices, names = self._labels()
        with self.assertRaisesRegex(ValueError, "maps to 0"):
            resolve_qces_label_indices({"Not_in_AudioSet"}, indices, names)
        names[indices[5]] = "Sawing"
        with self.assertRaisesRegex(ValueError, "maps to 2"):
            resolve_qces_label_indices({"Sawing"}, indices, names)

    def test_stream_score_uses_weakest_required_role_and_complement(self) -> None:
        probabilities = torch.zeros(5, EXPECTED_CLASS_COUNT)
        mapping = {"Coin_(dropping)": 4, "Sawing": 9, "Wind_chime": 12}
        # X, E*, R*, E, R. Coin is deliberately stronger than Sawing so the
        # role-completeness reduction must select Sawing.
        probabilities[:, 4] = torch.tensor([0.8, 0.9, 0.2, 0.7, 0.3])
        probabilities[:, 9] = torch.tensor([0.6, 0.85, 0.1, 0.55, 0.2])
        probabilities[:, 12] = torch.tensor([0.7, 0.1, 0.8, 0.25, 0.75])
        item = score_qces_streams(
            probabilities,
            ["Coin_(dropping)", "Sawing"],
            ["Wind_chime"],
            mapping,
            ["Wind_chime", "Sawing"],
        )
        self.assertAlmostEqual(item["required_probability_mixture"], 0.6)
        self.assertAlmostEqual(item["predicted_required_probability_evidence"], 0.55)
        self.assertAlmostEqual(item["predicted_required_probability_residual"], 0.2)
        self.assertAlmostEqual(item["predicted_evidence_residual_contrast"], 0.35)
        self.assertAlmostEqual(item["oracle_excluded_suppression_delta"], 0.6)
        self.assertAlmostEqual(item["predicted_excluded_suppression_delta"], 0.45)
        self.assertFalse(item["class_separable_for_event_presence"])
        self.assertEqual(item["required_labels_present_in_oracle_residual"], ["Sawing"])

        separable = score_qces_streams(
            probabilities,
            ["Coin_(dropping)", "Sawing"],
            ["Wind_chime"],
            mapping,
            ["Wind_chime"],
        )
        self.assertTrue(separable["class_separable_for_event_presence"])
        scoped = summarize_beats_items([item, separable])
        self.assertEqual(
            scoped["class_presence_scope"]["class_separable"]["records"], 1
        )
        self.assertEqual(
            scoped["class_presence_scope"]["same_required_class_present_in_residual"][
                "records"
            ],
            1,
        )

    def test_summary_has_explicit_directions_and_skips_absent_excluded_metric(
        self,
    ) -> None:
        probabilities = torch.zeros(5, EXPECTED_CLASS_COUNT)
        probabilities[:, 1] = torch.tensor([0.5, 0.8, 0.1, 0.7, 0.2])
        item = score_qces_streams(probabilities, ["Bark"], [], {"Bark": 1})
        summary = summarize_beats_items([item, item])
        self.assertEqual(summary["answerable_records_↑"], 2)
        self.assertIn(
            "predicted_evidence_residual_contrast_↑",
            summary["summary_with_directions"],
        )
        self.assertNotIn(
            "predicted_excluded_probability_evidence_↓",
            summary["summary_with_directions"],
        )

    def test_oracle_summary_and_cluster_bootstrap_preserve_family_unit(self) -> None:
        base = {
            "required_probability_mixture": 0.5,
            "oracle_required_probability_evidence": 0.8,
            "oracle_required_probability_residual": 0.1,
            "oracle_evidence_residual_contrast": 0.7,
            "oracle_vs_mixture_sufficiency_delta": 0.3,
            "mixture_vs_oracle_residual_necessity_delta": 0.4,
            "oracle_excluded_probability_evidence": 0.1,
            "oracle_excluded_suppression_delta": 0.5,
        }
        items = [
            {**base, "scene_family_id": "family_a"},
            {**base, "scene_family_id": "family_a"},
            {
                **base,
                "scene_family_id": "family_b",
                "oracle_evidence_residual_contrast": 0.5,
            },
        ]
        summary = summarize_oracle_items(items)
        self.assertEqual(summary["answerable_records_↑"], 3)
        self.assertIn(
            "oracle_required_probability_residual_↓",
            summary["summary_with_directions"],
        )
        interval = cluster_bootstrap_interval(
            items,
            "oracle_evidence_residual_contrast",
            cluster_key="scene_family_id",
            samples=200,
            seed=7,
        )
        self.assertEqual(interval["clusters_↑"], 2)
        self.assertGreater(interval["lower_95"], 0.0)

    def test_oracle_gate_requires_family_count_and_positive_lower_bounds(self) -> None:
        positive = {"lower_95": 0.01}
        gates = build_gates(30, 30, positive, positive)
        self.assertTrue(all(gate["passed"] for gate in gates))
        failed = build_gates(29, 30, {"lower_95": 0.0}, positive)
        self.assertFalse(failed[0]["passed"])
        self.assertFalse(failed[1]["passed"])


if __name__ == "__main__":
    unittest.main()
