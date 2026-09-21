"""Tests for the dependency-free QCES shortcut audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mixi_understanding.scripts.evaluate_qces_shortcuts import (
    ANSWERABLE_RECALL,
    NO_EVIDENCE_ACCURACY,
    SEEN_KEY_RATE,
    UNION_MIOU,
    evaluate_manifest,
    intervals_to_mask,
    main,
    normalize_question,
)


class QCESShortcutAuditTest(unittest.TestCase):
    def test_question_normalization_keeps_label_tokens(self) -> None:
        self.assertEqual(
            normalize_question("  What  SOUND occurs after Engine knocking?  "),
            "what sound occurs after engine knocking?",
        )

    def test_interval_rasterization_uses_fixed_normalized_grid(self) -> None:
        self.assertEqual(intervals_to_mask([(1.0, 2.0)], 4.0, 8), (0, 0, 1, 1, 0, 0, 0, 0))
        self.assertEqual(intervals_to_mask([(0.1, 0.9)], 4.0, 4), (1, 0, 0, 0))

    @staticmethod
    def _records() -> list[dict]:
        records = []
        for scene_index in range(3):
            scene = f"scene_{scene_index}"
            records.extend(
                [
                    {
                        "schema_version": "qces_v4",
                        "id": f"{scene}_after",
                        "scene_id": scene,
                        "duration_seconds": 4.0,
                        "question": "What occurs after Buzz?",
                        "question_type": "temporal_after",
                        "relation": "after",
                        "no_evidence": False,
                        "anchor_intervals": [[0.0, 1.0]],
                        "answer_intervals": [[1.0, 2.0]],
                    },
                    {
                        "schema_version": "qces_v4",
                        "id": f"{scene}_before",
                        "scene_id": scene,
                        "duration_seconds": 4.0,
                        "question": "What occurs before Croak?",
                        "question_type": "temporal_before",
                        "relation": "before",
                        "no_evidence": False,
                        "anchor_intervals": [[3.0, 4.0]],
                        "answer_intervals": [[2.0, 3.0]],
                    },
                    {
                        "schema_version": "qces_v4",
                        "id": f"{scene}_negative",
                        "scene_id": scene,
                        "duration_seconds": 4.0,
                        "question": f"What occurs after Missing {scene_index}?",
                        # Deliberately answerability-leaking legacy value.  The
                        # evaluator must prefer relation="after" in qces_v4.
                        "question_type": "no_evidence_after",
                        "relation": "after",
                        "no_evidence": True,
                        "anchor_intervals": [],
                        "answer_intervals": [],
                    },
                ]
            )
        return records

    def test_loso_metrics_prefer_v4_relation_and_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in self._records()),
                encoding="utf-8",
            )
            first = evaluate_manifest(manifest, frames=16, bootstrap_samples=100, seed=7)
            second = evaluate_manifest(manifest, frames=16, bootstrap_samples=100, seed=7)
            self.assertEqual(first, second)

            relation_metrics = first["baselines"]["relation_question_type_only"]["metrics"]
            text_metrics = first["baselines"][
                "normalized_exact_question_text_only"
            ]["metrics"]
            # After contains one answerable and one no-evidence example per
            # scene.  Its 0.5 no-evidence prototype proves relation was used;
            # using question_type instead would make accuracy 1.0.
            self.assertAlmostEqual(relation_metrics[NO_EVIDENCE_ACCURACY], 2.0 / 3.0)
            self.assertAlmostEqual(relation_metrics[ANSWERABLE_RECALL], 0.5)
            self.assertAlmostEqual(relation_metrics[UNION_MIOU], 1.0)
            self.assertAlmostEqual(text_metrics[SEEN_KEY_RATE], 2.0 / 3.0)
            self.assertEqual(
                first["shortcut_diagnostics"]["same_scene_target_diversity"]
                ["answerable_union_diversity_one_minus_iou_↑"],
                1.0,
            )

            first_output, second_output = root / "first.json", root / "second.json"
            common = [
                "--manifest",
                str(manifest),
                "--frames",
                "16",
                "--bootstrap-samples",
                "25",
                "--seed",
                "9",
            ]
            main(common + ["--output", str(first_output)])
            main(common + ["--output", str(second_output)])
            self.assertEqual(first_output.read_bytes(), second_output.read_bytes())

    def test_legacy_question_type_fallback(self) -> None:
        records = self._records()
        for record in records:
            record.pop("relation")
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in records),
                encoding="utf-8",
            )
            report = evaluate_manifest(manifest, frames=8, bootstrap_samples=10, seed=1)
            relation_metrics = report["baselines"]["relation_question_type_only"][
                "metrics"
            ]
            self.assertEqual(relation_metrics[NO_EVIDENCE_ACCURACY], 1.0)


if __name__ == "__main__":
    unittest.main()
