"""Tests for paired QCES-v5 scene-family bootstrap comparisons."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from mixi_understanding.scripts.compare_qces_v5_reports import (
    FORMAT,
    compare_reports,
    main,
)


def item(
    item_id: str,
    family: str,
    *,
    evidence_sd_sdri: float | None,
    residual_l1: float,
    same_role_label: bool | None = None,
    primary_counterfactual_probe: bool = False,
    variant_id: str = "base",
    no_evidence: bool = False,
) -> dict[str, object]:
    return {
        "id": item_id,
        "scene_family_id": family,
        "scene_id": f"scene_{family}_{variant_id}",
        "variant_id": variant_id,
        "counterfactual_group_id": f"{family}:primary",
        "evaluation_axis": "development",
        "primary_counterfactual_probe": primary_counterfactual_probe,
        "same_role_label": same_role_label,
        "same_label_repeat": same_role_label is True,
        "no_evidence": no_evidence,
        "evidence_sd_sdri": evidence_sd_sdri,
        "residual_l1": residual_l1,
    }


def report(items: list[dict[str, object]]) -> dict[str, object]:
    return {
        "format": "qces_question_swap_eval_v1",
        "schema_version": "qces_v5_scene_event_derived_v1",
        "items": items,
    }


class CompareQCESV5ReportsTest(unittest.TestCase):
    def test_family_macro_delta_ci_and_lower_is_better_orientation(self) -> None:
        reference = report(
            [
                item("a0", "family_a", evidence_sd_sdri=0.0, residual_l1=4.0),
                item("a1", "family_a", evidence_sd_sdri=0.0, residual_l1=4.0),
                item("b0", "family_b", evidence_sd_sdri=0.0, residual_l1=3.0),
            ]
        )
        candidate = report(
            [
                item("a0", "family_a", evidence_sd_sdri=2.0, residual_l1=2.0),
                item("a1", "family_a", evidence_sd_sdri=2.0, residual_l1=2.0),
                item("b0", "family_b", evidence_sd_sdri=4.0, residual_l1=4.0),
            ]
        )
        first = compare_reports(
            reference,
            candidate,
            reference_name="union",
            candidate_name="dual",
            metrics=("evidence_sd_sdri", "residual_l1"),
            bootstrap_samples=2_000,
            confidence_level=0.95,
            seed=17,
        )
        second = compare_reports(
            reference,
            candidate,
            reference_name="union",
            candidate_name="dual",
            metrics=("residual_l1", "evidence_sd_sdri"),
            bootstrap_samples=2_000,
            confidence_level=0.95,
            seed=17,
        )

        higher = first["slices"]["all"]["metrics"]["evidence_sd_sdri_↑"]
        lower = first["slices"]["all"]["metrics"]["residual_l1_↓"]
        self.assertEqual(higher["oriented_delta_positive_means_candidate_better"], 3.0)
        self.assertEqual(lower["raw_candidate_minus_reference"], -0.5)
        self.assertEqual(lower["oriented_delta_positive_means_candidate_better"], 0.5)
        self.assertEqual(higher["confidence_interval"]["lower"], 2.0)
        self.assertEqual(higher["confidence_interval"]["upper"], 4.0)
        self.assertEqual(lower["confidence_interval"]["lower"], -1.0)
        self.assertEqual(lower["confidence_interval"]["upper"], 2.0)
        self.assertEqual(
            higher,
            second["slices"]["all"]["metrics"]["evidence_sd_sdri_↑"],
        )

    def test_same_label_and_counterfactual_slices_are_exposed(self) -> None:
        common = [
            item(
                "same",
                "family_a",
                evidence_sd_sdri=1.0,
                residual_l1=1.0,
                same_role_label=True,
                primary_counterfactual_probe=True,
            ),
            item(
                "different",
                "family_b",
                evidence_sd_sdri=2.0,
                residual_l1=2.0,
                same_role_label=False,
                variant_id="order_swap",
            ),
        ]
        result = compare_reports(
            report(common),
            report([dict(value) for value in common]),
            reference_name="union",
            candidate_name="dual",
            metrics=("evidence_sd_sdri",),
            bootstrap_samples=20,
        )
        self.assertIn("same_role_label", result["slices"])
        self.assertIn("different_role_label", result["slices"])
        self.assertIn("same_label_repeat", result["slices"])
        self.assertIn("primary_counterfactual_probe", result["slices"])
        self.assertIn("counterfactual_variants", result["slices"])
        self.assertEqual(result["slices"]["same_role_label"]["paired_items"], 1)

    def test_item_set_and_metadata_mismatches_are_not_silent(self) -> None:
        reference_item = item(
            "shared", "family_a", evidence_sd_sdri=1.0, residual_l1=1.0
        )
        reference = report(
            [
                reference_item,
                item("only_ref", "family_b", evidence_sd_sdri=1.0, residual_l1=1.0),
            ]
        )
        candidate = report([dict(reference_item)])
        with self.assertRaisesRegex(ValueError, "item-ID sets differ"):
            compare_reports(
                reference,
                candidate,
                reference_name="union",
                candidate_name="dual",
                metrics=("evidence_sd_sdri",),
                bootstrap_samples=10,
            )
        partial = compare_reports(
            reference,
            candidate,
            reference_name="union",
            candidate_name="dual",
            metrics=("evidence_sd_sdri",),
            bootstrap_samples=10,
            allow_partial_overlap=True,
        )
        self.assertFalse(partial["pairing"]["exact_item_set_match"])
        self.assertEqual(partial["pairing"]["shared_items"], 1)

        mismatched = dict(reference_item)
        mismatched["scene_family_id"] = "family_wrong"
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            compare_reports(
                report([reference_item]),
                report([mismatched]),
                reference_name="union",
                candidate_name="dual",
                metrics=("evidence_sd_sdri",),
                bootstrap_samples=10,
            )

    def test_cli_writes_provenance_and_arrowed_metrics(self) -> None:
        values = [item("one", "family_a", evidence_sd_sdri=1.0, residual_l1=2.0)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "union.json"
            candidate = root / "dual.json"
            output = root / "comparison.json"
            reference.write_text(json.dumps(report(values)), encoding="utf-8")
            candidate.write_text(json.dumps(report(values)), encoding="utf-8")
            with redirect_stdout(StringIO()):
                main(
                    [
                        "--reference-report",
                        str(reference),
                        "--candidate-report",
                        str(candidate),
                        "--output",
                        str(output),
                        "--metric",
                        "evidence_sd_sdri",
                        "--bootstrap-samples",
                        "10",
                    ]
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(payload["format"], FORMAT)
        self.assertIn("evidence_sd_sdri_↑", payload["requested_metrics"])
        self.assertEqual(len(payload["source_reports"]["reference"]["sha256"]), 64)
        self.assertEqual(
            payload["comparison"]["positive_oriented_delta_means"],
            "dual is better",
        )


if __name__ == "__main__":
    unittest.main()
