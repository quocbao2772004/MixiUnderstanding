"""Focused checkpoint-free tests for the AudioSep factorization evaluator."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mixi_understanding.qces.metrics import scale_invariant_sdr
from mixi_understanding.qces.composer import PromptComposition
from mixi_understanding.scripts.evaluate_qces_audiosep_factorization import (
    BASE_RENDERED_CONDITIONS,
    CLIP_NO_EVIDENCE_CONDITION,
    COMPOSER_BINARY_CONDITION,
    CONDITION_PROTOCOL,
    ENERGY_CONDITION,
    apply_frame_gate,
    apply_clip_no_evidence_gate,
    apply_separator_aware_refiner,
    binary_ranking_metrics,
    fit_validation_threshold,
    fit_validation_no_evidence_threshold,
    load_locked_validation_threshold,
    load_oracle_semantic_cache,
    linear_percentile,
    normalized_frame_rms,
    no_evidence_threshold_metrics,
    oracle_gate_validity_metrics,
    render_factorized_conditions,
    summarize_condition_items,
    threshold_metrics,
)


class _ConditionScaledSeparator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, batch):  # type: ignore[no-untyped-def]
        self.calls += 1
        mixture = batch["mixture"]
        scale = batch["condition"][:, :1, None]
        return {"waveform": mixture * scale}


class RankingMetricsTest(unittest.TestCase):
    def test_sd_sdr_tail_uses_declared_linear_interpolation(self) -> None:
        values = [-30.0, -20.0, -10.0, 0.0, 10.0]
        # (5 - 1) * .1 = .4, hence 60% of the minimum + 40% of the next.
        self.assertAlmostEqual(float(linear_percentile(values, 0.1)), -26.0)
        with self.assertRaisesRegex(ValueError, "within"):
            linear_percentile(values, 1.1)

        condition_items = [
            {
                "evidence_sd_sdr_db_↑": value,
                "evidence_si_sdr_db_↑": value,
            }
            for value in values
        ]
        summary = summarize_condition_items(condition_items)
        self.assertAlmostEqual(
            float(summary["evidence_sd_sdr_answerable_p10_db_↑"]), -26.0
        )
        self.assertEqual(
            summary["evidence_sd_sdr_answerable_minimum_db_↑"], -30.0
        )

    def test_auroc_average_precision_and_ties(self) -> None:
        metrics = binary_ranking_metrics(
            np.array([0.9, 0.8, 0.2, 0.1]),
            np.array([1, 0, 1, 0]),
        )
        self.assertAlmostEqual(float(metrics["auroc"]), 0.75)
        self.assertAlmostEqual(float(metrics["auprc"]), (1.0 + 2.0 / 3.0) / 2.0)

        tied = binary_ranking_metrics(
            np.full(4, 0.5), np.array([1, 0, 1, 0])
        )
        self.assertAlmostEqual(float(tied["auroc"]), 0.5)
        self.assertAlmostEqual(float(tied["auprc"]), 0.5)

    def test_threshold_fitting_is_validation_only(self) -> None:
        scores = [np.array([0.9, 0.8, 0.2, 0.1])]
        targets = [np.array([1, 1, 0, 0])]
        fitted = fit_validation_threshold(scores, targets, "val", grid_size=11)
        self.assertEqual(fitted["selection_split"], "val")
        self.assertAlmostEqual(fitted["metrics"]["answerable_macro_iou_↑"], 1.0)
        self.assertAlmostEqual(fitted["metrics"]["answerable_macro_f1_↑"], 1.0)
        with self.assertRaisesRegex(ValueError, "only on validation"):
            fit_validation_threshold(scores, targets, "test", grid_size=11)

        locked = threshold_metrics(scores, targets, fitted["threshold"])
        self.assertAlmostEqual(locked["answerable_micro_iou_↑"], 1.0)

    def test_non_validation_can_only_import_hashed_validation_fits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "validation_report.json"
            report.write_text(
                json.dumps(
                    {
                        "format": "qces_audiosep_factorization_v1",
                        "split": "val",
                        "provenance": {
                            "inputs": {
                                "learned_qces_checkpoint": {"sha256": "model-sha"}
                            }
                        },
                        "temporal_calibration": {
                            "validation_only_threshold_fit": {
                                "status": "fitted_on_validation_only",
                                "threshold": 0.046,
                            },
                            "separator_energy_validation_only_threshold_fit": {
                                "status": "fitted_on_validation_only",
                                "threshold": 0.076,
                            },
                        },
                        "no_evidence_calibration": {
                            "validation_only_threshold_fit": {
                                "status": "fitted_on_validation_only",
                                "threshold": 0.6,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            composer, identity = load_locked_validation_threshold(
                report, expected_checkpoint_sha256="model-sha"
            )
            energy, _ = load_locked_validation_threshold(
                report,
                fit_key="separator_energy_validation_only_threshold_fit",
                expected_checkpoint_sha256="model-sha",
            )
            clip_no_evidence, _ = load_locked_validation_threshold(
                report,
                fit_key="validation_only_threshold_fit",
                calibration_section="no_evidence_calibration",
                expected_checkpoint_sha256="model-sha",
            )
            self.assertAlmostEqual(composer, 0.046)
            self.assertAlmostEqual(energy, 0.076)
            self.assertAlmostEqual(clip_no_evidence, 0.6)
            self.assertEqual(
                identity["sha256"],
                hashlib.sha256(report.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(ValueError, "different QCES checkpoint"):
                load_locked_validation_threshold(
                    report, expected_checkpoint_sha256="another-model"
                )

            payload = json.loads(report.read_text(encoding="utf-8"))
            payload["split"] = "test"
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation report"):
                load_locked_validation_threshold(report)

    def test_clip_no_evidence_fit_is_safe_and_validation_only(self) -> None:
        scores = np.array([0.9, 0.8, 0.4, 0.1])
        targets = np.array([1, 1, 0, 0])
        metrics = no_evidence_threshold_metrics(scores, targets, threshold=0.5)
        self.assertEqual(metrics["balanced_accuracy_↑"], 1.0)
        self.assertEqual(metrics["no_evidence_recall_↑"], 1.0)
        self.assertEqual(metrics["answerable_false_silence_rate_↓"], 0.0)

        fitted = fit_validation_no_evidence_threshold(
            scores, targets, split="val", grid_size=11
        )
        self.assertEqual(fitted["selection_objective"], "balanced_accuracy_↑")
        self.assertEqual(fitted["metrics"]["balanced_accuracy_↑"], 1.0)
        with self.assertRaisesRegex(ValueError, "only on validation"):
            fit_validation_no_evidence_threshold(
                scores, targets, split="test", grid_size=11
            )


class FactorizedRenderingTest(unittest.TestCase):
    def test_two_by_two_ungated_and_temporal_only_controls(self) -> None:
        separator = _ConditionScaledSeparator()
        mixture = torch.ones(1, 4)
        learned_condition = torch.tensor([[0.5]])
        oracle_condition = torch.tensor([[1.0]])
        learned_gate = torch.tensor([[0.0, 1.0]])
        oracle_gate = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        rendered = render_factorized_conditions(
            separator,
            mixture,
            learned_condition,
            oracle_condition,
            learned_gate,
            oracle_gate,
        )
        self.assertEqual(separator.calls, 2)
        self.assertEqual(set(rendered), set(BASE_RENDERED_CONDITIONS))
        self.assertIn(ENERGY_CONDITION, CONDITION_PROTOCOL)
        self.assertIn(COMPOSER_BINARY_CONDITION, CONDITION_PROTOCOL)
        self.assertIn(CLIP_NO_EVIDENCE_CONDITION, CONDITION_PROTOCOL)
        self.assertTrue(
            torch.equal(
                rendered["learned_semantic__ungated"][0],
                torch.full((1, 4), 0.5),
            )
        )
        self.assertTrue(
            torch.equal(
                rendered["oracle_semantic__oracle_union_window"][0],
                oracle_gate,
            )
        )
        self.assertTrue(
            torch.equal(
                rendered["mixture__oracle_union_window"][0],
                oracle_gate,
            )
        )
        learned_sample_gate = rendered["mixture__learned_soft_temporal"][0]
        self.assertTrue(
            torch.allclose(
                rendered["learned_semantic__learned_soft_temporal"][0],
                learned_sample_gate * 0.5,
            )
        )
        for evidence, residual in rendered.values():
            self.assertTrue(torch.allclose(evidence + residual, mixture))

    def test_precomputed_stems_and_separator_aware_refiner_are_not_bypassed(self) -> None:
        class FixedRefiner(torch.nn.Module):
            def forward(self, mixture, raw, base):  # type: ignore[no-untyped-def]
                del mixture, raw
                logits = base.role_logits.clone()
                logits[..., 0] = 8.0
                logits[..., 1:] = -8.0
                probability = 1.0 - logits.softmax(dim=-1)[..., 0]
                return PromptComposition(
                    semantic_condition=base.semantic_condition,
                    role_logits=logits,
                    evidence_probability=probability,
                    no_evidence_logit=base.no_evidence_logit + 1.0,
                    frame_features=base.frame_features,
                    frame_hop_samples=base.frame_hop_samples,
                )

        separator = _ConditionScaledSeparator()
        mixture = torch.ones(1, 4)
        base = PromptComposition(
            semantic_condition=torch.tensor([[0.5]]),
            role_logits=torch.zeros(1, 2, 3),
            evidence_probability=torch.full((1, 2), 2.0 / 3.0),
            no_evidence_logit=torch.zeros(1),
            frame_features=torch.zeros(1, 2, 1),
            frame_hop_samples=2,
        )
        learned_raw = torch.full_like(mixture, 0.5)
        oracle_raw = torch.ones_like(mixture)
        refined = apply_separator_aware_refiner(
            SimpleNamespace(separator_aware_refiner=FixedRefiner()),
            mixture,
            learned_raw,
            base,
        )
        self.assertFalse(
            torch.equal(refined.evidence_probability, base.evidence_probability)
        )
        rendered = render_factorized_conditions(
            separator,
            mixture,
            base.semantic_condition,
            torch.ones_like(base.semantic_condition),
            refined.evidence_probability,
            torch.ones_like(mixture),
            learned_raw=learned_raw,
            oracle_raw=oracle_raw,
        )
        self.assertEqual(separator.calls, 0)
        actual = rendered["learned_semantic__learned_soft_temporal"][0]
        self.assertLess(float(actual.abs().max()), 1e-5)

    def test_normalized_raw_separator_rms_and_energy_gate(self) -> None:
        waveform = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
        energy = normalized_frame_rms(waveform, frames=2)
        self.assertTrue(torch.allclose(energy, torch.tensor([[0.0, 1.0]])))
        gated = apply_frame_gate(waveform, energy >= 0.5)
        self.assertEqual(gated.shape, waveform.shape)
        self.assertAlmostEqual(float(gated[0, 0]), 0.0)
        self.assertAlmostEqual(float(gated[0, -1]), 4.0)

        silent = normalized_frame_rms(torch.zeros(2, 8), frames=4)
        self.assertTrue(torch.equal(silent, torch.zeros_like(silent)))

    def test_clip_no_evidence_gate_silences_only_predicted_negatives(self) -> None:
        waveform = torch.ones(2, 8)
        probability = torch.tensor([0.8, 0.2])
        gated = apply_clip_no_evidence_gate(waveform, probability, threshold=0.5)
        self.assertTrue(torch.equal(gated[0], torch.zeros(8)))
        self.assertTrue(torch.equal(gated[1], torch.ones(8)))

    def test_oracle_gate_reports_overlap_contamination(self) -> None:
        target = torch.tensor([1.0, -1.0, 0.0, 0.0])
        residual = torch.tensor([0.5, -0.5, 2.0, -2.0])
        mixture = target + residual
        gate = torch.tensor([1.0, 1.0, 0.0, 0.0])
        metrics = oracle_gate_validity_metrics(mixture, target, residual, gate)

        self.assertAlmostEqual(
            metrics["oracle_window_contamination_to_target_l1_ratio_↓"], 0.5
        )
        self.assertAlmostEqual(metrics["target_outside_oracle_window_l1_↓"], 0.0)
        expected = float(
            scale_invariant_sdr((mixture * gate)[None], target[None])[0]
        )
        self.assertAlmostEqual(
            metrics["target_vs_oracle_gated_mixture_si_sdr_db_↑"], expected
        )


class SemanticCacheIdentityTest(unittest.TestCase):
    def test_cache_must_be_manifest_checkpoint_and_split_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "qces_val.jsonl"
            manifest.write_text('{"id":"val_0"}\n', encoding="utf-8")
            checkpoint = root / "audiosep.bin"
            checkpoint.write_bytes(b"audiosep")
            cache = root / "val.pt"
            manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
            base = {
                "format": "qces_audiosep_semantic_targets_v1",
                "schema_version": "qces_v4",
                "prompt_source": "evidence_role_event_labels_or_absent_label",
                "manifest_sha256": manifest_hash,
                "audiosep_checkpoint": str(checkpoint.resolve()),
                "prompts": {"val_0": "horn and bell"},
                "targets": {"val_0": torch.ones(3)},
            }
            torch.save(base, cache)
            targets, identity = load_oracle_semantic_cache(
                cache,
                manifest,
                ["val_0"],
                3,
                checkpoint,
                {"val_0": "horn and bell"},
            )
            self.assertEqual(set(targets), {"val_0"})
            self.assertEqual(identity["manifest_sha256"], manifest_hash)

            with_extra = dict(base)
            with_extra["targets"] = {
                "val_0": torch.ones(3),
                "test_0": torch.ones(3),
            }
            torch.save(with_extra, cache)
            with self.assertRaisesRegex(ValueError, "split-exact"):
                load_oracle_semantic_cache(
                    cache,
                    manifest,
                    ["val_0"],
                    3,
                    checkpoint,
                    {"val_0": "horn and bell"},
                )

            other_checkpoint = root / "other.bin"
            other_checkpoint.write_bytes(b"other")
            torch.save(base, cache)
            with self.assertRaisesRegex(ValueError, "different AudioSep"):
                load_oracle_semantic_cache(
                    cache,
                    manifest,
                    ["val_0"],
                    3,
                    other_checkpoint,
                    {"val_0": "horn and bell"},
                )


if __name__ == "__main__":
    unittest.main()
