"""Regression tests for overlap-aware anchor/answer temporal roles."""

from __future__ import annotations

import unittest

import torch

from mixi_understanding.qces.composer import (
    PromptComposition,
    temporal_evidence_probability,
)
from mixi_understanding.qces.config import (
    LEGACY_TEMPORAL_ROLE_MODE,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    QCESConfig,
)
from mixi_understanding.qces.losses import (
    LossWeights,
    QCESLoss,
    _balanced_binary_frame_error,
    _separator_mask_loss,
)
from mixi_understanding.qces.model import QCESModel, QCESOutput, load_qces_checkpoint
from mixi_understanding.qces.separators import SeparationOutput
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.train_qces import (
    parse_args,
    validate_refiner_training_args,
)


def tiny_config(
    temporal_role_mode: str = LEGACY_TEMPORAL_ROLE_MODE,
) -> QCESConfig:
    return QCESConfig(
        sample_rate=8_000,
        n_fft=64,
        hop_length=16,
        win_length=64,
        vocab_size=128,
        max_question_tokens=16,
        question_dim=32,
        audio_dim=32,
        condition_dim=16,
        attention_heads=4,
        question_layers=1,
        separator_channels=8,
        separator_layers=2,
        dropout=0.0,
        temporal_role_mode=temporal_role_mode,
    )


def only_temporal_weights() -> LossWeights:
    values = {name: 0.0 for name in LossWeights.__dataclass_fields__}
    values["temporal_binary_cross_entropy"] = 1.0
    values["temporal_role_dice"] = 1.0
    return LossWeights(**values)


class OverlapAwareRoleTest(unittest.TestCase):
    def test_sparse_temporal_foreground_is_not_outvoted_by_background(self) -> None:
        target = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])
        prediction = torch.zeros_like(target)
        error = (prediction - target).abs()

        balanced = _balanced_binary_frame_error(error, target)
        mask_loss = _separator_mask_loss(
            prediction,
            torch.zeros_like(target),
            torch.zeros_like(target),
            target,
            balance_temporal_classes=True,
        )

        # Positive and negative classes receive equal mass: the single missed
        # foreground frame costs 0.5, not the old unbalanced 1/5.
        self.assertAlmostEqual(float(balanced), 0.5)
        self.assertAlmostEqual(float(mask_loss), 0.5)

    def test_overlap_minimality_penalizes_only_false_positive_frames(self) -> None:
        def components(probability: torch.Tensor):
            logits = torch.zeros(1, 4, 3)
            composition = PromptComposition(
                semantic_condition=torch.zeros(1, 2),
                role_logits=logits,
                evidence_probability=probability,
                no_evidence_logit=torch.zeros(1),
                frame_features=torch.zeros(1, 4, 2),
                frame_hop_samples=1,
                temporal_role_mode=OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
            )
            mixture = torch.zeros(1, 4)
            output = QCESOutput(
                composition=composition,
                separation=SeparationOutput(
                    evidence=mixture,
                    residual=mixture,
                    mask=probability,
                    raw_mask=None,
                    raw_evidence=None,
                    mixture_error=torch.zeros(1),
                ),
            )
            batch = {
                "mixture": mixture,
                "evidence": mixture,
                "residual": mixture,
                "anchor_mask": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
                "answer_mask": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
                "no_evidence": torch.zeros(1),
            }
            _, measured = QCESLoss(fft_sizes=())(output, batch)
            return measured

        perfect = components(torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
        overwide = components(torch.tensor([[0.5, 1.0, 0.5, 0.5]]))

        self.assertAlmostEqual(float(perfect["minimality"]), 0.0)
        self.assertAlmostEqual(float(overwide["minimality"]), 0.5)

    def test_independent_roles_can_both_activate_and_use_probabilistic_union(
        self,
    ) -> None:
        logits = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]])
        composition = PromptComposition(
            semantic_condition=torch.zeros(1, 2),
            role_logits=logits,
            evidence_probability=temporal_evidence_probability(
                logits, OVERLAP_AWARE_TEMPORAL_ROLE_MODE
            ),
            no_evidence_logit=torch.zeros(1),
            frame_features=torch.zeros(1, 2, 2),
            frame_hop_samples=1,
            temporal_role_mode=OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
        )
        probabilities = composition.role_probabilities
        self.assertAlmostEqual(float(probabilities[0, 0, 1]), 0.5, places=7)
        self.assertAlmostEqual(float(probabilities[0, 0, 2]), 0.5, places=7)
        self.assertAlmostEqual(
            float(composition.evidence_probability[0, 0]), 0.75, places=7
        )
        expected = 1.0 - (1.0 - probabilities[..., 1]) * (1.0 - probabilities[..., 2])
        self.assertTrue(torch.equal(composition.evidence_probability, expected))

    def test_overlap_supervision_reaches_both_role_logits_with_finite_gradients(
        self,
    ) -> None:
        logits = torch.zeros(1, 4, 3, requires_grad=True)
        evidence_probability = temporal_evidence_probability(
            logits, OVERLAP_AWARE_TEMPORAL_ROLE_MODE
        )
        composition = PromptComposition(
            semantic_condition=torch.zeros(1, 2),
            role_logits=logits,
            evidence_probability=evidence_probability,
            no_evidence_logit=torch.zeros(1),
            frame_features=torch.zeros(1, 4, 2),
            frame_hop_samples=1,
            temporal_role_mode=OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
        )
        mixture = torch.zeros(1, 4)
        separation = SeparationOutput(
            evidence=mixture.clone(),
            residual=mixture.clone(),
            mask=evidence_probability,
            raw_mask=None,
            raw_evidence=None,
            mixture_error=torch.zeros(1),
        )
        output = QCESOutput(composition=composition, separation=separation)
        anchor_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        answer_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        batch = {
            "mixture": mixture,
            "evidence": mixture.clone(),
            "residual": mixture.clone(),
            "anchor_mask": anchor_mask,
            "answer_mask": answer_mask,
            "no_evidence": torch.zeros(1),
        }
        loss, components = QCESLoss(weights=only_temporal_weights(), fft_sizes=())(
            output, batch
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        # At overlap frames both positive binary labels pull their own logits up.
        self.assertLess(float(logits.grad[0, 1:3, 1].sum()), 0.0)
        self.assertLess(float(logits.grad[0, 1:3, 2].sum()), 0.0)
        self.assertEqual(float(components["temporal_cross_entropy"]), 0.0)
        self.assertGreater(float(components["temporal_binary_cross_entropy"]), 0.0)

    def test_head_shape_is_checkpoint_compatible_across_role_modes(self) -> None:
        torch.manual_seed(41)
        legacy = QCESModel(tiny_config())
        torch.manual_seed(41)
        overlap = QCESModel(tiny_config(OVERLAP_AWARE_TEMPORAL_ROLE_MODE))
        self.assertEqual(
            tuple(legacy.composer.role_head.weight.shape),
            tuple(overlap.composer.role_head.weight.shape),
        )
        self.assertTrue(
            torch.equal(
                legacy.composer.role_head.weight,
                overlap.composer.role_head.weight,
            )
        )

    def test_overlap_checkpoint_roundtrip_preserves_mode_and_output(self) -> None:
        torch.manual_seed(7)
        config = tiny_config(OVERLAP_AWARE_TEMPORAL_ROLE_MODE)
        model = QCESModel(config).eval()
        mixture = torch.randn(2, 512)
        tokens = StableHashTokenizer(128, 16).batch_encode(
            ["What follows the horn?", "Which sound is first?"]
        )
        with torch.inference_mode():
            expected = model(mixture, tokens.input_ids, tokens.attention_mask)
        payload = model.checkpoint_payload(backend="mask")
        restored = load_qces_checkpoint(payload).eval()
        with torch.inference_mode():
            actual = restored(mixture, tokens.input_ids, tokens.attention_mask)
        self.assertEqual(
            restored.config.temporal_role_mode,
            OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
        )
        self.assertTrue(
            torch.equal(
                expected.composition.role_logits,
                actual.composition.role_logits,
            )
        )
        self.assertTrue(
            torch.equal(
                expected.composition.evidence_probability,
                actual.composition.evidence_probability,
            )
        )
        self.assertTrue(torch.equal(expected.evidence, actual.evidence))

    def test_legacy_checkpoint_without_mode_has_exact_historical_behavior(self) -> None:
        torch.manual_seed(13)
        model = QCESModel(tiny_config()).eval()
        mixture = torch.randn(2, 512)
        tokens = StableHashTokenizer(128, 16).batch_encode(
            ["What follows the horn?", "Which sound is first?"]
        )
        with torch.inference_mode():
            expected = model(mixture, tokens.input_ids, tokens.attention_mask)
        payload = model.checkpoint_payload(backend="mask")
        payload["config"].pop("temporal_role_mode")
        restored = load_qces_checkpoint(payload).eval()
        with torch.inference_mode():
            actual = restored(mixture, tokens.input_ids, tokens.attention_mask)
        self.assertEqual(restored.config.temporal_role_mode, LEGACY_TEMPORAL_ROLE_MODE)
        historical_probability = (
            1.0 - actual.composition.role_logits.softmax(dim=-1)[..., 0]
        )
        self.assertTrue(
            torch.equal(
                actual.composition.evidence_probability,
                historical_probability,
            )
        )
        self.assertTrue(
            torch.equal(
                expected.composition.role_logits,
                actual.composition.role_logits,
            )
        )
        self.assertTrue(torch.equal(expected.evidence, actual.evidence))

    def test_union_refiner_accepts_overlap_mode_after_residual_head_audit(self) -> None:
        config = QCESConfig(
            temporal_role_mode=OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
            separator_aware_refiner=True,
        )
        self.assertTrue(config.separator_aware_refiner)
        args = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--backend",
                "audiosep",
                "--temporal-role-mode",
                OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
                "--separator-aware-refiner",
            ]
        )
        validate_refiner_training_args(args)


if __name__ == "__main__":
    unittest.main()
