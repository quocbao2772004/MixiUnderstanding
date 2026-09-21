"""Tests for the optional one-call AudioSep temporal refiner."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from mixi_understanding.qces.config import (
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    QCESConfig,
)
from mixi_understanding.qces.composer import (
    PromptComposition,
    temporal_evidence_probability,
)
from mixi_understanding.qces.model import QCESModel, load_qces_checkpoint
from mixi_understanding.qces.separators import (
    AudioSepConditionedAdapter,
    SeparatorAwareTemporalRefiner,
)
from mixi_understanding.qces.signal import qces_smooth_1d
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.train_qces import (
    separator_aware_refiner_provenance,
)


def refiner_config(
    enabled: bool = True,
    mode: str = "full",
    temporal_role_mode: str = "exclusive_softmax",
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
        separator_aware_refiner=enabled,
        separator_aware_refiner_mode=mode,
        temporal_role_mode=temporal_role_mode,
    )


class CountingSeparator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.75))
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        # Keep a differentiable condition path while making the waveform
        # numerically stable and simple enough for exact comparisons.
        condition_zero = batch["condition"].sum(dim=-1, keepdim=True) * 0.0
        waveform = batch["mixture"][:, 0] * self.gain + condition_zero
        return {"waveform": waveform[:, None]}


def audiosep_model(config: QCESConfig) -> tuple[QCESModel, CountingSeparator]:
    separator = CountingSeparator()
    refiner = SeparatorAwareTemporalRefiner(config) if config.separator_aware_refiner else None
    adapter = AudioSepConditionedAdapter(
        separator,
        condition_dim=config.condition_dim,
        freeze_separator=True,
        separator_aware_refiner=refiner,
    )
    return QCESModel(config, separator=adapter), separator


class SeparatorAwareRefinerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(31)
        self.mixture = torch.randn(2, 512) * 0.05
        self.tokens = StableHashTokenizer(128, 16).batch_encode(
            ["What follows the horn?", "Which event is first?"]
        )

    def _forward(self, model: QCESModel):
        return model(
            self.mixture,
            self.tokens.input_ids,
            self.tokens.attention_mask,
        )

    def test_zero_init_is_exactly_base_gate_and_calls_separator_once(self) -> None:
        base_config = refiner_config(False)
        refined_config = refiner_config(True)
        base_model, base_separator = audiosep_model(base_config)
        refined_model, refined_separator = audiosep_model(refined_config)
        refined_model.composer.load_state_dict(base_model.composer.state_dict())
        base_model.eval()
        refined_model.eval()

        with torch.inference_mode():
            base = self._forward(base_model)
            refined = self._forward(refined_model)

        self.assertEqual(base_separator.calls, 1)
        self.assertEqual(refined_separator.calls, 1)
        self.assertTrue(
            torch.equal(
                base.composition.role_logits, refined.composition.role_logits
            )
        )
        self.assertTrue(
            torch.equal(
                base.composition.no_evidence_logit,
                refined.composition.no_evidence_logit,
            )
        )
        self.assertTrue(torch.equal(base.evidence, refined.evidence))
        self.assertTrue(torch.equal(base.residual, refined.residual))

    def test_relative_energy_zero_init_is_base_gate_and_calls_once(self) -> None:
        base_model, _ = audiosep_model(refiner_config(False))
        energy_model, energy_separator = audiosep_model(
            refiner_config(True, "relative_energy")
        )
        energy_model.composer.load_state_dict(base_model.composer.state_dict())
        base_model.eval()
        energy_model.eval()
        with torch.inference_mode():
            base = self._forward(base_model)
            refined = self._forward(energy_model)
        self.assertEqual(energy_separator.calls, 1)
        self.assertTrue(torch.equal(base.composition.role_logits, refined.composition.role_logits))
        self.assertTrue(torch.equal(base.evidence, refined.evidence))
        self.assertTrue(torch.equal(base.residual, refined.residual))

    def test_overlap_aware_refiner_preserves_probabilistic_union_at_zero_init(
        self,
    ) -> None:
        temporal_mode = OVERLAP_AWARE_TEMPORAL_ROLE_MODE
        base_model, _ = audiosep_model(
            refiner_config(False, temporal_role_mode=temporal_mode)
        )
        refined_model, refined_separator = audiosep_model(
            refiner_config(
                True,
                "relative_energy",
                temporal_role_mode=temporal_mode,
            )
        )
        refined_model.composer.load_state_dict(base_model.composer.state_dict())
        base_model.eval()
        refined_model.eval()
        with torch.inference_mode():
            base = self._forward(base_model)
            refined = self._forward(refined_model)
        self.assertEqual(refined_separator.calls, 1)
        self.assertEqual(
            refined.composition.temporal_role_mode,
            OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
        )
        self.assertTrue(
            torch.equal(base.composition.role_logits, refined.composition.role_logits)
        )
        expected_union = temporal_evidence_probability(
            refined.composition.role_logits,
            OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
        )
        self.assertTrue(
            torch.equal(refined.composition.evidence_probability, expected_union)
        )
        self.assertTrue(torch.equal(base.evidence, refined.evidence))
        self.assertTrue(torch.equal(base.residual, refined.residual))

    def test_relative_energy_is_low_capacity_and_has_no_spectrogram_encoder(self) -> None:
        full = SeparatorAwareTemporalRefiner(refiner_config(True, "full"))
        energy = SeparatorAwareTemporalRefiner(
            refiner_config(True, "relative_energy")
        )
        full_parameters = sum(parameter.numel() for parameter in full.parameters())
        energy_parameters = sum(
            parameter.numel() for parameter in energy.parameters()
        )
        self.assertFalse(hasattr(energy, "stem_encoder"))
        self.assertFalse(hasattr(energy, "fusion"))
        self.assertFalse(hasattr(energy, "window"))
        self.assertTrue(hasattr(energy, "energy_fusion"))
        self.assertLess(energy_parameters, full_parameters // 50)
        self.assertEqual(energy_parameters, 404)

        model, _ = audiosep_model(refiner_config(True, "relative_energy"))
        provenance = separator_aware_refiner_provenance(model)
        self.assertEqual(provenance["mode"], "relative_energy")
        self.assertEqual(provenance["parameter_count"], 404)
        self.assertEqual(provenance["trainable_parameter_count"], 404)
        self.assertIn("no spectrogram encoder", provenance["input_features"])

    def test_relative_energy_does_not_consume_base_frame_features(self) -> None:
        refiner = SeparatorAwareTemporalRefiner(
            refiner_config(True, "relative_energy")
        ).eval()
        with torch.no_grad():
            refiner.role_delta_head.weight[0].fill_(0.03)
            refiner.role_delta_head.weight[1].fill_(-0.02)
            refiner.role_delta_head.weight[2].fill_(0.01)
            refiner.no_evidence_delta_head.weight.fill_(0.04)
        base_model, _ = audiosep_model(refiner_config(False))
        base_model.eval()
        with torch.inference_mode():
            base = self._forward(base_model).composition
        changed_frames = PromptComposition(
            semantic_condition=base.semantic_condition,
            role_logits=base.role_logits,
            evidence_probability=base.evidence_probability,
            no_evidence_logit=base.no_evidence_logit,
            frame_features=base.frame_features + 1000.0,
            frame_hop_samples=base.frame_hop_samples,
        )
        raw = self.mixture * 0.6
        with torch.inference_mode():
            expected = refiner(self.mixture, raw, base)
            actual = refiner(self.mixture, raw, changed_frames)
        self.assertTrue(torch.equal(expected.role_logits, actual.role_logits))
        self.assertTrue(
            torch.equal(expected.no_evidence_logit, actual.no_evidence_logit)
        )

    def test_shapes_complement_and_gradients_exclude_frozen_audiosep(self) -> None:
        model, separator = audiosep_model(refiner_config(True))
        output = self._forward(model)
        target = torch.zeros_like(output.evidence)
        target[:, 100:260] = self.mixture[:, 100:260]
        loss = (output.evidence - target).abs().mean()
        loss.backward()

        refiner = model.separator.separator_aware_refiner
        assert refiner is not None
        self.assertEqual(separator.calls, 1)
        self.assertEqual(output.evidence.shape, self.mixture.shape)
        self.assertEqual(output.residual.shape, self.mixture.shape)
        self.assertTrue(torch.equal(output.evidence + output.residual, self.mixture))
        self.assertLess(float(output.separation.mixture_error.max()), 1e-8)
        self.assertIsNone(separator.gain.grad)
        self.assertIsNotNone(refiner.role_delta_head.weight.grad)
        self.assertGreater(
            float(refiner.role_delta_head.weight.grad.abs().sum()), 0.0
        )

    def test_relative_energy_gradients_reach_trunk_after_zero_head_update(self) -> None:
        model, separator = audiosep_model(
            refiner_config(True, "relative_energy")
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=3e-4,
        )
        target = torch.zeros_like(self.mixture)
        target[:, 100:260] = self.mixture[:, 100:260]
        refiner = model.separator.separator_aware_refiner
        assert refiner is not None
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = self._forward(model)
            (output.evidence - target).abs().mean().backward()
            head_gradient = refiner.role_delta_head.weight.grad
            trunk_gradient = refiner.energy_fusion[0].weight.grad
            self.assertIsNotNone(head_gradient)
            self.assertGreater(float(head_gradient.abs().sum()), 0.0)
            self.assertIsNotNone(trunk_gradient)
            if step == 0:
                self.assertEqual(float(trunk_gradient.abs().sum()), 0.0)
            else:
                self.assertGreater(float(trunk_gradient.abs().sum()), 0.0)
            self.assertIsNone(separator.gain.grad)
            optimizer.step()

    def test_checkpoint_saves_and_loads_refiner(self) -> None:
        config = refiner_config(True)
        model, _ = audiosep_model(config)
        model.eval()
        refiner = model.separator.separator_aware_refiner
        assert refiner is not None
        with torch.no_grad():
            refiner.role_delta_head.weight.fill_(0.013)
            refiner.no_evidence_delta_head.bias.fill_(-0.2)
        payload = model.checkpoint_payload(backend="audiosep")
        self.assertIn("separator_aware_refiner_state_dict", payload)

        replacement, _ = audiosep_model(config)
        with patch(
            "mixi_understanding.qces.model."
            "AudioSepConditionedAdapter.from_repository",
            return_value=replacement.separator,
        ):
            restored = load_qces_checkpoint(
                payload,
                audiosep_repository_root="repo",
                audiosep_config_path="config",
                audiosep_checkpoint_path="separator",
            ).eval()
        with torch.inference_mode():
            expected = self._forward(model)
            actual = self._forward(restored)
        self.assertTrue(torch.equal(expected.evidence, actual.evidence))
        self.assertTrue(
            torch.equal(
                expected.composition.no_evidence_logit,
                actual.composition.no_evidence_logit,
            )
        )

    def test_relative_energy_checkpoint_round_trip_and_mode_mismatch(self) -> None:
        config = refiner_config(True, "relative_energy")
        model, _ = audiosep_model(config)
        model.eval()
        refiner = model.separator.separator_aware_refiner
        assert refiner is not None
        with torch.no_grad():
            refiner.role_delta_head.weight.fill_(0.021)
            refiner.no_evidence_delta_head.bias.fill_(-0.4)
        payload = model.checkpoint_payload(backend="audiosep")
        self.assertEqual(
            payload["config"]["separator_aware_refiner_mode"],
            "relative_energy",
        )
        replacement, _ = audiosep_model(config)
        with patch(
            "mixi_understanding.qces.model."
            "AudioSepConditionedAdapter.from_repository",
            return_value=replacement.separator,
        ):
            restored = load_qces_checkpoint(
                payload,
                audiosep_repository_root="repo",
                audiosep_config_path="config",
                audiosep_checkpoint_path="separator",
            ).eval()
        with torch.inference_mode():
            expected = self._forward(model)
            actual = self._forward(restored)
        self.assertTrue(torch.equal(expected.evidence, actual.evidence))

        incompatible = dict(payload)
        incompatible["config"] = dict(payload["config"])
        incompatible["config"]["separator_aware_refiner_mode"] = "full"
        full_replacement, _ = audiosep_model(refiner_config(True, "full"))
        mismatched_live_model = QCESModel(
            config, separator=full_replacement.separator
        )
        with self.assertRaisesRegex(ValueError, "mode differs"):
            mismatched_live_model.checkpoint_payload(backend="audiosep")
        with patch(
            "mixi_understanding.qces.model."
            "AudioSepConditionedAdapter.from_repository",
            return_value=full_replacement.separator,
        ), self.assertRaises(RuntimeError):
            load_qces_checkpoint(
                incompatible,
                audiosep_repository_root="repo",
                audiosep_config_path="config",
                audiosep_checkpoint_path="separator",
            )

    def test_refiner_mode_config_is_strict_and_legacy_full_is_compatible(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires"):
            refiner_config(False, "relative_energy")
        with self.assertRaisesRegex(ValueError, "must be one of"):
            refiner_config(True, "unknown")

        config = refiner_config(True, "full")
        source, _ = audiosep_model(config)
        payload = source.checkpoint_payload(backend="audiosep")
        legacy_config = dict(payload["config"])
        legacy_config.pop("separator_aware_refiner_mode")
        payload["config"] = legacy_config
        replacement, _ = audiosep_model(config)
        with patch(
            "mixi_understanding.qces.model."
            "AudioSepConditionedAdapter.from_repository",
            return_value=replacement.separator,
        ):
            restored = load_qces_checkpoint(
                payload,
                audiosep_repository_root="repo",
                audiosep_config_path="config",
                audiosep_checkpoint_path="separator",
            )
        self.assertEqual(restored.config.separator_aware_refiner_mode, "full")
        self.assertEqual(
            set(source.separator.separator_aware_refiner.state_dict()),
            set(restored.separator.separator_aware_refiner.state_dict()),
        )

    def test_old_audiosep_checkpoint_without_flag_loads_unchanged(self) -> None:
        config = refiner_config(False)
        source, _ = audiosep_model(config)
        legacy_config = config.to_dict()
        legacy_config.pop("separator_aware_refiner")
        payload = {
            "format": "qces_v1",
            "backend": "audiosep",
            "config": legacy_config,
            "composer_state_dict": source.composer.state_dict(),
            "extra": {},
        }
        replacement, _ = audiosep_model(config)
        with patch(
            "mixi_understanding.qces.model."
            "AudioSepConditionedAdapter.from_repository",
            return_value=replacement.separator,
        ):
            restored = load_qces_checkpoint(
                payload,
                audiosep_repository_root="repo",
                audiosep_config_path="config",
                audiosep_checkpoint_path="separator",
            )
        self.assertFalse(restored.config.separator_aware_refiner)
        self.assertIsNone(restored.separator.separator_aware_refiner)
        for name, expected in source.composer.state_dict().items():
            self.assertTrue(torch.equal(expected, restored.composer.state_dict()[name]))

    def test_deterministic_smoothing_preserves_constant_and_backpropagates(self) -> None:
        tensor = torch.ones(2, 3, 7, requires_grad=True)
        smoothed = qces_smooth_1d(tensor)
        self.assertTrue(torch.allclose(smoothed, tensor))
        smoothed.square().sum().backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
