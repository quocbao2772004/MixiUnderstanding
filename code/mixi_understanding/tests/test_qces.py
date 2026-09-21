"""Unit tests for the checkpoint-independent QCES path."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mixi_understanding.qces.composer import PromptComposition
from mixi_understanding.qces.config import QCESConfig
from mixi_understanding.qces.data import QCESManifestDataset, collate_qces
from mixi_understanding.qces.losses import LossWeights, QCESLoss
from mixi_understanding.qces.metrics import (
    qces_metrics,
    scale_dependent_sdr,
    scale_invariant_sdr,
)
from mixi_understanding.qces.model import (
    QCESModel,
    QCESOutput,
    load_qces_checkpoint,
)
from mixi_understanding.qces.separators import (
    AudioSepConditionedAdapter,
    PhaseAwareComplexMaskSeparator,
    SeparationOutput,
)
from mixi_understanding.qces.signal import (
    deterministic_reflect_pad_1d,
    qces_linear_interpolate_1d,
    qces_stft,
)
from mixi_understanding.qces.tokenization import StableHashTokenizer


def tiny_config() -> QCESConfig:
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
    )


class StableHashTokenizerTest(unittest.TestCase):
    def test_is_deterministic_and_padded(self) -> None:
        tokenizer = StableHashTokenizer(vocab_size=64, max_length=8)
        first = tokenizer.batch_encode(["After horn?", ""])
        second = tokenizer.batch_encode(["After horn?", ""])
        self.assertTrue(torch.equal(first.input_ids, second.input_ids))
        self.assertTrue(torch.equal(first.attention_mask, second.attention_mask))
        self.assertTrue(first.attention_mask.any(dim=1).all())


class QCESMetricsTest(unittest.TestCase):
    def test_sd_sdr_penalizes_gain_that_si_sdr_ignores(self) -> None:
        target = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        orthogonal_noise = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
        estimate = target + 0.1 * orthogonal_noise
        attenuated = 0.5 * estimate
        self.assertTrue(
            torch.allclose(
                scale_invariant_sdr(estimate, target),
                scale_invariant_sdr(attenuated, target),
                atol=1e-5,
            )
        )
        self.assertGreater(
            float(
                scale_dependent_sdr(estimate, target)
                - scale_dependent_sdr(attenuated, target)
            ),
            10.0,
        )
        self.assertAlmostEqual(
            float(scale_dependent_sdr(0.5 * target, target)), 0.0, places=5
        )

    def test_answerable_iou_and_negative_retention_are_class_conditional(self) -> None:
        mixture = torch.ones(2, 4)
        evidence = torch.stack(
            [torch.tensor([1.0, 0.0, 0.0, 0.0]), torch.full((4,), 0.5)]
        )
        role_logits = torch.tensor(
            [
                [[-8.0, 8.0, -8.0], [8.0, -8.0, -8.0],
                 [8.0, -8.0, -8.0], [8.0, -8.0, -8.0]],
                [[-8.0, 8.0, -8.0], [-8.0, 8.0, -8.0],
                 [-8.0, 8.0, -8.0], [-8.0, 8.0, -8.0]],
            ]
        )
        probabilities = 1.0 - role_logits.softmax(dim=-1)[..., 0]
        composition = PromptComposition(
            semantic_condition=torch.zeros(2, 1),
            role_logits=role_logits,
            evidence_probability=probabilities,
            no_evidence_logit=torch.zeros(2),
            frame_features=torch.zeros(2, 4, 1),
            frame_hop_samples=1,
        )
        separation = SeparationOutput(
            evidence=evidence,
            residual=mixture - evidence,
            mask=probabilities,
            raw_mask=None,
            raw_evidence=None,
            mixture_error=torch.zeros(2),
        )
        output = QCESOutput(composition=composition, separation=separation)
        anchor_mask = torch.zeros_like(mixture)
        anchor_mask[0, 0] = 1.0
        batch = {
            "mixture": mixture,
            "evidence": torch.stack([anchor_mask[0], torch.zeros(4)]),
            "residual": mixture,
            "anchor_mask": anchor_mask,
            "answer_mask": torch.zeros_like(mixture),
            "no_evidence": torch.tensor([0.0, 1.0]),
        }
        metrics = qces_metrics(output, batch)
        self.assertAlmostEqual(float(metrics["answerable_temporal_iou"]), 1.0)
        self.assertAlmostEqual(float(metrics["temporal_iou"]), 0.5)
        self.assertAlmostEqual(
            float(metrics["no_evidence_retained_ratio"]), 0.5
        )
        self.assertIn("evidence_sd_sdr", metrics)


class QCESDatasetTest(unittest.TestCase):
    @staticmethod
    def _event(event_id, label, role, onset, offset):
        duration = offset - onset
        return {
            "event_id": event_id,
            "label": label,
            "source_dataset": "test",
            "source_id": f"source_{event_id}",
            "source_path": f"sources/{event_id}.wav",
            "source_sha256": "0" * 64,
            "source_interval_seconds": [0.0, 1.0],
            "source_crop_interval_seconds": [0.0, duration],
            "onset_seconds": onset,
            "offset_seconds": offset,
            "role": role,
        }

    def test_manifest_audio_and_role_masks(self) -> None:
        sample_rate = 8_000
        samples = sample_rate
        rng = np.random.default_rng(4)
        evidence = rng.normal(0, 0.02, samples).astype(np.float32)
        residual = rng.normal(0, 0.01, samples).astype(np.float32)
        mixture = evidence + residual
        events = [
            self._event("event_anchor", "horn", "anchor", 0.1, 0.2),
            self._event("event_answer", "glass", "answer", 0.3, 0.4),
            self._event("event_interference", "music", "interference", 0.6, 0.8),
        ]
        record = {
            "schema_version": "qa_removal_v2",
            "id": "train_000000",
            "scene_id": "scene_000000",
            "split": "train",
            "sample_rate": sample_rate,
            "num_channels": 1,
            "num_samples": samples,
            "duration_seconds": 1.0,
            "mixture_path": "audio/mixture.wav",
            "clean_path": "audio/clean.wav",
            "interference_stem_path": "audio/residual.wav",
            "question": "What sound occurs immediately after horn?",
            "answer": "glass",
            "question_type": "temporal_after",
            "events": events,
            "anchor_event_ids": ["event_anchor"],
            "answer_event_ids": ["event_answer"],
            "anchor_intervals": [[0.1, 0.2]],
            "answer_intervals": [[0.3, 0.4]],
            "interference_event_ids": ["event_interference"],
            "interference_intervals": [[0.6, 0.8]],
            "event_presence_labels": ["horn", "glass", "music"],
            "edit_needed": False,
            "edit_rationale": "non_overlap",
            "selector_target": "no_edit",
            "snr_measurement": "evidence_vs_active_interference",
            "snr_db_requested": 0.0,
            "snr_db": 0.0,
            "mixture_peak": float(np.abs(mixture).max()),
            "source_group_ids": [
                "source_event_anchor",
                "source_event_answer",
                "source_event_interference",
            ],
            "generation_seed": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "audio").mkdir()
            sf.write(root / "audio/mixture.wav", mixture, sample_rate)
            sf.write(root / "audio/clean.wav", evidence, sample_rate)
            sf.write(root / "audio/residual.wav", residual, sample_rate)
            (root / "manifest.jsonl").write_text(
                json.dumps(record) + "\n", encoding="utf-8"
            )
            dataset = QCESManifestDataset(
                root / "manifest.jsonl",
                crop_samples=4_096,
                random_crop=False,
            )
            batch = next(
                iter(
                    DataLoader(
                        dataset,
                        batch_size=1,
                        collate_fn=partial(
                            collate_qces,
                            tokenizer=StableHashTokenizer(128, 16),
                        ),
                    )
                )
            )
            self.assertEqual(tuple(batch["mixture"].shape), (1, 4_096))
            self.assertGreater(float(batch["anchor_mask"].sum()), 0.0)
            self.assertGreater(float(batch["answer_mask"].sum()), 0.0)
            self.assertEqual(tuple(batch["anchor_stem"].shape), (1, 4_096))
            self.assertEqual(tuple(batch["answer_stem"].shape), (1, 4_096))
            self.assertEqual(batch["questions"], [record["question"]])


class QCESModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.config = tiny_config()
        self.mixture = torch.randn(2, 512) * 0.1
        self.tokens = StableHashTokenizer(128, 16).batch_encode(
            ["What follows the horn?", "Which sound is first?"]
        )

    def test_shapes_and_exact_complement(self) -> None:
        model = QCESModel(self.config).eval()
        with torch.inference_mode():
            output = model(
                self.mixture, self.tokens.input_ids, self.tokens.attention_mask
            )
        self.assertEqual(output.evidence.shape, self.mixture.shape)
        self.assertEqual(output.residual.shape, self.mixture.shape)
        self.assertEqual(output.composition.semantic_condition.shape, (2, 16))
        self.assertLess(float(output.separation.mixture_error.max()), 1e-6)

    def test_complex_backend_is_complementary_and_reloadable(self) -> None:
        separator = PhaseAwareComplexMaskSeparator(self.config)
        model = QCESModel(self.config, separator=separator).eval()
        with torch.inference_mode():
            output = model(
                self.mixture, self.tokens.input_ids, self.tokens.attention_mask
            )
        self.assertTrue(torch.is_complex(output.separation.mask))
        self.assertLess(float(output.separation.mixture_error.max()), 1e-7)
        payload = model.checkpoint_payload(backend="complex")
        restored = load_qces_checkpoint(payload).eval()
        with torch.inference_mode():
            restored_output = restored(
                self.mixture, self.tokens.input_ids, self.tokens.attention_mask
            )
        self.assertTrue(torch.allclose(output.evidence, restored_output.evidence))

    def test_full_loss_backpropagates_into_composer(self) -> None:
        model = QCESModel(self.config)
        output = model(self.mixture, self.tokens.input_ids, self.tokens.attention_mask)
        anchor_mask = torch.zeros_like(self.mixture)
        answer_mask = torch.zeros_like(self.mixture)
        anchor_mask[:, 40:140] = 1
        answer_mask[:, 260:380] = 1
        union = (anchor_mask + answer_mask).clamp_max(1)
        batch = {
            "mixture": self.mixture,
            "evidence": self.mixture * union,
            "residual": self.mixture * (1 - union),
            "anchor_mask": anchor_mask,
            "answer_mask": answer_mask,
            "no_evidence": torch.zeros(2),
            "question_ids": self.tokens.input_ids,
            "question_mask": self.tokens.attention_mask,
        }
        loss, components = QCESLoss(fft_sizes=(64, 128))(output, batch)
        loss.backward()
        gradient = model.composer.role_head.weight.grad
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)
        self.assertIn("temporal_dice", components)

    def test_no_evidence_bce_balances_the_positive_class(self) -> None:
        model = QCESModel(self.config)
        output = model(self.mixture, self.tokens.input_ids, self.tokens.attention_mask)
        output.composition.no_evidence_logit = torch.zeros(2)
        anchor_mask = torch.ones_like(self.mixture)
        answer_mask = torch.zeros_like(self.mixture)
        batch = {
            "mixture": self.mixture,
            "evidence": self.mixture,
            "residual": torch.zeros_like(self.mixture),
            "anchor_mask": anchor_mask,
            "answer_mask": answer_mask,
            "no_evidence": torch.tensor([0.0, 1.0]),
        }
        _, neutral = QCESLoss(fft_sizes=())(output, batch)
        _, balanced = QCESLoss(
            fft_sizes=(), no_evidence_positive_weight=3.0
        )(output, batch)
        self.assertAlmostEqual(float(neutral["no_evidence"]), math.log(2.0), places=6)
        self.assertAlmostEqual(
            float(balanced["no_evidence"]), 2.0 * math.log(2.0), places=6
        )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            QCESLoss(no_evidence_positive_weight=0.0)

    def test_flattened_temporal_cross_entropy_is_algebraically_identical(self) -> None:
        logits = torch.randn(2, 11, 3)
        labels = torch.randint(0, 3, (2, 11))
        weights = torch.tensor([0.25, 1.0, 1.0])
        legacy = F.cross_entropy(logits.transpose(1, 2), labels, weight=weights)
        deterministic = F.cross_entropy(
            logits.reshape(-1, 3), labels.reshape(-1), weight=weights
        )
        self.assertTrue(torch.allclose(legacy, deterministic, atol=1e-7))

    def test_weakest_role_loss_detects_a_missing_answer(self) -> None:
        model = QCESModel(self.config)
        output = model(self.mixture, self.tokens.input_ids, self.tokens.attention_mask)
        anchor_mask = torch.zeros_like(self.mixture)
        answer_mask = torch.zeros_like(self.mixture)
        anchor_mask[:, 40:140] = 1
        answer_mask[:, 260:380] = 1
        anchor_stem = self.mixture * anchor_mask
        answer_stem = self.mixture * answer_mask
        output.separation.evidence = anchor_stem.clone()
        output.separation.residual = self.mixture - output.evidence
        batch = {
            "mixture": self.mixture,
            "evidence": anchor_stem + answer_stem,
            "residual": self.mixture - anchor_stem - answer_stem,
            "anchor_stem": anchor_stem,
            "answer_stem": answer_stem,
            "anchor_mask": anchor_mask,
            "answer_mask": answer_mask,
            "no_evidence": torch.zeros(2),
            "question_ids": self.tokens.input_ids,
            "question_mask": self.tokens.attention_mask,
        }
        _, components = QCESLoss(fft_sizes=(64,))(output, batch)
        self.assertLess(float(components["role_relative_waveform"]), 0.51)
        self.assertGreater(float(components["weakest_role_waveform"]), 0.99)

    def test_optional_counterfactual_qa_terms(self) -> None:
        class ToyAuditor(nn.Module):
            def forward(self, audio, question_ids, question_mask):
                score = audio.square().mean(dim=-1)
                return torch.stack([-score, score], dim=-1)

        model = QCESModel(self.config)
        output = model(self.mixture, self.tokens.input_ids, self.tokens.attention_mask)
        ones = torch.ones_like(self.mixture)
        batch = {
            "mixture": self.mixture,
            "evidence": self.mixture,
            "residual": torch.zeros_like(self.mixture),
            "anchor_mask": ones,
            "answer_mask": torch.zeros_like(ones),
            "no_evidence": torch.zeros(2),
            "question_ids": self.tokens.input_ids,
            "question_mask": self.tokens.attention_mask,
        }
        weights = LossWeights(qa_sufficiency=1.0, qa_necessity=1.0)
        loss, components = QCESLoss(weights=weights, fft_sizes=(64,))(
            output,
            batch,
            qa_auditor=ToyAuditor(),
            answer_targets=torch.ones(2, dtype=torch.long),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(components["qa_sufficiency"]))
        self.assertTrue(torch.isfinite(components["qa_necessity"]))

    def test_silent_no_evidence_target_has_bounded_loss(self) -> None:
        model = QCESModel(self.config)
        output = model(self.mixture, self.tokens.input_ids, self.tokens.attention_mask)
        anchor_mask = torch.zeros_like(self.mixture)
        answer_mask = torch.zeros_like(self.mixture)
        anchor_mask[0, 30:130] = 1
        answer_mask[0, 240:360] = 1
        union = (anchor_mask + answer_mask).clamp_max(1)
        evidence = self.mixture * union
        evidence[1].zero_()
        batch = {
            "mixture": self.mixture,
            "evidence": evidence,
            "residual": self.mixture - evidence,
            "anchor_mask": anchor_mask,
            "answer_mask": answer_mask,
            "no_evidence": torch.tensor([0.0, 1.0]),
            "question_ids": self.tokens.input_ids,
            "question_mask": self.tokens.attention_mask,
        }
        loss, components = QCESLoss(fft_sizes=(64, 128))(output, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(components["multi_resolution_stft"]), 20.0)
        self.assertTrue(torch.isfinite(components["separator_mask"]))

    def test_semantic_alignment_supervises_composer_condition(self) -> None:
        model = QCESModel(self.config)
        output = model(self.mixture, self.tokens.input_ids, self.tokens.attention_mask)
        batch = {
            "mixture": self.mixture,
            "evidence": self.mixture,
            "residual": torch.zeros_like(self.mixture),
            "anchor_mask": torch.ones_like(self.mixture),
            "answer_mask": torch.zeros_like(self.mixture),
            "no_evidence": torch.tensor([0.0, 1.0]),
            "question_ids": self.tokens.input_ids,
            "question_mask": self.tokens.attention_mask,
            "semantic_target": torch.randn(2, self.config.condition_dim),
        }
        loss, components = QCESLoss(
            weights=LossWeights(semantic_alignment=2.0), fft_sizes=(64,)
        )(output, batch)
        loss.backward()
        gradient = model.composer.semantic_head[-1].weight.grad
        self.assertGreater(float(components["semantic_alignment"]), 0.0)
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)


class AudioSepAdapterTest(unittest.TestCase):
    def test_deterministic_reflection_pad_matches_pytorch(self) -> None:
        waveform = torch.randn(2, 1, 32, requires_grad=True)
        reference_waveform = waveform.detach().clone().requires_grad_(True)
        actual = deterministic_reflect_pad_1d(waveform, 7, 5)
        expected = F.pad(reference_waveform, (7, 5), mode="reflect")
        self.assertTrue(torch.equal(actual, expected))
        weights = torch.linspace(0.1, 1.0, actual.numel()).reshape_as(actual)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        self.assertTrue(torch.equal(waveform.grad, reference_waveform.grad))

    def test_deterministic_centered_stft_matches_standard_stft(self) -> None:
        waveform = torch.randn(2, 96)
        window = torch.hann_window(16)
        expected = torch.stft(
            waveform,
            n_fft=16,
            hop_length=4,
            win_length=16,
            window=window,
            center=True,
            return_complex=True,
        )
        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            actual = qces_stft(
                waveform,
                n_fft=16,
                hop_length=4,
                win_length=16,
                window=window,
                center=True,
                return_complex=True,
            )
        finally:
            torch.use_deterministic_algorithms(prior)
        self.assertTrue(torch.equal(actual, expected))

    def test_deterministic_linear_interpolation_matches_pytorch(self) -> None:
        frames = torch.randn(2, 1, 17)
        expected = F.interpolate(
            frames, size=113, mode="linear", align_corners=False
        )
        prior = torch.are_deterministic_algorithms_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            actual = qces_linear_interpolate_1d(frames, 113)
        finally:
            torch.use_deterministic_algorithms(prior)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-6, rtol=1e-6))

    def test_frozen_adapter_is_complementary(self) -> None:
        class ToySeparator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gain = nn.Parameter(torch.tensor(0.75))

            def forward(self, batch):
                return {"waveform": batch["mixture"] * self.gain}

        separator = ToySeparator()
        adapter = AudioSepConditionedAdapter(
            separator, condition_dim=16, freeze_separator=True
        )
        mixture = torch.randn(2, 256)
        composition = PromptComposition(
            semantic_condition=torch.randn(2, 16),
            role_logits=torch.randn(2, 17, 3),
            evidence_probability=torch.rand(2, 17),
            no_evidence_logit=torch.zeros(2),
            frame_features=torch.randn(2, 17, 32),
            frame_hop_samples=16,
        )
        result = adapter(mixture, composition)
        self.assertFalse(separator.gain.requires_grad)
        self.assertLess(float(result.mixture_error.max()), 1e-7)
        self.assertTrue(torch.allclose(result.evidence + result.residual, mixture))


if __name__ == "__main__":
    unittest.main()
