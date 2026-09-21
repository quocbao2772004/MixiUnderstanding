"""Tensor-contract and temporal-semantics tests for the Q-DOR core."""

from __future__ import annotations

import unittest

import torch

from mixi_understanding.qces.dense_event_qa import (
    RELATION_AFTER,
    RELATION_BEFORE,
    DenseTemporalReasoner,
    DenseTemporalReasonerConfig,
    decode_dense_temporal_outputs,
    dense_temporal_reasoner_loss,
    intervals_to_40ms_mask,
    masks_to_40ms_intervals,
    onset_relation_mask,
)


class FixedGridMaskTest(unittest.TestCase):
    def test_half_open_intervals_use_exact_40ms_frames(self) -> None:
        mask = intervals_to_40ms_mask([[0.04, 0.12], [0.20, 0.24]], 8)
        self.assertTrue(
            torch.equal(
                mask,
                torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            )
        )
        self.assertEqual(
            masks_to_40ms_intervals(mask),
            [(0.04, 0.12), (0.20, 0.24)],
        )

    def test_batched_interval_padding_is_ignored(self) -> None:
        intervals = torch.tensor(
            [
                [[0.0, 0.04], [float("nan"), float("nan")]],
                [[0.08, 0.16], [0.20, 0.28]],
            ]
        )
        mask = intervals_to_40ms_mask(intervals, 7)
        self.assertEqual(tuple(mask.shape), (2, 7))
        self.assertTrue(torch.equal(mask[0], torch.tensor([1, 0, 0, 0, 0, 0, 0])))
        self.assertTrue(torch.equal(mask[1], torch.tensor([0, 0, 1, 1, 0, 1, 1])))

    def test_empty_no_evidence_intervals_make_a_silent_mask(self) -> None:
        self.assertTrue(torch.equal(intervals_to_40ms_mask([], 6), torch.zeros(6)))


class DifferentiableRelationTest(unittest.TestCase):
    def test_strict_before_and_after_follow_anchor_onset(self) -> None:
        anchor = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )
        relation, onset = onset_relation_mask(
            anchor, torch.tensor([RELATION_BEFORE, RELATION_AFTER])
        )
        # Tiny numerical fallback mass is permitted, but the intended hard
        # geometry remains exact to floating-point tolerance.
        self.assertTrue(torch.allclose(onset[:, 2], torch.ones(2), atol=1e-5))
        self.assertTrue(
            torch.allclose(
                relation[0], torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]), atol=1e-5
            )
        )
        self.assertTrue(
            torch.allclose(
                relation[1], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]), atol=1e-5
            )
        )

    def test_relation_cdf_is_differentiable(self) -> None:
        logits = torch.randn(2, 11, requires_grad=True)
        relation, _ = onset_relation_mask(
            logits.sigmoid(), torch.tensor([RELATION_BEFORE, RELATION_AFTER])
        )
        loss = (relation * torch.linspace(0, 1, 11)).sum()
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_padded_anchor_activity_cannot_enter_onset_or_relation(self) -> None:
        anchor = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        valid = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        relation, onset = onset_relation_mask(
            anchor,
            torch.tensor([RELATION_AFTER]),
            valid_frame_mask=valid,
        )
        self.assertTrue(torch.equal(relation[:, 4:], torch.zeros(1, 2)))
        self.assertTrue(torch.equal(onset[:, 4:], torch.zeros(1, 2)))


class DenseTemporalReasonerTest(unittest.TestCase):
    @staticmethod
    def tiny_model(num_classes: int = 7) -> DenseTemporalReasoner:
        return DenseTemporalReasoner(
            DenseTemporalReasonerConfig(
                feature_dim=12,
                num_classes=num_classes,
                hidden_dim=16,
                max_ordinal=4,
                dropout=0.0,
            )
        )

    def test_public_forward_contract_and_shared_none_choice(self) -> None:
        torch.manual_seed(11)
        model = self.tiny_model(num_classes=200)
        outputs = model(
            torch.randn(3, 25, 12),
            torch.randn(3, 25, 200),
            anchor_label=torch.tensor([2, 17, 199]),
            relation_id=torch.tensor([RELATION_BEFORE, RELATION_AFTER, RELATION_AFTER]),
            ordinal=torch.tensor([1, 2, 1]),
        )
        expected = {
            "answer_logits": (3, 200),
            "answerability_logit": (3,),
            "anchor_mask_logits": (3, 25),
            "answer_mask_logits": (3, 25),
            "relation_mask": (3, 25),
            "choice_logits": (3, 201),
        }
        for name, shape in expected.items():
            self.assertEqual(tuple(outputs[name].shape), shape)
            self.assertTrue(torch.isfinite(outputs[name]).all())
        reconstructed = (
            torch.logsumexp(outputs["choice_logits"][:, :-1], dim=1)
            - outputs["choice_logits"][:, -1]
        )
        self.assertTrue(torch.allclose(outputs["answerability_logit"], reconstructed))

    def test_gold_anchor_teacher_forcing_controls_relation_geometry(self) -> None:
        model = self.tiny_model()
        gold_anchor = intervals_to_40ms_mask(
            torch.tensor([[[0.12, 0.20]], [[0.12, 0.20]]]), 8
        )
        outputs = model(
            torch.randn(2, 8, 12),
            torch.randn(2, 8, 7),
            anchor_label=torch.tensor([1, 1]),
            relation_id=torch.tensor([RELATION_BEFORE, RELATION_AFTER]),
            ordinal=torch.ones(2, dtype=torch.long),
            gold_anchor_mask=gold_anchor,
            gold_answer_label=torch.tensor([3, -1]),
            teacher_forcing=1.0,
        )
        self.assertTrue(
            torch.allclose(
                outputs["relation_mask"][0],
                torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["relation_mask"][1],
                torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
                atol=1e-5,
            )
        )
        # A teacher-forced NONE label suppresses its answer mask.
        self.assertLess(
            float(outputs["answer_mask_logits"][1].detach().amax()), -10.0
        )

    def test_joint_losses_backpropagate_to_pointer_scorer_and_masks(self) -> None:
        torch.manual_seed(19)
        model = self.tiny_model()
        features = torch.randn(3, 20, 12)
        detector = torch.randn(3, 20, 7)
        anchor_mask = intervals_to_40ms_mask(
            torch.tensor(
                [
                    [[0.20, 0.32]],
                    [[0.36, 0.48]],
                    [[0.12, 0.24]],
                ]
            ),
            20,
        )
        answer_mask = intervals_to_40ms_mask(
            torch.tensor(
                [
                    [[0.48, 0.60]],
                    [[0.08, 0.20]],
                    [[float("nan"), float("nan")]],
                ]
            ),
            20,
        )
        gold_label = torch.tensor([3, 5, -1])
        outputs = model(
            features,
            detector,
            anchor_label=torch.tensor([1, 2, 4]),
            relation_id=torch.tensor([RELATION_AFTER, RELATION_BEFORE, RELATION_AFTER]),
            ordinal=torch.tensor([1, 2, 1]),
            gold_anchor_mask=anchor_mask,
            gold_answer_label=gold_label,
            teacher_forcing=0.5,
        )
        losses = dense_temporal_reasoner_loss(
            outputs,
            gold_answer_label=gold_label,
            gold_anchor_mask=anchor_mask,
            gold_answer_mask=answer_mask,
        )
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        for name in (
            "class_embeddings.weight",
            "frame_encoder.1.weight",
            "none_scorer.1.weight",
            "answer_mask_residual.0.weight",
        ):
            parameter = dict(model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_decode_uses_answerability_and_fixed_grid(self) -> None:
        outputs = {
            "answer_logits": torch.tensor([[0.0, 4.0], [5.0, 0.0]]),
            "answerability_logit": torch.tensor([5.0, -5.0]),
            "anchor_mask_logits": torch.tensor(
                [[-8.0, 8.0, 8.0, -8.0], [-8.0, -8.0, -8.0, -8.0]]
            ),
            "answer_mask_logits": torch.tensor(
                [[-8.0, -8.0, 8.0, -8.0], [8.0, 8.0, -8.0, -8.0]]
            ),
        }
        decoded = decode_dense_temporal_outputs(outputs)
        self.assertTrue(torch.equal(decoded["answer_label"], torch.tensor([1, -1])))
        self.assertEqual(decoded["anchor_intervals"][0], [(0.04, 0.12)])
        self.assertEqual(decoded["answer_intervals"][0], [(0.08, 0.12)])

    def test_padding_values_do_not_change_dense_answer_scores(self) -> None:
        torch.manual_seed(31)
        model = self.tiny_model().eval()
        features = torch.randn(2, 12, 12)
        detector = torch.randn(2, 12, 7)
        valid = torch.tensor(
            [
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )
        corrupted_features = features.clone()
        corrupted_detector = detector.clone()
        corrupted_features[~valid] = 10_000.0
        corrupted_detector[~valid] = 10_000.0
        kwargs = {
            "anchor_label": torch.tensor([1, 3]),
            "relation_id": torch.tensor([RELATION_AFTER, RELATION_BEFORE]),
            "ordinal": torch.tensor([1, 2]),
            "valid_frame_mask": valid,
        }
        with torch.inference_mode():
            clean = model(features, detector, **kwargs)
            corrupt = model(corrupted_features, corrupted_detector, **kwargs)
        for name in (
            "answer_logits",
            "answerability_logit",
            "anchor_mask_logits",
            "answer_mask_logits",
            "relation_mask",
        ):
            self.assertTrue(torch.allclose(clean[name], corrupt[name]), name)
        self.assertTrue(torch.equal(clean["relation_mask"][~valid], torch.zeros(11)))
        self.assertTrue((clean["anchor_mask_logits"][~valid] <= -30.0).all())
        self.assertTrue((clean["answer_mask_logits"][~valid] <= -30.0).all())

    def test_omitted_valid_mask_matches_an_all_valid_mask(self) -> None:
        torch.manual_seed(37)
        model = self.tiny_model().eval()
        features = torch.randn(2, 9, 12)
        detector = torch.randn(2, 9, 7)
        kwargs = {
            "anchor_label": torch.tensor([0, 6]),
            "relation_id": torch.tensor([RELATION_BEFORE, RELATION_AFTER]),
            "ordinal": torch.tensor([1, 1]),
        }
        with torch.inference_mode():
            omitted = model(features, detector, **kwargs)
            explicit = model(
                features,
                detector,
                **kwargs,
                valid_frame_mask=torch.ones(2, 9, dtype=torch.bool),
            )
        for name in (
            "answer_logits",
            "answerability_logit",
            "anchor_mask_logits",
            "answer_mask_logits",
            "relation_mask",
            "frame_class_logits",
        ):
            self.assertTrue(torch.equal(omitted[name], explicit[name]), name)

    def test_loss_and_decode_ignore_invalid_short_clip_tail(self) -> None:
        batch, frames, classes = 1, 8, 3
        valid = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
        base = {
            "answer_logits": torch.tensor([[0.0, 3.0, -1.0]], requires_grad=True),
            "answerability_logit": torch.tensor([2.0], requires_grad=True),
            "choice_logits": torch.tensor(
                [[0.0, 3.0, -1.0, -2.0]], requires_grad=True
            ),
            "anchor_mask_logits": torch.tensor(
                [[-5.0, 5.0, 5.0, -5.0, -5.0, -5.0, -5.0, -5.0]],
                requires_grad=True,
            ),
            "answer_mask_logits": torch.tensor(
                [[-5.0, -5.0, 5.0, -5.0, -5.0, -5.0, -5.0, -5.0]],
                requires_grad=True,
            ),
            "relation_mask": torch.tensor(
                [[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
            ),
            "valid_frame_mask": valid,
        }
        corrupted = dict(base)
        corrupted["anchor_mask_logits"] = base["anchor_mask_logits"].detach().clone()
        corrupted["answer_mask_logits"] = base["answer_mask_logits"].detach().clone()
        corrupted["relation_mask"] = base["relation_mask"].clone()
        corrupted["anchor_mask_logits"][:, 4:] = 1_000.0
        corrupted["answer_mask_logits"][:, 4:] = 1_000.0
        corrupted["relation_mask"][:, 4:] = 1.0
        corrupted["anchor_mask_logits"].requires_grad_()
        corrupted["answer_mask_logits"].requires_grad_()
        anchor_target = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        answer_target = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
        clean_loss = dense_temporal_reasoner_loss(
            base,
            gold_answer_label=torch.tensor([1]),
            gold_anchor_mask=anchor_target,
            gold_answer_mask=answer_target,
        )
        corrupt_loss = dense_temporal_reasoner_loss(
            corrupted,
            gold_answer_label=torch.tensor([1]),
            gold_anchor_mask=anchor_target,
            gold_answer_mask=answer_target,
        )
        for name in clean_loss:
            self.assertTrue(
                torch.allclose(clean_loss[name], corrupt_loss[name]), name
            )

        decoded = decode_dense_temporal_outputs(corrupted)
        self.assertEqual(decoded["anchor_intervals"], [[(0.04, 0.12)]])
        self.assertEqual(decoded["answer_intervals"], [[(0.08, 0.12)]])
        self.assertTrue(torch.equal(decoded["anchor_probability"][:, 4:], torch.zeros(1, 4)))
        self.assertTrue(torch.equal(decoded["answer_probability"][:, 4:], torch.zeros(1, 4)))

    def test_invalid_shapes_fail_before_silent_broadcasting(self) -> None:
        model = self.tiny_model()
        with self.assertRaisesRegex(ValueError, "detector_logits"):
            model(
                torch.randn(2, 10, 12),
                torch.randn(2, 9, 7),
                torch.tensor([1, 2]),
                torch.tensor([0, 1]),
                torch.tensor([1, 1]),
            )


if __name__ == "__main__":
    unittest.main()
