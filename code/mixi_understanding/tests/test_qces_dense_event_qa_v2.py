"""Behavioral and gradient contracts for Q-DOR-v2."""

from __future__ import annotations

import unittest

import torch

from mixi_understanding.qces.dense_event_qa import (
    RELATION_AFTER,
    RELATION_BEFORE,
    intervals_to_40ms_mask,
)
from mixi_understanding.qces.dense_event_qa_v2 import (
    DenseTemporalReasonerV2,
    DenseTemporalReasonerV2Config,
    DenseTemporalThresholdsV2,
    decode_dense_temporal_outputs_v2,
    dense_temporal_reasoner_v2_loss,
    asymmetric_anchor_relation_mask,
    first_passage_nearest_onset,
    localized_occurrence_state,
)


class NearestOnsetOperatorTest(unittest.TestCase):
    def test_asymmetric_relation_uses_onset_for_before_and_offset_for_after(self) -> None:
        anchor = torch.zeros(2, 12)
        anchor[:, 3:7] = 1.0
        relation, onset, offset = asymmetric_anchor_relation_mask(
            anchor,
            torch.tensor([RELATION_BEFORE, RELATION_AFTER]),
        )
        self.assertTrue(torch.equal(relation[0, :3], torch.ones(3)))
        self.assertEqual(float(relation[0, 3:].sum()), 0.0)
        self.assertEqual(float(relation[1, :7].sum()), 0.0)
        self.assertTrue(torch.equal(relation[1, 7:], torch.ones(5)))
        self.assertEqual(int(onset[0].argmax()), 3)
        self.assertEqual(int(offset[1].argmax()), 7)

    def test_near_weaker_event_beats_far_stronger_event(self) -> None:
        anchor = torch.zeros(1, 20)
        anchor[:, 2:4] = 1.0
        event_onset = torch.zeros(1, 20)
        event_onset[:, 5] = 0.60
        event_onset[:, 15] = 0.99
        relation, nearest, no_event = first_passage_nearest_onset(
            event_onset, anchor, torch.tensor([RELATION_AFTER])
        )
        self.assertEqual(int(nearest.argmax(dim=1)), 5)
        self.assertAlmostEqual(float(nearest[0, 5]), 0.60, places=4)
        self.assertAlmostEqual(float(nearest[0, 15]), 0.396, places=3)
        self.assertAlmostEqual(float(no_event), 0.004, places=3)
        self.assertLess(float(relation[0, 2]), 1e-4)
        self.assertGreater(float(relation[0, 5]), 0.999)

    def test_perfect_onsets_give_exact_nearest_and_exact_none(self) -> None:
        anchor = torch.zeros(2, 16)
        anchor[:, 7:9] = 1.0
        event_onset = torch.zeros(2, 16)
        event_onset[0, 2] = 1.0
        event_onset[0, 5] = 1.0
        # Row one deliberately has no event after the anchor.
        relation, nearest, no_event = first_passage_nearest_onset(
            event_onset,
            anchor,
            torch.tensor([RELATION_BEFORE, RELATION_AFTER]),
        )
        self.assertEqual(int(nearest[0].argmax()), 5)
        self.assertGreater(float(nearest[0, 5]), 0.999)
        self.assertLess(float(nearest[0, 2]), 1e-5)
        self.assertLess(float(no_event[0]), 1e-5)
        self.assertEqual(float(nearest[1].sum()), 0.0)
        self.assertGreater(float(no_event[1]), 0.999)
        self.assertTrue(torch.all(relation[1, :8] < 1e-4))


class OccurrenceLocalizationTest(unittest.TestCase):
    def test_repeated_label_does_not_union_later_occurrence(self) -> None:
        activity = torch.zeros(1, 24)
        activity[:, 5:8] = 1.0
        activity[:, 15:18] = 1.0
        class_onset = torch.zeros_like(activity)
        class_onset[:, 5] = 1.0
        class_onset[:, 15] = 1.0
        selected_onset = torch.zeros_like(activity)
        selected_onset[:, 5] = 1.0
        probability, state = localized_occurrence_state(
            activity, class_onset, selected_onset
        )
        self.assertTrue(torch.equal(probability[0, 5:8], torch.ones(3)))
        self.assertEqual(float(probability[0, 15:18].sum()), 0.0)
        self.assertEqual(float(state[0, 15]), 0.0)


class DenseTemporalReasonerV2Test(unittest.TestCase):
    @staticmethod
    def model(classes: int = 6) -> DenseTemporalReasonerV2:
        return DenseTemporalReasonerV2(
            DenseTemporalReasonerV2Config(
                feature_dim=8,
                num_classes=classes,
                hidden_dim=16,
                max_ordinal=3,
                dropout=0.0,
            )
        )

    def test_anchor_query_is_invariant_to_relation(self) -> None:
        torch.manual_seed(101)
        model = self.model().eval()
        features = torch.randn(2, 30, 8)
        detector = torch.randn(2, 30, 6)
        features[1] = features[0]
        detector[1] = detector[0]
        with torch.inference_mode():
            outputs = model(
                features,
                detector,
                anchor_label=torch.tensor([2, 2]),
                relation_id=torch.tensor([RELATION_BEFORE, RELATION_AFTER]),
                ordinal=torch.tensor([2, 2]),
            )
        self.assertTrue(
            torch.equal(outputs["anchor_mask_logits"][0], outputs["anchor_mask_logits"][1])
        )
        self.assertFalse(
            torch.equal(outputs["relation_mask"][0], outputs["relation_mask"][1])
        )

    def test_none_verifier_cannot_suppress_positive_answer_mask(self) -> None:
        torch.manual_seed(103)
        model = self.model().eval()
        features = torch.randn(1, 35, 8)
        detector = torch.randn(1, 35, 6)
        kwargs = {
            "anchor_label": torch.tensor([1]),
            "relation_id": torch.tensor([RELATION_AFTER]),
            "ordinal": torch.tensor([1]),
        }
        with torch.inference_mode():
            before = model(features, detector, **kwargs)
            model.answerability_scorer[-1].bias.add_(20.0)
            after = model(features, detector, **kwargs)
        self.assertGreater(
            abs(
                float(before["answerability_logit"].sigmoid())
                - float(after["answerability_logit"].sigmoid())
            ),
            0.1,
        )
        self.assertTrue(torch.equal(before["answer_logits"], after["answer_logits"]))
        self.assertTrue(
            torch.equal(before["answer_mask_logits"], after["answer_mask_logits"])
        )

    def test_forward_uses_hard_one_class_condition(self) -> None:
        torch.manual_seed(107)
        model = self.model(classes=4).eval()
        with torch.no_grad():
            model.frame_class_feature_scale.zero_()
            model.anchor_feature_scale.zero_()
            model.anchor_detector_scale.fill_(8.0)
            model.ordinal_prior_scale.zero_()
            model.answer_bias.zero_()
            model.answer_bias[1] = 0.2
            model.answer_mask_residual[-1].weight.zero_()
            model.answer_mask_residual[-1].bias.zero_()
        detector = torch.full((1, 30, 4), -10.0)
        detector[0, 2:5, 0] = 10.0
        detector[0, 8:11, 1] = 6.0
        detector[0, 18:21, 2] = 10.0
        with torch.inference_mode():
            outputs = model(
                torch.zeros(1, 30, 8),
                detector,
                anchor_label=torch.tensor([0]),
                relation_id=torch.tensor([RELATION_AFTER]),
                ordinal=torch.tensor([1]),
            )
        condition = outputs["label_condition_weight"][0]
        self.assertEqual(float(condition.sum()), 1.0)
        self.assertEqual(int(condition.argmax()), int(outputs["answer_logits"].argmax()))
        self.assertEqual(int((condition > 0.5).sum()), 1)

    def test_scheduled_teacher_forcing_never_blends_two_classes(self) -> None:
        torch.manual_seed(1071)
        model = self.model(classes=4).train()
        batch = 16
        gold_anchor = torch.zeros(batch, 30)
        gold_anchor[:, 4:8] = 1.0
        outputs = model(
            torch.randn(batch, 30, 8),
            torch.randn(batch, 30, 4),
            anchor_label=torch.zeros(batch, dtype=torch.long),
            relation_id=torch.full((batch,), RELATION_AFTER),
            ordinal=torch.ones(batch, dtype=torch.long),
            gold_anchor_mask=gold_anchor,
            gold_answer_label=torch.tensor([1, 2] * (batch // 2)),
            teacher_forcing=0.5,
        )
        condition = outputs["label_condition_weight"].detach()
        self.assertTrue(torch.allclose(condition.sum(dim=1), torch.ones(batch), atol=1e-6))
        self.assertLess(float((condition - condition.round()).abs().max()), 1e-6)
        self.assertTrue(torch.equal((condition > 0.5).sum(dim=1), torch.ones(batch, dtype=torch.long)))
        use_gold = outputs["teacher_forcing_gold_context"].detach()
        predicted_anchor = outputs["anchor_mask_logits"].sigmoid().detach()
        expected_anchor = torch.where(use_gold[:, None], gold_anchor, predicted_anchor)
        self.assertTrue(
            torch.equal(outputs["anchor_condition_probability"].detach(), expected_anchor)
        )
        expected_label = torch.nn.functional.one_hot(
            torch.where(
                use_gold,
                torch.tensor([1, 2] * (batch // 2)),
                outputs["hard_answer_label"].detach(),
            ),
            num_classes=4,
        ).to(condition.dtype)
        self.assertLess(float((condition - expected_label).abs().max()), 1e-6)

    def test_absent_classes_do_not_create_a_frame_zero_max_c_hazard(self) -> None:
        torch.manual_seed(108)
        model = self.model(classes=200).eval()
        with torch.no_grad():
            model.frame_class_feature_scale.zero_()
            model.onset_feature_scale.zero_()
            model.onset_class_bias.zero_()
        # All 200 detector tracks are confidently absent and temporally flat.
        detector = torch.full((1, 20, 200), -6.0)
        with torch.inference_mode():
            outputs = model(
                torch.zeros(1, 20, 8),
                detector,
                anchor_label=torch.tensor([0]),
                relation_id=torch.tensor([RELATION_AFTER]),
                ordinal=torch.tensor([1]),
            )
        self.assertLess(float(outputs["union_onset_probability"].max()), 0.01)
        self.assertLess(float(outputs["nearest_event_onset_mass"].sum()), 0.05)

    def test_onset_and_evidence_losses_backpropagate(self) -> None:
        torch.manual_seed(109)
        model = self.model(classes=5)
        batch_size, frames = 4, 28
        features = torch.randn(batch_size, frames, 8)
        detector = torch.randn(batch_size, frames, 5)
        anchor_mask = intervals_to_40ms_mask(
            [[[0.16, 0.28]], [[0.20, 0.32]], [[0.24, 0.36]], [[0.28, 0.40]]],
            frames,
        )
        answer_mask = intervals_to_40ms_mask(
            [
                [[0.48, 0.60]],
                [[0.52, 0.64]],
                [[0.56, 0.68]],
                [[float("nan"), float("nan")]],
            ],
            frames,
        )
        gold_label = torch.tensor([2, 3, 4, -1])
        class_onset = torch.zeros(batch_size, frames, 5)
        union_onset = torch.zeros(batch_size, frames)
        for row in range(batch_size):
            anchor_frame = 4 + row
            answer_frame = 12 + row
            class_onset[row, anchor_frame, 1] = 1.0
            union_onset[row, anchor_frame] = 1.0
            if row < 3:
                class_onset[row, answer_frame, int(gold_label[row])] = 1.0
                union_onset[row, answer_frame] = 1.0
        outputs = model(
            features,
            detector,
            anchor_label=torch.ones(batch_size, dtype=torch.long),
            relation_id=torch.full((batch_size,), RELATION_AFTER),
            ordinal=torch.ones(batch_size, dtype=torch.long),
            gold_anchor_mask=anchor_mask,
            gold_answer_label=gold_label,
            teacher_forcing=1.0,
        )
        batch = {
            "gold_answer_label": gold_label,
            "gold_anchor_mask": anchor_mask,
            "gold_answer_mask": answer_mask,
            "gold_class_onset_mask": class_onset,
            "gold_union_onset_mask": union_onset,
            "valid_mask": torch.ones(batch_size, frames, dtype=torch.bool),
        }
        losses = dense_temporal_reasoner_v2_loss(outputs, batch)
        losses["loss"].backward()
        for name in (
            "onset_frame_encoder.1.weight",
            "onset_class_embeddings.weight",
            "frame_encoder.1.weight",
            "answerability_scorer.1.weight",
            "answer_mask_residual.0.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
        self.assertGreater(float(losses["class_onset_bce"]), 0.0)
        self.assertGreater(float(losses["union_onset_bce"]), 0.0)

    def test_ordinal_out_of_range_is_rejected_not_clamped(self) -> None:
        with self.assertRaisesRegex(ValueError, "silently clamped"):
            self.model()(
                torch.randn(1, 10, 8),
                torch.randn(1, 10, 6),
                anchor_label=torch.tensor([1]),
                relation_id=torch.tensor([RELATION_AFTER]),
                ordinal=torch.tensor([4]),
            )

    def test_none_decode_emits_anchor_plus_verification_evidence(self) -> None:
        outputs = {
            "answer_logits": torch.tensor([[4.0, 0.0]]),
            "answerability_logit": torch.tensor([-8.0]),
            "anchor_mask_logits": torch.tensor([[-8.0, -8.0, 8.0, 8.0, -8.0, -8.0]]),
            "answer_mask_logits": torch.tensor([[-8.0, -8.0, -8.0, -8.0, 8.0, 8.0]]),
            "relation_mask": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0]]),
            "valid_frame_mask": torch.ones(1, 6, dtype=torch.bool),
        }
        decoded = decode_dense_temporal_outputs_v2(
            outputs, thresholds=DenseTemporalThresholdsV2()
        )
        self.assertFalse(bool(decoded["answerable"][0]))
        self.assertEqual(decoded["answer_intervals"], [[]])
        self.assertEqual(decoded["verification_intervals"], [[(0.16, 0.24)]])
        self.assertEqual(decoded["evidence_intervals"], [[(0.08, 0.24)]])


if __name__ == "__main__":
    unittest.main()
