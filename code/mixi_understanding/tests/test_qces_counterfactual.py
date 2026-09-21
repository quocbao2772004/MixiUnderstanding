"""Focused tests for paired QCES-v5 counterfactual evidence equivariance."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.composer import PromptComposition
from mixi_understanding.qces.config import QCESConfig
from mixi_understanding.qces.counterfactual import (
    CounterfactualBatchSampler,
    _oracle_evidence_probability,
    build_counterfactual_group_plan,
    counterfactual_objectives,
)
from mixi_understanding.qces.losses import LossWeights, QCESLoss
from mixi_understanding.qces.model import QCESModel, QCESOutput
from mixi_understanding.qces.separators import SeparationOutput
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.train_qces import evaluate_epoch


def _stub_record(**updates) -> QCESV5Record:
    """Create a planner-only typed record without duplicating the full schema fixture."""

    values = {
        "sample_id": "sample",
        "scene_id": "scene_base",
        "split": "train",
        "scene_family_id": "family",
        "variant_id": "base",
        "question_semantics_id": "after:a:ordinal=1",
        "counterfactual_group_id": "single",
        "question": "What follows a?",
        "answer": "b",
        "answer_options": ("a", "b", "c", "d", "no_evidence"),
        "relation": "after",
        "no_evidence": False,
        "surface_control_group_id": None,
        "mention_order_variant": "not_applicable",
        "anchor_event_ids": ("a",),
        "answer_event_ids": ("b",),
        "evidence_event_ids": ("a", "b"),
        "mixture_path": "audio/scene.wav",
        "primary_counterfactual_probe": False,
    }
    values.update(updates)
    # The group planner deliberately accepts only QCESV5Record, while these
    # tests isolate planner logic from the already-covered schema parser.
    record = object.__new__(QCESV5Record)
    for name, value in values.items():
        object.__setattr__(record, name, value)
    return record


def _complete_records() -> list[QCESV5Record]:
    surface_common = {
        "scene_id": "scene_surface_base",
        "scene_family_id": "family_surface",
        "variant_id": "base",
        "question_semantics_id": "first:a|b",
        "counterfactual_group_id": "surface-pair",
        "relation": "first",
        "answer": "a",
        "anchor_event_ids": ("b",),
        "answer_event_ids": ("a",),
        "evidence_event_ids": ("a", "b"),
        "mixture_path": "audio/surface.wav",
        "surface_control_group_id": "surface-pair",
    }
    surface = [
        _stub_record(
            sample_id="surface-forward",
            mention_order_variant="forward",
            **surface_common,
        ),
        _stub_record(
            sample_id="surface-reversed",
            mention_order_variant="reversed",
            **surface_common,
        ),
    ]
    primary_common = {
        "scene_family_id": "family-primary",
        "question_semantics_id": "after:frog:ordinal=3",
        "counterfactual_group_id": "family-primary:primary",
        "question": "What follows the third frog?",
        "answer_options": ("bell", "rain", "no_evidence", "horn", "dog"),
        "primary_counterfactual_probe": True,
    }
    primary = [
        _stub_record(
            sample_id="primary-base",
            scene_id="scene_primary_base",
            variant_id="base",
            answer="bell",
            evidence_event_ids=("frog3", "bell"),
            **primary_common,
        ),
        _stub_record(
            sample_id="primary-swap",
            scene_id="scene_primary_swap",
            variant_id="order_swap",
            answer="rain",
            evidence_event_ids=("frog3", "rain"),
            **primary_common,
        ),
        _stub_record(
            sample_id="primary-drop",
            scene_id="scene_primary_drop",
            variant_id="anchor_drop",
            answer="no_evidence",
            no_evidence=True,
            anchor_event_ids=(),
            answer_event_ids=(),
            evidence_event_ids=(),
            **primary_common,
        ),
    ]
    question = [
        _stub_record(
            sample_id="question-left",
            scene_id="scene-question",
            scene_family_id="family-question",
            question_semantics_id="after:a:ordinal=1",
            evidence_event_ids=("a", "b"),
        ),
        _stub_record(
            sample_id="question-right",
            scene_id="scene-question",
            scene_family_id="family-question",
            question_semantics_id="before:d:ordinal=1",
            anchor_event_ids=("d",),
            answer_event_ids=("c",),
            evidence_event_ids=("c", "d"),
        ),
    ]
    return [*surface, *primary, *question]


class CounterfactualGroupPlanTest(unittest.TestCase):
    def test_plan_and_sampler_are_complete_disjoint_and_deterministic(self) -> None:
        records = _complete_records()
        plan = build_counterfactual_group_plan(
            records,
            enable_surface=True,
            enable_family=True,
            enable_question=True,
        )
        self.assertEqual(len(plan.surface_pairs), 1)
        self.assertEqual(len(plan.primary_triplets), 1)
        self.assertEqual(len(plan.question_pairs), 1)
        sampler = CounterfactualBatchSampler(
            plan, batch_size=3, seed=17, shuffle=True
        )
        first = list(sampler)
        second = list(sampler)
        self.assertEqual(first, second)
        self.assertEqual(
            sorted(index for batch in first for index in batch),
            list(range(len(records))),
        )
        self.assertEqual(
            len([index for batch in first for index in batch]), len(records)
        )
        for required in (*plan.surface_pairs, *plan.primary_triplets, *plan.question_pairs):
            self.assertTrue(
                any(set(required).issubset(batch) for batch in first), required
            )
        sampler.set_epoch(1)
        self.assertEqual(
            sorted(index for batch in sampler for index in batch),
            list(range(len(records))),
        )

    def test_required_groups_fail_closed(self) -> None:
        records = _complete_records()
        with self.assertRaisesRegex(ValueError, "incomplete surface"):
            build_counterfactual_group_plan(
                records[1:],
                enable_surface=True,
                enable_family=False,
                enable_question=False,
            )
        without_drop = [record for record in records if record.sample_id != "primary-drop"]
        with self.assertRaisesRegex(ValueError, "incomplete primary"):
            build_counterfactual_group_plan(
                without_drop,
                enable_surface=False,
                enable_family=True,
                enable_question=False,
            )
        same_oracle = [
            _stub_record(sample_id="q1", scene_id="one", question_semantics_id="q1"),
            _stub_record(sample_id="q2", scene_id="one", question_semantics_id="q2"),
        ]
        with self.assertRaisesRegex(ValueError, "incomplete same-scene"):
            build_counterfactual_group_plan(
                same_oracle,
                enable_surface=False,
                enable_family=False,
                enable_question=True,
            )

    def test_batch_size_cannot_split_a_primary_triplet(self) -> None:
        plan = build_counterfactual_group_plan(
            _complete_records()[2:5],
            enable_surface=False,
            enable_family=True,
            enable_question=False,
        )
        with self.assertRaisesRegex(ValueError, "group of size 3"):
            CounterfactualBatchSampler(plan, batch_size=2, seed=1, shuffle=True)


def _paired_output_and_batch() -> tuple[QCESOutput, dict]:
    batch_size, frames, samples = 7, 4, 8
    desired_union = torch.zeros(batch_size, frames)
    desired_union[0, 0] = desired_union[1, 0] = 1.0
    desired_union[2, 0] = 1.0
    desired_union[3, 1] = 1.0
    desired_union[5, 0] = 1.0
    desired_union[6, 2] = 1.0
    role_logits = torch.empty(batch_size, frames, 3)
    role_logits[..., 0] = torch.where(desired_union > 0, -20.0, 20.0)
    role_logits[..., 1] = torch.where(desired_union > 0, 20.0, -20.0)
    role_logits[..., 2] = -20.0
    role_probability = role_logits.softmax(dim=-1)
    evidence_probability = 1.0 - role_probability[..., 0]
    semantic = torch.ones(batch_size, 3)
    semantic[2:] = torch.arange(1, 16, dtype=torch.float32).reshape(5, 3)
    no_evidence_logit = torch.full((batch_size,), -20.0)
    no_evidence_logit[4] = 20.0
    evidence = torch.zeros(batch_size, samples)
    for index, frame, amplitude in (
        (0, 0, 1.0),
        (1, 0, 1.0),
        (2, 0, 3.0),
        (3, 1, 4.0),
        (5, 0, 6.0),
        (6, 2, 7.0),
    ):
        evidence[index, 2 * frame : 2 * frame + 2] = amplitude
    # A/C controls share the full scene waveform even though C evidence targets
    # differ with the question.
    mixture = torch.full((batch_size, samples), 10.0)
    anchor_mask = torch.zeros_like(evidence)
    answer_mask = torch.zeros_like(evidence)
    for index, frame in ((0, 0), (1, 0), (2, 0), (3, 1), (5, 0), (6, 2)):
        anchor_mask[index, 2 * frame] = 1.0
        answer_mask[index, 2 * frame + 1] = 1.0
    composition = PromptComposition(
        semantic_condition=semantic,
        role_logits=role_logits,
        evidence_probability=evidence_probability,
        no_evidence_logit=no_evidence_logit,
        frame_features=torch.zeros(batch_size, frames, 2),
        frame_hop_samples=2,
    )
    separation = SeparationOutput(
        evidence=evidence.clone(),
        residual=mixture - evidence,
        mask=evidence_probability,
        raw_mask=None,
        raw_evidence=None,
        mixture_error=torch.zeros(batch_size),
    )
    output = QCESOutput(composition=composition, separation=separation)
    batch = {
        "mixture": mixture,
        "evidence": evidence,
        "residual": mixture - evidence,
        "anchor_mask": anchor_mask,
        "answer_mask": answer_mask,
        "no_evidence": torch.tensor([0, 0, 0, 0, 1, 0, 0], dtype=torch.float32),
        "counterfactual_groups": {
            "surface_pairs": [(0, 1)],
            "primary_triplets": [(2, 3, 4)],
            "question_pairs": [(5, 6)],
        },
    }
    return output, batch


class CounterfactualObjectiveTest(unittest.TestCase):
    def test_exact_equivariant_predictions_have_expected_metrics(self) -> None:
        output, batch = _paired_output_and_batch()
        losses, metrics, counts = counterfactual_objectives(output, batch)
        self.assertEqual(counts, {"surface": 1, "family": 1, "question": 1})
        for name, value in losses.items():
            self.assertLess(float(value), 1e-5, name)
        self.assertAlmostEqual(
            float(metrics["surface_semantic_cosine"]), 1.0, places=6
        )
        self.assertAlmostEqual(float(metrics["family_transition_accuracy"]), 1.0)
        self.assertGreater(float(metrics["family_temporal_delta_cosine"]), 0.999)
        self.assertGreater(float(metrics["question_evidence_delta_cosine"]), 0.999)

    def test_overlap_oracle_is_union_not_answer_overwrite(self) -> None:
        anchor = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        answer = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
        oracle = _oracle_evidence_probability(
            {"anchor_mask": anchor, "answer_mask": answer},
            frames=4,
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(oracle, torch.tensor([[1.0, 1.0, 1.0, 0.0]])))

    def test_positive_paired_weight_requires_group_metadata(self) -> None:
        output, batch = _paired_output_and_batch()
        batch.pop("counterfactual_groups")
        criterion = QCESLoss(
            weights=LossWeights(surface_evidence_invariance=1.0),
            fft_sizes=(4,),
        )
        with self.assertRaisesRegex(ValueError, "validated paired group metadata"):
            criterion(output, batch)
        batch["counterfactual_groups"] = {
            "surface_pairs": [],
            "primary_triplets": [],
            "question_pairs": [],
        }
        with self.assertRaisesRegex(ValueError, "validated complete group plan"):
            criterion(output, batch)

    def test_all_paired_heads_receive_finite_gradients(self) -> None:
        reference, batch = _paired_output_and_batch()
        semantic = reference.composition.semantic_condition.detach().clone()
        semantic[1, 0] += 0.25
        semantic.requires_grad_()
        role_logits = reference.composition.role_logits.detach().clone()
        role_logits[3, 0, 0] -= 0.5
        role_logits.requires_grad_()
        role_probability = role_logits.softmax(dim=-1)
        no_evidence_logit = torch.full((7,), -2.0, requires_grad=True)
        predicted_evidence = reference.evidence.detach().clone()
        predicted_evidence[1, 0] += 0.2
        predicted_evidence[3, 0] += 0.3
        predicted_evidence.requires_grad_()
        composition = PromptComposition(
            semantic_condition=semantic,
            role_logits=role_logits,
            evidence_probability=1.0 - role_probability[..., 0],
            no_evidence_logit=no_evidence_logit,
            frame_features=reference.composition.frame_features,
            frame_hop_samples=2,
        )
        output = QCESOutput(
            composition=composition,
            separation=SeparationOutput(
                evidence=predicted_evidence,
                residual=batch["mixture"] - predicted_evidence,
                mask=composition.evidence_probability,
                raw_mask=None,
                raw_evidence=None,
                mixture_error=torch.zeros(7),
            ),
        )
        losses, _, _ = counterfactual_objectives(output, batch)
        torch.stack(tuple(losses.values())).sum().backward()
        for tensor in (semantic, role_logits, no_evidence_logit, predicted_evidence):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_paired_objective_does_not_reinvoke_separator(self) -> None:
        class CountingSeparator(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, mixture, composition):
                self.calls += 1
                gate = torch.nn.functional.interpolate(
                    composition.evidence_probability[:, None],
                    size=mixture.size(-1),
                    mode="linear",
                    align_corners=False,
                )[:, 0]
                evidence = mixture * gate
                return SeparationOutput(
                    evidence=evidence,
                    residual=mixture - evidence,
                    mask=gate,
                    raw_mask=None,
                    raw_evidence=None,
                    mixture_error=torch.zeros(mixture.size(0)),
                )

        config = QCESConfig(
            sample_rate=8000,
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
        separator = CountingSeparator()
        model = QCESModel(config, separator=separator)
        mixture = torch.randn(1, 256).repeat(2, 1)
        tokens = StableHashTokenizer(128, 16).batch_encode(["first a b", "first b a"])
        output = model(mixture, tokens.input_ids, tokens.attention_mask)
        frames = output.composition.evidence_probability.size(1)
        batch = {
            "mixture": mixture,
            "evidence": mixture,
            "anchor_mask": torch.ones_like(mixture),
            "answer_mask": torch.zeros_like(mixture),
            "counterfactual_groups": {
                "surface_pairs": [(0, 1)],
                "primary_triplets": [],
                "question_pairs": [],
            },
        }
        counterfactual_objectives(output, batch)
        self.assertGreater(frames, 0)
        self.assertEqual(separator.calls, 1)

    def test_grouped_validation_reports_directional_metrics_once_per_group(self) -> None:
        output, batch = _paired_output_and_batch()
        batch.update(
            sample_ids=[f"sample-{index}" for index in range(7)],
            question_ids=torch.ones(7, 2, dtype=torch.long),
            question_mask=torch.ones(7, 2, dtype=torch.bool),
        )

        class FixedModel(nn.Module):
            def __init__(self, fixed_output) -> None:
                super().__init__()
                self.fixed_output = fixed_output
                self.calls = 0

            def forward(self, mixture, question_ids, question_mask):
                self.calls += 1
                return self.fixed_output

        model = FixedModel(output)
        result = evaluate_epoch(
            model,
            QCESLoss(
                weights=LossWeights(
                    surface_evidence_invariance=1.0,
                    family_temporal_delta=1.0,
                    question_evidence_delta=1.0,
                ),
                fft_sizes=(4,),
            ),
            [batch],
            torch.device("cpu"),
            None,
        )
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["counterfactual_surface_pair_count"], 1.0)
        self.assertEqual(result["counterfactual_primary_triplet_count"], 1.0)
        self.assertEqual(result["counterfactual_question_pair_count"], 1.0)
        self.assertIn("family_temporal_delta_cosine", result)
        self.assertIn("surface_evidence_relative_l1", result)


if __name__ == "__main__":
    unittest.main()
