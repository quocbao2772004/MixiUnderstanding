"""Focused evaluator contracts for QCES-v5 and frozen CLAP caches."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mixi_understanding.data.qces_v4_schema import QCESV4Record
from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    NO_FOUNDATION_FEATURES,
)
from mixi_understanding.qces.data import QCESExample
from mixi_understanding.scripts import evaluate_qces as evaluator
from mixi_understanding.scripts.train_qces import FoundationFeatureCache


def _v5_record(sample_id: str = "sample_a", *, no_evidence: bool = False):
    record = Mock(spec=QCESV5Record)
    record.schema_version = "qces_v5_scene_event_derived_v1"
    record.sample_id = sample_id
    record.scene_id = "scene_variant"
    record.scene_family_id = "family_007"
    record.variant_id = "base"
    record.counterfactual_group_id = "family_007:semantics"
    record.paraphrase_family_id = "before_prior"
    record.question_semantics_id = "before:Bell:ordinal=1"
    record.relation = "before"
    record.evaluation_axis = "iid"
    record.primary_counterfactual_probe = True
    record.same_label_repeat = True
    record.semantic_overlap = True
    record.max_polyphony = 3
    record.hard_case_tags = ("same_label_instances", "semantic_overlap")
    record.no_evidence = no_evidence
    record.anchor_intervals = () if no_evidence else ((0.0, 0.75),)
    record.answer_intervals = () if no_evidence else ((0.5, 1.0),)
    record.anchor_event_ids = () if no_evidence else ("anchor",)
    record.answer_event_ids = () if no_evidence else ("answer",)
    labels = {
        "anchor": SimpleNamespace(label="Bell"),
        "answer": SimpleNamespace(label="Bell"),
    }
    record.event_by_id.side_effect = labels.__getitem__
    return record


class QCESV5MetadataTest(unittest.TestCase):
    def test_v5_is_supported_and_emits_family_role_overlap_metadata(self) -> None:
        self.assertIn(QCESV5Record, evaluator.SUPPORTED_RECORD_TYPES)

        metadata = evaluator.v5_item_metadata(_v5_record())

        self.assertEqual(metadata["scene_family_id"], "family_007")
        self.assertEqual(metadata["variant_id"], "base")
        self.assertEqual(metadata["relation"], "before")
        self.assertTrue(metadata["same_role_label"])
        self.assertTrue(metadata["same_label_repeat"])
        self.assertTrue(metadata["role_windows_overlap"])
        self.assertTrue(metadata["semantic_overlap"])

    def test_no_evidence_same_role_label_is_undefined(self) -> None:
        metadata = evaluator.v5_item_metadata(
            _v5_record("sample_negative", no_evidence=True)
        )
        self.assertIsNone(metadata["same_role_label"])
        self.assertFalse(metadata["role_windows_overlap"])

    def test_v4_item_shape_is_unchanged(self) -> None:
        self.assertEqual(evaluator.v5_item_metadata(Mock(spec=QCESV4Record)), {})


class FoundationEvaluationContractTest(unittest.TestCase):
    def test_mode_preflight_fails_closed_and_legacy_rejects_cache(self) -> None:
        missing = SimpleNamespace(
            foundation_feature_cache=None,
            audiosep_root=None,
            audiosep_checkpoint=None,
        )
        with self.assertRaisesRegex(SystemExit, "foundation-feature-cache"):
            evaluator.validate_foundation_evaluation_args(
                missing, AUDIOSEP_CLAP_FOUNDATION_FEATURES
            )

        unexpected = SimpleNamespace(
            foundation_feature_cache=Path("unused-cache"),
            audiosep_root=None,
            audiosep_checkpoint=None,
        )
        with self.assertRaisesRegex(SystemExit, "only valid"):
            evaluator.validate_foundation_evaluation_args(
                unexpected, NO_FOUNDATION_FEATURES
            )

    def test_cache_loader_receives_exact_manifest_and_asset_identities(self) -> None:
        cache = FoundationFeatureCache({}, {}, {}, {"strict": True})
        args = SimpleNamespace(
            foundation_feature_cache=Path("cache"),
            audiosep_root=Path("audiosep"),
            audiosep_checkpoint=Path("audiosep.bin"),
        )
        records = [SimpleNamespace(sample_id="sample")]
        checkpoint_identity = {"sha256": "c" * 64, "size_bytes": 7}
        source_identity = {
            "sha256": "s" * 64,
            "hashed_file_count": 2,
            "included_suffixes": [".py"],
        }
        with patch.object(
            evaluator, "file_identity", return_value=checkpoint_identity
        ) as identify_checkpoint, patch.object(
            evaluator,
            "audiosep_source_tree_identity",
            return_value=source_identity,
        ) as identify_source, patch.object(
            evaluator,
            "load_foundation_feature_cache",
            return_value=cache,
        ) as strict_load:
            actual = evaluator.load_evaluation_foundation_cache(
                args, Path("evaluation.jsonl"), records
            )

        self.assertIs(actual, cache)
        identify_checkpoint.assert_called_once_with(Path("audiosep.bin"))
        identify_source.assert_called_once_with(Path("audiosep"))
        strict_load.assert_called_once_with(
            Path("cache"),
            Path("evaluation.jsonl"),
            records,
            "evaluation",
            audiosep_checkpoint_identity=checkpoint_identity,
            audiosep_source_identity=source_identity,
        )

    def test_forward_keeps_legacy_call_and_passes_both_clap_tensors(self) -> None:
        mixture = torch.randn(2, 16)
        question_ids = torch.ones(2, 3, dtype=torch.long)
        question_mask = torch.ones(2, 3, dtype=torch.bool)
        base_batch = {
            "mixture": mixture,
            "question_ids": question_ids,
            "question_mask": question_mask,
        }

        legacy = Mock(return_value="legacy-output")
        legacy.config = SimpleNamespace(foundation_feature_mode=NO_FOUNDATION_FEATURES)
        self.assertEqual(
            evaluator.forward_evaluation_batch(legacy, base_batch),
            "legacy-output",
        )
        legacy.assert_called_once_with(mixture, question_ids, question_mask)

        foundation = Mock(return_value="foundation-output")
        foundation.config = SimpleNamespace(
            foundation_feature_mode=AUDIOSEP_CLAP_FOUNDATION_FEATURES
        )
        question_clap = torch.randn(2, 512)
        scene_clap = torch.randn(2, 32, 512)
        enriched = {
            **base_batch,
            "question_clap": question_clap,
            "scene_clap": scene_clap,
        }
        self.assertEqual(
            evaluator.forward_evaluation_batch(foundation, enriched),
            "foundation-output",
        )
        foundation.assert_called_once_with(
            mixture,
            question_ids,
            question_mask,
            question_clap=question_clap,
            scene_clap=scene_clap,
        )

        with self.assertRaisesRegex(ValueError, "missing question_clap"):
            evaluator.forward_evaluation_batch(foundation, base_batch)


class _TinyV5Dataset:
    sample_rate = 8

    def __init__(self) -> None:
        self.records = [
            self._record("sample_a", 0, False),
            self._record("sample_b", 1, False),
            self._record("sample_negative", 2, True),
        ]
        mixture = torch.tensor(
            [-0.4, -0.2, 0.1, 0.5, -0.3, 0.2, 0.4, -0.1],
            dtype=torch.float32,
        )
        anchor_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]).float()
        answer_mask = 1.0 - anchor_mask
        self.examples = []
        for record in self.records:
            if record.no_evidence:
                evidence = torch.zeros_like(mixture)
                residual = mixture.clone()
                role_anchor = torch.zeros_like(mixture)
                role_answer = torch.zeros_like(mixture)
                active_anchor = torch.zeros_like(mixture)
                active_answer = torch.zeros_like(mixture)
            else:
                evidence = mixture.clone()
                residual = torch.zeros_like(mixture)
                role_anchor = mixture * anchor_mask
                role_answer = mixture * answer_mask
                active_anchor = anchor_mask
                active_answer = answer_mask
            self.examples.append(
                QCESExample(
                    sample_id=record.sample_id,
                    question=record.question,
                    answer=record.answer,
                    mixture=mixture.clone(),
                    evidence=evidence,
                    residual=residual,
                    anchor_stem=role_anchor,
                    answer_stem=role_answer,
                    anchor_mask=active_anchor,
                    answer_mask=active_answer,
                    no_evidence=torch.tensor(record.no_evidence),
                )
            )

    @staticmethod
    def _record(sample_id: str, question_index: int, no_evidence: bool):
        record = _v5_record(sample_id, no_evidence=no_evidence)
        record.question_index = question_index
        record.question_type = "temporal_before"
        record.question = f"Question {question_index}?"
        record.answer = "no_evidence" if no_evidence else "Bell"
        record.sample_rate = 8
        record.duration_seconds = 1.0
        return record

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> QCESExample:
        return self.examples[index]


class _HalfGainModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            vocab_size=128,
            max_question_tokens=16,
            foundation_feature_mode=NO_FOUNDATION_FEATURES,
        )

    def eval(self):
        return self

    def __call__(self, mixture, question_ids, question_mask):  # type: ignore[no-untyped-def]
        batch_size = mixture.size(0)
        evidence = 0.5 * mixture
        composition = SimpleNamespace(
            evidence_probability=torch.ones(batch_size, 4),
            no_evidence_logit=torch.tensor([-10.0, -10.0, 10.0], device=mixture.device)[
                :batch_size
            ],
            frame_hop_samples=2,
        )
        return SimpleNamespace(
            evidence=evidence,
            residual=mixture - evidence,
            composition=composition,
        )


class QCESV5EvaluatorIntegrationTest(unittest.TestCase):
    def test_report_contains_sd_sdr_arrows_and_family_metadata(self) -> None:
        dataset = _TinyV5Dataset()
        checkpoint = {
            "format": "qces_v1",
            "backend": "mask",
            "config": {"foundation_feature_mode": NO_FOUNDATION_FEATURES},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "evaluation"
            argv = [
                "evaluate_qces.py",
                "--checkpoint",
                str(root / "checkpoint.pt"),
                "--manifest",
                str(root / "evaluation.jsonl"),
                "--output-dir",
                str(output),
                "--batch-size",
                "3",
                "--device",
                "cpu",
                "--no-render-audio",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                evaluator.torch, "load", return_value=checkpoint
            ), patch.object(
                evaluator, "load_qces_checkpoint", return_value=_HalfGainModel()
            ), patch.object(
                evaluator, "QCESManifestDataset", return_value=dataset
            ):
                evaluator.main()

            report = json.loads(
                (output / "evaluation_report.json").read_text(encoding="utf-8")
            )
            self.assertFalse((output / "scene_variant").exists())

        item = report["items"][0]
        self.assertEqual(item["scene_family_id"], "family_007")
        self.assertIn("evidence_sd_sdr", item)
        self.assertIn("evidence_sd_sdri", item)
        self.assertIn("anchor_sd_sdr", item)
        self.assertIn("answer_sd_sdr", item)
        self.assertIn("weakest_role_sd_sdr", item)
        for name in (
            "evidence_sd_sdr_answerable",
            "evidence_sd_sdri_answerable",
            "anchor_sd_sdr_answerable",
            "answer_sd_sdr_answerable",
            "weakest_role_sd_sdr_answerable",
            "no_evidence_balanced_accuracy",
            "no_evidence_auroc",
            "no_evidence_f1",
            "no_evidence_recall",
        ):
            self.assertIn(f"{name}_↑", report["summary_with_directions"])
        self.assertIn(
            "answerable_false_silence_rate_↓",
            report["summary_with_directions"],
        )
        self.assertEqual(report["summary"]["no_evidence_balanced_accuracy"], 1.0)
        self.assertEqual(report["summary"]["no_evidence_auroc"], 1.0)

    def test_binary_auroc_handles_ties_without_majority_shortcut(self) -> None:
        self.assertEqual(
            evaluator.binary_auroc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]),
            0.5,
        )
        self.assertEqual(
            evaluator.binary_auroc([0.9, 0.8, 0.2, 0.1], [True, False, True, False]),
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
