"""Checkpoint-free tests for the QCES-v5 frozen-AudioSep baselines."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.scripts.evaluate_audiosep_baselines import (
    MIXTURE_MODE,
    MODES,
    ORACLE_SEMANTIC_GATE_MODE,
    ORACLE_SEMANTIC_MODE,
    RAW_QUESTION_MODE,
    V5_MODES,
    V5_MODE_REGISTRY,
    _v5_item_metrics,
    _v5_metadata,
    oracle_prompt,
    parse_args,
    prompt_for,
    summarize_v5_items,
    write_item_artifacts,
)


class QCESV5PromptProtocolTest(unittest.TestCase):
    @staticmethod
    def _record(*, no_evidence: bool = False) -> QCESV5Record:
        record = Mock(spec=QCESV5Record)
        record.sample_id = "val_000000_base_00"
        record.no_evidence = no_evidence
        record.absent_labels = ("buzz", "croak") if no_evidence else ()
        record.evidence_event_ids = () if no_evidence else ("late", "early")
        events = {
            "late": SimpleNamespace(
                event_id="late", label="buzz", onset_seconds=2.0
            ),
            "early": SimpleNamespace(
                event_id="early", label="croak", onset_seconds=0.5
            ),
        }
        record.event_by_id.side_effect = events.__getitem__
        record.question = "Which event starts immediately after the third croak?"
        return record

    def test_v5_union_prompt_is_timeline_ordered(self) -> None:
        record = self._record()

        self.assertEqual(
            oracle_prompt(record), "a frog croaking and a buzzing sound"
        )
        self.assertEqual(prompt_for(record, RAW_QUESTION_MODE), record.question)
        self.assertEqual(
            prompt_for(record, ORACLE_SEMANTIC_MODE),
            "a frog croaking and a buzzing sound",
        )

    def test_same_label_role_events_are_one_semantic_class_prompt(self) -> None:
        record = self._record()
        record.event_by_id.side_effect = {
            "late": SimpleNamespace(
                event_id="late", label="buzz", onset_seconds=2.0
            ),
            "early": SimpleNamespace(
                event_id="early", label="buzz", onset_seconds=0.5
            ),
        }.__getitem__

        self.assertEqual(oracle_prompt(record), "a buzzing sound")

    def test_no_evidence_oracle_prompt_uses_declared_absent_classes(self) -> None:
        record = self._record(no_evidence=True)

        self.assertEqual(
            oracle_prompt(record), "a buzzing sound and a frog croaking"
        )

    def test_mode_registry_separates_non_oracle_and_oracle_modes(self) -> None:
        self.assertEqual(
            MODES,
            ("question", "oracle_semantic", "oracle_semantic_oracle_gate"),
        )
        self.assertEqual(
            V5_MODES,
            (
                MIXTURE_MODE,
                RAW_QUESTION_MODE,
                ORACLE_SEMANTIC_MODE,
                ORACLE_SEMANTIC_GATE_MODE,
            ),
        )
        self.assertEqual(V5_MODE_REGISTRY[MIXTURE_MODE]["access"], "non_oracle")
        self.assertEqual(
            V5_MODE_REGISTRY[RAW_QUESTION_MODE]["access"], "non_oracle"
        )
        self.assertEqual(
            V5_MODE_REGISTRY[ORACLE_SEMANTIC_MODE]["access"],
            "oracle_upper_bound",
        )
        self.assertTrue(
            V5_MODE_REGISTRY[ORACLE_SEMANTIC_GATE_MODE]["uses_oracle_time"]
        )


class QCESV5MetricProtocolTest(unittest.TestCase):
    def test_item_metrics_have_directions_and_exact_arithmetic_residual(self) -> None:
        target = torch.tensor([0.0, 0.6, -0.4, 0.2, 0.0, 0.0])
        distractor = torch.tensor([0.1, 0.0, 0.0, 0.0, -0.2, 0.1])
        mixture = target + distractor

        metrics, descriptives = _v5_item_metrics(
            no_evidence=False,
            evidence=mixture,
            mixture=mixture,
            target=target,
            target_residual=distractor,
        )

        self.assertTrue(all(key.endswith(("↑", "↓")) for key in metrics))
        self.assertEqual(metrics["mixture_consistency_l1_sanity_↓"], 0.0)
        self.assertAlmostEqual(metrics["evidence_si_sdri_db_↑"], 0.0, places=6)
        self.assertAlmostEqual(metrics["evidence_sd_sdri_db_↑"], 0.0, places=6)
        self.assertIsNone(metrics["no_evidence_retained_ratio_↓"])
        self.assertEqual(descriptives["evidence_retained_ratio"], 1.0)

    def test_no_evidence_retention_is_a_down_metric(self) -> None:
        mixture = torch.tensor([0.25, -0.5, 0.75, -0.25])
        target = torch.zeros_like(mixture)

        metrics, _ = _v5_item_metrics(
            no_evidence=True,
            evidence=mixture,
            mixture=mixture,
            target=target,
            target_residual=mixture,
        )

        self.assertEqual(metrics["no_evidence_retained_ratio_↓"], 1.0)
        self.assertIsNone(metrics["evidence_si_sdr_db_↑"])

    def test_summary_keeps_metric_arrows(self) -> None:
        target = torch.tensor([0.0, 0.6, -0.4, 0.2, 0.0, 0.0])
        mixture = target + torch.tensor([0.1, 0.0, 0.0, 0.0, -0.2, 0.1])
        metrics, descriptives = _v5_item_metrics(
            no_evidence=False,
            evidence=target,
            mixture=mixture,
            target=target,
            target_residual=mixture - target,
        )
        item = {
            "no_evidence": False,
            "metrics": metrics,
            "descriptives": descriptives,
        }

        summary = summarize_v5_items([item])

        metric_keys = [
            key
            for key in summary
            if key not in {"record_count", "answerable_count", "no_evidence_count"}
        ]
        self.assertTrue(all(key.endswith(("↑", "↓")) for key in metric_keys))

    def test_item_metadata_exposes_counterfactual_axes(self) -> None:
        record = Mock(spec=QCESV5Record)
        values = {
            "sample_id": "val_000000_base_00",
            "split": "val",
            "evaluation_axis": "development",
            "scene_id": "scene_val_000000_base",
            "scene_family_id": "family_val_000000",
            "variant_id": "base",
            "counterfactual_group_id": "family_val_000000:primary",
            "question_semantics_id": "after:croak:ordinal=3",
            "paraphrase_family_id": "validation_after_next",
            "question_index": 0,
            "question_type": "temporal_after",
            "relation": "after",
            "primary_counterfactual_probe": True,
            "mention_order_variant": "not_applicable",
            "same_label_repeat": True,
            "semantic_overlap": False,
            "hard_case_tags": ("same_label_instances",),
            "question": "What comes next?",
            "no_evidence": False,
        }
        for key, value in values.items():
            setattr(record, key, value)

        metadata = _v5_metadata(record)

        self.assertEqual(metadata["scene_family_id"], "family_val_000000")
        self.assertEqual(metadata["variant_id"], "base")
        self.assertEqual(metadata["relation"], "after")
        self.assertEqual(metadata["hard_case_tags"], ["same_label_instances"])


class AudioRenderingContractTest(unittest.TestCase):
    @staticmethod
    def _required_cli() -> list[str]:
        return [
            "--manifest",
            "val.jsonl",
            "--audiosep-root",
            "audiosep",
            "--audiosep-config",
            "audiosep.yaml",
            "--audiosep-checkpoint",
            "audiosep.bin",
            "--output-dir",
            "output",
        ]

    def test_legacy_default_still_renders_both_wavs(self) -> None:
        args = parse_args(self._required_cli())
        self.assertFalse(args.no_render_audio)
        item = {"id": "legacy", "metrics": {"evidence_l1_↓": 0.0}}
        with tempfile.TemporaryDirectory() as temporary, patch(
            "mixi_understanding.scripts.evaluate_audiosep_baselines.sf.write"
        ) as write:
            output = Path(temporary) / "question"
            write_item_artifacts(
                question_dir=output,
                item=item,
                evidence=torch.ones(4),
                residual=torch.zeros(4),
                sample_rate=32000,
                render_audio=not args.no_render_audio,
            )

            self.assertEqual(write.call_count, 2)
            self.assertEqual(
                write.call_args_list[0].args[0].name, "predicted_evidence.wav"
            )
            self.assertEqual(
                write.call_args_list[1].args[0].name, "predicted_residual.wav"
            )
            self.assertEqual(
                json.loads((output / "metadata.json").read_text()), item
            )

    def test_v5_no_render_keeps_numeric_metadata_and_writes_no_wav(self) -> None:
        args = parse_args([*self._required_cli(), "--no-render-audio"])
        self.assertTrue(args.no_render_audio)
        item = {
            "id": "val_000000_base_00",
            "scene_family_id": "family_val_000000",
            "variant_id": "base",
            "relation": "after",
            "metrics": {
                "evidence_si_sdr_db_↑": 1.5,
                "evidence_l1_↓": 0.1,
            },
        }
        with tempfile.TemporaryDirectory() as temporary, patch(
            "mixi_understanding.scripts.evaluate_audiosep_baselines.sf.write"
        ) as write:
            output = Path(temporary) / "question"
            write_item_artifacts(
                question_dir=output,
                item=item,
                evidence=torch.ones(4),
                residual=torch.zeros(4),
                sample_rate=32000,
                render_audio=not args.no_render_audio,
            )

            write.assert_not_called()
            self.assertFalse((output / "predicted_evidence.wav").exists())
            self.assertFalse((output / "predicted_residual.wav").exists())
            self.assertEqual(
                json.loads((output / "metadata.json").read_text()), item
            )


if __name__ == "__main__":
    unittest.main()
