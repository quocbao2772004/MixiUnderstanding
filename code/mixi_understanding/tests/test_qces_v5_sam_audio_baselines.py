"""Checkpoint-free protocol tests for QCES-v5 SAM-Audio baselines."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.scripts.evaluate_sam_audio_baselines import (
    V5_MODE_REGISTRY,
    parse_args,
    sam_v5_item_metrics,
    summarize_v5_items,
    v5_anchors,
    v5_oracle_description,
    validate_predicted_spans,
    write_item_artifacts,
)


class SAMV5PromptProtocolTest(unittest.TestCase):
    @staticmethod
    def _record(*, no_evidence: bool = False) -> QCESV5Record:
        record = Mock(spec=QCESV5Record)
        record.sample_id = "val_000000_base_00"
        record.duration_seconds = 10.0
        record.no_evidence = no_evidence
        record.absent_labels = ("buzz", "croak") if no_evidence else ()
        record.evidence_event_ids = () if no_evidence else ("late", "early")
        record.anchor_intervals = () if no_evidence else ((2.0, 3.0),)
        record.answer_intervals = () if no_evidence else ((0.5, 1.5),)
        events = {
            "late": SimpleNamespace(event_id="late", label="buzz", onset_seconds=2.0),
            "early": SimpleNamespace(
                event_id="early", label="croak", onset_seconds=0.5
            ),
        }
        record.event_by_id.side_effect = events.__getitem__
        return record

    def test_oracle_description_is_ordered_and_deduplicates_same_class(self) -> None:
        record = self._record()
        self.assertEqual(v5_oracle_description(record), "frog croaking and buzzing")

        record.event_by_id.side_effect = {
            "late": SimpleNamespace(event_id="late", label="buzz", onset_seconds=2.0),
            "early": SimpleNamespace(event_id="early", label="buzz", onset_seconds=0.5),
        }.__getitem__
        self.assertEqual(v5_oracle_description(record), "buzzing")

    def test_oracle_and_predicted_spans_remain_distinct(self) -> None:
        record = self._record()
        predicted = {record.sample_id: [[4.0, 5.0]]}

        self.assertEqual(
            v5_anchors(record, "oracle_span_only", predicted),
            [["+", 0.5, 1.5], ["+", 2.0, 3.0]],
        )
        self.assertEqual(
            v5_anchors(record, "predicted_span_only", predicted),
            [["+", 4.0, 5.0]],
        )
        self.assertEqual(
            V5_MODE_REGISTRY["oracle_span_only"]["access"],
            "oracle_upper_bound",
        )


class SAMV5MetricProtocolTest(unittest.TestCase):
    def test_native_residual_consistency_is_not_arithmetic_by_construction(
        self,
    ) -> None:
        target = torch.tensor([0.0, 0.6, -0.4, 0.2, 0.0, 0.0])
        target_residual = torch.tensor([0.1, 0.0, 0.0, 0.0, -0.2, 0.1])
        mixture = target + target_residual
        native_residual = target_residual + 0.05

        metrics, _ = sam_v5_item_metrics(
            no_evidence=False,
            evidence=target,
            residual=native_residual,
            mixture=mixture,
            target=target,
            target_residual=target_residual,
        )

        self.assertAlmostEqual(
            metrics["native_mixture_consistency_l1_↓"], 0.05, places=6
        )
        self.assertTrue(all(key.endswith(("↑", "↓")) for key in metrics))

    def test_summary_preserves_metric_directions(self) -> None:
        target = torch.tensor([0.0, 0.6, -0.4, 0.2, 0.0, 0.0])
        residual = torch.tensor([0.1, 0.0, 0.0, 0.0, -0.2, 0.1])
        metrics, descriptives = sam_v5_item_metrics(
            no_evidence=False,
            evidence=target,
            residual=residual,
            mixture=target + residual,
            target=target,
            target_residual=residual,
        )
        summary = summarize_v5_items(
            [
                {
                    "no_evidence": False,
                    "metrics": metrics,
                    "descriptives": descriptives,
                }
            ]
        )
        metric_keys = [
            key
            for key in summary
            if key not in {"record_count", "answerable_count", "no_evidence_count"}
        ]
        self.assertTrue(all(key.endswith(("↑", "↓")) for key in metric_keys))


class SAMV5ProvenanceAndRenderingTest(unittest.TestCase):
    def test_predicted_spans_are_bound_to_exact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "val.jsonl"
            manifest.write_text('{"id":"a"}\n', encoding="utf-8")
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            report = root / "spans.json"
            report.write_text(
                json.dumps(
                    {
                        "manifest_sha256": digest,
                        "items": [
                            {
                                "id": "a",
                                "predicted_evidence_intervals": [[0.1, 0.3]],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            spans, provenance = validate_predicted_spans(
                report, manifest=manifest, record_ids=["a"]
            )
            self.assertEqual(spans, {"a": [[0.1, 0.3]]})
            self.assertEqual(provenance["manifest_sha256"], digest)

            manifest.write_text('{"id":"different"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different manifest"):
                validate_predicted_spans(report, manifest=manifest, record_ids=["a"])

    def test_no_render_keeps_metadata_and_writes_no_wav(self) -> None:
        args = parse_args(
            [
                "--manifest",
                "val.jsonl",
                "--model",
                "sam-small",
                "--output-dir",
                "out",
                "--no-render-audio",
            ]
        )
        with tempfile.TemporaryDirectory() as temporary, patch(
            "mixi_understanding.scripts.evaluate_sam_audio_baselines.torchaudio.save"
        ) as save:
            output = Path(temporary) / "item"
            item = {"id": "a", "metrics": {"evidence_l1_↓": 0.0}}
            write_item_artifacts(
                question_dir=output,
                item=item,
                evidence=torch.ones(4),
                residual=torch.zeros(4),
                sample_rate=32000,
                render_audio=not args.no_render_audio,
            )

            save.assert_not_called()
            self.assertEqual(json.loads((output / "metadata.json").read_text()), item)


if __name__ == "__main__":
    unittest.main()
