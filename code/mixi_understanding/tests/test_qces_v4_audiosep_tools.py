"""Checkpoint-free tests for the QCES v4 AudioSep utility bridge."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from mixi_understanding.data.qces_v4_schema import QCESV4Record
from mixi_understanding.scripts.cache_audiosep_semantic_targets import (
    SUPPORTED_RECORD_TYPES as CACHE_RECORD_TYPES,
    oracle_training_prompts,
)
from mixi_understanding.scripts.evaluate_audiosep_baselines import oracle_prompt
from mixi_understanding.scripts.evaluate_qces import (
    SUMMARY_DIRECTIONS,
    SUPPORTED_RECORD_TYPES as EVAL_RECORD_TYPES,
    summary_with_directions,
)


class QCESV4AudioSepToolsTest(unittest.TestCase):
    @staticmethod
    def _answerable_record() -> QCESV4Record:
        record = Mock(spec=QCESV4Record)
        record.sample_id = "test_000000_00"
        record.no_evidence = False
        record.absent_label = None
        # Reverse IDs deliberately: oracle_prompt must restore timeline order.
        record.evidence_event_ids = ("answer", "anchor")
        events = {
            "anchor": SimpleNamespace(label="horn", onset_seconds=0.1),
            "answer": SimpleNamespace(label="bell", onset_seconds=0.4),
        }
        record.event_by_id.side_effect = events.__getitem__
        # This sentinel proves prompt construction does not copy record.answer.
        record.answer = "ANSWER_STRING_MUST_NOT_BE_READ"
        return record

    def test_v4_is_accepted_by_cache_and_evaluator(self) -> None:
        self.assertIn(QCESV4Record, CACHE_RECORD_TYPES)
        self.assertIn(QCESV4Record, EVAL_RECORD_TYPES)

    def test_training_prompt_uses_role_event_labels_like_v3(self) -> None:
        record = self._answerable_record()
        prompts = oracle_training_prompts([record])

        self.assertEqual(prompts[record.sample_id], "horn and bell")
        self.assertEqual(prompts[record.sample_id], oracle_prompt(record))
        self.assertNotIn(record.answer, prompts[record.sample_id])

    def test_directional_summary_marks_every_metric(self) -> None:
        summary = {name: float(index) for index, name in enumerate(SUMMARY_DIRECTIONS)}
        directional = summary_with_directions(summary)

        self.assertEqual(len(directional), len(summary))
        for name, direction in SUMMARY_DIRECTIONS.items():
            self.assertIn(f"{name}_{direction}", directional)


if __name__ == "__main__":
    unittest.main()
