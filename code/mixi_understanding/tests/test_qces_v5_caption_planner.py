"""Checkpoint-free tests for the QCES-v5 caption/planner baseline."""

from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from mixi_understanding.data.qces_v5_schema import QCESV5Record
from mixi_understanding.scripts.evaluate_qces_audioqa import (
    AudioFlamingo3OptionScorer,
)
from mixi_understanding.scripts.evaluate_qces_v5_caption_planner_audiosep import (
    _role_metrics,
    load_planner_items,
)
from mixi_understanding.scripts.generate_qces_v5_caption_planner import (
    FORMAT_VERSION,
    build_planner_prompt,
    parse_caption_generation,
    parse_planner_generation,
    planner_diagnostics,
)


def _record(*, no_evidence: bool = False) -> QCESV5Record:
    record = Mock(spec=QCESV5Record)
    record.sample_id = "val_000000_base_00"
    record.no_evidence = no_evidence
    record.evidence_event_ids = () if no_evidence else ("anchor", "answer")
    events = {
        "anchor": SimpleNamespace(label="croak"),
        "answer": SimpleNamespace(label="buzz"),
    }
    record.event_by_id.side_effect = events.__getitem__
    return record


class CaptionPlannerParsingTest(unittest.TestCase):
    def test_caption_json_is_strict_and_canonical(self) -> None:
        raw = (
            '{"events":[{"order":1,"sound":"frog croaking"},'
            '{"order":2,"sound":"buzzing"}]}'
        )
        status, events, canonical = parse_caption_generation(raw)

        self.assertEqual(status, "valid")
        self.assertEqual([event["order"] for event in events], [1, 2])
        self.assertEqual(json.loads(canonical), {"events": events})

        invalid, _, _ = parse_caption_generation(
            '{"events":[{"order":2,"sound":"buzzing"}]}'
        )
        self.assertEqual(invalid, "invalid_schema")

    def test_planner_contract_uses_only_caption_and_question(self) -> None:
        caption = '{"events":[{"order":1,"sound":"croak"}]}'
        question = "What occurs immediately after the first croak?"
        prompt = build_planner_prompt(caption, question)

        self.assertIn(caption, prompt)
        self.assertIn(question, prompt)
        self.assertNotIn("answer_options", prompt)
        self.assertNotIn("gold_answer", prompt)

    def test_invalid_planner_falls_back_to_raw_question_not_oracle(self) -> None:
        question = "What occurs after the croak?"
        parsed = parse_planner_generation("not json", question)

        self.assertEqual(parsed["decision"], "extract")
        self.assertEqual(parsed["source_phrase"], question)
        self.assertTrue(parsed["fallback_used"])

    def test_post_hoc_diagnostics_do_not_change_prompt(self) -> None:
        diagnostics = planner_diagnostics(
            _record(),
            decision="extract",
            source_phrase="croak and buzz",
            vocabulary=("croak", "buzz", "music"),
        )

        self.assertTrue(diagnostics["decision_correct_↑"])
        self.assertTrue(diagnostics["normalized_label_set_exact_match_↑"])
        self.assertEqual(diagnostics["normalized_label_recall_↑"], 1.0)


class PlannerReportBindingTest(unittest.TestCase):
    def test_exact_manifest_and_no_oracle_input_contract_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "val.jsonl"
            manifest.write_text('{"id":"val_000000_base_00"}\n', encoding="utf-8")
            contract = {
                "caption_stage": {
                    "gold_answer": False,
                    "answer_options": False,
                    "event_labels_or_stems": False,
                    "timestamps": False,
                },
                "planner_stage": {
                    "predicted_caption": True,
                    "question": True,
                    "gold_answer": False,
                    "answer_options": False,
                    "event_labels_or_stems": False,
                    "timestamps": False,
                },
            }
            report = root / "planner_report.json"
            payload = {
                "format": FORMAT_VERSION,
                "manifest": str(manifest.resolve()),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "run_fingerprint": "f" * 64,
                "caption_prompt_version": "caption_v1",
                "planner_prompt_version": "planner_v1",
                "model_input_contract": contract,
                "items": [
                    {
                        "id": "val_000000_base_00",
                        "decision": "extract",
                        "source_phrase": "croak and buzz",
                    }
                ],
            }
            report.write_text(json.dumps(payload), encoding="utf-8")

            items, provenance = load_planner_items(
                report,
                manifest=manifest.resolve(),
                record_ids=["val_000000_base_00"],
            )
            self.assertEqual(items["val_000000_base_00"]["decision"], "extract")
            self.assertEqual(provenance["manifest_sha256"], payload["manifest_sha256"])

            payload["model_input_contract"]["planner_stage"]["gold_answer"] = True
            report.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden oracle"):
                load_planner_items(
                    report,
                    manifest=manifest.resolve(),
                    record_ids=["val_000000_base_00"],
                )


class CaptionPlannerWaveformMetricTest(unittest.TestCase):
    def test_role_metrics_include_weakest_scale_dependent_score(self) -> None:
        anchor = torch.tensor([0.0, 0.4, -0.2, 0.0, 0.0, 0.0])
        answer = torch.tensor([0.0, 0.0, 0.0, 0.3, -0.1, 0.0])
        evidence = anchor + answer
        metrics = _role_metrics(
            no_evidence=False,
            evidence=evidence,
            anchor_mask=torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            answer_mask=torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
            anchor_target=anchor,
            answer_target=answer,
        )

        self.assertIsNotNone(metrics["weakest_role_sd_sdr_db_↑"])
        self.assertTrue(all(key.endswith(("↑", "↓")) for key in metrics))


class AF3GenerationContractTest(unittest.TestCase):
    def test_generate_text_decodes_only_the_continuation(self) -> None:
        scorer = object.__new__(AudioFlamingo3OptionScorer)
        scorer.torch = torch
        scorer._prepare = lambda *_args: {"input_ids": torch.tensor([[10, 11]])}
        scorer._model_forward_context = contextlib.nullcontext

        class FakeModel:
            @staticmethod
            def generate(**kwargs):
                self.assertFalse(kwargs["do_sample"])
                self.assertEqual(kwargs["num_beams"], 1)
                return torch.tensor([[10, 11, 21, 22]])

        scorer.model = FakeModel()
        scorer.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(
                decode=lambda ids, skip_special_tokens: f"decoded:{ids}"
            )
        )

        result = scorer.generate_text("prompt", None, None, max_new_tokens=8)
        self.assertEqual(result, "decoded:[21, 22]")


if __name__ == "__main__":
    unittest.main()
