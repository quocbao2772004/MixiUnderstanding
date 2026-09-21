from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mixi_understanding.data.qces_real10_schema import (
    DURATION_SECONDS,
    NUM_CHANNELS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    SCORING_SCHEMA_VERSION,
    inference_record_fingerprint,
    parse_scoring_manifest,
    project_scoring_fields_to_inference,
)
from mixi_understanding.qces.real10_prediction import write_deterministic_float_wav
from mixi_understanding.qces.real10_scoring import ValidatedScoringInputs
from mixi_understanding.scripts.evaluate_qces_audioqa import _sha256_file
from mixi_understanding.scripts.evaluate_qces_real10_audioqa import (
    ALL_CONDITIONS,
    Real10AudioQAError,
    build_input_descriptors,
    build_report,
    materialize_bound_input,
)


def scoring_row(*, scene_index: int, question_index: int, mixture_sha256: str) -> dict:
    scene_id = f"scene_{scene_index}"
    no_evidence = question_index == 1
    relation = "before" if no_evidence else "after"
    row = {
        "schema_version": SCORING_SCHEMA_VERSION,
        "id": f"{scene_id}__q{question_index}",
        "scene_id": scene_id,
        "scene_family_id": scene_id,
        "split": "real_dev",
        "question_index": question_index,
        "question_type": f"temporal_{relation}",
        "relation": relation,
        "question": "What sound is temporally related to the described sound?",
        "sample_rate": SAMPLE_RATE,
        "num_channels": NUM_CHANNELS,
        "num_samples": NUM_SAMPLES,
        "duration_seconds": DURATION_SECONDS,
        "mixture_path": f"audio/{scene_id}.wav",
        "mixture_sha256": mixture_sha256,
        "inference_record_sha256": "0" * 64,
        "answer_options": ["answer", "anchor", "noise", "music", "no_evidence"],
        "answer": "no_evidence" if no_evidence else "answer",
        "answer_option_index": 4 if no_evidence else 0,
        "no_evidence": no_evidence,
        "creator_id": f"creator-{scene_index}",
        "anchor_event_ids": [] if no_evidence else ["anchor"],
        "answer_event_ids": [] if no_evidence else ["answer"],
        "evidence_event_ids": [] if no_evidence else ["anchor", "answer"],
        "anchor_intervals": [] if no_evidence else [[1.0, 2.0]],
        "answer_intervals": [] if no_evidence else [[3.0, 4.0]],
        "evidence_intervals": ([] if no_evidence else [[1.0, 2.0], [3.0, 4.0]]),
        "upstream_tacos_split": "development",
        "clean_reference_stems_available": False,
        "waveform_sdr_evaluation_allowed": False,
    }
    row["inference_record_sha256"] = inference_record_fingerprint(
        project_scoring_fields_to_inference(row)
    )
    return row


class Real10AudioQAFixture:
    def __init__(self, root: Path) -> None:
        self.dataset_root = root / "dataset"
        self.prediction_root = root / "predictions"
        (self.dataset_root / "audio").mkdir(parents=True)
        self.prediction_root.mkdir()
        self.scoring_manifest = self.dataset_root / "qces_real10_scoring.jsonl"
        rows = []
        predictions = []
        time = np.arange(NUM_SAMPLES, dtype=np.float64) / SAMPLE_RATE
        for scene_index in range(2):
            scene_id = f"scene_{scene_index}"
            mixture = (
                0.05 * np.sin(2 * np.pi * (220.0 + 20.0 * scene_index) * time)
            ).astype(np.float32)
            mixture_path = self.dataset_root / "audio" / f"{scene_id}.wav"
            write_deterministic_float_wav(mixture_path, mixture)
            mixture_sha256 = _sha256_file(mixture_path)
            for question_index in range(2):
                row = scoring_row(
                    scene_index=scene_index,
                    question_index=question_index,
                    mixture_sha256=mixture_sha256,
                )
                rows.append(row)
                sample_id = row["id"]
                evidence = (
                    np.zeros_like(mixture)
                    if row["no_evidence"]
                    else mixture * np.float32(0.5)
                )
                residual = mixture - evidence
                evidence_path = (
                    self.prediction_root / "stems" / sample_id / "evidence.wav"
                )
                residual_path = evidence_path.with_name("residual.wav")
                evidence_identity = write_deterministic_float_wav(
                    evidence_path, evidence
                )
                residual_identity = write_deterministic_float_wav(
                    residual_path, residual
                )
                evidence_identity["path"] = evidence_path.relative_to(
                    self.prediction_root
                ).as_posix()
                residual_identity["path"] = residual_path.relative_to(
                    self.prediction_root
                ).as_posix()
                predictions.append(
                    {
                        "id": sample_id,
                        "stems": {
                            "evidence": evidence_identity,
                            "residual": residual_identity,
                        },
                    }
                )
        self.records = parse_scoring_manifest(rows)
        self.validated = ValidatedScoringInputs(
            scoring_manifest_path=self.scoring_manifest,
            dataset_root=self.dataset_root,
            prediction_root=self.prediction_root,
            scoring_manifest_identity={"sha256": "a" * 64},
            prediction_manifest_identity={"sha256": "b" * 64},
            render_receipt_identity={"sha256": "c" * 64},
            split="real_dev",
            records=self.records,
            selected_records=self.records,
            prediction_rows=tuple(predictions),
            render_receipt={},
            evaluated_records=(),
            independently_observed_max_reconstruction_error=0.0,
        )


class Real10AudioQATest(unittest.TestCase):
    def test_only_nonoracle_inputs_are_exactly_bound_and_shuffle_is_matched(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10AudioQAFixture(Path(temporary))
            descriptors = build_input_descriptors(
                fixture.validated,
                conditions=ALL_CONDITIONS,
                seed=2026,
            )
            self.assertEqual(len(descriptors), len(fixture.records) * 6)
            self.assertFalse(any("oracle" in condition for _, condition in descriptors))
            records = {record.sample_id: record for record in fixture.records}
            for record in fixture.records:
                shuffled = descriptors[(record.sample_id, "shuffled_evidence")]
                source = records[shuffled.source_record_id]
                self.assertNotEqual(source.scene_id, record.scene_id)
                self.assertEqual(source.question_index, record.question_index)
                self.assertEqual(source.relation, record.relation)
            with self.assertRaisesRegex(Real10AudioQAError, "forbidden condition"):
                build_input_descriptors(
                    fixture.validated,
                    conditions=("oracle_evidence",),
                    seed=2026,
                )
            descriptor = descriptors[("scene_0__q0", "predicted_evidence")]
            Path(descriptor.path).write_bytes(b"tampered")
            with self.assertRaisesRegex(Real10AudioQAError, "before decoding"):
                materialize_bound_input(descriptor)

    def test_report_has_paired_qa_endpoints_arrows_and_no_oracle_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Real10AudioQAFixture(Path(temporary))
            items = []
            for record in fixture.records:
                for condition in ALL_CONDITIONS:
                    correct = condition in {
                        "mixture",
                        "predicted_evidence",
                    }
                    if record.no_evidence:
                        correct = condition in {
                            "mixture",
                            "predicted_evidence",
                            "predicted_residual",
                            "silence",
                            "question_only",
                        }
                    predicted_answer = record.answer if correct else "noise"
                    items.append(
                        {
                            "id": record.sample_id,
                            "scene_id": record.scene_id,
                            "creator_id": record.creator_id,
                            "split": record.split,
                            "question_index": record.question_index,
                            "question_type": record.question_type,
                            "relation": record.relation,
                            "question": record.question,
                            "answer_options": list(record.answer_options),
                            "gold_answer": record.answer,
                            "gold_option_index": record.answer_option_index,
                            "no_evidence": record.no_evidence,
                            "condition": condition,
                            "predicted_answer": predicted_answer,
                            "correct": correct,
                            "gold_log_score": -0.1 if correct else -2.0,
                            "gold_option_probability": 0.8 if correct else 0.1,
                        }
                    )
            report = build_report(
                validated=fixture.validated,
                conditions=ALL_CONDITIONS,
                items=items,
                run_fingerprint="d" * 64,
                metadata={"auditor": "fixture"},
                bootstrap_samples=20,
                seed=2026,
            )
            paired = report["paired_metrics"]
            self.assertEqual(paired["predicted_evidence_sufficiency_accuracy_↑"], 1.0)
            self.assertEqual(
                paired["predicted_residual_answer_leakage_accuracy_↓"], 0.0
            )
            self.assertEqual(
                report["integrity"]["oracle_or_clean_waveform_conditions_consumed_↓"],
                0,
            )
            self.assertTrue(all(name.endswith(("↑", "↓")) for name in report["counts"]))


if __name__ == "__main__":
    unittest.main()
