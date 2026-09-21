from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from mixi_understanding.data.qces_real10_schema import (
    DURATION_SECONDS,
    INFERENCE_FIELDS,
    INFERENCE_SCHEMA_VERSION,
    NUM_CHANNELS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    SCORING_FIELDS,
    SCORING_SCHEMA_VERSION,
    canonical_inference_manifest_fingerprint,
    inference_record_fingerprint,
    parse_inference_manifest,
    parse_inference_record,
    parse_record,
    parse_scoring_manifest,
    parse_scoring_record,
    project_scoring_fields_to_inference,
    project_scoring_manifest,
    resolve_manifest_mixture_path,
    scoring_to_inference,
)


def inference_row(**overrides: object) -> dict:
    row = {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "id": "tacos_123__after_internal_0",
        "scene_id": "tacos_123",
        "scene_family_id": "tacos_123",
        "split": "real_test",
        "question_index": 0,
        "question_type": "temporal_after",
        "relation": "after",
        "question": "What sound begins immediately after the dog bark?",
        "sample_rate": SAMPLE_RATE,
        "num_channels": NUM_CHANNELS,
        "num_samples": NUM_SAMPLES,
        "duration_seconds": DURATION_SECONDS,
        "mixture_path": "audio/tacos_123.wav",
        "mixture_sha256": "a" * 64,
    }
    row.update(overrides)
    return row


def scoring_row(**overrides: object) -> dict:
    row = inference_row(schema_version=SCORING_SCHEMA_VERSION)
    row.update(
        {
            "inference_record_sha256": "0" * 64,
            "answer_options": [
                "glass breaking",
                "dog bark",
                "car horn",
                "child laughing",
                "no_evidence",
            ],
            "answer": "glass breaking",
            "answer_option_index": 0,
            "no_evidence": False,
            "creator_id": "freesound-user-sha256:" + "c" * 64,
            "anchor_event_ids": ["event_000"],
            "answer_event_ids": ["event_001"],
            "evidence_event_ids": ["event_000", "event_001"],
            "anchor_intervals": [[1.0, 1.7]],
            "answer_intervals": [[2.2, 3.0]],
            "evidence_intervals": [[1.0, 1.7], [2.2, 3.0]],
            "upstream_tacos_split": "test",
            "clean_reference_stems_available": False,
            "waveform_sdr_evaluation_allowed": False,
        }
    )
    row.update(overrides)
    projection = project_scoring_fields_to_inference(row)
    row["inference_record_sha256"] = inference_record_fingerprint(projection)
    return row


class Real10ProjectionAndBindingTest(unittest.TestCase):
    def test_scoring_projection_is_exactly_label_free_and_bound(self) -> None:
        payload = scoring_row()
        scoring = parse_scoring_record(payload)
        projected = scoring_to_inference(scoring)

        self.assertEqual(set(projected.to_dict()), set(INFERENCE_FIELDS))
        self.assertEqual(set(scoring.to_dict()), set(SCORING_FIELDS))
        self.assertEqual(
            scoring.inference_record_sha256,
            inference_record_fingerprint(projected),
        )
        forbidden_projection_fields = {
            "answer",
            "answer_options",
            "answer_option_index",
            "no_evidence",
            "creator_id",
            "anchor_event_ids",
            "answer_event_ids",
            "evidence_event_ids",
            "anchor_intervals",
            "answer_intervals",
            "evidence_intervals",
            "upstream_tacos_split",
        }
        self.assertTrue(forbidden_projection_fields.isdisjoint(projected.to_dict()))
        self.assertEqual(parse_record(projected.to_dict()), projected)
        self.assertEqual(parse_record(scoring.to_dict()), scoring)

    def test_modified_model_input_breaks_scoring_binding(self) -> None:
        payload = scoring_row()
        payload["question"] = "A different question"
        with self.assertRaisesRegex(ValueError, "inference-record binding"):
            parse_scoring_record(payload)

    def test_public_fingerprint_and_projection_revalidate_dataclasses(self) -> None:
        inference = parse_inference_record(inference_row())
        with self.assertRaisesRegex(ValueError, "exactly mono"):
            inference_record_fingerprint(replace(inference, num_samples=123))

        scoring = parse_scoring_record(scoring_row())
        with self.assertRaisesRegex(ValueError, "does not identify"):
            scoring_to_inference(replace(scoring, answer_option_index=2))

    def test_projected_manifest_has_same_canonical_fingerprint(self) -> None:
        first = parse_scoring_record(scoring_row())
        second_payload = scoring_row(
            id="tacos_123__before_internal_0",
            question_index=1,
            question_type="temporal_before",
            relation="before",
            question="What sound begins immediately before the car horn?",
            answer="dog bark",
            answer_option_index=1,
        )
        second = parse_scoring_record(second_payload)
        projected = project_scoring_manifest((first, second))
        direct = parse_inference_manifest([record.to_dict() for record in projected])
        self.assertEqual(
            canonical_inference_manifest_fingerprint(projected),
            canonical_inference_manifest_fingerprint(tuple(reversed(direct))),
        )


class Real10CanonicalAudioAndPathTest(unittest.TestCase):
    def test_only_exact_mono_32k_320k_ten_second_audio_is_accepted(self) -> None:
        invalid = {
            "sample_rate": 16_000,
            "num_channels": 2,
            "num_samples": NUM_SAMPLES - 1,
            "duration_seconds": 9.999,
        }
        for field, value in invalid.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "exactly mono"):
                    parse_inference_record(inference_row(**{field: value}))

    def test_mixture_path_is_normalized_manifest_relative_wav(self) -> None:
        unsafe = (
            "/tmp/mixture.wav",
            "../audio/mixture.wav",
            "audio/../mixture.wav",
            "audio//mixture.wav",
            "audio\\mixture.wav",
            "C:/mixture.wav",
            "audio/mixture.mp3",
        )
        for path in unsafe:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "mixture_path"):
                    parse_inference_record(inference_row(mixture_path=path))

    def test_resolver_rejects_symlink_escape(self) -> None:
        record = parse_inference_record(inference_row())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            outside = root / "outside"
            dataset.mkdir()
            outside.mkdir()
            manifest = dataset / "manifest.jsonl"
            manifest.write_text("", encoding="utf-8")
            (dataset / "audio").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                resolve_manifest_mixture_path(manifest, record)

    def test_lowercase_sha256_is_required(self) -> None:
        for digest in ("a" * 63, "A" * 64, "g" * 64):
            with self.subTest(digest=digest[:4]):
                with self.assertRaisesRegex(ValueError, "lowercase SHA256"):
                    parse_inference_record(inference_row(mixture_sha256=digest))


class Real10ScoringSemanticsTest(unittest.TestCase):
    def test_oracle_stem_fields_and_waveform_sdr_claims_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "oracle clean-stem"):
            parse_scoring_record(scoring_row(evidence_stem_path="audio/oracle.wav"))
        with self.assertRaisesRegex(ValueError, "forbids waveform-SDR"):
            parse_scoring_record(scoring_row(waveform_sdr_evaluation_allowed=True))
        with self.assertRaisesRegex(ValueError, "no oracle clean stems"):
            parse_scoring_record(scoring_row(clean_reference_stems_available=True))

    def test_options_are_five_unique_and_answer_index_is_exact(self) -> None:
        duplicate = scoring_row()
        duplicate["answer_options"][4] = "DOG BARK"
        duplicate["answer_options"][1] = "dog bark"
        with self.assertRaisesRegex(ValueError, "case-insensitively unique"):
            parse_scoring_record(duplicate)

        whitespace_duplicate = scoring_row()
        whitespace_duplicate["answer_options"][1] = "dog  bark"
        whitespace_duplicate["answer_options"][4] = "DOG BARK"
        with self.assertRaisesRegex(ValueError, "case-insensitively unique"):
            parse_scoring_record(whitespace_duplicate)

        missing_negative = scoring_row(
            answer_options=["a", "b", "c", "d", "e"],
            answer="a",
            answer_option_index=0,
        )
        with self.assertRaisesRegex(ValueError, "include no_evidence"):
            parse_scoring_record(missing_negative)

        wrong_index = scoring_row(answer_option_index=2)
        with self.assertRaisesRegex(ValueError, "does not identify"):
            parse_scoring_record(wrong_index)

    def test_no_evidence_and_answerable_temporal_contracts(self) -> None:
        negative = parse_scoring_record(
            scoring_row(
                answer="no_evidence",
                answer_option_index=4,
                no_evidence=True,
                anchor_event_ids=[],
                answer_event_ids=[],
                evidence_event_ids=[],
                anchor_intervals=[],
                answer_intervals=[],
                evidence_intervals=[],
            )
        )
        self.assertTrue(negative.no_evidence)

        with self.assertRaisesRegex(ValueError, "all role labels empty"):
            parse_scoring_record(
                scoring_row(
                    answer="no_evidence",
                    answer_option_index=4,
                    no_evidence=True,
                )
            )
        with self.assertRaisesRegex(ValueError, "nonempty anchor and answer"):
            parse_scoring_record(
                scoring_row(
                    anchor_event_ids=[],
                    answer_event_ids=[],
                    evidence_event_ids=[],
                    anchor_intervals=[],
                    answer_intervals=[],
                    evidence_intervals=[],
                )
            )

    def test_evidence_ids_and_intervals_are_bounded_and_aligned(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal cardinality"):
            parse_scoring_record(scoring_row(evidence_intervals=[[1.0, 2.0]]))
        for interval in ([-0.1, 1.0], [1.0, 10.1], [2.0, 2.0]):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, "inside"):
                    parse_scoring_record(
                        scoring_row(
                            evidence_event_ids=["event_000"],
                            evidence_intervals=[interval],
                        )
                    )

    def test_role_specific_evidence_must_pair_and_equal_union(self) -> None:
        with self.assertRaisesRegex(ValueError, "anchor event IDs and intervals"):
            parse_scoring_record(scoring_row(anchor_intervals=[]))
        with self.assertRaisesRegex(ValueError, "inconsistent evidence interval"):
            parse_scoring_record(
                scoring_row(evidence_intervals=[[1.1, 1.7], [2.2, 3.0]])
            )
        with self.assertRaisesRegex(ValueError, "deduplicated anchor-answer union"):
            parse_scoring_record(
                scoring_row(
                    evidence_event_ids=["event_000"],
                    evidence_intervals=[[1.0, 1.7]],
                )
            )

    def test_upstream_tacos_split_is_preserved_independently_of_qces_split(
        self,
    ) -> None:
        # QCES-Real-10 assigns a custom creator-disjoint real_dev/real_test split
        # over the pooled official TACOS partitions, so these axes must not be
        # conflated.
        pooled = parse_scoring_record(
            scoring_row(split="real_test", upstream_tacos_split="development")
        )
        self.assertEqual(pooled.upstream_tacos_split, "development")
        with self.assertRaisesRegex(ValueError, "upstream_tacos_split"):
            parse_scoring_record(scoring_row(upstream_tacos_split="training"))


class Real10ManifestInvariantTest(unittest.TestCase):
    def test_scene_questions_share_one_audio_identity_and_unique_indices(self) -> None:
        first = inference_row()
        second = inference_row(
            id="tacos_123__before_internal_0",
            question_index=1,
            question_type="temporal_before",
            relation="before",
            question="What sound begins before the horn?",
        )
        records = parse_inference_manifest([first, second])
        self.assertEqual(len(records), 2)

        inconsistent = dict(second, mixture_sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "scene audio identity"):
            parse_inference_manifest([first, inconsistent])

        duplicate_index = dict(second, question_index=0)
        with self.assertRaisesRegex(ValueError, "question index"):
            parse_inference_manifest([first, duplicate_index])

    def test_manifest_rejects_duplicate_sample_ids_and_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            parse_inference_manifest([])
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            parse_inference_manifest([inference_row(), inference_row()])

    def test_scoring_manifest_parser_preserves_bindings(self) -> None:
        records = parse_scoring_manifest([scoring_row()])
        self.assertEqual(
            records[0].inference_record_sha256,
            inference_record_fingerprint(records[0].to_inference()),
        )

    def test_scoring_manifest_preserves_one_upstream_split_per_scene(self) -> None:
        first = scoring_row()
        second = scoring_row(
            id="tacos_123__before_internal_0",
            question_index=1,
            question_type="temporal_before",
            relation="before",
            question="What begins before the horn?",
            upstream_tacos_split="development",
        )
        with self.assertRaisesRegex(ValueError, "upstream TACOS split"):
            parse_scoring_manifest([first, second])

        second = scoring_row(
            id="tacos_123__before_internal_0",
            question_index=1,
            question_type="temporal_before",
            relation="before",
            question="What begins before the horn?",
            creator_id="freesound-user-sha256:" + "d" * 64,
        )
        with self.assertRaisesRegex(ValueError, "inconsistent creator ID"):
            parse_scoring_manifest([first, second])

    def test_exact_fields_and_schema_dispatch_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            parse_inference_record(inference_row(unexpected=True))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse_record({"schema_version": "future_schema"})


if __name__ == "__main__":
    unittest.main()
