"""Checkpoint-free regression tests for the frozen AudioQA audit."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from mixi_understanding.data.qces_v4_schema import parse_qces_v4_record
from mixi_understanding.scripts import evaluate_qces_audioqa as audioqa


SAMPLE_RATE = 8_000
NUM_SAMPLES = 8_000


def _event(
    event_id: str,
    label: str,
    event_kind: str,
    onset: float,
    offset: float,
) -> dict:
    return {
        "event_id": event_id,
        "label": label,
        "event_kind": event_kind,
        "source_dataset": "fixture",
        "source_id": f"source_{event_id}",
        "source_path": f"sources/{event_id}.wav",
        "source_sha256": "0" * 64,
        "source_interval_seconds": [0.0, 1.0],
        "source_crop_interval_seconds": [0.0, offset - onset],
        "onset_seconds": onset,
        "offset_seconds": offset,
        "stem_path": f"audio/events/{event_id}.wav",
    }


def _record_payload(
    *,
    sample_id: str = "test_000000_00",
    scene_id: str = "scene_test_000000",
    question_index: int = 0,
) -> dict:
    split = sample_id.split("_", 1)[0]
    return {
        "schema_version": "qces_v4",
        "id": sample_id,
        "scene_id": scene_id,
        "question_family_id": "after_horn",
        "counterfactual_group_id": f"{scene_id}_after_horn",
        "paraphrase_family_id": "after_template_0",
        "question_index": question_index,
        "split": split,
        "sample_rate": SAMPLE_RATE,
        "num_channels": 1,
        "num_samples": NUM_SAMPLES,
        "duration_seconds": 1.0,
        "mixture_path": "audio/mixture.wav",
        "evidence_stem_path": "audio/evidence.wav",
        "residual_stem_path": "audio/residual.wav",
        "anchor_stem_path": "audio/anchor.wav",
        "answer_stem_path": "audio/answer.wav",
        "question": "What sound occurs immediately after the horn?",
        "answer": "bell",
        "answer_options": ["bell", "no_evidence", "croak", "rain", "fan"],
        "answer_option_index": 0,
        "question_type": "temporal_after",
        "relation": "after",
        "no_evidence": False,
        "no_evidence_reason": None,
        "absent_label": None,
        "query_labels": ["horn"],
        "query_event_ids": ["event_horn"],
        "events": [
            _event("event_horn", "horn", "semantic", 0.05, 0.10),
            _event("event_bell", "bell", "semantic", 0.40, 0.45),
            _event("event_rain", "rain", "semantic", 0.80, 0.85),
            _event("event_fan", "fan", "nuisance", 0.20, 0.25),
        ],
        "anchor_event_ids": ["event_horn"],
        "answer_event_ids": ["event_bell"],
        "evidence_event_ids": ["event_horn", "event_bell"],
        "anchor_intervals": [[0.05, 0.10]],
        "answer_intervals": [[0.40, 0.45]],
        "event_presence_labels": ["horn", "bell", "rain", "fan"],
        "source_group_ids": [
            "source_event_horn",
            "source_event_bell",
            "source_event_rain",
            "source_event_fan",
        ],
        "nuisance_snr_db_requested": 0.0,
        "nuisance_snr_db": 0.0,
        "mixture_peak": 0.3,
        "generation_seed": 2026,
    }


def _write_wav(path: Path, signal: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, signal, sample_rate, subtype="FLOAT")


def _audit_item(
    record_id: str,
    scene_id: str,
    condition: str,
    *,
    correct: bool,
    no_evidence: bool = False,
    predicted_answer: str | None = None,
    gold_log_score: float = -0.1,
    relation: str = "after",
) -> dict:
    gold = "no_evidence" if no_evidence else "bell"
    if predicted_answer is None:
        predicted_answer = (
            gold if correct else ("rain" if no_evidence else "no_evidence")
        )
    return {
        "id": record_id,
        "scene_id": scene_id,
        "condition": condition,
        "no_evidence": no_evidence,
        "relation": relation,
        "correct": correct,
        "predicted_answer": predicted_answer,
        "gold_log_score": gold_log_score,
        "gold_option_probability": 0.7 if correct else 0.1,
    }


def _paired_fixture_items() -> list[dict]:
    specifications = {
        "mixture": ((True, -0.10), (False, -2.00)),
        "predicted_evidence": ((True, -0.05), (True, -0.10)),
        "predicted_residual": ((False, -2.00), (True, -0.20)),
        "oracle_evidence": ((True, -0.02), (True, -0.03)),
        "oracle_residual": ((False, -2.20), (False, -2.10)),
        "silence": ((False, -2.50), (False, -2.50)),
        "question_only": ((False, -2.50), (False, -2.50)),
        "shuffled_evidence": ((False, -2.50), (False, -2.50)),
        "shuffled_oracle_evidence": ((False, -2.50), (False, -2.50)),
    }
    items: list[dict] = []
    for condition, values in specifications.items():
        for index, (correct, log_score) in enumerate(values, start=1):
            items.append(
                _audit_item(
                    f"r{index}",
                    f"scene{index}",
                    condition,
                    correct=correct,
                    gold_log_score=log_score,
                )
            )
        items.append(
            _audit_item(
                "negative",
                "scene1",
                condition,
                correct=True,
                no_evidence=True,
                predicted_answer="no_evidence",
            )
        )
    return items


class ParsingAndDescriptorTest(unittest.TestCase):
    def test_load_records_strictly_parses_filters_and_rejects_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test_payload = _record_payload()
            val_payload = _record_payload(
                sample_id="val_000000_00", scene_id="scene_val_000000"
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(test_payload) + "\n" + json.dumps(val_payload) + "\n",
                encoding="utf-8",
            )
            records = audioqa.load_records(manifest, "test", None)
            self.assertEqual(
                [record.sample_id for record in records], [test_payload["id"]]
            )
            self.assertEqual(len(audioqa.load_records(manifest, "all", 1)), 1)

            duplicate = root / "duplicate.jsonl"
            duplicate.write_text(
                json.dumps(test_payload) + "\n" + json.dumps(test_payload) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate record IDs"):
                audioqa.load_records(duplicate, "all", None)

            malformed_payload = copy.deepcopy(test_payload)
            malformed_payload["schema_version"] = "qces_v3"
            malformed = root / "malformed.jsonl"
            malformed.write_text(json.dumps(malformed_payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                audioqa.load_records(malformed, "all", None)

    def test_shuffle_is_deterministic_cross_scene_and_requires_two_scenes(self) -> None:
        first = parse_qces_v4_record(_record_payload())
        second = parse_qces_v4_record(
            _record_payload(sample_id="test_000001_00", scene_id="scene_test_000001")
        )
        mapping = audioqa.shuffled_record_map([second, first])
        self.assertEqual(mapping[first.sample_id].sample_id, second.sample_id)
        self.assertEqual(mapping[second.sample_id].sample_id, first.sample_id)
        self.assertEqual(mapping, audioqa.shuffled_record_map([first, second]))
        with self.assertRaisesRegex(ValueError, "at least two scenes"):
            audioqa.shuffled_record_map([first])

    def test_descriptors_resolve_every_condition_and_block_root_escape(self) -> None:
        first = parse_qces_v4_record(_record_payload())
        second = parse_qces_v4_record(
            _record_payload(sample_id="test_000001_00", scene_id="scene_test_000001")
        )
        signal = np.linspace(-0.1, 0.1, NUM_SAMPLES, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            predictions = root / "predictions"
            for filename in ("mixture.wav", "evidence.wav", "residual.wav"):
                _write_wav(dataset / "audio" / filename, signal)
            for record in (first, second):
                question_dir = (
                    predictions
                    / record.scene_id
                    / f"q{record.question_index}_{record.question_type}"
                )
                _write_wav(question_dir / "predicted_evidence.wav", signal)
                _write_wav(question_dir / "predicted_residual.wav", -signal)

            descriptors = audioqa.build_input_descriptors(
                [first, second], dataset, predictions, audioqa.ALL_CONDITIONS
            )
            self.assertEqual(len(descriptors), 2 * len(audioqa.ALL_CONDITIONS))
            shuffled = descriptors[(first.sample_id, "shuffled_evidence")]
            self.assertEqual(shuffled.source_record_id, second.sample_id)
            shuffled_oracle = descriptors[(first.sample_id, "shuffled_oracle_evidence")]
            self.assertEqual(shuffled_oracle.source_record_id, second.sample_id)
            self.assertEqual(
                descriptors[(first.sample_id, "silence")].special, "silence"
            )
            self.assertEqual(
                descriptors[(first.sample_id, "question_only")].special,
                "question_only",
            )

            with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                audioqa._safe_manifest_audio(dataset, "../outside.wav")
            malicious = replace(first, scene_id="scene_test/../../../outside")
            with self.assertRaisesRegex(ValueError, "escapes predictions root"):
                audioqa.build_input_descriptors(
                    [malicious], dataset, predictions, ("predicted_evidence",)
                )

    def test_materialization_checks_shape_rate_channels_and_special_inputs(
        self,
    ) -> None:
        signal = np.linspace(-0.1, 0.1, NUM_SAMPLES, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mono_path = root / "mono.wav"
            _write_wav(mono_path, signal)
            descriptor = audioqa.InputDescriptor(
                "r",
                "mixture",
                "r",
                str(mono_path),
                "0" * 64,
                SAMPLE_RATE,
                NUM_SAMPLES,
            )
            audioqa._read_audio_cached.cache_clear()
            waveform, sample_rate = audioqa.materialize_input(descriptor)
            self.assertEqual(sample_rate, SAMPLE_RATE)
            np.testing.assert_array_equal(waveform, signal)
            assert waveform is not None
            waveform[0] = 123.0
            second_read, _ = audioqa.materialize_input(descriptor)
            self.assertNotEqual(float(second_read[0]), 123.0)

            silence = replace(descriptor, path=None, special="silence")
            silence_waveform, silence_rate = audioqa.materialize_input(silence)
            np.testing.assert_array_equal(silence_waveform, np.zeros(NUM_SAMPLES))
            self.assertEqual(silence_rate, SAMPLE_RATE)
            question_only = replace(
                descriptor, path=None, num_samples=0, special="question_only"
            )
            self.assertEqual(audioqa.materialize_input(question_only), (None, None))

            bad_rate = replace(descriptor, sample_rate=SAMPLE_RATE + 1)
            with self.assertRaisesRegex(ValueError, "sample-rate mismatch"):
                audioqa.materialize_input(bad_rate)
            bad_length = replace(descriptor, num_samples=NUM_SAMPLES - 1)
            with self.assertRaisesRegex(ValueError, "sample-count mismatch"):
                audioqa.materialize_input(bad_length)

            stereo_path = root / "stereo.wav"
            _write_wav(stereo_path, np.column_stack((signal, signal)))
            stereo = replace(descriptor, path=str(stereo_path))
            audioqa._read_audio_cached.cache_clear()
            with self.assertRaisesRegex(ValueError, "expected mono"):
                audioqa.materialize_input(stereo)


class ScoringAndMetricTest(unittest.TestCase):
    def test_prompt_supports_interactive_two_way_without_changing_five_way(
        self,
    ) -> None:
        two_way = audioqa.format_mc_prompt("Which sound?", ("bell", "rain"))
        self.assertIn("A. bell", two_way)
        self.assertIn("B. rain", two_way)
        self.assertNotIn("C.", two_way)
        self.assertIn("A, B.", two_way)

        five_way = audioqa.format_mc_prompt(
            "Which sound?", ("one", "two", "three", "four", "five")
        )
        self.assertIn("E. five", five_way)
        self.assertIn("A, B, C, D, E.", five_way)
        with self.assertRaisesRegex(ValueError, "between two and five"):
            audioqa.format_mc_prompt("Which sound?", ("one",))

    def test_exact_letter_continuations_have_no_leading_space(self) -> None:
        class Tokenizer:
            def __init__(self) -> None:
                self.inputs: list[tuple[str, bool]] = []

            def encode(self, text: str, add_special_tokens: bool) -> list[int]:
                self.inputs.append((text, add_special_tokens))
                return [ord(text)]

        tokenizer = Tokenizer()
        token_ids = audioqa.encode_option_continuations(tokenizer)
        self.assertEqual(
            [text for text, _ in tokenizer.inputs], list(audioqa.OPTION_LABELS)
        )
        self.assertFalse(any(text.startswith(" ") for text, _ in tokenizer.inputs))
        self.assertEqual(token_ids[0], (ord("A"),))

    def test_prepare_routes_af3_audio_kwargs_and_omits_them_question_only(self) -> None:
        import torch

        class Processor:
            def __init__(self) -> None:
                self.calls: list[tuple[object, dict]] = []

            def apply_chat_template(
                self, conversation: object, **kwargs: object
            ) -> dict:
                self.calls.append((conversation, kwargs))
                contains_audio = any(
                    block.get("type") == "audio"
                    for message in conversation[0]
                    for block in message["content"]
                )
                result = {
                    "input_ids": torch.ones((1, 3), dtype=torch.long),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                }
                if contains_audio:
                    result.update(
                        {
                            "input_features": torch.ones((1, 128, 3000)),
                            "input_features_mask": torch.ones((1, 3000)),
                        }
                    )
                return result

        scorer = object.__new__(audioqa.AudioFlamingo3OptionScorer)
        scorer.processor = Processor()
        scorer.target_sample_rate = 16_000
        scorer.input_device = torch.device("cpu")
        scorer.audio_dtype = torch.float16

        prepared = scorer._prepare("prompt", np.zeros(16_000, dtype=np.float32), 16_000)
        self.assertIn("input_features", prepared)
        self.assertEqual(prepared["input_features"].dtype, torch.float32)
        _, audio_kwargs = scorer.processor.calls[-1]
        self.assertNotIn("audio_kwargs", audio_kwargs)
        self.assertNotIn("padding", audio_kwargs)
        self.assertEqual(
            audio_kwargs["processor_kwargs"],
            {
                "audio_kwargs": {
                    "sampling_rate": 16_000,
                    "return_attention_mask": True,
                    "padding": "max_length",
                }
            },
        )

        question_only = scorer._prepare("prompt", None, None)
        self.assertNotIn("input_features", question_only)
        _, question_only_kwargs = scorer.processor.calls[-1]
        self.assertNotIn("processor_kwargs", question_only_kwargs)
        self.assertNotIn("audio_kwargs", question_only_kwargs)

    def test_prepare_qwen2_audio_uses_official_chat_and_processor_contract(
        self,
    ) -> None:
        import torch

        class Processor:
            def __init__(self) -> None:
                self.template_calls: list[tuple[object, dict]] = []
                self.processor_calls: list[dict] = []

            def apply_chat_template(
                self, conversation: object, **kwargs: object
            ) -> str:
                self.template_calls.append((conversation, kwargs))
                messages = conversation  # one non-batched Qwen conversation
                has_audio = any(
                    block.get("type") == "audio"
                    for message in messages
                    for block in message["content"]
                )
                token = "<|audio_bos|><|AUDIO|><|audio_eos|>\n" if has_audio else ""
                return token + "<|im_start|>assistant\n"

            def __call__(self, **kwargs: object) -> dict:
                self.processor_calls.append(kwargs)
                result = {
                    "input_ids": torch.ones((1, 3), dtype=torch.long),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                    "ignored": torch.ones((1, 1)),
                }
                if "audio" in kwargs:
                    result.update(
                        {
                            "input_features": torch.ones((1, 128, 3000)),
                            "feature_attention_mask": torch.ones((1, 3000)),
                        }
                    )
                return result

        scorer = object.__new__(audioqa.Qwen2AudioOptionScorer)
        scorer.processor = Processor()
        scorer.target_sample_rate = 16_000
        scorer.input_device = torch.device("cpu")

        prepared = scorer._prepare("prompt", np.zeros(8_000, dtype=np.float32), 8_000)
        self.assertIn("input_features", prepared)
        self.assertIn("feature_attention_mask", prepared)
        self.assertNotIn("ignored", prepared)
        conversation, template_kwargs = scorer.processor.template_calls[-1]
        self.assertEqual(conversation[0]["content"][0]["type"], "audio")
        self.assertEqual(conversation[0]["content"][-1]["text"], "prompt")
        self.assertEqual(
            template_kwargs, {"add_generation_prompt": True, "tokenize": False}
        )
        processor_kwargs = scorer.processor.processor_calls[-1]
        self.assertEqual(processor_kwargs["sampling_rate"], 16_000)
        self.assertEqual(processor_kwargs["audio"].shape[0], 16_000)

        question_only = scorer._prepare("prompt", None, None)
        self.assertNotIn("input_features", question_only)
        question_only_kwargs = scorer.processor.processor_calls[-1]
        self.assertNotIn("audio", question_only_kwargs)
        self.assertNotIn("sampling_rate", question_only_kwargs)

    def test_prepare_phi4mm_uses_pinned_speech_prompt_contract(self) -> None:
        import torch

        class Processor:
            def __init__(self) -> None:
                self.calls: list[dict] = []

            def __call__(self, **kwargs: object) -> dict:
                self.calls.append(kwargs)
                result = {
                    "input_ids": torch.ones((1, 4), dtype=torch.long),
                    "attention_mask": torch.ones((1, 4), dtype=torch.bool),
                    "input_mode": torch.tensor([2 if "audios" in kwargs else 0]),
                    "input_image_embeds": torch.tensor([]),
                }
                if "audios" in kwargs:
                    result.update(
                        {
                            "input_audio_embeds": torch.ones((1, 100, 80)),
                            "audio_embed_sizes": torch.tensor([13]),
                        }
                    )
                return result

        scorer = object.__new__(audioqa.Phi4MMOptionScorer)
        scorer.processor = Processor()
        scorer.target_sample_rate = 16_000
        scorer.input_device = torch.device("cpu")

        prepared = scorer._prepare("prompt", np.zeros(8_000, dtype=np.float32), 8_000)
        self.assertIn("input_audio_embeds", prepared)
        call = scorer.processor.calls[-1]
        self.assertEqual(call["text"], "<|user|><|audio_1|>prompt<|end|><|assistant|>")
        audio, sample_rate = call["audios"][0]
        self.assertEqual(sample_rate, 16_000)
        self.assertEqual(audio.shape[0], 16_000)

        question_only = scorer._prepare("prompt", None, None)
        self.assertNotIn("input_audio_embeds", question_only)
        call = scorer.processor.calls[-1]
        self.assertEqual(call["text"], "<|user|>prompt<|end|><|assistant|>")
        self.assertNotIn("audios", call)

    def test_mixed_precision_forward_context_uses_cuda_autocast(self) -> None:
        calls: list[tuple[str, object]] = []

        class Context:
            def __enter__(self) -> None:
                calls.append(("enter", None))

            def __exit__(self, *_: object) -> None:
                calls.append(("exit", None))

        half = object()
        fake_torch = SimpleNamespace(
            float16=half,
            bfloat16=object(),
            autocast=lambda *, device_type, dtype: (
                calls.append((device_type, dtype)) or Context()
            ),
        )
        scorer = object.__new__(audioqa.AudioFlamingo3OptionScorer)
        scorer.torch = fake_torch
        scorer.input_device = SimpleNamespace(type="cuda")
        scorer.compute_dtype = half
        with scorer._model_forward_context():
            pass
        self.assertEqual(calls[0], ("cuda", half))
        self.assertEqual(calls[1:], [("enter", None), ("exit", None)])

        scorer.input_device = SimpleNamespace(type="cpu")
        calls.clear()
        with scorer._model_forward_context():
            pass
        self.assertEqual(calls, [])

    def test_projector_output_is_reconciled_to_language_embedding_dtype(self) -> None:
        import torch

        projected_float = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        language_half = torch.zeros((2, 3), dtype=torch.float16)
        mask = torch.ones_like(language_half, dtype=torch.bool)
        with self.assertRaisesRegex(RuntimeError, "same dtype"):
            language_half.masked_scatter(mask, projected_float)

        reconciled = audioqa.reconcile_projector_output_dtype(
            projected_float, language_half.dtype
        )
        self.assertEqual(reconciled.dtype, torch.float16)
        scattered = language_half.masked_scatter(mask, reconciled)
        self.assertEqual(scattered.dtype, torch.float16)
        torch.testing.assert_close(scattered.float(), projected_float)

        # The already-working Float32 route remains an identity/no-copy path.
        preserved = audioqa.reconcile_projector_output_dtype(
            projected_float, torch.float32
        )
        self.assertIs(preserved, projected_float)
        with self.assertRaisesRegex(TypeError, "must return a tensor"):
            audioqa.reconcile_projector_output_dtype({"not": "a tensor"}, torch.float16)

    def test_make_item_uses_mock_score_and_rejects_inconsistent_values(self) -> None:
        record = parse_qces_v4_record(_record_payload())
        descriptor = audioqa.InputDescriptor(
            record.sample_id,
            "mixture",
            record.sample_id,
            None,
            "0" * 64,
            SAMPLE_RATE,
            NUM_SAMPLES,
            "silence",
        )
        probabilities = audioqa._softmax((5.0, 4.0, 3.0, 2.0, 1.0))
        score = audioqa.OptionScore(
            (5.0, 4.0, 3.0, 2.0, 1.0), probabilities, 0, "mock", (1,) * 5
        )
        item = audioqa.make_item(record, descriptor, score, "fingerprint")
        self.assertTrue(item["correct"])
        self.assertEqual(item["predicted_answer"], "bell")
        with self.assertRaisesRegex(ValueError, "disagrees"):
            audioqa.make_item(
                record, descriptor, replace(score, predicted_index=1), "fp"
            )
        with self.assertRaisesRegex(ValueError, "invalid option probabilities"):
            audioqa.make_item(
                record,
                descriptor,
                replace(score, probabilities=(0.2,) * 4 + (0.3,)),
                "fp",
            )

    def test_condition_and_paired_metrics_match_paired_definitions(self) -> None:
        items = _paired_fixture_items()
        mixture = [item for item in items if item["condition"] == "mixture"]
        condition = audioqa.condition_metrics(mixture)
        self.assertAlmostEqual(condition["multiple_choice_accuracy_all_↑"], 2 / 3)
        self.assertAlmostEqual(condition["answerable_accuracy_↑"], 0.5)
        self.assertEqual(condition["no_evidence_accuracy_↑"], 1.0)
        self.assertAlmostEqual(condition["answerability_balanced_accuracy_↑"], 0.75)
        self.assertAlmostEqual(
            condition["no_evidence_false_positive_rate_on_answerable_↓"], 0.5
        )
        self.assertAlmostEqual(
            condition["answerable_candidate_aware_chance_accuracy_↑"], 0.2
        )
        self.assertAlmostEqual(
            condition["answerable_accuracy_by_relation_↑"]["after"], 0.5
        )
        self.assertEqual(condition["no_evidence_accuracy_by_relation_↑"]["after"], 1.0)

        paired = audioqa.paired_metrics(items)
        self.assertEqual(paired["predicted_evidence_sufficiency_accuracy_↑"], 1.0)
        self.assertEqual(paired["conditional_sufficiency_given_mixture_correct_↑"], 1.0)
        self.assertAlmostEqual(
            paired["predicted_evidence_accuracy_gain_over_mixture_↑"], 0.5
        )
        self.assertAlmostEqual(
            paired["predicted_evidence_gold_log_score_gain_over_mixture_↑"], 0.975
        )
        self.assertEqual(paired["predicted_residual_answer_leakage_accuracy_↓"], 0.5)
        self.assertEqual(
            paired["conditional_residual_leakage_given_mixture_correct_↓"], 0.0
        )
        self.assertEqual(
            paired["predicted_necessity_success_given_mixture_correct_↑"], 1.0
        )
        self.assertEqual(paired["predicted_necessity_accuracy_drop_↑"], 0.0)
        self.assertAlmostEqual(
            paired["predicted_necessity_gold_log_score_drop_↑"], 0.05
        )
        self.assertEqual(paired["oracle_residual_answer_leakage_accuracy_↓"], 0.0)
        self.assertEqual(paired["question_only_control_answerable_accuracy_↓"], 0.0)
        self.assertAlmostEqual(
            paired["mixture_accuracy_gain_over_question_only_↑"], 0.5
        )
        self.assertAlmostEqual(
            paired["mixture_gold_log_score_gain_over_question_only_↑"], 1.45
        )
        self.assertEqual(
            paired["oracle_evidence_accuracy_gain_over_question_only_↑"], 1.0
        )
        self.assertAlmostEqual(
            paired["oracle_evidence_gold_log_score_gain_over_question_only_↑"],
            2.475,
        )
        self.assertEqual(
            paired["oracle_evidence_accuracy_gain_over_shuffled_oracle_evidence_↑"],
            1.0,
        )
        self.assertAlmostEqual(
            paired[
                "oracle_evidence_gold_log_score_gain_over_shuffled_oracle_evidence_↑"
            ],
            2.475,
        )
        self.assertEqual(
            paired["oracle_evidence_accuracy_gain_over_oracle_residual_↑"], 1.0
        )
        self.assertAlmostEqual(
            paired["oracle_evidence_gold_log_score_gain_over_oracle_residual_↑"],
            2.125,
        )
        self.assertEqual(paired["predicted_evidence_no_evidence_accuracy_↑"], 1.0)
        self.assertEqual(
            paired["predicted_evidence_accuracy_gain_over_shuffled_evidence_↑"],
            1.0,
        )
        self.assertAlmostEqual(
            paired["predicted_evidence_gold_log_score_gain_over_shuffled_evidence_↑"],
            2.425,
        )
        self.assertEqual(
            paired[
                "conditional_sufficiency_given_mixture_correct_question_only_wrong_↑"
            ],
            1.0,
        )
        self.assertEqual(
            paired[
                "conditional_residual_leakage_given_mixture_correct_question_only_wrong_↓"
            ],
            0.0,
        )
        self.assertEqual(
            paired[
                "conditional_necessity_success_given_mixture_correct_question_only_wrong_↑"
            ],
            1.0,
        )
        self.assertEqual(
            paired[
                "oracle_conditional_sufficiency_given_mixture_correct_question_only_wrong_↑"
            ],
            1.0,
        )
        self.assertEqual(
            paired[
                "oracle_conditional_residual_leakage_given_mixture_correct_question_only_wrong_↓"
            ],
            0.0,
        )
        self.assertEqual(
            audioqa.paired_subset_counts(items),
            {
                "answerable_records": 2,
                "mixture_correct_answerable_records": 1,
                "mixture_question_only_paired_answerable_records": 2,
                "audio_dependent_mixture_correct_question_only_wrong_records": 1,
            },
        )

    def test_first_question_shortcut_diagnostics_detect_first_mention_bias(
        self,
    ) -> None:
        first_gold = _audit_item(
            "first_gold",
            "scene1",
            "question_only",
            correct=True,
            predicted_answer="bell",
            relation="first",
        )
        first_gold.update(
            question="Between bell and rain, which comes first?",
            answer_options=["fan", "bell", "no_evidence", "rain", "croak"],
            gold_answer="bell",
        )
        second_gold = _audit_item(
            "second_gold",
            "scene2",
            "question_only",
            correct=False,
            predicted_answer="bell",
            relation="first",
        )
        second_gold.update(
            question="Which sound occurs first, bell or rain?",
            answer_options=["rain", "no_evidence", "bell", "fan", "croak"],
            gold_answer="rain",
        )
        metrics = audioqa.condition_metrics([first_gold, second_gold])
        self.assertEqual(metrics["answerable_accuracy_by_relation_↑"]["first"], 0.5)
        self.assertEqual(
            metrics["first_question_named_candidate_prediction_rate_↑"], 1.0
        )
        self.assertEqual(metrics["first_question_first_mention_bias_gap_↓"], 0.5)
        self.assertEqual(metrics["first_question_mention_position_accuracy_gap_↓"], 1.0)

    def test_scene_bootstrap_is_deterministic_and_metric_arrows_are_complete(
        self,
    ) -> None:
        items = _paired_fixture_items()
        first = audioqa.bootstrap_paired_metrics(items, samples=100, seed=77)
        second = audioqa.bootstrap_paired_metrics(items, samples=100, seed=77)
        self.assertEqual(first, second)
        self.assertIn("predicted_evidence_accuracy_gain_over_mixture_↑", first)
        self.assertIn(
            "predicted_evidence_accuracy_gain_over_shuffled_evidence_↑", first
        )
        for lower, upper in first.values():
            self.assertLessEqual(lower, upper)

        intervals, coverage = audioqa.bootstrap_paired_metric_summary(
            items, samples=100, seed=77
        )
        self.assertEqual(intervals, first)
        self.assertEqual(coverage["requested_draws"], 100)
        self.assertEqual(coverage["independent_scene_clusters"], 2)
        conditional_name = (
            "conditional_sufficiency_given_mixture_correct_question_only_wrong_↑"
        )
        self.assertGreater(coverage["valid_draws_by_metric"][conditional_name], 0)
        self.assertLessEqual(coverage["valid_draws_by_metric"][conditional_name], 100)
        self.assertAlmostEqual(
            coverage["valid_fraction_by_metric"][conditional_name],
            coverage["valid_draws_by_metric"][conditional_name] / 100,
        )

        conditions = list(audioqa.ALL_CONDITIONS)
        grouped = {
            condition: audioqa.condition_metrics(
                [item for item in items if item["condition"] == condition]
            )
            for condition in conditions
        }
        paired = audioqa.paired_metrics(items)
        audioqa.validate_metric_metadata(grouped, paired)
        for name in {key for metrics in grouped.values() for key in metrics} | set(
            paired
        ):
            self.assertTrue(name.endswith(("_↑", "_↓")), name)


class ResumeAndFingerprintTest(unittest.TestCase):
    def test_main_writes_isolated_option_order_report_end_to_end(self) -> None:
        class FakeScorer:
            calls: list[tuple[str, ...]] = []

            def __init__(self, **_: object) -> None:
                pass

            def provenance(self) -> dict:
                return {"model_class": "FakeScorer", "scoring_version": "mock"}

            def score(
                self,
                _question: str,
                options: tuple[str, ...],
                _waveform: object,
                _sample_rate: object,
            ) -> audioqa.OptionScore:
                type(self).calls.append(tuple(options))
                scores = (5.0, 4.0, 3.0, 2.0, 1.0)
                return audioqa.OptionScore(
                    scores, audioqa._softmax(scores), 0, "mock", (1,) * 5
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(_record_payload()) + "\n", encoding="utf-8")
            _write_wav(
                root / "audio" / "mixture.wav",
                np.linspace(-0.1, 0.1, NUM_SAMPLES, dtype=np.float32),
            )
            model = root / "model"
            model.mkdir()
            output = root / "output"
            argv = [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--model",
                str(model),
                "--conditions",
                "mixture",
                "--option-order-control-conditions",
                "mixture",
                "--quantization",
                "none",
                "--dtype",
                "float32",
                "--device",
                "cpu",
                "--bootstrap-samples",
                "5",
            ]
            with patch.object(audioqa, "AudioFlamingo3OptionScorer", FakeScorer):
                with contextlib.redirect_stdout(io.StringIO()):
                    audioqa.main(argv)
            report = json.loads(
                (output / "evaluation_report.json").read_text(encoding="utf-8")
            )
        self.assertEqual(len(FakeScorer.calls), 2)
        self.assertNotEqual(FakeScorer.calls[0], FakeScorer.calls[1])
        self.assertEqual(report["format"], "qces_audioqa_audit_v7")
        self.assertEqual(report["counts"]["completed_record_conditions"], 2)
        self.assertEqual(
            report["option_order_control_coverage_by_condition"]["mixture"][
                "complete_pairs"
            ],
            1,
        )
        metrics = report["option_order_control_metrics_by_condition"]["mixture"]
        self.assertEqual(metrics["option_order_gold_position_changed_rate_↑"], 1.0)
        self.assertEqual(metrics["option_order_semantic_prediction_invariance_↑"], 0.0)
        self.assertIn("mixture", report["option_order_scene_bootstrap_95ci"])

    def test_truncated_and_missing_newline_tails_are_repaired_before_append(
        self,
    ) -> None:
        fingerprint = "f" * 64
        first = {"id": "r1", "condition": "mixture", "run_fingerprint": fingerprint}
        second = {"id": "r2", "condition": "mixture", "run_fingerprint": fingerprint}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "items.jsonl"
            complete_line = json.dumps(first).encode() + b"\n"
            path.write_bytes(complete_line + b'{"id":"partial"')
            loaded = audioqa.load_completed_items(path, fingerprint)
            self.assertEqual(set(loaded), {("r1", "mixture")})
            self.assertEqual(path.read_bytes(), complete_line)
            audioqa.append_item(path, second)
            self.assertEqual(
                set(audioqa.load_completed_items(path, fingerprint)),
                {("r1", "mixture"), ("r2", "mixture")},
            )

            no_newline = Path(directory) / "complete_without_newline.jsonl"
            no_newline.write_text(json.dumps(first), encoding="utf-8")
            audioqa.load_completed_items(no_newline, fingerprint)
            self.assertTrue(no_newline.read_bytes().endswith(b"\n"))

            invalid_complete = Path(directory) / "invalid_complete.jsonl"
            invalid_complete.write_bytes(b"not-json\n")
            with self.assertRaisesRegex(ValueError, "invalid resumable JSONL"):
                audioqa.load_completed_items(invalid_complete, fingerprint)

    def test_main_resumes_without_model_and_fingerprint_tracks_local_model(
        self,
    ) -> None:
        class FakeScorer:
            initializations = 0
            calls = 0

            def __init__(self, **_: object) -> None:
                type(self).initializations += 1

            def provenance(self) -> dict:
                return {"model_class": "FakeScorer", "scoring_version": "mock"}

            def score(self, *_: object) -> audioqa.OptionScore:
                type(self).calls += 1
                scores = (5.0, 4.0, 3.0, 2.0, 1.0)
                return audioqa.OptionScore(
                    scores, audioqa._softmax(scores), 0, "mock", (1,) * 5
                )

        signal = np.linspace(-0.1, 0.1, NUM_SAMPLES, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps(_record_payload()) + "\n", encoding="utf-8")
            _write_wav(root / "audio" / "mixture.wav", signal)
            model = root / "local_model"
            model.mkdir()
            output = root / "output"
            argv = [
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--model",
                str(model),
                "--conditions",
                "mixture",
                "--quantization",
                "none",
                "--dtype",
                "float32",
                "--device",
                "cpu",
                "--bootstrap-samples",
                "5",
            ]
            with patch.object(audioqa, "AudioFlamingo3OptionScorer", FakeScorer):
                with contextlib.redirect_stdout(io.StringIO()):
                    audioqa.main(argv)
                    audioqa.main(argv)
            self.assertEqual(FakeScorer.initializations, 1)
            self.assertEqual(FakeScorer.calls, 1)
            report = json.loads((output / "evaluation_report.json").read_text())
            self.assertEqual(report["counts"]["completed_record_conditions"], 1)
            self.assertIn(
                "multiple_choice_accuracy_all_↑", report["condition_metrics"]["mixture"]
            )
            self.assertIn("paired_subset_counts", report)
            self.assertIn("paired_scene_bootstrap_coverage", report)

            qwen_output = root / "qwen_output"
            qwen_argv = [
                *(item for item in argv),
                "--auditor",
                "qwen2_audio",
            ]
            qwen_argv[qwen_argv.index(str(output))] = str(qwen_output)
            with patch.object(audioqa, "Qwen2AudioOptionScorer", FakeScorer):
                with contextlib.redirect_stdout(io.StringIO()):
                    audioqa.main(qwen_argv)
            qwen_metadata = json.loads(
                (qwen_output / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(qwen_metadata["run_config"]["auditor"], "qwen2_audio")
            self.assertEqual(FakeScorer.initializations, 2)
            self.assertEqual(FakeScorer.calls, 2)

            phi_output = root / "phi_output"
            phi_argv = [*argv, "--auditor", "phi4mm"]
            phi_argv[phi_argv.index(str(output))] = str(phi_output)
            with patch.object(audioqa, "Phi4MMOptionScorer", FakeScorer):
                with contextlib.redirect_stdout(io.StringIO()):
                    audioqa.main(phi_argv)
            phi_metadata = json.loads(
                (phi_output / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(phi_metadata["run_config"]["auditor"], "phi4mm")
            self.assertEqual(FakeScorer.initializations, 3)
            self.assertEqual(FakeScorer.calls, 3)

            # Model metadata is part of the identity even when the local path is
            # unchanged, preventing cross-checkpoint item mixing on resume.
            (model / "tokenizer.json").write_text(
                '{"changed":true}\n', encoding="utf-8"
            )
            with patch.object(audioqa, "AudioFlamingo3OptionScorer", FakeScorer):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(SystemExit, "another run fingerprint"):
                        audioqa.main(argv)


if __name__ == "__main__":
    unittest.main()
