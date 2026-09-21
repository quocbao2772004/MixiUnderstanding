"""Checkpoint-free tests for the frozen AudioSep-CLAP feature cache."""

from __future__ import annotations

import io
import hashlib
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import mixi_understanding.scripts.cache_audiosep_clap_features as cache_module
from mixi_understanding.data.qces_real10_schema import (
    INFERENCE_SCHEMA_VERSION as REAL10_INFERENCE_SCHEMA_VERSION,
    canonical_inference_manifest_fingerprint,
    parse_inference_manifest,
)
from mixi_understanding.scripts.cache_audiosep_clap_features import (
    CANONICAL_NUM_SAMPLES,
    CANONICAL_SAMPLE_RATE,
    EFFECTIVE_FRAMES,
    EXPECTED_QUERY_STATE_TENSORS,
    FINE_DIM,
    JOINT_DIM,
    QUESTION_CACHE_FORMAT,
    RAW_FINE_FRAMES,
    RECEIPT_FORMAT,
    REPEAT_RATIO,
    SCENE_CACHE_FORMAT,
    SampleSpec,
    compress_repeated_fine_features,
    configure_determinism,
    encode_full_questions,
    extract_query_encoder_state,
    load_query_encoder_state_exact,
    parse_args,
    project_effective_audio_features,
    question_payload,
    read_cache_manifest,
    read_canonical_waveform,
    read_strict_manifest,
    real10_receipt_contract,
    scene_payload,
    validate_question_payload,
    validate_scene_payload,
    write_outputs_atomically,
)


class _FakeTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[tuple[str, ...]] = []

    def get_query_embed(self, *, modality: str, text: list[str]) -> torch.Tensor:
        if modality != "text":
            raise AssertionError(modality)
        self.batches.append(tuple(text))
        result = torch.zeros((len(text), JOINT_DIM), dtype=torch.float32)
        for index, value in enumerate(text):
            result[index, sum(value.encode("utf-8")) % JOINT_DIM] = 1.0
        return result


class _SliceProjection(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value[..., :JOINT_DIM]


class FineFeatureTest(unittest.TestCase):
    def test_exact_repeat_compression_and_frame_normalized_projection(self) -> None:
        generator = torch.Generator().manual_seed(7)
        effective = torch.randn((1, EFFECTIVE_FRAMES, FINE_DIM), generator=generator)
        fine = effective.repeat_interleave(REPEAT_RATIO, dim=1)

        compressed = compress_repeated_fine_features(fine)
        projected = project_effective_audio_features(fine, _SliceProjection())

        self.assertTrue(torch.equal(compressed, effective))
        self.assertEqual(tuple(projected.shape), (1, EFFECTIVE_FRAMES, JOINT_DIM))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(projected, dim=-1),
                torch.ones((1, EFFECTIVE_FRAMES)),
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_compression_fails_closed_when_one_repeated_frame_differs(self) -> None:
        effective = torch.ones((1, EFFECTIVE_FRAMES, FINE_DIM))
        fine = effective.repeat_interleave(REPEAT_RATIO, dim=1)
        fine[0, REPEAT_RATIO + 1, 0] += 1e-7

        with self.assertRaisesRegex(RuntimeError, "not exact 32-repeat blocks"):
            compress_repeated_fine_features(fine)

    def test_shape_is_strict(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unexpected raw fine-grained"):
            compress_repeated_fine_features(
                torch.zeros((1, RAW_FINE_FRAMES - 1, FINE_DIM))
            )


class TextFeatureTest(unittest.TestCase):
    def test_full_questions_are_deduplicated_only_for_compute(self) -> None:
        samples = [
            SampleSpec("sample_2", "scene_1", "What follows the bell?"),
            SampleSpec("sample_1", "scene_1", "What follows the bell?"),
            SampleSpec("sample_3", "scene_2", "What starts first?"),
        ]
        encoder = _FakeTextEncoder()

        features = encode_full_questions(encoder, samples, batch_size=1)

        self.assertEqual(set(features), {"sample_1", "sample_2", "sample_3"})
        self.assertEqual(len(encoder.batches), 2)
        self.assertTrue(torch.equal(features["sample_1"], features["sample_2"]))
        self.assertIsNot(features["sample_1"], features["sample_2"])
        self.assertEqual(features["sample_1"].dtype, torch.float32)


class StateAndDeterminismTest(unittest.TestCase):
    def test_extracts_exact_505_query_encoder_tensors(self) -> None:
        payload = {
            f"query_encoder.tensor_{index}": torch.tensor(float(index))
            for index in range(EXPECTED_QUERY_STATE_TENSORS)
        }
        payload["separator.unrelated"] = torch.tensor(1.0)

        state = extract_query_encoder_state(payload)

        self.assertEqual(len(state), EXPECTED_QUERY_STATE_TENSORS)
        self.assertIn("tensor_0", state)
        self.assertNotIn("separator.unrelated", state)

    def test_rejects_incomplete_query_encoder_state(self) -> None:
        payload = {
            f"query_encoder.tensor_{index}": torch.tensor(float(index))
            for index in range(EXPECTED_QUERY_STATE_TENSORS - 1)
        }
        with self.assertRaisesRegex(RuntimeError, "expected 505, got 504"):
            extract_query_encoder_state(payload)

    def test_state_load_has_zero_effective_missing_and_unexpected_keys(self) -> None:
        encoder = torch.nn.Linear(2, 1, bias=False)
        state = {
            "weight": torch.tensor([[2.0, 3.0]]),
            "text.embeddings.position_ids": torch.arange(4).reshape(1, 4),
        }

        provenance = load_query_encoder_state_exact(encoder, state)

        self.assertEqual(provenance["effective_missing_keys"], [])
        self.assertEqual(provenance["effective_unexpected_keys"], [])
        self.assertEqual(
            provenance["ignored_nonpersistent_checkpoint_keys"],
            ["text.embeddings.position_ids"],
        )
        self.assertTrue(
            torch.equal(encoder.weight.detach(), torch.tensor([[2.0, 3.0]]))
        )

    def test_cuda_fails_before_execution_when_cublas_env_is_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "CUBLAS_WORKSPACE_CONFIG"):
                configure_determinism(torch.device("cuda"), seed=2026)


class ManifestAndFormatTest(unittest.TestCase):
    @staticmethod
    def _write_manifest(root: Path) -> Path:
        audio = root / "mixture.wav"
        sf.write(
            audio,
            np.zeros(CANONICAL_NUM_SAMPLES, dtype=np.float32),
            CANONICAL_SAMPLE_RATE,
            subtype="PCM_16",
        )
        common = {
            "scene_id": "scene_0",
            "schema_version": "synthetic_strict_test_v1",
            "mixture_path": "mixture.wav",
            "sample_rate": CANONICAL_SAMPLE_RATE,
            "num_samples": CANONICAL_NUM_SAMPLES,
            "num_channels": 1,
            "duration_seconds": 10.0,
            "events": [{"label": "ULTRA_SECRET_EVENT_LABEL"}],
            "answer": "ULTRA_SECRET_ANSWER_LABEL",
        }
        rows = [
            {
                **common,
                "id": "sample_0",
                "question": "ULTRA_SECRET_FULL_QUESTION zero?",
            },
            {
                **common,
                "id": "sample_1",
                "question": "ULTRA_SECRET_FULL_QUESTION one?",
            },
        ]
        manifest = root / "val_fixture.jsonl"
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return manifest

    def test_manifest_deduplicates_scene_and_discards_semantic_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self._write_manifest(Path(temporary))

            samples, scenes, schemas = read_strict_manifest(manifest)

        self.assertEqual(
            [sample.sample_id for sample in samples], ["sample_0", "sample_1"]
        )
        self.assertEqual([scene.scene_id for scene in scenes], ["scene_0"])
        self.assertEqual(schemas, ["synthetic_strict_test_v1"])
        self.assertFalse(hasattr(scenes[0], "answer"))
        self.assertFalse(hasattr(scenes[0], "events"))

    def test_payload_whitelists_reject_semantic_label_fields(self) -> None:
        questions = question_payload({"sample": torch.ones(JOINT_DIM)})
        audio = torch.randn(EFFECTIVE_FRAMES, JOINT_DIM)
        audio = torch.nn.functional.normalize(audio, dim=-1).half()
        scenes = scene_payload({"scene": audio.contiguous()})

        validate_question_payload(questions)
        validate_scene_payload(scenes)
        self.assertEqual(questions["format"], QUESTION_CACHE_FORMAT)
        self.assertEqual(scenes["format"], SCENE_CACHE_FORMAT)
        with self.assertRaisesRegex(ValueError, "unexpected question-cache fields"):
            validate_question_payload({**questions, "answer_labels": ["secret"]})
        with self.assertRaisesRegex(ValueError, "unexpected scene-cache fields"):
            validate_scene_payload({**scenes, "event_labels": ["secret"]})

    def test_atomic_artifacts_store_only_ids_and_tensors(self) -> None:
        questions = question_payload({"sample_0": torch.ones(JOINT_DIM)})
        audio = torch.randn(EFFECTIVE_FRAMES, JOINT_DIM)
        audio = torch.nn.functional.normalize(audio, dim=-1).half().contiguous()
        scenes = scene_payload({"scene_0": audio})
        receipt = {
            "format": RECEIPT_FORMAT,
            "sample_ids": ["sample_0"],
            "scene_ids": ["scene_0"],
        }
        secrets = (
            b"ULTRA_SECRET_FULL_QUESTION",
            b"ULTRA_SECRET_EVENT_LABEL",
            b"ULTRA_SECRET_ANSWER_LABEL",
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cache"
            write_outputs_atomically(output, questions, scenes, receipt)
            loaded_questions = torch.load(
                output / "question_features.pt", weights_only=True
            )
            loaded_scenes = torch.load(
                output / "scene_audio_features.pt", weights_only=True
            )
            validate_question_payload(loaded_questions)
            validate_scene_payload(loaded_scenes)
            combined = b"".join(path.read_bytes() for path in output.iterdir())
            for secret in secrets:
                self.assertNotIn(secret, combined)
            with self.assertRaises(FileExistsError):
                write_outputs_atomically(output, questions, scenes, receipt)

    def test_cli_help_requires_no_checkpoint_or_model_load(self) -> None:
        stream = io.StringIO()
        with redirect_stdout(stream), self.assertRaises(SystemExit) as context:
            parse_args(["--help"])
        self.assertEqual(context.exception.code, 0)
        help_text = stream.getvalue()
        self.assertIn("--manifest", help_text)
        self.assertIn("--audiosep-checkpoint", help_text)
        self.assertIn("--output-dir", help_text)


class Real10InferenceManifestTest(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
        audio_dir = root / "audio"
        audio_dir.mkdir()
        audio = audio_dir / "scene_0.wav"
        sf.write(
            audio,
            np.linspace(
                -0.25,
                0.25,
                CANONICAL_NUM_SAMPLES,
                dtype=np.float32,
            ),
            CANONICAL_SAMPLE_RATE,
            format="WAV",
            subtype="FLOAT",
        )
        row: dict[str, object] = {
            "schema_version": REAL10_INFERENCE_SCHEMA_VERSION,
            "id": "real_dev_scene_0_q0",
            "scene_id": "scene_0",
            "scene_family_id": "family_0",
            "split": "real_dev",
            "question_index": 0,
            "question_type": "temporal_after",
            "relation": "after",
            "question": "What sound occurs after the bell?",
            "sample_rate": CANONICAL_SAMPLE_RATE,
            "num_channels": 1,
            "num_samples": CANONICAL_NUM_SAMPLES,
            "duration_seconds": 10.0,
            "mixture_path": "audio/scene_0.wav",
            "mixture_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        }
        manifest = root / "qces_real10_inference.jsonl"
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return manifest, audio, row

    def test_exact_inference_view_binds_manifest_and_canonical_wav_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, row = self._write_fixture(Path(temporary))

            parsed = read_cache_manifest(manifest)
            waveform = read_canonical_waveform(parsed.scenes[0])
            receipt = real10_receipt_contract(parsed)

        records = parse_inference_manifest([row])
        self.assertTrue(parsed.is_real10_inference)
        self.assertEqual(
            parsed.real10_inference_manifest_fingerprint,
            canonical_inference_manifest_fingerprint(records),
        )
        self.assertEqual(parsed.schema_versions, (REAL10_INFERENCE_SCHEMA_VERSION,))
        self.assertEqual(waveform.shape, (CANONICAL_NUM_SAMPLES,))
        self.assertEqual(
            receipt["canonical_wav_sha256_by_scene"],
            {"scene_0": row["mixture_sha256"]},
        )
        self.assertFalse(receipt["contains_event_answer_or_oracle_inputs"])
        self.assertNotIn("answer", receipt)
        self.assertNotIn("event_labels", receipt)

    def test_inference_schema_rejects_scoring_and_oracle_extras(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, row = self._write_fixture(root)
            row["answer"] = "glass breaking"
            row["oracle_evidence_path"] = "audio/oracle.wav"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "oracle clean-stem|fields mismatch"
            ):
                read_cache_manifest(manifest)

            row.pop("answer")
            row.pop("oracle_evidence_path")
            row["schema_version"] = "qces_real10_scoring_v2"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "scoring/oracle rows are forbidden"
            ):
                read_cache_manifest(manifest)

    def test_traversal_and_symlinked_audio_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, audio, row = self._write_fixture(root)
            row["mixture_path"] = "../scene_0.wav"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "safe relative path"):
                read_cache_manifest(manifest)

            alias = root / "audio" / "alias.wav"
            alias.symlink_to(audio)
            row["mixture_path"] = "audio/alias.wav"
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not use symlinks"):
                read_cache_manifest(manifest)

    def test_symlinked_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, _ = self._write_fixture(root)
            alias = root / "manifest_alias.jsonl"
            alias.symlink_to(manifest)
            with self.assertRaisesRegex(ValueError, "must not use symlinks"):
                read_cache_manifest(alias)

    def test_declared_wav_sha_mismatch_is_rejected_before_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, row = self._write_fixture(root)
            row["mixture_sha256"] = "0" * 64
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "mixture_sha256"):
                read_cache_manifest(manifest)

    def test_noncanonical_wav_subtype_is_rejected_before_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, audio, row = self._write_fixture(root)
            sf.write(
                audio,
                np.zeros(CANONICAL_NUM_SAMPLES, dtype=np.float32),
                CANONICAL_SAMPLE_RATE,
                format="WAV",
                subtype="PCM_16",
            )
            row["mixture_sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "IEEE-float WAV"):
                read_cache_manifest(manifest)

    def test_audio_mutation_during_decode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, audio, _ = self._write_fixture(Path(temporary))
            parsed = read_cache_manifest(manifest)
            original_read = cache_module.sf.read

            def mutate_after_decode(*args: object, **kwargs: object) -> object:
                decoded = original_read(*args, **kwargs)
                audio.write_bytes(b"mutated-after-byte-buffer-capture")
                return decoded

            with patch.object(
                cache_module.sf,
                "read",
                side_effect=mutate_after_decode,
            ), self.assertRaisesRegex(RuntimeError, "mixture_sha256|mutated"):
                read_canonical_waveform(parsed.scenes[0])

    def test_manifest_byte_mutation_fails_snapshot_identity_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _ = self._write_fixture(Path(temporary))
            before = cache_module.file_identity(manifest)
            read_cache_manifest(manifest)
            manifest.write_text(
                manifest.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "changed while"):
                cache_module._assert_identity_unchanged(
                    "manifest",
                    before,
                    cache_module.file_identity(manifest),
                )

    def test_main_receipt_carries_real10_contract_without_gpu_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _, row = self._write_fixture(root)
            audiosep_root = root / "audiosep"
            audiosep_root.mkdir()
            checkpoint = root / "audiosep.ckpt"
            checkpoint.write_bytes(b"fixture checkpoint")
            output = root / "cache"
            source_identity = {
                "path": str(audiosep_root),
                "sha256": "1" * 64,
                "hashed_file_count": 1,
                "included_suffixes": [".py"],
            }
            question_features = {
                "real_dev_scene_0_q0": torch.ones(JOINT_DIM, dtype=torch.float32)
            }
            audio_feature = torch.zeros(
                EFFECTIVE_FRAMES, JOINT_DIM, dtype=torch.float32
            )
            audio_feature[:, 0] = 1.0
            writer = patch.object(
                cache_module,
                "write_outputs_atomically",
                return_value=output,
            )
            with patch.object(
                cache_module,
                "source_tree_identity",
                return_value=source_identity,
            ), patch.object(
                cache_module,
                "configure_determinism",
                return_value={"seed": 2026},
            ), patch.object(
                cache_module,
                "load_frozen_encoder",
                return_value=(torch.nn.Identity(), {"fixture": True}),
            ), patch.object(
                cache_module,
                "encode_full_questions",
                return_value=question_features,
            ), patch.object(
                cache_module,
                "encode_scene",
                return_value=audio_feature.half().contiguous(),
            ), writer as mocked_writer:
                cache_module.main(
                    [
                        "--manifest",
                        str(manifest),
                        "--audiosep-root",
                        str(audiosep_root),
                        "--audiosep-checkpoint",
                        str(checkpoint),
                        "--output-dir",
                        str(output),
                        "--device",
                        "cpu",
                    ]
                )

            receipt = mocked_writer.call_args.args[3]
            contract = receipt["qces_real10_inference_contract"]
            self.assertEqual(contract["schema_version"], row["schema_version"])
            self.assertEqual(
                contract["canonical_wav_sha256_by_scene"],
                {"scene_0": row["mixture_sha256"]},
            )
            self.assertFalse(contract["contains_event_answer_or_oracle_inputs"])
            self.assertFalse(
                receipt["privacy_contract"]["contains_event_answer_or_oracle_inputs"]
            )


if __name__ == "__main__":
    unittest.main()
