"""CPU-only tests for the leakage-safe QCES-Real-10 renderer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from mixi_understanding.data.qces_real10_schema import (
    DURATION_SECONDS,
    INFERENCE_SCHEMA_VERSION,
    NUM_CHANNELS,
    NUM_SAMPLES,
    SAMPLE_RATE,
    canonical_inference_manifest_fingerprint,
    parse_inference_record,
)
from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    QCESConfig,
)
from mixi_understanding.qces.real10_prediction import (
    MAX_RECONSTRUCTION_ABS_ERROR,
    PREDICTION_MANIFEST_FILENAME,
    RENDER_RECEIPT_FILENAME,
    CanonicalMixture,
    PreparedRenderInputs,
    Real10FoundationFeatureCache,
    Real10RenderError,
    RenderSettings,
    _aggregate_cache_mixtures,
    _rename_directory_noreplace,
    load_canonical_mixtures,
    load_real10_foundation_feature_cache,
    pcm_f32le_sha256,
    prediction_manifest_fingerprint,
    read_inference_manifest,
    render_prediction_set,
    runtime_source_identity,
    validate_prediction_record,
    write_deterministic_float_wav,
    write_method_freeze_receipt,
)
from mixi_understanding.scripts.cache_audiosep_clap_features import (
    EFFECTIVE_FRAMES,
    EXPECTED_QUERY_STATE_TENSORS,
    FEATURE_SPACE,
    FINE_DIM,
    JOINT_DIM,
    RAW_FINE_FRAMES,
    RECEIPT_FORMAT,
    REPEAT_RATIO,
    file_identity,
    question_payload,
    scene_payload,
    source_tree_identity,
)


def _waveform() -> np.ndarray:
    phase = np.arange(NUM_SAMPLES, dtype=np.float32) / np.float32(SAMPLE_RATE)
    return np.ascontiguousarray(
        0.1 * np.sin(2.0 * np.pi * 137.0 * phase), dtype=np.float32
    )


def _inference_row(
    *,
    split: str,
    audio_sha256: str,
    sample_id: str | None = None,
    scene_id: str | None = None,
) -> dict[str, object]:
    scene = scene_id or f"scene_{split}"
    return {
        "schema_version": INFERENCE_SCHEMA_VERSION,
        "id": sample_id or f"{scene}_q0",
        "scene_id": scene,
        "scene_family_id": scene,
        "split": split,
        "question_index": 0,
        "question_type": "temporal_after",
        "relation": "after",
        "question": "What sound occurs after the bell?",
        "sample_rate": SAMPLE_RATE,
        "num_channels": NUM_CHANNELS,
        "num_samples": NUM_SAMPLES,
        "duration_seconds": DURATION_SECONDS,
        "mixture_path": f"audio/{scene}.wav",
        "mixture_sha256": audio_sha256,
    }


def _write_manifest_fixture(root: Path, split: str = "real_dev") -> tuple[Path, object]:
    scene_id = f"scene_{split}"
    audio = root / "audio" / f"{scene_id}.wav"
    write_deterministic_float_wav(audio, _waveform())
    row = _inference_row(
        split=split,
        audio_sha256=file_identity(audio)["sha256"],
        scene_id=scene_id,
    )
    manifest = root / "qces_real10_inference.jsonl"
    manifest.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    return manifest, parse_inference_record(row)


class _DummyComposition:
    def __init__(self) -> None:
        self.role_logits = torch.tensor(
            [
                [
                    [10.0, -10.0, -10.0],
                    [-10.0, 10.0, -10.0],
                    [-10.0, 10.0, 10.0],
                    [-10.0, -10.0, 10.0],
                    [10.0, -10.0, -10.0],
                ]
            ],
            dtype=torch.float32,
        )
        self.no_evidence_logit = torch.zeros(1, dtype=torch.float32)
        self.frame_hop_samples = NUM_SAMPLES // 4
        self.temporal_role_mode = OVERLAP_AWARE_TEMPORAL_ROLE_MODE

    @property
    def role_probabilities(self) -> torch.Tensor:
        return self.role_logits.sigmoid()

    @property
    def same_semantic_probability(self) -> None:
        return None


class _DummyModel:
    def __init__(self, config: QCESConfig, *, bad_reconstruction: bool = False) -> None:
        self.config = config
        self.bad_reconstruction = bad_reconstruction
        self.forward_count = 0

    def forward_questions(
        self,
        mixture: torch.Tensor,
        questions: list[str],
        tokenizer: object,
        *,
        question_clap: torch.Tensor,
        scene_clap: torch.Tensor,
    ) -> object:
        del tokenizer
        self.forward_count += 1
        if mixture.shape != (1, NUM_SAMPLES) or len(questions) != 1:
            raise AssertionError("renderer did not use batch size one")
        if question_clap.shape != (1, JOINT_DIM):
            raise AssertionError("question cache shape drift")
        if scene_clap.shape != (1, EFFECTIVE_FRAMES, JOINT_DIM):
            raise AssertionError("scene cache shape drift")
        evidence = mixture * 0.25
        residual = mixture - evidence
        if self.bad_reconstruction:
            residual = residual + 1e-4
        return SimpleNamespace(
            evidence=evidence,
            residual=residual,
            composition=_DummyComposition(),
            separation=SimpleNamespace(
                semantic_separation_mode="union_single",
                physical_separator_forwards_per_batch=1,
                effective_separator_evaluations_per_record=1,
            ),
        )


def _make_prepared(root: Path, split: str) -> PreparedRenderInputs:
    manifest, record = _write_manifest_fixture(root, split)
    records = (record,)
    mixture_path = root / record.mixture_path
    samples = np.ascontiguousarray(_waveform())
    mixture_identity = file_identity(mixture_path)
    mixture = CanonicalMixture(
        scene_id=record.scene_id,
        manifest_path=record.mixture_path,
        path=mixture_path.resolve(),
        samples=samples,
        file_identity=mixture_identity,
        pcm_f32le_sha256=pcm_f32le_sha256(samples),
    )
    question = torch.zeros(JOINT_DIM, dtype=torch.float32)
    question[0] = 1.0
    scene = torch.zeros(EFFECTIVE_FRAMES, JOINT_DIM, dtype=torch.float16)
    scene[:, 0] = 1.0
    cache_dir = root / "cache"
    cache_dir.mkdir()
    cache_receipt = cache_dir / "cache_receipt.json"
    question_artifact = cache_dir / "question_features.pt"
    scene_artifact = cache_dir / "scene_audio_features.pt"
    cache_receipt.write_text("{}\n", encoding="utf-8")
    question_artifact.write_bytes(b"question fixture")
    scene_artifact.write_bytes(b"scene fixture")
    feature_cache = Real10FoundationFeatureCache(
        question_features={record.sample_id: question},
        scene_features={record.scene_id: scene},
        sample_to_scene={record.sample_id: record.scene_id},
        identity={},
    )

    audiosep_root = root / "audiosep"
    audiosep_root.mkdir()
    (audiosep_root / "model.py").write_text("MODEL = 'fixture'\n", encoding="utf-8")
    audiosep_config = root / "audiosep.yaml"
    audiosep_checkpoint = root / "audiosep.bin"
    qces_checkpoint = root / "qces.pt"
    audiosep_config.write_text("model: fixture\n", encoding="utf-8")
    audiosep_checkpoint.write_bytes(b"official frozen fixture")
    qces_checkpoint.write_bytes(b"learned controller fixture")
    source_identity = source_tree_identity(audiosep_root)
    config_identity = file_identity(audiosep_config)
    checkpoint_identity = file_identity(audiosep_checkpoint)
    cache_identity = {
        "receipt": file_identity(cache_receipt),
        "question_feature_artifact": file_identity(question_artifact),
        "scene_feature_artifact": file_identity(scene_artifact),
        "manifest_binding": {"fixture": True},
        "audiosep_checkpoint_binding": {
            "run_expected": checkpoint_identity,
        },
        "audiosep_source_binding": {"run_expected": source_identity},
        "sample_count": 1,
        "scene_count": 1,
        "contract_sha256": "a" * 64,
        "contains_gold_label_inputs": False,
    }
    feature_cache = Real10FoundationFeatureCache(
        question_features=feature_cache.question_features,
        scene_features=feature_cache.scene_features,
        sample_to_scene=feature_cache.sample_to_scene,
        identity=cache_identity,
    )
    config = QCESConfig(
        foundation_feature_mode=AUDIOSEP_CLAP_FOUNDATION_FEATURES,
        temporal_role_mode=OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    )
    fingerprint = canonical_inference_manifest_fingerprint(records)
    return PreparedRenderInputs(
        manifest_path=manifest.resolve(),
        manifest_identity=file_identity(manifest),
        records=records,
        selected_records=records,
        full_manifest_fingerprint=fingerprint,
        selected_manifest_fingerprint=fingerprint,
        mixtures={record.scene_id: mixture},
        foundation_cache=feature_cache,
        foundation_cache_contract={
            "format": RECEIPT_FORMAT,
            "fixture_contract": True,
        },
        foundation_cache_identity=cache_identity,
        qces_checkpoint_path=qces_checkpoint.resolve(),
        qces_checkpoint_identity=file_identity(qces_checkpoint),
        qces_checkpoint_payload={"fixture": True},
        qces_config=config,
        audiosep_root=audiosep_root.resolve(),
        audiosep_source_identity=source_identity,
        audiosep_config_path=audiosep_config.resolve(),
        audiosep_config_identity=config_identity,
        audiosep_checkpoint_path=audiosep_checkpoint.resolve(),
        audiosep_checkpoint_identity=checkpoint_identity,
        runtime_source_identity=runtime_source_identity(),
    )


class StrictInputAndWavTest(unittest.TestCase):
    def test_inference_only_manifest_and_canonical_audio_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, record = _write_manifest_fixture(root)
            records, identity, fingerprint = read_inference_manifest(manifest)
            mixtures = load_canonical_mixtures(manifest, records)

            self.assertEqual(records, (record,))
            self.assertEqual(identity["sha256"], file_identity(manifest)["sha256"])
            self.assertEqual(
                fingerprint, canonical_inference_manifest_fingerprint(records)
            )
            self.assertEqual(
                mixtures[record.scene_id].pcm_f32le_sha256,
                pcm_f32le_sha256(_waveform()),
            )

            scoring_like = record.to_dict()
            scoring_like["answer"] = "secret"
            manifest.write_text(json.dumps(scoring_like) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fields mismatch"):
                read_inference_manifest(manifest)

    def test_deterministic_float_writer_allows_exact_silence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            silence = np.zeros(NUM_SAMPLES, dtype=np.float32)
            first = write_deterministic_float_wav(root / "first.wav", silence)
            second = write_deterministic_float_wav(root / "second.wav", silence)
            self.assertEqual(first["file_sha256"], second["file_sha256"])
            self.assertEqual(first["pcm_f32le_sha256"], second["pcm_f32le_sha256"])

    def test_atomic_directory_install_never_replaces_an_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "staging"
            destination = root / "output"
            source.mkdir()
            destination.mkdir()
            (source / "new.txt").write_text("new\n", encoding="utf-8")
            (destination / "owned.txt").write_text("owned\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _rename_directory_noreplace(source, destination)
            self.assertTrue((source / "new.txt").is_file())
            self.assertEqual(
                (destination / "owned.txt").read_text(encoding="utf-8"), "owned\n"
            )


class HardenedRealCacheLoaderTest(unittest.TestCase):
    def _build_cache(self, root: Path) -> tuple[Path, Path, tuple, dict, dict]:
        manifest, record = _write_manifest_fixture(root)
        records = (record,)
        mixtures = load_canonical_mixtures(manifest, records)
        cache = root / "foundation"
        cache.mkdir()
        question = torch.zeros(JOINT_DIM, dtype=torch.float32)
        question[0] = 1.0
        scene = torch.zeros(EFFECTIVE_FRAMES, JOINT_DIM, dtype=torch.float16)
        scene[:, 0] = 1.0
        torch.save(
            question_payload({record.sample_id: question}),
            cache / "question_features.pt",
        )
        torch.save(
            scene_payload({record.scene_id: scene}),
            cache / "scene_audio_features.pt",
        )
        checkpoint = root / "audiosep.bin"
        checkpoint.write_bytes(b"official checkpoint")
        checkpoint_identity = file_identity(checkpoint)
        source_identity = {
            "path": str(root / "audiosep"),
            "sha256": "b" * 64,
            "hashed_file_count": 2,
            "included_suffixes": [".py", ".yaml"],
        }
        mixture_identity = mixtures[record.scene_id].file_identity
        mixture_rows = [
            {
                "scene_id": record.scene_id,
                "manifest_path": record.mixture_path,
                "sha256": mixture_identity["sha256"],
                "size_bytes": mixture_identity["size_bytes"],
                "sample_rate": SAMPLE_RATE,
                "num_samples": NUM_SAMPLES,
                "duration_seconds": DURATION_SECONDS,
                "feature_shape": [EFFECTIVE_FRAMES, JOINT_DIM],
                "feature_dtype": "float16",
                "declared_mixture_sha256": mixture_identity["sha256"],
            }
        ]
        fingerprint = canonical_inference_manifest_fingerprint(records)
        receipt = {
            "format": RECEIPT_FORMAT,
            "purpose": "frozen_controller_inputs_without_oracle_labels",
            "manifest": file_identity(manifest),
            "schema_versions": [INFERENCE_SCHEMA_VERSION],
            "audiosep_checkpoint": checkpoint_identity,
            "audiosep_source_tree": source_identity,
            "query_encoder_state": {
                "extracted_tensor_count": EXPECTED_QUERY_STATE_TENSORS,
                "loaded_tensor_count": EXPECTED_QUERY_STATE_TENSORS,
                "effective_missing_keys": [],
                "effective_unexpected_keys": [],
                "ignored_nonpersistent_checkpoint_keys": [],
                "fusion_enabled": False,
            },
            "canonical_audio": {
                "sample_rate": SAMPLE_RATE,
                "num_samples": NUM_SAMPLES,
                "duration_seconds": DURATION_SECONDS,
                "num_channels": NUM_CHANNELS,
                "clap_sample_rate": 48_000,
                "clap_num_samples": 480_000,
            },
            "features": {
                "feature_space": FEATURE_SPACE,
                "question_shape": [JOINT_DIM],
                "question_dtype": "float32",
                "raw_audio_shape": [RAW_FINE_FRAMES, FINE_DIM],
                "effective_audio_shape": [EFFECTIVE_FRAMES, JOINT_DIM],
                "effective_audio_dtype": "float16",
                "raw_to_effective_repeat_ratio": REPEAT_RATIO,
                "audio_normalization": "per_frame_l2_before_fp16_storage",
            },
            "counts": {
                "sample_ids": 1,
                "scene_ids": 1,
                "unique_full_question_texts": 1,
                "physical_mixture_encodes": 1,
            },
            "sample_ids": [record.sample_id],
            "scene_ids": [record.scene_id],
            "mixtures": mixture_rows,
            "mixtures_aggregate_sha256": _aggregate_cache_mixtures(mixture_rows),
            "execution": {
                "device": "cpu",
                "text_batch_size": 1,
                "determinism": {
                    "seed": 2026,
                    "torch_deterministic_algorithms": True,
                    "cublas_workspace_config": None,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                },
                "software": {"fixture": True},
            },
            "privacy_contract": {
                "question_text_stored": False,
                "event_labels_stored": False,
                "answer_labels_stored": False,
                "evidence_annotations_stored": False,
                "allowed_keys": "sample_id_and_scene_id_only",
                "contains_event_answer_or_oracle_inputs": False,
            },
            "artifacts": {
                "question_features": {
                    "filename": "question_features.pt",
                    **{
                        key: value
                        for key, value in file_identity(
                            cache / "question_features.pt"
                        ).items()
                        if key in {"sha256", "size_bytes"}
                    },
                },
                "scene_audio_features": {
                    "filename": "scene_audio_features.pt",
                    **{
                        key: value
                        for key, value in file_identity(
                            cache / "scene_audio_features.pt"
                        ).items()
                        if key in {"sha256", "size_bytes"}
                    },
                },
            },
            "qces_real10_inference_contract": {
                "schema_version": INFERENCE_SCHEMA_VERSION,
                "canonical_inference_manifest_fingerprint": fingerprint,
                "canonical_wav_sha256_by_scene": {
                    record.scene_id: record.mixture_sha256
                },
                "contains_event_answer_or_oracle_inputs": False,
            },
        }
        (cache / "cache_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return cache, manifest, records, checkpoint_identity, source_identity

    def test_explicit_real_branch_validates_fingerprint_hashes_and_tensors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache, manifest, records, checkpoint, source = self._build_cache(root)
            mixtures = load_canonical_mixtures(manifest, records)
            fingerprint = canonical_inference_manifest_fingerprint(records)

            loaded, contract = load_real10_foundation_feature_cache(
                cache,
                manifest,
                records,
                expected_manifest_fingerprint=fingerprint,
                mixtures=mixtures,
                audiosep_checkpoint_identity=checkpoint,
                audiosep_source_identity=source,
            )
            self.assertEqual(set(loaded.question_features), {records[0].sample_id})
            self.assertEqual(set(loaded.scene_features), {records[0].scene_id})
            self.assertFalse(loaded.identity["contains_gold_label_inputs"])
            self.assertEqual(
                contract["real10_schema_version"], INFERENCE_SCHEMA_VERSION
            )

            receipt_path = cache / "cache_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["qces_real10_inference_contract"][
                "canonical_inference_manifest_fingerprint"
            ] = ("0" * 64)
            receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Real10RenderError, "binding is invalid"):
                load_real10_foundation_feature_cache(
                    cache,
                    manifest,
                    records,
                    expected_manifest_fingerprint=fingerprint,
                    mixtures=mixtures,
                    audiosep_checkpoint_identity=checkpoint,
                    audiosep_source_identity=source,
                )


class PredictionRenderAndResumeTest(unittest.TestCase):
    def test_batch1_render_contract_silence_safe_stems_and_exact_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = _make_prepared(root, "real_dev")
            settings = RenderSettings(
                split="real_dev",
                role_threshold=0.5,
                no_evidence_threshold=0.6,
                seed=2026,
                device_type="cpu",
            )
            model = _DummyModel(prepared.qces_config)
            output = root / "render"
            result = render_prediction_set(
                prepared=prepared,
                settings=settings,
                output_dir=output,
                model_loader=lambda *_: model,
            )
            self.assertFalse(result.resumed)
            self.assertEqual(model.forward_count, 1)
            self.assertTrue(result.receipt["gates"]["reconstruction_gate_passed"])
            self.assertLessEqual(
                result.receipt["gates"]["observed_maximum_↓"],
                MAX_RECONSTRUCTION_ABS_ERROR,
            )
            rows = [
                json.loads(line)
                for line in (output / PREDICTION_MANIFEST_FILENAME)
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            row = validate_prediction_record(rows[0])
            self.assertEqual(row["prediction"]["anchor_intervals"], [[2.5, 7.5]])
            self.assertEqual(row["prediction"]["answer_intervals"], [[5.0, 10.0]])
            self.assertEqual(row["prediction"]["union_intervals"], [[2.5, 10.0]])
            self.assertFalse(row["prediction"]["no_evidence_prediction"])
            self.assertEqual(
                result.receipt["artifacts"]["prediction_manifest"][
                    "canonical_fingerprint"
                ],
                prediction_manifest_fingerprint(rows),
            )
            for stem in row["stems"].values():
                self.assertTrue((output / stem["path"]).is_file())
                self.assertEqual(
                    file_identity(output / stem["path"])["sha256"],
                    stem["file_sha256"],
                )
            public_json = (
                (output / PREDICTION_MANIFEST_FILENAME).read_text(encoding="utf-8")
                + (output / RENDER_RECEIPT_FILENAME).read_text(encoding="utf-8")
            ).casefold()
            self.assertNotIn("oracle", public_json)
            self.assertNotIn("sdr", public_json)

            resumed = render_prediction_set(
                prepared=prepared,
                settings=settings,
                output_dir=output,
                resume=True,
                model_loader=lambda *_: (_ for _ in ()).throw(
                    AssertionError("exact resume must not load a model")
                ),
            )
            self.assertTrue(resumed.resumed)
            self.assertTrue((output / RENDER_RECEIPT_FILENAME).is_file())

            (output / "undeclared.txt").write_text("bad\n", encoding="utf-8")
            with self.assertRaisesRegex(Real10RenderError, "undeclared files"):
                render_prediction_set(
                    prepared=prepared,
                    settings=settings,
                    output_dir=output,
                    resume=True,
                )

    def test_reconstruction_gate_refuses_installation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = _make_prepared(root, "real_dev")
            output = root / "failed_render"
            with self.assertRaisesRegex(Real10RenderError, "reconstruction gate"):
                render_prediction_set(
                    prepared=prepared,
                    settings=RenderSettings(split="real_dev", device_type="cpu"),
                    output_dir=output,
                    model_loader=lambda *_: _DummyModel(
                        prepared.qces_config, bad_reconstruction=True
                    ),
                )
            self.assertFalse(output.exists())
            self.assertFalse(list(root.glob(".failed_render.building-*")))

    def test_prediction_validator_rejects_non_union_temporal_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = _make_prepared(root, "real_dev")
            output = root / "render"
            render_prediction_set(
                prepared=prepared,
                settings=RenderSettings(split="real_dev", device_type="cpu"),
                output_dir=output,
                model_loader=lambda *_: _DummyModel(prepared.qces_config),
            )
            row = json.loads(
                (output / PREDICTION_MANIFEST_FILENAME)
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            row["prediction"]["union_intervals"] = []
            with self.assertRaisesRegex(Real10RenderError, "anchor-answer union"):
                validate_prediction_record(row)


class RealTestFreezeGateTest(unittest.TestCase):
    def test_real_test_needs_flag_and_matching_dev_freeze_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev = _make_prepared(root / "dev", "real_dev")
            test = _make_prepared(root / "test", "real_test")
            dev_settings = RenderSettings(split="real_dev", device_type="cpu")
            test_settings = RenderSettings(split="real_test", device_type="cpu")
            freeze_path = root / "method_freeze.json"
            write_method_freeze_receipt(
                freeze_path,
                prepared=dev,
                settings=dev_settings,
            )

            with self.assertRaisesRegex(
                Real10RenderError, "requires --allow-real-test"
            ):
                render_prediction_set(
                    prepared=test,
                    settings=test_settings,
                    output_dir=root / "not_allowed",
                    model_loader=lambda *_: _DummyModel(test.qces_config),
                )
            result = render_prediction_set(
                prepared=test,
                settings=test_settings,
                output_dir=root / "allowed",
                allow_real_test=True,
                method_freeze_receipt_path=freeze_path,
                model_loader=lambda *_: _DummyModel(test.qces_config),
            )
            self.assertFalse(result.resumed)

            with self.assertRaisesRegex(Real10RenderError, "differs from the method"):
                render_prediction_set(
                    prepared=test,
                    settings=RenderSettings(
                        split="real_test",
                        no_evidence_threshold=0.55,
                        device_type="cpu",
                    ),
                    output_dir=root / "threshold_changed",
                    allow_real_test=True,
                    method_freeze_receipt_path=freeze_path,
                    model_loader=lambda *_: _DummyModel(test.qces_config),
                )


if __name__ == "__main__":
    unittest.main()
