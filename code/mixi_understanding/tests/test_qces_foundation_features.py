"""CPU-only tests for opt-in offline frozen AudioSep-CLAP features."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mixi_understanding.qces.composer import (
    FOUNDATION_CLAP_DIM,
    FOUNDATION_CLAP_FRAMES,
    RoleAwarePromptComposer,
    foundation_semantic_mix,
    role_pool_scene_clap,
)
from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    CONVEX_SEMANTIC_INTERPOLATION,
    DUAL_ROLE_SEMANTIC_MODE,
    LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
    NO_FOUNDATION_FEATURES,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    QUESTION_RESIDUAL_SEMANTIC_MIXING,
    QCESConfig,
)
from mixi_understanding.qces.model import QCESModel, load_qces_checkpoint
from mixi_understanding.qces.separators import AudioSepConditionedAdapter
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.cache_audiosep_clap_features import (
    FEATURE_SPACE,
    RECEIPT_FORMAT,
    source_tree_identity,
    question_payload,
    scene_payload,
    write_outputs_atomically,
)
from mixi_understanding.scripts.infer_qces import (
    extract_online_foundation_features,
    main as infer_main,
    require_online_foundation_support,
)
from mixi_understanding.scripts.train_qces import (
    FoundationFeatureCache,
    _aggregate_foundation_mixture_receipt,
    add_foundation_features,
    file_identity,
    forward_training_batch,
    load_foundation_feature_cache,
    parse_args,
    validate_foundation_training_args,
)


def tiny_config(
    *,
    foundation: bool = False,
    dual: bool = False,
    semantic_mixing: str = LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
) -> QCESConfig:
    return QCESConfig(
        sample_rate=8_000,
        n_fft=64,
        hop_length=16,
        win_length=64,
        vocab_size=128,
        max_question_tokens=16,
        question_dim=16,
        audio_dim=16,
        condition_dim=512 if foundation else 16,
        attention_heads=4,
        question_layers=1,
        separator_channels=8,
        separator_layers=2,
        dropout=0.0,
        temporal_role_mode=(
            OVERLAP_AWARE_TEMPORAL_ROLE_MODE if dual else "exclusive_softmax"
        ),
        semantic_separation_mode=(DUAL_ROLE_SEMANTIC_MODE if dual else "union_single"),
        foundation_feature_mode=(
            AUDIOSEP_CLAP_FOUNDATION_FEATURES if foundation else NO_FOUNDATION_FEATURES
        ),
        foundation_semantic_mixing_mode=semantic_mixing,
    )


def normalized_foundation_features(
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(991)
    question = F.normalize(
        torch.randn(batch_size, FOUNDATION_CLAP_DIM, generator=generator),
        dim=-1,
    )
    scene = F.normalize(
        torch.randn(
            batch_size,
            FOUNDATION_CLAP_FRAMES,
            FOUNDATION_CLAP_DIM,
            generator=generator,
        ),
        dim=-1,
    )
    return question, scene


class _FakeAudioSep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):  # type: ignore[no-untyped-def]
        condition = batch["condition"]
        gain = 0.5 + 0.01 * condition[:, :1, None]
        return {"waveform": batch["mixture"] * gain * self.scale}


class FoundationComposerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.waveform = torch.randn(2, 256) * 0.05
        self.tokens = StableHashTokenizer(128, 16).batch_encode(
            ["What follows the bell?", "Which sound starts first?"]
        )

    def test_default_none_has_no_new_state_and_old_checkpoint_is_exact(self) -> None:
        config = tiny_config()
        self.assertEqual(config.foundation_feature_mode, NO_FOUNDATION_FEATURES)
        self.assertEqual(
            config.foundation_semantic_mixing_mode,
            LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
        )
        model = QCESModel(config).eval()
        state_keys = set(model.state_dict())
        self.assertFalse(any("foundation_" in key for key in state_keys))
        with torch.inference_mode():
            expected = model(
                self.waveform,
                self.tokens.input_ids,
                self.tokens.attention_mask,
            )
        payload = model.checkpoint_payload(backend="mask")
        payload["config"].pop("foundation_feature_mode")
        payload["config"].pop("foundation_semantic_mixing_mode")
        restored = load_qces_checkpoint(payload).eval()
        self.assertEqual(set(restored.state_dict()), state_keys)
        with torch.inference_mode():
            actual = restored(
                self.waveform,
                self.tokens.input_ids,
                self.tokens.attention_mask,
            )
        self.assertTrue(torch.equal(expected.evidence, actual.evidence))
        self.assertTrue(
            torch.equal(
                expected.composition.semantic_condition,
                actual.composition.semantic_condition,
            )
        )

    def test_convex_semantic_mode_reaches_candidate_endpoint(self) -> None:
        base = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        target = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        near_one = torch.tensor(20.0)
        legacy = foundation_semantic_mix(
            base,
            target,
            near_one,
            LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
        )
        reachable = foundation_semantic_mix(
            base,
            target,
            near_one,
            CONVEX_SEMANTIC_INTERPOLATION,
        )
        self.assertLess(float((legacy * target).sum()), 0.71)
        self.assertGreater(float((reachable * target).sum()), 0.999999)

        config = tiny_config(
            foundation=True,
            semantic_mixing=CONVEX_SEMANTIC_INTERPOLATION,
        )
        self.assertEqual(
            config.foundation_semantic_mixing_mode,
            CONVEX_SEMANTIC_INTERPOLATION,
        )
        with self.assertRaisesRegex(ValueError, "requires foundation_feature_mode"):
            tiny_config(semantic_mixing=CONVEX_SEMANTIC_INTERPOLATION)

    def test_question_residual_is_reachable_and_identity_initialized(self) -> None:
        base = F.normalize(torch.randn(2, 512), dim=-1)
        target = F.normalize(torch.randn(2, 512), dim=-1)
        delta = target - base
        mixed = foundation_semantic_mix(
            base,
            delta,
            torch.tensor(-20.0),
            QUESTION_RESIDUAL_SEMANTIC_MIXING,
        )
        self.assertTrue(torch.allclose(mixed, target, atol=1e-6, rtol=1e-6))

        model = QCESModel(
            tiny_config(
                foundation=True,
                semantic_mixing=QUESTION_RESIDUAL_SEMANTIC_MIXING,
            )
        ).eval()
        question, scene = normalized_foundation_features(2)
        with torch.inference_mode():
            output = model(
                self.waveform,
                self.tokens.input_ids,
                self.tokens.attention_mask,
                question_clap=question,
                scene_clap=scene,
            )
        self.assertTrue(
            torch.allclose(
                output.composition.semantic_condition,
                question,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertIsNone(output.composition.foundation_semantic_candidate_weight)
        self.assertEqual(
            int(torch.count_nonzero(model.composer.semantic_head[-1].weight)), 0
        )
        with self.assertRaisesRegex(ValueError, "requires foundation_feature_mode"):
            tiny_config(semantic_mixing=QUESTION_RESIDUAL_SEMANTIC_MIXING)

    def test_none_refuses_silently_ignored_foundation_inputs(self) -> None:
        model = QCESModel(tiny_config())
        question, scene = normalized_foundation_features(2)
        with self.assertRaisesRegex(ValueError, "mode='none'"):
            model(
                self.waveform,
                self.tokens.input_ids,
                self.tokens.attention_mask,
                question_clap=question,
                scene_clap=scene,
            )

    def test_union_features_are_required_normalized_and_differentiable(self) -> None:
        model = QCESModel(tiny_config(foundation=True)).train()
        question, scene = normalized_foundation_features(2)
        question.requires_grad_(True)
        scene.requires_grad_(True)

        with self.assertRaisesRegex(ValueError, "requires offline"):
            model(
                self.waveform,
                self.tokens.input_ids,
                self.tokens.attention_mask,
            )
        with self.assertRaisesRegex(ValueError, "question_clap must have shape"):
            model(
                self.waveform,
                self.tokens.input_ids,
                self.tokens.attention_mask,
                question_clap=question[:, :-1],
                scene_clap=scene,
            )
        with self.assertRaisesRegex(
            ValueError, "scene_clap frame must be L2 normalized"
        ):
            model(
                self.waveform,
                self.tokens.input_ids,
                self.tokens.attention_mask,
                question_clap=question,
                scene_clap=scene * 2.0,
            )

        # Cache storage is FP16; the composer must cast before its FP32 Linear
        # even on CPU, where mixed-dtype Linear would otherwise fail.
        half_output = model(
            self.waveform,
            self.tokens.input_ids,
            self.tokens.attention_mask,
            question_clap=question,
            scene_clap=scene.detach().half(),
        )
        self.assertEqual(
            tuple(half_output.composition.semantic_condition.shape), (2, 512)
        )
        question_api_output = model.forward_questions(
            self.waveform,
            ["What follows the bell?", "Which sound starts first?"],
            question_clap=question,
            scene_clap=scene.detach().half(),
        )
        self.assertEqual(
            tuple(question_api_output.composition.semantic_condition.shape),
            (2, 512),
        )

        output = model(
            self.waveform,
            self.tokens.input_ids,
            self.tokens.attention_mask,
            question_clap=question,
            scene_clap=scene,
        )
        condition = output.composition.semantic_condition
        self.assertIsNotNone(output.composition.foundation_semantic_candidate_weight)
        assert output.composition.foundation_semantic_candidate_weight is not None
        self.assertAlmostEqual(
            float(output.composition.foundation_semantic_candidate_weight),
            float(torch.tensor(-2.0).sigmoid()),
            places=7,
        )
        self.assertEqual(tuple(condition.shape), (2, 512))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(condition, dim=-1),
                torch.ones(2),
                atol=1e-5,
                rtol=1e-5,
            )
        )
        target = torch.linspace(-1.0, 1.0, 512)[None]
        objective = (
            condition * target
        ).sum() + output.composition.role_logits.square().mean()
        objective.backward()
        self.assertGreater(
            float(
                model.composer.foundation_question_projection.weight.grad.abs().sum()
            ),
            0.0,
        )
        self.assertGreater(
            float(model.composer.foundation_scene_projection.weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(model.composer.foundation_semantic_residual_scale_logit.grad.abs()),
            0.0,
        )
        self.assertGreater(float(question.grad.abs().sum()), 0.0)
        self.assertGreater(float(scene.grad.abs().sum()), 0.0)

    def test_convex_mixer_starts_with_unattenuated_candidate_gradient(self) -> None:
        legacy = RoleAwarePromptComposer(tiny_config(foundation=True))
        reachable = RoleAwarePromptComposer(
            tiny_config(
                foundation=True,
                semantic_mixing=CONVEX_SEMANTIC_INTERPOLATION,
            )
        )
        self.assertAlmostEqual(
            float(legacy.foundation_semantic_residual_scale_logit.sigmoid()),
            float(torch.tensor(-2.0).sigmoid()),
            places=7,
        )
        self.assertAlmostEqual(
            float(reachable.foundation_semantic_residual_scale_logit.sigmoid()),
            0.5,
            places=7,
        )

    def test_deterministic_interpolation_and_dual_acoustic_bases(self) -> None:
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            composer = RoleAwarePromptComposer(
                tiny_config(foundation=True, dual=True)
            ).train()
            question, scene = normalized_foundation_features(2)
            output = composer(
                self.waveform,
                self.tokens.input_ids,
                self.tokens.attention_mask,
                question_clap=question,
                scene_clap=scene,
            )
            self.assertIsNotNone(output.anchor_semantic_condition)
            self.assertIsNotNone(output.answer_semantic_condition)
            assert output.anchor_semantic_condition is not None
            assert output.answer_semantic_condition is not None
            self.assertEqual(tuple(output.anchor_semantic_condition.shape), (2, 512))
            self.assertEqual(tuple(output.answer_semantic_condition.shape), (2, 512))
            loss = (
                output.anchor_semantic_condition[:, 0].sum()
                + output.answer_semantic_condition[:, 1].sum()
            )
            loss.backward()
            self.assertIsNotNone(composer.foundation_scene_projection.weight.grad)

            # The semantic base API accepts only acoustic frames and learned
            # role weights; there is no question/answer/label input path.
            weights = torch.rand(2, 19)
            base = role_pool_scene_clap(scene, weights)
            self.assertEqual(tuple(base.shape), (2, 512))
            self.assertTrue(
                torch.allclose(
                    torch.linalg.vector_norm(base, dim=-1),
                    torch.ones(2),
                    atol=1e-5,
                )
            )
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_audiosep_checkpoint_roundtrip_keeps_foundation_parameters(self) -> None:
        config = tiny_config(foundation=True)
        first_adapter = AudioSepConditionedAdapter(
            _FakeAudioSep(), condition_dim=512, freeze_separator=True
        )
        first = QCESModel(config, separator=first_adapter)
        payload = first.checkpoint_payload(backend="audiosep")
        foundation_keys = {
            key for key in payload["composer_state_dict"] if "foundation_" in key
        }
        self.assertTrue(foundation_keys)

        second_adapter = AudioSepConditionedAdapter(
            _FakeAudioSep(), condition_dim=512, freeze_separator=True
        )
        second = QCESModel(config, separator=second_adapter)
        second.composer.load_state_dict(payload["composer_state_dict"], strict=True)
        for key in foundation_keys:
            self.assertTrue(
                torch.equal(
                    first.composer.state_dict()[key],
                    second.composer.state_dict()[key],
                )
            )


class FoundationCacheTest(unittest.TestCase):
    def _build_cache(self, root: Path):  # type: ignore[no-untyped-def]
        manifest = root / "train.jsonl"
        manifest.write_text('{"fixture": true}\n', encoding="utf-8")
        mixture = root / "mixture.wav"
        mixture.write_bytes(b"canonical-waveform-fixture")
        checkpoint = root / "audiosep.bin"
        checkpoint.write_bytes(b"audiosep-checkpoint-fixture")
        source = root / "audiosep_source"
        source.mkdir()
        (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
        records = [
            SimpleNamespace(
                sample_id="sample_a",
                scene_id="scene_shared",
                schema_version="fixture_v1",
                mixture_path="mixture.wav",
            ),
            SimpleNamespace(
                sample_id="sample_b",
                scene_id="scene_shared",
                schema_version="fixture_v1",
                mixture_path="mixture.wav",
            ),
        ]
        checkpoint_identity = file_identity(checkpoint)
        source_identity = source_tree_identity(source)
        question_a = torch.zeros(512)
        question_a[1] = 1.0
        question_b = torch.zeros(512)
        question_b[2] = 1.0
        scene_feature = F.normalize(torch.randn(32, 512), dim=-1).half().contiguous()
        questions = question_payload({"sample_a": question_a, "sample_b": question_b})
        scenes = scene_payload({"scene_shared": scene_feature})
        mixture_identity = file_identity(mixture)
        mixture_rows = [
            {
                "scene_id": "scene_shared",
                "manifest_path": "mixture.wav",
                "sha256": mixture_identity["sha256"],
                "size_bytes": mixture_identity["size_bytes"],
                "sample_rate": 32_000,
                "num_samples": 320_000,
                "duration_seconds": 10.0,
                "feature_shape": [32, 512],
                "feature_dtype": "float16",
            }
        ]
        receipt = {
            "format": RECEIPT_FORMAT,
            "purpose": "frozen_controller_inputs_without_oracle_labels",
            "manifest": file_identity(manifest),
            "schema_versions": ["fixture_v1"],
            "audiosep_checkpoint": checkpoint_identity,
            "audiosep_source_tree": source_identity,
            "query_encoder_state": {
                "extracted_tensor_count": 505,
                "loaded_tensor_count": 504,
                "effective_missing_keys": [],
                "effective_unexpected_keys": [],
                "ignored_nonpersistent_checkpoint_keys": [
                    "text.embeddings.position_ids"
                ],
                "fusion_enabled": False,
            },
            "canonical_audio": {
                "sample_rate": 32_000,
                "num_samples": 320_000,
                "duration_seconds": 10.0,
                "num_channels": 1,
                "clap_sample_rate": 48_000,
                "clap_num_samples": 480_000,
            },
            "features": {
                "feature_space": FEATURE_SPACE,
                "question_shape": [512],
                "question_dtype": "float32",
                "raw_audio_shape": [1024, 1024],
                "effective_audio_shape": [32, 512],
                "effective_audio_dtype": "float16",
                "raw_to_effective_repeat_ratio": 32,
                "audio_normalization": "per_frame_l2_before_fp16_storage",
            },
            "counts": {
                "sample_ids": 2,
                "scene_ids": 1,
                "unique_full_question_texts": 2,
                "physical_mixture_encodes": 1,
            },
            "sample_ids": ["sample_a", "sample_b"],
            "scene_ids": ["scene_shared"],
            "mixtures": mixture_rows,
            "mixtures_aggregate_sha256": _aggregate_foundation_mixture_receipt(
                mixture_rows
            ),
            "execution": {
                "device": "cpu",
                "text_batch_size": 2,
                "determinism": {
                    "seed": 2026,
                    "torch_deterministic_algorithms": True,
                    "cublas_workspace_config": None,
                    "cudnn_benchmark": False,
                    "cudnn_deterministic": True,
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                },
                "software": {
                    "python": "fixture",
                    "numpy": "fixture",
                    "soundfile": "fixture",
                    "torch": "fixture",
                    "torchaudio": "fixture",
                    "transformers": "fixture",
                },
            },
            "privacy_contract": {
                "question_text_stored": False,
                "event_labels_stored": False,
                "answer_labels_stored": False,
                "evidence_annotations_stored": False,
                "allowed_keys": "sample_id_and_scene_id_only",
            },
        }
        cache_dir = root / "cache"
        write_outputs_atomically(cache_dir, questions, scenes, receipt)
        return SimpleNamespace(
            manifest=manifest,
            mixture=mixture,
            checkpoint=checkpoint,
            source=source,
            records=records,
            checkpoint_identity=checkpoint_identity,
            source_identity=source_identity,
            cache_dir=cache_dir,
            question_a=question_a,
            question_b=question_b,
            scene_feature=scene_feature,
        )

    def test_strict_load_and_shuffled_sample_to_shared_scene_injection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._build_cache(Path(temporary))
            cache = load_foundation_feature_cache(
                fixture.cache_dir,
                fixture.manifest,
                fixture.records,
                "training",
                audiosep_checkpoint_identity=fixture.checkpoint_identity,
                audiosep_source_identity=fixture.source_identity,
            )
            batch = {"sample_ids": ["sample_b", "sample_a"]}
            add_foundation_features(batch, cache, torch.device("cpu"))

        self.assertEqual(set(batch), {"sample_ids", "question_clap", "scene_clap"})
        self.assertTrue(torch.equal(batch["question_clap"][0], fixture.question_b))
        self.assertTrue(torch.equal(batch["question_clap"][1], fixture.question_a))
        self.assertTrue(torch.equal(batch["scene_clap"][0], fixture.scene_feature))
        self.assertTrue(torch.equal(batch["scene_clap"][0], batch["scene_clap"][1]))
        self.assertFalse(
            {"answer", "event_labels", "semantic_target", "oracle_prompt"} & set(batch)
        )
        self.assertFalse(cache.identity["contains_oracle_or_label_inputs"])

    def test_tampered_inputs_and_artifact_fail(self) -> None:
        mutators = {
            "waveform": lambda fixture: fixture.mixture.write_bytes(b"changed"),
            "manifest": lambda fixture: fixture.manifest.write_text(
                '{"fixture": false}\n', encoding="utf-8"
            ),
            "checkpoint": lambda fixture: fixture.checkpoint.write_bytes(b"changed"),
            "source": lambda fixture: (fixture.source / "model.py").write_text(
                "VALUE = 2\n", encoding="utf-8"
            ),
            "artifact": lambda fixture: (
                fixture.cache_dir / "question_features.pt"
            ).write_bytes(b"changed"),
        }
        for name, mutate in mutators.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = self._build_cache(Path(temporary))
                mutate(fixture)
                checkpoint_identity = file_identity(fixture.checkpoint)
                source_identity = source_tree_identity(fixture.source)
                with self.assertRaises(SystemExit):
                    load_foundation_feature_cache(
                        fixture.cache_dir,
                        fixture.manifest,
                        fixture.records,
                        "training",
                        audiosep_checkpoint_identity=checkpoint_identity,
                        audiosep_source_identity=source_identity,
                    )


class FoundationTrainingContractTest(unittest.TestCase):
    def test_training_args_default_none_and_full_scene_requirement(self) -> None:
        legacy = parse_args(["--manifest", "train.jsonl", "--output-dir", "out"])
        self.assertEqual(legacy.foundation_feature_mode, NO_FOUNDATION_FEATURES)
        validate_foundation_training_args(legacy)

        unreachable = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--foundation-semantic-mixing-mode",
                CONVEX_SEMANTIC_INTERPOLATION,
            ]
        )
        with self.assertRaisesRegex(SystemExit, "requires.*audiosep_clap"):
            validate_foundation_training_args(unreachable)

        enabled = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--backend",
                "audiosep",
                "--foundation-feature-mode",
                "audiosep_clap",
                "--foundation-feature-cache",
                "cache",
                "--foundation-semantic-mixing-mode",
                CONVEX_SEMANTIC_INTERPOLATION,
                "--crop-seconds",
                "10",
            ]
        )
        validate_foundation_training_args(enabled)
        enabled.crop_seconds = 4.0
        with self.assertRaisesRegex(
            SystemExit, "prevent temporal feature misalignment"
        ):
            validate_foundation_training_args(enabled)

    def test_batch_forward_refuses_missing_cached_features(self) -> None:
        model = QCESModel(tiny_config(foundation=True))
        tokens = StableHashTokenizer(128, 16).batch_encode(["What follows?"])
        batch = {
            "mixture": torch.randn(1, 256),
            "question_ids": tokens.input_ids,
            "question_mask": tokens.attention_mask,
        }
        with self.assertRaisesRegex(ValueError, "refuses a batch missing"):
            forward_training_batch(model, batch)

    def test_infer_preflight_requires_online_assets(self) -> None:
        payload = {
            "format": "qces_v1",
            "config": {"foundation_feature_mode": AUDIOSEP_CLAP_FOUNDATION_FEATURES},
        }
        with self.assertRaisesRegex(SystemExit, "requires --audiosep-root"):
            require_online_foundation_support(payload)
        self.assertTrue(
            require_online_foundation_support(
                payload,
                audiosep_root=Path("audiosep"),
                audiosep_config=Path("audiosep.yaml"),
                audiosep_checkpoint=Path("audiosep.bin"),
            )
        )
        require_online_foundation_support(
            {"format": "qces_v1", "config": {"foundation_feature_mode": "none"}}
        )

    def test_infer_missing_assets_rejects_before_model_construction(self) -> None:
        payload = {
            "format": "qces_v1",
            "config": {"foundation_feature_mode": AUDIOSEP_CLAP_FOUNDATION_FEATURES},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint.pt"
            torch.save(payload, checkpoint)
            argv = [
                "infer_qces.py",
                "--checkpoint",
                str(checkpoint),
                "--audio",
                str(root / "not_read.wav"),
                "--question",
                "not encoded",
                "--output-dir",
                str(root / "not_created"),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch(
                    "mixi_understanding.scripts.infer_qces.load_qces_checkpoint"
                ) as loader,
                self.assertRaisesRegex(SystemExit, "requires --audiosep-root"),
            ):
                infer_main()
            loader.assert_not_called()
            self.assertFalse((root / "not_created").exists())

    def test_online_extractor_uses_only_question_and_canonical_waveform(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "audiosep"
            source.mkdir()
            (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
            checkpoint = root / "audiosep.bin"
            checkpoint.write_bytes(b"frozen-audiosep")
            question_feature = torch.zeros(512)
            question_feature[3] = 1.0
            scene_feature = (
                F.normalize(torch.randn(32, 512), dim=-1).half().contiguous()
            )
            waveform = np.zeros(320_000, dtype=np.float32)
            with (
                patch(
                    "mixi_understanding.scripts.infer_qces.configure_clap_determinism",
                    return_value={"seed": 7},
                ),
                patch(
                    "mixi_understanding.scripts.infer_qces.load_frozen_encoder",
                    return_value=(nn.Identity(), {"loaded_tensor_count": 504}),
                ),
                patch(
                    "mixi_understanding.scripts.infer_qces.encode_full_questions",
                    return_value={"online_sample": question_feature},
                ) as text_encoder,
                patch(
                    "mixi_understanding.scripts.infer_qces.encode_canonical_waveform",
                    return_value=scene_feature,
                ) as audio_encoder,
            ):
                question, scene, provenance = extract_online_foundation_features(
                    question="What follows the bell?",
                    waveform=waveform,
                    sample_rate=32_000,
                    audiosep_root=source,
                    audiosep_checkpoint=checkpoint,
                    device=torch.device("cpu"),
                    seed=7,
                )
        self.assertEqual(tuple(question.shape), (1, 512))
        self.assertEqual(tuple(scene.shape), (1, 32, 512))
        self.assertEqual(provenance["source"], "online_frozen_audiosep_clap")
        self.assertFalse(provenance["contains_event_answer_or_oracle_inputs"])
        text_sample = text_encoder.call_args.args[1][0]
        self.assertEqual(text_sample.question, "What follows the bell?")
        self.assertEqual(text_sample.sample_id, "online_sample")
        self.assertEqual(audio_encoder.call_args.args[1].shape, (320_000,))
        self.assertFalse(
            {"answer", "event_labels", "evidence", "oracle_prompt"} & set(provenance)
        )


if __name__ == "__main__":
    unittest.main()
