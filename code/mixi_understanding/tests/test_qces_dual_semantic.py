"""Regression tests for learned role-factorized semantic separation."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from mixi_understanding.qces.composer import (
    PromptComposition,
    RoleAwarePromptComposer,
    temporal_evidence_probability,
)
from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    DUAL_ROLE_SEMANTIC_MODE,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    QCESConfig,
    UNION_SINGLE_SEMANTIC_MODE,
)
from mixi_understanding.qces.losses import LossWeights, QCESLoss
from mixi_understanding.qces.metrics import qces_metrics
from mixi_understanding.qces.model import QCESModel, load_qces_checkpoint
from mixi_understanding.qces.separators import AudioSepConditionedAdapter
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.train_qces import (
    ROLE_SEMANTIC_CACHE_FORMAT,
    ROLE_SEMANTIC_PROMPT_SOURCE,
    ROLE_SEMANTIC_TARGET_SCOPE,
    SEMANTIC_CACHE_FORMAT,
    SEMANTIC_PROMPT_SOURCE,
    SEMANTIC_TARGET_SCOPE,
    V5_UNION_SEMANTIC_CACHE_FORMAT,
    V5_UNION_SEMANTIC_PROMPT_SOURCE,
    V5_UNION_SEMANTIC_TARGET_SCOPE,
    add_semantic_targets,
    file_identity,
    load_role_semantic_target_cache,
    load_semantic_target_cache,
    parse_args,
    sha256_file,
    validate_refiner_training_args,
)
from mixi_understanding.scripts.cache_audiosep_semantic_targets import (
    encode_prompts_batched,
    parse_args as parse_cache_args,
    role_training_prompts,
    v5_union_training_prompts,
    write_cache_atomically,
)


def tiny_config(*, dual: bool = False) -> QCESConfig:
    return QCESConfig(
        sample_rate=8_000,
        n_fft=64,
        hop_length=16,
        win_length=64,
        vocab_size=128,
        max_question_tokens=16,
        question_dim=32,
        audio_dim=32,
        condition_dim=16,
        attention_heads=4,
        question_layers=1,
        separator_channels=8,
        separator_layers=2,
        dropout=0.0,
        temporal_role_mode=(
            OVERLAP_AWARE_TEMPORAL_ROLE_MODE if dual else "exclusive_softmax"
        ),
        semantic_separation_mode=(
            DUAL_ROLE_SEMANTIC_MODE if dual else UNION_SINGLE_SEMANTIC_MODE
        ),
    )


class RecordingConditionedSeparator(nn.Module):
    """Small differentiable AudioSep-compatible stand-in."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, batch):
        self.calls += 1
        mixture = batch["mixture"]
        condition = batch["condition"]
        self.batch_sizes.append(int(mixture.size(0)))
        gain = 0.75 + 0.25 * condition[:, :1, None]
        return {"waveform": mixture * gain * self.scale}


class _RecordingTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batches: list[list[str]] = []

    def get_query_embed(self, *, modality: str, text: list[str]) -> torch.Tensor:
        self.assert_modality = modality
        self.batches.append(list(text))
        result = torch.zeros(len(text), 512)
        for index, value in enumerate(text):
            result[index, sum(value.encode("utf-8")) % 512] = 1.0
        return result


class BoundedSemanticEncoderTest(unittest.TestCase):
    def test_cache_publication_is_atomic_and_overwrite_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "semantic.pt"
            write_cache_atomically(
                output,
                {"format": "first", "targets": {"a": torch.ones(1)}},
                overwrite=False,
            )
            with self.assertRaises(FileExistsError):
                write_cache_atomically(
                    output,
                    {"format": "second"},
                    overwrite=False,
                )
            write_cache_atomically(
                output,
                {"format": "second", "targets": {"b": torch.zeros(1)}},
                overwrite=True,
            )
            payload = torch.load(output, map_location="cpu", weights_only=True)
        self.assertEqual(payload["format"], "second")
        self.assertEqual(set(payload["targets"]), {"b"})

    def test_unique_prompts_are_sorted_and_memory_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "audiosep"
            repository.mkdir()
            (repository / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
            checkpoint = root / "audiosep.bin"
            checkpoint.write_bytes(b"checkpoint")
            encoder = _RecordingTextEncoder()
            with (
                patch(
                    "mixi_understanding.scripts.cache_audiosep_semantic_targets."
                    "configure_determinism",
                    return_value={"seed": 9},
                ),
                patch(
                    "mixi_understanding.scripts.cache_audiosep_semantic_targets."
                    "load_frozen_encoder",
                    return_value=(encoder, {"loaded_tensor_count": 504}),
                ),
            ):
                encoded, provenance = encode_prompts_batched(
                    repository,
                    checkpoint,
                    ["echo", "bell", "dog", "bell", "cat"],
                    batch_size=2,
                    seed=9,
                )
        self.assertEqual(encoder.batches, [["bell", "cat"], ["dog", "echo"]])
        self.assertEqual(set(encoded), {"bell", "cat", "dog", "echo"})
        self.assertTrue(provenance["bounded_memory"])
        self.assertEqual(provenance["unique_prompt_count"], 4)
        self.assertEqual(provenance["batch_size"], 2)


class DualSemanticModelTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        self.mixture = torch.randn(2, 512) * 0.05
        self.tokens = StableHashTokenizer(128, 16).batch_encode(
            ["What follows the bell?", "Which event begins first?"]
        )

    def test_legacy_checkpoint_and_forward_remain_exact(self) -> None:
        config = tiny_config()
        model = QCESModel(config).eval()
        self.assertFalse(
            any("same_semantic_head" in key for key in model.composer.state_dict())
        )
        with torch.inference_mode():
            expected = model(
                self.mixture, self.tokens.input_ids, self.tokens.attention_mask
            )
        payload = model.checkpoint_payload(backend="mask")
        payload["config"].pop("semantic_separation_mode")
        restored = load_qces_checkpoint(payload).eval()
        with torch.inference_mode():
            actual = restored(
                self.mixture, self.tokens.input_ids, self.tokens.attention_mask
            )
        self.assertEqual(
            restored.config.semantic_separation_mode, UNION_SINGLE_SEMANTIC_MODE
        )
        self.assertTrue(torch.equal(expected.evidence, actual.evidence))
        self.assertTrue(
            torch.equal(
                expected.composition.semantic_condition,
                actual.composition.semantic_condition,
            )
        )

    def test_union_and_dual_common_initialization_is_bit_identical(self) -> None:
        common = dict(
            sample_rate=8_000,
            n_fft=64,
            hop_length=16,
            win_length=64,
            vocab_size=128,
            max_question_tokens=16,
            question_dim=32,
            audio_dim=32,
            condition_dim=512,
            attention_heads=4,
            question_layers=1,
            separator_channels=8,
            separator_layers=2,
            dropout=0.0,
            temporal_role_mode=OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
            foundation_feature_mode=AUDIOSEP_CLAP_FOUNDATION_FEATURES,
        )
        torch.manual_seed(2_026)
        union = RoleAwarePromptComposer(
            QCESConfig(
                **common,
                semantic_separation_mode=UNION_SINGLE_SEMANTIC_MODE,
            )
        )
        torch.manual_seed(2_026)
        dual = RoleAwarePromptComposer(
            QCESConfig(
                **common,
                semantic_separation_mode=DUAL_ROLE_SEMANTIC_MODE,
            )
        )

        union_state = union.state_dict()
        dual_state = dual.state_dict()
        self.assertEqual(
            set(union_state),
            {key for key in dual_state if not key.startswith("same_semantic_head.")},
        )
        self.assertTrue(
            all(
                torch.equal(value, dual_state[key])
                for key, value in union_state.items()
            )
        )

    def test_legacy_audiosep_path_remains_one_effective_evaluation(self) -> None:
        backbone = RecordingConditionedSeparator()
        adapter = AudioSepConditionedAdapter(
            backbone, condition_dim=2, freeze_separator=True
        )
        mixture = torch.ones(1, 8)
        role_logits = torch.tensor([[[0.0, 2.0, -2.0], [0.0, 2.0, -2.0]]])
        probability = temporal_evidence_probability(role_logits, "exclusive_softmax")
        composition = PromptComposition(
            semantic_condition=torch.tensor([[1.0, 0.0]]),
            role_logits=role_logits,
            evidence_probability=probability,
            no_evidence_logit=torch.zeros(1),
            frame_features=torch.zeros(1, 2, 2),
            frame_hop_samples=4,
        )
        output = adapter(mixture, composition)
        expected = probability[:, :1].expand_as(mixture)
        self.assertTrue(torch.allclose(output.evidence, expected, atol=1e-6))
        self.assertEqual(backbone.calls, 1)
        self.assertEqual(backbone.batch_sizes, [1])
        self.assertEqual(output.semantic_separation_mode, UNION_SINGLE_SEMANTIC_MODE)
        self.assertEqual(output.physical_separator_forwards_per_batch, 1)
        self.assertEqual(output.effective_separator_evaluations_per_record, 1)

    def test_dual_role_has_two_effective_evaluations_and_gradients(self) -> None:
        backbone = RecordingConditionedSeparator()
        adapter = AudioSepConditionedAdapter(
            backbone, condition_dim=16, freeze_separator=True
        )
        model = QCESModel(tiny_config(dual=True), separator=adapter)
        model.train()
        self.assertFalse(backbone.training)
        self.assertTrue(all(not p.requires_grad for p in backbone.parameters()))

        output = model(self.mixture, self.tokens.input_ids, self.tokens.attention_mask)
        self.assertEqual(backbone.calls, 1)
        self.assertEqual(backbone.batch_sizes, [4])
        self.assertEqual(output.separation.physical_separator_forwards_per_batch, 1)
        self.assertEqual(
            output.separation.effective_separator_evaluations_per_record, 2
        )
        self.assertLess(float(output.separation.mixture_error.max()), 1e-8)
        self.assertTrue(torch.equal(output.residual, self.mixture - output.evidence))
        self.assertIsNotNone(output.composition.anchor_semantic_condition)
        self.assertIsNotNone(output.composition.answer_semantic_condition)
        payload = model.checkpoint_payload(backend="audiosep")
        self.assertEqual(
            payload["config"]["semantic_separation_mode"],
            DUAL_ROLE_SEMANTIC_MODE,
        )
        self.assertTrue(
            any("same_semantic_head" in key for key in payload["composer_state_dict"])
        )
        self.assertNotIn("state_dict", payload)

        anchor_mask = torch.zeros_like(self.mixture)
        answer_mask = torch.zeros_like(self.mixture)
        anchor_mask[:, 40:220] = 1.0
        answer_mask[:, 160:360] = 1.0
        union = (anchor_mask + answer_mask).clamp_max(1.0)
        batch = {
            "mixture": self.mixture,
            "evidence": self.mixture * union,
            "residual": self.mixture * (1.0 - union),
            "anchor_mask": anchor_mask,
            "answer_mask": answer_mask,
            "no_evidence": torch.zeros(2),
            "anchor_semantic_target": torch.randn(2, 16),
            "answer_semantic_target": torch.randn(2, 16),
            "same_semantic_target": torch.tensor([0.0, 1.0]),
            "role_semantic_valid": torch.ones(2),
        }
        weights = LossWeights(
            anchor_semantic_alignment=1.0,
            answer_semantic_alignment=1.0,
            same_semantic_classification=1.0,
        )
        loss, components = QCESLoss(weights=weights, fft_sizes=(64,))(output, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(
            float(model.composer.semantic_head[0].weight.grad.abs().sum()), 0.0
        )
        assert model.composer.same_semantic_head is not None
        same_head_parameter = next(model.composer.same_semantic_head.parameters())
        self.assertGreater(float(same_head_parameter.grad.abs().sum()), 0.0)
        self.assertIn("same_semantic_classification", components)

        metrics = qces_metrics(output, batch)
        self.assertIn("same_semantic_brier", metrics)
        self.assertIn("same_semantic_accuracy", metrics)
        probabilities = output.composition.same_semantic_probability
        assert probabilities is not None
        self.assertAlmostEqual(
            float(metrics["same_semantic_probability_on_same"]),
            float(probabilities[1]),
            places=6,
        )
        self.assertAlmostEqual(
            float(metrics["same_semantic_probability_on_different"]),
            float(probabilities[0]),
            places=6,
        )

    def test_same_label_overlap_routes_to_one_union_gated_target(self) -> None:
        backbone = RecordingConditionedSeparator()
        adapter = AudioSepConditionedAdapter(
            backbone, condition_dim=2, freeze_separator=True
        )
        mixture = torch.ones(1, 8)
        role_logits = torch.tensor([[[-100.0, 100.0, 100.0], [-100.0, 100.0, 100.0]]])
        condition = torch.tensor([[1.0, 0.0]], requires_grad=True)
        answer_condition = torch.tensor([[1.0, 0.0]], requires_grad=True)
        same_logit = torch.tensor([100.0], requires_grad=True)
        composition = PromptComposition(
            semantic_condition=condition,
            role_logits=role_logits,
            evidence_probability=temporal_evidence_probability(
                role_logits, OVERLAP_AWARE_TEMPORAL_ROLE_MODE
            ),
            no_evidence_logit=torch.zeros(1),
            frame_features=torch.zeros(1, 2, 2),
            frame_hop_samples=4,
            temporal_role_mode=OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
            semantic_separation_mode=DUAL_ROLE_SEMANTIC_MODE,
            anchor_semantic_condition=condition,
            answer_semantic_condition=answer_condition,
            same_semantic_logit=same_logit,
        )
        output = adapter(mixture, composition)
        # The dummy gain is 1.0 for both equal conditions. A naive sum would
        # produce 2.0 on overlap; learned shared routing counts it once.
        self.assertTrue(torch.allclose(output.evidence, mixture, atol=1e-6))
        self.assertTrue(torch.equal(output.residual, mixture - output.evidence))

        # A non-saturated routing value proves gradients reach both role
        # conditions and the learned shared-target router.
        composition.same_semantic_logit = torch.tensor([0.0], requires_grad=True)
        routed = adapter(mixture, composition)
        routed.evidence.sum().backward()
        self.assertGreater(float(condition.grad.abs().sum()), 0.0)
        self.assertGreater(float(answer_condition.grad.abs().sum()), 0.0)
        self.assertGreater(float(composition.same_semantic_logit.grad.abs().sum()), 0.0)

    def test_dual_cli_fails_closed_without_supervision(self) -> None:
        base = [
            "--manifest",
            "train.jsonl",
            "--output-dir",
            "out",
            "--backend",
            "audiosep",
            "--temporal-role-mode",
            OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
            "--semantic-separation-mode",
            DUAL_ROLE_SEMANTIC_MODE,
        ]
        with self.assertRaisesRegex(SystemExit, "role-semantic-targets"):
            validate_refiner_training_args(parse_args(base))
        complete = parse_args(
            [
                *base,
                "--role-semantic-targets",
                "roles.pt",
                "--role-semantic-weight",
                "1",
                "--same-semantic-weight",
                "1",
            ]
        )
        validate_refiner_training_args(complete)

    def test_inference_api_has_no_role_label_or_routing_target_input(self) -> None:
        parameters = list(inspect.signature(QCESModel.forward).parameters)
        self.assertEqual(
            parameters,
            [
                "self",
                "mixture",
                "question_ids",
                "question_mask",
                "question_clap",
                "scene_clap",
            ],
        )
        self.assertFalse(
            {"answer", "answer_label", "event_labels", "role_targets"} & set(parameters)
        )


class RoleSemanticCacheTest(unittest.TestCase):
    def test_targets_come_from_annotated_role_event_labels(self) -> None:
        def record(sample_id: str, anchor: str, answer: str):
            events = {
                "a": SimpleNamespace(label=anchor),
                "b": SimpleNamespace(label=answer),
            }
            return SimpleNamespace(
                sample_id=sample_id,
                no_evidence=False,
                anchor_event_ids=("a",),
                answer_event_ids=("b",),
                event_by_id=events.__getitem__,
            )

        negative = SimpleNamespace(sample_id="negative", no_evidence=True)
        prompts, negatives, same = role_training_prompts(
            [
                record("same", "Mechanical bell", "Mechanical bell"),
                record("different", "Mechanical bell", "Croak"),
                negative,
            ]
        )
        self.assertEqual(
            prompts["same"], {"anchor": "mechanical bell", "answer": "mechanical bell"}
        )
        self.assertEqual(prompts["different"]["answer"], "a frog croaking")
        self.assertEqual(same, {"same": True, "different": False})
        self.assertEqual(negatives, ["negative"])

    def test_cache_is_exactly_bound_and_partitioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "audiosep.pt"
            checkpoint.write_bytes(b"frozen-backbone")
            cache = root / "roles.pt"
            checkpoint_identity = file_identity(checkpoint)
            payload = {
                "format": ROLE_SEMANTIC_CACHE_FORMAT,
                "schema_version": "qces_v5",
                "target_scope": ROLE_SEMANTIC_TARGET_SCOPE,
                "prompt_source": ROLE_SEMANTIC_PROMPT_SOURCE,
                "manifest_sha256": sha256_file(manifest),
                "audiosep_checkpoint_sha256": checkpoint_identity["sha256"],
                "audiosep_checkpoint_identity": checkpoint_identity,
                "role_prompts": {"answerable": {"anchor": "bell", "answer": "bell"}},
                "role_targets": {
                    "answerable": {
                        "anchor": torch.ones(3),
                        "answer": torch.ones(3),
                        "same_semantic": True,
                    }
                },
                "no_evidence_ids": ["negative"],
            }
            torch.save(payload, cache)
            targets, identity = load_role_semantic_target_cache(
                cache,
                manifest,
                ["answerable", "negative"],
                ["negative"],
                {"answerable": True},
                "training",
                expected_dim=3,
                expected_schema_versions=["qces_v5"],
                audiosep_checkpoint_identity=checkpoint_identity,
            )
            self.assertEqual(set(targets), {"answerable"})
            self.assertEqual(identity["same_semantic_count"], 1)
            self.assertEqual(identity["no_evidence_count"], 1)

            payload["no_evidence_ids"] = []
            torch.save(payload, cache)
            with self.assertRaisesRegex(SystemExit, "partition"):
                load_role_semantic_target_cache(
                    cache,
                    manifest,
                    ["answerable", "negative"],
                    ["negative"],
                    {"answerable": True},
                    "training",
                    expected_dim=3,
                    expected_schema_versions=["qces_v5"],
                    audiosep_checkpoint_identity=checkpoint_identity,
                )
            payload["no_evidence_ids"] = ["negative"]
            payload["role_targets"]["answerable"]["same_semantic"] = "yes"
            torch.save(payload, cache)
            with self.assertRaisesRegex(SystemExit, "boolean"):
                load_role_semantic_target_cache(
                    cache,
                    manifest,
                    ["answerable", "negative"],
                    ["negative"],
                    {"answerable": True},
                    "training",
                    expected_dim=3,
                    expected_schema_versions=["qces_v5"],
                    audiosep_checkpoint_identity=checkpoint_identity,
                )

            payload["role_targets"]["answerable"]["same_semantic"] = False
            torch.save(payload, cache)
            with self.assertRaisesRegex(SystemExit, "disagrees"):
                load_role_semantic_target_cache(
                    cache,
                    manifest,
                    ["answerable", "negative"],
                    ["negative"],
                    {"answerable": True},
                    "training",
                    expected_dim=3,
                    expected_schema_versions=["qces_v5"],
                    audiosep_checkpoint_identity=checkpoint_identity,
                )

            payload["role_targets"]["answerable"]["same_semantic"] = True
            torch.save(payload, cache)
            wrong_checkpoint = {**checkpoint_identity, "sha256": "0" * 64}
            with self.assertRaisesRegex(SystemExit, "exact AudioSep checkpoint"):
                load_role_semantic_target_cache(
                    cache,
                    manifest,
                    ["answerable", "negative"],
                    ["negative"],
                    {"answerable": True},
                    "training",
                    expected_dim=3,
                    expected_schema_versions=["qces_v5"],
                    audiosep_checkpoint_identity=wrong_checkpoint,
                )


class V5UnionSemanticCacheTest(unittest.TestCase):
    @staticmethod
    def _record(
        sample_id: str,
        anchor_label: str,
        answer_label: str,
        *,
        anchor_onset: float = 0.1,
        answer_onset: float = 0.4,
    ):
        events = {
            "anchor": SimpleNamespace(
                event_id="anchor", label=anchor_label, onset_seconds=anchor_onset
            ),
            "answer": SimpleNamespace(
                event_id="answer", label=answer_label, onset_seconds=answer_onset
            ),
        }
        return SimpleNamespace(
            sample_id=sample_id,
            no_evidence=False,
            anchor_event_ids=("anchor",),
            answer_event_ids=("answer",),
            event_by_id=events.__getitem__,
        )

    def test_prompt_deduplicates_same_label_and_preserves_timeline(self) -> None:
        negative = SimpleNamespace(sample_id="negative", no_evidence=True)
        prompts, no_evidence = v5_union_training_prompts(
            [
                self._record("same", "Mechanical bell", "Mechanical bell"),
                self._record(
                    "reverse",
                    "Mechanical bell",
                    "Croak",
                    anchor_onset=0.8,
                    answer_onset=0.2,
                ),
                negative,
            ]
        )
        self.assertEqual(prompts["same"], "mechanical bell")
        self.assertEqual(prompts["reverse"], "a frog croaking and mechanical bell")
        self.assertEqual(no_evidence, ["negative"])

    def test_cache_cli_keeps_union_default_and_accepts_explicit_v5_mode(self) -> None:
        required = [
            "--manifest",
            "v5.jsonl",
            "--audiosep-root",
            "audiosep",
            "--audiosep-checkpoint",
            "audiosep.pt",
            "--output",
            "targets.pt",
        ]
        self.assertEqual(
            parse_cache_args(required).semantic_separation_mode,
            UNION_SINGLE_SEMANTIC_MODE,
        )
        train_args = parse_args(
            [
                "--manifest",
                "v5_train.jsonl",
                "--output-dir",
                "out",
                "--backend",
                "audiosep",
                "--temporal-role-mode",
                OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
                "--semantic-separation-mode",
                UNION_SINGLE_SEMANTIC_MODE,
                "--semantic-targets",
                "v5_union_targets.pt",
                "--semantic-weight",
                "1",
            ]
        )
        validate_refiner_training_args(train_args)
        self.assertEqual(
            parse_cache_args(
                [
                    *required,
                    "--semantic-separation-mode",
                    UNION_SINGLE_SEMANTIC_MODE,
                ]
            ).semantic_separation_mode,
            UNION_SINGLE_SEMANTIC_MODE,
        )

    def test_loader_uses_exact_answerable_negative_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "val.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "audiosep.pt"
            checkpoint.write_bytes(b"frozen-audiosep")
            checkpoint_identity = file_identity(checkpoint)
            cache = root / "v5_union.pt"
            payload = {
                "format": V5_UNION_SEMANTIC_CACHE_FORMAT,
                "schema_version": "qces_v5_scene_event_derived_v1",
                "target_scope": V5_UNION_SEMANTIC_TARGET_SCOPE,
                "prompt_source": V5_UNION_SEMANTIC_PROMPT_SOURCE,
                "manifest_sha256": sha256_file(manifest),
                "audiosep_checkpoint": str(checkpoint),
                "audiosep_checkpoint_sha256": checkpoint_identity["sha256"],
                "audiosep_checkpoint_identity": checkpoint_identity,
                "prompts": {"answerable": "bell"},
                "targets": {"answerable": torch.ones(3)},
                "no_evidence_ids": ["negative"],
            }
            torch.save(payload, cache)
            targets, identity = load_semantic_target_cache(
                cache,
                manifest,
                ["answerable", "negative"],
                "validation",
                expected_no_evidence_ids=["negative"],
                expected_dim=3,
                expected_schema_versions=["qces_v5_scene_event_derived_v1"],
                audiosep_checkpoint_identity=checkpoint_identity,
            )
            self.assertEqual(set(targets), {"answerable"})
            self.assertEqual(identity["answerable_target_count"], 1)
            self.assertEqual(identity["no_evidence_count"], 1)
            self.assertFalse(identity["no_evidence_has_semantic_target"])

            payload["no_evidence_ids"] = []
            torch.save(payload, cache)
            with self.assertRaisesRegex(SystemExit, "partition"):
                load_semantic_target_cache(
                    cache,
                    manifest,
                    ["answerable", "negative"],
                    "validation",
                    expected_no_evidence_ids=["negative"],
                    expected_dim=3,
                    expected_schema_versions=["qces_v5_scene_event_derived_v1"],
                    audiosep_checkpoint_identity=checkpoint_identity,
                )
            payload["no_evidence_ids"] = ["negative"]
            payload["schema_version"] = "qces_v5"
            torch.save(payload, cache)
            with self.assertRaisesRegex(SystemExit, "schema mismatch"):
                load_semantic_target_cache(
                    cache,
                    manifest,
                    ["answerable", "negative"],
                    "validation",
                    expected_no_evidence_ids=["negative"],
                    expected_dim=3,
                    expected_schema_versions=["qces_v5_scene_event_derived_v1"],
                    audiosep_checkpoint_identity=checkpoint_identity,
                )
            payload["schema_version"] = "qces_v5_scene_event_derived_v1"
            payload["manifest_sha256"] = "0" * 64
            torch.save(payload, cache)
            with self.assertRaisesRegex(SystemExit, "does not match manifest"):
                load_semantic_target_cache(
                    cache,
                    manifest,
                    ["answerable", "negative"],
                    "validation",
                    expected_no_evidence_ids=["negative"],
                    expected_dim=3,
                    expected_schema_versions=["qces_v5_scene_event_derived_v1"],
                    audiosep_checkpoint_identity=checkpoint_identity,
                )
            payload["manifest_sha256"] = sha256_file(manifest)
            torch.save(payload, cache)
            wrong_checkpoint = {**checkpoint_identity, "sha256": "0" * 64}
            with self.assertRaisesRegex(SystemExit, "exact AudioSep checkpoint"):
                load_semantic_target_cache(
                    cache,
                    manifest,
                    ["answerable", "negative"],
                    "validation",
                    expected_no_evidence_ids=["negative"],
                    expected_dim=3,
                    expected_schema_versions=["qces_v5_scene_event_derived_v1"],
                    audiosep_checkpoint_identity=wrong_checkpoint,
                )

    def test_legacy_v1_loader_still_requires_and_returns_every_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "legacy.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "audiosep.pt"
            checkpoint.write_bytes(b"legacy-audiosep")
            identity = file_identity(checkpoint)
            cache = root / "legacy.pt"
            payload = {
                "format": SEMANTIC_CACHE_FORMAT,
                "schema_version": "qces_v4",
                "target_scope": SEMANTIC_TARGET_SCOPE,
                "prompt_source": SEMANTIC_PROMPT_SOURCE,
                "manifest_sha256": sha256_file(manifest),
                "audiosep_checkpoint": str(checkpoint),
                "audiosep_checkpoint_sha256": identity["sha256"],
                "prompts": {
                    "answerable": "bell and croak",
                    "negative": "absent bell",
                },
                "targets": {
                    "answerable": torch.ones(3),
                    "negative": torch.zeros(3),
                },
            }
            torch.save(payload, cache)
            targets, loaded_identity = load_semantic_target_cache(
                cache,
                manifest,
                ["answerable", "negative"],
                "training",
                expected_no_evidence_ids=["negative"],
                expected_dim=3,
                expected_schema_versions=["qces_v4"],
                audiosep_checkpoint_identity=identity,
            )
            self.assertEqual(set(targets), {"answerable", "negative"})
            self.assertEqual(loaded_identity["format"], SEMANTIC_CACHE_FORMAT)
            self.assertEqual(loaded_identity["target_count"], 2)

    def test_missing_negative_target_is_masked_from_semantic_loss(self) -> None:
        torch.manual_seed(41)
        config = tiny_config()
        model = QCESModel(config).eval()
        mixture = torch.randn(2, 256) * 0.05
        tokens = StableHashTokenizer(128, 16).batch_encode(
            ["What follows the bell?", "What follows an absent bell?"]
        )
        output = model(mixture, tokens.input_ids, tokens.attention_mask)
        batch = {
            "sample_ids": ["answerable", "negative"],
            "mixture": mixture,
            "evidence": output.evidence.detach(),
            "residual": output.residual.detach(),
            "anchor_mask": torch.zeros_like(mixture),
            "answer_mask": torch.zeros_like(mixture),
            "no_evidence": torch.tensor([0.0, 1.0]),
        }
        add_semantic_targets(
            batch,
            {"answerable": output.composition.semantic_condition[0].detach()},
            torch.device("cpu"),
        )
        self.assertTrue(
            torch.equal(batch["semantic_target_valid"], torch.tensor([1.0, 0.0]))
        )
        self.assertTrue(torch.equal(batch["semantic_target"][1], torch.zeros(16)))
        _, components = QCESLoss(
            weights=LossWeights(semantic_alignment=1.0), fft_sizes=(64,)
        )(output, batch)
        self.assertLess(float(components["semantic_alignment"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
