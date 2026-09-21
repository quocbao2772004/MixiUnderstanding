"""Focused tests for held-out QCES training helpers."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mixi_understanding.qces.composer import RoleAwarePromptComposer
from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    CONVEX_SEMANTIC_INTERPOLATION,
    DUAL_ROLE_SEMANTIC_MODE,
    LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    QUESTION_RESIDUAL_SEMANTIC_MIXING,
    QCESConfig,
    UNION_SINGLE_SEMANTIC_MODE,
)
from mixi_understanding.qces.losses import LossWeights, QCESLoss
from mixi_understanding.qces.model import QCESModel
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.train_qces import (
    AMP_FP16_PRECISION,
    CORE_LOSS_WEIGHT_ARGS,
    DIAGNOSTIC_BEST_DIRECTIONS,
    FP32_PRECISION,
    SELECTION_DIRECTIONS,
    SEMANTIC_PROMPT_SOURCE,
    SEMANTIC_TARGET_SCOPE,
    atomic_torch_save,
    backward_and_optimizer_step,
    check_output_directory,
    composer_initialization_receipt,
    counterfactual_activation,
    create_gradient_scaler,
    evaluate_epoch,
    file_identity,
    forward_training_batch_with_precision,
    initialize_audiosep_composer,
    load_semantic_target_cache,
    loaded_project_source_identity,
    no_evidence_class_balance,
    parse_args,
    prepare_output_directory,
    retained_epoch_checkpoint_extra,
    retained_epoch_checkpoint_path,
    selection_improved,
    semantic_adapter_provenance,
    set_base_composer_trainable,
    sha256_file,
    split_overlap_audit,
    validate_precision_mode,
    validate_refiner_training_args,
    validate_semantic_adapter_training_args,
)


class NoEvidenceBalanceTest(unittest.TestCase):
    def test_training_split_inverse_frequency_weight_is_explicit(self) -> None:
        records = [
            SimpleNamespace(no_evidence=value) for value in (False, False, False, True)
        ]
        audit = no_evidence_class_balance(records)
        self.assertEqual(audit["answerable_records_↑"], 3)
        self.assertEqual(audit["no_evidence_records_↑"], 1)
        self.assertTrue(audit["both_classes_present"])
        self.assertEqual(audit["positive_weight_descriptive"], 3.0)

    def test_single_class_legacy_manifest_is_neutral_and_declared(self) -> None:
        audit = no_evidence_class_balance(
            [SimpleNamespace(no_evidence=False) for _ in range(3)]
        )
        self.assertFalse(audit["both_classes_present"])
        self.assertEqual(audit["positive_weight_descriptive"], 1.0)
        with self.assertRaisesRegex(ValueError, "no records"):
            no_evidence_class_balance([])


class SelectionTest(unittest.TestCase):
    def test_core_loss_weights_are_cli_visible_with_compatible_defaults(self) -> None:
        default = LossWeights()
        parsed = parse_args(["--manifest", "train.jsonl", "--output-dir", "out"])
        for field_name, argument_name in CORE_LOSS_WEIGHT_ARGS.items():
            self.assertEqual(
                getattr(parsed, argument_name), getattr(default, field_name), field_name
            )
        ablated = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--temporal-binary-cross-entropy-weight",
                "0",
                "--temporal-dice-weight",
                "0",
                "--temporal-role-dice-weight",
                "0",
                "--no-evidence-weight",
                "0",
                "--minimality-weight",
                "0",
            ]
        )
        self.assertEqual(ablated.temporal_binary_cross_entropy_weight, 0.0)
        self.assertEqual(ablated.temporal_dice_weight, 0.0)
        self.assertEqual(ablated.temporal_role_dice_weight, 0.0)
        self.assertEqual(ablated.no_evidence_weight, 0.0)
        self.assertEqual(ablated.minimality_weight, 0.0)

    def test_direction_and_minimum_delta(self) -> None:
        self.assertTrue(selection_improved(1.0, None, "minimize", 0.0))
        self.assertTrue(selection_improved(0.8, 1.0, "minimize", 0.1))
        self.assertFalse(selection_improved(0.95, 1.0, "minimize", 0.1))
        self.assertTrue(selection_improved(1.2, 1.0, "maximize", 0.1))
        self.assertFalse(selection_improved(1.05, 1.0, "maximize", 0.1))
        with self.assertRaises(ValueError):
            selection_improved(math.nan, 1.0, "minimize", 0.0)

    def test_validation_options_are_backwards_compatible(self) -> None:
        legacy = parse_args(["--manifest", "train.jsonl", "--output-dir", "out"])
        self.assertIsNone(legacy.val_manifest)
        self.assertEqual(legacy.selection_metric, "total")
        self.assertEqual(legacy.early_stopping_patience, 0)
        self.assertTrue(legacy.deterministic)
        self.assertFalse(legacy.separator_aware_refiner)
        self.assertEqual(legacy.separator_aware_refiner_mode, "full")
        self.assertIsNone(legacy.init_audiosep_qces_checkpoint)
        self.assertEqual(legacy.freeze_base_composer_steps, 0)
        self.assertFalse(legacy.freeze_semantic_adapter)
        self.assertFalse(legacy.save_every_epoch)
        self.assertEqual(
            legacy.foundation_semantic_mixing_mode,
            LEGACY_BOUNDED_SEMANTIC_RESIDUAL,
        )
        self.assertEqual(legacy.precision, FP32_PRECISION)
        self.assertEqual(legacy.surface_semantic_invariance_weight, 0.0)
        self.assertEqual(legacy.surface_role_invariance_weight, 0.0)
        self.assertEqual(legacy.surface_no_evidence_invariance_weight, 0.0)
        self.assertEqual(legacy.surface_evidence_invariance_weight, 0.0)
        self.assertEqual(legacy.family_temporal_delta_weight, 0.0)
        self.assertEqual(legacy.family_evidence_delta_weight, 0.0)
        self.assertEqual(legacy.family_no_evidence_transition_weight, 0.0)
        self.assertEqual(legacy.question_temporal_delta_weight, 0.0)
        self.assertEqual(legacy.question_evidence_delta_weight, 0.0)
        self.assertEqual(legacy.counterfactual_transition_margin, 0.25)
        self.assertFalse(legacy.force_counterfactual_batching)
        schedule_matched = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--force-counterfactual-batching",
            ]
        )
        self.assertTrue(schedule_matched.force_counterfactual_batching)
        zero_weights = {
            "surface_semantic_invariance": 0.0,
            "surface_role_invariance": 0.0,
            "surface_no_evidence_invariance": 0.0,
            "surface_evidence_invariance": 0.0,
            "family_temporal_delta": 0.0,
            "family_evidence_delta": 0.0,
            "family_no_evidence_transition": 0.0,
            "question_temporal_delta": 0.0,
            "question_evidence_delta": 0.0,
        }
        matched = counterfactual_activation(zero_weights, True)
        self.assertFalse(matched["objective"])
        self.assertTrue(matched["batching"])
        self.assertTrue(matched["surface_batched"])
        self.assertTrue(matched["family_batched"])
        self.assertTrue(matched["question_batched"])
        self.assertTrue(matched["forced_schedule_matched_control"])
        active = counterfactual_activation(
            {**zero_weights, "family_temporal_delta": 0.1}, False
        )
        self.assertTrue(active["objective"])
        self.assertTrue(active["family_batched"])
        self.assertFalse(active["surface_batched"])
        self.assertFalse(active["forced_schedule_matched_control"])
        nondeterministic = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--no-deterministic",
            ]
        )
        self.assertFalse(nondeterministic.deterministic)
        reachable = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--foundation-semantic-mixing-mode",
                CONVEX_SEMANTIC_INTERPOLATION,
            ]
        )
        self.assertEqual(
            reachable.foundation_semantic_mixing_mode,
            CONVEX_SEMANTIC_INTERPOLATION,
        )
        question_residual = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--foundation-semantic-mixing-mode",
                QUESTION_RESIDUAL_SEMANTIC_MIXING,
            ]
        )
        self.assertEqual(
            question_residual.foundation_semantic_mixing_mode,
            QUESTION_RESIDUAL_SEMANTIC_MIXING,
        )
        held_out = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--val-manifest",
                "val.jsonl",
                "--output-dir",
                "out",
                "--selection-metric",
                "evidence_si_sdr",
                "--early-stopping-patience",
                "4",
            ]
        )
        self.assertEqual(held_out.val_manifest, Path("val.jsonl"))
        self.assertEqual(held_out.selection_metric, "evidence_si_sdr")
        self.assertEqual(held_out.early_stopping_patience, 4)
        retained = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--save-every-epoch",
            ]
        )
        self.assertTrue(retained.save_every_epoch)
        amp = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--precision",
                AMP_FP16_PRECISION,
            ]
        )
        self.assertEqual(amp.precision, AMP_FP16_PRECISION)

    def test_direct_question_ablation_cli_contract_is_fail_closed(self) -> None:
        arguments = [
            "--manifest",
            "train.jsonl",
            "--output-dir",
            "out",
            "--backend",
            "audiosep",
            "--foundation-feature-mode",
            AUDIOSEP_CLAP_FOUNDATION_FEATURES,
            "--foundation-feature-cache",
            "foundation_train",
            "--foundation-semantic-mixing-mode",
            QUESTION_RESIDUAL_SEMANTIC_MIXING,
            "--crop-seconds",
            "10",
            "--freeze-semantic-adapter",
        ]
        valid = parse_args(arguments)
        validate_refiner_training_args(valid)
        validate_semantic_adapter_training_args(valid)

        invalid_mode = parse_args(arguments)
        invalid_mode.foundation_semantic_mixing_mode = CONVEX_SEMANTIC_INTERPOLATION
        with self.assertRaisesRegex(SystemExit, "question_residual"):
            validate_semantic_adapter_training_args(invalid_mode)

        invalid_warm_start = parse_args(arguments)
        invalid_warm_start.init_audiosep_qces_checkpoint = Path("learned.pt")
        with self.assertRaisesRegex(SystemExit, "forbids a warm-start"):
            validate_semantic_adapter_training_args(invalid_warm_start)

        invalid_supervision = parse_args(arguments)
        invalid_supervision.semantic_weight = 0.1
        with self.assertRaisesRegex(SystemExit, "semantic targets"):
            validate_semantic_adapter_training_args(invalid_supervision)

        dual = parse_args(
            [
                *arguments,
                "--semantic-separation-mode",
                DUAL_ROLE_SEMANTIC_MODE,
                "--temporal-role-mode",
                OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
                "--role-semantic-targets",
                "role_targets.pt",
                "--role-semantic-weight",
                "0",
                "--same-semantic-weight",
                "0.1",
            ]
        )
        validate_refiner_training_args(dual)
        validate_semantic_adapter_training_args(dual)

    def test_refiner_selection_metrics_have_explicit_directions(self) -> None:
        self.assertEqual(SELECTION_DIRECTIONS["answerable_temporal_iou"], "maximize")
        self.assertEqual(SELECTION_DIRECTIONS["no_evidence_retained_ratio"], "minimize")
        self.assertEqual(
            DIAGNOSTIC_BEST_DIRECTIONS,
            {
                "evidence_si_sdr": "maximize",
                "evidence_sd_sdr": "maximize",
                "answerable_temporal_iou": "maximize",
                "no_evidence_retained_ratio": "minimize",
                "weakest_role_waveform": "minimize",
            },
        )


class MixedPrecisionTrainingTest(unittest.TestCase):
    def test_amp_fp16_is_rejected_off_cuda(self) -> None:
        validate_precision_mode(FP32_PRECISION, torch.device("cpu"))
        with self.assertRaisesRegex(SystemExit, "requires a CUDA device"):
            validate_precision_mode(AMP_FP16_PRECISION, torch.device("cpu"))
        validate_precision_mode(AMP_FP16_PRECISION, torch.device("cuda"))

    def test_fp32_forward_never_enters_autocast(self) -> None:
        sentinel = object()

        class Model:
            config = SimpleNamespace(foundation_feature_mode="none")

            def __call__(self, mixture, question_ids, question_mask):
                self.arguments = (mixture, question_ids, question_mask)
                return sentinel

        model = Model()
        batch = {
            "mixture": torch.zeros(1, 8),
            "question_ids": torch.ones(1, 2, dtype=torch.long),
            "question_mask": torch.ones(1, 2, dtype=torch.bool),
        }
        with patch(
            "mixi_understanding.scripts.train_qces.torch.autocast",
            side_effect=AssertionError("fp32 must not use autocast"),
        ):
            result = forward_training_batch_with_precision(
                model, batch, torch.device("cpu")
            )
        self.assertIs(result, sentinel)
        self.assertEqual(model.arguments, tuple(batch.values()))

    def test_default_optimizer_update_is_bit_exact_with_historical_flow(self) -> None:
        initial = torch.tensor([0.25, -0.5, 1.25], dtype=torch.float32)
        reference = torch.nn.Parameter(initial.clone())
        candidate = torch.nn.Parameter(initial.clone())
        reference_optimizer = torch.optim.SGD([reference], lr=0.125)
        candidate_optimizer = torch.optim.SGD([candidate], lr=0.125)
        coefficients = torch.tensor([1.5, -2.0, 0.75])

        reference_loss = (reference * coefficients).square().sum()
        reference_loss.backward()
        torch.nn.utils.clip_grad_norm_([reference], 5.0)
        reference_optimizer.step()

        candidate_loss = (candidate * coefficients).square().sum()
        backward_and_optimizer_step(
            candidate_loss,
            candidate_optimizer,
            [candidate],
        )
        self.assertTrue(torch.equal(candidate, reference))
        self.assertTrue(torch.equal(candidate.grad, reference.grad))
        self.assertIsNone(create_gradient_scaler(FP32_PRECISION))

    def test_amp_scaler_unscales_before_gradient_clipping(self) -> None:
        calls: list[object] = []
        optimizer = object()
        parameters = [object()]
        total = object()

        class ScaledLoss:
            def backward(self) -> None:
                calls.append("backward")

        class Scaler:
            def scale(self, received) -> ScaledLoss:
                self.asserted_total = received
                calls.append("scale")
                return ScaledLoss()

            def unscale_(self, received_optimizer) -> None:
                calls.append(("unscale", received_optimizer))

            def step(self, received_optimizer) -> None:
                calls.append(("step", received_optimizer))

            def update(self) -> None:
                calls.append("update")

        scaler = Scaler()

        def record_clip(received_parameters, max_norm) -> None:
            calls.append(("clip", received_parameters, max_norm))

        with patch(
            "mixi_understanding.scripts.train_qces.clip_grad_norm_",
            side_effect=record_clip,
        ):
            backward_and_optimizer_step(
                total,  # type: ignore[arg-type]
                optimizer,  # type: ignore[arg-type]
                parameters,  # type: ignore[arg-type]
                precision=AMP_FP16_PRECISION,
                scaler=scaler,
                max_grad_norm=3.0,
            )

        self.assertIs(scaler.asserted_total, total)
        self.assertEqual(
            calls,
            [
                "scale",
                "backward",
                ("unscale", optimizer),
                ("clip", parameters, 3.0),
                ("step", optimizer),
                "update",
            ],
        )


class ComposerInitializationReceiptTest(unittest.TestCase):
    @staticmethod
    def _composer(mode: str, seed: int) -> RoleAwarePromptComposer:
        torch.manual_seed(seed)
        return RoleAwarePromptComposer(
            QCESConfig(
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
                temporal_role_mode=OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
                semantic_separation_mode=mode,
            )
        )

    def test_common_aggregate_is_comparison_safe_across_semantic_modes(self) -> None:
        union = self._composer(UNION_SINGLE_SEMANTIC_MODE, 2_026)
        dual = self._composer(DUAL_ROLE_SEMANTIC_MODE, 2_026)
        union_before = {
            key: value.detach().clone() for key, value in union.state_dict().items()
        }

        union_receipt = composer_initialization_receipt(union, seed=2_026)
        dual_receipt = composer_initialization_receipt(dual, seed=2_026)

        self.assertEqual(union_receipt["seed"], 2_026)
        self.assertEqual(dual_receipt["seed"], 2_026)
        self.assertEqual(
            union_receipt["aggregates"]["common_parameters_sha256"],
            dual_receipt["aggregates"]["common_parameters_sha256"],
        )
        self.assertEqual(
            union_receipt["aggregates"]["common_state_tensors_sha256"],
            dual_receipt["aggregates"]["common_state_tensors_sha256"],
        )
        self.assertNotEqual(
            union_receipt["aggregates"]["all_state_tensors_sha256"],
            dual_receipt["aggregates"]["all_state_tensors_sha256"],
        )
        self.assertEqual(union_receipt["candidate_only_state_tensor_keys"], [])
        self.assertIsNone(
            union_receipt["aggregates"]["candidate_only_state_tensors_sha256"]
        )
        self.assertTrue(dual_receipt["candidate_only_state_tensor_keys"])
        self.assertTrue(
            all(
                key.startswith("same_semantic_head.")
                for key in dual_receipt["candidate_only_state_tensor_keys"]
            )
        )
        self.assertEqual(set(union_receipt["tensor_receipts"]), set(union.state_dict()))
        self.assertTrue(
            all(
                torch.equal(value, union.state_dict()[key])
                for key, value in union_before.items()
            )
        )

    def test_common_aggregate_changes_with_seed(self) -> None:
        first = composer_initialization_receipt(
            self._composer(UNION_SINGLE_SEMANTIC_MODE, 2_026),
            seed=2_026,
        )
        second = composer_initialization_receipt(
            self._composer(UNION_SINGLE_SEMANTIC_MODE, 2_027),
            seed=2_027,
        )
        self.assertNotEqual(
            first["aggregates"]["common_parameters_sha256"],
            second["aggregates"]["common_parameters_sha256"],
        )


class AudioSepInitializationTest(unittest.TestCase):
    @staticmethod
    def _config(enabled: bool, mode: str = "full") -> QCESConfig:
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
            separator_aware_refiner=enabled,
            separator_aware_refiner_mode=mode,
        )

    def _write_initial_checkpoint(
        self,
        path: Path,
        config: QCESConfig,
        audiosep_config: Path,
        audiosep_checkpoint: Path,
        *,
        backend: str = "audiosep",
        declared_config_hash: str | None = None,
    ) -> QCESModel:
        source = QCESModel(config)
        with torch.no_grad():
            source.composer.role_head.bias.fill_(0.25)
        payload = {
            "format": "qces_v1",
            "backend": backend,
            "config": config.to_dict(),
            "composer_state_dict": source.composer.state_dict(),
            "extra": {
                "audiosep": {
                    "config": {
                        **file_identity(audiosep_config),
                        "sha256": (
                            declared_config_hash
                            if declared_config_hash is not None
                            else file_identity(audiosep_config)["sha256"]
                        ),
                    },
                    "checkpoint": file_identity(audiosep_checkpoint),
                }
            },
        }
        torch.save(payload, path)
        return source

    def test_composer_warm_start_allows_only_refiner_enable_transition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audiosep_config = root / "audiosep.yaml"
            audiosep_config.write_text("model: test\n", encoding="utf-8")
            audiosep_checkpoint = root / "audiosep.pt"
            audiosep_checkpoint.write_bytes(b"separator")
            initial = root / "initial.pt"
            source = self._write_initial_checkpoint(
                initial,
                self._config(False),
                audiosep_config,
                audiosep_checkpoint,
            )
            legacy_payload = torch.load(initial, weights_only=True)
            legacy_payload["config"].pop("separator_aware_refiner_mode")
            torch.save(legacy_payload, initial)
            target = QCESModel(self._config(True))
            identity = initialize_audiosep_composer(
                target,
                initial,
                target.config,
                audiosep_config_identity=file_identity(audiosep_config),
                audiosep_checkpoint_identity=file_identity(audiosep_checkpoint),
            )
            self.assertEqual(identity["mode"], "composer_warm_start_only")
            self.assertFalse(identity["refiner_state_restored"])
            for name, expected in source.composer.state_dict().items():
                self.assertTrue(
                    torch.equal(expected, target.composer.state_dict()[name])
                )

            relative_target = QCESModel(self._config(True, "relative_energy"))
            relative_identity = initialize_audiosep_composer(
                relative_target,
                initial,
                relative_target.config,
                audiosep_config_identity=file_identity(audiosep_config),
                audiosep_checkpoint_identity=file_identity(audiosep_checkpoint),
            )
            self.assertEqual(
                relative_identity["current_separator_aware_refiner_mode"],
                "relative_energy",
            )
            self.assertEqual(
                relative_identity["source_separator_aware_refiner_mode"],
                "full",
            )
            for name, expected in source.composer.state_dict().items():
                self.assertTrue(
                    torch.equal(
                        expected,
                        relative_target.composer.state_dict()[name],
                    )
                )

    def test_composer_warm_start_rejects_backend_config_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audiosep_config = root / "audiosep.yaml"
            audiosep_config.write_text("model: test\n", encoding="utf-8")
            audiosep_checkpoint = root / "audiosep.pt"
            audiosep_checkpoint.write_bytes(b"separator")
            target = QCESModel(self._config(True))

            cases = []
            backend_path = root / "wrong_backend.pt"
            self._write_initial_checkpoint(
                backend_path,
                self._config(False),
                audiosep_config,
                audiosep_checkpoint,
                backend="mask",
            )
            cases.append(backend_path)
            mismatch_path = root / "wrong_config.pt"
            mismatched = self._config(False).to_dict()
            mismatched["audio_dim"] = 64
            mismatched["attention_heads"] = 4
            payload = torch.load(backend_path, weights_only=True)
            payload["backend"] = "audiosep"
            payload["config"] = mismatched
            torch.save(payload, mismatch_path)
            cases.append(mismatch_path)
            provenance_path = root / "wrong_provenance.pt"
            self._write_initial_checkpoint(
                provenance_path,
                self._config(False),
                audiosep_config,
                audiosep_checkpoint,
                declared_config_hash="0" * 64,
            )
            cases.append(provenance_path)

            for path in cases:
                with self.subTest(path=path.name), self.assertRaises(SystemExit):
                    initialize_audiosep_composer(
                        target,
                        path,
                        target.config,
                        audiosep_config_identity=file_identity(audiosep_config),
                        audiosep_checkpoint_identity=file_identity(audiosep_checkpoint),
                    )

    def test_composer_warm_start_rejects_source_with_learned_refiner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audiosep_config = root / "audiosep.yaml"
            audiosep_config.write_text("model: test\n", encoding="utf-8")
            audiosep_checkpoint = root / "audiosep.pt"
            audiosep_checkpoint.write_bytes(b"separator")
            initial = root / "initial.pt"
            self._write_initial_checkpoint(
                initial,
                self._config(True),
                audiosep_config,
                audiosep_checkpoint,
            )
            with self.assertRaisesRegex(SystemExit, "silently discard"):
                initialize_audiosep_composer(
                    QCESModel(self._config(True)),
                    initial,
                    self._config(True),
                    audiosep_config_identity=file_identity(audiosep_config),
                    audiosep_checkpoint_identity=file_identity(audiosep_checkpoint),
                )

    def test_freezing_base_composer_requires_a_warm_start(self) -> None:
        args = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--backend",
                "audiosep",
                "--separator-aware-refiner",
                "--freeze-base-composer-steps",
                "1",
            ]
        )
        with self.assertRaisesRegex(SystemExit, "randomly initialized"):
            validate_refiner_training_args(args)
        args.init_audiosep_qces_checkpoint = Path("base.pt")
        validate_refiner_training_args(args)

    def test_relative_energy_cli_requires_enabled_refiner(self) -> None:
        args = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--output-dir",
                "out",
                "--backend",
                "audiosep",
                "--separator-aware-refiner-mode",
                "relative_energy",
            ]
        )
        with self.assertRaisesRegex(SystemExit, "requires"):
            validate_refiner_training_args(args)
        args.separator_aware_refiner = True
        validate_refiner_training_args(args)
        args.temporal_role_mode = OVERLAP_AWARE_TEMPORAL_ROLE_MODE
        validate_refiner_training_args(args)

    def test_composer_freeze_toggle_leaves_other_modules_trainable(self) -> None:
        model = QCESModel(self._config(True))
        set_base_composer_trainable(model, False)
        self.assertFalse(model.composer.training)
        self.assertTrue(
            all(
                not parameter.requires_grad for parameter in model.composer.parameters()
            )
        )
        self.assertTrue(
            any(parameter.requires_grad for parameter in model.separator.parameters())
        )
        set_base_composer_trainable(model, True)
        self.assertTrue(model.composer.training)
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.composer.parameters())
        )

    def test_direct_question_adapter_stays_exactly_zero_and_frozen(self) -> None:
        config = QCESConfig(
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
            foundation_feature_mode=AUDIOSEP_CLAP_FOUNDATION_FEATURES,
            foundation_semantic_mixing_mode=QUESTION_RESIDUAL_SEMANTIC_MIXING,
        )
        model = QCESModel(config)
        set_base_composer_trainable(model, True, freeze_semantic_adapter=True)
        receipt = semantic_adapter_provenance(model, freeze_semantic_adapter=True)
        self.assertEqual(receipt["ablation"], "direct_full_question_clap")
        self.assertEqual(receipt["trainable_parameter_count"], 0)
        self.assertEqual(receipt["final_layer_nonzero_parameter_count"], 0)
        self.assertTrue(receipt["exact_identity_condition_at_initialization"])
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.composer.semantic_head.parameters()
            )
        )

        tokenizer = StableHashTokenizer(
            vocab_size=config.vocab_size,
            max_length=config.max_question_tokens,
        )
        tokens = tokenizer.batch_encode(["What happens after the bell?"])
        waveform = torch.randn(1, 1_024)
        question_clap = torch.randn(1, 512)
        question_clap = question_clap / question_clap.norm(dim=-1, keepdim=True)
        scene_clap = torch.randn(1, 32, 512)
        scene_clap = scene_clap / scene_clap.norm(dim=-1, keepdim=True)
        optimizer = torch.optim.SGD(
            [
                parameter
                for parameter in model.composer.parameters()
                if parameter.requires_grad
            ],
            lr=0.01,
        )
        composition = model.composer(
            waveform,
            tokens.input_ids,
            tokens.attention_mask,
            question_clap=question_clap,
            scene_clap=scene_clap,
        )
        self.assertTrue(
            torch.allclose(
                composition.semantic_condition,
                question_clap,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        loss = composition.role_logits.square().mean()
        loss = loss + composition.no_evidence_logit.square().mean()
        loss.backward()
        optimizer.step()
        with torch.inference_mode():
            updated = model.composer(
                waveform,
                tokens.input_ids,
                tokens.attention_mask,
                question_clap=question_clap,
                scene_clap=scene_clap,
            )
        self.assertTrue(
            torch.allclose(
                updated.semantic_condition,
                question_clap,
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            any(
                parameter.requires_grad
                for name, parameter in model.composer.named_parameters()
                if not name.startswith("semantic_head.")
            )
        )

        model.train()
        set_base_composer_trainable(model, True, freeze_semantic_adapter=True)
        self.assertFalse(model.composer.semantic_head.training)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.composer.semantic_head.parameters()
            )
        )


class SemanticCacheTest(unittest.TestCase):
    def _write_cache(
        self,
        cache: Path,
        manifest: Path,
        checkpoint: Path,
        **overrides: object,
    ) -> None:
        payload = {
            "format": "qces_audiosep_semantic_targets_v1",
            "schema_version": "qces_v4",
            "target_scope": SEMANTIC_TARGET_SCOPE,
            "prompt_source": SEMANTIC_PROMPT_SOURCE,
            "manifest_sha256": sha256_file(manifest),
            "audiosep_checkpoint": str(checkpoint.resolve()),
            "prompts": {"train_0": "bell and croak"},
            "targets": {"train_0": torch.ones(3)},
        }
        payload.update(overrides)
        torch.save(payload, cache)

    def _load(
        self, cache: Path, manifest: Path, checkpoint: Path
    ) -> tuple[object, object]:
        return load_semantic_target_cache(
            cache,
            manifest,
            ["train_0"],
            "training",
            expected_dim=3,
            expected_schema_versions=["qces_v4"],
            audiosep_checkpoint_identity=file_identity(checkpoint),
        )

    def test_cache_is_bound_to_manifest_and_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            manifest.write_text('{"id":"train_0"}\n', encoding="utf-8")
            checkpoint = root / "audiosep.pt"
            checkpoint.write_bytes(b"checkpoint-a")
            cache = root / "targets.pt"
            self._write_cache(cache, manifest, checkpoint)
            targets, identity = self._load(cache, manifest, checkpoint)
            self.assertEqual(set(targets), {"train_0"})
            self.assertEqual(identity["manifest_sha256"], sha256_file(manifest))
            self.assertEqual(identity["target_dim"], 3)
            self.assertEqual(
                identity["audiosep_checkpoint_binding"]["method"],
                "legacy_path_rehashed_at_load",
            )

            self._write_cache(
                cache,
                manifest,
                checkpoint,
                targets={
                    "train_0": torch.ones(3),
                    "extra": torch.ones(3),
                },
            )
            with self.assertRaises(SystemExit):
                self._load(cache, manifest, checkpoint)

            self._write_cache(cache, manifest, checkpoint)
            manifest.write_text('{"id":"changed"}\n', encoding="utf-8")
            with self.assertRaises(SystemExit):
                self._load(cache, manifest, checkpoint)

    def test_cache_rejects_bad_metadata_tensor_and_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "train.jsonl"
            manifest.write_text('{"id":"train_0"}\n', encoding="utf-8")
            checkpoint = root / "audiosep.pt"
            checkpoint.write_bytes(b"checkpoint-a")
            other_checkpoint = root / "other.pt"
            other_checkpoint.write_bytes(b"checkpoint-b")
            cache = root / "targets.pt"

            invalid_overrides = (
                {"schema_version": "qces_v3"},
                {"target_scope": "inference_input"},
                {"prompt_source": "question_answer"},
                {"targets": {"train_0": torch.ones(1, 3)}},
                {"targets": {"train_0": torch.ones(4)}},
                {"targets": {"train_0": torch.ones(3, dtype=torch.int64)}},
                {"targets": {"train_0": torch.tensor([1.0, math.nan, 2.0])}},
                {"prompts": {"wrong": "bell and croak"}},
                {"audiosep_checkpoint": str(other_checkpoint.resolve())},
            )
            for override in invalid_overrides:
                with self.subTest(override=tuple(override)):
                    self._write_cache(cache, manifest, checkpoint, **override)
                    with self.assertRaises(SystemExit):
                        self._load(cache, manifest, checkpoint)

            self._write_cache(
                cache,
                manifest,
                checkpoint,
                audiosep_checkpoint_sha256=sha256_file(other_checkpoint),
            )
            with self.assertRaises(SystemExit):
                self._load(cache, manifest, checkpoint)

            self._write_cache(
                cache,
                manifest,
                checkpoint,
                audiosep_checkpoint_sha256=sha256_file(checkpoint),
            )
            _, identity = self._load(cache, manifest, checkpoint)
            self.assertEqual(
                identity["audiosep_checkpoint_binding"]["method"],
                "recorded_sha256",
            )


class SplitLeakageTest(unittest.TestCase):
    @staticmethod
    def _dataset(sample: str, scene: str, source: str) -> SimpleNamespace:
        event = SimpleNamespace(source_id=source)
        record = SimpleNamespace(scene_id=scene, events=(event,))
        example = SimpleNamespace(sample_id=sample)
        return SimpleNamespace(records=[record], examples=[example])

    def test_source_overlap_fails_closed(self) -> None:
        train = self._dataset("train_0", "scene_train", "source_shared")
        validation = self._dataset("val_0", "scene_val", "source_shared")
        with self.assertRaises(SystemExit):
            split_overlap_audit(train, validation)

    def test_disjoint_splits_report_zero_overlap(self) -> None:
        train = self._dataset("train_0", "scene_train", "source_train")
        validation = self._dataset("val_0", "scene_val", "source_val")
        audit = split_overlap_audit(train, validation)
        self.assertEqual(audit["sample_id_overlap_count"], 0)
        self.assertEqual(audit["scene_id_overlap_count"], 0)
        self.assertEqual(audit["source_id_overlap_count"], 0)


class OutputDirectoryTest(unittest.TestCase):
    def test_overwrite_removes_stale_files_only_after_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            stale = output / "stale.json"
            stale.write_text("old", encoding="utf-8")
            stale_epoch_dir = output / "epoch_checkpoints"
            stale_epoch_dir.mkdir()
            stale_epoch = stale_epoch_dir / "epoch_0001_checkpoint.pt"
            stale_epoch.write_bytes(b"old checkpoint")
            resolved, nonempty = check_output_directory(output, overwrite=True)
            self.assertTrue(nonempty)
            self.assertTrue(stale.exists())
            self.assertTrue(stale_epoch.exists())
            prepare_output_directory(resolved, overwrite=True, was_nonempty=nonempty)
            self.assertTrue(resolved.is_dir())
            self.assertEqual(list(resolved.iterdir()), [])

    def test_retention_preserves_non_extreme_feasible_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            (output / "epoch_checkpoints").mkdir()
            base = {"run_provenance": {"seed": 2026}}
            snapshots = {
                1: {
                    "epoch": 1,
                    "global_step": 10,
                    "validation": {
                        "evidence_si_sdr": 1.0,
                        "no_evidence_retained_ratio": 0.10,
                    },
                },
                2: {
                    "epoch": 2,
                    "global_step": 20,
                    "validation": {
                        "evidence_si_sdr": 2.0,
                        "no_evidence_retained_ratio": 0.30,
                    },
                },
                3: {
                    "epoch": 3,
                    "global_step": 30,
                    "validation": {
                        "evidence_si_sdr": 0.5,
                        "no_evidence_retained_ratio": 0.05,
                    },
                },
            }
            for epoch, metrics in snapshots.items():
                retained_path = retained_epoch_checkpoint_path(output, epoch)
                extra = retained_epoch_checkpoint_extra(
                    base,
                    epoch=epoch,
                    global_step=epoch * 10,
                    stop_reason=(
                        "epochs_completed" if epoch == 3 else "training_in_progress"
                    ),
                    best_epoch=2,
                    best_global_step=20,
                    best_value=2.0,
                    metrics_at_checkpoint=metrics,
                    checkpoint_path=retained_path,
                )
                atomic_torch_save(
                    {"model_state": {"epoch": torch.tensor(epoch)}, "extra": extra},
                    retained_path,
                    overwrite=False,
                )

            retained_payloads = [
                torch.load(
                    retained_epoch_checkpoint_path(output, epoch),
                    map_location="cpu",
                    weights_only=True,
                )
                for epoch in snapshots
            ]
            # Epoch 1 is neither the maximum SI-SDR (epoch 2) nor minimum
            # no-evidence retention (epoch 3), but wins the frozen joint rule:
            # retention <= .12, then maximize SI-SDR.
            feasible = [
                payload
                for payload in retained_payloads
                if payload["extra"]["epoch_retention"][
                    "validation_metrics_at_checkpoint"
                ]["no_evidence_retained_ratio"]
                <= 0.12
            ]
            selected = max(
                feasible,
                key=lambda payload: payload["extra"]["epoch_retention"][
                    "validation_metrics_at_checkpoint"
                ]["evidence_si_sdr"],
            )
            self.assertEqual(selected["extra"]["checkpoint_epoch"], 1)
            self.assertEqual(
                selected["extra"]["epoch_retention"]["metrics_at_checkpoint"],
                snapshots[1],
            )

            epoch_one = retained_epoch_checkpoint_path(output, 1)
            original = epoch_one.read_bytes()
            with self.assertRaises(FileExistsError):
                atomic_torch_save(
                    {"extra": {"checkpoint_epoch": 99}},
                    epoch_one,
                    overwrite=False,
                )
            self.assertEqual(epoch_one.read_bytes(), original)
            self.assertFalse((epoch_one.parent / f".{epoch_one.name}.tmp").exists())

            epoch_four = retained_epoch_checkpoint_path(output, 4)
            stale_temporary = epoch_four.parent / f".{epoch_four.name}.tmp"
            stale_temporary.write_bytes(b"foreign incomplete writer")
            with self.assertRaises(FileExistsError):
                atomic_torch_save(
                    {"extra": {"checkpoint_epoch": 4}},
                    epoch_four,
                    overwrite=False,
                )
            self.assertFalse(epoch_four.exists())
            self.assertEqual(stale_temporary.read_bytes(), b"foreign incomplete writer")

    def test_race_and_missing_overwrite_fail_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            output.mkdir()
            resolved, nonempty = check_output_directory(output, overwrite=True)
            late = output / "late.txt"
            late.write_text("keep", encoding="utf-8")
            with self.assertRaises(SystemExit):
                prepare_output_directory(
                    resolved, overwrite=True, was_nonempty=nonempty
                )
            self.assertTrue(late.exists())
            with self.assertRaises(SystemExit):
                check_output_directory(output, overwrite=False)


class SourceProvenanceTest(unittest.TestCase):
    def test_loaded_qces_sources_are_hashed(self) -> None:
        identity = loaded_project_source_identity(Path(__file__).resolve().parents[3])
        paths = {item["relative_path"] for item in identity["files"]}
        self.assertIn("code/mixi_understanding/scripts/train_qces.py", paths)
        self.assertIn("code/mixi_understanding/qces/model.py", paths)
        self.assertEqual(len(identity["aggregate_sha256"]), 64)


class DeterministicEvaluationTest(unittest.TestCase):
    def test_evaluation_is_repeatable_and_does_not_build_gradients(self) -> None:
        torch.manual_seed(9)
        config = QCESConfig(
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
            dropout=0.3,
        )
        model = QCESModel(config).train()
        criterion = QCESLoss(fft_sizes=(64,))
        mixture = torch.randn(2, 512) * 0.02
        anchor_mask = torch.zeros_like(mixture)
        answer_mask = torch.zeros_like(mixture)
        anchor_mask[0, 40:120] = 1.0
        answer_mask[0, 240:340] = 1.0
        union = (anchor_mask + answer_mask).clamp_max(1.0)
        tokens = StableHashTokenizer(128, 16).batch_encode(
            ["What follows the horn?", "What follows an absent bell?"]
        )
        evidence = mixture * union
        batch = {
            "sample_ids": ["val_0", "val_1"],
            "mixture": mixture,
            "evidence": evidence,
            "residual": mixture - evidence,
            "anchor_stem": mixture * anchor_mask,
            "answer_stem": mixture * answer_mask,
            "anchor_mask": anchor_mask,
            "answer_mask": answer_mask,
            "no_evidence": torch.tensor([0.0, 1.0]),
            "question_ids": tokens.input_ids,
            "question_mask": tokens.attention_mask,
        }
        singleton_batches = []
        for index in range(2):
            singleton_batches.append(
                {
                    key: (
                        value[index : index + 1]
                        if isinstance(value, torch.Tensor)
                        else [value[index]]
                    )
                    for key, value in batch.items()
                }
            )
        first = evaluate_epoch(
            model, criterion, singleton_batches, torch.device("cpu"), None
        )
        second = evaluate_epoch(
            model, criterion, singleton_batches, torch.device("cpu"), None
        )
        self.assertEqual(first, second)
        self.assertFalse(model.training)
        self.assertIn("total", first)
        self.assertIn("answerable_temporal_iou", first)
        self.assertIn("no_evidence_retained_ratio", first)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_evaluation_rejects_ambiguous_batch_reductions(self) -> None:
        config = QCESConfig(
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
        )
        model = QCESModel(config)
        criterion = QCESLoss(fft_sizes=(64,))
        mixture = torch.zeros(2, 256)
        tokens = StableHashTokenizer(128, 16).batch_encode(["a", "b"])
        batch = {
            "sample_ids": ["a", "b"],
            "mixture": mixture,
            "evidence": mixture.clone(),
            "residual": mixture.clone(),
            "anchor_stem": mixture.clone(),
            "answer_stem": mixture.clone(),
            "anchor_mask": mixture.clone(),
            "answer_mask": mixture.clone(),
            "no_evidence": torch.ones(2),
            "question_ids": tokens.input_ids,
            "question_mask": tokens.attention_mask,
        }
        with self.assertRaises(ValueError):
            evaluate_epoch(model, criterion, [batch], torch.device("cpu"), None)


if __name__ == "__main__":
    unittest.main()
