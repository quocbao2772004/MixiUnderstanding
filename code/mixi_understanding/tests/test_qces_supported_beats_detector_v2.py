from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from mixi_understanding.qces.fixed_grid_detector import (
    NUM_FRAMES,
    _match_events,
    balancing_tensors,
    boundary_probabilities_from_activity_logits,
    build_fixed_grid_boundary_targets,
    build_fixed_grid_targets,
    evaluate_fixed_grid_predictions,
    masked_balanced_bce,
    masked_probability_bce,
)
from mixi_understanding.scripts.train_qces_supported_beats_detector_v2 import (
    GATE_FORMAT,
    DetectorScene,
    _sha256,
    configure_stage,
    evaluate_thresholds,
    load_manifest,
    main as detector_main,
    set_train_mode_without_frozen_backbone_dropout,
    split_calibration_selection_rows,
    validate_native_initialization,
    validate_gate_receipt,
)


def _event(label: str, label_id: int, onset: float, offset: float) -> dict[str, object]:
    return {
        "label": label,
        "label_id": label_id,
        "onset_seconds": onset,
        "offset_seconds": offset,
    }


class FixedGridDetectorTest(unittest.TestCase):
    def test_boundary_targets_merge_touching_same_class_activity(self) -> None:
        rows = [
            {
                "duration_seconds": 2.0,
                "events": [
                    _event("A", 0, 0.4, 0.8),
                    _event("A", 0, 0.8, 1.2),
                ],
            }
        ]
        onset, offset, valid = build_fixed_grid_boundary_targets(
            rows, num_labels=2, dilation_frames=0
        )
        self.assertEqual(int(valid.sum()), 50)
        self.assertEqual(onset[0, :, 0].nonzero().flatten().tolist(), [10])
        self.assertEqual(offset[0, :, 0].nonzero().flatten().tolist(), [30])

    def test_boundary_targets_keep_separated_repeated_occurrences(self) -> None:
        rows = [
            {
                "duration_seconds": 2.0,
                "events": [
                    _event("A", 0, 0.4, 0.8),
                    _event("A", 0, 1.0, 1.2),
                ],
            }
        ]
        onset, offset, _ = build_fixed_grid_boundary_targets(
            rows, num_labels=2, dilation_frames=0
        )
        self.assertEqual(onset[0, :, 0].nonzero().flatten().tolist(), [10, 25])
        self.assertEqual(offset[0, :, 0].nonzero().flatten().tolist(), [20, 30])

    def test_boundary_loss_has_no_gradient_in_padding(self) -> None:
        rows = [{"duration_seconds": 1.0, "events": [_event("A", 0, 0.4, 0.8)]}]
        onset_target, _, valid = build_fixed_grid_boundary_targets(rows, num_labels=1)
        logits = torch.zeros(1, NUM_FRAMES, 1, requires_grad=True)
        onset_probability, _ = boundary_probabilities_from_activity_logits(logits)
        loss = masked_probability_bce(
            onset_probability,
            onset_target,
            valid,
            pos_weight=torch.tensor([20.0]),
        )
        loss.backward()
        self.assertEqual(float(logits.grad[:, 25:].abs().sum()), 0.0)

    def test_event_matching_is_maximum_cardinality_not_confidence_greedy(self) -> None:
        gold = [
            _event("A", 0, 0.0, 1.0),
            _event("A", 0, 1.0, 2.0),
        ]
        predicted = [
            {**_event("A", 0, 0.5, 1.5), "confidence": 0.99},
            {**_event("A", 0, 0.0, 1.0), "confidence": 0.80},
        ]
        tp, fp, fn = _match_events(predicted, gold, iou_threshold=0.30)
        self.assertEqual(tp["A"], 2)
        self.assertEqual(fp["A"], 0)
        self.assertEqual(fn["A"], 0)

    def test_short_clip_timestamp_is_not_stretched(self) -> None:
        rows = [
            {
                "duration_seconds": 5.0,
                "events": [_event("A", 0, 1.0, 2.0)],
            }
        ]
        target, valid = build_fixed_grid_targets(rows, num_labels=2)
        self.assertEqual(target.shape, (1, 250, 2))
        self.assertEqual(int(valid.sum()), 125)
        self.assertEqual(int(target[0, :, 0].sum()), 25)
        self.assertTrue(bool(target[0, 25:50, 0].all()))
        self.assertFalse(bool(target[0, 50:, 0].any()))

    def test_padding_is_excluded_from_balanced_loss(self) -> None:
        target = torch.zeros((1, NUM_FRAMES, 2))
        valid = torch.zeros((1, NUM_FRAMES), dtype=torch.bool)
        valid[:, :100] = True
        logits_a = torch.zeros_like(target, requires_grad=True)
        logits_b = torch.zeros_like(target)
        logits_b[:, 100:] = 1000.0
        loss_a = masked_balanced_bce(
            logits_a, target, valid, pos_weight=torch.ones(2), class_weight=torch.ones(2)
        )
        loss_b = masked_balanced_bce(
            logits_b, target, valid, pos_weight=torch.ones(2), class_weight=torch.ones(2)
        )
        self.assertAlmostEqual(float(loss_a.detach()), float(loss_b.detach()), places=6)
        loss_a.backward()
        self.assertEqual(float(logits_a.grad[:, 100:].abs().sum()), 0.0)

    def test_class_balancing_upsamples_rare_positive_scene(self) -> None:
        labels = ["frequent", "rare"]
        rows = [
            {"duration_seconds": 10.0, "events": [_event("frequent", 0, 0.0, 1.0)]}
            for _ in range(10)
        ]
        rows.append({"duration_seconds": 10.0, "events": [_event("rare", 1, 0.0, 1.0)]})
        _, class_weight, scene_weight = balancing_tensors(rows, labels)
        self.assertGreater(float(class_weight[1]), float(class_weight[0]))
        self.assertGreater(float(scene_weight[-1]), float(scene_weight[0]))

    def test_perfect_synthetic_prediction_has_perfect_metrics(self) -> None:
        labels = ["A", "B", "C", "D", "E", "F"]
        logits = torch.full((250, len(labels)), -12.0)
        logits[25:35, 0] = 12.0
        # Huge padded false positives must be ignored.
        logits[125:, 1:] = 12.0
        report = evaluate_fixed_grid_predictions(
            [
                {
                    "duration_seconds": 5.0,
                    "valid_frames": 125,
                    "gold_events": [_event("A", 0, 1.0, 1.4)],
                    "logits": logits,
                }
            ],
            labels=labels,
            threshold=0.5,
            event_iou_threshold=0.30,
        )
        self.assertAlmostEqual(report["frame"]["f1"], 1.0)
        self.assertAlmostEqual(report["event"]["f1"], 1.0)
        self.assertAlmostEqual(report["oracle_window_exact_label"]["top1_micro"], 1.0)
        self.assertAlmostEqual(report["oracle_window_exact_label"]["top5_micro"], 1.0)
        self.assertAlmostEqual(report["onset_frame"]["f1"], 1.0)
        self.assertAlmostEqual(report["offset_frame"]["f1"], 1.0)
        self.assertTrue(report["onset_frame"]["diagnostic_only"])
        self.assertTrue(report["offset_frame"]["diagnostic_only"])

    def test_evaluator_rejects_padding_metadata_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid_frames disagrees"):
            evaluate_fixed_grid_predictions(
                [
                    {
                        "duration_seconds": 5.0,
                        "valid_frames": 250,
                        "gold_events": [],
                        "logits": torch.zeros(250, 2),
                    }
                ],
                labels=["A", "B"],
                threshold=0.5,
            )

    def test_small_synthetic_head_can_learn_with_masked_loss(self) -> None:
        torch.manual_seed(7)
        features = torch.randn(16, 20, 4)
        target = (features[..., :2] > 0).float()
        valid = torch.ones((16, 20), dtype=torch.bool)
        head = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(head.parameters(), lr=0.08)
        losses = []
        for _ in range(40):
            logits = head(features)
            loss = masked_balanced_bce(
                logits, target, valid, pos_weight=torch.ones(2), class_weight=torch.ones(2)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1], losses[0] * 0.45)


class DetectorTrainerContractTest(unittest.TestCase):
    def test_trainer_rejects_dilated_activity_edge_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "structurally contradictory"):
            detector_main(["--boundary-dilation-frames", "1"])

    def test_selection_is_scored_with_frozen_calibration_threshold(self) -> None:
        labels = ["A", "B"]

        def prediction(target_probability: float, false_probability: float) -> dict[str, object]:
            logits = torch.full((250, 2), -12.0)
            logits[25:35, 0] = torch.logit(torch.tensor(target_probability))
            logits[50:60, 1] = torch.logit(torch.tensor(false_probability))
            return {
                "duration_seconds": 10.0,
                "valid_frames": 250,
                "gold_events": [_event("A", 0, 1.0, 1.4)],
                "logits": logits,
            }

        calibration_metrics, _ = evaluate_thresholds(
            [prediction(0.70, 0.55)],
            labels=labels,
            thresholds=[0.5, 0.6],
            event_iou_threshold=0.30,
            min_duration_seconds=0.08,
            merge_gap_seconds=0.08,
        )
        self.assertEqual(calibration_metrics["threshold"], 0.6)
        selection = [prediction(0.55, 0.01)]
        frozen = evaluate_fixed_grid_predictions(
            selection,
            labels=labels,
            threshold=calibration_metrics["threshold"],
            event_iou_threshold=0.30,
        )
        recalibrated, _ = evaluate_thresholds(
            selection,
            labels=labels,
            thresholds=[0.5, 0.6],
            event_iou_threshold=0.30,
            min_duration_seconds=0.08,
            merge_gap_seconds=0.08,
        )
        self.assertEqual(frozen["event"]["f1"], 0.0)
        self.assertEqual(recalibrated["event"]["f1"], 1.0)
        self.assertEqual(recalibrated["threshold"], 0.5)

    def test_native_head_mapping_is_hash_and_row_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ontology = root / "ontology.txt"
            labels = [f"label_{index}" for index in range(200)]
            ontology.write_text("\n".join(labels) + "\n")
            state = {
                "strong_head.weight": torch.zeros(200, 2),
                "strong_head.bias": torch.zeros(200),
                "weak_head.weight": torch.zeros(200, 2),
                "weak_head.bias": torch.zeros(200),
            }
            payload = {
                "format": "qces_native_supported_detector_200_v1",
                "labels": labels,
                "model_state_dict": state,
                "initialization": {
                    "ontology_sha256": _sha256(ontology),
                    "native_head_labels": 447,
                    "selected_native_rows": list(range(200)),
                    "checkpoint_sha256": "a" * 64,
                },
            }
            self.assertTrue(
                validate_native_initialization(payload, labels=labels, ontology=ontology)[
                    "passes"
                ]
            )
            payload["initialization"]["selected_native_rows"][-1] = 0
            with self.assertRaisesRegex(RuntimeError, "row mapping"):
                validate_native_initialization(payload, labels=labels, ontology=ontology)

    def test_dev_calibration_selection_split_is_scene_disjoint(self) -> None:
        rows = [
            DetectorScene(
                scene_id=f"scene_{index}",
                split="dev",
                mixture_path="x.wav",
                duration_seconds=10.0,
                sample_rate=16000,
                events=(),
                raw={},
            )
            for index in range(10)
        ]
        calibration, selection = split_calibration_selection_rows(
            rows, seed=9, calibration_fraction=0.4
        )
        self.assertEqual(len(calibration), 4)
        self.assertFalse(
            {row.scene_id for row in calibration} & {row.scene_id for row in selection}
        )

    def test_gate_is_mandatory_and_hash_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ontology = root / "ontology.txt"
            ontology.write_text("\n".join(f"label_{index}" for index in range(200)) + "\n")
            manifests = {}
            for split in ("train", "dev", "test"):
                path = root / f"{split}.jsonl"
                path.write_text("{}\n")
                manifests[split] = path
            missing = root / "missing.json"
            with self.assertRaisesRegex(RuntimeError, "missing detector quota/integrity"):
                validate_gate_receipt(
                    missing,
                    ontology=ontology,
                    manifests=manifests,
                    allow_debug_data=False,
                )
            payload = {
                "format": GATE_FORMAT,
                "passes": True,
                "paper_eligible": True,
                "ontology": {"labels": 200, "sha256": _sha256(ontology)},
                "manifests": {
                    split: {"sha256": _sha256(path)} for split, path in manifests.items()
                },
                "quota": {
                    "passes": True,
                    "selected_classes": 200,
                    "classes_meeting_train_eval_targets": 200,
                    "train_target_videos_per_label": 100,
                    "eval_target_videos_per_label": 20,
                },
                "integrity": {"passes": True, "cross_split_hard_overlap": 0},
            }
            gate = root / "gate.json"
            gate.write_text(json.dumps(payload))
            self.assertTrue(
                validate_gate_receipt(
                    gate,
                    ontology=ontology,
                    manifests=manifests,
                    allow_debug_data=False,
                )["paper_eligible"]
            )
            manifests["dev"].write_text('{"tampered":true}\n')
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                validate_gate_receipt(
                    gate,
                    ontology=ontology,
                    manifests=manifests,
                    allow_debug_data=False,
                )

    def test_manifest_rejects_selected_label_hidden_as_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = [f"label_{index}" for index in range(200)]
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "scene_id": "s1",
                        "split": "train",
                        "mixture_path": "x.wav",
                        "duration_seconds": 10.0,
                        "events": [],
                        "context_events": [_event("label_3", 3, 1.0, 2.0)],
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "incorrectly stored as context"):
                load_manifest(manifest, labels, expected_split="train")

    def test_manifest_rejects_event_outside_declared_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = [f"label_{index}" for index in range(200)]
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "scene_id": "s1",
                        "split": "train",
                        "mixture_path": "x.wav",
                        "duration_seconds": 5.0,
                        "events": [_event("label_0", 0, 4.8, 5.2)],
                    }
                )
                + "\n"
            )
            with self.assertRaisesRegex(ValueError, "invalid event interval"):
                load_manifest(manifest, labels, expected_split="train")

    def test_manifest_clamps_subframe_official_offset_rounding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = [f"label_{index}" for index in range(200)]
            manifest = root / "train.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "scene_id": "s1",
                        "split": "train",
                        "mixture_path": "x.wav",
                        "duration_seconds": 9.7535,
                        "sample_rate": 16000,
                        "events": [
                            {
                                "label": "label_0",
                                "onset_seconds": 0.0,
                                "offset_seconds": 9.763,
                            }
                        ],
                    }
                )
                + "\n"
            )
            rows, _ = load_manifest(manifest, labels, expected_split="train")
            event = rows[0].events[0]
            self.assertEqual(event["offset_seconds"], 9.7535)
            self.assertEqual(event["raw_offset_seconds"], 9.763)
            self.assertTrue(event["offset_clipped_to_duration"])

    def test_stage2_opens_only_last_two_blocks(self) -> None:
        class Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(12)])
                self.layer_norm = nn.LayerNorm(2)

        class Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.beats = nn.Module()
                self.beats.encoder = Encoder()

        class Dummy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = Backbone()
                self.strong_head = nn.Linear(2, 200)
                self.weak_head = nn.Linear(2, 200)

        model = Dummy()
        _, backbone = configure_stage(model, "stage2_last_two_blocks")
        set_train_mode_without_frozen_backbone_dropout(model)
        self.assertTrue(backbone)
        names = {name for name, parameter in model.model.named_parameters() if parameter.requires_grad}
        self.assertTrue(any("layers.10." in name for name in names))
        self.assertTrue(any("layers.11." in name for name in names))
        self.assertFalse(any("layers.9." in name for name in names))
        self.assertFalse(any("layer_norm." in name for name in names))
        self.assertFalse(any(parameter.requires_grad for parameter in model.weak_head.parameters()))
        self.assertFalse(model.model.training)
        self.assertFalse(model.model.beats.encoder.training)
        self.assertFalse(model.model.beats.encoder.layers[0].training)
        self.assertFalse(model.model.beats.encoder.layers[9].training)
        self.assertTrue(model.model.beats.encoder.layers[10].training)
        self.assertTrue(model.model.beats.encoder.layers[11].training)
        self.assertTrue(model.strong_head.training)
        self.assertFalse(model.weak_head.training)

    def test_stage1_freezes_everything_except_strong_head(self) -> None:
        class Dummy(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nn.Linear(2, 2)
                self.strong_head = nn.Linear(2, 200)
                self.weak_head = nn.Linear(2, 200)

        model = Dummy()
        head, backbone = configure_stage(model, "stage1_head_only")
        self.assertTrue(head)
        self.assertFalse(backbone)
        self.assertTrue(all(parameter.requires_grad for parameter in model.strong_head.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.model.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.weak_head.parameters()))


if __name__ == "__main__":
    unittest.main()
