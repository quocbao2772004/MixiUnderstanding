"""Data, calibration, and microfit contracts for the Q-DOR-v2 trainer."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from mixi_understanding.qces.dense_event_qa_v2 import DenseTemporalThresholdsV2
from mixi_understanding.scripts.train_qces_qdor_dense import DenseFeatureStore
from mixi_understanding.scripts.train_qces_qdor_dense_v2 import (
    CHECKPOINT_FORMAT_V2,
    RECEIPT_FORMAT_V2,
    EvaluationCacheV2,
    assert_dense_identity_disjoint_v2,
    bind_qa_items_to_dense_gold,
    calibrate_thresholds_v2,
    checkpoint_selection_key,
    dense_scene_onset_targets,
    load_bound_qa_manifest_v2,
    main,
    run_toy_microfit_v2,
    score_evaluation_cache_v2,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _events(scene_id: str, positive_layout: bool) -> list[dict[str, object]]:
    labels = ["A", "B", "C"] if positive_layout else ["B", "C", "A"]
    intervals = [(0.4, 0.8), (2.0, 2.4), (4.0, 4.4)]
    ontology = {"A": 0, "B": 1, "C": 2}
    return [
        {
            "event_id": f"{scene_id}:e{index}",
            "event_kind": "semantic",
            "label": label,
            "label_id": ontology[label],
            "onset_seconds": onset,
            "offset_seconds": offset,
        }
        for index, (label, (onset, offset)) in enumerate(zip(labels, intervals, strict=True))
    ]


def _write_dense_export(root: Path) -> tuple[Path, list[str]]:
    torch.manual_seed(2041)
    labels = ["A", "B", "C"]
    scene_ids = [f"scene_{index}" for index in range(6)]
    features = torch.randn(6, 250, 4, dtype=torch.float16)
    logits = torch.full((6, 250, 3), -8.0, dtype=torch.float16)
    scenes: list[dict[str, object]] = []
    gold_events: list[list[dict[str, object]]] = []
    for index, scene_id in enumerate(scene_ids):
        events = _events(scene_id, positive_layout=index % 2 == 0)
        for event in events:
            onset = int(round(float(event["onset_seconds"]) / 0.04))
            offset = int(round(float(event["offset_seconds"]) / 0.04))
            logits[index, onset:offset, int(event["label_id"])] = 8.0
        scenes.append({"scene_id": scene_id, "duration_seconds": 10.0})
        gold_events.append(events)
    shard = {
        "format": "qces_detector_dense_features_shard_v2",
        "scene_ids": scene_ids,
        "scenes": scenes,
        "features": features,
        "logits": logits,
        "valid_frames": torch.full((6,), 250, dtype=torch.int16),
        "gold_events": gold_events,
    }
    shard_path = root / "shard-00000.pt"
    torch.save(shard, shard_path)
    index = {
        "format": "qces_detector_dense_features_v2",
        "schema_version": 2,
        "labels": labels,
        "num_labels": len(labels),
        "feature_dim": 4,
        "scene_count": len(scene_ids),
        "shard_count": 1,
        "grid": {
            "num_frames": 250,
            "fixed_audio_seconds": 10.0,
            "frame_hop_seconds": 0.04,
        },
        "shards": [
            {
                "index": 0,
                "path": shard_path.name,
                "scene_count": len(scene_ids),
                "sha256": _sha256(shard_path),
            }
        ],
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path, scene_ids


def _qa_row(scene_id: str, positive: bool) -> dict[str, object]:
    if positive:
        anchor = [0.4, 0.8]
        answer = [2.0, 2.4]
        return {
            "item_id": f"{scene_id}:positive",
            "scene_id": scene_id,
            "relation": "after",
            "question": "What occurs immediately after the first A?",
            "answer": "B",
            "no_evidence": False,
            "no_evidence_reason": None,
            "anchor_label": "A",
            "anchor_ordinal": 1,
            "answer_label": "B",
            "gold_anchor_interval": anchor,
            "gold_answer_interval": answer,
            "gold_verification_interval": None,
            "gold_evidence_intervals": [anchor, answer],
        }
    return {
        "item_id": f"{scene_id}:none",
        "scene_id": scene_id,
        "relation": "after",
        "question": "What occurs immediately after the first A?",
        "answer": "no_evidence",
        "no_evidence": True,
        "no_evidence_reason": "no_event_after_anchor",
        "anchor_label": "A",
        "anchor_ordinal": 1,
        "answer_label": None,
        "gold_anchor_interval": [4.0, 4.4],
        "gold_answer_interval": None,
        "gold_verification_interval": [4.4, 10.0],
        "gold_evidence_intervals": [[4.0, 10.0]],
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _cache(answerability: list[float]) -> EvaluationCacheV2:
    rows, frames = 4, 8
    gold_label = torch.tensor([1, 1, -1, -1])
    anchor = torch.zeros(rows, frames)
    answer = torch.zeros(rows, frames)
    verification = torch.zeros(rows, frames)
    evidence = torch.zeros(rows, frames)
    anchor[:, 1:3] = 1.0
    answer[:2, 4:6] = 1.0
    verification[2:, 3:] = 1.0
    evidence[:2] = anchor[:2] + answer[:2]
    evidence[2:, 1:] = 1.0
    return EvaluationCacheV2(
        answerability_probability=torch.tensor(answerability),
        raw_answer_label=torch.tensor([1, 1, 0, 0]),
        gold_answer_label=gold_label,
        anchor_probability=anchor * 0.98 + (1.0 - anchor) * 0.02,
        answer_probability=answer * 0.98 + (1.0 - answer) * 0.02,
        verification_probability=verification,
        gold_anchor_mask=anchor,
        gold_answer_mask=answer,
        gold_verification_mask=verification,
        gold_evidence_mask=evidence.clamp_max(1.0),
        valid_mask=torch.ones(rows, frames, dtype=torch.bool),
    )


class QdorV2TrainerTest(unittest.TestCase):
    def test_source_identity_overlap_is_fatal_even_when_scene_ids_differ(self) -> None:
        class FakeStore:
            def metadata(self, scene_id: str) -> dict[str, object]:
                return {
                    "scene_id": scene_id,
                    "events": [
                        {
                            "event_id": f"{scene_id}:e0",
                            "event_kind": "semantic",
                            "label": "A",
                            "onset_seconds": 0.4,
                            "offset_seconds": 0.8,
                            "source_sha256": "same-source",
                        }
                    ],
                }

        with self.assertRaisesRegex(ValueError, "source identity leakage"):
            assert_dense_identity_disjoint_v2(
                FakeStore(),  # type: ignore[arg-type]
                {"calibration": ["scene_a"], "selection": ["scene_b"]},
            )

    def test_dense_binding_rejects_silent_ordinal_clamp_and_wrong_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, scene_ids = _write_dense_export(root)
            manifest = root / "qa.jsonl"
            _write_jsonl(manifest, [_qa_row(scene_ids[0], True), _qa_row(scene_ids[1], False)])
            store = DenseFeatureStore([index_path])
            bound = load_bound_qa_manifest_v2(
                manifest,
                allowed_scene_ids=scene_ids[:2],
                store=store,
                max_ordinal=3,
            )
            bad_ordinal = replace(bound[0].qa, anchor_ordinal=2)
            with self.assertRaisesRegex(ValueError, "silently clamped"):
                bind_qa_items_to_dense_gold([bad_ordinal], store, max_ordinal=3)
            bad_answer = replace(bound[0].qa, answer_label="C")
            with self.assertRaisesRegex(ValueError, "immediate dense event"):
                bind_qa_items_to_dense_gold([bad_answer], store, max_ordinal=3)

    def test_scene_onset_targets_have_class_frames_and_no_false_frame_zero(self) -> None:
        events = _events("scene", positive_layout=True)
        class_mask, union = dense_scene_onset_targets(
            events,
            label_to_id={"A": 0, "B": 1, "C": 2},
            valid_frames=250,
        )
        self.assertEqual(float(class_mask.sum()), 3.0)
        self.assertEqual(float(class_mask[10, 0]), 1.0)
        self.assertEqual(float(class_mask[50, 1]), 1.0)
        self.assertEqual(float(class_mask[100, 2]), 1.0)
        self.assertEqual(float(union[0]), 0.0)

    def test_calibration_threshold_is_frozen_before_selection_scoring(self) -> None:
        calibration = _cache([0.60, 0.70, 0.30, 0.40])
        thresholds, calibrated = calibrate_thresholds_v2(
            calibration,
            answerability_values=[0.4, 0.5, 0.65],
            anchor_values=[0.5],
            answer_values=[0.5],
        )
        self.assertEqual(thresholds.answerability, 0.5)
        self.assertEqual(calibrated["strict_balanced"], 1.0)
        # Selection deliberately has a different score distribution.  It is
        # scored with the frozen calibration threshold, never recalibrated.
        selection = _cache([0.45, 0.46, 0.44, 0.43])
        selected = score_evaluation_cache_v2(selection, thresholds)
        self.assertEqual(selected["strict_positive"], 0.0)
        self.assertEqual(selected["strict_none"], 1.0)

    def test_checkpoint_selection_prioritizes_strict_balance(self) -> None:
        stronger_strict = {
            "strict_balanced": 0.70,
            "strict_min": 0.65,
            "strict_positive": 0.75,
            "strict_none": 0.65,
            "answer_balanced_accuracy": 0.70,
            "evidence_iou": 0.60,
        }
        weaker_strict_higher_answer = {
            "strict_balanced": 0.69,
            "strict_min": 0.68,
            "strict_positive": 0.70,
            "strict_none": 0.68,
            "answer_balanced_accuracy": 0.99,
            "evidence_iou": 0.95,
        }
        self.assertGreater(
            checkpoint_selection_key(stronger_strict),
            checkpoint_selection_key(weaker_strict_higher_answer),
        )

    def test_strict_positive_rejects_union_evidence_leakage(self) -> None:
        # Adversarial metric-only cache: each component just clears 0.30 IoU,
        # but their leaked union covers four times the gold evidence support.
        frames = 8
        gold_anchor = torch.zeros(1, frames)
        gold_answer = torch.zeros(1, frames)
        gold_anchor[0, 7] = 1.0
        gold_answer[0, 7] = 1.0
        predicted_anchor = torch.zeros(1, frames)
        predicted_answer = torch.zeros(1, frames)
        predicted_anchor[0, [6, 7]] = 1.0
        predicted_answer[0, [4, 5, 7]] = 1.0
        cache = EvaluationCacheV2(
            answerability_probability=torch.tensor([0.9]),
            raw_answer_label=torch.tensor([1]),
            gold_answer_label=torch.tensor([1]),
            anchor_probability=predicted_anchor,
            answer_probability=predicted_answer,
            verification_probability=torch.zeros(1, frames),
            gold_anchor_mask=gold_anchor,
            gold_answer_mask=gold_answer,
            gold_verification_mask=torch.zeros(1, frames),
            gold_evidence_mask=(gold_anchor.bool() | gold_answer.bool()).to(torch.float32),
            valid_mask=torch.ones(1, frames, dtype=torch.bool),
        )
        metrics = score_evaluation_cache_v2(cache, DenseTemporalThresholdsV2())
        self.assertGreaterEqual(metrics["anchor_iou"], 0.30)
        self.assertGreaterEqual(metrics["answer_iou_positive"], 0.30)
        self.assertLess(metrics["evidence_iou_positive"], 0.30)
        self.assertEqual(metrics["strict_positive"], 0.0)

    def test_toy_microfit_exceeds_ninety_five_percent_for_both_modes(self) -> None:
        metrics = run_toy_microfit_v2(steps=20)
        self.assertGreaterEqual(metrics["strict_positive"], 0.95)
        self.assertGreaterEqual(metrics["strict_none"], 0.95)
        self.assertGreaterEqual(metrics["strict_min"], 0.95)

    def test_cpu_smoke_writes_v2_checkpoint_with_frozen_thresholds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, scene_ids = _write_dense_export(root)
            train_list = root / "train.txt"
            calibration_list = root / "calibration.txt"
            selection_list = root / "selection.txt"
            train_list.write_text("\n".join(scene_ids[:2]), encoding="utf-8")
            calibration_list.write_text("\n".join(scene_ids[2:4]), encoding="utf-8")
            selection_list.write_text("\n".join(scene_ids[4:6]), encoding="utf-8")
            train_manifest = root / "train.jsonl"
            dev_manifest = root / "dev.jsonl"
            _write_jsonl(
                train_manifest,
                [_qa_row(scene_ids[0], True), _qa_row(scene_ids[1], False)],
            )
            _write_jsonl(
                dev_manifest,
                [
                    _qa_row(scene_ids[2], True),
                    _qa_row(scene_ids[3], False),
                    _qa_row(scene_ids[4], True),
                    _qa_row(scene_ids[5], False),
                ],
            )
            output = root / "output"
            receipt = main(
                [
                    "--dense-index",
                    str(index_path),
                    "--train-scene-list",
                    str(train_list),
                    "--dev-scene-list",
                    str(selection_list),
                    "--calibration-scene-list",
                    str(calibration_list),
                    "--train-qa-manifest",
                    str(train_manifest),
                    "--dev-qa-manifest",
                    str(dev_manifest),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                    "--hidden-dim",
                    "8",
                    "--batch-size",
                    "2",
                    "--epochs-stage-a",
                    "1",
                    "--epochs-stage-b",
                    "0",
                    "--allow-failed-oracle-gate",
                    "--no-preload",
                ]
            )
            self.assertEqual(receipt["format"], RECEIPT_FORMAT_V2)
            self.assertFalse(receipt["data_protocol"]["thresholds_calibrated_on_selection"])
            checkpoint = torch.load(
                output / "qdor_v2_best.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(checkpoint["format"], CHECKPOINT_FORMAT_V2)
            self.assertIn("thresholds", checkpoint)
            self.assertIn("strict_balanced", checkpoint["selection_metrics"])


if __name__ == "__main__":
    unittest.main()
