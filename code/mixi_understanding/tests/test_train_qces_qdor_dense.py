from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from mixi_understanding.scripts.train_qces_qdor_dense import (
    DenseFeatureStore,
    RECEIPT_FORMAT,
    load_oracle_gate,
    main,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_dense_export(root: Path, *, scene_count: int = 6) -> tuple[Path, list[str]]:
    labels = ["Camera", "Keys_jangling", "Slam", "Printer"]
    scene_ids = [f"scene_{index:03d}" for index in range(scene_count)]
    feature_dim = 8
    features = torch.randn(scene_count, 250, feature_dim, dtype=torch.float16)
    logits = torch.full((scene_count, 250, len(labels)), -3.0, dtype=torch.float16)
    scenes = []
    all_events = []
    for scene_index, scene_id in enumerate(scene_ids):
        ordered = [scene_index % 4, (scene_index + 1) % 4, (scene_index + 2) % 4]
        events = []
        for event_index, label_id in enumerate(ordered):
            onset = 0.5 + 2.0 * event_index
            offset = onset + 0.6
            events.append(
                {
                    "event_id": f"{scene_id}:e{event_index}",
                    "event_kind": "semantic",
                    "label": labels[label_id],
                    "label_id": label_id,
                    "onset_seconds": onset,
                    "offset_seconds": offset,
                }
            )
            start = int(onset / 0.04)
            end = int(offset / 0.04)
            logits[scene_index, start:end, label_id] = 4.0
        scenes.append(
            {
                "scene_id": scene_id,
                "duration_seconds": 8.0,
                "valid_frames": 200,
                "source_route": "synthetic-test",
            }
        )
        all_events.append(events)
    shard = {
        "format": "qces_detector_dense_features_shard_v2",
        "scene_ids": scene_ids,
        "scenes": scenes,
        "features": features,
        "logits": logits,
        "duration_seconds": torch.full((scene_count,), 8.0, dtype=torch.float32),
        "valid_frames": torch.full((scene_count,), 200, dtype=torch.int16),
        "gold_events": all_events,
    }
    shard_path = root / "shard-00000.pt"
    torch.save(shard, shard_path)
    index = {
        "format": "qces_detector_dense_features_v2",
        "schema_version": 2,
        "labels": labels,
        "num_labels": len(labels),
        "feature_dim": feature_dim,
        "scene_count": scene_count,
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
                "scene_count": scene_count,
                "sha256": _sha256(shard_path),
            }
        ],
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path, scene_ids


class DenseQdorTrainingTest(unittest.TestCase):
    def test_oracle_probe_canonical_receipt_passes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = Path(temporary) / "oracle_probe.json"
            # This mirrors the probe receipt: the historical diagnostic field
            # remains under ``best`` and the machine contract is top-level.
            receipt_path.write_text(
                json.dumps(
                    {
                        "format": "qces_qdor_oracle_span_probe_v1",
                        "best": {"top1_accuracy": 0.81},
                        "oracle_span_answer_top1": 0.81,
                    }
                ),
                encoding="utf-8",
            )

            gate = load_oracle_gate(
                receipt_path,
                minimum_top1=0.75,
                allow_failed=False,
            )

            self.assertTrue(gate["passed"])
            self.assertEqual(gate["metric"], "oracle_span_answer_top1")
            self.assertAlmostEqual(gate["top1"], 0.81)
            self.assertFalse(gate["debug_override"])

    def test_oracle_probe_canonical_receipt_below_threshold_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            receipt_path = Path(temporary) / "oracle_probe.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "format": "qces_qdor_oracle_span_probe_v1",
                        "best": {"top1_accuracy": 0.74},
                        "oracle_span_answer_top1": 0.74,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "below the required 0.7500"):
                load_oracle_gate(
                    receipt_path,
                    minimum_top1=0.75,
                    allow_failed=False,
                )

    def test_training_refuses_missing_oracle_representation_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, scene_ids = _write_dense_export(root)
            train_list = root / "train.txt"
            dev_list = root / "dev.txt"
            train_list.write_text("\n".join(scene_ids[:4]), encoding="utf-8")
            dev_list.write_text("\n".join(scene_ids[4:]), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "oracle-gate-receipt"):
                main(
                    [
                        "--dense-index",
                        str(index_path),
                        "--train-scene-list",
                        str(train_list),
                        "--dev-scene-list",
                        str(dev_list),
                        "--output-dir",
                        str(root / "output"),
                        "--epochs-stage-a",
                        "1",
                        "--epochs-stage-b",
                        "0",
                    ]
                )

    def test_verified_store_rejects_tampered_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, _ = _write_dense_export(root)
            DenseFeatureStore([index_path])
            with (root / "shard-00000.pt").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                DenseFeatureStore([index_path])

    def test_train_dev_overlap_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, scene_ids = _write_dense_export(root)
            train_list = root / "train.txt"
            dev_list = root / "dev.txt"
            train_list.write_text("\n".join(scene_ids[:4]), encoding="utf-8")
            dev_list.write_text("\n".join(scene_ids[3:]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                main(
                    [
                        "--dense-index",
                        str(index_path),
                        "--train-scene-list",
                        str(train_list),
                        "--dev-scene-list",
                        str(dev_list),
                        "--output-dir",
                        str(root / "output"),
                        "--epochs-stage-a",
                        "1",
                        "--epochs-stage-b",
                        "0",
                        "--allow-failed-oracle-gate",
                    ]
                )

    def test_two_stage_cpu_smoke_writes_atomic_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index_path, scene_ids = _write_dense_export(root)
            train_list = root / "train.txt"
            dev_list = root / "dev.txt"
            train_list.write_text("\n".join(scene_ids[:4]), encoding="utf-8")
            dev_list.write_text("\n".join(scene_ids[4:]), encoding="utf-8")
            output_dir = root / "output"
            receipt = main(
                [
                    "--dense-index",
                    str(index_path),
                    "--train-scene-list",
                    str(train_list),
                    "--dev-scene-list",
                    str(dev_list),
                    "--output-dir",
                    str(output_dir),
                    "--device",
                    "cpu",
                    "--hidden-dim",
                    "8",
                    "--batch-size",
                    "8",
                    "--epochs-stage-a",
                    "1",
                    "--epochs-stage-b",
                    "1",
                    "--stage-b-teacher-start",
                    "0.5",
                    "--stage-b-teacher-end",
                    "0.5",
                    "--allow-failed-oracle-gate",
                    "--no-preload",
                ]
            )
            self.assertEqual(receipt["format"], RECEIPT_FORMAT)
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(len(receipt["history"]), 2)
            best = receipt["best_dev"]
            for key in (
                "direct_answerable_accuracy",
                "no_evidence_accuracy",
                "balanced_accuracy",
                "answer_label_top1",
                "answer_label_top5",
                "anchor_iou",
                "answer_iou",
                "union_evidence_iou",
                "strict_joint_answer_anchor_answer_iou_0.30",
            ):
                self.assertIn(key, best)
            self.assertTrue((output_dir / "receipt.json").is_file())
            checkpoint = torch.load(output_dir / "qdor_best.pt", map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["format"], "qces_qdor_dense_checkpoint_v1")


if __name__ == "__main__":
    unittest.main()
