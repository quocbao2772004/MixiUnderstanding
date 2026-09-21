from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from mixi_understanding.scripts.export_qces_detector_dense_features import (
    FORMAT,
    FRAME_HOP_SECONDS,
    NUM_FRAMES,
    SHARD_FORMAT,
    fixed_grid_metadata,
    load_scene_ids,
    valid_frames_for_duration,
    validate_index_payload,
    validate_shard_payload,
)


class DenseFeatureExportTest(unittest.TestCase):
    def test_fixed_grid_is_exactly_250_frames_at_40ms(self) -> None:
        grid = fixed_grid_metadata()
        self.assertEqual(FORMAT, "qces_detector_dense_features_v2")
        self.assertEqual(grid["num_frames"], NUM_FRAMES)
        self.assertEqual(NUM_FRAMES, 250)
        self.assertEqual(grid["frame_hop_seconds"], FRAME_HOP_SECONDS)
        self.assertEqual(FRAME_HOP_SECONDS, 0.04)
        self.assertAlmostEqual(NUM_FRAMES * FRAME_HOP_SECONDS, 10.0)
        self.assertEqual(valid_frames_for_duration(0.0), 0)
        self.assertEqual(valid_frames_for_duration(0.001), 1)
        self.assertEqual(valid_frames_for_duration(0.08), 2)
        self.assertEqual(valid_frames_for_duration(float(torch.tensor(9.96, dtype=torch.float32))), 249)
        self.assertEqual(valid_frames_for_duration(9.441360544217687), 237)
        self.assertEqual(valid_frames_for_duration(10.0), 250)
        self.assertEqual(valid_frames_for_duration(12.0), 250)

    def test_validate_dense_shard_schema_and_grid(self) -> None:
        payload = {
            "format": SHARD_FORMAT,
            "scene_ids": ["scene-a", "scene-b"],
            "scenes": [{"scene_id": "scene-a"}, {"scene_id": "scene-b"}],
            "features": torch.zeros((2, 250, 768), dtype=torch.float16),
            "logits": torch.zeros((2, 250, 200), dtype=torch.float16),
            "probs": torch.full((2, 250, 200), 0.5, dtype=torch.float16),
            "duration_seconds": torch.tensor([10.0, 9.441360544217687], dtype=torch.float32),
            "valid_frames": torch.tensor([250, 237], dtype=torch.int16),
            "gold_events": [[], []],
        }
        validate_shard_payload(payload, num_labels=200, feature_dim=768, include_probs=True)
        payload["valid_frames"][1] = 236
        with self.assertRaisesRegex(ValueError, "fixed 40 ms grid"):
            validate_shard_payload(payload, num_labels=200, feature_dim=768, include_probs=True)

    def test_scene_list_json_formats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            list_path = tmp_path / "scenes.json"
            list_path.write_text(json.dumps({"scene_ids": ["b", "a", "a"]}), encoding="utf-8")
            self.assertEqual(load_scene_ids(list_path), {"a", "b"})

            jsonl_path = tmp_path / "scenes.jsonl"
            jsonl_path.write_text('{"scene_id":"x"}\n{"scene_id":"y"}\n', encoding="utf-8")
            self.assertEqual(load_scene_ids(jsonl_path), {"x", "y"})

    def test_validate_index_schema(self) -> None:
        payload = {
            "format": FORMAT,
            "schema_version": 2,
            "labels": ["a", "b"],
            "num_labels": 2,
            "feature_dim": 768,
            "scene_count": 3,
            "shard_count": 2,
            "grid": fixed_grid_metadata(),
            "tensors": {
                "features": {"shape_per_scene": [250, 768]},
                "logits": {"shape_per_scene": [250, 2]},
            },
            "shards": [
                {"index": 0, "path": "shard-0.pt", "scene_count": 2},
                {"index": 1, "path": "shard-1.pt", "scene_count": 1},
            ],
        }
        validate_index_payload(payload)
        payload["grid"]["frame_hop_seconds"] = 0.05
        with self.assertRaisesRegex(ValueError, "fixed 250-frame/40-ms grid"):
            validate_index_payload(payload)


if __name__ == "__main__":
    unittest.main()
