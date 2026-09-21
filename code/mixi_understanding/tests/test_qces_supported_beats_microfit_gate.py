from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch

from mixi_understanding.scripts.audit_qces_supported_beats_microfit_gate import (
    FORMAT,
    build_receipt,
)
from mixi_understanding.scripts.train_qces_supported_beats_detector_v2 import _sha256


class SupportedBeatsMicrofitGateTest(unittest.TestCase):
    def _fixture(self, root: Path, *, cover_all_train_labels: bool) -> argparse.Namespace:
        labels = [f"label_{index}" for index in range(200)]
        ontology = root / "ontology.txt"
        ontology.write_text("\n".join(labels) + "\n", encoding="utf-8")
        initialization = root / "initialization.pt"
        torch.save(
            {
                "format": "qces_native_supported_detector_200_v1",
                "labels": labels,
                "initialization": {
                    "ontology_sha256": _sha256(ontology),
                    "native_head_labels": 447,
                    "selected_native_rows": list(range(200)),
                    "checkpoint_sha256": "a" * 64,
                },
                "model_state_dict": {
                    "strong_head.weight": torch.zeros(200, 1),
                    "strong_head.bias": torch.zeros(200),
                    "weak_head.weight": torch.zeros(200, 1),
                    "weak_head.bias": torch.zeros(200),
                },
            },
            initialization,
        )

        def write_split(split: str, count: int) -> Path:
            path = root / f"{split}.jsonl"
            rows = []
            for index in range(count):
                label_index = index % 200
                if split == "train" and not cover_all_train_labels and index == 199:
                    label_index = 0
                rows.append(
                    {
                        "scene_id": f"{split}_scene_{index}",
                        "video_id": f"{split}_video_{index}",
                        "audio_sha256": f"{split}_{index:064d}",
                        "split": split,
                        "mixture_path": f"audio_that_does_not_exist/{split}_{index}.wav",
                        "duration_seconds": 2.0,
                        "sample_rate": 16000,
                        "events": [
                            {
                                "label": labels[label_index],
                                "onset_seconds": 0.4,
                                "offset_seconds": 0.8,
                            }
                        ],
                        "context_events": [],
                    }
                )
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            return path

        return argparse.Namespace(
            train_manifest=write_split("train", 200),
            dev_manifest=write_split("dev", 20),
            test_manifest=write_split("test", 20),
            ontology=ontology,
            initialization=initialization,
            output=root / "receipt.json",
            min_train_scenes=200,
            max_train_scenes=500,
            min_dev_scenes=20,
            min_test_scenes=20,
            seed=2028,
            calibration_fraction=0.5,
        )

    def test_passes_without_opening_nonexistent_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = build_receipt(
                self._fixture(Path(directory), cover_all_train_labels=True)
            )
        self.assertEqual(receipt["format"], FORMAT)
        self.assertTrue(receipt["passes"])
        self.assertTrue(receipt["scope"]["metadata_training_authorized"])
        self.assertFalse(receipt["scope"]["audio_training_authorized"])
        self.assertEqual(receipt["audio_io"]["audio_files_opened"], 0)
        self.assertEqual(receipt["splits"]["train"]["positive_classes"], 200)
        self.assertEqual(receipt["splits"]["train"]["targets"]["classes_with_onset"], 200)

    def test_fails_when_one_of_200_train_classes_has_no_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = build_receipt(
                self._fixture(Path(directory), cover_all_train_labels=False)
            )
        self.assertFalse(receipt["passes"])
        self.assertIn("train positives cover 199/200 classes", receipt["failures"])
        self.assertFalse(receipt["scope"]["metadata_training_authorized"])


if __name__ == "__main__":
    unittest.main()
