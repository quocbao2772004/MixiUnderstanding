"""Focused regression tests for loading strict QCES v4 manifests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from mixi_understanding.data.qces_v4_schema import QCESV4Record
from mixi_understanding.qces.data import QCESManifestDataset


class QCESV4DatasetTest(unittest.TestCase):
    sample_rate = 8_000
    num_samples = 8_000

    @staticmethod
    def _event(
        event_id: str,
        label: str,
        event_kind: str,
        onset: float,
        offset: float,
    ) -> dict:
        return {
            "event_id": event_id,
            "label": label,
            "event_kind": event_kind,
            "source_dataset": "fixture",
            "source_id": f"source_{event_id}",
            "source_path": f"sources/{event_id}.wav",
            "source_sha256": "0" * 64,
            "source_interval_seconds": [0.0, 1.0],
            "source_crop_interval_seconds": [0.0, offset - onset],
            "onset_seconds": onset,
            "offset_seconds": offset,
            "stem_path": f"audio/events/{event_id}.wav",
        }

    @classmethod
    def _record(cls) -> dict:
        events = [
            cls._event("event_horn", "horn", "semantic", 0.05, 0.10),
            cls._event("event_bell", "bell", "semantic", 0.40, 0.45),
            cls._event("event_rain", "rain", "semantic", 0.80, 0.85),
            cls._event("event_fan", "fan", "nuisance", 0.20, 0.25),
        ]
        return {
            "schema_version": "qces_v4",
            "id": "train_000000",
            "scene_id": "scene_train_000000",
            "question_family_id": "after_horn",
            "counterfactual_group_id": "scene_train_000000_after_horn",
            "paraphrase_family_id": "after_template_0",
            "question_index": 0,
            "split": "train",
            "sample_rate": cls.sample_rate,
            "num_channels": 1,
            "num_samples": cls.num_samples,
            "duration_seconds": 1.0,
            "mixture_path": "audio/mixture.wav",
            "evidence_stem_path": "audio/evidence.wav",
            "residual_stem_path": "audio/residual.wav",
            "anchor_stem_path": "audio/anchor.wav",
            "answer_stem_path": "audio/answer.wav",
            "question": "What sound occurs immediately after the horn?",
            "answer": "bell",
            "answer_options": ["bell", "no_evidence", "croak", "rain", "fan"],
            "answer_option_index": 0,
            "question_type": "temporal_after",
            "relation": "after",
            "no_evidence": False,
            "no_evidence_reason": None,
            "absent_label": None,
            "query_labels": ["horn"],
            "query_event_ids": ["event_horn"],
            "events": events,
            "anchor_event_ids": ["event_horn"],
            "answer_event_ids": ["event_bell"],
            "evidence_event_ids": ["event_horn", "event_bell"],
            "anchor_intervals": [[0.05, 0.10]],
            "answer_intervals": [[0.40, 0.45]],
            "event_presence_labels": ["horn", "bell", "rain", "fan"],
            "source_group_ids": [
                "source_event_horn",
                "source_event_bell",
                "source_event_rain",
                "source_event_fan",
            ],
            "nuisance_snr_db_requested": 0.0,
            "nuisance_snr_db": 0.0,
            "mixture_peak": 0.3,
            "generation_seed": 2026,
        }

    @classmethod
    def _write_fixture(cls, root: Path) -> tuple[Path, dict[str, np.ndarray]]:
        audio_dir = root / "audio"
        audio_dir.mkdir()
        anchor = np.zeros(cls.num_samples, dtype=np.float32)
        answer = np.zeros_like(anchor)
        residual = np.zeros_like(anchor)
        anchor[400:800] = 0.10
        answer[3_200:3_600] = 0.20
        residual[1_600:2_000] = 0.05
        residual[6_400:6_800] = 0.30
        evidence = anchor + answer
        mixture = evidence + residual
        signals = {
            "mixture": mixture,
            "evidence": evidence,
            "residual": residual,
            "anchor": anchor,
            "answer": answer,
        }
        for name, signal in signals.items():
            sf.write(
                audio_dir / f"{name}.wav",
                signal,
                cls.sample_rate,
                subtype="FLOAT",
            )
        manifest_path = root / "manifest.jsonl"
        manifest_path.write_text(
            json.dumps(cls._record()) + "\n", encoding="utf-8"
        )
        return manifest_path, signals

    def test_loads_v4_explicit_stems_and_role_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, signals = self._write_fixture(Path(directory))
            dataset = QCESManifestDataset(manifest_path)
            example = dataset[0]

            self.assertIsInstance(dataset.records[0], QCESV4Record)
            self.assertTrue(
                torch.equal(example.anchor_stem, torch.from_numpy(signals["anchor"]))
            )
            self.assertTrue(
                torch.equal(example.answer_stem, torch.from_numpy(signals["answer"]))
            )
            self.assertTrue(
                torch.equal(example.evidence, torch.from_numpy(signals["evidence"]))
            )
            self.assertEqual(float(example.anchor_mask.sum()), 400.0)
            self.assertEqual(float(example.answer_mask.sum()), 400.0)
            self.assertEqual(float(example.no_evidence), 0.0)

    def test_v4_crop_is_centered_over_the_full_semantic_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path, signals = self._write_fixture(Path(directory))
            dataset = QCESManifestDataset(
                manifest_path,
                crop_samples=6_500,
                random_crop=True,
                seed=7,
            )

            # Semantic events span samples [400, 6800].  The valid starts are
            # [300, 400], so v3/v4 evidence-focused cropping chooses midpoint 350.
            expected_start = 350
            first = dataset[0]
            second = dataset[0]
            self.assertTrue(torch.equal(first.mixture, second.mixture))
            self.assertTrue(
                torch.equal(
                    first.mixture,
                    torch.from_numpy(
                        signals["mixture"][expected_start : expected_start + 6_500]
                    ),
                )
            )
            self.assertEqual(float(first.anchor_mask.sum()), 400.0)
            self.assertEqual(float(first.answer_mask.sum()), 400.0)

            with self.assertRaisesRegex(ValueError, "preserve both roles"):
                QCESManifestDataset(manifest_path, crop_samples=6_000)


if __name__ == "__main__":
    unittest.main()
