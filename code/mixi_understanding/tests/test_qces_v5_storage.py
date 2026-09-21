"""Focused tests for lossless QCES-v5 scene/event-derived storage."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from unittest import mock

from mixi_understanding.data.qces_v5_schema import (
    DERIVED_SCHEMA_VERSION,
    DERIVED_STORAGE_MODE,
    MATERIALIZED_STORAGE_MODE,
    parse_qces_v5_record,
)
from mixi_understanding.qces.data import QCESManifestDataset
from mixi_understanding.scripts.build_qces_v5_dataset import resolve_storage_mode
from mixi_understanding.scripts.validate_qces_v5_dataset import _artifact_identity


RATE = 8_000
SAMPLES = 24_000


def _event(event_id: str, label: str, onset: float, occurrence: int) -> dict:
    duration = 0.4
    return {
        "event_id": event_id,
        "label": label,
        "event_kind": "semantic",
        "source_dataset": "fixture",
        "dataset_version": "fixture-v1",
        "source_id": f"source_{event_id}",
        "creator_id": f"creator_{event_id}",
        "uploader_id": f"uploader_{event_id}",
        "attribution": f"Fixture creator for {event_id}",
        "source_license_spdx": "CC0-1.0",
        "source_license_url": "https://example.invalid/license",
        "source_partition": "train",
        "source_path": f"sources/{event_id}.wav",
        "source_sha256": "0" * 64,
        "license_record_id": "fixture-license",
        "source_interval_seconds": [0.0, 3.0],
        "source_crop_interval_seconds": [0.0, duration],
        "onset_seconds": onset,
        "offset_seconds": onset + duration,
        "gain_db": 0.0,
        "occurrence_index": occurrence,
        "stem_path": f"audio/events/scene_train_000000_base/{event_id}.wav",
    }


def _materialized_record() -> dict:
    events = [
        _event("sem_00", "frog", 0.10, 1),
        _event("sem_01", "frog", 0.30, 2),
        _event("sem_02", "bell", 0.90, 1),
        _event("sem_03", "dog", 1.50, 1),
    ]
    return {
        "schema_version": "qces_v5",
        "id": "train_000000_0_00",
        "scene_id": "scene_train_000000_base",
        "scene_family_id": "family_train_000000",
        "variant_id": "base",
        "counterfactual_intervention": {
            "kind": "none",
            "parent_variant_id": None,
            "intervened_event_ids": [],
        },
        "question_semantics_id": "after:frog:ordinal=2",
        "counterfactual_group_id": "fixture:after",
        "paraphrase_family_id": "tr_after_next",
        "template_partition": "train",
        "question_index": 0,
        "split": "train",
        "evaluation_axis": "development",
        "sample_rate": RATE,
        "num_channels": 1,
        "num_samples": SAMPLES,
        "duration_seconds": SAMPLES / RATE,
        "mixture_path": "audio/mixture/scene_train_000000_base.wav",
        "evidence_stem_path": "audio/evidence/train_000000_0_00.wav",
        "residual_stem_path": "audio/residual/train_000000_0_00.wav",
        "anchor_stem_path": "audio/anchor/train_000000_0_00.wav",
        "answer_stem_path": "audio/answer/train_000000_0_00.wav",
        "question": "What begins next after the second occurrence of frog?",
        "answer": "bell",
        "answer_options": ["frog", "bell", "dog", "cat", "no_evidence"],
        "answer_option_index": 1,
        "question_type": "temporal_after",
        "relation": "after",
        "no_evidence": False,
        "no_evidence_reason": None,
        "absent_labels": [],
        "query_label": "frog",
        "query_instance_ordinal": 2,
        "query_candidate_labels": [],
        "query_event_ids": ["sem_01"],
        "surface_control_group_id": None,
        "mention_order_variant": "not_applicable",
        "events": events,
        "anchor_event_ids": ["sem_01"],
        "answer_event_ids": ["sem_02"],
        "evidence_event_ids": ["sem_01", "sem_02"],
        "anchor_intervals": [[0.30, 0.70]],
        "answer_intervals": [[0.90, 1.30]],
        "source_group_ids": [f"source_sem_0{index}" for index in range(4)],
        "primary_counterfactual_probe": True,
        "composition_pair_signature": "frog=>bell+dog",
        "composition_triplet_signature": "frog|bell|dog",
        "same_label_repeat": True,
        "semantic_overlap": True,
        "max_polyphony": 2,
        "hard_case_tags": ["same_label_instances", "semantic_overlap"],
        "render_recipe_id": "fixture-v1",
        "mixture_peak": 0.4,
        "family_gain": 1.0,
        "generation_seed": 1,
    }


def _derived_record() -> dict:
    payload = copy.deepcopy(_materialized_record())
    payload["schema_version"] = DERIVED_SCHEMA_VERSION
    payload["storage_mode"] = DERIVED_STORAGE_MODE
    for field in (
        "evidence_stem_path",
        "residual_stem_path",
        "anchor_stem_path",
        "answer_stem_path",
    ):
        payload.pop(field)
    return payload


def _write(path: Path, waveform: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, RATE, subtype="PCM_16")


class QCESV5StorageSchemaTest(unittest.TestCase):
    def test_legacy_materialized_schema_remains_strict_and_readable(self) -> None:
        record = parse_qces_v5_record(_materialized_record())
        self.assertEqual(record.storage_mode, MATERIALIZED_STORAGE_MODE)
        self.assertIsNotNone(record.evidence_stem_path)

    def test_derived_schema_has_no_missing_path_loophole(self) -> None:
        record = parse_qces_v5_record(_derived_record())
        self.assertEqual(record.storage_mode, DERIVED_STORAGE_MODE)
        self.assertIsNone(record.evidence_stem_path)

        missing_mode = _derived_record()
        missing_mode.pop("storage_mode")
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            parse_qces_v5_record(missing_mode)

        smuggled_materialized_path = _derived_record()
        smuggled_materialized_path["evidence_stem_path"] = "missing.wav"
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            parse_qces_v5_record(smuggled_materialized_path)

    def test_cross_root_paths_are_rejected_by_schema(self) -> None:
        payload = _derived_record()
        payload["events"][0]["stem_path"] = "../outside.wav"
        with self.assertRaisesRegex(ValueError, "safe relative path"):
            parse_qces_v5_record(payload)

    def test_paper_default_is_derived_but_legacy_mode_is_selectable(self) -> None:
        self.assertEqual(resolve_storage_mode("paper", None), DERIVED_STORAGE_MODE)
        self.assertEqual(
            resolve_storage_mode("paper", MATERIALIZED_STORAGE_MODE),
            MATERIALIZED_STORAGE_MODE,
        )
        self.assertEqual(
            resolve_storage_mode("smoke", None), MATERIALIZED_STORAGE_MODE
        )


class QCESV5DerivedLoaderTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        record = _materialized_record()
        event_waves = {}
        amplitudes = (0.07, 0.11, 0.13, 0.05)
        for event, amplitude in zip(record["events"], amplitudes):
            waveform = np.zeros(SAMPLES, dtype=np.float32)
            start = int(round(event["onset_seconds"] * RATE))
            stop = int(round(event["offset_seconds"] * RATE))
            waveform[start:stop] = amplitude
            event_waves[event["event_id"]] = waveform
            _write(root / event["stem_path"], waveform)
        mixture = sum(event_waves.values(), np.zeros(SAMPLES, dtype=np.float32))
        evidence = event_waves["sem_01"] + event_waves["sem_02"]
        role_waves = {
            "mixture_path": mixture,
            "evidence_stem_path": evidence,
            "residual_stem_path": mixture - evidence,
            "anchor_stem_path": event_waves["sem_01"],
            "answer_stem_path": event_waves["sem_02"],
        }
        for field, waveform in role_waves.items():
            _write(root / record[field], waveform)

        materialized_manifest = root / "materialized.jsonl"
        materialized_manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
        derived_manifest = root / "derived.jsonl"
        derived_manifest.write_text(
            json.dumps(_derived_record()) + "\n", encoding="utf-8"
        )
        return materialized_manifest, derived_manifest

    def test_derived_waveforms_match_materialized_within_pcm_and_reconstruct_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialized_manifest, derived_manifest = self._fixture(root)
            materialized = QCESManifestDataset(materialized_manifest)[0]
            derived_dataset = QCESManifestDataset(derived_manifest)
            self.assertEqual(len(derived_dataset), 1)
            derived = derived_dataset[0]
            pcm_bound = 3.0 / 32768.0
            for name in ("evidence", "residual", "anchor_stem", "answer_stem"):
                error = torch.max(
                    torch.abs(getattr(materialized, name) - getattr(derived, name))
                ).item()
                self.assertLessEqual(error, pcm_bound, name)
            self.assertTrue(
                torch.equal(derived.evidence + derived.residual, derived.mixture)
            )

    def test_missing_event_file_and_symlink_escape_are_hard_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, derived_manifest = self._fixture(root)
            missing = root / _derived_record()["events"][0]["stem_path"]
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                QCESManifestDataset(derived_manifest)[0]

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            _, derived_manifest = self._fixture(root)
            event_path = root / _derived_record()["events"][0]["stem_path"]
            event_path.unlink()
            external = Path(outside) / "external.wav"
            _write(external, np.zeros(SAMPLES, dtype=np.float32))
            event_path.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                QCESManifestDataset(derived_manifest)[0]

    def test_derived_initialization_is_audio_lazy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, derived_manifest = self._fixture(root)
            with mock.patch(
                "mixi_understanding.qces.data._read_mono",
                side_effect=AssertionError("init must not read audio"),
            ) as reader:
                dataset = QCESManifestDataset(derived_manifest)
                reader.assert_not_called()
                self.assertIsNone(dataset.examples)
                self.assertEqual(len(dataset), 1)

    def test_counterfactual_family_crops_share_one_absolute_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, derived_manifest = self._fixture(root)
            first = _derived_record()
            second = copy.deepcopy(first)
            second["id"] = "train_000000_0_01"
            second["question_index"] = 1
            derived_manifest.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            dataset = QCESManifestDataset(
                derived_manifest,
                crop_samples=16_000,
                random_crop=True,
                seed=91,
                align_v5_family_crops=True,
            )
            starts = {
                dataset._aligned_crop_starts[record.sample_id]
                for record in dataset.records
            }
            self.assertEqual(len(starts), 1)
            self.assertTrue(torch.equal(dataset[0].mixture, dataset[1].mixture))
            self.assertTrue(torch.equal(dataset[0].anchor_mask, dataset[1].anchor_mask))

    def test_scene_cache_is_bounded_and_crop_is_seed_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, derived_manifest = self._fixture(root)
            first = _derived_record()
            second = copy.deepcopy(first)
            second["id"] = "train_000001_0_00"
            second["scene_id"] = "scene_train_000001_base"
            second["scene_family_id"] = "family_train_000001"
            second["mixture_path"] = "audio/mixture/scene_train_000001_base.wav"
            original_mixture, _ = sf.read(
                root / first["mixture_path"], dtype="float32"
            )
            _write(root / second["mixture_path"], original_mixture)
            for event in second["events"]:
                original = next(
                    item for item in first["events"] if item["event_id"] == event["event_id"]
                )
                event["stem_path"] = (
                    f"audio/events/{second['scene_id']}/{event['event_id']}.wav"
                )
                waveform, _ = sf.read(root / original["stem_path"], dtype="float32")
                _write(root / event["stem_path"], waveform)
            derived_manifest.write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )
            dataset = QCESManifestDataset(
                derived_manifest,
                crop_samples=16_000,
                random_crop=True,
                seed=77,
                derived_scene_cache_size=1,
            )
            first_example = dataset._load(dataset.records[0])
            start_a = dataset._crop_start(first_example, dataset.records[0])
            start_b = dataset._crop_start(first_example, dataset.records[0])
            clone = QCESManifestDataset(
                derived_manifest,
                crop_samples=16_000,
                random_crop=True,
                seed=77,
                derived_scene_cache_size=1,
            )
            clone_example = clone._load(clone.records[0])
            self.assertEqual(start_a, start_b)
            self.assertEqual(
                start_a, clone._crop_start(clone_example, clone.records[0])
            )
            dataset[1]
            self.assertEqual(len(dataset._derived_scene_cache), 1)
            self.assertEqual(
                next(iter(dataset._derived_scene_cache)), second["scene_id"]
            )
            dataset[0]
            self.assertEqual(len(dataset._derived_scene_cache), 1)
            self.assertEqual(next(iter(dataset._derived_scene_cache)), first["scene_id"])

    def test_artifact_fingerprint_is_order_independent_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, derived_manifest = self._fixture(root)
            relatives = [
                derived_manifest.name,
                _derived_record()["mixture_path"],
                *[event["stem_path"] for event in _derived_record()["events"]],
            ]
            first, first_files = _artifact_identity(root, relatives)
            second, second_files = _artifact_identity(root, reversed(relatives))
            self.assertEqual(first, second)
            self.assertEqual(first_files, second_files)


if __name__ == "__main__":
    unittest.main()
