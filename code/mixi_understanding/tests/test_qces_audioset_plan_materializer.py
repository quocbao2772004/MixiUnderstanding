from __future__ import annotations

import hashlib
import fcntl
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from mixi_understanding.qces.audioset_plan_materializer import (
    MANIFEST_FORMAT,
    MaterializationConfig,
    MaterializationError,
    RetryPolicy,
    run_materialization,
)
from mixi_understanding.scripts.materialize_qces_supported_audioset_plan import (
    main as materializer_main,
)


def _flac_bytes(*, frequency: float, duration: float = 0.25) -> bytes:
    sample_rate = 16_000
    times = np.arange(int(sample_rate * duration), dtype=np.float32) / sample_rate
    waveform = 0.1 * np.sin(2.0 * np.pi * frequency * times)
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="FLAC")
    return buffer.getvalue()


def _write_parquet(path: Path, video_ids: list[str]) -> dict[str, bytes]:
    encoded = {
        video_id: _flac_bytes(frequency=220.0 + 30.0 * index)
        for index, video_id in enumerate(video_ids)
    }
    rows = []
    for index, video_id in enumerate(video_ids):
        rows.append(
            {
                "video_id": video_id,
                "audio": {"bytes": encoded[video_id], "path": f"{video_id}.flac"},
                "labels": ["/m/test"],
                "human_labels": ["Test sound"],
                "events": [
                    {"end": 0.20, "event_name": "Test sound", "start": 0.02}
                ],
            }
        )
    schema = pa.schema(
        [
            pa.field("video_id", pa.string()),
            pa.field(
                "audio",
                pa.struct(
                    [pa.field("bytes", pa.binary()), pa.field("path", pa.string())]
                ),
            ),
            pa.field("labels", pa.list_(pa.string())),
            pa.field("human_labels", pa.list_(pa.string())),
            pa.field(
                "events",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("end", pa.float64()),
                            pa.field("event_name", pa.string()),
                            pa.field("start", pa.float64()),
                        ]
                    )
                ),
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, row_group_size=2)
    return encoded


def _plan_row(video_id: str, *, offset: float = 0.20) -> dict[str, object]:
    return {
        "format": "qces_supported_ontology_v1",
        "hf_dataset": "enyoukai/AudioSet-Strong",
        "metadata_split": "train",
        "hf_split": "train",
        "video_id": video_id,
        "segment_ids": [f"{video_id}_0"],
        "labels": ["Test_sound"],
        "covers_deficit_labels": ["Test_sound"],
        "events": [
            {
                "display_name": "Test sound",
                "label": "Test_sound",
                "mid": "/m/test",
                "offset_seconds": offset,
                "onset_seconds": 0.02,
                "segment_id": f"{video_id}_0",
                "video_id": video_id,
            }
        ],
    }


def _crop_plan_row(video_id: str, *, two_items: bool = False) -> dict[str, object]:
    row = _plan_row(video_id)
    row["format"] = "qces_joint_acoustic_materialization_plan_v1"
    row["split_lock"] = "unassigned_train_pool"
    row["full_official_strong_events"] = True
    row["events"] = [
        {
            "display_name": "Test sound",
            "label": "Test_sound",
            "mid": "/m/test",
            "offset_seconds": 0.20,
            "onset_seconds": 0.02,
            "segment_id": f"{video_id}_0",
            "video_id": video_id,
            "selected_ontology_label": True,
        },
        {
            "display_name": "Music",
            "label": "Music",
            "mid": "/m/music",
            "offset_seconds": 0.16,
            "onset_seconds": 0.08,
            "segment_id": f"{video_id}_0",
            "video_id": video_id,
            "selected_ontology_label": False,
        },
    ]
    requests = [
        {
            "selection_key": f"selection-{video_id}-0",
            "coverage_label": "Test_sound",
            "coverage_mid": "/m/test",
            "coverage_rank": 1,
            "ambiguity_tier": 1,
            "ambiguity_tier_name": "selected_ontology_isolated",
            "clean_source_eligible": True,
            "fully_isolated": False,
            "crop_start_seconds": 0.0,
            "crop_end_seconds": 0.18,
            "event_onset_seconds": 0.02,
            "event_offset_seconds": 0.20,
            "strong_annotations": [{"label": "Music", "mid": "/m/music"}],
            "selected_ontology_annotations": [
                {"label": "Test_sound", "mid": "/m/test"}
            ],
            "materialized_audio_path": "/must/not/leak/source.flac",
        }
    ]
    if two_items:
        requests.append(
            {
                **requests[0],
                "selection_key": f"selection-{video_id}-1",
                "coverage_rank": 2,
                "crop_start_seconds": 0.01,
                "crop_end_seconds": 0.22,
            }
        )
    row["crop_requests"] = requests
    row["coverage_request_count"] = len(requests)
    return row


def _write_plan(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class AudioSetPlanMaterializerTests(unittest.TestCase):
    def test_preindexed_location_uses_exact_shard_without_fallback_glob(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "physical/exact.parquet"
            encoded = _write_parquet(
                parquet_path, ["video0", "video1", "video2"]
            )
            row = _plan_row("video2")
            row.update(
                {
                    "source_route": "local_exact_route",
                    "hf_dataset": "local/exact-dataset",
                    "hf_revision": "local-exact-revision",
                    "availability_location": {
                        "source_route": "local_exact_route",
                        "hf_dataset": "local/exact-dataset",
                        "hf_revision": "local-exact-revision",
                        "hf_split": "train",
                        "video_id": "video2",
                        "parquet_url": str(parquet_path),
                        "row_group": 1,
                        "row_index": 0,
                        "shard_provenance": {
                            "file_size_bytes": parquet_path.stat().st_size,
                        },
                    },
                }
            )
            plan_path = root / "plan.jsonl"
            _write_plan(plan_path, [row])
            config = MaterializationConfig(
                plan_paths=(plan_path,),
                output_dir=root / "output",
                # This path deliberately cannot locate the exact shard.
                parquet_pattern=str(root / "wrong-mirror/{split}-*.parquet"),
                hf_dataset="local/exact-dataset",
                resolved_revision="local-exact-revision",
                require_preindexed_locations=True,
                minimum_free_disk_bytes=0,
                retry=RetryPolicy(max_attempts=1),
            )
            with patch(
                "mixi_understanding.qces.audioset_plan_materializer.glob_parquet_urls",
                side_effect=AssertionError("fallback glob must not be called"),
            ):
                receipt = run_materialization(config)
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["location_mode"], "preindexed_required")
            self.assertEqual(receipt["row_groups_scanned"], 0)
            self.assertEqual(
                receipt["preindexed_location_summary"]["validated_plan_rows"], 1
            )
            self.assertFalse(
                receipt["preindexed_location_summary"]["fallback_glob_used"]
            )
            result = json.loads(
                (root / "output/audioset_strong_plan_manifest.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(result["source_route"], "local_exact_route")
            self.assertEqual(result["hf_dataset"], "local/exact-dataset")
            self.assertEqual(result["hf_revision"], "local-exact-revision")
            self.assertEqual(
                hashlib.sha256(Path(result["mixture_path"]).read_bytes()).hexdigest(),
                hashlib.sha256(encoded["video2"]).hexdigest(),
            )

    def test_preindexed_location_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parquet_path = root / "physical/exact.parquet"
            _write_parquet(parquet_path, ["video0"])
            base = _plan_row("video0")
            base.update(
                {
                    "source_route": "local_exact_route",
                    "hf_dataset": "local/exact-dataset",
                    "hf_revision": "local-exact-revision",
                }
            )
            cases: list[tuple[str, dict[str, object], str]] = []
            cases.append(("missing", dict(base), "contract is incomplete"))
            valid_location = {
                "source_route": "local_exact_route",
                "hf_dataset": "local/exact-dataset",
                "hf_revision": "local-exact-revision",
                "hf_split": "train",
                "video_id": "video0",
                "parquet_url": str(parquet_path),
                "row_group": 0,
                "row_index": 0,
                "shard_provenance": {
                    "file_size_bytes": parquet_path.stat().st_size,
                },
            }
            mismatch = dict(base)
            mismatch["availability_location"] = {
                **valid_location,
                "video_id": "different-video",
            }
            cases.append(("mismatch", mismatch, "video_id mismatch"))
            wrong_row = dict(base)
            wrong_row["availability_location"] = {
                **valid_location,
                "row_index": 1,
            }
            cases.append(("wrong-row", wrong_row, "row index is out of range"))
            wrong_provenance = dict(base)
            wrong_provenance["availability_location"] = {
                **valid_location,
                "shard_provenance": {
                    "file_size_bytes": parquet_path.stat().st_size + 1,
                },
            }
            cases.append(
                ("wrong-provenance", wrong_provenance, "provenance changed")
            )
            wrong_revision = dict(base)
            wrong_revision["hf_revision"] = "different-revision"
            wrong_revision["availability_location"] = {
                **valid_location,
                "hf_revision": "different-revision",
            }
            cases.append(
                ("wrong-revision", wrong_revision, "plan revision mismatch")
            )
            for name, row, message in cases:
                with self.subTest(name=name):
                    plan_path = root / f"plan-{name}.jsonl"
                    _write_plan(plan_path, [row])
                    config = MaterializationConfig(
                        plan_paths=(plan_path,),
                        output_dir=root / f"output-{name}",
                        parquet_pattern=str(root / "physical/{split}-*.parquet"),
                        hf_dataset="local/exact-dataset",
                        resolved_revision="local-exact-revision",
                        require_preindexed_locations=True,
                        minimum_free_disk_bytes=0,
                        retry=RetryPolicy(max_attempts=1),
                    )
                    with patch(
                        "mixi_understanding.qces.audioset_plan_materializer.glob_parquet_urls",
                        side_effect=AssertionError("fallback glob must not be called"),
                    ):
                        with self.assertRaisesRegex(MaterializationError, message):
                            run_materialization(config)

    def test_output_directory_rejects_concurrent_materializer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_parquet(
                root / "parquet/train-00000-of-00001.parquet", ["video0"]
            )
            plan_path = root / "plan.jsonl"
            _write_plan(plan_path, [_plan_row("video0")])
            output_dir = root / "output"
            output_dir.mkdir()
            lock_handle = (output_dir / ".materialization.lock").open("a+")
            try:
                fcntl.flock(
                    lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
                with self.assertRaisesRegex(
                    MaterializationError, "holds the output lock"
                ):
                    run_materialization(
                        MaterializationConfig(
                            plan_paths=(plan_path,),
                            output_dir=output_dir,
                            parquet_pattern=str(
                                root / "parquet/{split}-*.parquet"
                            ),
                            resolved_revision="local-test-revision",
                            minimum_free_disk_bytes=0,
                            retry=RetryPolicy(max_attempts=1),
                        )
                    )
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()

    def test_requested_crop_mode_flattens_items_and_separates_context_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoded = _write_parquet(
                root / "parquet/train-00000-of-00001.parquet",
                ["video0", "video1", "video2"],
            )
            plan_path = root / "crop-plan.jsonl"
            _write_plan(
                plan_path,
                [_crop_plan_row("video0", two_items=True), _crop_plan_row("video2")],
            )
            existing_audio = root / "existing/video0.flac"
            existing_audio.parent.mkdir(parents=True)
            existing_audio.write_bytes(encoded["video0"])
            existing_manifest = root / "existing.jsonl"
            existing_manifest.write_text(
                json.dumps(
                    {
                        "video_id": "video0",
                        "hf_split": "train",
                        "mixture_path": str(existing_audio),
                        "audio_sha256": hashlib.sha256(encoded["video0"]).hexdigest(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            common = dict(
                plan_paths=(plan_path,),
                output_dir=root / "output",
                parquet_pattern=str(root / "parquet/{split}-*.parquet"),
                resolved_revision="local-test-revision",
                storage_mode="requested_crops",
                existing_manifest_paths=(existing_manifest,),
                minimum_free_disk_bytes=0,
                retry=RetryPolicy(max_attempts=1),
            )
            scan_receipt = run_materialization(
                MaterializationConfig(**common, scan_only=True)
            )
            self.assertEqual(scan_receipt["plan_rows_accounted_for"], 2)
            self.assertEqual(
                scan_receipt["plan_rows_with_validated_local_crop_source"], 1
            )
            self.assertEqual(scan_receipt["estimates"]["crop_items_total"], 3)
            self.assertFalse((root / "output/audio/crops").exists())

            def interrupt_crop(event: str, payload: object) -> None:
                if event == "materialize_existing_crop_batch_committed":
                    raise RuntimeError("simulated crop interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated crop interruption"):
                run_materialization(
                    MaterializationConfig(**common), checkpoint_hook=interrupt_crop
                )
            receipt = run_materialization(MaterializationConfig(**common))
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["plan_rows_completed"], 2)
            self.assertEqual(receipt["estimates"]["crop_items_completed"], 3)
            crop_path = root / "output/audioset_strong_crop_manifest.jsonl"
            crops = [json.loads(line) for line in crop_path.read_text().splitlines()]
            self.assertEqual(len(crops), 3)
            for crop in crops:
                self.assertEqual([event["label"] for event in crop["events"]], ["Test_sound"])
                self.assertEqual([event["label"] for event in crop["context_events"]], ["Music"])
                self.assertEqual(len(crop["all_strong_events"]), 2)
                self.assertEqual(crop["complete_intersecting_strong_annotation_count"], 2)
                self.assertNotIn("Music", crop["labels"])
                self.assertIn("Music", crop["all_strong_labels"])
                self.assertNotIn("strong_annotations", crop["crop_request"])
                self.assertNotIn(
                    "selected_ontology_annotations", crop["crop_request"]
                )
                self.assertNotIn("materialized_audio_path", crop["crop_request"])
                self.assertEqual(
                    crop["crop_request"]["raw_strong_annotation_count"], 1
                )
                self.assertTrue(Path(crop["mixture_path"]).is_file())
            self.assertFalse((root / "output/audio/train/video2.flac").exists())

            # FLAC decode/re-encode preserves the exact PCM samples in the
            # requested frame interval.
            source_pcm, source_rate = sf.read(
                io.BytesIO(encoded["video0"]), dtype="int32", always_2d=True
            )
            first = next(
                crop
                for crop in crops
                if crop["materialization_item_id"] == "selection-video0-0"
            )
            crop_pcm, crop_rate = sf.read(
                Path(first["mixture_path"]), dtype="int32", always_2d=True
            )
            self.assertEqual(source_rate, crop_rate)
            start = int(round(first["source_crop_start_seconds"] * source_rate))
            end = int(round(first["source_crop_end_seconds"] * source_rate))
            np.testing.assert_array_equal(crop_pcm, source_pcm[start:end])

    def test_requested_crop_mode_pads_small_nominal_tail_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_parquet(
                root / "parquet/train-00000-of-00001.parquet", ["video0"]
            )
            row = _crop_plan_row("video0")
            row["events"] = [
                {
                    "display_name": "Test sound",
                    "label": "Test_sound",
                    "mid": "/m/test",
                    "offset_seconds": 0.30,
                    "onset_seconds": 0.256,
                    "segment_id": "video0_0",
                    "video_id": "video0",
                    "selected_ontology_label": True,
                }
            ]
            request = row["crop_requests"][0]
            request.update(
                {
                    "crop_start_seconds": 0.20,
                    "crop_end_seconds": 0.30,
                    "event_onset_seconds": 0.256,
                    "event_offset_seconds": 0.30,
                }
            )
            plan_path = root / "crop-plan.jsonl"
            _write_plan(plan_path, [row])
            receipt = run_materialization(
                MaterializationConfig(
                    plan_paths=(plan_path,),
                    output_dir=root / "output",
                    parquet_pattern=str(root / "parquet/{split}-*.parquet"),
                    resolved_revision="local-test-revision",
                    storage_mode="requested_crops",
                    minimum_free_disk_bytes=0,
                    retry=RetryPolicy(max_attempts=1),
                )
            )
            self.assertTrue(receipt["complete"])
            crop = json.loads(
                (root / "output/audioset_strong_crop_manifest.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertTrue(crop["nominal_tail_padding_applied"])
            self.assertAlmostEqual(crop["nominal_tail_padding_seconds"], 0.05)
            self.assertEqual(crop["coverage_event_retained_seconds"], 0.0)
            self.assertEqual(crop["coverage_event_retained_fraction"], 0.0)
            self.assertTrue(crop["coverage_event_clipped_to_decoded_audio"])
            samples, sample_rate = sf.read(
                Path(crop["mixture_path"]), dtype="int32", always_2d=True
            )
            decoded_samples = int(round(0.05 * sample_rate))
            self.assertGreater(np.max(np.abs(samples[:decoded_samples])), 0)
            self.assertEqual(np.max(np.abs(samples[decoded_samples:])), 0)

    def test_cli_bounded_smoke_exits_nonzero_until_complete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_parquet(
                root / "parquet/train-00000-of-00001.parquet",
                ["video0", "video1", "video2"],
            )
            plan_path = root / "plan.jsonl"
            _write_plan(plan_path, [_plan_row("video0"), _plan_row("video2")])
            common = [
                "--plan",
                str(plan_path),
                "--output-dir",
                str(root / "output"),
                "--parquet-pattern",
                str(root / "parquet/{split}-*.parquet"),
                "--resolved-revision",
                "local-test-revision",
                "--min-free-disk-gib",
                "0",
                "--max-retries",
                "1",
            ]
            self.assertEqual(materializer_main([*common, "--max-new-videos", "1"]), 2)
            receipt = json.loads(
                (root / "output/materialization_receipt.json").read_text()
            )
            self.assertFalse(receipt["complete"])
            self.assertEqual(receipt["plan_rows_completed"], 1)
            self.assertEqual(materializer_main(common), 0)
            receipt = json.loads(
                (root / "output/materialization_receipt.json").read_text()
            )
            self.assertTrue(receipt["complete"])

    def test_scan_only_reads_exact_plan_and_writes_estimates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_parquet(
                root / "parquet/train-00000-of-00001.parquet",
                ["video0", "video1", "video2", "video3", "video4"],
            )
            plan_path = root / "plan.jsonl"
            _write_plan(
                plan_path,
                [_plan_row("video0"), _plan_row("video2"), _plan_row("video4")],
            )
            receipt = run_materialization(
                MaterializationConfig(
                    plan_paths=(plan_path,),
                    output_dir=root / "scan",
                    parquet_pattern=str(root / "parquet/{split}-*.parquet"),
                    resolved_revision="local-test-revision",
                    scan_only=True,
                    minimum_free_disk_bytes=0,
                    retry=RetryPolicy(max_attempts=1),
                )
            )
            self.assertEqual(receipt["status"], "scan_complete")
            self.assertEqual(receipt["plan_rows"], 3)
            self.assertEqual(receipt["plan_rows_located_remotely"], 3)
            self.assertEqual(receipt["plan_rows_accounted_for"], 3)
            self.assertEqual(receipt["plan_rows_completed"], 0)
            self.assertGreater(
                receipt["estimates"]["total_compressed_column_bytes_estimate"], 0
            )
            self.assertTrue(receipt["disk_preflight"]["safe"])
            self.assertFalse((root / "scan/audio").exists())
            self.assertFalse(list((root / "scan").glob("*.parquet")))

    def test_interruption_after_atomic_fragment_resumes_without_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoded = _write_parquet(
                root / "parquet/train-00000-of-00001.parquet",
                ["video0", "video1", "video2", "video3", "video4", "video5"],
            )
            plan_rows = [
                _plan_row("video0"),
                _plan_row("video2"),
                _plan_row("video4", offset=0.40),
            ]
            plan_path = root / "plan.jsonl"
            _write_plan(plan_path, plan_rows)
            config = MaterializationConfig(
                plan_paths=(plan_path,),
                output_dir=root / "output",
                parquet_pattern=str(root / "parquet/{split}-*.parquet"),
                resolved_revision="local-test-revision",
                minimum_free_disk_bytes=0,
                retry=RetryPolicy(max_attempts=1),
            )

            def interrupt(event: str, payload: object) -> None:
                if event == "materialize_row_group_committed":
                    raise RuntimeError("simulated process interruption")

            with self.assertRaisesRegex(RuntimeError, "simulated process interruption"):
                run_materialization(config, checkpoint_hook=interrupt)

            fragments = list((root / "output/manifest_fragments").glob("*.jsonl"))
            self.assertEqual(len(fragments), 1)
            first_rows = [json.loads(line) for line in fragments[0].read_text().splitlines()]
            self.assertEqual([row["video_id"] for row in first_rows], ["video0"])

            receipt = run_materialization(config)
            self.assertTrue(receipt["complete"])
            self.assertEqual(receipt["plan_rows_completed"], 3)
            final_path = root / "output/audioset_strong_plan_manifest.jsonl"
            final_rows = [json.loads(line) for line in final_path.read_text().splitlines()]
            self.assertEqual(
                [row["video_id"] for row in final_rows],
                ["video0", "video2", "video4"],
            )
            self.assertTrue(all(row["format"] == MANIFEST_FORMAT for row in final_rows))
            self.assertTrue(all(len(row["events"]) == 1 for row in final_rows))
            clipped = next(row for row in final_rows if row["video_id"] == "video4")
            self.assertTrue(clipped["events"][0]["annotation_clipped_to_audio_duration"])
            self.assertLessEqual(
                clipped["events"][0]["offset_seconds"], clipped["duration_seconds"]
            )
            for row in final_rows:
                audio_path = Path(row["mixture_path"])
                self.assertTrue(audio_path.is_file())
                self.assertEqual(
                    hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                    hashlib.sha256(encoded[row["video_id"]]).hexdigest(),
                )
                sf.info(audio_path)

    def test_crop_publication_is_last_commit_and_resume_validates_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_parquet(
                root / "parquet/train-00000-of-00001.parquet", ["video0"]
            )
            plan_path = root / "crop-plan.jsonl"
            _write_plan(plan_path, [_crop_plan_row("video0")])
            config = MaterializationConfig(
                plan_paths=(plan_path,),
                output_dir=root / "output",
                parquet_pattern=str(root / "parquet/{split}-*.parquet"),
                resolved_revision="local-test-revision",
                storage_mode="requested_crops",
                minimum_free_disk_bytes=0,
                retry=RetryPolicy(max_attempts=1),
            )

            def interrupt_after_transaction(event: str, payload: object) -> None:
                if event == "materialize_row_group_committed":
                    raise RuntimeError("stop before flat publication")

            with self.assertRaisesRegex(RuntimeError, "flat publication"):
                run_materialization(config, checkpoint_hook=interrupt_after_transaction)
            index = json.loads(
                (root / "output/manifest_index.json").read_text(encoding="utf-8")
            )
            self.assertTrue(index["transactions_complete"])
            self.assertFalse(index["artifacts_published"])
            self.assertFalse(index["complete"])
            self.assertFalse(
                (root / "output/audioset_strong_crop_manifest.jsonl").exists()
            )

            receipt = run_materialization(config)
            self.assertTrue(receipt["complete"])
            index = json.loads(
                (root / "output/manifest_index.json").read_text(encoding="utf-8")
            )
            self.assertTrue(index["artifacts_published"])
            self.assertTrue(index["complete"])
            crop = json.loads(
                (root / "output/audioset_strong_crop_manifest.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            Path(crop["mixture_path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(MaterializationError, "SHA-256 mismatch"):
                run_materialization(config)

    def test_existing_manifest_is_hash_validated_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoded = _write_parquet(
                root / "parquet/train-00000-of-00001.parquet",
                ["video0", "video1"],
            )
            plan_path = root / "plan.jsonl"
            _write_plan(plan_path, [_plan_row("video0")])
            existing_audio = root / "existing/video0.flac"
            existing_audio.parent.mkdir(parents=True)
            existing_audio.write_bytes(encoded["video0"])
            existing_manifest = root / "existing.jsonl"
            existing_manifest.write_text(
                json.dumps(
                    {
                        "video_id": "video0",
                        "hf_split": "train",
                        "mixture_path": str(existing_audio),
                        "audio_sha256": hashlib.sha256(encoded["video0"]).hexdigest(),
                        "source_route": "legacy-test",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            receipt = run_materialization(
                MaterializationConfig(
                    plan_paths=(plan_path,),
                    output_dir=root / "output",
                    parquet_pattern=str(root / "parquet/{split}-*.parquet"),
                    resolved_revision="local-test-revision",
                    existing_manifest_paths=(existing_manifest,),
                    minimum_free_disk_bytes=0,
                    retry=RetryPolicy(max_attempts=1),
                )
            )
            self.assertTrue(receipt["complete"])
            row = json.loads(
                (root / "output/audioset_strong_plan_manifest.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(
                row["source_provenance"]["transport"], "existing_manifest_reuse"
            )
            self.assertEqual(Path(row["mixture_path"]), existing_audio)

            bad_manifest = root / "bad-existing.jsonl"
            bad_manifest.write_text(
                json.dumps(
                    {
                        "video_id": "video0",
                        "mixture_path": str(existing_audio),
                        "audio_sha256": "0" * 64,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(MaterializationError, "SHA-256 mismatch"):
                run_materialization(
                    MaterializationConfig(
                        plan_paths=(plan_path,),
                        output_dir=root / "bad-output",
                        parquet_pattern=str(root / "parquet/{split}-*.parquet"),
                        resolved_revision="local-test-revision",
                        existing_manifest_paths=(bad_manifest,),
                        minimum_free_disk_bytes=0,
                        retry=RetryPolicy(max_attempts=1),
                    )
                )


if __name__ == "__main__":
    unittest.main()
