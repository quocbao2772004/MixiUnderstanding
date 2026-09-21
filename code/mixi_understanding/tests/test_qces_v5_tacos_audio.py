from __future__ import annotations

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from mixi_understanding.data.qces_v5_tacos import (
    DOI,
    PACKET_FORMAT,
    PLAN_FORMAT,
    RECORD_ID,
    TacosAuditError,
    canonical_json_sha256,
    sha256_file,
)
from mixi_understanding.data.qces_v5_tacos_audio import (
    AUDIO_COMPLIANCE_FORMAT,
    AUDIO_RECEIPT_FORMAT,
    BENCHMARK_SAMPLE_RATE,
    CANONICAL_FRAME_COUNT,
    extract_tacos_subset,
    inspect_decoded_audio,
)


class TacosCanonicalAudioExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        if "MP3" not in sf.available_formats():
            self.skipTest("libsndfile build cannot encode the MP3 test fixture")
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.packet_path = self.root / "packet.jsonl"
        self.plan_path = self.root / "plan.json"
        self.archive_path = self.root / "audio.zip"
        self.output_dir = self.root / "derived"
        self.source_path = self.root / "1234.mp3"

        frames = BENCHMARK_SAMPLE_RATE * 12
        time = np.arange(frames, dtype=np.float32) / BENCHMARK_SAMPLE_RATE
        waveform = (
            0.20 * np.sin(2.0 * np.pi * 233.0 * time)
            + 0.05 * np.sin(2.0 * np.pi * 719.0 * time)
        ).astype(np.float32)
        sf.write(
            str(self.source_path),
            waveform,
            BENCHMARK_SAMPLE_RATE,
            format="MP3",
            subtype="MPEG_LAYER_III",
        )
        decoded, decoded_rate = sf.read(
            str(self.source_path), dtype="float32", always_2d=False
        )
        self.assertEqual(decoded_rate, BENCHMARK_SAMPLE_RATE)
        self.decoded_source = np.asarray(decoded, dtype=np.float32)
        self.start = 19_337
        self.end = self.start + CANONICAL_FRAME_COUNT
        self.assertLessEqual(self.end, self.decoded_source.size)

        with zipfile.ZipFile(
            self.archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.write(self.source_path, arcname="1234.mp3")
        with zipfile.ZipFile(self.archive_path, "r") as archive:
            info = archive.getinfo("1234.mp3")
        self.archive_receipt = {
            "path": str(self.archive_path.resolve()),
            "size_bytes": self.archive_path.stat().st_size,
            "publisher_md5": "a" * 32,
            "sha256": sha256_file(self.archive_path),
            "archive_gate_passed": True,
            "unsafe_entries_↓": 0,
            "symlink_entries_↓": 0,
            "selected_members_verified_↑": 1,
            "selected_members": {
                "1234.mp3": {
                    "archive_member": "1234.mp3",
                    "uncompressed_size_bytes": info.file_size,
                    "compressed_size_bytes": info.compress_size,
                    "crc32": f"{info.CRC:08x}",
                }
            },
        }
        self.packet = self._packet()
        self.plan = self._plan(self.packet)
        self._write_inputs()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _packet(self) -> list[dict[str, object]]:
        start_seconds = self.start / BENCHMARK_SAMPLE_RATE
        selected = [f"region_{index}" for index in range(4)]
        proposals = []
        for index, region_id in enumerate(selected):
            onset = 0.5 + 1.5 * index
            offset = onset + 0.4
            proposals.append(
                {
                    "region_id": region_id,
                    "caption": f"event {index}",
                    "onset_seconds": onset,
                    "offset_seconds": offset,
                    "upstream_clip_onset_seconds": start_seconds + onset,
                    "upstream_clip_offset_seconds": start_seconds + offset,
                    "truncated_by_benchmark_window": False,
                    "selected_chain_member": True,
                    "eligible_proposal": True,
                    "proposal_exclusion_reasons": [],
                }
            )
        return [
            {
                "schema_version": PACKET_FORMAT,
                "scene_id": "tacos_1234",
                "selection_partition": "real_dev",
                "selection_tier": "core",
                "selection_ordinal": 0,
                "audio": {
                    "archive_member": "1234.mp3",
                    "benchmark_window_start_sample_32k": self.start,
                    "benchmark_window_end_sample_32k": self.end,
                    "benchmark_sample_rate_hz": BENCHMARK_SAMPLE_RATE,
                    "local_path": None,
                    "local_sha256": None,
                    "verified_audio_properties": None,
                },
                "source": {
                    "dataset": "TACOS",
                    "filename": "1234.mp3",
                    "freesound_id": "1234",
                    "upstream_clip_duration_seconds": self.decoded_source.size
                    / BENCHMARK_SAMPLE_RATE,
                    "benchmark_window_interval_in_upstream_clip_seconds": [
                        self.start / BENCHMARK_SAMPLE_RATE,
                        self.end / BENCHMARK_SAMPLE_RATE,
                    ],
                    "clip_duration_seconds": 10.0,
                },
                "benchmark_window": {
                    "duration_seconds": 10.0,
                    "sample_rate_hz": BENCHMARK_SAMPLE_RATE,
                    "start_sample": self.start,
                    "end_sample": self.end,
                    "start_seconds_in_upstream_clip": self.start
                    / BENCHMARK_SAMPLE_RATE,
                    "end_seconds_in_upstream_clip": self.end / BENCHMARK_SAMPLE_RATE,
                    "valid_start_sample_interval_inclusive": [
                        self.start - 100,
                        self.start + 100,
                    ],
                    "construction_is_proposal_informed": True,
                    "runtime_or_question_specific": False,
                    "selected_before_human_annotation": True,
                    "human_labels_used": False,
                    "qces_method_outputs_used": False,
                },
                "proposal_regions": proposals,
                "suggested_distinct_onset_chain": selected,
                "semantic_chain": {"region_ids": selected},
            }
        ]

    @staticmethod
    def _plan(packet: list[dict[str, object]]) -> dict[str, object]:
        return {
            "format": PLAN_FORMAT,
            "record": {"record_id": RECORD_ID, "doi": DOI},
            "selection": {
                "benchmark_window_seconds": 10.0,
                "benchmark_window_sample_rate_hz": BENCHMARK_SAMPLE_RATE,
                "benchmark_window_start_rule": "sample-quantized hash-uniform",
            },
            "input_fingerprint": "b" * 64,
            "packet_fingerprint": canonical_json_sha256(packet),
            "paper_result_eligible": False,
        }

    def _write_inputs(self) -> None:
        self.packet_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.packet),
            encoding="utf-8",
        )
        self.plan_path.write_text(
            json.dumps(self.plan, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _extract(
        self,
        *,
        overwrite_audio: bool = False,
        archive_receipt: dict[str, object] | None = None,
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        receipt = archive_receipt or self.archive_receipt
        with patch(
            "mixi_understanding.data.qces_v5_tacos_audio.verify_audio_archive",
            return_value=receipt,
        ) as verifier:
            result = extract_tacos_subset(
                packet_path=self.packet_path,
                plan_path=self.plan_path,
                archive_path=self.archive_path,
                output_dir=self.output_dir,
                project_root=self.root,
                overwrite_audio=overwrite_audio,
                maximum_duration_error_seconds=0.10,
            )
        verifier.assert_called_once_with(self.archive_path, ["1234.mp3"])
        return result

    def test_derives_exact_float_wav_and_binds_every_input(self) -> None:
        compliance, receipts = self._extract()
        self.assertEqual(compliance["format"], AUDIO_COMPLIANCE_FORMAT)
        self.assertTrue(compliance["audio_extraction_gate_passed"])
        self.assertFalse(compliance["submission_real_data_gate_passed"])
        self.assertTrue(all(key.endswith(("↑", "↓")) for key in compliance["metrics"]))
        self.assertEqual(len(receipts), 1)
        receipt = receipts[0]
        self.assertEqual(receipt["format"], AUDIO_RECEIPT_FORMAT)
        self.assertEqual(receipt["packet_fingerprint"], self.plan["packet_fingerprint"])
        self.assertEqual(
            receipt["source_plan_fingerprint"], canonical_json_sha256(self.plan)
        )
        self.assertEqual(receipt["archive_sha256"], self.archive_receipt["sha256"])
        self.assertEqual(receipt["crop_recipe"]["start_sample_inclusive"], self.start)
        self.assertEqual(receipt["crop_recipe"]["end_sample_exclusive"], self.end)
        self.assertEqual(receipt["crop_recipe"]["padding_samples_↓"], 0)
        self.assertFalse(receipt["crop_recipe"]["resampling_applied"])
        self.assertEqual(
            receipt["proposal_containment"][
                "selected_semantic_chain_regions_inside_window_↑"
            ],
            4,
        )

        derived = self.root / receipt["local_path"]
        self.assertEqual(derived.name, "tacos_1234.wav")
        self.assertFalse((self.output_dir / "1234.mp3").exists())
        self.assertEqual(derived.stat().st_size, 56 + 4 * CANONICAL_FRAME_COUNT)
        self.assertNotIn(b"PEAK", derived.read_bytes()[:256])
        properties = inspect_decoded_audio(derived)
        self.assertEqual(properties["decoded_frames"], CANONICAL_FRAME_COUNT)
        actual, rate = sf.read(str(derived), dtype="float32", always_2d=False)
        self.assertEqual(rate, BENCHMARK_SAMPLE_RATE)
        expected = self.decoded_source[self.start : self.end]
        self.assertTrue(
            np.array_equal(np.asarray(actual).view(np.uint32), expected.view(np.uint32))
        )
        self.assertEqual(receipt["local_sha256"], sha256_file(derived))
        self.assertNotEqual(
            receipt["source_decoded_pcm_f32le_sha256"],
            receipt["local_pcm_f32le_sha256"],
        )

    def test_resume_requires_exact_deterministic_wav_bytes(self) -> None:
        first_compliance, first_receipts = self._extract()
        first_hash = first_receipts[0]["local_sha256"]
        second_compliance, second_receipts = self._extract()
        self.assertEqual(second_receipts[0]["local_sha256"], first_hash)
        self.assertEqual(
            second_compliance["metrics"]["resume_reused_exact_derived_wav_files_↑"],
            1,
        )
        self.assertEqual(
            second_compliance["metrics"]["newly_written_derived_wav_files_↑"],
            0,
        )
        self.assertEqual(
            first_compliance["audio_receipt_fingerprint"],
            second_compliance["audio_receipt_fingerprint"],
        )

        target = self.output_dir / "tacos_1234.wav"
        payload = bytearray(target.read_bytes())
        payload[-1] ^= 1
        target.write_bytes(payload)
        with self.assertRaisesRegex(TacosAuditError, "existing derived WAV mismatch"):
            self._extract()
        _, repaired = self._extract(overwrite_audio=True)
        self.assertEqual(repaired[0]["local_sha256"], first_hash)

    def test_archive_member_crc_is_checked_during_streaming(self) -> None:
        bad = copy.deepcopy(self.archive_receipt)
        bad["selected_members"]["1234.mp3"]["crc32"] = "00000000"
        with self.assertRaisesRegex(TacosAuditError, "archive member CRC mismatch"):
            self._extract(archive_receipt=bad)
        self.assertFalse(any(self.output_dir.glob("*.wav")))

    def test_crop_outside_fully_decoded_source_fails_without_padding(self) -> None:
        self.start = int(self.decoded_source.size) - CANONICAL_FRAME_COUNT + 1
        self.end = self.start + CANONICAL_FRAME_COUNT
        self.packet = self._packet()
        self.plan = self._plan(self.packet)
        self._write_inputs()
        with self.assertRaisesRegex(TacosAuditError, "padding is prohibited"):
            self._extract()
        self.assertFalse(any(self.output_dir.glob("*.wav")))

    def test_packet_window_duplicates_must_agree(self) -> None:
        self.packet[0]["benchmark_window"]["start_sample"] += 1
        self.plan = self._plan(self.packet)
        self._write_inputs()
        with self.assertRaisesRegex(TacosAuditError, "window duplicates disagree"):
            self._extract()


if __name__ == "__main__":
    unittest.main()
