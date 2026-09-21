from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from mixi_understanding.data.qces_v5_tacos import (
    PACKET_FORMAT,
    TacosAuditError,
    canonical_json_sha256,
    sha256_file,
)
from mixi_understanding.data.qces_v5_tacos_annotation import (
    RESPONSE_FORMAT,
    load_bound_tasks,
    response_path,
    validate_response,
)
from mixi_understanding.data.qces_v5_tacos_audio import (
    AUDIO_RECEIPT_FORMAT,
    CANONICAL_FRAME_COUNT,
    inspect_decoded_audio,
)


PACKET_FP = "a" * 64
AUDIO_FP = "b" * 64
FILE_SHA = "c" * 64
PCM_SHA = "d" * 64


def task() -> dict:
    return {
        "scene_id": "tacos_123",
        "source": {"clip_duration_seconds": 10.0},
        "audio_receipt_sha256": FILE_SHA,
        "audio_receipt_pcm_f32le_sha256": PCM_SHA,
        "proposal_regions": [
            {
                "region_id": f"region_{index:03d}",
                "eligible_proposal": True,
                "onset_seconds": float(index),
            }
            for index in range(4)
        ],
        "question_contract": {
            "absent_anchor_candidates": [
                "Bark",
                "Sneeze",
                "Bell",
                "Whistle",
                "Knock",
                "Splash",
            ]
        },
    }


def response() -> dict:
    return {
        "format": RESPONSE_FORMAT,
        "packet_fingerprint": PACKET_FP,
        "audio_receipt_fingerprint": AUDIO_FP,
        "scene_id": "tacos_123",
        "rater_id": "rater_a01",
        "saved_at_utc": "2026-07-22T00:00:00+00:00",
        "scene_decision": "accept",
        "selected_region_ids": [f"region_{index:03d}" for index in range(4)],
        "regions": [
            {
                "region_id": f"region_{index:03d}",
                "audible": "yes",
                "proposal_caption_accurate": "yes",
                "canonical_event_phrase": f"event {index}",
                "verified_onset_seconds": float(index),
                "verified_offset_seconds": float(index) + 0.5,
                "contamination_rating": "mixed",
                "salience_1_to_5": 3,
            }
            for index in range(4)
        ],
        "event_inventory_complete_for_relations": "yes",
        "adjacent_onset_relations_verified": ["yes", "yes", "yes"],
        "absent_anchor_judgments": [
            {"caption": caption, "confirmed_inaudible": "yes"}
            for caption in ["Bark", "Sneeze", "Bell", "Whistle", "Knock", "Splash"]
        ],
        "full_listen_confirmation": {
            "entire_final_wav_listened": True,
            "file_sha256": FILE_SHA,
            "pcm_f32le_sha256": PCM_SHA,
            "sample_rate_hz": 32_000,
            "num_frames": 320_000,
            "duration_seconds": 10.0,
        },
        "model_outputs_visible": False,
        "notes": "",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


class TacosHumanResponseContractTest(unittest.TestCase):
    def test_valid_two_pass_input_contract_is_strict(self) -> None:
        parsed = validate_response(
            response(),
            task=task(),
            packet_fingerprint=PACKET_FP,
            audio_receipt_fingerprint=AUDIO_FP,
        )
        self.assertEqual(parsed["scene_decision"], "accept")
        self.assertFalse(parsed["model_outputs_visible"])
        self.assertTrue(parsed["full_listen_confirmation"]["entire_final_wav_listened"])

    def test_response_is_bound_to_exact_final_wav_and_pcm(self) -> None:
        for field, value in (
            ("entire_final_wav_listened", False),
            ("file_sha256", "e" * 64),
            ("pcm_f32le_sha256", "f" * 64),
            ("num_frames", 319_999),
            ("duration_seconds", 9.99),
        ):
            bad = response()
            bad["full_listen_confirmation"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                TacosAuditError, "exact final ten-second WAV"
            ):
                validate_response(
                    bad,
                    task=task(),
                    packet_fingerprint=PACKET_FP,
                    audio_receipt_fingerprint=AUDIO_FP,
                )

    def test_accept_rejects_missing_adjacency_or_absent_anchor(self) -> None:
        bad = response()
        bad["adjacent_onset_relations_verified"][1] = "uncertain"
        with self.assertRaisesRegex(TacosAuditError, "adjacency"):
            validate_response(
                bad,
                task=task(),
                packet_fingerprint=PACKET_FP,
                audio_receipt_fingerprint=AUDIO_FP,
            )
        bad = response()
        bad["absent_anchor_judgments"][0]["caption"] = "Piano"
        with self.assertRaisesRegex(TacosAuditError, "packet caption"):
            validate_response(
                bad,
                task=task(),
                packet_fingerprint=PACKET_FP,
                audio_receipt_fingerprint=AUDIO_FP,
            )

        bad = response()
        for judgment in bad["absent_anchor_judgments"]:
            judgment["confirmed_inaudible"] = "no"
        with self.assertRaisesRegex(TacosAuditError, "at least one"):
            validate_response(
                bad,
                task=task(),
                packet_fingerprint=PACKET_FP,
                audio_receipt_fingerprint=AUDIO_FP,
            )

        bad = response()
        bad["absent_anchor_judgments"][1]["confirmed_inaudible"] = "uncertain"
        with self.assertRaisesRegex(TacosAuditError, "definite judgment"):
            validate_response(
                bad,
                task=task(),
                packet_fingerprint=PACKET_FP,
                audio_receipt_fingerprint=AUDIO_FP,
            )

    def test_accept_requires_four_distinct_salient_phrases(self) -> None:
        bad = response()
        bad["regions"][1]["canonical_event_phrase"] = "event 0"
        with self.assertRaisesRegex(TacosAuditError, "distinct"):
            validate_response(
                bad,
                task=task(),
                packet_fingerprint=PACKET_FP,
                audio_receipt_fingerprint=AUDIO_FP,
            )

    def test_accept_requires_each_immutable_proposal_caption_to_be_accurate(
        self,
    ) -> None:
        for verdict in ("no", "uncertain"):
            bad = response()
            bad["regions"][2]["proposal_caption_accurate"] = verdict
            with self.subTest(verdict=verdict), self.assertRaisesRegex(
                TacosAuditError, "proposal caption"
            ):
                validate_response(
                    bad,
                    task=task(),
                    packet_fingerprint=PACKET_FP,
                    audio_receipt_fingerprint=AUDIO_FP,
                )

    def test_rater_cannot_edit_or_supply_the_immutable_query_phrase(self) -> None:
        bad = response()
        bad["regions"][0]["query_event_phrase"] = "rater-controlled shortcut"
        with self.assertRaisesRegex(TacosAuditError, "fields mismatch"):
            validate_response(
                bad,
                task=task(),
                packet_fingerprint=PACKET_FP,
                audio_receipt_fingerprint=AUDIO_FP,
            )
        bad = response()
        bad["regions"][2]["salience_1_to_5"] = 2
        with self.assertRaisesRegex(TacosAuditError, "salience"):
            validate_response(
                bad,
                task=task(),
                packet_fingerprint=PACKET_FP,
                audio_receipt_fingerprint=AUDIO_FP,
            )

    def test_corrected_intervals_are_strictly_inside_final_ten_seconds(self) -> None:
        exact_boundary = response()
        exact_boundary["regions"][3]["verified_offset_seconds"] = 10.0
        validate_response(
            exact_boundary,
            task=task(),
            packet_fingerprint=PACKET_FP,
            audio_receipt_fingerprint=AUDIO_FP,
        )
        for field, value in (
            ("verified_onset_seconds", -0.001),
            ("verified_offset_seconds", 10.001),
        ):
            bad = response()
            bad["regions"][3][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                TacosAuditError, "invalid verified interval"
            ):
                validate_response(
                    bad,
                    task=task(),
                    packet_fingerprint=PACKET_FP,
                    audio_receipt_fingerprint=AUDIO_FP,
                )

    def test_rater_paths_are_independent(self) -> None:
        root = Path("responses")
        left = response_path(root, "rater_a", "tacos_123")
        right = response_path(root, "rater_b", "tacos_123")
        self.assertNotEqual(left, right)
        self.assertEqual(left.name, right.name)


class TacosBoundAnnotationAudioTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.packet_path = self.root / "audit" / "packet.jsonl"
        self.receipt_path = self.root / "audit" / "receipt.jsonl"
        self.wav_path = self.root / "audio" / "tacos_123.wav"
        self.wav_path.parent.mkdir(parents=True)
        time = np.arange(CANONICAL_FRAME_COUNT, dtype=np.float32) / 32_000.0
        samples = (0.05 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)
        sf.write(str(self.wav_path), samples, 32_000, subtype="FLOAT")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _packet(self) -> dict:
        start = 64_000
        selected_ids = [f"region_{index:03d}" for index in range(4)]
        proposals = []
        for index, region_id in enumerate(selected_ids):
            onset = float(index * 2)
            offset = onset + 0.5
            proposals.append(
                {
                    "region_id": region_id,
                    "eligible_proposal": True,
                    "selected_chain_member": True,
                    "truncated_by_benchmark_window": False,
                    "onset_seconds": onset,
                    "offset_seconds": offset,
                    "upstream_clip_onset_seconds": onset + 2.0,
                    "upstream_clip_offset_seconds": offset + 2.0,
                    "caption": f"event {index}",
                }
            )
        # The packet may also expose a non-chain proposal clipped at the final
        # WAV boundary. It remains visible context but can never be selected.
        proposals.append(
            {
                "region_id": "region_truncated",
                "eligible_proposal": False,
                "selected_chain_member": False,
                "truncated_by_benchmark_window": True,
                "onset_seconds": 0.0,
                "offset_seconds": 0.2,
                "upstream_clip_onset_seconds": 1.5,
                "upstream_clip_offset_seconds": 2.2,
                "caption": "boundary context",
            }
        )
        return {
            "schema_version": PACKET_FORMAT,
            "scene_id": "tacos_123",
            "selection_partition": "real_dev",
            "selection_tier": "core",
            "selection_ordinal": 0,
            "audio": {
                "archive_member": "123.mp3",
                "benchmark_window_start_sample_32k": start,
                "benchmark_window_end_sample_32k": start + CANONICAL_FRAME_COUNT,
                "benchmark_sample_rate_hz": 32_000,
                "local_path": None,
                "local_sha256": None,
                "verified_audio_properties": None,
            },
            "source": {
                "filename": "123.mp3",
                "freesound_id": "123",
                "custom_qces_partition": "real_dev",
                "clip_duration_seconds": 10.0,
                "benchmark_window_interval_in_upstream_clip_seconds": [2.0, 12.0],
                "subclass": "domestic sounds",
                "audio_license_spdx": "CC0-1.0",
            },
            "benchmark_window": {
                "start_sample": start,
                "end_sample": start + CANONICAL_FRAME_COUNT,
                "sample_rate_hz": 32_000,
                "duration_seconds": 10.0,
                "selected_before_human_annotation": True,
                "human_labels_used": False,
                "qces_method_outputs_used": False,
                "runtime_or_question_specific": False,
            },
            "proposal_regions": proposals,
            "suggested_distinct_onset_chain": selected_ids,
            "proposal_diagnostics": {
                "usable_region_count_↑": 4,
                "overlap_pair_count_↑": 0,
            },
            "question_contract": {
                "absent_anchor_candidates": [
                    "Bark",
                    "Sneeze",
                    "Bell",
                    "Whistle",
                    "Knock",
                    "Splash",
                ],
                "absent_anchor_cross_scene_support": [
                    {
                        "caption": caption,
                        "selection_partition": "real_dev",
                        "support_tier": "core",
                        "cross_scene_support_scene_ids": [f"tacos_{1000 + index}"],
                        "cross_scene_support_count_↑": 1,
                        "primary_support_scene_id": f"tacos_{1000 + index}",
                    }
                    for index, caption in enumerate(
                        ["Bark", "Sneeze", "Bell", "Whistle", "Knock", "Splash"]
                    )
                ],
                "absent_anchor_pool_source": "other_selected_chain_captions",
                "absent_anchor_candidates_are_exact_raw_positive_captions": True,
                "qces_label_selection_used_for_absent_anchor_selection": False,
            },
        }

    def _write_packet_and_receipt(self, packet: dict | None = None) -> dict:
        packet = self._packet() if packet is None else packet
        _write_jsonl(self.packet_path, [packet])
        properties = dict(inspect_decoded_audio(self.wav_path))
        local_sha = sha256_file(self.wav_path)
        verified = {
            **properties,
            "file_sha256": local_sha,
            "file_size_bytes": self.wav_path.stat().st_size,
        }
        crop_properties = {
            key: properties[key]
            for key in (
                "sample_rate_hz",
                "channels",
                "decoded_frames",
                "decoded_duration_seconds",
                "peak_abs_↑",
                "rms_↑",
                "nonfinite_sample_count_↓",
                "pcm_f32le_sha256",
            )
        }
        start = packet["audio"]["benchmark_window_start_sample_32k"]
        receipt = {
            "format": AUDIO_RECEIPT_FORMAT,
            "packet_fingerprint": canonical_json_sha256([packet]),
            "packet_file_sha256": sha256_file(self.packet_path),
            "scene_id": "tacos_123",
            "selection_partition": packet["selection_partition"],
            "selection_tier": packet["selection_tier"],
            "freesound_id": "123",
            "filename": "123.mp3",
            "archive_member": "123.mp3",
            "crop_recipe": {
                "start_sample_inclusive": start,
                "end_sample_exclusive": start + CANONICAL_FRAME_COUNT,
                "output_frames": CANONICAL_FRAME_COUNT,
                "sample_rate_hz": 32_000,
                "channels": 1,
                "output_container": "WAV",
                "output_subtype": "FLOAT",
                "resampling_applied": False,
                "downmixing_applied": False,
                "padding_samples_↓": 0,
                "human_labels_used": False,
                "qces_method_outputs_used": False,
            },
            "proposal_containment": {
                "selected_semantic_chain_regions_inside_window_↑": 4,
                "selected_semantic_chain_regions_outside_window_↓": 0,
                "selected_semantic_chain_regions_truncated_↓": 0,
            },
            "crop_decoded_audio": crop_properties,
            "local_path": "audio/tacos_123.wav",
            "local_size_bytes": self.wav_path.stat().st_size,
            "local_sha256": local_sha,
            "local_pcm_f32le_sha256": properties["pcm_f32le_sha256"],
            "local_verified_audio_properties": verified,
        }
        _write_jsonl(self.receipt_path, [receipt])
        return receipt

    def test_loader_exposes_only_hash_verified_canonical_wav(self) -> None:
        receipt = self._write_packet_and_receipt()
        packet_fp, receipt_fp, tasks = load_bound_tasks(
            packet_path=self.packet_path,
            audio_receipt_path=self.receipt_path,
            project_root=self.root,
        )
        self.assertEqual(packet_fp, receipt["packet_fingerprint"])
        self.assertEqual(receipt_fp, canonical_json_sha256([receipt]))
        self.assertEqual(len(tasks), 1)
        loaded = tasks[0]
        self.assertEqual(Path(loaded["resolved_audio_path"]), self.wav_path)
        self.assertEqual(loaded["annotation_audio"]["num_frames"], 320_000)
        self.assertEqual(loaded["annotation_audio"]["duration_seconds"], 10.0)
        self.assertEqual(
            loaded["annotation_audio"]["pcm_f32le_sha256"],
            receipt["local_pcm_f32le_sha256"],
        )

    def test_loader_rejects_audio_tampering(self) -> None:
        self._write_packet_and_receipt()
        with self.wav_path.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(TacosAuditError, "hash mismatch"):
            load_bound_tasks(
                packet_path=self.packet_path,
                audio_receipt_path=self.receipt_path,
                project_root=self.root,
            )

    def test_loader_rejects_packet_or_crop_coordinate_tampering(self) -> None:
        receipt = self._write_packet_and_receipt()
        receipt["packet_file_sha256"] = "0" * 64
        _write_jsonl(self.receipt_path, [receipt])
        with self.assertRaisesRegex(TacosAuditError, "packet file bytes"):
            load_bound_tasks(
                packet_path=self.packet_path,
                audio_receipt_path=self.receipt_path,
                project_root=self.root,
            )

        packet = self._packet()
        packet["proposal_regions"][0]["onset_seconds"] = -0.01
        self._write_packet_and_receipt(packet)
        with self.assertRaisesRegex(TacosAuditError, "outside the final"):
            load_bound_tasks(
                packet_path=self.packet_path,
                audio_receipt_path=self.receipt_path,
                project_root=self.root,
            )

    def test_loader_rejects_noncanonical_or_bypass_audio(self) -> None:
        packet = self._packet()
        packet["audio"]["local_path"] = "original/123.mp3"
        self._write_packet_and_receipt(packet)
        with self.assertRaisesRegex(TacosAuditError, "bypass audio path"):
            load_bound_tasks(
                packet_path=self.packet_path,
                audio_receipt_path=self.receipt_path,
                project_root=self.root,
            )


if __name__ == "__main__":
    unittest.main()
