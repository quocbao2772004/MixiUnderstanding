from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from mixi_understanding.qces.audiosep_source_cleaner import (
    FORMAT,
    INPUT_FORMAT,
    QualityGate,
    SourceBankConfig,
    SourceCleanerError,
    apply_quality_gate,
    build_source_bank,
    deterministic_resample_mono,
    estimate_audible_interval,
    sha256_file,
)
from mixi_understanding.qces.clean_evidence_scenes import load_source_bank


SAMPLE_RATE = 8_000


class StubBackend:
    sample_rate = SAMPLE_RATE

    def __init__(self, *, silent: bool = False) -> None:
        self.silent = silent
        self.separate_calls = 0
        self.seen_prompts: list[str] = []
        self.seen_waveform_lengths: list[int] = []

    def identity(self) -> Mapping[str, Any]:
        return {"backend": "deterministic-test-stub-v1", "silent": self.silent}

    def separate(
        self, waveforms: Sequence[np.ndarray], prompts: Sequence[str]
    ) -> Sequence[np.ndarray]:
        self.separate_calls += 1
        self.seen_prompts.extend(prompts)
        self.seen_waveform_lengths.extend(int(len(value)) for value in waveforms)
        output = []
        for waveform, prompt in zip(waveforms, prompts):
            if self.silent:
                output.append(np.zeros_like(waveform, dtype=np.float32))
            else:
                scale = 0.78 if prompt.startswith("the sound of ") else 0.80
                output.append(np.asarray(waveform, dtype=np.float32) * scale)
        return output

    def embed_audio(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        vectors = []
        for waveform in waveforms:
            rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
            vectors.append([1.0, 0.0, 0.0] if rms > 0.15 else [0.0, 0.0, 1.0])
        return np.asarray(vectors, dtype=np.float32)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            [
                [0.0, 1.0, 0.0]
                if text == "Other event"
                else [1.0, 0.0, 0.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def _write_audio(
    path: Path,
    *,
    frequency: float = 440.0,
    sample_rate: int = SAMPLE_RATE,
    duration_seconds: float = 1.0,
) -> str:
    time = np.arange(
        int(round(sample_rate * duration_seconds)), dtype=np.float32
    ) / sample_rate
    waveform = 0.5 * np.sin(2.0 * math.pi * frequency * time)
    sf.write(path, waveform, sample_rate, format="FLAC", subtype="PCM_24")
    return sha256_file(path)


def _row(
    *,
    item: str,
    source_video: str,
    path: Path,
    split: str,
    metadata_split: str,
    tier: int,
    label: str = "Target_event",
) -> dict[str, Any]:
    audio_info = sf.info(path)
    return {
        "format": INPUT_FORMAT,
        "scene_id": f"scene-{item}",
        "materialization_item_id": item,
        "source_video_id": source_video,
        "video_id": source_video,
        "split_lock": split,
        "metadata_split": metadata_split,
        "hf_split": "test" if metadata_split == "eval" else "train",
        "mixture_path": str(path.resolve()),
        "audio_sha256": sha256_file(path),
        "sample_rate": int(audio_info.samplerate),
        "duration_seconds": float(audio_info.duration),
        "source_crop_start_seconds": 2.0,
        "source_crop_end_seconds": 3.0,
        "coverage_label": label,
        "coverage_mid": "/m/target",
        "coverage_rank": 1,
        "ambiguity_tier": tier,
        "ambiguity_tier_name": f"tier_{tier}",
        "events": [
            {
                "label": label,
                "display_name": "Target event",
                "audioset_mid": "/m/target",
                "onset_seconds": 0.0,
                "offset_seconds": 1.0,
                "selected_ontology_label": True,
            },
            {
                "label": "Other_event",
                "display_name": "Other event",
                "audioset_mid": "/m/other",
                "onset_seconds": 0.0,
                "offset_seconds": 0.2,
                "selected_ontology_label": False,
            },
        ],
    }


def _write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class AudioSepSourceCleanerTests(unittest.TestCase):
    def test_native_44100_and_48000_crops_are_content_bound_resampled_to_32000(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for sample_rate in (44_100, 48_000):
                audio = root / f"crop-{sample_rate}.flac"
                _write_audio(audio, sample_rate=sample_rate)
                rows.append(
                    _row(
                        item=f"native-{sample_rate}",
                        source_video=f"video-{sample_rate}",
                        path=audio,
                        split="train",
                        metadata_split="train",
                        tier=0,
                    )
                )
            manifest = root / "requested.jsonl"
            _write_manifest(manifest, rows)
            backend = StubBackend()
            backend.sample_rate = 32_000
            config = SourceBankConfig(
                manifest_paths=(manifest,),
                output_dir=root / "bank",
                batch_size=2,
                target_train_per_class=2,
                target_eval_per_class=1,
            )
            receipt = build_source_bank(config, backend)
            self.assertEqual(receipt["accepted_items"], 2)
            self.assertEqual(receipt["input_sample_rate_counts"], {"44100": 1, "48000": 1})
            self.assertEqual(receipt["resampling_applied_items"], 2)
            self.assertTrue(
                receipt["invariants"]["all_separator_inputs_use_backend_sample_rate"]
            )
            self.assertLessEqual(
                receipt["maximum_absolute_resample_duration_error_seconds"],
                1.0 / 32_000,
            )
            self.assertTrue(backend.seen_waveform_lengths)
            self.assertEqual(set(backend.seen_waveform_lengths), {32_000})
            cleaned = [
                json.loads(line)
                for line in (root / "bank/source_bank.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(cleaned), 2)
            for row in cleaned:
                self.assertIn(row["input_sample_rate"], {44_100, 48_000})
                self.assertEqual(row["separator_input_sample_rate"], 32_000)
                self.assertEqual(row["separator_input_num_samples"], 32_000)
                self.assertEqual(len(row["separator_input_f32le_sha256"]), 64)
                self.assertEqual(len(row["input_decoded_mono_f32le_sha256"]), 64)
                self.assertTrue(row["resampling_applied"])
                self.assertEqual(row["active_onset_seconds"], 0.0)
                self.assertEqual(row["active_offset_seconds"], 1.0)
                self.assertEqual(sf.info(row["stem_path"]).samplerate, 32_000)
            calls = backend.separate_calls
            resumed = build_source_bank(config, backend)
            self.assertEqual(backend.separate_calls, calls)
            self.assertEqual(resumed["resumed_items"], 2)

    def test_polyphase_length_rule_preserves_seconds_for_noninteger_duration(
        self,
    ) -> None:
        source_rate = 44_100
        waveform = np.linspace(-0.2, 0.2, 44_099, dtype=np.float32)
        output, audit = deterministic_resample_mono(
            waveform,
            source_sample_rate=source_rate,
            target_sample_rate=32_000,
        )
        self.assertEqual(output.shape, (31_999,))
        self.assertEqual(audit["up_factor"], 320)
        self.assertEqual(audit["down_factor"], 441)
        self.assertLessEqual(abs(audit["duration_error_seconds"]), 1.0 / 32_000)

    def test_audible_interval_trims_silent_annotation_edges(self) -> None:
        waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)
        time = np.arange(int(0.50 * SAMPLE_RATE), dtype=np.float32) / SAMPLE_RATE
        waveform[int(0.20 * SAMPLE_RATE) : int(0.70 * SAMPLE_RATE)] = (
            0.4 * np.sin(2.0 * math.pi * 440.0 * time)
        )
        onset, offset = estimate_audible_interval(
            waveform, SAMPLE_RATE, (0.0, 1.0)
        )
        self.assertGreaterEqual(onset, 0.16)
        self.assertLessEqual(onset, 0.21)
        self.assertGreaterEqual(offset, 0.69)
        self.assertLessEqual(offset, 0.75)

    def test_silver_gate_cannot_promote_ambiguous_tier_two_or_three(self) -> None:
        # Passes the fixed silver thresholds but intentionally misses gold.
        metrics = {
            "stem_rms_dbfs": -50.0,
            "stem_peak_dbfs": -40.0,
            "retained_energy_ratio": 0.002,
            "target_text_similarity": 0.10,
            "target_residual_margin": -0.01,
            "target_other_margin": -0.02,
            "paraphrase_agreement": 0.60,
            "maximum_clipped_sample_fraction": 0.0,
        }
        tier_one = apply_quality_gate(metrics, ambiguity_tier=1, gate=QualityGate())
        tier_two = apply_quality_gate(metrics, ambiguity_tier=2, gate=QualityGate())
        self.assertEqual(tier_one["acceptance_tier"], "silver")
        self.assertEqual(tier_two["acceptance_tier"], "rejected")
        self.assertTrue(tier_two["silver_disallowed_by_upstream_ambiguity"])

    def test_accepts_writes_hashes_preserves_split_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "crop.flac"
            _write_audio(audio)
            manifest = root / "requested.jsonl"
            # Deliberately put tier 2 first; the cleaner must process tier 0 first.
            _write_manifest(
                manifest,
                [
                    _row(
                        item="tier2",
                        source_video="video2",
                        path=audio,
                        split="train",
                        metadata_split="train",
                        tier=2,
                    ),
                    _row(
                        item="tier0",
                        source_video="video0",
                        path=audio,
                        split="train",
                        metadata_split="train",
                        tier=0,
                    ),
                ],
            )
            output = root / "bank"
            backend = StubBackend()
            config = SourceBankConfig(
                manifest_paths=(manifest,),
                output_dir=output,
                batch_size=1,
                shard_size=1,
                target_train_per_class=2,
                target_eval_per_class=1,
                store_residual=True,
            )
            receipt = build_source_bank(config, backend)
            self.assertEqual(receipt["format"], FORMAT)
            self.assertEqual(receipt["accepted_items"], 2)
            self.assertEqual(receipt["processed_items_this_run"], 2)
            self.assertEqual(backend.seen_prompts[0], "Target event")
            self.assertEqual(receipt["input_ambiguity_tier_counts"], {"0": 1, "2": 1})
            rows = [
                json.loads(line)
                for line in (output / "source_bank.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["item_id"] for row in rows], ["tier0", "tier2"])
            for row in rows:
                self.assertEqual(row["source_video_id"], f"video{row['item_id'][-1]}")
                self.assertEqual(row["source_split"], "train")
                self.assertEqual(len(row["all_intersecting_strong_annotations"]), 2)
                for path_key, hash_key in (
                    ("stem_path", "stem_sha256"),
                    ("residual_path", "residual_sha256"),
                ):
                    artifact = Path(row[path_key])
                    self.assertTrue(artifact.is_file())
                    self.assertEqual(sha256_file(artifact), row[hash_key])
                self.assertEqual(row["source_id"], row["item_id"])
                self.assertEqual(row["label"], "Target_event")
                self.assertEqual(row["audio_path"], row["stem_path"])
                self.assertEqual(row["source_sha256"], row["stem_sha256"])
                self.assertEqual(row["active_onset_seconds"], 0.0)
                self.assertEqual(row["active_offset_seconds"], 1.0)
                self.assertTrue(row["cleanliness_passed"])
                self.assertTrue(row["audibility_passed"])
            # This is the real downstream boundary: an accepted cleaner
            # output must be loadable by the scene builder without an adapter.
            clean_sources = load_source_bank(
                output / "source_bank.jsonl", require_audio_file=True
            )
            self.assertEqual(len(clean_sources), 2)
            self.assertTrue(all(source.label == "Target_event" for source in clean_sources))
            calls = backend.separate_calls
            second = build_source_bank(config, backend)
            self.assertEqual(backend.separate_calls, calls)
            self.assertEqual(second["resumed_items"], 2)
            self.assertEqual(second["processed_items_this_run"], 0)

    def test_default_stores_only_downstream_stem_and_still_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "crop.flac"
            _write_audio(audio)
            manifest = root / "requested.jsonl"
            _write_manifest(
                manifest,
                [
                    _row(
                        item="stem-only",
                        source_video="video-stem-only",
                        path=audio,
                        split="train",
                        metadata_split="train",
                        tier=0,
                    )
                ],
            )
            output = root / "bank"
            backend = StubBackend()
            config = SourceBankConfig(
                manifest_paths=(manifest,),
                output_dir=output,
                target_train_per_class=1,
                target_eval_per_class=1,
            )
            receipt = build_source_bank(config, backend)
            row = json.loads((output / "source_bank.jsonl").read_text())
            self.assertFalse(receipt["store_residual"])
            self.assertTrue(Path(row["stem_path"]).is_file())
            self.assertEqual(row["residual_path"], "")
            self.assertEqual(row["residual_sha256"], "")
            self.assertFalse(row["provenance"]["residual_artifact_stored"])
            calls = backend.separate_calls
            resumed = build_source_bank(config, backend)
            self.assertEqual(backend.separate_calls, calls)
            self.assertEqual(resumed["resumed_items"], 1)

    def test_repeated_label_uses_requested_source_interval_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "crop.flac"
            _write_audio(audio)
            row = _row(
                item="repeat-second",
                source_video="repeat-video",
                path=audio,
                split="train",
                metadata_split="train",
                tier=0,
            )
            row["crop_request"] = {
                "event_onset_seconds": 2.60,
                "event_offset_seconds": 2.90,
            }
            row["events"] = [
                {
                    "label": "Target_event",
                    "display_name": "Target event",
                    "audioset_mid": "/m/target",
                    "onset_seconds": 0.10,
                    "offset_seconds": 0.30,
                    "source_onset_seconds": 2.10,
                    "source_offset_seconds": 2.30,
                    "selected_ontology_label": True,
                },
                {
                    "label": "Target_event",
                    "display_name": "Target event",
                    "audioset_mid": "/m/target",
                    "onset_seconds": 0.60,
                    "offset_seconds": 0.90,
                    "source_onset_seconds": 2.60,
                    "source_offset_seconds": 2.90,
                    "selected_ontology_label": True,
                },
            ]
            manifest = root / "requested.jsonl"
            _write_manifest(manifest, [row])
            output = root / "bank"
            build_source_bank(
                SourceBankConfig(
                    manifest_paths=(manifest,),
                    output_dir=output,
                    target_train_per_class=1,
                    target_eval_per_class=1,
                ),
                StubBackend(),
            )
            cleaned = json.loads((output / "source_bank.jsonl").read_text())
            self.assertAlmostEqual(cleaned["active_onset_seconds"], 0.60)
            self.assertAlmostEqual(cleaned["active_offset_seconds"], 0.90)

    def test_rejects_silent_separator_output_without_emitting_source_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "crop.flac"
            _write_audio(audio)
            manifest = root / "requested.jsonl"
            _write_manifest(
                manifest,
                [
                    _row(
                        item="reject",
                        source_video="video-reject",
                        path=audio,
                        split="train",
                        metadata_split="train",
                        tier=0,
                    )
                ],
            )
            output = root / "bank"
            receipt = build_source_bank(
                SourceBankConfig(
                    manifest_paths=(manifest,),
                    output_dir=output,
                    target_train_per_class=1,
                    target_eval_per_class=1,
                ),
                StubBackend(silent=True),
            )
            self.assertEqual(receipt["accepted_items"], 0)
            self.assertEqual(receipt["rejected_items"], 1)
            self.assertEqual((output / "source_bank.jsonl").read_text(), "")
            audit = json.loads((output / "quality_audit.jsonl").read_text())
            self.assertFalse(audit["accepted"])
            self.assertEqual(audit["stem_path"], "")
            self.assertIn("stem_rms_dbfs", audit["quality_gate"]["gold_failures"])

    def test_rejects_source_video_cross_split_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "crop.flac"
            _write_audio(audio)
            manifest = root / "requested.jsonl"
            _write_manifest(
                manifest,
                [
                    _row(
                        item="train-item",
                        source_video="shared-video",
                        path=audio,
                        split="train",
                        metadata_split="train",
                        tier=0,
                    ),
                    _row(
                        item="test-item",
                        source_video="shared-video",
                        path=audio,
                        split="test",
                        metadata_split="eval",
                        tier=0,
                    ),
                ],
            )
            with self.assertRaisesRegex(SourceCleanerError, "multiple splits"):
                build_source_bank(
                    SourceBankConfig(manifest_paths=(manifest,), output_dir=root / "bank"),
                    StubBackend(),
                )

    def test_per_class_quota_counts_unique_source_video_in_train_and_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_audio = root / "train.flac"
            eval_audio = root / "eval.flac"
            _write_audio(train_audio, frequency=330.0)
            _write_audio(eval_audio, frequency=550.0)
            manifest = root / "requested.jsonl"
            _write_manifest(
                manifest,
                [
                    _row(
                        item="train",
                        source_video="train-video",
                        path=train_audio,
                        split="train",
                        metadata_split="train",
                        tier=0,
                    ),
                    _row(
                        item="eval",
                        source_video="eval-video",
                        path=eval_audio,
                        split="test",
                        metadata_split="eval",
                        tier=1,
                    ),
                ],
            )
            receipt = build_source_bank(
                SourceBankConfig(
                    manifest_paths=(manifest,),
                    output_dir=root / "bank",
                    target_train_per_class=1,
                    target_eval_per_class=1,
                ),
                StubBackend(),
            )
            quota = receipt["quota_report"]
            self.assertEqual(quota["classes_quota_ready"], 1)
            row = quota["per_class"][0]
            self.assertEqual(row["accepted_train_unique_videos"], 1)
            self.assertEqual(row["accepted_eval_unique_videos"], 1)
            self.assertTrue(row["quota_ready"])
            self.assertTrue(
                receipt["invariants"]["upstream_split_preserved_and_never_reassigned"]
            )


if __name__ == "__main__":
    unittest.main()
