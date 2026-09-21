from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from mixi_understanding.scripts import (
    audit_qces_supported_beats_microfit_audio_preflight as preflight,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_flac(path: Path, *, index: int, frames: int = 1280) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(frames, dtype=np.float64) / preflight.EXPECTED_SAMPLE_RATE
    waveform = 0.08 * np.sin(2.0 * np.pi * (220.0 + 7.0 * index) * time)
    waveform[0] = 0.001 * (index + 1)
    sf.write(
        path,
        waveform.astype(np.float32),
        preflight.EXPECTED_SAMPLE_RATE,
        format="FLAC",
        subtype="PCM_16",
    )
    return _sha256(path)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class AudioPreflightFixture:
    def __init__(
        self,
        root: Path,
        *,
        train_count: int = 2,
        dev_count: int = 2,
        test_count: int = 1,
    ) -> None:
        self.root = root
        self.rows: dict[str, list[dict[str, object]]] = {}
        self.manifests: dict[str, Path] = {}
        running_index = 0
        for split, count in (
            ("train", train_count),
            ("dev", dev_count),
            ("test", test_count),
        ):
            rows: list[dict[str, object]] = []
            for index in range(count):
                scene_id = f"{split}_scene_{index}"
                relative = Path("audio") / split / f"{scene_id}.flac"
                sha = _write_flac(root / relative, index=running_index)
                source_digest = hashlib.sha256(
                    f"source:{scene_id}".encode("utf-8")
                ).hexdigest()
                rows.append(
                    {
                        "scene_id": scene_id,
                        "video_id": f"video_{scene_id}",
                        "split": split,
                        "mixture_path": relative.as_posix(),
                        "audio_sha256": sha,
                        "sample_rate": preflight.EXPECTED_SAMPLE_RATE,
                        "audio_num_frames": 1280,
                        "audio_num_channels": 1,
                        "duration_seconds": 0.08,
                        "events": [
                            {
                                "event_id": f"event_{scene_id}",
                                "label": f"label_{running_index}",
                                "onset_seconds": 0.0,
                                "offset_seconds": 0.08,
                                "source_id": f"source_{scene_id}",
                                "source_video_id": f"source_video_{scene_id}",
                                "source_sha256": source_digest,
                                "source_path": f"sources/{scene_id}.flac",
                            }
                        ],
                        "context_events": [],
                    }
                )
                running_index += 1
            manifest = root / f"detector_manifest_{split}.jsonl"
            _write_jsonl(manifest, rows)
            self.rows[split] = rows
            self.manifests[split] = manifest
        self.gate_path = root / "microfit_metadata_gate_v1.json"
        self.output = root / "audio_preflight.json"
        self.refresh_gate(train_range=[2, 3])

    def refresh_manifest(self, split: str) -> None:
        _write_jsonl(self.manifests[split], self.rows[split])

    def refresh_gate(
        self,
        *,
        train_range: list[int],
        gate_passes: bool = True,
    ) -> None:
        gate = {
            "format": preflight.METADATA_GATE_FORMAT,
            "passes": gate_passes,
            "scope": {
                "debug_microfit_only": True,
                "paper_eligible": False,
                "metadata_training_authorized": gate_passes,
                "audio_training_authorized": False,
            },
            "contract": {
                "train_scene_range": train_range,
                "minimum_dev_scenes": 2,
                "minimum_test_scenes": 1,
                "fixed_grid_frames": 250,
                "boundary_dilation_frames": 0,
                "test_used_for_selection": False,
            },
            "inputs": {
                "manifests": {
                    split: {
                        "path": str(path.resolve()),
                        "sha256": _sha256(path),
                    }
                    for split, path in self.manifests.items()
                }
            },
            "integrity": {"passes": True, "overlap_count": 0},
            "calibration_selection": {
                "identity_audit": {"passes": True, "overlap_count": 0}
            },
            "splits": {
                split: {"scenes": len(rows)} for split, rows in self.rows.items()
            },
        }
        self.gate_path.write_text(
            json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def args(self) -> argparse.Namespace:
        return argparse.Namespace(
            train_manifest=self.manifests["train"],
            dev_manifest=self.manifests["dev"],
            test_manifest=self.manifests["test"],
            metadata_gate_receipt=self.gate_path,
            audio_root=self.root,
            output=self.output,
            clipping_threshold=preflight.DEFAULT_CLIP_THRESHOLD,
            minimum_rms=preflight.DEFAULT_MINIMUM_RMS,
            reconstruction_tolerance=preflight.DEFAULT_RECONSTRUCTION_TOLERANCE,
        )


def _debug_scene_range() -> tuple[mock._patch, mock._patch]:
    return (
        mock.patch.object(preflight, "TRAIN_SCENES_MIN", 2),
        mock.patch.object(preflight, "TRAIN_SCENES_MAX", 3),
    )


def test_tiny_flacs_pass_with_component_and_evidence_reconstruction() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = AudioPreflightFixture(Path(directory))
        row = fixture.rows["train"][0]
        original = fixture.root / str(row["mixture_path"])
        samples, rate = sf.read(original, dtype="float32")

        component_path = Path("components/train/source.flac")
        (fixture.root / component_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(fixture.root / component_path, samples, rate, format="FLAC", subtype="PCM_16")
        event = row["events"][0]  # type: ignore[index]
        event.update(  # type: ignore[union-attr]
            {
                "component_path": component_path.as_posix(),
                "component_sha256": _sha256(fixture.root / component_path),
                "component_num_samples": int(samples.size),
                "placement_start_sample": 0,
            }
        )

        evidence_path = Path("evidence/train/evidence.flac")
        residual_path = Path("evidence/train/residual.flac")
        (fixture.root / evidence_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(fixture.root / evidence_path, samples, rate, format="FLAC", subtype="PCM_16")
        sf.write(
            fixture.root / residual_path,
            np.zeros_like(samples),
            rate,
            format="FLAC",
            subtype="PCM_16",
        )
        row.update(
            {
                "evidence_path": evidence_path.as_posix(),
                "evidence_sha256": _sha256(fixture.root / evidence_path),
                "residual_path": residual_path.as_posix(),
                "residual_sha256": _sha256(fixture.root / residual_path),
            }
        )
        fixture.refresh_manifest("train")
        fixture.refresh_gate(train_range=[2, 3])

        minimum, maximum = _debug_scene_range()
        with minimum, maximum:
            receipt = preflight.build_receipt(fixture.args())

    assert receipt["passes"]
    assert receipt["audio_training_authorized"]
    assert receipt["scope"]["audio_training_authorized"]
    assert receipt["audio_io"]["audio_files_opened"] == 8
    assert receipt["audio_io"]["inventory_entries"] == 8
    assert receipt["reconstruction"]["source_components"]["scenes_with_contract"] == 1
    assert receipt["reconstruction"]["source_components"]["maximum_absolute_error"] == 0.0
    assert receipt["reconstruction"]["evidence_residual"]["evidence"][
        "scenes_with_contract"
    ] == 1


def test_failed_metadata_gate_opens_no_audio() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = AudioPreflightFixture(Path(directory))
        fixture.refresh_gate(train_range=[2, 3], gate_passes=False)
        minimum, maximum = _debug_scene_range()
        with minimum, maximum:
            receipt = preflight.build_receipt(fixture.args())
    assert not receipt["passes"]
    assert not receipt["audio_training_authorized"]
    assert receipt["audio_io"]["audio_files_opened"] == 0


def test_independent_source_overlap_audit_fails_before_audio_io() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = AudioPreflightFixture(Path(directory))
        train_event = fixture.rows["train"][0]["events"][0]  # type: ignore[index]
        dev_event = fixture.rows["dev"][0]["events"][0]  # type: ignore[index]
        dev_event["source_sha256"] = train_event["source_sha256"]  # type: ignore[index]
        fixture.refresh_manifest("dev")
        fixture.refresh_gate(train_range=[2, 3])
        minimum, maximum = _debug_scene_range()
        with minimum, maximum:
            receipt = preflight.build_receipt(fixture.args())
    assert not receipt["passes"]
    assert receipt["audio_io"]["audio_files_opened"] == 0
    assert any("source_sha256 reused" in failure for failure in receipt["failures"])


def test_hash_mismatch_and_off_grid_timestamp_refuse_authorization() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = AudioPreflightFixture(Path(directory))
        fixture.rows["train"][0]["audio_sha256"] = "f" * 64
        fixture.rows["dev"][0]["events"][0]["onset_seconds"] = 0.03  # type: ignore[index]
        fixture.refresh_manifest("train")
        fixture.refresh_manifest("dev")
        fixture.refresh_gate(train_range=[2, 3])
        minimum, maximum = _debug_scene_range()
        with minimum, maximum:
            receipt = preflight.build_receipt(fixture.args())
    assert not receipt["passes"]
    assert not receipt["audio_training_authorized"]
    assert any("SHA-256 mismatch" in failure for failure in receipt["failures"])
    assert any("not aligned" in failure for failure in receipt["failures"])


def test_missing_declared_frame_count_is_fatal() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = AudioPreflightFixture(Path(directory))
        del fixture.rows["train"][0]["audio_num_frames"]
        fixture.refresh_manifest("train")
        fixture.refresh_gate(train_range=[2, 3])
        minimum, maximum = _debug_scene_range()
        with minimum, maximum:
            receipt = preflight.build_receipt(fixture.args())
    assert not receipt["passes"]
    assert any("does not declare audio_num_frames" in failure for failure in receipt["failures"])


def test_reader_rejects_nan_clipping_and_silence_pathologies() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cases = {
            "nan.wav": np.array([0.0, np.nan, 0.1, -0.1], dtype=np.float32),
            "clipped.wav": np.array([0.0, 1.0, 0.1, -0.1], dtype=np.float32),
            "silent.wav": np.zeros(4, dtype=np.float32),
        }
        for filename, samples in cases.items():
            sf.write(
                root / filename,
                samples,
                preflight.EXPECTED_SAMPLE_RATE,
                format="WAV",
                subtype="FLOAT",
            )
        reader = preflight.OnceAudioReader(
            audio_root=root,
            clipping_threshold=preflight.DEFAULT_CLIP_THRESHOLD,
            minimum_rms=preflight.DEFAULT_MINIMUM_RMS,
        )
        messages: list[str] = []
        for filename in cases:
            try:
                reader.read(
                    filename,
                    expected_sha256=_sha256(root / filename),
                    role="pathology_test",
                    context=filename,
                    require_audible=True,
                )
            except ValueError as exc:
                messages.append(str(exc))
    assert len(messages) == 3
    assert any("NaN or Inf" in message for message in messages)
    assert any("clipping threshold" in message for message in messages)
    assert any("silent/near-silent" in message for message in messages)


def test_production_contract_rejects_less_than_200_train_scenes_before_audio_io() -> None:
    with tempfile.TemporaryDirectory() as directory:
        fixture = AudioPreflightFixture(Path(directory))
        fixture.refresh_gate(train_range=[200, 500])
        receipt = preflight.build_receipt(fixture.args())
    assert not receipt["passes"]
    assert not receipt["audio_training_authorized"]
    assert receipt["audio_io"]["audio_files_opened"] == 0
    assert any("outside [200,500]" in failure for failure in receipt["failures"])
