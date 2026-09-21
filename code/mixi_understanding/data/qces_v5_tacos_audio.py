"""Derive immutable 10 s WAVs for the selected QCES/TACOS real subset.

The official TACOS archive contains variable-duration MP3 clips.  QCES uses a
10 s, 32 kHz acoustic contract, so annotation and evaluation must consume a
derived waveform rather than an unrecorded player-side seek into the MP3.
This module verifies and fully decodes each selected archive member, applies
the packet's pre-annotation sample-exact window, and writes a deterministic
mono IEEE-float WAV.  It never pads, resamples, or persists the source MP3.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import tempfile
import zlib
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from mixi_understanding.data.qces_v5_tacos import (
    BENCHMARK_SAMPLE_RATE,
    DEFAULT_BENCHMARK_WINDOW_SECONDS,
    DOI,
    PACKET_FORMAT,
    PLAN_FORMAT,
    RECORD_ID,
    TacosAuditError,
    atomic_json,
    atomic_jsonl,
    canonical_json_sha256,
    sha256_file,
    verify_audio_archive,
)


AUDIO_RECEIPT_FORMAT = "qces_v5_tacos_derived_audio_receipt_v2"
AUDIO_COMPLIANCE_FORMAT = "qces_v5_tacos_audio_compliance_v2"
DERIVATION_RECIPE = "full_mp3_decode_f32_then_packet_sample_slice_to_float_wav_v1"
CANONICAL_FRAME_COUNT = round(DEFAULT_BENCHMARK_WINDOW_SECONDS * BENCHMARK_SAMPLE_RATE)
CANONICAL_WAV_FORMAT = "WAV"
CANONICAL_WAV_SUBTYPE = "FLOAT"


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TacosAuditError(f"cannot parse JSON: {path}") from error
    if not isinstance(payload, dict):
        raise TacosAuditError(f"{path} must contain a JSON object")
    return payload


def _load_packet(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise TacosAuditError(f"{path}:{line_number} lacks final newline")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TacosAuditError(f"cannot parse {path}:{line_number}") from error
            if not isinstance(row, dict) or row.get("schema_version") != PACKET_FORMAT:
                raise TacosAuditError(f"invalid packet row at {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise TacosAuditError("annotation packet is empty")
    scene_ids = [row.get("scene_id") for row in rows]
    if any(
        not isinstance(value, str) or re.fullmatch(r"tacos_[0-9]+", value) is None
        for value in scene_ids
    ):
        raise TacosAuditError("packet has an invalid scene_id")
    if len(scene_ids) != len(set(scene_ids)):
        raise TacosAuditError("packet has duplicate scene_id values")
    return rows


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TacosAuditError(f"{context} must be an integer")
    return value


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TacosAuditError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TacosAuditError(f"{context} must be finite")
    return result


def _f32le_bytes(samples: np.ndarray) -> bytes:
    canonical = np.ascontiguousarray(samples, dtype=np.dtype("<f4"))
    return canonical.tobytes(order="C")


def _pcm_f32le_sha256(samples: np.ndarray) -> str:
    return hashlib.sha256(_f32le_bytes(samples)).hexdigest()


def _sample_statistics(samples: np.ndarray, sample_rate: int) -> dict[str, Any]:
    if samples.ndim != 1 or samples.size <= 0:
        raise TacosAuditError("decoded mono audio must be a non-empty vector")
    nonfinite = int(samples.size - np.count_nonzero(np.isfinite(samples)))
    if nonfinite:
        raise TacosAuditError("decoded audio contains non-finite samples")
    peak = float(np.abs(samples).max())
    rms = math.sqrt(float(np.square(samples, dtype=np.float64).mean()))
    if not math.isfinite(rms) or peak <= 1e-7:
        raise TacosAuditError("decoded audio is silent/invalid")
    return {
        "sample_rate_hz": sample_rate,
        "channels": 1,
        "decoded_frames": int(samples.size),
        "decoded_duration_seconds": samples.size / float(sample_rate),
        "peak_abs_↑": peak,
        "rms_↑": rms,
        "nonfinite_sample_count_↓": 0,
        "pcm_f32le_sha256": _pcm_f32le_sha256(samples),
    }


def _decode_full_source_mp3(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode every source frame while enforcing the no-resampling contract."""

    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError) as error:
        raise TacosAuditError(f"cannot inspect source MP3 {path}: {error}") from error
    if info.format != "MP3":
        raise TacosAuditError(f"source archive member is not decoded as MP3: {path}")
    if info.samplerate != BENCHMARK_SAMPLE_RATE or info.channels != 1:
        raise TacosAuditError(
            "source audio must already be mono 32 kHz; resampling/downmixing is "
            f"prohibited (sr={info.samplerate}, channels={info.channels})"
        )
    try:
        samples, decoded_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (RuntimeError, OSError) as error:
        raise TacosAuditError(
            f"cannot fully decode source MP3 {path}: {error}"
        ) from error
    samples = np.asarray(samples, dtype=np.float32)
    if decoded_rate != BENCHMARK_SAMPLE_RATE or samples.ndim != 1:
        raise TacosAuditError("source decoder changed the mono 32 kHz audio contract")
    if info.frames > 0 and int(samples.size) != int(info.frames):
        raise TacosAuditError(
            f"source decoded frame mismatch: info={info.frames}, read={samples.size}"
        )
    properties = _sample_statistics(samples, BENCHMARK_SAMPLE_RATE)
    properties.update(
        {
            "container_format": info.format,
            "codec_subtype": info.subtype,
            "container_endian": info.endian,
        }
    )
    return samples, properties


def inspect_decoded_audio(path: Path) -> Mapping[str, Any]:
    """Decode and audit a derived canonical WAV, including every sample."""

    samples, properties = _read_canonical_wav(path)
    del samples
    return properties


def _read_canonical_wav(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        info = sf.info(str(path))
    except (RuntimeError, OSError) as error:
        raise TacosAuditError(f"cannot inspect derived WAV {path}: {error}") from error
    if (
        info.format != CANONICAL_WAV_FORMAT
        or info.subtype != CANONICAL_WAV_SUBTYPE
        or info.samplerate != BENCHMARK_SAMPLE_RATE
        or info.channels != 1
        or info.frames != CANONICAL_FRAME_COUNT
    ):
        raise TacosAuditError(
            "derived audio is not canonical mono 32 kHz/10 s IEEE-float WAV: "
            f"format={info.format}, subtype={info.subtype}, sr={info.samplerate}, "
            f"channels={info.channels}, frames={info.frames}"
        )
    try:
        samples, decoded_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (RuntimeError, OSError) as error:
        raise TacosAuditError(
            f"cannot fully decode derived WAV {path}: {error}"
        ) from error
    samples = np.asarray(samples, dtype=np.float32)
    if decoded_rate != BENCHMARK_SAMPLE_RATE or samples.shape != (
        CANONICAL_FRAME_COUNT,
    ):
        raise TacosAuditError("derived WAV decoded shape/rate mismatch")
    properties = _sample_statistics(samples, BENCHMARK_SAMPLE_RATE)
    properties.update(
        {
            "container_format": info.format,
            "codec_subtype": info.subtype,
            "container_endian": info.endian,
        }
    )
    return samples, properties


def _safe_member_name(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise TacosAuditError(f"{context} must be a string")
    member = PurePosixPath(value)
    if (
        member.is_absolute()
        or ".." in member.parts
        or len(member.parts) != 1
        or member.suffix.casefold() != ".mp3"
        or not member.stem.isdigit()
    ):
        raise TacosAuditError(f"unsafe selected archive member: {value}")
    return value


def _materialize_member_temporarily(
    *,
    archive: zipfile.ZipFile,
    member_name: str,
    expected_crc32: int,
    expected_size: int,
    temporary_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Copy one member to a temporary file and independently verify its bytes."""

    descriptor, temporary_name = tempfile.mkstemp(
        dir=temporary_dir, prefix=".qces_tacos_source_", suffix=".mp3"
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    crc = 0
    size = 0
    try:
        with os.fdopen(descriptor, "wb") as output, archive.open(member_name) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                output.write(chunk)
                digest.update(chunk)
                crc = zlib.crc32(chunk, crc)
                size += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        crc &= 0xFFFFFFFF
        if size != expected_size:
            raise TacosAuditError(
                f"archive member size mismatch for {member_name}: "
                f"expected={expected_size}, actual={size}"
            )
        if crc != expected_crc32:
            raise TacosAuditError(
                f"archive member CRC mismatch for {member_name}: "
                f"expected={expected_crc32:08x}, actual={crc:08x}"
            )
        return temporary, {
            "source_member_size_bytes": size,
            "source_member_crc32": f"{crc:08x}",
            "source_member_sha256": digest.hexdigest(),
        }
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _stage_canonical_wav(
    samples: np.ndarray,
    destination: Path,
) -> tuple[Path, dict[str, Any]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".wav"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        # libsndfile adds a time-varying PEAK chunk to floating-point WAVs.
        # Write one minimal, standards-compliant RIFF layout ourselves so the
        # exact same PCM always has the exact same file SHA-256.
        pcm = _f32le_bytes(samples)
        fmt_chunk = b"fmt " + struct.pack(
            "<IHHIIHH",
            16,  # WAVEFORMAT payload bytes
            3,  # WAVE_FORMAT_IEEE_FLOAT
            1,  # mono
            BENCHMARK_SAMPLE_RATE,
            BENCHMARK_SAMPLE_RATE * 4,
            4,  # block alignment
            32,
        )
        fact_chunk = b"fact" + struct.pack("<II", 4, CANONICAL_FRAME_COUNT)
        data_header = b"data" + struct.pack("<I", len(pcm))
        riff_size = 4 + len(fmt_chunk) + len(fact_chunk) + len(data_header) + len(pcm)
        header = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
        with temporary.open("wb") as handle:
            handle.write(header)
            handle.write(fmt_chunk)
            handle.write(fact_chunk)
            handle.write(data_header)
            handle.write(pcm)
            handle.flush()
            os.fsync(handle.fileno())
        decoded, properties = _read_canonical_wav(temporary)
        expected_pcm = _pcm_f32le_sha256(samples)
        if properties["pcm_f32le_sha256"] != expected_pcm or not np.array_equal(
            decoded.view(np.uint32), samples.view(np.uint32)
        ):
            raise TacosAuditError(
                "derived WAV does not preserve the exact float samples"
            )
        properties["file_sha256"] = sha256_file(temporary)
        properties["file_size_bytes"] = temporary.stat().st_size
        return temporary, properties
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _install_staged_wav(
    *, staged: Path, destination: Path, expected_sha256: str, overwrite: bool
) -> bool:
    """Install a staged WAV atomically; return True when an exact file was reused."""

    reused = False
    try:
        if destination.exists():
            if destination.is_symlink():
                raise TacosAuditError(
                    f"derived audio target must not be a symlink: {destination}"
                )
            if not destination.is_file():
                raise TacosAuditError(
                    f"derived audio target is not a file: {destination}"
                )
            if sha256_file(destination) == expected_sha256:
                reused = True
            elif not overwrite:
                raise TacosAuditError(
                    f"existing derived WAV mismatch: {destination}; use --overwrite-audio"
                )
        if reused:
            staged.unlink()
        else:
            os.replace(staged, destination)
            directory_descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        return reused
    finally:
        if staged.exists():
            staged.unlink()


def _validate_plan(
    plan: Mapping[str, Any], *, packet_fingerprint: str
) -> tuple[str, Mapping[str, Any]]:
    if plan.get("format") != PLAN_FORMAT:
        raise TacosAuditError("source plan schema mismatch")
    if plan.get("packet_fingerprint") != packet_fingerprint:
        raise TacosAuditError("plan/annotation-packet fingerprint mismatch")
    if plan.get("paper_result_eligible") is not False:
        raise TacosAuditError("candidate plan must remain paper_result_eligible=false")
    record = plan.get("record")
    if (
        not isinstance(record, dict)
        or record.get("record_id") != RECORD_ID
        or record.get("doi") != DOI
    ):
        raise TacosAuditError("source plan is not bound to the pinned TACOS record")
    selection = plan.get("selection")
    if not isinstance(selection, dict):
        raise TacosAuditError("source plan lacks its selection contract")
    if (
        _finite(
            selection.get("benchmark_window_seconds"),
            "plan benchmark_window_seconds",
        )
        != DEFAULT_BENCHMARK_WINDOW_SECONDS
        or _integer(
            selection.get("benchmark_window_sample_rate_hz"),
            "plan benchmark_window_sample_rate_hz",
        )
        != BENCHMARK_SAMPLE_RATE
    ):
        raise TacosAuditError(
            "source plan does not use the canonical 10 s/32 kHz window"
        )
    return canonical_json_sha256(plan), selection


def _validate_window_and_proposals(
    row: Mapping[str, Any], *, decoded_frames: int
) -> tuple[int, int, dict[str, Any]]:
    scene_id = row.get("scene_id")
    audio = row.get("audio")
    source = row.get("source")
    window = row.get("benchmark_window")
    proposals = row.get("proposal_regions")
    selected_ids = row.get("suggested_distinct_onset_chain")
    semantic_chain = row.get("semantic_chain")
    if not all(
        isinstance(value, dict) for value in (audio, source, window, semantic_chain)
    ) or not isinstance(proposals, list):
        raise TacosAuditError(f"{scene_id} lacks a v2 window/proposal binding")
    if (
        not isinstance(selected_ids, list)
        or len(selected_ids) != 4
        or len(set(selected_ids)) != 4
        or semantic_chain.get("region_ids") != selected_ids
    ):
        raise TacosAuditError(f"{scene_id} lacks one exact four-region semantic chain")

    start = _integer(
        audio.get("benchmark_window_start_sample_32k"),
        f"{scene_id}.audio.window_start",
    )
    end = _integer(
        audio.get("benchmark_window_end_sample_32k"),
        f"{scene_id}.audio.window_end",
    )
    if _integer(audio.get("benchmark_sample_rate_hz"), f"{scene_id}.audio.sr") != (
        BENCHMARK_SAMPLE_RATE
    ):
        raise TacosAuditError(f"{scene_id} audio sample-rate contract mismatch")
    if (
        _integer(window.get("start_sample"), f"{scene_id}.window.start") != start
        or _integer(window.get("end_sample"), f"{scene_id}.window.end") != end
        or _integer(window.get("sample_rate_hz"), f"{scene_id}.window.sr")
        != BENCHMARK_SAMPLE_RATE
        or _finite(window.get("duration_seconds"), f"{scene_id}.window.duration")
        != DEFAULT_BENCHMARK_WINDOW_SECONDS
        or end - start != CANONICAL_FRAME_COUNT
    ):
        raise TacosAuditError(f"{scene_id} benchmark window duplicates disagree")
    if start < 0 or end > decoded_frames:
        raise TacosAuditError(
            f"{scene_id} crop [{start}, {end}) exceeds {decoded_frames} decoded frames; "
            "padding is prohibited"
        )

    start_seconds = start / float(BENCHMARK_SAMPLE_RATE)
    end_seconds = end / float(BENCHMARK_SAMPLE_RATE)
    if (
        abs(
            _finite(
                window.get("start_seconds_in_upstream_clip"),
                f"{scene_id}.window.start_seconds",
            )
            - start_seconds
        )
        > 0.5 / BENCHMARK_SAMPLE_RATE
        or abs(
            _finite(
                window.get("end_seconds_in_upstream_clip"),
                f"{scene_id}.window.end_seconds",
            )
            - end_seconds
        )
        > 0.5 / BENCHMARK_SAMPLE_RATE
    ):
        raise TacosAuditError(f"{scene_id} sample/second window coordinates disagree")
    valid_start_interval = window.get("valid_start_sample_interval_inclusive")
    if not isinstance(valid_start_interval, list) or len(valid_start_interval) != 2:
        raise TacosAuditError(f"{scene_id} lacks the frozen valid-start interval")
    valid_start_low = _integer(
        valid_start_interval[0], f"{scene_id}.window.valid_start_low"
    )
    valid_start_high = _integer(
        valid_start_interval[1], f"{scene_id}.window.valid_start_high"
    )
    if not 0 <= valid_start_low <= start <= valid_start_high:
        raise TacosAuditError(
            f"{scene_id} selected start lies outside its proposal-containment interval"
        )
    if (
        window.get("construction_is_proposal_informed") is not True
        or window.get("runtime_or_question_specific") is not False
        or window.get("selected_before_human_annotation") is not True
        or window.get("human_labels_used") is not False
        or window.get("qces_method_outputs_used") is not False
    ):
        raise TacosAuditError(f"{scene_id} benchmark-window provenance mismatch")
    source_window = source.get("benchmark_window_interval_in_upstream_clip_seconds")
    if (
        not isinstance(source_window, list)
        or len(source_window) != 2
        or abs(
            _finite(source_window[0], f"{scene_id}.source_window[0]") - start_seconds
        )
        > 0.5 / BENCHMARK_SAMPLE_RATE
        or abs(_finite(source_window[1], f"{scene_id}.source_window[1]") - end_seconds)
        > 0.5 / BENCHMARK_SAMPLE_RATE
        or _finite(source.get("clip_duration_seconds"), f"{scene_id}.clip_duration")
        != DEFAULT_BENCHMARK_WINDOW_SECONDS
    ):
        raise TacosAuditError(f"{scene_id} source/window time binding mismatch")

    selected_set = set(selected_ids)
    selected_rows: list[Mapping[str, Any]] = []
    region_ids: set[str] = set()
    truncated_count = 0
    minimum_margin = DEFAULT_BENCHMARK_WINDOW_SECONDS
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            raise TacosAuditError(f"{scene_id}.proposal_regions[{index}] is invalid")
        region_id = proposal.get("region_id")
        if not isinstance(region_id, str) or not region_id or region_id in region_ids:
            raise TacosAuditError(
                f"{scene_id} has invalid/duplicate proposal region IDs"
            )
        region_ids.add(region_id)
        onset = _finite(proposal.get("onset_seconds"), f"{scene_id}.{region_id}.onset")
        offset = _finite(
            proposal.get("offset_seconds"), f"{scene_id}.{region_id}.offset"
        )
        if not 0.0 <= onset < offset <= DEFAULT_BENCHMARK_WINDOW_SECONDS:
            raise TacosAuditError(
                f"{scene_id}.{region_id} lies outside the packet window"
            )
        marked_selected = proposal.get("selected_chain_member") is True
        if marked_selected != (region_id in selected_set):
            raise TacosAuditError(f"{scene_id}.{region_id} chain membership mismatch")
        truncated = proposal.get("truncated_by_benchmark_window") is True
        truncated_count += int(truncated)
        if marked_selected:
            if truncated:
                raise TacosAuditError(
                    f"{scene_id}.{region_id} selected proposal is window-truncated"
                )
            upstream_onset = _finite(
                proposal.get("upstream_clip_onset_seconds"),
                f"{scene_id}.{region_id}.upstream_onset",
            )
            upstream_offset = _finite(
                proposal.get("upstream_clip_offset_seconds"),
                f"{scene_id}.{region_id}.upstream_offset",
            )
            tolerance = 1e-7
            if (
                abs((upstream_onset - start_seconds) - onset) > tolerance
                or abs((upstream_offset - start_seconds) - offset) > tolerance
            ):
                raise TacosAuditError(
                    f"{scene_id}.{region_id} shifted/upstream proposal times disagree"
                )
            minimum_margin = min(
                minimum_margin,
                onset,
                DEFAULT_BENCHMARK_WINDOW_SECONDS - offset,
            )
            selected_rows.append(proposal)
    if len(selected_rows) != 4 or not selected_set.issubset(region_ids):
        raise TacosAuditError(
            f"{scene_id} does not contain all four selected proposals"
        )
    return (
        start,
        end,
        {
            "selected_semantic_chain_regions_inside_window_↑": 4,
            "selected_semantic_chain_regions_outside_window_↓": 0,
            "selected_semantic_chain_regions_truncated_↓": 0,
            "window_intersecting_proposal_regions_↑": len(proposals),
            "window_truncated_nonchain_proposal_regions_↓": truncated_count,
            "minimum_selected_region_window_margin_seconds_↑": minimum_margin,
            "valid_start_sample_interval_inclusive": [
                valid_start_low,
                valid_start_high,
            ],
            "valid_start_positions_↑": valid_start_high - valid_start_low + 1,
            "chosen_start_offset_within_valid_interval_samples": start
            - valid_start_low,
        },
    )


def extract_tacos_subset(
    *,
    packet_path: Path,
    plan_path: Path,
    archive_path: Path,
    output_dir: Path,
    project_root: Path,
    overwrite_audio: bool,
    maximum_duration_error_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if maximum_duration_error_seconds < 0.0 or not math.isfinite(
        maximum_duration_error_seconds
    ):
        raise TacosAuditError("maximum duration error must be finite and non-negative")
    rows = _load_packet(packet_path)
    plan = _load_json(plan_path)
    packet_fingerprint = canonical_json_sha256(rows)
    plan_fingerprint, plan_selection = _validate_plan(
        plan, packet_fingerprint=packet_fingerprint
    )

    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    try:
        output_relative = output_dir.relative_to(project_root)
    except ValueError as error:
        raise TacosAuditError(
            "output directory must lie inside project root"
        ) from error
    if not output_relative.parts:
        raise TacosAuditError("refusing to derive audio into the project root")
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames: list[str] = []
    packet_by_filename: dict[str, Mapping[str, Any]] = {}
    output_names: set[str] = set()
    for row in rows:
        source = row.get("source")
        audio = row.get("audio")
        if not isinstance(source, dict) or not isinstance(audio, dict):
            raise TacosAuditError(f"{row.get('scene_id')} lacks source/audio binding")
        filename = _safe_member_name(source.get("filename"), "source filename")
        member = _safe_member_name(audio.get("archive_member"), "archive member")
        if member != filename:
            raise TacosAuditError(f"{row.get('scene_id')} archive/source name mismatch")
        freesound_id = source.get("freesound_id")
        if (
            not isinstance(freesound_id, str)
            or not freesound_id.isdigit()
            or filename != f"{freesound_id}.mp3"
            or row["scene_id"] != f"tacos_{freesound_id}"
        ):
            raise TacosAuditError(
                f"{row.get('scene_id')} scene/source/Freesound identity mismatch"
            )
        if filename in packet_by_filename:
            raise TacosAuditError(f"packet reuses source filename: {filename}")
        output_name = f"{row['scene_id']}.wav"
        if output_name in output_names:
            raise TacosAuditError(f"packet reuses derived output name: {output_name}")
        filenames.append(filename)
        packet_by_filename[filename] = row
        output_names.add(output_name)

    archive_receipt = verify_audio_archive(archive_path, filenames)
    member_receipts = archive_receipt.get("selected_members")
    if not isinstance(member_receipts, dict) or set(member_receipts) != set(filenames):
        raise TacosAuditError("verified archive member coverage mismatch")
    receipts: list[dict[str, Any]] = []
    reused_count = 0
    written_count = 0
    source_duration_errors: list[float] = []
    selected_chain_margins: list[float] = []
    # Source MP3s are transient decoder inputs and never live in the derived
    # dataset directory, even if a prior run is interrupted mid-scene.
    with tempfile.TemporaryDirectory(
        prefix="qces_tacos_decode_"
    ) as decode_temp_name, zipfile.ZipFile(archive_path, "r") as archive:
        decode_temp_dir = Path(decode_temp_name)
        for filename in filenames:
            row = packet_by_filename[filename]
            member = member_receipts[filename]
            if not isinstance(member, dict):
                raise TacosAuditError(f"{filename} verified archive receipt is invalid")
            member_name = _safe_member_name(
                member.get("archive_member"), f"{filename} verified archive member"
            )
            try:
                expected_crc = int(str(member["crc32"]), 16)
            except (KeyError, TypeError, ValueError) as error:
                raise TacosAuditError(
                    f"{filename} verified archive CRC is invalid"
                ) from error
            expected_member_size = _integer(
                member.get("uncompressed_size_bytes"),
                f"{filename} uncompressed archive size",
            )
            temporary_source, source_bytes = _materialize_member_temporarily(
                archive=archive,
                member_name=member_name,
                expected_crc32=expected_crc,
                expected_size=expected_member_size,
                temporary_dir=decode_temp_dir,
            )
            try:
                source_samples, source_properties = _decode_full_source_mp3(
                    temporary_source
                )
            finally:
                if temporary_source.exists():
                    temporary_source.unlink()

            expected_duration = _finite(
                row["source"].get("upstream_clip_duration_seconds"),
                f"{row['scene_id']} upstream clip duration",
            )
            duration_error = abs(
                source_properties["decoded_duration_seconds"] - expected_duration
            )
            if duration_error > maximum_duration_error_seconds:
                raise TacosAuditError(
                    f"{filename} duration error {duration_error:.6f}s exceeds "
                    f"{maximum_duration_error_seconds:.6f}s"
                )
            start, end, proposal_containment = _validate_window_and_proposals(
                row, decoded_frames=int(source_samples.size)
            )
            crop = np.ascontiguousarray(source_samples[start:end], dtype=np.float32)
            if crop.shape != (CANONICAL_FRAME_COUNT,):
                raise TacosAuditError(
                    f"{row['scene_id']} did not yield exactly {CANONICAL_FRAME_COUNT} "
                    "source samples; padding is prohibited"
                )
            crop_properties = _sample_statistics(crop, BENCHMARK_SAMPLE_RATE)
            target = output_dir / f"{row['scene_id']}.wav"
            staged, derived_properties = _stage_canonical_wav(crop, target)
            reused = _install_staged_wav(
                staged=staged,
                destination=target,
                expected_sha256=derived_properties["file_sha256"],
                overwrite=overwrite_audio,
            )
            if sha256_file(target) != derived_properties["file_sha256"]:
                raise TacosAuditError(f"post-install derived WAV mismatch: {target}")
            relative_path = target.relative_to(project_root).as_posix()
            window_start_seconds = start / float(BENCHMARK_SAMPLE_RATE)
            window_end_seconds = end / float(BENCHMARK_SAMPLE_RATE)
            receipts.append(
                {
                    "format": AUDIO_RECEIPT_FORMAT,
                    "packet_fingerprint": packet_fingerprint,
                    "packet_file_sha256": sha256_file(packet_path),
                    "source_plan_fingerprint": plan_fingerprint,
                    "source_plan_file_sha256": sha256_file(plan_path),
                    "source_plan_input_fingerprint": plan.get("input_fingerprint"),
                    "scene_id": row["scene_id"],
                    "selection_partition": row["selection_partition"],
                    "selection_tier": row["selection_tier"],
                    "freesound_id": row["source"]["freesound_id"],
                    "filename": filename,
                    "archive_sha256": archive_receipt["sha256"],
                    "archive_publisher_md5": archive_receipt["publisher_md5"],
                    "archive_member": member_name,
                    "archive_crc32": member["crc32"],
                    **source_bytes,
                    "source_decoded_audio": source_properties,
                    "source_decoded_pcm_f32le_sha256": source_properties[
                        "pcm_f32le_sha256"
                    ],
                    "expected_upstream_clip_duration_seconds": expected_duration,
                    "source_duration_absolute_error_seconds_↓": duration_error,
                    "crop_recipe": {
                        "recipe": DERIVATION_RECIPE,
                        "source_coordinate_system": "decoded upstream TACOS clip",
                        "start_sample_inclusive": start,
                        "end_sample_exclusive": end,
                        "start_seconds": window_start_seconds,
                        "end_seconds": window_end_seconds,
                        "output_frames": CANONICAL_FRAME_COUNT,
                        "sample_rate_hz": BENCHMARK_SAMPLE_RATE,
                        "channels": 1,
                        "output_container": CANONICAL_WAV_FORMAT,
                        "output_subtype": CANONICAL_WAV_SUBTYPE,
                        "resampling_applied": False,
                        "downmixing_applied": False,
                        "padding_samples_↓": 0,
                        "human_labels_used": False,
                        "qces_method_outputs_used": False,
                    },
                    "proposal_containment": proposal_containment,
                    "crop_decoded_audio": crop_properties,
                    "local_path": relative_path,
                    "local_size_bytes": target.stat().st_size,
                    "local_sha256": derived_properties["file_sha256"],
                    "local_pcm_f32le_sha256": derived_properties["pcm_f32le_sha256"],
                    "local_verified_audio_properties": derived_properties,
                }
            )
            source_duration_errors.append(duration_error)
            selected_chain_margins.append(
                proposal_containment["minimum_selected_region_window_margin_seconds_↑"]
            )
            reused_count += int(reused)
            written_count += int(not reused)

    receipt_fingerprint = canonical_json_sha256(receipts)
    source_pcm_hashes = {row["source_decoded_pcm_f32le_sha256"] for row in receipts}
    source_member_hashes = {row["source_member_sha256"] for row in receipts}
    derived_hashes = {row["local_sha256"] for row in receipts}
    duplicate_counts = {
        "source member": len(receipts) - len(source_member_hashes),
        "source decoded PCM": len(receipts) - len(source_pcm_hashes),
        "derived crop": len(receipts) - len(derived_hashes),
    }
    if any(duplicate_counts.values()):
        raise TacosAuditError(
            "selected packet contains exact duplicate audio identities: "
            + ", ".join(
                f"{name}={count}" for name, count in duplicate_counts.items() if count
            )
        )
    compliance = {
        "format": AUDIO_COMPLIANCE_FORMAT,
        "packet_fingerprint": packet_fingerprint,
        "packet_file_sha256": sha256_file(packet_path),
        "source_plan_fingerprint": plan_fingerprint,
        "source_plan_file_sha256": sha256_file(plan_path),
        "audio_receipt_fingerprint": receipt_fingerprint,
        "archive": {
            key: value
            for key, value in archive_receipt.items()
            if key != "selected_members"
        },
        "derivation_contract": {
            "recipe": DERIVATION_RECIPE,
            "sample_rate_hz": BENCHMARK_SAMPLE_RATE,
            "frames_per_scene": CANONICAL_FRAME_COUNT,
            "duration_seconds": DEFAULT_BENCHMARK_WINDOW_SECONDS,
            "channels": 1,
            "container": CANONICAL_WAV_FORMAT,
            "subtype": CANONICAL_WAV_SUBTYPE,
            "maximum_source_duration_absolute_error_seconds_allowed": (
                maximum_duration_error_seconds
            ),
            "plan_window_start_rule": plan_selection.get("benchmark_window_start_rule"),
            "resampling_allowed": False,
            "padding_allowed": False,
        },
        "decoder_runtime": {
            "python_soundfile_version": str(getattr(sf, "__version__", "unknown")),
            "libsndfile_version": str(getattr(sf, "__libsndfile_version__", "unknown")),
        },
        "audio_extraction_gate_passed": True,
        "human_verification_gate_passed": False,
        "submission_real_data_gate_passed": False,
        "metrics": {
            "selected_derived_wav_files_↑": len(receipts),
            "newly_written_derived_wav_files_↑": written_count,
            "resume_reused_exact_derived_wav_files_↑": reused_count,
            "unique_source_decoded_pcm_hashes_↑": len(source_pcm_hashes),
            "unique_source_member_hashes_↑": len(source_member_hashes),
            "unique_derived_wav_hashes_↑": len(derived_hashes),
            "duplicate_audio_identities_↓": 0,
            "missing_selected_audio_↓": 0,
            "archive_crc_mismatch_files_↓": 0,
            "source_decode_failure_files_↓": 0,
            "source_non_mono_or_non_32k_files_↓": 0,
            "nonfinite_audio_files_↓": 0,
            "silent_audio_files_↓": 0,
            "noncanonical_derived_wav_files_↓": 0,
            "resampling_operations_↓": 0,
            "downmixing_operations_↓": 0,
            "padding_samples_↓": 0,
            "selected_chain_regions_outside_window_↓": 0,
            "selected_chain_regions_truncated_↓": 0,
            "minimum_selected_region_window_margin_seconds_↑": min(
                selected_chain_margins
            ),
            "maximum_source_duration_absolute_error_seconds_↓": max(
                source_duration_errors
            ),
            "mean_source_duration_absolute_error_seconds_↓": sum(source_duration_errors)
            / len(source_duration_errors),
        },
        "gate_reason": (
            "All selected sources are pinned-archive/CRC/decoded-PCM bound and "
            "materialized as exact packet-selected 10 s mono 32 kHz float WAVs. "
            "The real-data gate remains false until two-pass human verification."
        ),
    }
    return compliance, receipts


def write_extraction_artifacts(
    *,
    compliance_path: Path,
    receipt_path: Path,
    compliance: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    overwrite: bool,
) -> None:
    existing = [path for path in (compliance_path, receipt_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "audio receipt outputs exist; use --overwrite-receipts: "
            + ", ".join(map(str, existing))
        )
    atomic_jsonl(receipt_path, receipts, overwrite=overwrite)
    atomic_json(compliance_path, compliance, overwrite=overwrite)
