"""Human annotation contracts for the fixed-window QCES-Real-10 TACOS set."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from mixi_understanding.data.qces_v5_tacos import (
    BENCHMARK_SAMPLE_RATE,
    DEFAULT_BENCHMARK_WINDOW_SECONDS,
    PACKET_FORMAT,
    TacosAuditError,
    atomic_json,
    canonical_json_sha256,
    normalize_caption,
    sha256_file,
)
from mixi_understanding.data.qces_v5_tacos_audio import (
    AUDIO_RECEIPT_FORMAT,
    CANONICAL_FRAME_COUNT,
    CANONICAL_WAV_FORMAT,
    CANONICAL_WAV_SUBTYPE,
    inspect_decoded_audio,
)


RESPONSE_FORMAT = "qces_v5_tacos_human_annotation_response_v4"
RATER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,31}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TRISTATE = ("yes", "no", "uncertain")
SCENE_DECISIONS = ("accept", "reject", "uncertain")
CONTAMINATION = ("clean", "mixed", "uncertain")
PARTITIONS = ("real_dev", "real_test")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise TacosAuditError(f"{path}:{line_number} lacks final newline")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TacosAuditError(f"cannot parse {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise TacosAuditError(f"{path}:{line_number} is not an object")
            rows.append(row)
    if not rows:
        raise TacosAuditError(f"{path} is empty")
    return rows


def load_bound_tasks(
    *,
    packet_path: Path,
    audio_receipt_path: Path,
    project_root: Path,
) -> tuple[str, str, list[dict[str, Any]]]:
    packet = load_jsonl(packet_path)
    if any(row.get("schema_version") != PACKET_FORMAT for row in packet):
        raise TacosAuditError("annotation packet schema mismatch")
    packet_scene_ids = [row.get("scene_id") for row in packet]
    if any(
        not isinstance(scene_id, str) or re.fullmatch(r"tacos_[0-9]+", scene_id) is None
        for scene_id in packet_scene_ids
    ) or len(packet_scene_ids) != len(set(packet_scene_ids)):
        raise TacosAuditError("annotation packet has invalid/duplicate scene_id")
    packet_fingerprint = canonical_json_sha256(packet)
    packet_file_sha256 = sha256_file(packet_path)
    audio_rows = load_jsonl(audio_receipt_path)
    if any(row.get("format") != AUDIO_RECEIPT_FORMAT for row in audio_rows):
        raise TacosAuditError("audio receipt schema mismatch")
    if any(row.get("packet_fingerprint") != packet_fingerprint for row in audio_rows):
        raise TacosAuditError("audio receipt is not bound to the annotation packet")
    if any(row.get("packet_file_sha256") != packet_file_sha256 for row in audio_rows):
        raise TacosAuditError("audio receipt is not bound to the packet file bytes")
    for row in packet:
        _validate_packet_task(row)
    audio_fingerprint = canonical_json_sha256(audio_rows)
    audio_by_scene: dict[str, Mapping[str, Any]] = {}
    for row in audio_rows:
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or scene_id in audio_by_scene:
            raise TacosAuditError("audio receipt has invalid/duplicate scene_id")
        audio_by_scene[scene_id] = row
    if set(packet_scene_ids) != set(audio_by_scene):
        raise TacosAuditError("packet/audio-receipt scene coverage mismatch")

    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise TacosAuditError("project root must be an existing directory")
    tasks: list[dict[str, Any]] = []
    local_paths: set[str] = set()
    local_hashes: set[str] = set()
    for row in packet:
        scene_id = row["scene_id"]
        receipt = audio_by_scene[scene_id]
        _validate_receipt_binding(receipt, packet_row=row)
        relative = receipt.get("local_path")
        if not isinstance(relative, str) or "\\" in relative:
            raise TacosAuditError(f"{scene_id} has no safe local WAV path")
        pure_relative = PurePosixPath(relative)
        if (
            pure_relative.is_absolute()
            or ".." in pure_relative.parts
            or pure_relative.name != f"{scene_id}.wav"
        ):
            raise TacosAuditError(f"{scene_id} has an unsafe/noncanonical WAV path")
        unresolved_path = project_root.joinpath(*pure_relative.parts)
        if unresolved_path.is_symlink():
            raise TacosAuditError(f"{scene_id} derived WAV must not be a symlink")
        try:
            path = unresolved_path.resolve(strict=True)
        except OSError as error:
            raise TacosAuditError(f"{scene_id} derived WAV is missing") from error
        try:
            path.relative_to(project_root)
        except ValueError as error:
            raise TacosAuditError(f"{scene_id} audio escapes project root") from error
        if not path.is_file() or sha256_file(path) != receipt.get("local_sha256"):
            raise TacosAuditError(f"{scene_id} local audio hash mismatch")
        if path.stat().st_size != receipt.get("local_size_bytes"):
            raise TacosAuditError(f"{scene_id} local audio size mismatch")
        properties = dict(inspect_decoded_audio(path))
        _validate_decoded_wav_properties(
            properties,
            receipt=receipt,
            scene_id=scene_id,
        )
        if sha256_file(path) != receipt.get(
            "local_sha256"
        ) or path.stat().st_size != receipt.get("local_size_bytes"):
            raise TacosAuditError(f"{scene_id} WAV changed during verification")
        if relative in local_paths or receipt["local_sha256"] in local_hashes:
            raise TacosAuditError("audio receipt reuses a derived WAV identity")
        local_paths.add(relative)
        local_hashes.add(receipt["local_sha256"])
        task = dict(row)
        task["resolved_audio_path"] = str(path)
        task["audio_receipt_local_path"] = relative
        task["audio_receipt_sha256"] = receipt["local_sha256"]
        task["audio_receipt_pcm_f32le_sha256"] = receipt["local_pcm_f32le_sha256"]
        task["annotation_audio"] = {
            "local_sha256": receipt["local_sha256"],
            "pcm_f32le_sha256": receipt["local_pcm_f32le_sha256"],
            "sample_rate_hz": BENCHMARK_SAMPLE_RATE,
            "num_frames": CANONICAL_FRAME_COUNT,
            "channels": 1,
            "duration_seconds": DEFAULT_BENCHMARK_WINDOW_SECONDS,
            "container_format": CANONICAL_WAV_FORMAT,
            "codec_subtype": CANONICAL_WAV_SUBTYPE,
        }
        tasks.append(task)
    return packet_fingerprint, audio_fingerprint, tasks


def _finite_number(value: Any, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TacosAuditError(f"{context} must be finite numeric")
    return float(value)


def _validate_packet_task(row: Mapping[str, Any]) -> None:
    scene_id = row.get("scene_id", "unknown scene")
    partition = row.get("selection_partition")
    source = row.get("source")
    audio = row.get("audio")
    window = row.get("benchmark_window")
    proposals = row.get("proposal_regions")
    selected = row.get("suggested_distinct_onset_chain")
    if partition not in PARTITIONS:
        raise TacosAuditError(f"{scene_id} has an invalid annotation partition")
    if row.get("selection_tier") not in {"core", "reserve"} or (
        isinstance(row.get("selection_ordinal"), bool)
        or not isinstance(row.get("selection_ordinal"), int)
        or row["selection_ordinal"] < 0
    ):
        raise TacosAuditError(f"{scene_id} has invalid selection metadata")
    if not all(isinstance(value, Mapping) for value in (source, audio, window)):
        raise TacosAuditError(f"{scene_id} lacks packet-v2 source/audio/window data")
    assert isinstance(source, Mapping)
    assert isinstance(audio, Mapping)
    assert isinstance(window, Mapping)
    if source.get("custom_qces_partition") != partition:
        raise TacosAuditError(f"{scene_id} source/partition mismatch")
    if (
        _finite_number(source.get("clip_duration_seconds"), f"{scene_id}.duration")
        != DEFAULT_BENCHMARK_WINDOW_SECONDS
    ):
        raise TacosAuditError(f"{scene_id} is not an exact ten-second task")

    start = audio.get("benchmark_window_start_sample_32k")
    end = audio.get("benchmark_window_end_sample_32k")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end - start != CANONICAL_FRAME_COUNT
        or audio.get("benchmark_sample_rate_hz") != BENCHMARK_SAMPLE_RATE
        or window.get("start_sample") != start
        or window.get("end_sample") != end
        or window.get("sample_rate_hz") != BENCHMARK_SAMPLE_RATE
        or _finite_number(window.get("duration_seconds"), f"{scene_id}.window.duration")
        != DEFAULT_BENCHMARK_WINDOW_SECONDS
    ):
        raise TacosAuditError(f"{scene_id} violates the fixed-10s sample contract")
    if any(
        audio.get(field) is not None
        for field in ("local_path", "local_sha256", "verified_audio_properties")
    ):
        raise TacosAuditError(f"{scene_id} packet must not provide a bypass audio path")
    if (
        window.get("selected_before_human_annotation") is not True
        or window.get("human_labels_used") is not False
        or window.get("qces_method_outputs_used") is not False
        or window.get("runtime_or_question_specific") is not False
    ):
        raise TacosAuditError(f"{scene_id} benchmark-window provenance mismatch")
    source_window = source.get("benchmark_window_interval_in_upstream_clip_seconds")
    expected_start_seconds = start / float(BENCHMARK_SAMPLE_RATE)
    expected_end_seconds = end / float(BENCHMARK_SAMPLE_RATE)
    if (
        not isinstance(source_window, list)
        or len(source_window) != 2
        or abs(
            _finite_number(source_window[0], f"{scene_id}.source_window_start")
            - expected_start_seconds
        )
        > 0.5 / BENCHMARK_SAMPLE_RATE
        or abs(
            _finite_number(source_window[1], f"{scene_id}.source_window_end")
            - expected_end_seconds
        )
        > 0.5 / BENCHMARK_SAMPLE_RATE
        or not isinstance(source.get("subclass"), str)
        or not source["subclass"]
        or not isinstance(source.get("audio_license_spdx"), str)
        or not source["audio_license_spdx"]
    ):
        raise TacosAuditError(f"{scene_id} lacks valid visible crop metadata")
    diagnostics = row.get("proposal_diagnostics")
    question_contract = row.get("question_contract")
    absent_candidates = (
        question_contract.get("absent_anchor_candidates")
        if isinstance(question_contract, Mapping)
        else None
    )
    absent_support = (
        question_contract.get("absent_anchor_cross_scene_support")
        if isinstance(question_contract, Mapping)
        else None
    )
    if (
        not isinstance(diagnostics, Mapping)
        or "usable_region_count_↑" not in diagnostics
        or "overlap_pair_count_↑" not in diagnostics
        or not isinstance(absent_candidates, list)
        or len(absent_candidates) != 6
        or len(set(absent_candidates)) != 6
        or any(not isinstance(value, str) or not value for value in absent_candidates)
        or not isinstance(absent_support, list)
        or len(absent_support) != len(absent_candidates)
    ):
        raise TacosAuditError(f"{scene_id} lacks annotation task metadata")
    primary_support_scene_ids: list[str] = []
    for candidate, support in zip(absent_candidates, absent_support, strict=True):
        support_scene_ids = (
            support.get("cross_scene_support_scene_ids")
            if isinstance(support, Mapping)
            else None
        )
        primary_support_scene_id = (
            support.get("primary_support_scene_id")
            if isinstance(support, Mapping)
            else None
        )
        if (
            not isinstance(support, Mapping)
            or support.get("caption") != candidate
            or support.get("selection_partition") != partition
            or support.get("support_tier") != "core"
            or not isinstance(support_scene_ids, list)
            or not support_scene_ids
            or len(set(support_scene_ids)) != len(support_scene_ids)
            or any(
                not isinstance(support_scene_id, str)
                or not support_scene_id
                or support_scene_id == scene_id
                for support_scene_id in support_scene_ids
            )
            or support.get("cross_scene_support_count_↑") != len(support_scene_ids)
            or not isinstance(primary_support_scene_id, str)
            or primary_support_scene_id not in support_scene_ids
        ):
            raise TacosAuditError(
                f"{scene_id} has invalid same-split core absent support"
            )
        primary_support_scene_ids.append(primary_support_scene_id)
    if len(set(primary_support_scene_ids)) != 6:
        raise TacosAuditError(
            f"{scene_id} absent anchors lack six distinct primary support scenes"
        )
    if (
        question_contract.get("absent_anchor_pool_source")
        != "other_selected_chain_captions"
        or question_contract.get(
            "absent_anchor_candidates_are_exact_raw_positive_captions"
        )
        is not True
        or question_contract.get(
            "qces_label_selection_used_for_absent_anchor_selection"
        )
        is not False
    ):
        raise TacosAuditError(f"{scene_id} absent-anchor provenance contract mismatch")

    if (
        not isinstance(proposals, list)
        or not isinstance(selected, list)
        or len(selected) != 4
        or len(set(selected)) != 4
    ):
        raise TacosAuditError(f"{scene_id} lacks four selected crop-relative regions")
    by_id: dict[str, Mapping[str, Any]] = {}
    window_start_seconds = start / float(BENCHMARK_SAMPLE_RATE)
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, Mapping):
            raise TacosAuditError(f"{scene_id}.proposal[{index}] is invalid")
        region_id = proposal.get("region_id")
        if not isinstance(region_id, str) or not region_id or region_id in by_id:
            raise TacosAuditError(f"{scene_id} has invalid/duplicate region IDs")
        onset = _finite_number(
            proposal.get("onset_seconds"), f"{scene_id}.{region_id}.onset"
        )
        offset = _finite_number(
            proposal.get("offset_seconds"), f"{scene_id}.{region_id}.offset"
        )
        if not 0.0 <= onset < offset <= DEFAULT_BENCHMARK_WINDOW_SECONDS:
            raise TacosAuditError(
                f"{scene_id}.{region_id} is outside the final ten-second WAV"
            )
        if (
            not isinstance(proposal.get("caption"), str)
            or not proposal["caption"]
            or not isinstance(proposal.get("eligible_proposal"), bool)
            or not isinstance(proposal.get("selected_chain_member"), bool)
            or not isinstance(proposal.get("truncated_by_benchmark_window"), bool)
        ):
            raise TacosAuditError(
                f"{scene_id}.{region_id} has invalid proposal metadata"
            )
        upstream_onset = _finite_number(
            proposal.get("upstream_clip_onset_seconds"),
            f"{scene_id}.{region_id}.upstream_onset",
        )
        upstream_offset = _finite_number(
            proposal.get("upstream_clip_offset_seconds"),
            f"{scene_id}.{region_id}.upstream_offset",
        )
        expected_onset = max(0.0, upstream_onset - window_start_seconds)
        expected_offset = min(
            DEFAULT_BENCHMARK_WINDOW_SECONDS,
            upstream_offset - window_start_seconds,
        )
        expected_truncated = (
            upstream_onset < window_start_seconds
            or upstream_offset > window_start_seconds + DEFAULT_BENCHMARK_WINDOW_SECONDS
        )
        if (
            abs(expected_onset - onset) > 1e-7
            or abs(expected_offset - offset) > 1e-7
            or proposal.get("truncated_by_benchmark_window") is not expected_truncated
            or (expected_truncated and proposal.get("eligible_proposal") is not False)
        ):
            raise TacosAuditError(
                f"{scene_id}.{region_id} proposal is not crop-relative"
            )
        by_id[region_id] = proposal
    if any(region_id not in by_id for region_id in selected):
        raise TacosAuditError(f"{scene_id} selected region is missing")
    selected_rows = [by_id[region_id] for region_id in selected]
    if any(
        proposal.get("eligible_proposal") is not True
        or proposal.get("selected_chain_member") is not True
        or proposal.get("truncated_by_benchmark_window") is not False
        for proposal in selected_rows
    ):
        raise TacosAuditError(
            f"{scene_id} selected annotation regions are not fully inside the WAV"
        )
    selected_onsets = [float(proposal["onset_seconds"]) for proposal in selected_rows]
    if selected_onsets != sorted(selected_onsets) or len(set(selected_onsets)) != 4:
        raise TacosAuditError(f"{scene_id} selected regions are not in onset order")


def _validate_receipt_binding(
    receipt: Mapping[str, Any], *, packet_row: Mapping[str, Any]
) -> None:
    scene_id = packet_row["scene_id"]
    source = packet_row["source"]
    audio = packet_row["audio"]
    if (
        receipt.get("scene_id") != scene_id
        or receipt.get("selection_partition") != packet_row.get("selection_partition")
        or receipt.get("selection_tier") != packet_row.get("selection_tier")
        or receipt.get("freesound_id") != source.get("freesound_id")
        or receipt.get("filename") != source.get("filename")
        or receipt.get("archive_member") != audio.get("archive_member")
    ):
        raise TacosAuditError(f"{scene_id} receipt identity disagrees with packet")
    crop = receipt.get("crop_recipe")
    if not isinstance(crop, Mapping):
        raise TacosAuditError(f"{scene_id} receipt lacks a crop recipe")
    if (
        crop.get("start_sample_inclusive")
        != audio.get("benchmark_window_start_sample_32k")
        or crop.get("end_sample_exclusive")
        != audio.get("benchmark_window_end_sample_32k")
        or crop.get("output_frames") != CANONICAL_FRAME_COUNT
        or crop.get("sample_rate_hz") != BENCHMARK_SAMPLE_RATE
        or crop.get("channels") != 1
        or crop.get("output_container") != CANONICAL_WAV_FORMAT
        or crop.get("output_subtype") != CANONICAL_WAV_SUBTYPE
        or crop.get("resampling_applied") is not False
        or crop.get("downmixing_applied") is not False
        or crop.get("padding_samples_↓") != 0
        or crop.get("human_labels_used") is not False
        or crop.get("qces_method_outputs_used") is not False
    ):
        raise TacosAuditError(f"{scene_id} receipt crop contract mismatch")
    containment = receipt.get("proposal_containment")
    if (
        not isinstance(containment, Mapping)
        or containment.get("selected_semantic_chain_regions_inside_window_↑") != 4
        or containment.get("selected_semantic_chain_regions_outside_window_↓") != 0
        or containment.get("selected_semantic_chain_regions_truncated_↓") != 0
    ):
        raise TacosAuditError(f"{scene_id} receipt proposal containment mismatch")
    for field in ("local_sha256", "local_pcm_f32le_sha256"):
        value = receipt.get(field)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise TacosAuditError(f"{scene_id} receipt has an invalid {field}")
    if (
        isinstance(receipt.get("local_size_bytes"), bool)
        or not isinstance(receipt.get("local_size_bytes"), int)
        or receipt["local_size_bytes"] <= 0
    ):
        raise TacosAuditError(f"{scene_id} receipt has an invalid local size")


def _validate_decoded_wav_properties(
    properties: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    scene_id: str,
) -> None:
    if (
        properties.get("sample_rate_hz") != BENCHMARK_SAMPLE_RATE
        or properties.get("channels") != 1
        or properties.get("decoded_frames") != CANONICAL_FRAME_COUNT
        or properties.get("decoded_duration_seconds")
        != DEFAULT_BENCHMARK_WINDOW_SECONDS
        or properties.get("container_format") != CANONICAL_WAV_FORMAT
        or properties.get("codec_subtype") != CANONICAL_WAV_SUBTYPE
        or properties.get("nonfinite_sample_count_↓") != 0
        or properties.get("pcm_f32le_sha256") != receipt.get("local_pcm_f32le_sha256")
    ):
        raise TacosAuditError(f"{scene_id} is not the canonical final 10s WAV")
    recorded = receipt.get("local_verified_audio_properties")
    crop = receipt.get("crop_decoded_audio")
    if not isinstance(recorded, Mapping) or not isinstance(crop, Mapping):
        raise TacosAuditError(f"{scene_id} receipt lacks decoded-audio properties")
    for key, value in properties.items():
        if recorded.get(key) != value:
            raise TacosAuditError(f"{scene_id} decoded WAV/receipt property mismatch")
    if (
        recorded.get("file_sha256") != receipt.get("local_sha256")
        or recorded.get("file_size_bytes") != receipt.get("local_size_bytes")
        or crop.get("sample_rate_hz") != BENCHMARK_SAMPLE_RATE
        or crop.get("channels") != 1
        or crop.get("decoded_frames") != CANONICAL_FRAME_COUNT
        or crop.get("decoded_duration_seconds") != DEFAULT_BENCHMARK_WINDOW_SECONDS
        or crop.get("nonfinite_sample_count_↓") != 0
        or crop.get("pcm_f32le_sha256") != receipt.get("local_pcm_f32le_sha256")
    ):
        raise TacosAuditError(f"{scene_id} decoded crop/receipt binding mismatch")


def response_path(response_root: Path, rater_id: str, scene_id: str) -> Path:
    if not RATER_ID_RE.fullmatch(rater_id):
        raise TacosAuditError("rater_id must match [A-Za-z0-9][A-Za-z0-9_-]{1,31}")
    if not re.fullmatch(r"tacos_[0-9]+", scene_id):
        raise TacosAuditError("invalid scene_id")
    return response_root.resolve() / rater_id / f"{scene_id}.json"


def _tristate(value: Any, context: str) -> str:
    if value not in TRISTATE:
        raise TacosAuditError(f"{context} must be one of {TRISTATE}")
    return value


def validate_response(
    response: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    packet_fingerprint: str,
    audio_receipt_fingerprint: str,
) -> dict[str, Any]:
    expected_fields = {
        "format",
        "packet_fingerprint",
        "audio_receipt_fingerprint",
        "scene_id",
        "rater_id",
        "saved_at_utc",
        "scene_decision",
        "selected_region_ids",
        "regions",
        "event_inventory_complete_for_relations",
        "adjacent_onset_relations_verified",
        "absent_anchor_judgments",
        "full_listen_confirmation",
        "model_outputs_visible",
        "notes",
    }
    if set(response) != expected_fields:
        raise TacosAuditError("human response fields mismatch")
    if response.get("format") != RESPONSE_FORMAT:
        raise TacosAuditError("human response format mismatch")
    if response.get("packet_fingerprint") != packet_fingerprint:
        raise TacosAuditError("human response packet fingerprint mismatch")
    if response.get("audio_receipt_fingerprint") != audio_receipt_fingerprint:
        raise TacosAuditError("human response audio fingerprint mismatch")
    if response.get("scene_id") != task.get("scene_id"):
        raise TacosAuditError("human response scene mismatch")
    rater_id = response.get("rater_id")
    if not isinstance(rater_id, str) or not RATER_ID_RE.fullmatch(rater_id):
        raise TacosAuditError("invalid rater_id")
    if response.get("scene_decision") not in SCENE_DECISIONS:
        raise TacosAuditError("invalid scene_decision")
    if response.get("model_outputs_visible") is not False:
        raise TacosAuditError("dataset annotators must remain blind to model outputs")
    listen_confirmation = response.get("full_listen_confirmation")
    expected_listen_fields = {
        "entire_final_wav_listened",
        "file_sha256",
        "pcm_f32le_sha256",
        "sample_rate_hz",
        "num_frames",
        "duration_seconds",
    }
    if (
        not isinstance(listen_confirmation, dict)
        or set(listen_confirmation) != expected_listen_fields
    ):
        raise TacosAuditError("full-listen confirmation fields mismatch")
    if (
        listen_confirmation.get("entire_final_wav_listened") is not True
        or listen_confirmation.get("file_sha256") != task.get("audio_receipt_sha256")
        or listen_confirmation.get("pcm_f32le_sha256")
        != task.get("audio_receipt_pcm_f32le_sha256")
        or listen_confirmation.get("sample_rate_hz") != BENCHMARK_SAMPLE_RATE
        or listen_confirmation.get("num_frames") != CANONICAL_FRAME_COUNT
        or _finite_number(
            listen_confirmation.get("duration_seconds"),
            "full-listen duration",
        )
        != DEFAULT_BENCHMARK_WINDOW_SECONDS
    ):
        raise TacosAuditError(
            "response must confirm listening to the exact final ten-second WAV"
        )
    if (
        not isinstance(response.get("saved_at_utc"), str)
        or not response["saved_at_utc"]
    ):
        raise TacosAuditError("saved_at_utc is required")
    if not isinstance(response.get("notes"), str):
        raise TacosAuditError("notes must be a string")
    _tristate(
        response.get("event_inventory_complete_for_relations"),
        "event inventory",
    )
    adjacency = response.get("adjacent_onset_relations_verified")
    if not isinstance(adjacency, list) or len(adjacency) != 3:
        raise TacosAuditError("three adjacency judgments are required")
    for index, value in enumerate(adjacency):
        _tristate(value, f"adjacency[{index}]")

    proposed = {
        row["region_id"]: row
        for row in task.get("proposal_regions", [])
        if row.get("eligible_proposal") is True
    }
    selected = response.get("selected_region_ids")
    if (
        not isinstance(selected, list)
        or len(selected) != 4
        or len(set(selected)) != 4
        or any(value not in proposed for value in selected)
    ):
        raise TacosAuditError("exactly four unique eligible regions must be selected")
    selected_sorted = sorted(
        selected, key=lambda region_id: proposed[region_id]["onset_seconds"]
    )
    if selected != selected_sorted:
        raise TacosAuditError("selected_region_ids must be in onset order")

    regions = response.get("regions")
    if not isinstance(regions, list) or len(regions) != 4:
        raise TacosAuditError("exactly four region judgments are required")
    parsed_regions: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        fields = {
            "region_id",
            "audible",
            "proposal_caption_accurate",
            "canonical_event_phrase",
            "verified_onset_seconds",
            "verified_offset_seconds",
            "contamination_rating",
            "salience_1_to_5",
        }
        if not isinstance(region, dict) or set(region) != fields:
            raise TacosAuditError(f"region judgment {index} fields mismatch")
        if region.get("region_id") != selected[index]:
            raise TacosAuditError("region judgments do not follow selected order")
        _tristate(region.get("audible"), f"region[{index}].audible")
        _tristate(
            region.get("proposal_caption_accurate"),
            f"region[{index}].proposal_caption_accurate",
        )
        phrase = region.get("canonical_event_phrase")
        if not isinstance(phrase, str):
            raise TacosAuditError("canonical_event_phrase must be a string")
        phrase = phrase.strip()
        onset = region.get("verified_onset_seconds")
        offset = region.get("verified_offset_seconds")
        if (
            isinstance(onset, bool)
            or isinstance(offset, bool)
            or not isinstance(onset, (int, float))
            or not isinstance(offset, (int, float))
            or not math.isfinite(float(onset))
            or not math.isfinite(float(offset))
            or not 0.0
            <= float(onset)
            < float(offset)
            <= DEFAULT_BENCHMARK_WINDOW_SECONDS
        ):
            raise TacosAuditError(f"region[{index}] has invalid verified interval")
        if region.get("contamination_rating") not in CONTAMINATION:
            raise TacosAuditError("invalid contamination rating")
        salience = region.get("salience_1_to_5")
        if (
            isinstance(salience, bool)
            or not isinstance(salience, int)
            or not 1 <= salience <= 5
        ):
            raise TacosAuditError("salience must be an integer in [1, 5]")
        parsed_regions.append({**region, "canonical_event_phrase": phrase})

    allowed_absent = task["question_contract"]["absent_anchor_candidates"]
    absent_judgments = response.get("absent_anchor_judgments")
    if not isinstance(absent_judgments, list) or len(absent_judgments) != len(
        allowed_absent
    ):
        raise TacosAuditError("every proposed absent anchor requires one judgment")
    parsed_absent_judgments: list[dict[str, str]] = []
    for index, (candidate, judgment) in enumerate(
        zip(allowed_absent, absent_judgments, strict=True)
    ):
        if (
            not isinstance(judgment, dict)
            or set(judgment) != {"caption", "confirmed_inaudible"}
            or judgment.get("caption") != candidate
        ):
            raise TacosAuditError(
                f"absent anchor judgment {index} is not bound to its packet caption"
            )
        verdict = _tristate(
            judgment.get("confirmed_inaudible"),
            f"absent_anchor_judgments[{index}]",
        )
        parsed_absent_judgments.append(
            {"caption": candidate, "confirmed_inaudible": verdict}
        )

    if response["scene_decision"] == "accept":
        if response["event_inventory_complete_for_relations"] != "yes":
            raise TacosAuditError("accepted scene requires a complete event inventory")
        if any(value != "yes" for value in adjacency):
            raise TacosAuditError("accepted scene requires all adjacency checks")
        absent_verdicts = [
            judgment["confirmed_inaudible"] for judgment in parsed_absent_judgments
        ]
        if "uncertain" in absent_verdicts:
            raise TacosAuditError(
                "accepted scene requires a definite judgment for every absent anchor"
            )
        if "yes" not in absent_verdicts:
            raise TacosAuditError(
                "accepted scene requires at least one verified absent anchor"
            )
        if any(region["audible"] != "yes" for region in parsed_regions):
            raise TacosAuditError("accepted scene requires four audible regions")
        if any(
            region["proposal_caption_accurate"] != "yes" for region in parsed_regions
        ):
            raise TacosAuditError(
                "accepted scene requires every immutable proposal caption to "
                "accurately describe its audible event"
            )
        if any(region["salience_1_to_5"] < 3 for region in parsed_regions):
            raise TacosAuditError("accepted scene requires salience >= 3")
        if any(
            region["contamination_rating"] == "uncertain" for region in parsed_regions
        ):
            raise TacosAuditError(
                "accepted scene requires clean/mixed contamination judgments"
            )
        phrases = [
            normalize_caption(region["canonical_event_phrase"])
            for region in parsed_regions
        ]
        if any(not phrase for phrase in phrases) or len(set(phrases)) != 4:
            raise TacosAuditError("accepted scene requires four distinct event phrases")

    return {
        **response,
        "regions": parsed_regions,
        "absent_anchor_judgments": parsed_absent_judgments,
    }


def save_response(
    *,
    response_root: Path,
    response: Mapping[str, Any],
    task: Mapping[str, Any],
    packet_fingerprint: str,
    audio_receipt_fingerprint: str,
) -> Path:
    parsed = validate_response(
        response,
        task=task,
        packet_fingerprint=packet_fingerprint,
        audio_receipt_fingerprint=audio_receipt_fingerprint,
    )
    path = response_path(response_root, parsed["rater_id"], parsed["scene_id"])
    atomic_json(path, parsed, overwrite=True)
    return path
