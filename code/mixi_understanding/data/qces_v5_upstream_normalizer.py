"""Strict official-upstream normalizer for the QCES v5 FUSS/FSD50K route.

The source-ledger auditor consumes deliberately small normalized JSONL files.
This module is the reproducible bridge from the official FUSS v1.3 split lists
and FSD50K v1.0 CSV/JSON metadata to those files.  It never downloads data.

FSD50K ground-truth labels are ontology-smeared.  The one intervention label
is therefore taken from ``collection_dev.csv`` or ``collection_eval.csv`` and
cross-checked against ground truth.  The FUSS split always comes from the FUSS
list containing the source, never from FSD50K's native ``dev.csv`` split.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import struct
import tempfile
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from mixi_understanding.data.qces_v5_source_ledger import (
    CC0_SPDX,
    CC0_URL,
    FSD50K_VERSION,
    FUSS_SPLITS,
    FUSS_VERSION,
    fsd50k_uploader_key,
    sha256_file,
)


REPORT_FORMAT = "qces_v5_upstream_normalization_report_v1"
FUSS_MANIFEST_NAME = "fuss_sources.jsonl"
FSD50K_MANIFEST_NAME = "fsd50k_labels_provenance.jsonl"
REJECTIONS_NAME = "qces_v5_upstream_normalization_rejections.jsonl"
REPORT_NAME = "qces_v5_upstream_normalization_report.json"
LIST_KINDS = ("foreground", "background")
_SOURCE_ID_RE = re.compile(r"^[1-9][0-9]*$")
_DEV_GROUND_TRUTH_FIELDS = ("fname", "labels", "mids", "split")
_EVAL_GROUND_TRUTH_FIELDS = ("fname", "labels", "mids")
_COLLECTION_FIELDS = ("fname", "labels", "mids")
_CLIP_REQUIRED_FIELDS = {"title", "license", "uploader"}
_CC0_URLS = {
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "https://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/publicdomain/zero/1.0",
    "https://creativecommons.org/publicdomain/zero/1.0",
}


class UpstreamNormalizationError(ValueError):
    """Raised for structural or provenance-breaking upstream input errors."""


@dataclass(frozen=True)
class ListedSource:
    source_id: str
    split: str
    list_kind: str
    list_path: Path
    entry: str
    audio_path: str
    audio_abspath: Path


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UpstreamNormalizationError(f"{context} must be a non-empty string")
    if value != value.strip():
        raise UpstreamNormalizationError(
            f"{context} must preserve an exact value without surrounding whitespace"
        )
    return value


def _source_id(value: Any, context: str) -> str:
    result = _text(value, context)
    if not _SOURCE_ID_RE.fullmatch(result):
        raise UpstreamNormalizationError(
            f"{context} must be a canonical positive numeric Freesound ID"
        )
    return result


def _metric(value: int | float, direction: str) -> Dict[str, Any]:
    if direction not in {"↑", "↓"}:
        raise AssertionError("metric direction must be an arrow")
    return {"value": value, "direction": direction}


def _artifact(path: Path, role: str) -> Dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise UpstreamNormalizationError(f"missing {role}: {path}")
    return {
        "role": role,
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _relative_under(path: Path, root: Path, context: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise UpstreamNormalizationError(
            f"{context} must resolve under {root.resolve()}"
        ) from error


def _parse_list_entry(raw: str, *, split: str, context: str) -> Tuple[str, str]:
    entry = _text(raw, context)
    path = PurePosixPath(entry)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != entry:
        raise UpstreamNormalizationError(f"{context} is not a safe POSIX path")
    if len(path.parts) != 3 or path.parts[0] != split or path.parts[1] != "sound":
        raise UpstreamNormalizationError(
            f"{context} must match {split}/sound/<FREESOUND_ID>.wav"
        )
    if path.suffix.lower() != ".wav":
        raise UpstreamNormalizationError(f"{context} must name a WAV file")
    return _source_id(path.stem, f"{context} source ID"), entry


def load_fuss_lists(
    *, fuss_root: Path, fuss_data_dir: Path, splits: Sequence[str]
) -> Tuple[List[ListedSource], List[Dict[str, Any]]]:
    """Load canonical foreground/background lists for requested FUSS splits."""

    fuss_root = fuss_root.resolve()
    fuss_data_dir = fuss_data_dir.resolve()
    _relative_under(fuss_data_dir, fuss_root, "FUSS data directory")
    requested = tuple(splits)
    if not requested or len(set(requested)) != len(requested):
        raise UpstreamNormalizationError("splits must be non-empty and unique")
    invalid = sorted(set(requested) - set(FUSS_SPLITS))
    if invalid:
        raise UpstreamNormalizationError(f"unsupported FUSS splits: {invalid}")

    sources: List[ListedSource] = []
    artifacts: List[Dict[str, Any]] = []
    seen_ids: Dict[str, str] = {}
    for split in FUSS_SPLITS:
        if split not in requested:
            continue
        for list_kind in LIST_KINDS:
            list_path = fuss_data_dir / f"{split}_{list_kind}.txt"
            artifacts.append(_artifact(list_path, f"fuss_{split}_{list_kind}_list"))
            try:
                raw_lines = list_path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError) as error:
                raise UpstreamNormalizationError(
                    f"cannot read FUSS list {list_path}: {error}"
                ) from error
            if not raw_lines:
                raise UpstreamNormalizationError(f"empty FUSS list: {list_path}")
            for line_number, raw in enumerate(raw_lines, start=1):
                source_id, entry = _parse_list_entry(
                    raw,
                    split=split,
                    context=f"{list_path}:{line_number}",
                )
                prior = seen_ids.get(source_id)
                if prior is not None:
                    raise UpstreamNormalizationError(
                        f"duplicate FUSS source ID {source_id}: {prior} and "
                        f"{list_path}:{line_number}"
                    )
                seen_ids[source_id] = f"{list_path}:{line_number}"
                audio_abspath = (fuss_data_dir / entry).resolve()
                audio_path = _relative_under(
                    audio_abspath, fuss_root, f"FUSS source {source_id} audio"
                )
                sources.append(
                    ListedSource(
                        source_id=source_id,
                        split=split,
                        list_kind=list_kind,
                        list_path=list_path,
                        entry=entry,
                        audio_path=audio_path,
                        audio_abspath=audio_abspath,
                    )
                )
    sources.sort(key=lambda row: (int(row.source_id), row.source_id))
    return sources, artifacts


def _csv_rows(path: Path, expected_fields: Tuple[str, ...], context: str):
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise UpstreamNormalizationError(f"cannot read {context}: {path}: {error}")
    with handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected_fields:
            raise UpstreamNormalizationError(
                f"{context} header mismatch: expected={expected_fields}, "
                f"observed={tuple(reader.fieldnames or ())}"
            )
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise UpstreamNormalizationError(
                    f"{context}:{line_number} contains unexpected extra columns"
                )
            yield line_number, row


def _load_ground_truth(path: Path, *, partition: str) -> Dict[str, Tuple[str, ...]]:
    expected = (
        _DEV_GROUND_TRUTH_FIELDS if partition == "dev" else _EVAL_GROUND_TRUTH_FIELDS
    )
    result: Dict[str, Tuple[str, ...]] = {}
    for line_number, row in _csv_rows(path, expected, f"FSD50K {partition} ground truth"):
        source_id = _source_id(row["fname"], f"{path}:{line_number}.fname")
        if source_id in result:
            raise UpstreamNormalizationError(
                f"duplicate FSD50K ground-truth source ID {source_id} in {path}"
            )
        labels = tuple(label for label in row["labels"].split(",") if label)
        if not labels or len(set(labels)) != len(labels):
            raise UpstreamNormalizationError(
                f"{path}:{line_number}.labels must contain unique labels"
            )
        if any(label != label.strip() for label in labels):
            raise UpstreamNormalizationError(
                f"{path}:{line_number}.labels contains surrounding whitespace"
            )
        result[source_id] = labels
    return result


def _load_collection(
    path: Path, *, partition: str
) -> Dict[str, Tuple[Tuple[str, str], ...]]:
    grouped: MutableMapping[str, set[Tuple[str, str]]] = defaultdict(set)
    for line_number, row in _csv_rows(
        path, _COLLECTION_FIELDS, f"FSD50K {partition} collection"
    ):
        source_id = _source_id(row["fname"], f"{path}:{line_number}.fname")
        # The official collection files retain some unlabelled collection
        # candidates as rows with both fields empty.  Preserve the source ID so
        # a FUSS match becomes an explicit zero-label rejection rather than a
        # structural parser failure or a false "unmatched" claim.
        grouped[source_id]
        if row["labels"] == "" and row["mids"] == "":
            continue
        if (row["labels"] == "") != (row["mids"] == ""):
            raise UpstreamNormalizationError(
                f"{path}:{line_number} must have both label and mid or neither"
            )
        labels = row["labels"].split(",")
        mids = row["mids"].split(",")
        if len(labels) != len(mids):
            raise UpstreamNormalizationError(
                f"{path}:{line_number} label/mid counts do not match"
            )
        for index, (label_raw, mid_raw) in enumerate(zip(labels, mids)):
            label = _text(
                label_raw, f"{path}:{line_number}.labels[{index}]"
            )
            mid = _text(mid_raw, f"{path}:{line_number}.mids[{index}]")
            grouped[source_id].add((label, mid))
    return {
        source_id: tuple(sorted(values))
        for source_id, values in grouped.items()
    }


def _no_duplicate_object(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UpstreamNormalizationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_clips_info(path: Path, *, partition: str) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_no_duplicate_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise UpstreamNormalizationError(
            f"cannot read FSD50K {partition} clips info {path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise UpstreamNormalizationError(
            f"FSD50K {partition} clips info must be a JSON object"
        )
    return payload


def _partition_inputs(
    *, fsd50k_ground_truth_dir: Path, fsd50k_metadata_dir: Path, partition: str
) -> Tuple[
    Dict[str, Tuple[str, ...]],
    Dict[str, Tuple[Tuple[str, str], ...]],
    Mapping[str, Any],
    List[Dict[str, Any]],
]:
    gt_path = fsd50k_ground_truth_dir / f"{'dev' if partition == 'dev' else 'eval'}.csv"
    collection_path = (
        fsd50k_metadata_dir / "collection" / f"collection_{partition}.csv"
    )
    clips_path = fsd50k_metadata_dir / f"{partition}_clips_info_FSD50K.json"
    artifacts = [
        _artifact(gt_path, f"fsd50k_{partition}_ground_truth"),
        _artifact(collection_path, f"fsd50k_{partition}_collection"),
        _artifact(clips_path, f"fsd50k_{partition}_clips_info"),
    ]
    return (
        _load_ground_truth(gt_path, partition=partition),
        _load_collection(collection_path, partition=partition),
        _load_clips_info(clips_path, partition=partition),
        artifacts,
    )


def _issue(row: ListedSource, reason: str, **details: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "source_id": row.source_id,
        "split": row.split,
        "stage": "upstream_normalization",
        "reason": reason,
    }
    payload.update(details)
    return payload


def _clip_metadata(
    source_id: str, payload: Any, context: str
) -> Tuple[str, str, str]:
    if not isinstance(payload, dict):
        raise UpstreamNormalizationError(f"{context} must be a JSON object")
    missing = sorted(_CLIP_REQUIRED_FIELDS - set(payload))
    if missing:
        raise UpstreamNormalizationError(f"{context} missing fields: {missing}")
    uploader_name = _text(payload["uploader"], f"{context}.uploader")
    license_url = _text(payload["license"], f"{context}.license")
    if license_url not in _CC0_URLS:
        raise UpstreamNormalizationError(
            f"{context}.license is not CC0-1.0: {license_url!r}"
        )
    title_raw = payload["title"]
    if not isinstance(title_raw, str):
        raise UpstreamNormalizationError(f"{context}.title must be a string")
    title = " ".join(title_raw.split()) or "untitled source"
    attribution = (
        f'Freesound ID {source_id}, "{title}", uploaded by {uploader_name}; '
        f"https://freesound.org/s/{source_id}/"
    )
    return uploader_name, CC0_URL, attribution


def _zero_sized_riff_pcm_metadata(
    path: Path, context: str
) -> Tuple[int, int, int, int, str]:
    """Inspect an official FUSS PCM WAV whose RIFF-size field is zero.

    A subset of the official FUSS v1.3 source files has a valid ``WAVE``
    payload and valid ``fmt ``/``data`` chunks, but stores zero in bytes 4--7
    instead of the conventional file-size-minus-eight value.  Python's
    :mod:`wave` rejects those files before inspecting their chunks.  The
    fallback is deliberately narrow: it accepts only that exact zero-size
    anomaly and still bounds every chunk by the physical file size.
    """

    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) != 12:
                raise UpstreamNormalizationError(
                    f"{context} has a truncated RIFF header"
                )
            if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                raise UpstreamNormalizationError(
                    f"{context} is not a little-endian RIFF/WAVE file"
                )
            (declared_riff_size,) = struct.unpack("<I", header[4:8])
            if declared_riff_size != 0:
                raise UpstreamNormalizationError(
                    f"{context} does not have the documented zero RIFF-size anomaly"
                )

            fmt_payload: bytes | None = None
            data_size: int | None = None
            offset = 12
            while offset + 8 <= file_size:
                handle.seek(offset)
                chunk_header = handle.read(8)
                if len(chunk_header) != 8:
                    raise UpstreamNormalizationError(
                        f"{context} has a truncated chunk header at byte {offset}"
                    )
                chunk_id = chunk_header[:4]
                (chunk_size,) = struct.unpack("<I", chunk_header[4:])
                payload_start = offset + 8
                payload_end = payload_start + chunk_size
                padded_end = payload_end + (chunk_size & 1)
                if payload_end > file_size or padded_end > file_size:
                    raise UpstreamNormalizationError(
                        f"{context} chunk {chunk_id!r} exceeds the physical file size"
                    )
                if chunk_id == b"fmt " and fmt_payload is None:
                    handle.seek(payload_start)
                    fmt_payload = handle.read(chunk_size)
                    if len(fmt_payload) != chunk_size:
                        raise UpstreamNormalizationError(
                            f"{context} has a truncated fmt chunk"
                        )
                elif chunk_id == b"data" and data_size is None:
                    data_size = chunk_size
                offset = padded_end
    except OSError as error:
        raise UpstreamNormalizationError(f"cannot inspect {context}: {error}") from error

    if fmt_payload is None or len(fmt_payload) < 16:
        raise UpstreamNormalizationError(f"{context} is missing a valid fmt chunk")
    if data_size is None:
        raise UpstreamNormalizationError(f"{context} is missing a data chunk")
    (
        audio_format,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
    ) = struct.unpack("<HHIIHH", fmt_payload[:16])
    if audio_format != 1:
        raise UpstreamNormalizationError(
            f"{context} zero-size RIFF fallback only supports integer PCM"
        )
    sample_width = bits_per_sample // 8
    expected_block_align = channels * sample_width
    expected_byte_rate = sample_rate * expected_block_align
    if (
        bits_per_sample % 8 != 0
        or block_align != expected_block_align
        or byte_rate != expected_byte_rate
        or block_align <= 0
        or data_size % block_align != 0
    ):
        raise UpstreamNormalizationError(
            f"{context} has inconsistent PCM fmt/data sizes"
        )
    frames = data_size // block_align
    return channels, sample_width, sample_rate, frames, "NONE"


def _wav_duration(path: Path, context: str) -> Tuple[float, bool]:
    used_zero_sized_riff_fallback = False
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
            compression = handle.getcomptype()
    except (EOFError, wave.Error) as wave_error:
        try:
            channels, sample_width, sample_rate, frames, compression = (
                _zero_sized_riff_pcm_metadata(path, context)
            )
            used_zero_sized_riff_fallback = True
        except UpstreamNormalizationError as fallback_error:
            raise UpstreamNormalizationError(
                f"cannot inspect {context}: {wave_error}; "
                f"zero-size fallback rejected it: {fallback_error}"
            ) from wave_error
    except OSError as error:
        raise UpstreamNormalizationError(f"cannot inspect {context}: {error}") from error
    if (channels, sample_width, sample_rate, compression) != (1, 2, 16_000, "NONE"):
        raise UpstreamNormalizationError(
            f"{context} must be mono PCM16 16 kHz WAV; observed channels={channels}, "
            f"sample_width={sample_width}, sample_rate={sample_rate}, "
            f"compression={compression}"
        )
    if frames <= 0:
        raise UpstreamNormalizationError(f"{context} contains no audio frames")
    duration = frames / sample_rate
    if not math.isfinite(duration) or duration <= 0.0:
        raise UpstreamNormalizationError(f"{context} has invalid duration")
    return duration, used_zero_sized_riff_fallback


def _reason_count(rejections: Sequence[Mapping[str, Any]], reason: str) -> int:
    return sum(row.get("reason") == reason for row in rejections)


def normalize_official_upstream(
    *,
    fuss_root: Path,
    fuss_data_dir: Path,
    fsd50k_ground_truth_dir: Path,
    fsd50k_metadata_dir: Path,
    splits: Sequence[str] = FUSS_SPLITS,
    metadata_only: bool = False,
) -> Dict[str, Any]:
    """Normalize official inputs and return rows plus an auditable report.

    Row-level defects are rejected and reported, not raised.  This is required
    for the known prerelease-to-v1.0 FUSS/FSD50K mismatch.  Structural input
    corruption, duplicate FUSS IDs, or unsafe paths remain hard errors.
    """

    requested_splits = tuple(splits)
    invalid_splits = sorted(set(requested_splits) - set(FUSS_SPLITS))
    if invalid_splits:
        raise UpstreamNormalizationError(
            f"unsupported FUSS splits: {invalid_splits}"
        )
    ordered_splits = tuple(
        split for split in FUSS_SPLITS if split in set(requested_splits)
    )
    sources, artifacts = load_fuss_lists(
        fuss_root=fuss_root,
        fuss_data_dir=fuss_data_dir,
        splits=ordered_splits,
    )
    needed_partitions = []
    if any(split in {"train", "validation"} for split in ordered_splits):
        needed_partitions.append("dev")
    if "eval" in ordered_splits:
        needed_partitions.append("eval")
    fsd_inputs: Dict[
        str,
        Tuple[
            Dict[str, Tuple[str, ...]],
            Dict[str, Tuple[Tuple[str, str], ...]],
            Mapping[str, Any],
        ],
    ] = {}
    for partition in needed_partitions:
        ground_truth, collection, clips_info, partition_artifacts = _partition_inputs(
            fsd50k_ground_truth_dir=fsd50k_ground_truth_dir.resolve(),
            fsd50k_metadata_dir=fsd50k_metadata_dir.resolve(),
            partition=partition,
        )
        fsd_inputs[partition] = (ground_truth, collection, clips_info)
        artifacts.extend(partition_artifacts)

    fuss_rows: List[Dict[str, Any]] = []
    fsd_rows: List[Dict[str, Any]] = []
    rejections: List[Dict[str, Any]] = []
    metadata_matched_count = 0
    zero_sized_riff_fallback_count = 0
    for row in sources:
        partition = "eval" if row.split == "eval" else "dev"
        ground_truth, collection, clips_info = fsd_inputs[partition]
        missing_inputs = [
            name
            for name, mapping in (
                ("ground_truth", ground_truth),
                ("collection", collection),
                ("clips_info", clips_info),
            )
            if row.source_id not in mapping
        ]
        if missing_inputs:
            rejections.append(
                _issue(
                    row,
                    "unmatched_fsd50k_source_id",
                    missing_inputs=missing_inputs,
                    expected_fsd50k_partition=partition,
                )
            )
            continue

        label_mid_pairs = collection[row.source_id]
        labels = sorted({label for label, _ in label_mid_pairs})
        if len(labels) != 1 or len(label_mid_pairs) != 1:
            rejections.append(
                _issue(
                    row,
                    "ambiguous_collection_label",
                    observed_labels=labels,
                    observed_label_mid_pair_count=len(label_mid_pairs),
                )
            )
            continue
        label = labels[0]
        if label not in ground_truth[row.source_id]:
            rejections.append(
                _issue(
                    row,
                    "collection_label_not_in_ground_truth",
                    collection_label=label,
                    ground_truth_labels=list(ground_truth[row.source_id]),
                )
            )
            continue
        try:
            uploader_name, license_url, attribution = _clip_metadata(
                row.source_id,
                clips_info[row.source_id],
                f"FSD50K {partition} clips_info[{row.source_id}]",
            )
        except UpstreamNormalizationError as error:
            reason = (
                "source_license_not_cc0"
                if "not CC0-1.0" in str(error)
                else "invalid_clips_info"
            )
            rejections.append(_issue(row, reason, detail=str(error)))
            continue

        identity_key = fsd50k_uploader_key(uploader_name)
        metadata_matched_count += 1
        fsd_row = {
            "source_id": row.source_id,
            "split": row.split,
            "labels": [label],
            "creator_id": identity_key,
            "uploader_id": identity_key,
            "uploader_name": uploader_name,
            "attribution": attribution,
            "source_license_spdx": CC0_SPDX,
            "source_license_url": license_url,
            "dataset_version": FSD50K_VERSION,
        }
        if metadata_only:
            fsd_rows.append(fsd_row)
            continue
        if not row.audio_abspath.is_file():
            rejections.append(
                _issue(row, "missing_fuss_audio_file", audio_path=row.audio_path)
            )
            continue
        try:
            duration, used_zero_sized_riff_fallback = _wav_duration(
                row.audio_abspath, f"FUSS source {row.source_id} audio"
            )
        except UpstreamNormalizationError as error:
            rejections.append(_issue(row, "invalid_fuss_audio", detail=str(error)))
            continue
        zero_sized_riff_fallback_count += int(used_zero_sized_riff_fallback)
        fuss_rows.append(
            {
                "source_id": row.source_id,
                "split": row.split,
                "audio_path": row.audio_path,
                "sha256": sha256_file(row.audio_abspath),
                "duration_seconds": duration,
                "source_license_spdx": CC0_SPDX,
                "source_license_url": CC0_URL,
                "dataset_version": FUSS_VERSION,
            }
        )
        fsd_rows.append(fsd_row)

    row_key = lambda payload: (int(payload["source_id"]), payload["source_id"])
    fuss_rows.sort(key=row_key)
    fsd_rows.sort(key=row_key)
    rejections.sort(
        key=lambda payload: (
            int(payload["source_id"]),
            payload["source_id"],
            payload["reason"],
        )
    )
    listed_count = len(sources)
    report = {
        "format": REPORT_FORMAT,
        "mode": "metadata_only" if metadata_only else "audio_verified",
        "source_route": "fuss_v1.3_fsd50k_v1.0_official",
        "fuss_splits": list(ordered_splits),
        "complete_fuss_split_scope": set(ordered_splits) == set(FUSS_SPLITS),
        "normalized_pair_complete": (
            not metadata_only and len(fuss_rows) == len(fsd_rows)
        ),
        "conversion_completed": True,
        "release_ready": False,
        "identity_key_scheme": {
            "name": "qces-v5-fsd50k-uploader-v1",
            "source_field": "FSD50K clips_info uploader",
            "input_normalization": "none; exact non-empty UTF-8 username",
            "derivation": (
                "fsd50k-uploader-sha256: + hex(sha256(" 
                "b'qces-v5-fsd50k-uploader-v1\\0' + username_utf8))"
            ),
            "creator_id_equals_uploader_id": True,
            "upstream_numeric_id_claimed": False,
        },
        "label_policy": {
            "selected_from": "FSD50K collection_dev/eval.csv unsmeared label",
            "cross_checked_against": "FSD50K dev/eval.csv smeared labels",
            "required_unique_label_mid_pairs": 1,
        },
        "split_policy": (
            "Normalized split is copied from the containing FUSS list; "
            "FSD50K dev.csv train/val is not reused."
        ),
        "audio_validation_policy": {
            "required_format": "mono integer PCM16, 16000 Hz, uncompressed WAV",
            "standard_parser": "Python wave module",
            "zero_sized_riff_fallback": (
                "Accept only RIFF/WAVE files whose declared RIFF size is zero; "
                "parse and physically bound every chunk, then require consistent "
                "PCM fmt/data sizes. This handles an official FUSS v1.3 anomaly."
            ),
        },
        "input_artifacts": sorted(artifacts, key=lambda item: item["role"]),
        "metrics": {
            "fuss_listed_source_count": _metric(listed_count, "↑"),
            "normalized_fsd50k_row_count": _metric(len(fsd_rows), "↑"),
            "normalized_fuss_row_count": _metric(len(fuss_rows), "↑"),
            "metadata_join_coverage": _metric(
                metadata_matched_count / max(1, listed_count), "↑"
            ),
            "rejected_source_count": _metric(len(rejections), "↓"),
            "unmatched_fsd50k_source_count": _metric(
                _reason_count(rejections, "unmatched_fsd50k_source_id"), "↓"
            ),
            "ambiguous_collection_label_count": _metric(
                _reason_count(rejections, "ambiguous_collection_label"), "↓"
            ),
            "ground_truth_label_mismatch_count": _metric(
                _reason_count(rejections, "collection_label_not_in_ground_truth"),
                "↓",
            ),
            "non_cc0_source_count": _metric(
                _reason_count(rejections, "source_license_not_cc0"), "↓"
            ),
            "invalid_clips_info_count": _metric(
                _reason_count(rejections, "invalid_clips_info"), "↓"
            ),
            "missing_audio_count": _metric(
                _reason_count(rejections, "missing_fuss_audio_file"), "↓"
            ),
            "invalid_audio_count": _metric(
                _reason_count(rejections, "invalid_fuss_audio"), "↓"
            ),
            "official_zero_sized_riff_recovered_count": _metric(
                zero_sized_riff_fallback_count, "↑"
            ),
        },
        "rejection_policy": (
            "Row-level unmatched, ambiguous, non-CC0, malformed, or missing-audio "
            "sources are excluded and reported; they do not abort conversion. "
            "The downstream allocation gate decides feasibility."
        ),
        "non_claim": (
            "This converter proves deterministic normalization only. It does not "
            "prove paper-scale allocation feasibility or release readiness."
        ),
    }
    return {
        "fuss_rows": fuss_rows,
        "fsd50k_rows": fsd_rows,
        "rejections": rejections,
        "report": report,
    }


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {path}; use --overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_normalized_outputs(
    result: Mapping[str, Any], *, output_dir: Path, overwrite: bool = False
) -> Dict[str, Path]:
    """Atomically write deterministic manifests, rejections, and report."""

    output_dir = output_dir.resolve()
    metadata_only = result["report"]["mode"] == "metadata_only"
    fuss_path = output_dir / FUSS_MANIFEST_NAME
    fsd_path = output_dir / FSD50K_MANIFEST_NAME
    rejections_path = output_dir / REJECTIONS_NAME
    report_path = output_dir / REPORT_NAME
    if metadata_only and fuss_path.exists():
        raise FileExistsError(
            f"metadata-only conversion refuses to leave or overwrite stale full "
            f"manifest {fuss_path}; use a fresh output directory"
        )
    outputs = [fsd_path, rejections_path, report_path]
    if not metadata_only:
        outputs.append(fuss_path)
    if not overwrite:
        existing = [path for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "output exists: " + ", ".join(str(path) for path in existing)
            )

    fsd_bytes = _jsonl_bytes(result["fsd50k_rows"])
    rejection_bytes = _jsonl_bytes(result["rejections"])
    fuss_bytes = _jsonl_bytes(result["fuss_rows"])
    _atomic_bytes(fsd_path, fsd_bytes, overwrite=overwrite)
    _atomic_bytes(rejections_path, rejection_bytes, overwrite=overwrite)
    if not metadata_only:
        _atomic_bytes(fuss_path, fuss_bytes, overwrite=overwrite)

    report = dict(result["report"])
    report["output_artifacts"] = {
        "fsd50k_manifest": {
            "path": str(fsd_path),
            "row_count": len(result["fsd50k_rows"]),
            "sha256": hashlib.sha256(fsd_bytes).hexdigest(),
        },
        "rejections": {
            "path": str(rejections_path),
            "row_count": len(result["rejections"]),
            "sha256": hashlib.sha256(rejection_bytes).hexdigest(),
        },
    }
    if not metadata_only:
        report["output_artifacts"]["fuss_manifest"] = {
            "path": str(fuss_path),
            "row_count": len(result["fuss_rows"]),
            "sha256": hashlib.sha256(fuss_bytes).hexdigest(),
        }
    report_bytes = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_bytes(report_path, report_bytes, overwrite=overwrite)
    paths = {
        "fsd50k_manifest": fsd_path,
        "rejections": rejections_path,
        "report": report_path,
    }
    if not metadata_only:
        paths["fuss_manifest"] = fuss_path
    return paths
