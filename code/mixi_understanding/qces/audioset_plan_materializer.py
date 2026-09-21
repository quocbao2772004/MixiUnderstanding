"""Resumable row-group materialization for an exact AudioSet-Strong plan.

The implementation deliberately does not use :mod:`datasets` or download
Parquet shards to a local cache.  It first reads only the ``video_id`` column
through a seekable fsspec handle, records the matching Parquet row groups, and
then reads ``video_id`` plus ``audio`` only for those matching row groups.

Durability is row-group transactional: audio files are written atomically,
then a deterministic manifest fragment is written atomically, and finally the
manifest index and resumable state are replaced atomically.  On restart the
fragments, rather than the last in-memory state, are the source of truth.
"""

from __future__ import annotations

import hashlib
import fcntl
import io
import json
import math
import os
import re
import shutil
import statistics
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import fsspec
import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAT = "qces_audioset_strong_plan_materializer_v1"
MANIFEST_FORMAT = "qces_audioset_strong_plan_manifest_v1"
SOURCE_ROUTE = "audioset_strong_hf_pinned_parquet_v1"
STORAGE_MODES = {"full_clips", "requested_crops"}
DEFAULT_PARQUET_PATTERN = (
    "hf://datasets/enyoukai/AudioSet-Strong@{revision}/data/{split}-*.parquet"
)
# YouTube-derived AudioSet clips can decode a few milliseconds shorter than
# their nominal annotation timeline.  Requested-crop mode may pad only this
# small tail discrepancy.  The retained-coverage fields are still computed
# against real decoded samples, so a wholly missing tail event is rejected by
# the downstream acoustic-quality gate rather than becoming a false positive.
# Some pinned AudioSet-Strong rows decode close to 9 s even though their
# official annotation clock extends to 10 s. Requested crops that still begin
# inside real decoded audio may be padded and terminally accounted for; the
# source cleaner rejects every row whose coverage event was not fully decoded.
# A 1.25 s cap covers that known container mismatch without permitting an
# unbounded/corrupt plan to allocate arbitrary silence.
MAX_NOMINAL_TAIL_PADDING_SECONDS = 1.250
RETRYABLE_REMOTE_IO_ERRORS = (
    OSError,
    TimeoutError,
    ConnectionError,
    pa.ArrowIOError,
    httpx.HTTPError,
)


class MaterializationError(RuntimeError):
    """Raised when completeness or integrity cannot be guaranteed."""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_delay_seconds: float = 0.5
    maximum_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_delay_seconds < 0 or self.maximum_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")


@dataclass(frozen=True)
class MaterializationConfig:
    plan_paths: tuple[Path, ...]
    output_dir: Path
    parquet_pattern: str = DEFAULT_PARQUET_PATTERN
    hf_dataset: str = "enyoukai/AudioSet-Strong"
    resolved_revision: str = "main"
    storage_mode: str = "full_clips"
    existing_manifest_paths: tuple[Path, ...] = ()
    require_preindexed_locations: bool = False
    scan_only: bool = False
    max_new_videos: int = 0
    block_size_bytes: int = 1 << 20
    minimum_free_disk_bytes: int = 10 * (1 << 30)
    disk_estimate_safety_factor: float = 1.15
    scan_workers: int = 4
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        if not self.plan_paths:
            raise ValueError("at least one plan path is required")
        if not self.hf_dataset.strip():
            raise ValueError("hf_dataset must be non-empty")
        if self.max_new_videos < 0:
            raise ValueError("max_new_videos must be >= 0")
        if self.storage_mode not in STORAGE_MODES:
            raise ValueError(
                f"storage_mode must be one of {sorted(STORAGE_MODES)}"
            )
        if self.block_size_bytes <= 0:
            raise ValueError("block_size_bytes must be positive")
        if self.minimum_free_disk_bytes < 0:
            raise ValueError("minimum_free_disk_bytes must be >= 0")
        if self.disk_estimate_safety_factor < 1.0:
            raise ValueError("disk_estimate_safety_factor must be >= 1")
        if self.scan_workers < 1:
            raise ValueError("scan_workers must be >= 1")


CheckpointHook = Callable[[str, Mapping[str, Any]], None]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_bytes(path, json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n")


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    content = b"".join(
        _canonical_json_bytes(row) + b"\n"
        for row in rows
    )
    atomic_bytes(path, content)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise MaterializationError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise MaterializationError(
                    f"expected object at {path}:{line_number}"
                )
            rows.append(row)
    return rows


def _validate_video_id(video_id: str) -> str:
    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise MaterializationError(f"unsafe or empty AudioSet video_id: {video_id!r}")
    return video_id


def load_plans(
    paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str, list[dict[str, Any]]]:
    """Load, validate and fingerprint plan rows in deterministic order."""

    ordered: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    file_receipts: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise MaterializationError(f"plan does not exist: {path}")
        rows = _load_jsonl(path)
        split_values: set[str] = set()
        for row in rows:
            video_id = _validate_video_id(str(row.get("video_id") or ""))
            split = str(row.get("hf_split") or "")
            if split not in {"train", "test"}:
                raise MaterializationError(
                    f"plan row {video_id} has unsupported hf_split={split!r}"
                )
            split_values.add(split)
            events = row.get("events")
            if not isinstance(events, list) or not events:
                raise MaterializationError(f"plan row {video_id} has no events")
            for index, event in enumerate(events):
                if not isinstance(event, dict):
                    raise MaterializationError(
                        f"plan row {video_id} event {index} is not an object"
                    )
                label = str(event.get("label") or "")
                onset = float(event.get("onset_seconds", -1.0))
                offset = float(event.get("offset_seconds", -1.0))
                if not label or onset < 0.0 or offset <= onset:
                    raise MaterializationError(
                        f"invalid event {index} for plan row {video_id}"
                    )
                event_video = str(event.get("video_id") or video_id)
                if event_video != video_id:
                    raise MaterializationError(
                        f"event video_id mismatch for plan row {video_id}: {event_video}"
                    )
            if video_id in by_id:
                if _canonical_json_bytes(by_id[video_id]) != _canonical_json_bytes(row):
                    raise MaterializationError(
                        f"conflicting duplicate video_id across plans: {video_id}"
                    )
                continue
            normalized = json.loads(_canonical_json_bytes(row))
            by_id[video_id] = normalized
            ordered.append(normalized)
        file_receipts.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(rows),
                "hf_splits": sorted(split_values),
            }
        )
    canonical_rows = sorted(
        by_id.values(), key=lambda row: (str(row["hf_split"]), str(row["video_id"]))
    )
    plan_hash = sha256_bytes(_canonical_json_bytes(canonical_rows))
    return ordered, by_id, plan_hash, file_receipts


def _plan_source_route(plan: Mapping[str, Any]) -> str:
    return str(plan.get("source_route") or SOURCE_ROUTE)


def _plan_hf_dataset(plan: Mapping[str, Any]) -> str:
    return str(plan.get("hf_dataset") or "enyoukai/AudioSet-Strong")


def _plan_revision(plan: Mapping[str, Any], fallback: str) -> str:
    return str(plan.get("hf_revision") or fallback)


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def resolve_manifest_audio_path(manifest_path: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (PROJECT_ROOT, manifest_path.parent, Path.cwd()):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return resolved
    return (PROJECT_ROOT / candidate).resolve()


def validate_audio_path(path: Path, declared_sha256: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise MaterializationError(f"audio file does not exist: {path}")
    actual_hash = sha256_file(path)
    if declared_sha256 and actual_hash != declared_sha256:
        raise MaterializationError(
            f"audio SHA-256 mismatch for {path}: {actual_hash} != {declared_sha256}"
        )
    try:
        info = sf.info(path)
    except Exception as error:
        raise MaterializationError(f"soundfile rejected {path}: {error}") from error
    if int(info.frames) <= 0 or int(info.samplerate) <= 0 or float(info.duration) <= 0:
        raise MaterializationError(f"invalid audio metadata for {path}: {info}")
    return {
        "audio_sha256": actual_hash,
        "sample_rate": int(info.samplerate),
        "duration_seconds": float(info.duration),
        "audio_bytes": int(path.stat().st_size),
    }


def validate_audio_bytes(data: bytes) -> dict[str, Any]:
    if not data:
        raise MaterializationError("empty audio payload")
    try:
        info = sf.info(io.BytesIO(data))
    except Exception as error:
        raise MaterializationError(f"soundfile rejected audio bytes: {error}") from error
    if int(info.frames) <= 0 or int(info.samplerate) <= 0 or float(info.duration) <= 0:
        raise MaterializationError(f"invalid decoded audio metadata: {info}")
    return {
        "audio_sha256": sha256_bytes(data),
        "sample_rate": int(info.samplerate),
        "duration_seconds": float(info.duration),
        "audio_bytes": len(data),
    }


def _manifest_events(
    plan: Mapping[str, Any], *, audio_duration_seconds: float
) -> list[dict[str, Any]]:
    video_id = str(plan["video_id"])
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(plan["events"]):
        label = str(raw["label"])
        plan_onset = float(raw["onset_seconds"])
        plan_offset = float(raw["offset_seconds"])
        # A small tail mismatch is normal for YouTube-derived FLACs, which are
        # occasionally shorter than the nominal 10 s metadata segment.  Keep
        # every planned event, but never emit a boundary outside decoded audio.
        outside_audio = plan_onset >= audio_duration_seconds
        onset = max(0.0, min(plan_onset, audio_duration_seconds))
        offset = max(0.0, min(plan_offset, audio_duration_seconds))
        if offset <= onset and not outside_audio:
            raise MaterializationError(
                f"planned event has no audible duration after clipping for {video_id}: "
                f"[{plan_onset:.6f}, {plan_offset:.6f}] vs {audio_duration_seconds:.6f}s"
            )
        output.append(
            {
                "event_id": f"{video_id}_{index:04d}_{label}",
                "event_kind": "semantic",
                "label": label,
                "display_name": str(raw.get("display_name") or label.replace("_", " ")),
                "audioset_mid": str(raw.get("mid") or raw.get("audioset_mid") or ""),
                "onset_seconds": onset,
                "offset_seconds": offset,
                "plan_onset_seconds": plan_onset,
                "plan_offset_seconds": plan_offset,
                "annotation_clipped_to_audio_duration": (
                    onset != plan_onset or offset != plan_offset
                ),
                "annotation_outside_decoded_audio": outside_audio,
                "selected_ontology_label": bool(
                    raw.get("selected_ontology_label", True)
                ),
                "segment_id": str(raw.get("segment_id") or ""),
                "video_id": video_id,
            }
        )
    return output


def build_manifest_row(
    *,
    plan: Mapping[str, Any],
    plan_hash: str,
    audio_path: Path,
    audio_info: Mapping[str, Any],
    resolved_revision: str,
    transport: str,
    parquet: Mapping[str, Any] | None = None,
    existing_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    video_id = str(plan["video_id"])
    hf_split = str(plan["hf_split"])
    source_route = _plan_source_route(plan)
    hf_dataset = _plan_hf_dataset(plan)
    hf_revision = _plan_revision(plan, resolved_revision)
    events = _manifest_events(
        plan, audio_duration_seconds=float(audio_info["duration_seconds"])
    )
    for request in plan.get("crop_requests", []):
        crop_start = float(request.get("crop_start_seconds", -1.0))
        event_onset = float(request.get("event_onset_seconds", -1.0))
        if crop_start >= float(audio_info["duration_seconds"]) or event_onset >= float(
            audio_info["duration_seconds"]
        ):
            raise MaterializationError(
                f"coverage crop lies outside decoded audio for {video_id}: "
                f"crop_start={crop_start}, event_onset={event_onset}, "
                f"duration={audio_info['duration_seconds']}"
            )
    provenance: dict[str, Any] = {
        "source_route": source_route,
        "hf_dataset": hf_dataset,
        "hf_revision": hf_revision,
        "hf_split": hf_split,
        "video_id": video_id,
        "transport": transport,
        "plan_sha256": plan_hash,
        "plan_row_sha256": sha256_bytes(_canonical_json_bytes(plan)),
    }
    if parquet:
        provenance["parquet"] = dict(parquet)
    if existing_manifest:
        provenance["existing_manifest"] = dict(existing_manifest)
    return {
        "format": MANIFEST_FORMAT,
        "scene_id": f"audioset_strong_{hf_split}_{video_id}",
        "source_route": source_route,
        "hf_dataset": provenance["hf_dataset"],
        "hf_revision": hf_revision,
        "hf_split": hf_split,
        "video_id": video_id,
        "mixture_path": _portable_path(audio_path),
        "audio_sha256": str(audio_info["audio_sha256"]),
        "audio_bytes": int(audio_info["audio_bytes"]),
        "duration_seconds": float(audio_info["duration_seconds"]),
        "sample_rate": int(audio_info["sample_rate"]),
        "labels": sorted(
            {
                str(event["label"])
                for event in events
                if bool(event.get("selected_ontology_label", True))
            }
        ),
        "all_strong_labels": sorted({str(event["label"]) for event in events}),
        "covers_deficit_labels": sorted(
            {str(value) for value in plan.get("covers_deficit_labels", [])}
        ),
        "segment_ids": sorted({str(value) for value in plan.get("segment_ids", [])}),
        "split_lock": str(plan.get("split_lock") or ""),
        "coverage_request_count": int(plan.get("coverage_request_count") or 0),
        "crop_requests": list(plan.get("crop_requests") or []),
        "full_official_strong_events": bool(
            plan.get("full_official_strong_events", False)
        ),
        "events": events,
        "plan_sha256": plan_hash,
        "source_provenance": provenance,
    }


def _retry(
    operation: Callable[[], Any],
    *,
    policy: RetryPolicy,
    description: str,
) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except RETRYABLE_REMOTE_IO_ERRORS as error:
            last_error = error
            if attempt >= policy.max_attempts:
                break
            delay = min(
                policy.maximum_delay_seconds,
                policy.initial_delay_seconds * (2 ** (attempt - 1)),
            )
            print(
                f"retry {attempt}/{policy.max_attempts - 1} for {description} "
                f"after {type(error).__name__}: {error}; sleeping {delay:.2f}s",
                flush=True,
            )
            time.sleep(delay)
    raise MaterializationError(
        f"failed {description} after {policy.max_attempts} attempts: {last_error}"
    ) from last_error


def _url_to_fs(url: str) -> tuple[Any, str]:
    return fsspec.core.url_to_fs(url)


def _open_for_parquet(fs: Any, path: str, block_size: int) -> Any:
    protocol = fs.protocol
    if isinstance(protocol, (tuple, list)):
        protocols = set(protocol)
    else:
        protocols = {protocol}
    if protocols & {"file", "local", None}:
        return fs.open(path, "rb")
    return fs.open(
        path,
        "rb",
        block_size=block_size,
        cache_type="readahead",
    )


def glob_parquet_urls(pattern: str, *, split: str, revision: str) -> list[str]:
    rendered = pattern.format(split=split, revision=revision)
    fs, path_pattern = _url_to_fs(rendered)
    paths = sorted(fs.glob(path_pattern)) if any(c in path_pattern for c in "*?[") else [path_pattern]
    urls = [str(fs.unstrip_protocol(path)) for path in paths]
    if not urls:
        raise MaterializationError(
            f"no Parquet shards match split={split}: {rendered}"
        )
    return urls


def _jsonable_file_info(info: Mapping[str, Any]) -> dict[str, Any]:
    lfs = info.get("lfs")
    output: dict[str, Any] = {
        "size_bytes": int(info.get("size") or 0),
        "blob_id": str(info.get("blob_id") or ""),
        "xet_hash": str(info.get("xet_hash") or ""),
    }
    if lfs is not None:
        output["lfs_sha256"] = str(getattr(lfs, "sha256", "") or "")
        output["lfs_size_bytes"] = int(getattr(lfs, "size", 0) or 0)
    return output


def parquet_metadata(
    url: str,
    *,
    block_size: int,
    retry: RetryPolicy,
) -> dict[str, Any]:
    def operation() -> dict[str, Any]:
        fs, path = _url_to_fs(url)
        info = _jsonable_file_info(fs.info(path))
        with _open_for_parquet(fs, path, block_size) as handle:
            parquet = pq.ParquetFile(handle)
            if "video_id" not in parquet.schema_arrow.names or "audio" not in parquet.schema_arrow.names:
                raise MaterializationError(
                    f"required columns video_id/audio missing from {url}: "
                    f"{parquet.schema_arrow.names}"
                )
            groups: list[dict[str, Any]] = []
            for index in range(parquet.num_row_groups):
                group = parquet.metadata.row_group(index)
                video_bytes = 0
                audio_bytes = 0
                for column_index in range(group.num_columns):
                    column = group.column(column_index)
                    compressed = int(column.total_compressed_size or 0)
                    path_in_schema = str(column.path_in_schema)
                    if path_in_schema == "video_id":
                        video_bytes += compressed
                    if path_in_schema == "audio" or path_in_schema.startswith("audio."):
                        audio_bytes += compressed
                groups.append(
                    {
                        "index": index,
                        "num_rows": int(group.num_rows),
                        "video_id_compressed_bytes": video_bytes,
                        "audio_compressed_bytes": audio_bytes,
                    }
                )
            return {
                "url": url,
                **info,
                "num_rows": int(parquet.metadata.num_rows),
                "num_row_groups": int(parquet.num_row_groups),
                "row_groups": groups,
            }

    return _retry(operation, policy=retry, description=f"read Parquet metadata {url}")


def read_row_group(
    url: str,
    row_group: int,
    *,
    columns: Sequence[str],
    block_size: int,
    retry: RetryPolicy,
) -> pa.Table:
    def operation() -> pa.Table:
        fs, path = _url_to_fs(url)
        with _open_for_parquet(fs, path, block_size) as handle:
            parquet = pq.ParquetFile(handle)
            if row_group < 0 or row_group >= parquet.num_row_groups:
                raise MaterializationError(
                    f"row group {row_group} out of range for {url}"
                )
            return parquet.read_row_group(row_group, columns=list(columns))

    return _retry(
        operation,
        policy=retry,
        description=f"read row group {row_group} columns={list(columns)} from {url}",
    )


def process_row_groups(
    url: str,
    row_groups: Sequence[int],
    *,
    columns: Sequence[str],
    block_size: int,
    retry: RetryPolicy,
    callback: Callable[[int, pa.Table], bool | None],
) -> None:
    """Read several row groups while reusing one seekable Parquet handle.

    ``callback`` runs only after a complete table has been read.  A truthy
    ``False`` return stops the shard early.  If an HTTP/Arrow I/O exception is
    raised, already-callbacked groups are not replayed and the current group is
    retried through a newly opened handle.  Exceptions from the callback are
    deliberately not caught, which makes committed-state interruption tests
    representative of a killed process.
    """

    pending = [int(value) for value in row_groups]
    cursor = 0
    failures = 0
    while cursor < len(pending):
        remaining = pending[cursor:]
        try:
            fs, path = _url_to_fs(url)
            with _open_for_parquet(fs, path, block_size) as handle:
                parquet = pq.ParquetFile(handle)
                row_counts: list[int] = []
                for row_group in remaining:
                    if row_group < 0 or row_group >= parquet.num_row_groups:
                        raise MaterializationError(
                            f"row group {row_group} out of range for {url}"
                        )
                    row_counts.append(int(parquet.metadata.row_group(row_group).num_rows))
                # One projection per shard lets PyArrow reuse footer/range
                # state.  For phase one this is only the tiny video_id column;
                # callbacks below still checkpoint every constituent group.
                combined = parquet.read_row_groups(
                    remaining, columns=list(columns), use_threads=True
                )
        except RETRYABLE_REMOTE_IO_ERRORS as error:
            failures += 1
            if failures >= retry.max_attempts:
                raise MaterializationError(
                    f"failed batched row-group read for {url} at index "
                    f"{pending[cursor]} after {failures} attempts: {error}"
                ) from error
            delay = min(
                retry.maximum_delay_seconds,
                retry.initial_delay_seconds * (2 ** (failures - 1)),
            )
            print(
                f"retry {failures}/{retry.max_attempts - 1} for batched row-group "
                f"read {url} at rg={pending[cursor]} after {type(error).__name__}: "
                f"{error}; sleeping {delay:.2f}s",
                flush=True,
            )
            time.sleep(delay)
            continue

        # State/manifest callback failures are outside the HTTP retry block.
        # This prevents a local disk error from being mislabeled as transient
        # network trouble, and lets interruption tests stop after a commit.
        failures = 0
        offset = 0
        for row_group, row_count in zip(remaining, row_counts):
            table = combined.slice(offset, row_count)
            offset += row_count
            cursor += 1
            should_continue = callback(row_group, table)
            if should_continue is False:
                return


def _new_state(
    *,
    plan_hash: str,
    file_receipts: Sequence[Mapping[str, Any]],
    pattern: str,
    hf_dataset: str,
    revision: str,
    storage_mode: str,
    location_mode: str,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "schema_version": 1,
        "plan_sha256": plan_hash,
        "plan_files": [dict(value) for value in file_receipts],
        "parquet_pattern": pattern,
        "hf_dataset": hf_dataset,
        "resolved_revision": revision,
        "storage_mode": storage_mode,
        "location_mode": location_mode,
        "shards": {},
        "scanned_row_groups": [],
        "locations": {},
        "scan_complete_splits": [],
        "manifest_fragments": [],
        "completed_video_ids": [],
        "updated_unix_seconds": time.time(),
    }


def _load_or_create_state(
    path: Path,
    *,
    plan_hash: str,
    file_receipts: Sequence[Mapping[str, Any]],
    pattern: str,
    hf_dataset: str,
    revision: str,
    storage_mode: str,
    location_mode: str,
) -> dict[str, Any]:
    if not path.exists():
        return _new_state(
            plan_hash=plan_hash,
            file_receipts=file_receipts,
            pattern=pattern,
            hf_dataset=hf_dataset,
            revision=revision,
            storage_mode=storage_mode,
            location_mode=location_mode,
        )
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MaterializationError(f"cannot read resumable state {path}: {error}") from error
    if state.get("format") != FORMAT:
        raise MaterializationError(f"unexpected state format in {path}")
    if state.get("plan_sha256") != plan_hash:
        raise MaterializationError(
            "plan changed for an existing output directory; use a new output directory"
        )
    if state.get("parquet_pattern") != pattern or state.get("resolved_revision") != revision:
        raise MaterializationError(
            "Parquet pattern/revision changed for an existing output directory"
        )
    if state.get("hf_dataset", "enyoukai/AudioSet-Strong") != hf_dataset:
        raise MaterializationError(
            "HF dataset changed for an existing output directory"
        )
    if state.get("storage_mode", "full_clips") != storage_mode:
        raise MaterializationError(
            "storage mode changed for an existing output directory"
        )
    if state.get("location_mode", "scan_if_missing") != location_mode:
        raise MaterializationError(
            "location mode changed for an existing output directory"
        )
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_unix_seconds"] = time.time()
    atomic_json(path, state)


def _fragment_key(url: str, row_group: int) -> str:
    return f"{sha256_bytes(url.encode('utf-8'))[:16]}-rg{row_group:05d}"


def _load_fragments(
    output_dir: Path,
    *,
    plan_hash: str,
    plans: Mapping[str, Mapping[str, Any]],
    validate_artifacts: bool = True,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    rows: dict[str, dict[str, Any]] = {}
    fragments: list[dict[str, Any]] = []
    fragment_dir = output_dir / "manifest_fragments"
    for path in sorted(fragment_dir.glob("*.jsonl")) if fragment_dir.exists() else []:
        fragment_rows = _load_jsonl(path)
        for row in fragment_rows:
            video_id = str(row.get("video_id") or "")
            if video_id not in plans:
                raise MaterializationError(
                    f"fragment {path} contains video outside current plan: {video_id}"
                )
            if row.get("plan_sha256") != plan_hash:
                raise MaterializationError(f"fragment plan hash mismatch: {path}")
            if validate_artifacts:
                _validate_committed_artifacts(
                    row=row,
                    plan=plans[video_id],
                    fragment_path=path,
                )
            previous = rows.get(video_id)
            if previous is not None and _canonical_json_bytes(previous) != _canonical_json_bytes(row):
                raise MaterializationError(
                    f"conflicting fragment rows for video_id={video_id}"
                )
            rows[video_id] = row
        fragments.append(
            {
                "path": _portable_path(path),
                "sha256": sha256_file(path),
                "rows": len(fragment_rows),
            }
        )
    return rows, fragments


def _validate_committed_artifacts(
    *,
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    fragment_path: Path,
) -> None:
    """Hash- and decoder-validate the durable audio referenced by one row.

    Fragment JSON is the transaction journal, but a journal entry is not a
    completed transaction if its audio was later deleted or corrupted.  This
    check runs when a process starts/resumes.  Internal reloads after freshly
    validated atomic writes opt out to avoid quadratic re-hashing.
    """

    video_id = str(row.get("video_id") or "")
    crop_records = row.get("crop_records")
    if isinstance(crop_records, list):
        declared_count = int(row.get("crop_item_count") or 0)
        if declared_count != len(crop_records):
            raise MaterializationError(
                f"crop transaction count mismatch for {video_id}: "
                f"{declared_count} != {len(crop_records)}"
            )
        expected_items = {
            str(request.get("selection_key") or "")
            for request in plan.get("crop_requests") or []
        }
        actual_items: list[str] = []
        for crop in crop_records:
            if not isinstance(crop, dict):
                raise MaterializationError(
                    f"crop transaction contains a non-object for {video_id}"
                )
            item_id = str(crop.get("materialization_item_id") or "")
            actual_items.append(item_id)
            if str(crop.get("source_video_id") or "") != video_id:
                raise MaterializationError(
                    f"crop source_video_id mismatch for {video_id}/{item_id}"
                )
            audio_value = str(crop.get("mixture_path") or "")
            declared_hash = str(crop.get("audio_sha256") or "")
            if not item_id or not audio_value or not declared_hash:
                raise MaterializationError(
                    f"incomplete committed crop record for {video_id}/{item_id}"
                )
            audio_path = resolve_manifest_audio_path(fragment_path, audio_value)
            validate_audio_path(audio_path, declared_hash)
        if len(actual_items) != len(set(actual_items)):
            raise MaterializationError(
                f"duplicate committed crop item for {video_id}"
            )
        if set(actual_items) != expected_items:
            raise MaterializationError(
                f"committed crop items differ from plan for {video_id}: "
                f"missing={sorted(expected_items - set(actual_items))[:5]}, "
                f"extra={sorted(set(actual_items) - expected_items)[:5]}"
            )
        return

    audio_value = str(row.get("mixture_path") or "")
    declared_hash = str(row.get("audio_sha256") or "")
    if not audio_value or not declared_hash:
        raise MaterializationError(
            f"incomplete committed full-audio row for {video_id}"
        )
    audio_path = resolve_manifest_audio_path(fragment_path, audio_value)
    validate_audio_path(audio_path, declared_hash)


def _write_manifest_index(
    output_dir: Path,
    *,
    plan_hash: str,
    plan_rows: int,
    completed_rows: Mapping[str, Mapping[str, Any]],
    fragments: Sequence[Mapping[str, Any]],
    artifacts_published: bool = False,
) -> None:
    transactions_complete = len(completed_rows) == plan_rows
    atomic_json(
        output_dir / "manifest_index.json",
        {
            "format": FORMAT,
            "plan_sha256": plan_hash,
            "plan_rows": plan_rows,
            "completed_rows": len(completed_rows),
            "remaining_rows": plan_rows - len(completed_rows),
            "transactions_complete": transactions_complete,
            # The final plan/crop manifests are published before this bit is
            # flipped.  A kill after the last fragment therefore remains an
            # explicitly resumable, incomplete publication.
            "artifacts_published": bool(
                artifacts_published and transactions_complete
            ),
            "complete": bool(artifacts_published and transactions_complete),
            "fragments": list(fragments),
        },
    )


def _write_fragment(
    output_dir: Path,
    *,
    key: str,
    new_rows: Sequence[Mapping[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    path = output_dir / "manifest_fragments" / f"{key}.jsonl"
    merged: dict[str, dict[str, Any]] = {}
    if path.exists():
        for row in _load_jsonl(path):
            merged[str(row["video_id"])] = row
    for row in new_rows:
        video_id = str(row["video_id"])
        previous = merged.get(video_id)
        if previous is not None and _canonical_json_bytes(previous) != _canonical_json_bytes(row):
            raise MaterializationError(
                f"attempt to replace committed fragment row for video_id={video_id}"
            )
        merged[video_id] = dict(row)
    atomic_jsonl(path, [merged[key] for key in sorted(merged)])
    return path, {
        "path": _portable_path(path),
        "sha256": sha256_file(path),
        "rows": len(merged),
    }


def _upsert_fragment_receipt(
    receipts: Sequence[Mapping[str, Any]],
    new_receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Update one fragment receipt without re-reading all fragment JSON."""

    by_path = {
        str(receipt["path"]): dict(receipt)
        for receipt in receipts
    }
    by_path[str(new_receipt["path"])] = dict(new_receipt)
    return [by_path[path] for path in sorted(by_path)]


def load_existing_manifest_reuse(
    paths: Sequence[Path],
    *,
    plans: Mapping[str, Mapping[str, Any]],
    plan_hash: str,
    resolved_revision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise MaterializationError(f"existing manifest does not exist: {path}")
        manifest_hash = sha256_file(path)
        matched = 0
        for row in _load_jsonl(path):
            video_id = str(row.get("video_id") or "")
            if video_id not in plans:
                continue
            audio_value = str(row.get("mixture_path") or "")
            if not audio_value:
                raise MaterializationError(
                    f"existing manifest {path} has no mixture_path for {video_id}"
                )
            audio_path = resolve_manifest_audio_path(path, audio_value)
            audio_info = validate_audio_path(
                audio_path,
                str(row.get("audio_sha256") or "") or None,
            )
            replacement = build_manifest_row(
                plan=plans[video_id],
                plan_hash=plan_hash,
                audio_path=audio_path,
                audio_info=audio_info,
                resolved_revision=resolved_revision,
                transport="existing_manifest_reuse",
                existing_manifest={
                    "path": str(path),
                    "sha256": manifest_hash,
                    "source_route": str(row.get("source_route") or ""),
                    "original_audio_sha256": str(row.get("audio_sha256") or ""),
                },
            )
            previous = output.get(video_id)
            if previous is not None and previous["audio_sha256"] != replacement["audio_sha256"]:
                raise MaterializationError(
                    f"existing manifests disagree on audio for video_id={video_id}"
                )
            output[video_id] = replacement
            matched += 1
        receipts.append(
            {
                "path": str(path),
                "sha256": manifest_hash,
                "matched_plan_rows": matched,
            }
        )
    return [output[key] for key in sorted(output)], receipts


def load_existing_audio_sources(
    paths: Sequence[Path],
    *,
    plans: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Validate existing full clips for crop-mode source reuse.

    These rows are source availability, not completed materialization rows: the
    requested crop files still have to be emitted transactionally.
    """

    sources: dict[str, dict[str, Any]] = {}
    receipts: list[dict[str, Any]] = []
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise MaterializationError(f"existing manifest does not exist: {path}")
        manifest_hash = sha256_file(path)
        matched = 0
        for row in _load_jsonl(path):
            video_id = str(row.get("video_id") or "")
            if video_id not in plans:
                continue
            declared_split = str(
                row.get("protocol_upstream_split")
                or row.get("hf_split")
                or row.get("metadata_split")
                or ""
            ).lower()
            expected_hf_split = str(plans[video_id]["hf_split"])
            if declared_split:
                normalized = "test" if declared_split in {"test", "eval"} else "train"
                if normalized != expected_hf_split:
                    raise MaterializationError(
                        f"existing source split mismatch for {video_id}: "
                        f"{declared_split} vs {expected_hf_split}"
                    )
            audio_value = str(row.get("mixture_path") or row.get("audio_path") or "")
            if not audio_value:
                raise MaterializationError(
                    f"existing manifest {path} has no audio path for {video_id}"
                )
            audio_path = resolve_manifest_audio_path(path, audio_value)
            audio_info = validate_audio_path(
                audio_path, str(row.get("audio_sha256") or "") or None
            )
            candidate = {
                "video_id": video_id,
                "audio_path": audio_path,
                "audio_info": audio_info,
                "manifest_path": str(path),
                "manifest_sha256": manifest_hash,
                "original_source_route": str(row.get("source_route") or ""),
            }
            previous = sources.get(video_id)
            if previous is not None:
                if (
                    previous["audio_info"]["audio_sha256"]
                    != audio_info["audio_sha256"]
                ):
                    raise MaterializationError(
                        f"existing manifests disagree on source audio for {video_id}"
                    )
                if str(audio_path) >= str(previous["audio_path"]):
                    matched += 1
                    continue
            sources[video_id] = candidate
            matched += 1
        receipts.append(
            {
                "path": str(path),
                "sha256": manifest_hash,
                "matched_plan_rows": matched,
            }
        )
    return sources, receipts


def _scan_key(url: str, row_group: int) -> str:
    return f"{url}#rg={row_group}"


def _scan_one_shard_video_ids(
    *,
    url: str,
    known_metadata: Mapping[str, Any] | None,
    scanned_keys: set[str],
    config: MaterializationConfig,
) -> tuple[dict[str, Any], list[tuple[int, list[str]]]]:
    """Network worker: return video IDs without mutating durable state."""

    metadata = (
        dict(known_metadata)
        if known_metadata is not None
        else parquet_metadata(
            url,
            block_size=config.block_size_bytes,
            retry=config.retry,
        )
    )
    pending_groups = [
        int(group["index"])
        for group in metadata["row_groups"]
        if _scan_key(url, int(group["index"])) not in scanned_keys
    ]
    results: list[tuple[int, list[str]]] = []

    def collect(row_group: int, table: pa.Table) -> bool:
        results.append(
            (row_group, [str(value) for value in table.column("video_id").to_pylist()])
        )
        return True

    if pending_groups:
        process_row_groups(
            url,
            pending_groups,
            columns=["video_id"],
            block_size=config.block_size_bytes,
            retry=config.retry,
            callback=collect,
        )
    return metadata, results


def _validate_shard_provenance(
    *,
    video_id: str,
    declared: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    comparisons = (
        ("file_size_bytes", "size_bytes", int),
        ("blob_id", "blob_id", str),
        ("xet_hash", "xet_hash", str),
        ("lfs_sha256", "lfs_sha256", str),
        ("lfs_size_bytes", "lfs_size_bytes", int),
    )
    for declared_key, actual_key, cast in comparisons:
        raw_expected = declared.get(declared_key)
        if raw_expected in (None, "", 0, "0"):
            continue
        expected = cast(raw_expected)
        observed = cast(actual.get(actual_key) or cast())
        if observed != expected:
            raise MaterializationError(
                f"preindexed shard provenance changed for {video_id}: "
                f"{declared_key}={observed!r} != {expected!r}"
            )


def seed_preindexed_locations(
    *,
    config: MaterializationConfig,
    plans: Mapping[str, Mapping[str, Any]],
    state: dict[str, Any],
    state_path: Path,
    checkpoint_hook: CheckpointHook | None,
) -> None:
    """Validate and install exact physical locations embedded in the plan.

    This mode is intentionally fail-closed.  A missing or inconsistent
    location never falls back to globbing another AudioSet mirror.  Footer
    metadata is read for the exact shards so row bounds, immutable file
    provenance and I/O estimates remain auditable before any audio column is
    touched.
    """

    normalized: dict[str, dict[str, Any]] = {}
    physical_rows: dict[tuple[str, int, int], str] = {}
    provenance_by_url: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    urls_by_split: dict[str, set[str]] = {"train": set(), "test": set()}
    route_counts: dict[str, int] = {}
    for video_id, plan in plans.items():
        source_route = str(plan.get("source_route") or "")
        dataset = str(plan.get("hf_dataset") or "")
        revision = str(plan.get("hf_revision") or "")
        split = str(plan.get("hf_split") or "")
        location = plan.get("availability_location")
        if (
            not source_route
            or not dataset
            or not revision
            or split not in {"train", "test"}
            or not isinstance(location, Mapping)
        ):
            raise MaterializationError(
                f"preindexed plan contract is incomplete for {video_id}"
            )
        if dataset != config.hf_dataset:
            raise MaterializationError(
                f"preindexed plan dataset mismatch for {video_id}: "
                f"{dataset} != {config.hf_dataset}"
            )
        if revision != config.resolved_revision:
            raise MaterializationError(
                f"preindexed plan revision mismatch for {video_id}: "
                f"{revision} != {config.resolved_revision}"
            )
        for key, expected in (
            ("source_route", source_route),
            ("hf_dataset", dataset),
            ("hf_revision", revision),
            ("hf_split", split),
            ("video_id", video_id),
        ):
            observed = str(location.get(key) or "")
            if observed != expected:
                raise MaterializationError(
                    f"preindexed location {key} mismatch for {video_id}: "
                    f"{observed!r} != {expected!r}"
                )
        url = str(location.get("parquet_url") or "")
        row_group = int(location.get("row_group", -1))
        row_index = int(location.get("row_index", -1))
        if not url or row_group < 0 or row_index < 0:
            raise MaterializationError(
                f"preindexed location is incomplete for {video_id}"
            )
        physical_key = (url, row_group, row_index)
        previous_video = physical_rows.get(physical_key)
        if previous_video is not None and previous_video != video_id:
            raise MaterializationError(
                "multiple planned video IDs claim the same preindexed row: "
                f"{previous_video}, {video_id} -> {physical_key}"
            )
        physical_rows[physical_key] = video_id
        normalized[video_id] = {
            "source_route": source_route,
            "hf_dataset": dataset,
            "hf_revision": revision,
            "hf_split": split,
            "parquet_url": url,
            "row_group": row_group,
            "row_index": row_index,
            "shard_provenance": dict(location.get("shard_provenance") or {}),
        }
        provenance_by_url.setdefault(url, []).append(
            (video_id, dict(location.get("shard_provenance") or {}))
        )
        urls_by_split[split].add(url)
        route_counts[source_route] = route_counts.get(source_route, 0) + 1

    stored_locations = {
        str(key): dict(value)
        for key, value in state.get("locations", {}).items()
    }
    for video_id, location in stored_locations.items():
        expected = normalized.get(video_id)
        if expected is None or location != expected:
            raise MaterializationError(
                f"stored preindexed location changed for {video_id}"
            )

    metadata_by_url = state.setdefault("shard_metadata", {})
    missing_urls = [url for url in sorted(provenance_by_url) if url not in metadata_by_url]
    futures: dict[Future[Any], str] = {}
    with ThreadPoolExecutor(max_workers=min(config.scan_workers, 4)) as executor:
        for url in missing_urls:
            futures[
                executor.submit(
                    parquet_metadata,
                    url,
                    block_size=config.block_size_bytes,
                    retry=config.retry,
                )
            ] = url
        for future in as_completed(futures):
            url = futures[future]
            metadata = future.result()
            if str(metadata.get("url") or "") != url:
                raise MaterializationError(
                    f"preindexed shard URL changed while reading metadata: {url}"
                )
            metadata_by_url[url] = metadata
            state["shard_metadata"] = metadata_by_url
            _save_state(state_path, state)
            if checkpoint_hook:
                checkpoint_hook(
                    "preindexed_shard_metadata_committed",
                    {"parquet_url": url},
                )

    for video_id, location in normalized.items():
        url = str(location["parquet_url"])
        metadata = metadata_by_url.get(url)
        if not isinstance(metadata, Mapping):
            raise MaterializationError(
                f"preindexed shard metadata missing for {video_id}: {url}"
            )
        _validate_shard_provenance(
            video_id=video_id,
            declared=location.get("shard_provenance") or {},
            actual=metadata,
        )
        groups = {
            int(value["index"]): value
            for value in metadata.get("row_groups", [])
        }
        group = groups.get(int(location["row_group"]))
        if group is None:
            raise MaterializationError(
                f"preindexed row group is out of range for {video_id}"
            )
        if int(location["row_index"]) >= int(group.get("num_rows") or 0):
            raise MaterializationError(
                f"preindexed row index is out of range for {video_id}"
            )

    state["locations"] = normalized
    state["shards"] = {
        split: sorted(urls) for split, urls in urls_by_split.items() if urls
    }
    state["scan_complete_splits"] = sorted(
        split for split, urls in urls_by_split.items() if urls
    )
    state["preindexed_location_summary"] = {
        "validated_plan_rows": len(normalized),
        "unique_shards": len(provenance_by_url),
        "source_route_counts": dict(sorted(route_counts.items())),
        "fallback_glob_used": False,
    }
    _save_state(state_path, state)


def scan_plan_locations(
    *,
    config: MaterializationConfig,
    plans: Mapping[str, Mapping[str, Any]],
    already_completed: set[str],
    state: dict[str, Any],
    state_path: Path,
    checkpoint_hook: CheckpointHook | None,
) -> None:
    if config.require_preindexed_locations:
        seed_preindexed_locations(
            config=config,
            plans=plans,
            state=state,
            state_path=state_path,
            checkpoint_hook=checkpoint_hook,
        )
        return
    scanned = set(str(value) for value in state.get("scanned_row_groups", []))
    locations: dict[str, dict[str, Any]] = {
        str(key): dict(value) for key, value in state.get("locations", {}).items()
    }
    complete_splits = set(str(value) for value in state.get("scan_complete_splits", []))
    targets_by_split: dict[str, set[str]] = {"train": set(), "test": set()}
    for video_id, plan in plans.items():
        if video_id not in already_completed:
            targets_by_split[str(plan["hf_split"])].add(video_id)

    for split in ("train", "test"):
        unresolved = targets_by_split[split] - set(locations)
        if not unresolved:
            complete_splits.add(split)
            continue
        urls = glob_parquet_urls(
            config.parquet_pattern,
            split=split,
            revision=config.resolved_revision,
        )
        state.setdefault("shards", {})[split] = urls
        metadata_by_url = state.setdefault("shard_metadata", {})
        futures: dict[Future[Any], str] = {}
        executor = ThreadPoolExecutor(max_workers=config.scan_workers)
        try:
            for url in urls:
                # Entirely scanned shards need no worker.  Metadata survives
                # from the committed state and remains available for receipt
                # estimates.
                metadata = metadata_by_url.get(url)
                if metadata is not None and all(
                    _scan_key(url, int(group["index"])) in scanned
                    for group in metadata["row_groups"]
                ):
                    continue
                future = executor.submit(
                    _scan_one_shard_video_ids,
                    url=url,
                    known_metadata=metadata,
                    scanned_keys=set(scanned),
                    config=config,
                )
                futures[future] = url

            for future in as_completed(futures):
                url = futures[future]
                metadata, group_results = future.result()
                metadata_by_url[url] = metadata
                groups_by_index = {
                    int(group["index"]): group for group in metadata["row_groups"]
                }
                _save_state(state_path, state)
                for row_group, values in group_results:
                    group = groups_by_index[row_group]
                    key = _scan_key(url, row_group)
                    if key in scanned:
                        continue
                    matching_ids: list[str] = []
                    for row_index, raw_video_id in enumerate(values):
                        video_id = str(raw_video_id)
                        if video_id not in targets_by_split[split]:
                            continue
                        previous = locations.get(video_id)
                        location = {
                            "hf_split": split,
                            "parquet_url": url,
                            "row_group": row_group,
                            "row_index": row_index,
                        }
                        if previous is not None and previous != location:
                            raise MaterializationError(
                                f"video_id occurs at multiple Parquet locations: {video_id}"
                            )
                        locations[video_id] = location
                        matching_ids.append(video_id)
                    group["matching_plan_video_ids"] = sorted(matching_ids)
                    group["matching_plan_rows"] = len(matching_ids)
                    scanned.add(key)
                    state["scanned_row_groups"] = sorted(scanned)
                    state["locations"] = locations
                    _save_state(state_path, state)
                    if checkpoint_hook:
                        checkpoint_hook(
                            "scan_row_group_committed",
                            {
                                "hf_split": split,
                                "parquet_url": url,
                                "row_group": row_group,
                                "matches": len(matching_ids),
                            },
                        )
                    unresolved = targets_by_split[split] - set(locations)
                    if not unresolved:
                        break
                if not unresolved:
                    break
            if not unresolved:
                for pending_future in futures:
                    if not pending_future.done():
                        pending_future.cancel()
        except BaseException:
            for pending_future in futures:
                pending_future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        if unresolved:
            state["locations"] = locations
            _save_state(state_path, state)
            raise MaterializationError(
                f"{len(unresolved)} planned {split} video_ids were not found in Parquet shards; "
                f"examples={sorted(unresolved)[:10]}"
            )
        complete_splits.add(split)
        state["scan_complete_splits"] = sorted(complete_splits)
        state["locations"] = locations
        _save_state(state_path, state)


def _audio_suffix(raw_path: str) -> str:
    suffix = Path(raw_path).suffix.lower()
    if not suffix or not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        return ".flac"
    return suffix


def _extract_audio_payload(value: Any) -> tuple[bytes, str]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if not isinstance(value, dict):
        raise MaterializationError(f"unexpected AudioSet audio cell: {type(value)}")
    data = value.get("bytes")
    if isinstance(data, memoryview):
        data = data.tobytes()
    if isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes) or not data:
        raise MaterializationError("AudioSet row has no encoded audio bytes")
    return data, str(value.get("path") or "")


def initialize_crop_storage_state(
    *,
    state: dict[str, Any],
    plans: Mapping[str, Mapping[str, Any]],
    existing_sources: Mapping[str, Mapping[str, Any]],
) -> None:
    """Persist a deterministic crop disk estimate before remote audio reads."""

    current = state.get("crop_storage")
    if current is not None:
        return
    bytes_per_second: dict[str, list[float]] = {"train": [], "test": []}
    for video_id, source in existing_sources.items():
        info = source["audio_info"]
        duration = float(info["duration_seconds"])
        if duration > 0:
            split = str(plans[video_id]["hf_split"])
            bytes_per_second[split].append(float(info["audio_bytes"]) / duration)
    global_rates = [value for values in bytes_per_second.values() for value in values]
    fallback_rate = statistics.fmean(global_rates) if global_rates else 130_000.0
    rates = {
        split: (statistics.fmean(values) if values else fallback_rate)
        for split, values in bytes_per_second.items()
    }
    per_video: dict[str, dict[str, Any]] = {}
    seen_items: set[str] = set()
    split_summary: dict[str, dict[str, float | int]] = {
        "train": {"source_videos": 0, "crop_items": 0, "requested_seconds": 0.0},
        "test": {"source_videos": 0, "crop_items": 0, "requested_seconds": 0.0},
    }
    for video_id, plan in plans.items():
        requests = plan.get("crop_requests") or []
        if not requests:
            raise MaterializationError(
                f"requested_crops mode requires crop_requests for {video_id}"
            )
        seconds = 0.0
        for request in requests:
            item_id = str(request.get("selection_key") or "")
            if not item_id:
                raise MaterializationError(f"crop request has no selection_key: {video_id}")
            if item_id in seen_items:
                raise MaterializationError(f"duplicate crop selection_key: {item_id}")
            seen_items.add(item_id)
            start = float(request.get("crop_start_seconds", -1.0))
            end = float(request.get("crop_end_seconds", -1.0))
            if start < 0 or end <= start:
                raise MaterializationError(
                    f"invalid crop window for {video_id}/{item_id}: [{start}, {end}]"
                )
            seconds += end - start
        split = str(plan["hf_split"])
        # Account for one small FLAC container/header per requested crop.
        estimated = int(round(seconds * rates[split] + len(requests) * 4096))
        per_video[video_id] = {
            "hf_split": split,
            "crop_items": len(requests),
            "requested_seconds": seconds,
            "estimated_bytes": estimated,
        }
        split_summary[split]["source_videos"] += 1
        split_summary[split]["crop_items"] += len(requests)
        split_summary[split]["requested_seconds"] += seconds
    state["crop_storage"] = {
        "format": FORMAT,
        "estimation_method": (
            "requested crop seconds multiplied by split-specific mean encoded "
            "FLAC bytes/second from validated existing sources, plus 4096 bytes/item"
        ),
        "bytes_per_second": rates,
        "total_crop_items": len(seen_items),
        "total_requested_seconds": sum(
            float(value["requested_seconds"]) for value in per_video.values()
        ),
        "estimated_total_bytes": sum(
            int(value["estimated_bytes"]) for value in per_video.values()
        ),
        "split_summary": split_summary,
        "per_video": per_video,
    }


def _crop_events(
    plan: Mapping[str, Any],
    *,
    crop_start: float,
    crop_end: float,
    item_id: str,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(plan["events"]):
        source_onset = float(raw["onset_seconds"])
        source_offset = float(raw["offset_seconds"])
        if source_offset <= crop_start or source_onset >= crop_end:
            continue
        onset = max(source_onset, crop_start) - crop_start
        offset = min(source_offset, crop_end) - crop_start
        if offset <= onset:
            continue
        label = str(raw["label"])
        events.append(
            {
                "event_id": f"{item_id}_{index:04d}_{label}",
                "event_kind": "semantic",
                "label": label,
                "display_name": str(raw.get("display_name") or label.replace("_", " ")),
                "audioset_mid": str(raw.get("mid") or raw.get("audioset_mid") or ""),
                "onset_seconds": onset,
                "offset_seconds": offset,
                "source_onset_seconds": source_onset,
                "source_offset_seconds": source_offset,
                "segment_id": str(raw.get("segment_id") or ""),
                "video_id": str(plan["video_id"]),
                "selected_ontology_label": bool(
                    raw.get("selected_ontology_label", True)
                ),
            }
        )
    return events


def _public_crop_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Keep request/provenance scalars without duplicating annotation payloads.

    Complete intersecting annotations are exposed through ``events``,
    ``context_events`` and ``all_strong_events``.  Re-copying the raw nested
    annotation arrays here would make non-ontology events reachable through an
    ambiguous detector input field and would bloat the flat manifest.
    """

    excluded = {
        "strong_annotations",
        "selected_ontology_annotations",
        "materialized_audio_path",
    }
    public = {
        str(key): value
        for key, value in request.items()
        if str(key) not in excluded
    }
    public["raw_strong_annotation_count"] = len(
        request.get("strong_annotations") or []
    )
    public["raw_selected_ontology_annotation_count"] = len(
        request.get("selected_ontology_annotations") or []
    )
    return public


def materialize_crop_transaction(
    *,
    config: MaterializationConfig,
    plan: Mapping[str, Any],
    plan_hash: str,
    encoded_audio: bytes | None,
    audio_path: Path | None,
    source_audio_info: Mapping[str, Any],
    source_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Decode one source once and atomically emit every requested lossless crop."""

    if (encoded_audio is None) == (audio_path is None):
        raise ValueError("provide exactly one of encoded_audio or audio_path")
    source: Any = io.BytesIO(encoded_audio) if encoded_audio is not None else audio_path
    video_id = str(plan["video_id"])
    source_route = _plan_source_route(plan)
    hf_dataset = _plan_hf_dataset(plan)
    hf_revision = _plan_revision(plan, config.resolved_revision)
    crop_records: list[dict[str, Any]] = []
    with sf.SoundFile(source, mode="r") as audio:
        sample_rate = int(audio.samplerate)
        total_frames = int(audio.frames)
        source_duration = total_frames / max(sample_rate, 1)
        subtype = str(audio.subtype or "PCM_16")
        if subtype not in {"PCM_16", "PCM_24"}:
            subtype = "PCM_16"
        for request in plan.get("crop_requests") or []:
            item_id = str(request["selection_key"])
            requested_start = float(request["crop_start_seconds"])
            requested_end = float(request["crop_end_seconds"])
            event_onset = float(request["event_onset_seconds"])
            tail_shortfall = max(0.0, requested_end - source_duration)
            if (
                requested_start >= source_duration
                or (
                    event_onset >= source_duration
                    and tail_shortfall > MAX_NOMINAL_TAIL_PADDING_SECONDS
                )
            ):
                raise MaterializationError(
                    f"coverage crop outside decoded source for {video_id}/{item_id}: "
                    f"window=[{requested_start}, {requested_end}], event={event_onset}, "
                    f"duration={source_duration}"
                )
            start_frame = max(0, min(total_frames - 1, int(math.floor(requested_start * sample_rate))))
            decoded_end_frame = max(
                start_frame + 1,
                min(total_frames, int(math.ceil(requested_end * sample_rate))),
            )
            output_end_frame = max(
                decoded_end_frame,
                int(math.ceil(requested_end * sample_rate)),
            )
            actual_start = start_frame / sample_rate
            decoded_actual_end = decoded_end_frame / sample_rate
            actual_end = output_end_frame / sample_rate
            audio.seek(start_frame)
            samples = audio.read(
                decoded_end_frame - start_frame, dtype="int32", always_2d=True
            )
            if samples.shape[0] != decoded_end_frame - start_frame:
                raise MaterializationError(
                    f"short decoded crop for {video_id}/{item_id}: "
                    f"{samples.shape[0]} != {decoded_end_frame - start_frame}"
                )
            tail_padding_frames = output_end_frame - decoded_end_frame
            if tail_padding_frames:
                samples = np.pad(
                    samples,
                    ((0, tail_padding_frames), (0, 0)),
                    mode="constant",
                )
            buffer = io.BytesIO()
            sf.write(
                buffer,
                samples,
                sample_rate,
                format="FLAC",
                subtype=subtype,
            )
            crop_bytes = buffer.getvalue()
            crop_info = validate_audio_bytes(crop_bytes)
            filename = sha256_bytes(f"{video_id}:{item_id}".encode("utf-8"))
            output_path = (
                config.output_dir.resolve()
                / "audio"
                / "crops"
                / str(plan["hf_split"])
                / f"{filename}.flac"
            )
            if output_path.exists():
                disk_info = validate_audio_path(output_path)
                if disk_info["audio_sha256"] != crop_info["audio_sha256"]:
                    raise MaterializationError(
                        f"existing crop differs for {video_id}/{item_id}: {output_path}"
                    )
            else:
                atomic_bytes(output_path, crop_bytes)
                disk_info = validate_audio_path(
                    output_path, str(crop_info["audio_sha256"])
                )
            all_strong_events = _crop_events(
                plan,
                crop_start=actual_start,
                crop_end=actual_end,
                item_id=item_id,
            )
            coverage_label = str(request["coverage_label"])
            coverage_mid = str(request["coverage_mid"])
            if not any(
                str(event["label"]) == coverage_label
                and str(event["audioset_mid"]) == coverage_mid
                for event in all_strong_events
            ):
                raise MaterializationError(
                    f"coverage event missing after crop for {video_id}/{item_id}"
                )
            events = [
                event
                for event in all_strong_events
                if bool(event["selected_ontology_label"])
            ]
            context_events = [
                event
                for event in all_strong_events
                if not bool(event["selected_ontology_label"])
            ]
            coverage_onset = float(request["event_onset_seconds"])
            coverage_offset = float(request["event_offset_seconds"])
            coverage_duration = max(coverage_offset - coverage_onset, 1e-12)
            retained_coverage_seconds = max(
                0.0,
                min(coverage_offset, decoded_actual_end)
                - max(coverage_onset, actual_start),
            )
            retained_coverage_fraction = min(
                1.0, retained_coverage_seconds / coverage_duration
            )
            crop_records.append(
                {
                    "format": "qces_audioset_strong_requested_crop_v1",
                    "scene_id": f"audioset_strong_crop_{filename[:24]}",
                    "materialization_item_id": item_id,
                    "source_video_id": video_id,
                    "video_id": video_id,
                    "source_route": source_route,
                    "hf_dataset": hf_dataset,
                    "hf_revision": hf_revision,
                    "hf_split": str(plan["hf_split"]),
                    "metadata_split": str(plan.get("metadata_split") or ""),
                    "split_lock": str(plan.get("split_lock") or ""),
                    "mixture_path": _portable_path(output_path),
                    "audio_sha256": str(disk_info["audio_sha256"]),
                    "audio_bytes": int(disk_info["audio_bytes"]),
                    "sample_rate": int(disk_info["sample_rate"]),
                    "duration_seconds": float(disk_info["duration_seconds"]),
                    "requested_crop_start_seconds": requested_start,
                    "requested_crop_end_seconds": requested_end,
                    "source_crop_start_seconds": actual_start,
                    "source_crop_end_seconds": actual_end,
                    "decoded_source_duration_seconds": source_duration,
                    "decoded_source_crop_end_seconds": decoded_actual_end,
                    "nominal_tail_padding_seconds": (
                        tail_padding_frames / sample_rate
                    ),
                    "nominal_tail_padding_applied": bool(tail_padding_frames),
                    "coverage_label": coverage_label,
                    "coverage_mid": coverage_mid,
                    "coverage_rank": int(request["coverage_rank"]),
                    "ambiguity_tier": int(request["ambiguity_tier"]),
                    "ambiguity_tier_name": str(request["ambiguity_tier_name"]),
                    "clean_source_eligible": bool(
                        request.get("clean_source_eligible", False)
                    ),
                    "fully_isolated": bool(request.get("fully_isolated", False)),
                    "labels": sorted(
                        {str(event["label"]) for event in events}
                    ),
                    "all_strong_labels": sorted(
                        {str(event["label"]) for event in all_strong_events}
                    ),
                    "events": events,
                    "context_events": context_events,
                    "all_strong_events": all_strong_events,
                    "complete_intersecting_strong_annotation_count": len(
                        all_strong_events
                    ),
                    "coverage_event_retained_seconds": retained_coverage_seconds,
                    "coverage_event_retained_fraction": retained_coverage_fraction,
                    "coverage_event_clipped_to_decoded_audio": (
                        retained_coverage_fraction < 1.0 - 1e-9
                    ),
                    "crop_request": _public_crop_request(request),
                    "source_audio_sha256": str(source_audio_info["audio_sha256"]),
                    "source_provenance": dict(source_provenance),
                    "plan_sha256": plan_hash,
                }
            )
    return {
        "format": "qces_audioset_strong_crop_transaction_v1",
        "video_id": video_id,
        "hf_split": str(plan["hf_split"]),
        "plan_sha256": plan_hash,
        "crop_item_count": len(crop_records),
        "crop_audio_bytes": sum(int(row["audio_bytes"]) for row in crop_records),
        "crop_records": crop_records,
        "source_audio_sha256": str(source_audio_info["audio_sha256"]),
        "source_provenance": dict(source_provenance),
    }


def materialize_existing_crop_sources(
    *,
    config: MaterializationConfig,
    ordered_plans: Sequence[Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    plan_hash: str,
    existing_sources: Mapping[str, Mapping[str, Any]],
    state: dict[str, Any],
    state_path: Path,
    completed_rows: dict[str, dict[str, Any]],
    checkpoint_hook: CheckpointHook | None,
    maximum_new_videos: int,
    batch_size: int = 32,
) -> int:
    ordered_ids = [
        str(plan["video_id"])
        for plan in ordered_plans
        if str(plan["video_id"]) in existing_sources
    ]
    new_count = 0
    for batch_start in range(0, len(ordered_ids), batch_size):
        batch_ids = ordered_ids[batch_start : batch_start + batch_size]
        batch_rows: list[dict[str, Any]] = []
        for video_id in batch_ids:
            if video_id in completed_rows:
                continue
            if maximum_new_videos and new_count >= maximum_new_videos:
                break
            source = existing_sources[video_id]
            batch_rows.append(
                materialize_crop_transaction(
                    config=config,
                    plan=plans[video_id],
                    plan_hash=plan_hash,
                    encoded_audio=None,
                    audio_path=Path(source["audio_path"]),
                    source_audio_info=source["audio_info"],
                    source_provenance={
                        "source_route": _plan_source_route(plans[video_id]),
                        "transport": "existing_manifest_crop_source",
                        "hf_dataset": _plan_hf_dataset(plans[video_id]),
                        "hf_revision": _plan_revision(
                            plans[video_id], config.resolved_revision
                        ),
                        "hf_split": str(plans[video_id]["hf_split"]),
                        "video_id": video_id,
                        "plan_sha256": plan_hash,
                        "existing_manifest": {
                            "path": str(source["manifest_path"]),
                            "sha256": str(source["manifest_sha256"]),
                            "original_source_route": str(
                                source["original_source_route"]
                            ),
                        },
                    },
                )
            )
            new_count += 1
        if batch_rows:
            fragment_path, fragment_receipt = _write_fragment(
                config.output_dir.resolve(),
                key=f"existing-crops-{batch_start // batch_size:05d}",
                new_rows=batch_rows,
            )
            for row in batch_rows:
                completed_rows[str(row["video_id"])] = row
            fragments = _upsert_fragment_receipt(
                state.get("manifest_fragments", []), fragment_receipt
            )
            state["manifest_fragments"] = fragments
            state["completed_video_ids"] = sorted(completed_rows)
            _write_manifest_index(
                config.output_dir.resolve(),
                plan_hash=plan_hash,
                plan_rows=len(plans),
                completed_rows=completed_rows,
                fragments=fragments,
            )
            _save_state(state_path, state)
            if checkpoint_hook:
                checkpoint_hook(
                    "materialize_existing_crop_batch_committed",
                    {
                        "batch_index": batch_start // batch_size,
                        "new_videos": len(batch_rows),
                        "fragment_path": str(fragment_path),
                        "completed_videos": len(completed_rows),
                    },
                )
        if maximum_new_videos and new_count >= maximum_new_videos:
            break
    return new_count


def materialize_locations(
    *,
    config: MaterializationConfig,
    ordered_plans: Sequence[Mapping[str, Any]],
    plans: Mapping[str, Mapping[str, Any]],
    plan_hash: str,
    state: dict[str, Any],
    state_path: Path,
    completed_rows: dict[str, dict[str, Any]],
    checkpoint_hook: CheckpointHook | None,
    new_videos_already: int = 0,
) -> int:
    locations = state.get("locations", {})
    order = {str(row["video_id"]): index for index, row in enumerate(ordered_plans)}
    grouped: dict[tuple[str, int], list[str]] = {}
    for video_id, location in locations.items():
        if video_id in completed_rows:
            continue
        grouped.setdefault(
            (str(location["parquet_url"]), int(location["row_group"])), []
        ).append(video_id)
    for ids in grouped.values():
        ids.sort(key=lambda value: order[value])
    groups = sorted(
        grouped,
        key=lambda item: min(order[video_id] for video_id in grouped[item]),
    )
    new_count = 0
    for url, row_group in groups:
        remaining_limit = (
            config.max_new_videos - new_videos_already - new_count
            if config.max_new_videos
            else None
        )
        if remaining_limit is not None and remaining_limit <= 0:
            break
        selected_ids = grouped[(url, row_group)]
        if remaining_limit is not None:
            selected_ids = selected_ids[:remaining_limit]
        selected_set = set(selected_ids)
        table = read_row_group(
            url,
            row_group,
            columns=["video_id", "audio"],
            block_size=config.block_size_bytes,
            retry=config.retry,
        )
        video_values = table.column("video_id").to_pylist()
        audio_column = table.column("audio")
        found: set[str] = set()
        rows_to_commit: list[dict[str, Any]] = []
        shard_metadata = state.get("shard_metadata", {}).get(url, {})
        for row_index, raw_video_id in enumerate(video_values):
            video_id = str(raw_video_id)
            if video_id not in selected_set:
                continue
            expected = locations[video_id]
            if int(expected["row_index"]) != row_index:
                raise MaterializationError(
                    f"row index changed for {video_id}: {row_index} != {expected['row_index']}"
                )
            data, source_path = _extract_audio_payload(audio_column[row_index])
            bytes_info = validate_audio_bytes(data)
            parquet_provenance = {
                "url": url,
                "file_size_bytes": int(shard_metadata.get("size_bytes") or 0),
                "blob_id": str(shard_metadata.get("blob_id") or ""),
                "lfs_sha256": str(shard_metadata.get("lfs_sha256") or ""),
                "xet_hash": str(shard_metadata.get("xet_hash") or ""),
                "row_group": row_group,
                "row_index": row_index,
                "encoded_audio_path": source_path,
            }
            if config.storage_mode == "requested_crops":
                rows_to_commit.append(
                    materialize_crop_transaction(
                        config=config,
                        plan=plans[video_id],
                        plan_hash=plan_hash,
                        encoded_audio=data,
                        audio_path=None,
                        source_audio_info=bytes_info,
                        source_provenance={
                            "source_route": _plan_source_route(plans[video_id]),
                            "transport": "hf_parquet_row_group",
                            "hf_dataset": _plan_hf_dataset(plans[video_id]),
                            "hf_revision": _plan_revision(
                                plans[video_id], config.resolved_revision
                            ),
                            "hf_split": str(plans[video_id]["hf_split"]),
                            "video_id": video_id,
                            "plan_sha256": plan_hash,
                            "parquet": parquet_provenance,
                        },
                    )
                )
            else:
                suffix = _audio_suffix(source_path)
                output_path = (
                    config.output_dir.resolve()
                    / "audio"
                    / str(plans[video_id]["hf_split"])
                    / f"{video_id}{suffix}"
                )
                if output_path.exists():
                    disk_info = validate_audio_path(output_path)
                    if disk_info["audio_sha256"] != bytes_info["audio_sha256"]:
                        raise MaterializationError(
                            f"existing destination audio differs for {video_id}: {output_path}"
                        )
                else:
                    atomic_bytes(output_path, data)
                    disk_info = validate_audio_path(
                        output_path, str(bytes_info["audio_sha256"])
                    )
                rows_to_commit.append(
                    build_manifest_row(
                        plan=plans[video_id],
                        plan_hash=plan_hash,
                        audio_path=output_path,
                        audio_info=disk_info,
                        resolved_revision=config.resolved_revision,
                        transport="hf_parquet_row_group",
                        parquet=parquet_provenance,
                    )
                )
            found.add(video_id)
        missing = selected_set - found
        if missing:
            raise MaterializationError(
                f"matching row group no longer contains planned ids: {sorted(missing)}"
            )
        fragment_path, fragment_receipt = _write_fragment(
            config.output_dir.resolve(),
            key=_fragment_key(url, row_group),
            new_rows=rows_to_commit,
        )
        for row in rows_to_commit:
            completed_rows[str(row["video_id"])] = row
        new_count += len(rows_to_commit)
        fragments = _upsert_fragment_receipt(
            state.get("manifest_fragments", []), fragment_receipt
        )
        state["manifest_fragments"] = fragments
        state["completed_video_ids"] = sorted(completed_rows)
        _write_manifest_index(
            config.output_dir.resolve(),
            plan_hash=plan_hash,
            plan_rows=len(plans),
            completed_rows=completed_rows,
            fragments=fragments,
        )
        _save_state(state_path, state)
        if checkpoint_hook:
            checkpoint_hook(
                "materialize_row_group_committed",
                {
                    "parquet_url": url,
                    "row_group": row_group,
                    "new_videos": len(rows_to_commit),
                    "fragment_path": str(fragment_path),
                    "completed_videos": len(completed_rows),
                },
            )
    return new_count


def _estimate_receipt(
    *,
    state: Mapping[str, Any],
    completed_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    locations = state.get("locations", {})
    matching_groups = {
        (str(value["parquet_url"]), int(value["row_group"]))
        for value in locations.values()
    }
    scan_bytes = 0
    materialization_network = 0
    target_disk_estimate = 0.0
    remaining_target_disk_estimate = 0.0
    shard_metadata = state.get("shard_metadata", {})
    for url, metadata in shard_metadata.items():
        for group in metadata.get("row_groups", []):
            scan_bytes += int(group.get("video_id_compressed_bytes") or 0)
            key = (str(url), int(group["index"]))
            if key not in matching_groups:
                continue
            audio_bytes = int(group.get("audio_compressed_bytes") or 0)
            materialization_network += audio_bytes
            group_plan_rows = sum(
                1
                for value in locations.values()
                if str(value["parquet_url"]) == url
                and int(value["row_group"]) == int(group["index"])
            )
            group_remaining_rows = sum(
                1
                for video_id, value in locations.items()
                if video_id not in completed_rows
                and str(value["parquet_url"]) == url
                and int(value["row_group"]) == int(group["index"])
            )
            if int(group.get("num_rows") or 0) > 0:
                target_disk_estimate += (
                    audio_bytes * group_plan_rows / int(group["num_rows"])
                )
                remaining_target_disk_estimate += (
                    audio_bytes * group_remaining_rows / int(group["num_rows"])
                )
    actual_disk = sum(
        int(row.get("audio_bytes") or row.get("crop_audio_bytes") or 0)
        for row in completed_rows.values()
    )
    crop_storage = state.get("crop_storage")
    method = "Parquet compressed column-chunk metadata; excludes protocol/header overhead"
    if crop_storage:
        per_video = crop_storage.get("per_video", {})
        target_disk_estimate = int(crop_storage.get("estimated_total_bytes") or 0)
        remaining_target_disk_estimate = sum(
            int(value.get("estimated_bytes") or 0)
            for video_id, value in per_video.items()
            if video_id not in completed_rows
        )
        method = str(crop_storage.get("estimation_method") or method)
    scanned_shard_payload_bytes = sum(
        int(metadata.get("size_bytes") or 0)
        for metadata in shard_metadata.values()
    )
    matching_urls = {url for url, _ in matching_groups}
    matching_shard_payload_bytes = sum(
        int(shard_metadata.get(url, {}).get("size_bytes") or 0)
        for url in matching_urls
    )
    return {
        "method": method,
        # Column-chunk sums are useful transfer estimates, but are not HTTP
        # upper bounds: footer/range readahead, protocol overhead and retries
        # are intentionally outside them.
        "video_id_scan_compressed_column_bytes_estimate": scan_bytes,
        "matching_audio_row_groups_compressed_column_bytes_estimate": materialization_network,
        "total_compressed_column_bytes_estimate": scan_bytes + materialization_network,
        # Full shard payload sizes provide a deliberately conservative
        # no-retry ceiling for comparison, still excluding HTTP overhead.
        "video_id_scan_full_shard_payload_bytes_no_retry_ceiling": scanned_shard_payload_bytes,
        "matching_audio_full_shard_payload_bytes_no_retry_ceiling": matching_shard_payload_bytes,
        "total_full_shard_payload_bytes_no_retry_ceiling": (
            scanned_shard_payload_bytes + matching_shard_payload_bytes
        ),
        "network_estimate_excludes": [
            "HTTP/protocol overhead",
            "range readahead/cache effects",
            "retry transfers",
        ],
        "planned_audio_disk_bytes_estimate": int(round(target_disk_estimate)),
        "remaining_planned_audio_disk_bytes_estimate": int(
            round(remaining_target_disk_estimate)
        ),
        "completed_audio_disk_bytes_actual": actual_disk,
        "matching_row_groups": len(matching_groups),
        "crop_items_total": int(
            crop_storage.get("total_crop_items", 0) if crop_storage else 0
        ),
        "crop_items_completed": sum(
            int(row.get("crop_item_count") or 0) for row in completed_rows.values()
        ),
        "requested_crop_seconds_total": float(
            crop_storage.get("total_requested_seconds", 0.0)
            if crop_storage
            else 0.0
        ),
    }


def disk_preflight(
    *,
    config: MaterializationConfig,
    state: Mapping[str, Any],
    completed_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    estimates = _estimate_receipt(state=state, completed_rows=completed_rows)
    remaining_estimate = int(
        estimates["remaining_planned_audio_disk_bytes_estimate"]
    )
    guarded_additional = int(
        round(remaining_estimate * config.disk_estimate_safety_factor)
    )
    free = int(shutil.disk_usage(config.output_dir.resolve()).free)
    projected_free = free - guarded_additional
    return {
        "safe": projected_free >= config.minimum_free_disk_bytes,
        "free_bytes_before": free,
        "minimum_free_reserve_bytes": config.minimum_free_disk_bytes,
        "remaining_audio_bytes_estimate": remaining_estimate,
        "safety_factor": config.disk_estimate_safety_factor,
        "guarded_additional_bytes": guarded_additional,
        "projected_free_bytes_after": projected_free,
    }


def _write_receipt(
    *,
    config: MaterializationConfig,
    plan_hash: str,
    plan_files: Sequence[Mapping[str, Any]],
    plan_rows: int,
    state: Mapping[str, Any],
    completed_rows: Mapping[str, Mapping[str, Any]],
    existing_receipts: Sequence[Mapping[str, Any]],
    new_this_run: int,
    status: str,
) -> dict[str, Any]:
    locations = state.get("locations", {})
    preflight = disk_preflight(
        config=config, state=state, completed_rows=completed_rows
    )
    local_source_video_ids = set(
        str(value) for value in state.get("local_source_video_ids", [])
    )
    accounted_video_ids = (
        set(str(value) for value in locations)
        | set(completed_rows)
        | local_source_video_ids
    )
    reused_video_ids = {
        video_id
        for video_id, row in completed_rows.items()
        if row.get("source_provenance", {}).get("transport")
        == "existing_manifest_reuse"
    }
    receipt = {
        "format": FORMAT,
        "status": status,
        "scan_only": config.scan_only,
        "storage_mode": config.storage_mode,
        "location_mode": str(state.get("location_mode") or "scan_if_missing"),
        "require_preindexed_locations": config.require_preindexed_locations,
        "preindexed_location_summary": dict(
            state.get("preindexed_location_summary") or {}
        ),
        "plan_sha256": plan_hash,
        "plan_files": list(plan_files),
        "plan_rows": plan_rows,
        "resolved_revision": config.resolved_revision,
        "hf_dataset": config.hf_dataset,
        "parquet_pattern": config.parquet_pattern,
        "parquet_shards_considered": sum(
            len(value) for value in state.get("shards", {}).values()
        ),
        "scan_workers": config.scan_workers,
        "row_groups_scanned": len(state.get("scanned_row_groups", [])),
        "plan_rows_located_remotely": len(locations),
        "plan_rows_reused_from_existing_manifests": len(reused_video_ids),
        "plan_rows_with_validated_local_crop_source": len(local_source_video_ids),
        "plan_rows_accounted_for": len(accounted_video_ids),
        "plan_rows_unaccounted_for": plan_rows - len(accounted_video_ids),
        "plan_rows_completed": len(completed_rows),
        "plan_rows_remaining": plan_rows - len(completed_rows),
        "complete": len(completed_rows) == plan_rows,
        "new_videos_this_run": new_this_run,
        "max_new_videos": config.max_new_videos,
        "existing_manifests": list(existing_receipts),
        "estimates": _estimate_receipt(state=state, completed_rows=completed_rows),
        "disk_preflight": preflight,
        "durability": {
            "audio_atomic": True,
            "row_group_manifest_fragments_atomic": True,
            "manifest_index_atomic": True,
            "state_atomic": True,
            "resume_rebuilds_from_fragments": True,
        },
    }
    atomic_json(config.output_dir.resolve() / "materialization_receipt.json", receipt)
    estimates = receipt["estimates"]
    lines = [
        "# AudioSet-Strong plan materialization receipt",
        "",
        f"- Status: `{status}`",
        f"- Plan SHA-256: `{plan_hash}`",
        f"- Plan rows: {plan_rows:,}",
        f"- Remotely located rows: {len(locations):,}",
        f"- Existing audio reused: {len(reused_video_ids):,}",
        f"- Validated local crop sources: {len(local_source_video_ids):,}",
        f"- Accounted plan rows: {len(accounted_video_ids):,}",
        f"- Completed rows: {len(completed_rows):,}",
        f"- Remaining rows: {plan_rows - len(completed_rows):,}",
        f"- Location mode: `{state.get('location_mode', 'scan_if_missing')}`",
        f"- Preindexed locations validated: {int(state.get('preindexed_location_summary', {}).get('validated_plan_rows', 0)):,}",
        f"- Pinned dataset revision: `{config.resolved_revision}`",
        "",
        "## Estimated I/O",
        "",
        f"- video_id compressed-column estimate: {estimates['video_id_scan_compressed_column_bytes_estimate']:,} bytes",
        f"- matching-audio compressed-column estimate: {estimates['matching_audio_row_groups_compressed_column_bytes_estimate']:,} bytes",
        f"- total compressed-column estimate: {estimates['total_compressed_column_bytes_estimate']:,} bytes",
        f"- total full-shard payload no-retry ceiling: {estimates['total_full_shard_payload_bytes_no_retry_ceiling']:,} bytes",
        f"- planned audio disk estimate: {estimates['planned_audio_disk_bytes_estimate']:,} bytes",
        f"- completed audio disk actual: {estimates['completed_audio_disk_bytes_actual']:,} bytes",
        f"- disk preflight safe: {preflight['safe']}",
        f"- free before: {preflight['free_bytes_before']:,} bytes",
        f"- guarded additional: {preflight['guarded_additional_bytes']:,} bytes",
        f"- projected free after: {preflight['projected_free_bytes_after']:,} bytes",
        f"- mandatory reserve: {preflight['minimum_free_reserve_bytes']:,} bytes",
        "",
        "Column estimates exclude protocol overhead, range readahead/cache effects, and retry transfers; they are not HTTP upper bounds.",
    ]
    atomic_bytes(
        config.output_dir.resolve() / "materialization_receipt.md",
        ("\n".join(lines) + "\n").encode("utf-8"),
    )
    return receipt


def _run_materialization_locked(
    config: MaterializationConfig,
    *,
    checkpoint_hook: CheckpointHook | None = None,
) -> dict[str, Any]:
    """Run or resume one exact plan materialization.

    The returned receipt has ``complete=False`` for a bounded smoke run.  The
    CLI maps that condition to a non-zero exit code, so an incomplete subset is
    never accidentally treated as a finished plan.
    """

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered, plans, plan_hash, plan_files = load_plans(config.plan_paths)
    state_path = output_dir / "materialization_state.json"
    state = _load_or_create_state(
        state_path,
        plan_hash=plan_hash,
        file_receipts=plan_files,
        pattern=config.parquet_pattern,
        hf_dataset=config.hf_dataset,
        revision=config.resolved_revision,
        storage_mode=config.storage_mode,
        location_mode=(
            "preindexed_required"
            if config.require_preindexed_locations
            else "scan_if_missing"
        ),
    )
    completed_rows, fragments = _load_fragments(
        output_dir, plan_hash=plan_hash, plans=plans
    )
    existing_sources: dict[str, dict[str, Any]] = {}
    if config.storage_mode == "requested_crops":
        existing_sources, existing_receipts = load_existing_audio_sources(
            config.existing_manifest_paths,
            plans=plans,
        )
        state["local_source_video_ids"] = sorted(existing_sources)
        initialize_crop_storage_state(
            state=state, plans=plans, existing_sources=existing_sources
        )
        _save_state(state_path, state)
    else:
        reuse_rows, existing_receipts = load_existing_manifest_reuse(
            config.existing_manifest_paths,
            plans=plans,
            plan_hash=plan_hash,
            resolved_revision=config.resolved_revision,
        )
        not_yet_committed_reuse = [
            row for row in reuse_rows if str(row["video_id"]) not in completed_rows
        ]
        if not_yet_committed_reuse:
            _, fragment_receipt = _write_fragment(
                output_dir,
                key="existing-manifest-reuse",
                new_rows=not_yet_committed_reuse,
            )
            for row in not_yet_committed_reuse:
                completed_rows[str(row["video_id"])] = row
            fragments = _upsert_fragment_receipt(
                fragments, fragment_receipt
            )
            state["manifest_fragments"] = fragments
            state["completed_video_ids"] = sorted(completed_rows)
            _write_manifest_index(
                output_dir,
                plan_hash=plan_hash,
                plan_rows=len(plans),
                completed_rows=completed_rows,
                fragments=fragments,
            )
            _save_state(state_path, state)
            if checkpoint_hook:
                checkpoint_hook(
                    "existing_reuse_committed",
                    {"new_videos": len(not_yet_committed_reuse)},
                )

    scan_plan_locations(
        config=config,
        plans=plans,
        already_completed=set(completed_rows) | set(existing_sources),
        state=state,
        state_path=state_path,
        checkpoint_hook=checkpoint_hook,
    )

    preflight = disk_preflight(
        config=config, state=state, completed_rows=completed_rows
    )

    if config.scan_only:
        receipt = _write_receipt(
            config=config,
            plan_hash=plan_hash,
            plan_files=plan_files,
            plan_rows=len(plans),
            state=state,
            completed_rows=completed_rows,
            existing_receipts=existing_receipts,
            new_this_run=0,
            status=(
                "scan_complete" if preflight["safe"] else "scan_complete_preflight_unsafe"
            ),
        )
        return receipt

    if not preflight["safe"]:
        _write_receipt(
            config=config,
            plan_hash=plan_hash,
            plan_files=plan_files,
            plan_rows=len(plans),
            state=state,
            completed_rows=completed_rows,
            existing_receipts=existing_receipts,
            new_this_run=0,
            status="blocked_unsafe_disk_preflight",
        )
        raise MaterializationError(
            "unsafe disk preflight: materialization would leave "
            f"{preflight['projected_free_bytes_after']} bytes, below required "
            f"reserve {preflight['minimum_free_reserve_bytes']} bytes"
        )

    local_new_count = 0
    if config.storage_mode == "requested_crops" and existing_sources:
        local_new_count = materialize_existing_crop_sources(
            config=config,
            ordered_plans=ordered,
            plans=plans,
            plan_hash=plan_hash,
            existing_sources=existing_sources,
            state=state,
            state_path=state_path,
            completed_rows=completed_rows,
            checkpoint_hook=checkpoint_hook,
            maximum_new_videos=config.max_new_videos,
        )
    new_count = local_new_count + materialize_locations(
        config=config,
        ordered_plans=ordered,
        plans=plans,
        plan_hash=plan_hash,
        state=state,
        state_path=state_path,
        completed_rows=completed_rows,
        checkpoint_hook=checkpoint_hook,
        new_videos_already=local_new_count,
    )
    completed_rows, fragments = _load_fragments(
        output_dir, plan_hash=plan_hash, plans=plans
    )
    ordered_manifest = [
        completed_rows[str(plan["video_id"])]
        for plan in ordered
        if str(plan["video_id"]) in completed_rows
    ]
    partial_path = output_dir / "audioset_strong_plan_manifest.partial.jsonl"
    atomic_jsonl(partial_path, ordered_manifest)
    complete = len(completed_rows) == len(plans)
    if complete:
        atomic_jsonl(output_dir / "audioset_strong_plan_manifest.jsonl", ordered_manifest)
    if config.storage_mode == "requested_crops":
        crop_manifest = [
            crop
            for transaction in ordered_manifest
            for crop in transaction.get("crop_records", [])
        ]
        atomic_jsonl(
            output_dir / "audioset_strong_crop_manifest.partial.jsonl",
            crop_manifest,
        )
        if complete:
            expected_items = int(
                state.get("crop_storage", {}).get("total_crop_items", 0)
            )
            if len(crop_manifest) != expected_items:
                raise MaterializationError(
                    f"complete crop manifest has {len(crop_manifest)} items, "
                    f"expected {expected_items}"
                )
            atomic_jsonl(
                output_dir / "audioset_strong_crop_manifest.jsonl",
                crop_manifest,
            )
    state["manifest_fragments"] = fragments
    state["completed_video_ids"] = sorted(completed_rows)
    _write_manifest_index(
        output_dir,
        plan_hash=plan_hash,
        plan_rows=len(plans),
        completed_rows=completed_rows,
        fragments=fragments,
        artifacts_published=complete,
    )
    _save_state(state_path, state)
    return _write_receipt(
        config=config,
        plan_hash=plan_hash,
        plan_files=plan_files,
        plan_rows=len(plans),
        state=state,
        completed_rows=completed_rows,
        existing_receipts=existing_receipts,
        new_this_run=new_count,
        status="complete" if complete else "incomplete_max_new_videos",
    )


def run_materialization(
    config: MaterializationConfig,
    *,
    checkpoint_hook: CheckpointHook | None = None,
) -> dict[str, Any]:
    """Run one materializer process exclusively for its output directory."""

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".materialization.lock"
    lock_handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MaterializationError(
                f"another materializer process holds the output lock: {lock_path}"
            ) from error
        lock_handle.seek(0)
        lock_handle.truncate()
        lock_handle.write(f"pid={os.getpid()}\n")
        lock_handle.flush()
        os.fsync(lock_handle.fileno())
        return _run_materialization_locked(
            config,
            checkpoint_hook=checkpoint_hook,
        )
    finally:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()
