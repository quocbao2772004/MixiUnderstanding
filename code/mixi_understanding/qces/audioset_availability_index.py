"""Exact, resumable availability indexing for AudioSet Parquet mirrors.

The indexer projects ``video_id`` plus small optional label metadata columns.
It never reads or writes audio payloads.  Each Parquet row group is committed
as one atomic fragment; on restart, validated fragments rather than mutable
state are authoritative.

Multiple mirrors are supported through :class:`AvailabilityRoute`.  This is
useful because AudioSet mirrors expose different split names and layouts.  The
result deliberately keeps every physical location, so duplicate video IDs can
be audited rather than silently overwritten.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq


FORMAT = "qces_audioset_availability_index_v1"
FRAGMENT_FORMAT = "qces_audioset_availability_row_group_fragment_v1"
ENTRY_FORMAT = "qces_audioset_availability_entry_v1"
SHARD_FORMAT = "qces_audioset_availability_shard_v1"
DEFAULT_SOURCE_ROUTE = "enyoukai_audioset_strong_pinned_parquet_v1"
DEFAULT_HF_DATASET = "enyoukai/AudioSet-Strong"
DEFAULT_PARQUET_PATTERN = (
    "hf://datasets/enyoukai/AudioSet-Strong@{revision}/data/{split}-*.parquet"
)
# ``ParquetFile`` pre-buffering already coalesces projected column ranges.  A
# second fsspec readahead cache fetches a full block around every coalesced
# range; on AudioSet that block can include bytes from the adjacent ``audio``
# column and greatly amplifies network traffic.  The no-cache reader keeps the
# projection exact while Arrow remains responsible for range coalescing.
REMOTE_PARQUET_CACHE_TYPE = "none"
PARQUET_PRE_BUFFER = True


class AvailabilityIndexError(RuntimeError):
    """Raised when the mirror snapshot or a durable fragment is inconsistent."""


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
class AvailabilityRoute:
    """One immutable Parquet mirror route and its logical split names."""

    source_route: str
    hf_dataset: str
    resolved_revision: str
    parquet_pattern: str
    splits: tuple[str, ...]
    optional_metadata_columns: tuple[str, ...] = ("labels", "human_labels")

    def __post_init__(self) -> None:
        if not self.source_route.strip():
            raise ValueError("source_route must be non-empty")
        if not self.hf_dataset.strip():
            raise ValueError("hf_dataset must be non-empty")
        if not self.resolved_revision.strip():
            raise ValueError("resolved_revision must be non-empty")
        if not self.parquet_pattern.strip():
            raise ValueError("parquet_pattern must be non-empty")
        if self.parquet_pattern.startswith("hf://"):
            if self.resolved_revision.lower() in {"main", "master", "latest"}:
                raise ValueError("HF availability routes require an immutable revision")
            rendered_probe = self.parquet_pattern.format(
                split="__split__",
                revision=self.resolved_revision,
                hf_dataset=self.hf_dataset,
                source_route=self.source_route,
            )
            if self.resolved_revision not in rendered_probe:
                raise ValueError(
                    "HF parquet_pattern does not embed its resolved immutable revision"
                )
        if not self.splits or any(not value.strip() for value in self.splits):
            raise ValueError("route must contain at least one non-empty split")
        if len(set(self.splits)) != len(self.splits):
            raise ValueError(f"duplicate splits in route {self.source_route}")
        forbidden = {
            value
            for value in self.optional_metadata_columns
            if value == "video_id" or value == "audio" or value.startswith("audio.")
        }
        if forbidden:
            raise ValueError(
                f"optional metadata columns cannot contain {sorted(forbidden)}"
            )
        if len(set(self.optional_metadata_columns)) != len(
            self.optional_metadata_columns
        ):
            raise ValueError(
                f"duplicate optional metadata columns in route {self.source_route}"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "source_route": self.source_route,
            "hf_dataset": self.hf_dataset,
            "resolved_revision": self.resolved_revision,
            "parquet_pattern": self.parquet_pattern,
            "splits": list(self.splits),
            "optional_metadata_columns": list(self.optional_metadata_columns),
        }


@dataclass(frozen=True)
class AvailabilityAllowlist:
    """A normalized video-ID filter, globally or route/split scoped."""

    video_ids: frozenset[str]
    normalized_sha256: str
    source_path: str
    source_sha256: str
    input_rows: int
    source_route: str = "*"
    hf_split: str = "*"

    def __post_init__(self) -> None:
        if not self.video_ids or any(not value for value in self.video_ids):
            raise ValueError("availability allowlist must contain non-empty IDs")
        expected = _sha256_bytes(_canonical(sorted(self.video_ids)))
        if self.normalized_sha256 != expected:
            raise ValueError("allowlist normalized SHA-256 does not match its IDs")
        if self.input_rows < len(self.video_ids):
            raise ValueError("allowlist input_rows cannot be smaller than unique IDs")
        if not self.source_route or not self.hf_split:
            raise ValueError("allowlist scope values must be non-empty or '*'")

    def payload(self) -> dict[str, Any]:
        return {
            "source_route": self.source_route,
            "hf_split": self.hf_split,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "normalized_sha256": self.normalized_sha256,
            "input_rows": self.input_rows,
            "unique_video_ids": len(self.video_ids),
        }


def default_enyoukai_route(*, resolved_revision: str) -> AvailabilityRoute:
    return AvailabilityRoute(
        source_route=DEFAULT_SOURCE_ROUTE,
        hf_dataset=DEFAULT_HF_DATASET,
        resolved_revision=resolved_revision,
        parquet_pattern=DEFAULT_PARQUET_PATTERN,
        splits=("train", "test"),
    )


@dataclass(frozen=True)
class AvailabilityIndexConfig:
    routes: tuple[AvailabilityRoute, ...]
    output_dir: Path
    allowlists: tuple[AvailabilityAllowlist, ...] = ()
    scan_workers: int = 4
    block_size_bytes: int = 1 << 20
    max_new_row_groups: int = 0
    require_unique_video_ids: bool = False
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __post_init__(self) -> None:
        if not self.routes:
            raise ValueError("at least one availability route is required")
        names = [route.source_route for route in self.routes]
        if len(names) != len(set(names)):
            raise ValueError(f"source_route values must be unique: {names}")
        if self.scan_workers < 1:
            raise ValueError("scan_workers must be >= 1")
        if self.block_size_bytes < 1:
            raise ValueError("block_size_bytes must be positive")
        if self.max_new_row_groups < 0:
            raise ValueError("max_new_row_groups must be >= 0")
        scopes = [
            (allowlist.source_route, allowlist.hf_split)
            for allowlist in self.allowlists
        ]
        if len(scopes) != len(set(scopes)):
            raise ValueError(f"duplicate allowlist scopes: {scopes}")


CheckpointHook = Callable[[str, Mapping[str, Any]], None]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_parent(path: Path) -> None:
    try:
        descriptor = os.open(path.parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_bytes(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_bytes(
        path,
        b"".join(_canonical(row) + b"\n" for row in rows),
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AvailabilityIndexError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise AvailabilityIndexError(f"expected JSON object in {path}")
    return value


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
        except (OSError, TimeoutError, ConnectionError, pa.ArrowIOError) as error:
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
    raise AvailabilityIndexError(
        f"failed {description} after {policy.max_attempts} attempts: {last_error}"
    ) from last_error


def _url_to_fs(url: str) -> tuple[Any, str]:
    return fsspec.core.url_to_fs(url)


def _open_for_parquet(fs: Any, path: str, block_size: int) -> Any:
    protocol = fs.protocol
    protocols = set(protocol) if isinstance(protocol, (tuple, list)) else {protocol}
    if protocols & {"file", "local", None}:
        return fs.open(path, "rb")
    return fs.open(
        path,
        "rb",
        block_size=block_size,
        cache_type=REMOTE_PARQUET_CACHE_TYPE,
    )


def _glob_urls(route: AvailabilityRoute, split: str) -> list[str]:
    rendered = route.parquet_pattern.format(
        split=split,
        revision=route.resolved_revision,
        hf_dataset=route.hf_dataset,
        source_route=route.source_route,
    )
    fs, path_pattern = _url_to_fs(rendered)
    if any(character in path_pattern for character in "*?["):
        paths = sorted(fs.glob(path_pattern))
    else:
        paths = [path_pattern] if fs.exists(path_pattern) else []
    urls = [str(fs.unstrip_protocol(path)) for path in paths]
    if not urls:
        raise AvailabilityIndexError(
            f"no Parquet shards for route={route.source_route} split={split}: {rendered}"
        )
    return urls


def _lfs_value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _file_provenance(info: Mapping[str, Any]) -> dict[str, Any]:
    lfs = info.get("lfs")
    return {
        "size_bytes": int(info.get("size") or 0),
        "blob_id": str(info.get("blob_id") or ""),
        "xet_hash": str(info.get("xet_hash") or ""),
        "lfs_sha256": str(_lfs_value(lfs, "sha256", "") if lfs else ""),
        "lfs_size_bytes": int(_lfs_value(lfs, "size", 0) if lfs else 0),
    }


def _shard_identity(source_route: str, hf_split: str, url: str) -> str:
    return _sha256_bytes(
        _canonical(
            {
                "source_route": source_route,
                "hf_split": hf_split,
                "parquet_url": url,
            }
        )
    )


def _inspect_shard(
    descriptor: Mapping[str, Any],
    *,
    block_size: int,
    retry: RetryPolicy,
) -> dict[str, Any]:
    url = str(descriptor["parquet_url"])

    def operation() -> dict[str, Any]:
        fs, path = _url_to_fs(url)
        file_info = _file_provenance(fs.info(path))
        with _open_for_parquet(fs, path, block_size) as handle:
            parquet = pq.ParquetFile(handle, pre_buffer=PARQUET_PRE_BUFFER)
            if "video_id" not in parquet.schema_arrow.names:
                raise AvailabilityIndexError(
                    f"video_id column missing from {url}: {parquet.schema_arrow.names}"
                )
            requested_metadata = [
                str(value)
                for value in descriptor.get("optional_metadata_columns", [])
            ]
            available_metadata = [
                value for value in requested_metadata if value in parquet.schema_arrow.names
            ]
            row_groups: list[dict[str, Any]] = []
            for row_group in range(parquet.num_row_groups):
                metadata = parquet.metadata.row_group(row_group)
                compressed_bytes = 0
                for column_index in range(metadata.num_columns):
                    column = metadata.column(column_index)
                    if str(column.path_in_schema) == "video_id":
                        compressed_bytes += int(column.total_compressed_size or 0)
                row_groups.append(
                    {
                        "index": row_group,
                        "num_rows": int(metadata.num_rows),
                        "video_id_compressed_bytes": compressed_bytes,
                    }
                )
            payload = {
                "format": SHARD_FORMAT,
                **dict(descriptor),
                **file_info,
                "num_rows": int(parquet.metadata.num_rows),
                "num_row_groups": int(parquet.num_row_groups),
                "row_groups": row_groups,
                "available_metadata_columns": available_metadata,
                "projected_columns": ["video_id", *available_metadata],
            }
            payload["shard_metadata_sha256"] = _sha256_bytes(_canonical(payload))
            return payload

    return _retry(
        operation,
        policy=retry,
        description=f"inspect Parquet shard {url}",
    )


def _route_payloads(config: AvailabilityIndexConfig) -> list[dict[str, Any]]:
    return [route.payload() for route in config.routes]


def _allowlist_payloads(config: AvailabilityIndexConfig) -> list[dict[str, Any]]:
    return [value.payload() for value in config.allowlists]


def _effective_allowlist(
    config: AvailabilityIndexConfig,
    *,
    source_route: str,
    hf_split: str,
) -> frozenset[str] | None:
    if not config.allowlists:
        return None
    matching = [
        value
        for value in config.allowlists
        if value.source_route in {"*", source_route}
        and value.hf_split in {"*", hf_split}
    ]
    if not matching:
        raise AvailabilityIndexError(
            f"no allowlist scope covers route={source_route} split={hf_split}"
        )
    return frozenset(
        video_id for value in matching for video_id in value.video_ids
    )


def _new_state(config: AvailabilityIndexConfig) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "schema_version": 1,
        "routes": _route_payloads(config),
        "allowlists": _allowlist_payloads(config),
        "shards": [],
        "shard_metadata": {},
        "fragment_receipts": {},
        "updated_unix_seconds": time.time(),
    }


def _load_or_create_state(
    path: Path,
    config: AvailabilityIndexConfig,
) -> dict[str, Any]:
    if not path.exists():
        return _new_state(config)
    state = _load_json(path)
    if state.get("format") != FORMAT:
        raise AvailabilityIndexError(f"unexpected state format in {path}")
    if state.get("routes") != _route_payloads(config):
        raise AvailabilityIndexError(
            "route/pattern/revision changed for an existing output directory"
        )
    if state.get("allowlists", []) != _allowlist_payloads(config):
        raise AvailabilityIndexError(
            "allowlist scope/content changed for an existing output directory"
        )
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_unix_seconds"] = time.time()
    atomic_json(path, state)


def _discover_descriptors(config: AvailabilityIndexConfig) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    seen_physical: dict[tuple[str, str], str] = {}
    for route in config.routes:
        for split in route.splits:
            effective_allowlist = _effective_allowlist(
                config,
                source_route=route.source_route,
                hf_split=split,
            )
            allowlist_hash = (
                _sha256_bytes(_canonical(sorted(effective_allowlist)))
                if effective_allowlist is not None
                else ""
            )
            for url in _glob_urls(route, split):
                physical_key = (route.source_route, url)
                previous_split = seen_physical.get(physical_key)
                if previous_split is not None and previous_split != split:
                    raise AvailabilityIndexError(
                        f"same shard assigned to multiple splits for route "
                        f"{route.source_route}: {url} -> {previous_split}, {split}"
                    )
                seen_physical[physical_key] = split
                descriptors.append(
                    {
                        "source_route": route.source_route,
                        "hf_dataset": route.hf_dataset,
                        "hf_revision": route.resolved_revision,
                        "hf_split": split,
                        "parquet_url": url,
                        "shard_id": _shard_identity(route.source_route, split, url),
                        "optional_metadata_columns": list(
                            route.optional_metadata_columns
                        ),
                        "allowlist_enabled": effective_allowlist is not None,
                        "effective_allowlist_sha256": allowlist_hash,
                        "effective_allowlist_size": (
                            len(effective_allowlist)
                            if effective_allowlist is not None
                            else 0
                        ),
                    }
                )
    return sorted(
        descriptors,
        key=lambda value: (
            str(value["source_route"]),
            str(value["hf_split"]),
            str(value["parquet_url"]),
        ),
    )


def _discover_and_inspect_shards(
    *,
    config: AvailabilityIndexConfig,
    state: dict[str, Any],
    state_path: Path,
    checkpoint_hook: CheckpointHook | None,
) -> list[dict[str, Any]]:
    descriptors = _discover_descriptors(config)
    if state.get("shards") and state["shards"] != descriptors:
        raise AvailabilityIndexError(
            "discovered shard list changed for an existing output directory"
        )
    state["shards"] = descriptors
    _save_state(state_path, state)

    metadata_by_id: dict[str, dict[str, Any]] = {
        str(key): dict(value)
        for key, value in state.get("shard_metadata", {}).items()
    }
    descriptors_by_id = {
        str(value["shard_id"]): value for value in descriptors
    }
    for shard_id, metadata in metadata_by_id.items():
        descriptor = descriptors_by_id.get(shard_id)
        if descriptor is None:
            raise AvailabilityIndexError(
                f"stored metadata references an undiscovered shard: {shard_id}"
            )
        for key, expected in descriptor.items():
            if metadata.get(key) != expected:
                raise AvailabilityIndexError(
                    f"stored shard metadata {key} mismatch for {shard_id}"
                )
        declared_hash = str(metadata.get("shard_metadata_sha256") or "")
        unhashed = dict(metadata)
        unhashed.pop("shard_metadata_sha256", None)
        if not declared_hash or declared_hash != _sha256_bytes(_canonical(unhashed)):
            raise AvailabilityIndexError(
                f"stored shard metadata SHA-256 mismatch for {shard_id}"
            )
    pending = [
        descriptor
        for descriptor in descriptors
        if str(descriptor["shard_id"]) not in metadata_by_id
    ]
    futures: dict[Future[Any], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=config.scan_workers) as executor:
        for descriptor in pending:
            future = executor.submit(
                _inspect_shard,
                descriptor,
                block_size=config.block_size_bytes,
                retry=config.retry,
            )
            futures[future] = descriptor
        for future in as_completed(futures):
            # A completed Future owns its result.  Removing it from the lookup
            # here prevents metadata for every completed shard accumulating
            # until the executor context exits.
            expected = futures.pop(future)
            metadata = future.result()
            shard_id = str(metadata["shard_id"])
            if shard_id != str(expected["shard_id"]):
                raise AvailabilityIndexError(
                    f"shard identity changed while inspecting {expected['parquet_url']}"
                )
            metadata_by_id[shard_id] = metadata
            state["shard_metadata"] = metadata_by_id
            _save_state(state_path, state)
            if checkpoint_hook:
                checkpoint_hook(
                    "shard_metadata_committed",
                    {
                        "shard_id": shard_id,
                        "parquet_url": str(metadata["parquet_url"]),
                    },
                )
    return [metadata_by_id[str(value["shard_id"])] for value in descriptors]


def _fragment_path(output_dir: Path, shard_id: str, row_group: int) -> Path:
    return output_dir / "row_group_fragments" / f"{shard_id[:24]}-rg{row_group:05d}.json"


def _entry_provenance(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "shard_id": str(metadata["shard_id"]),
        "shard_metadata_sha256": str(metadata["shard_metadata_sha256"]),
        "file_size_bytes": int(metadata.get("size_bytes") or 0),
        "blob_id": str(metadata.get("blob_id") or ""),
        "xet_hash": str(metadata.get("xet_hash") or ""),
        "lfs_sha256": str(metadata.get("lfs_sha256") or ""),
        "lfs_size_bytes": int(metadata.get("lfs_size_bytes") or 0),
    }


def _normalize_label_values(value: Any) -> list[str]:
    if hasattr(value, "as_py"):
        value = value.as_py()
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else [value]
    return sorted({str(item).strip() for item in values if str(item).strip()})


def _read_selected_row_groups(
    metadata: Mapping[str, Any],
    row_groups: Sequence[int],
    *,
    allowlist_video_ids: frozenset[str] | None,
    block_size: int,
    retry: RetryPolicy,
) -> list[dict[str, Any]]:
    url = str(metadata["parquet_url"])
    expected_by_group = {
        int(value["index"]): int(value["num_rows"])
        for value in metadata["row_groups"]
    }

    def operation() -> list[dict[str, Any]]:
        projected_columns = [
            "video_id",
            *[str(value) for value in metadata.get("available_metadata_columns", [])],
        ]
        fs, path = _url_to_fs(url)
        with _open_for_parquet(fs, path, block_size) as handle:
            parquet = pq.ParquetFile(handle, pre_buffer=PARQUET_PRE_BUFFER)
            tables = parquet.read_row_groups(
                list(row_groups),
                columns=projected_columns,
                use_threads=True,
            )
        counts = [expected_by_group[value] for value in row_groups]
        if int(tables.num_rows) != sum(counts):
            raise AvailabilityIndexError(
                f"row count changed while reading {url}: "
                f"{tables.num_rows} != {sum(counts)}"
            )
        results: list[dict[str, Any]] = []
        offset = 0
        provenance = _entry_provenance(metadata)
        for row_group, count in zip(row_groups, counts):
            group_table = tables.slice(offset, count)
            values = group_table.column("video_id").to_pylist()
            metadata_values = {
                column: group_table.column(column).to_pylist()
                for column in projected_columns
                if column != "video_id"
            }
            offset += count
            entries: list[dict[str, Any]] = []
            for row_index, raw_video_id in enumerate(values):
                video_id = str(raw_video_id or "")
                if not video_id:
                    raise AvailabilityIndexError(
                        f"empty video_id at {url} row_group={row_group} row={row_index}"
                    )
                if (
                    allowlist_video_ids is not None
                    and video_id not in allowlist_video_ids
                ):
                    continue
                entries.append(
                    {
                        "format": ENTRY_FORMAT,
                        "source_route": str(metadata["source_route"]),
                        "hf_dataset": str(metadata["hf_dataset"]),
                        "hf_revision": str(metadata["hf_revision"]),
                        "hf_split": str(metadata["hf_split"]),
                        "video_id": video_id,
                        "parquet_url": url,
                        "row_group": int(row_group),
                        "row_index": row_index,
                        "labels": _normalize_label_values(
                            metadata_values.get("labels", [None] * count)[row_index]
                        ),
                        "human_labels": _normalize_label_values(
                            metadata_values.get("human_labels", [None] * count)[
                                row_index
                            ]
                        ),
                        "shard_provenance": provenance,
                    }
                )
            fragment = {
                "format": FRAGMENT_FORMAT,
                "source_route": str(metadata["source_route"]),
                "hf_dataset": str(metadata["hf_dataset"]),
                "hf_revision": str(metadata["hf_revision"]),
                "hf_split": str(metadata["hf_split"]),
                "parquet_url": url,
                "shard_id": str(metadata["shard_id"]),
                "shard_metadata_sha256": str(metadata["shard_metadata_sha256"]),
                "row_group": int(row_group),
                "scanned_rows": count,
                "indexed_rows": len(entries),
                "allowlist_enabled": allowlist_video_ids is not None,
                "effective_allowlist_sha256": str(
                    metadata.get("effective_allowlist_sha256") or ""
                ),
                "effective_allowlist_size": int(
                    metadata.get("effective_allowlist_size") or 0
                ),
                "projected_columns": projected_columns,
                "entries": entries,
            }
            fragment["fragment_payload_sha256"] = _sha256_bytes(
                _canonical(fragment)
            )
            results.append(fragment)
        return results

    return _retry(
        operation,
        policy=retry,
        description=f"read video_id row groups {list(row_groups)} from {url}",
    )


def _validate_fragment(
    fragment: Mapping[str, Any],
    *,
    config: AvailabilityIndexConfig,
    metadata_by_id: Mapping[str, Mapping[str, Any]],
    path: Path,
) -> tuple[str, int, list[dict[str, Any]]]:
    if fragment.get("format") != FRAGMENT_FORMAT:
        raise AvailabilityIndexError(f"unexpected fragment format: {path}")
    declared_payload_hash = str(fragment.get("fragment_payload_sha256") or "")
    unhashed = dict(fragment)
    unhashed.pop("fragment_payload_sha256", None)
    if not declared_payload_hash or declared_payload_hash != _sha256_bytes(
        _canonical(unhashed)
    ):
        raise AvailabilityIndexError(f"fragment payload SHA-256 mismatch: {path}")
    shard_id = str(fragment.get("shard_id") or "")
    metadata = metadata_by_id.get(shard_id)
    if metadata is None:
        raise AvailabilityIndexError(f"fragment references unknown shard: {path}")
    for key in (
        "source_route",
        "hf_dataset",
        "hf_revision",
        "hf_split",
        "parquet_url",
        "shard_metadata_sha256",
    ):
        if fragment.get(key) != metadata.get(key):
            raise AvailabilityIndexError(
                f"fragment {key} mismatch for {path}: "
                f"{fragment.get(key)!r} != {metadata.get(key)!r}"
            )
    row_group = int(fragment.get("row_group", -1))
    groups = {int(value["index"]): value for value in metadata["row_groups"]}
    if row_group not in groups:
        raise AvailabilityIndexError(f"fragment row group out of range: {path}")
    expected_rows = int(groups[row_group]["num_rows"])
    if int(fragment.get("scanned_rows", -1)) != expected_rows:
        raise AvailabilityIndexError(f"fragment scanned row count mismatch: {path}")
    for key in (
        "allowlist_enabled",
        "effective_allowlist_sha256",
        "effective_allowlist_size",
    ):
        if fragment.get(key) != metadata.get(key):
            raise AvailabilityIndexError(f"fragment {key} mismatch: {path}")
    expected_projection = [
        "video_id",
        *[str(value) for value in metadata.get("available_metadata_columns", [])],
    ]
    if fragment.get("projected_columns") != expected_projection:
        raise AvailabilityIndexError(f"fragment used unexpected projection: {path}")
    entries = fragment.get("entries")
    if not isinstance(entries, list) or int(
        fragment.get("indexed_rows", -1)
    ) != len(entries):
        raise AvailabilityIndexError(f"fragment entry count mismatch: {path}")
    if not bool(fragment.get("allowlist_enabled")) and len(entries) != expected_rows:
        raise AvailabilityIndexError(f"unfiltered fragment omitted rows: {path}")
    provenance = _entry_provenance(metadata)
    effective_allowlist = _effective_allowlist(
        config,
        source_route=str(metadata["source_route"]),
        hf_split=str(metadata["hf_split"]),
    )
    validated: list[dict[str, Any]] = []
    previous_row_index = -1
    for raw in entries:
        if not isinstance(raw, dict):
            raise AvailabilityIndexError(f"non-object entry in {path}")
        row_index = int(raw.get("row_index", -1))
        if row_index <= previous_row_index or row_index >= expected_rows:
            raise AvailabilityIndexError(
                f"invalid/non-monotonic row index in {path}: {row_index}"
            )
        previous_row_index = row_index
        expected = {
            "format": ENTRY_FORMAT,
            "source_route": str(metadata["source_route"]),
            "hf_dataset": str(metadata["hf_dataset"]),
            "hf_revision": str(metadata["hf_revision"]),
            "hf_split": str(metadata["hf_split"]),
            "video_id": str(raw.get("video_id") or ""),
            "parquet_url": str(metadata["parquet_url"]),
            "row_group": row_group,
            "row_index": row_index,
            "labels": _normalize_label_values(raw.get("labels")),
            "human_labels": _normalize_label_values(raw.get("human_labels")),
            "shard_provenance": provenance,
        }
        if not expected["video_id"] or _canonical(raw) != _canonical(expected):
            raise AvailabilityIndexError(
                f"fragment entry integrity mismatch at {path} row={row_index}"
            )
        if (
            effective_allowlist is not None
            and expected["video_id"] not in effective_allowlist
        ):
            raise AvailabilityIndexError(
                f"fragment persisted video outside effective allowlist: "
                f"{path} row={row_index} video_id={expected['video_id']}"
            )
        validated.append(dict(raw))
    return shard_id, row_group, validated


def _load_fragments(
    *,
    config: AvailabilityIndexConfig,
    output_dir: Path,
    metadata: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    metadata_by_id = {str(value["shard_id"]): value for value in metadata}
    fragments: dict[tuple[str, int], dict[str, Any]] = {}
    receipts: dict[str, dict[str, Any]] = {}
    directory = output_dir / "row_group_fragments"
    paths = sorted(directory.glob("*.json")) if directory.exists() else []
    for path in paths:
        fragment = _load_json(path)
        shard_id, row_group, entries = _validate_fragment(
            fragment,
            config=config,
            metadata_by_id=metadata_by_id,
            path=path,
        )
        key = (shard_id, row_group)
        if key in fragments:
            raise AvailabilityIndexError(
                f"multiple fragments for shard={shard_id} row_group={row_group}"
            )
        fragments[key] = {
            "scanned_rows": int(fragment["scanned_rows"]),
            "entries": entries,
        }
        receipts[f"{shard_id}#rg={row_group}"] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "scanned_rows": int(fragment["scanned_rows"]),
            "indexed_rows": len(entries),
        }
    return fragments, receipts


def _commit_fragment(
    *,
    output_dir: Path,
    fragment: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    shard_id = str(fragment["shard_id"])
    row_group = int(fragment["row_group"])
    path = _fragment_path(output_dir, shard_id, row_group)
    if path.exists():
        existing = _load_json(path)
        if _canonical(existing) != _canonical(fragment):
            raise AvailabilityIndexError(
                f"attempt to replace committed row-group fragment: {path}"
            )
    else:
        atomic_json(path, fragment)
    return path, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "scanned_rows": int(fragment["scanned_rows"]),
        "indexed_rows": len(fragment["entries"]),
    }


def _scan_pending_row_groups(
    *,
    config: AvailabilityIndexConfig,
    metadata: Sequence[Mapping[str, Any]],
    state: dict[str, Any],
    state_path: Path,
    completed: set[tuple[str, int]],
    checkpoint_hook: CheckpointHook | None,
) -> int:
    metadata_by_id = {str(value["shard_id"]): value for value in metadata}
    pending: list[tuple[str, int]] = []
    for shard in metadata:
        shard_id = str(shard["shard_id"])
        for group in shard["row_groups"]:
            key = (shard_id, int(group["index"]))
            if key not in completed:
                pending.append(key)
    if config.max_new_row_groups:
        pending = pending[: config.max_new_row_groups]
    grouped: dict[str, list[int]] = defaultdict(list)
    for shard_id, row_group in pending:
        grouped[shard_id].append(row_group)
    futures: dict[Future[Any], str] = {}
    new_count = 0
    with ThreadPoolExecutor(max_workers=config.scan_workers) as executor:
        for shard_id, row_groups in grouped.items():
            future = executor.submit(
                _read_selected_row_groups,
                metadata_by_id[shard_id],
                row_groups,
                allowlist_video_ids=_effective_allowlist(
                    config,
                    source_route=str(metadata_by_id[shard_id]["source_route"]),
                    hf_split=str(metadata_by_id[shard_id]["hf_split"]),
                ),
                block_size=config.block_size_bytes,
                retry=config.retry,
            )
            futures[future] = shard_id
        for future in as_completed(futures):
            # Fragment payloads can contain many nested label dictionaries.
            # Drop completed Future objects immediately so only in-flight
            # shard results (plus the current result) remain resident.
            shard_id = futures.pop(future)
            committed_payloads: list[dict[str, int | str]] = []
            for fragment in future.result():
                path, receipt = _commit_fragment(
                    output_dir=config.output_dir.resolve(),
                    fragment=fragment,
                )
                if checkpoint_hook:
                    checkpoint_hook(
                        "row_group_fragment_written",
                        {
                            "shard_id": shard_id,
                            "row_group": int(fragment["row_group"]),
                            "path": str(path),
                        },
                    )
                key = f"{shard_id}#rg={int(fragment['row_group'])}"
                state.setdefault("fragment_receipts", {})[key] = receipt
                completed.add((shard_id, int(fragment["row_group"])))
                new_count += 1
                committed_payloads.append(
                    {
                        "shard_id": shard_id,
                        "row_group": int(fragment["row_group"]),
                        "scanned_rows": int(fragment["scanned_rows"]),
                        "indexed_rows": len(fragment["entries"]),
                    }
                )
            # Fragments are individually atomic and authoritative on resume.
            # Commit their derived state receipts once per shard, not once per
            # row group: the agk unbalanced mirror has ~17k groups, and
            # repeatedly rewriting a growing state file would be quadratic.
            if committed_payloads:
                _save_state(state_path, state)
                if checkpoint_hook:
                    for payload in committed_payloads:
                        checkpoint_hook(
                            "row_group_fragment_committed",
                            payload,
                        )
    return new_count


def _ordered_entries(
    fragments: Mapping[tuple[str, int], Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for shard in metadata:
        shard_id = str(shard["shard_id"])
        for group in shard["row_groups"]:
            fragment = fragments.get((shard_id, int(group["index"])))
            if fragment is not None:
                output.extend(dict(value) for value in fragment["entries"])
    return output


def _duplicate_audit(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    locations: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    physical_locations: Counter[tuple[str, str, int, int]] = Counter()
    for entry in entries:
        locations[str(entry["video_id"])].append(entry)
        physical_locations[
            (
                str(entry["source_route"]),
                str(entry["parquet_url"]),
                int(entry["row_group"]),
                int(entry["row_index"]),
            )
        ] += 1
    physical_duplicates = [key for key, count in physical_locations.items() if count > 1]
    if physical_duplicates:
        raise AvailabilityIndexError(
            f"duplicate physical index locations: {physical_duplicates[:10]}"
        )
    duplicate_ids = {
        video_id: rows for video_id, rows in locations.items() if len(rows) > 1
    }
    cross_route = 0
    cross_split = 0
    within_route = 0
    within_route_split = 0
    examples: list[dict[str, Any]] = []
    for video_id in sorted(duplicate_ids):
        rows = duplicate_ids[video_id]
        routes = {str(value["source_route"]) for value in rows}
        route_splits = {
            (str(value["source_route"]), str(value["hf_split"])) for value in rows
        }
        route_counts = Counter(str(value["source_route"]) for value in rows)
        route_split_counts = Counter(
            (str(value["source_route"]), str(value["hf_split"]))
            for value in rows
        )
        if len(routes) > 1:
            cross_route += 1
        if len(route_splits) > len(routes):
            cross_split += 1
        if any(count > 1 for count in route_counts.values()):
            within_route += 1
        if any(count > 1 for count in route_split_counts.values()):
            within_route_split += 1
        if len(examples) < 20:
            examples.append(
                {
                    "video_id": video_id,
                    "locations": [
                        {
                            "source_route": str(value["source_route"]),
                            "hf_split": str(value["hf_split"]),
                            "parquet_url": str(value["parquet_url"]),
                            "row_group": int(value["row_group"]),
                            "row_index": int(value["row_index"]),
                        }
                        for value in rows
                    ],
                }
            )
    return {
        "unique_video_ids": len(locations),
        "video_ids_with_multiple_locations": len(duplicate_ids),
        "cross_route_duplicate_video_ids": cross_route,
        "cross_split_duplicate_video_ids": cross_split,
        "within_route_duplicate_video_ids": within_route,
        "within_route_split_duplicate_video_ids": within_route_split,
        "duplicate_examples": examples,
    }


def _allowlist_audit(
    config: AvailabilityIndexConfig,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    scopes: list[dict[str, Any]] = []
    all_expected: set[tuple[str, str, str]] = set()
    all_matched: set[tuple[str, str, str]] = set()
    for allowlist in config.allowlists:
        matching_entries = [
            entry
            for entry in entries
            if allowlist.source_route in {"*", str(entry["source_route"])}
            and allowlist.hf_split in {"*", str(entry["hf_split"])}
            and str(entry["video_id"]) in allowlist.video_ids
        ]
        matched_ids = {str(entry["video_id"]) for entry in matching_entries}
        unmatched = sorted(allowlist.video_ids - matched_ids)
        scopes.append(
            {
                **allowlist.payload(),
                "matched_unique_video_ids": len(matched_ids),
                "matched_physical_locations": len(matching_entries),
                "unmatched_unique_video_ids": len(unmatched),
                "unmatched_examples": unmatched[:20],
            }
        )
        for video_id in allowlist.video_ids:
            all_expected.add(
                (allowlist.source_route, allowlist.hf_split, video_id)
            )
        for video_id in matched_ids:
            all_matched.add((allowlist.source_route, allowlist.hf_split, video_id))
    return {
        "enabled": bool(config.allowlists),
        "scope_count": len(config.allowlists),
        "normalized_scope_video_ids": len(all_expected),
        "matched_scope_video_ids": len(all_matched),
        "unmatched_scope_video_ids": len(all_expected - all_matched),
        "scopes": scopes,
    }


def _shard_output(metadata: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(value) for value in metadata]


def _snapshot_hash(metadata: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(
        _canonical(
            [
                {
                    "source_route": value["source_route"],
                    "hf_dataset": value["hf_dataset"],
                    "hf_revision": value["hf_revision"],
                    "hf_split": value["hf_split"],
                    "parquet_url": value["parquet_url"],
                    "shard_metadata_sha256": value["shard_metadata_sha256"],
                }
                for value in metadata
            ]
        )
    )


def _write_receipt(
    *,
    config: AvailabilityIndexConfig,
    metadata: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    fragment_count: int,
    mirror_rows_scanned: int,
    new_row_groups: int,
    duplicate_audit: Mapping[str, Any],
    allowlist_audit: Mapping[str, Any],
    complete: bool,
    integrity_pass: bool,
    status: str,
    index_path: Path,
    shard_path: Path,
) -> dict[str, Any]:
    expected_row_groups = sum(int(value["num_row_groups"]) for value in metadata)
    expected_rows = sum(int(value["num_rows"]) for value in metadata)
    rows_by_route_split = Counter(
        (str(value["source_route"]), str(value["hf_split"])) for value in entries
    )
    projected_columns = sorted(
        {
            str(column)
            for shard in metadata
            for column in shard.get("projected_columns", ["video_id"])
        }
    )
    receipt = {
        "format": FORMAT,
        "status": status,
        "complete": complete,
        "integrity_pass": integrity_pass,
        "routes": _route_payloads(config),
        "allowlists": _allowlist_payloads(config),
        "source_snapshot_sha256": _snapshot_hash(metadata),
        "shards": len(metadata),
        "expected_row_groups": expected_row_groups,
        "indexed_row_groups": fragment_count,
        "remaining_row_groups": expected_row_groups - fragment_count,
        "new_row_groups_this_run": new_row_groups,
        "max_new_row_groups": config.max_new_row_groups,
        "mirror_rows_total": expected_rows,
        "mirror_rows_scanned": mirror_rows_scanned,
        "mirror_rows_remaining": expected_rows - mirror_rows_scanned,
        "indexed_allowlist_matches": len(entries),
        "rows_by_route_split": {
            f"{route}/{split}": count
            for (route, split), count in sorted(rows_by_route_split.items())
        },
        "duplicates": dict(duplicate_audit),
        "allowlist_audit": dict(allowlist_audit),
        "require_unique_video_ids": config.require_unique_video_ids,
        "projection": {
            "columns_read": projected_columns,
            "audio_column_read": False,
        },
        "outputs": {
            "availability_index": {
                "path": str(index_path.resolve()),
                "sha256": sha256_file(index_path),
                "rows": len(entries),
                "final": complete and integrity_pass,
            },
            "shard_index": {
                "path": str(shard_path.resolve()),
                "sha256": sha256_file(shard_path),
                "rows": len(metadata),
                "final": complete and integrity_pass,
            },
        },
        "durability": {
            "row_group_fragments_atomic": True,
            "state_atomic": True,
            "resume_rebuilds_from_fragments": True,
            "final_receipt_is_commit_marker": complete and integrity_pass,
        },
    }
    output_dir = config.output_dir.resolve()
    atomic_json(output_dir / "availability_receipt.json", receipt)
    lines = [
        "# AudioSet mirror availability index",
        "",
        f"- Status: `{status}`",
        f"- Complete: {complete}",
        f"- Integrity pass: {integrity_pass}",
        f"- Routes: {len(config.routes):,}",
        f"- Shards: {len(metadata):,}",
        f"- Indexed row groups: {fragment_count:,}/{expected_row_groups:,}",
        f"- Mirror rows scanned: {mirror_rows_scanned:,}/{expected_rows:,}",
        f"- Persisted allowlist matches: {len(entries):,}",
        f"- Unique video IDs: {duplicate_audit['unique_video_ids']:,}",
        f"- Video IDs with multiple locations: "
        f"{duplicate_audit['video_ids_with_multiple_locations']:,}",
        f"- Unmatched allowlist IDs: "
        f"{allowlist_audit['unmatched_scope_video_ids']:,}",
        f"- Source snapshot SHA-256: `{receipt['source_snapshot_sha256']}`",
        "",
        f"Projected metadata columns: {', '.join(projected_columns)}.",
        "No audio payload was read.",
        "The complete receipt is the publication commit marker.",
        "",
    ]
    atomic_bytes(
        output_dir / "availability_receipt.md",
        "\n".join(lines).encode("utf-8"),
    )
    return receipt


def _run_availability_index_locked(
    config: AvailabilityIndexConfig,
    *,
    checkpoint_hook: CheckpointHook | None = None,
) -> dict[str, Any]:
    """Build or resume an exact physical availability index."""

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "availability_state.json"
    state = _load_or_create_state(state_path, config)
    metadata = _discover_and_inspect_shards(
        config=config,
        state=state,
        state_path=state_path,
        checkpoint_hook=checkpoint_hook,
    )
    fragments, fragment_receipts = _load_fragments(
        config=config,
        output_dir=output_dir,
        metadata=metadata,
    )
    state["fragment_receipts"] = fragment_receipts
    _save_state(state_path, state)
    completed = set(fragments)
    new_row_groups = _scan_pending_row_groups(
        config=config,
        metadata=metadata,
        state=state,
        state_path=state_path,
        completed=completed,
        checkpoint_hook=checkpoint_hook,
    )
    fragments, fragment_receipts = _load_fragments(
        config=config,
        output_dir=output_dir,
        metadata=metadata,
    )
    state["fragment_receipts"] = fragment_receipts
    _save_state(state_path, state)
    entries = _ordered_entries(fragments, metadata)
    duplicates = _duplicate_audit(entries)
    allowlist_results = _allowlist_audit(config, entries)
    expected_row_groups = sum(int(value["num_row_groups"]) for value in metadata)
    expected_rows = sum(int(value["num_rows"]) for value in metadata)
    mirror_rows_scanned = sum(
        int(value["scanned_rows"]) for value in fragments.values()
    )
    complete = (
        len(fragments) == expected_row_groups
        and mirror_rows_scanned == expected_rows
    )
    unique_ok = (
        not config.require_unique_video_ids
        or int(duplicates["video_ids_with_multiple_locations"]) == 0
    )
    integrity_pass = complete and unique_ok

    partial_index = output_dir / "audioset_availability_index.partial.jsonl"
    partial_shards = output_dir / "audioset_availability_shards.partial.jsonl"
    atomic_jsonl(partial_index, entries)
    atomic_jsonl(partial_shards, _shard_output(metadata))
    index_path = partial_index
    shard_path = partial_shards
    if complete and unique_ok:
        index_path = output_dir / "audioset_availability_index.jsonl"
        shard_path = output_dir / "audioset_availability_shards.jsonl"
        atomic_jsonl(index_path, entries)
        atomic_jsonl(shard_path, _shard_output(metadata))

    if complete and not unique_ok:
        status = "integrity_failed_duplicate_video_ids"
    elif complete:
        status = "complete"
    else:
        status = "incomplete_max_new_row_groups"
    receipt = _write_receipt(
        config=config,
        metadata=metadata,
        entries=entries,
        fragment_count=len(fragments),
        mirror_rows_scanned=mirror_rows_scanned,
        new_row_groups=new_row_groups,
        duplicate_audit=duplicates,
        allowlist_audit=allowlist_results,
        complete=complete,
        integrity_pass=integrity_pass,
        status=status,
        index_path=index_path,
        shard_path=shard_path,
    )
    if complete and not unique_ok:
        raise AvailabilityIndexError(
            f"mirror contains {duplicates['video_ids_with_multiple_locations']} "
            "video IDs at multiple locations; rerun with duplicate allowance only "
            "if the one-to-many inventory is intentional"
        )
    return receipt


def run_availability_index(
    config: AvailabilityIndexConfig,
    *,
    checkpoint_hook: CheckpointHook | None = None,
) -> dict[str, Any]:
    """Build/resume an index while holding an exclusive output lock."""

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".availability_index.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AvailabilityIndexError(
                f"another availability indexer holds the output lock: {lock_path}"
            ) from error
        try:
            return _run_availability_index_locked(
                config,
                checkpoint_hook=checkpoint_hook,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
