"""Fail-closed, row-level AudioSet transport via the HF Dataset Viewer.

This module is deliberately separate from :mod:`audioset_plan_materializer`.
The existing Parquet row-group transport remains the default and is not
silently replaced or used as a fallback here.

The Dataset Viewer can return a signed URL for one audio cell through
``/rows``.  A row number is only safe when it is bound to the exact Parquet
export order.  The binding implemented below therefore requires all of:

* the Dataset Viewer ``/parquet`` response revision equals the pinned source
  revision;
* its file order is retained verbatim;
* every export file is content-identical to exactly one source shard, using
  size, LFS SHA-256, and Xet hash from the pinned ``refs/convert/parquet``
  tree;
* source row-group counts are internally complete;
* the returned row index, video ID, labels, and cached-asset URL all match the
  pre-indexed location.

Any mismatch raises :class:`DatasetViewerTransportError`.  In particular,
there is no fallback to another mirror or to the row-group backend.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence
from urllib.parse import quote, unquote, urlsplit

import numpy as np
import requests
import soundfile as sf


BINDING_FORMAT = "qces_audioset_dataset_viewer_split_binding_v1"
LOCATION_FORMAT = "qces_audioset_dataset_viewer_row_location_v1"
FETCH_FORMAT = "qces_audioset_dataset_viewer_fetch_v1"

DEFAULT_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DEFAULT_PARQUET_ENDPOINT = "https://datasets-server.huggingface.co/parquet"
DEFAULT_HUB_ENDPOINT = "https://huggingface.co"


class DatasetViewerTransportError(RuntimeError):
    """Raised when row-level transport cannot prove exact provenance."""


@dataclass(frozen=True)
class ViewerRetryPolicy:
    max_attempts: int = 5
    initial_delay_seconds: float = 0.5
    maximum_delay_seconds: float = 8.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.initial_delay_seconds < 0 or self.maximum_delay_seconds < 0:
            raise ValueError("retry delays must be non-negative")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _clean_hash(value: Any) -> str:
    output = str(value or "").strip().strip('"').lower()
    if output.startswith("sha256:"):
        output = output[7:]
    return output


def _require_sha256(value: Any, *, field: str) -> str:
    output = _clean_hash(value)
    if len(output) != 64 or any(char not in "0123456789abcdef" for char in output):
        raise DatasetViewerTransportError(f"invalid {field}: {value!r}")
    return output


def _canonical_strings(
    values: Any,
    *,
    field: str,
    require_nonempty: bool,
) -> list[str]:
    output = [str(value).strip() for value in values or []]
    if any(not value for value in output):
        raise DatasetViewerTransportError(f"{field} contains an empty value")
    if len(set(output)) != len(output):
        raise DatasetViewerTransportError(f"{field} contains duplicates")
    if require_nonempty and not output:
        raise DatasetViewerTransportError(f"{field} is required")
    return sorted(output)


def _header(response: Any, key: str) -> str:
    headers = getattr(response, "headers", {}) or {}
    for candidate, value in headers.items():
        if str(candidate).lower() == key.lower():
            return str(value)
    return ""


def _response_content(response: Any) -> bytes:
    content = getattr(response, "content", b"")
    if isinstance(content, str):
        return content.encode("utf-8")
    return bytes(content)


def _response_json(response: Any) -> Any:
    try:
        return response.json()
    except Exception as error:  # pragma: no cover - requests-specific detail
        raise DatasetViewerTransportError("HTTP response is not valid JSON") from error


def _request_with_retry(
    session: Any,
    method: str,
    url: str,
    *,
    retry: ViewerRetryPolicy,
    timeout_seconds: float,
    sleep: Callable[[float], None],
    **kwargs: Any,
) -> tuple[Any, dict[str, int]]:
    total_payload_bytes = 0
    attempts = 0
    last_status = 0
    last_error = ""
    retryable = {408, 425, 429, 500, 502, 503, 504}
    for attempt in range(retry.max_attempts):
        attempts += 1
        try:
            response = session.request(
                method,
                url,
                timeout=timeout_seconds,
                **kwargs,
            )
            total_payload_bytes += len(_response_content(response))
            last_status = int(getattr(response, "status_code", 0) or 0)
            if 200 <= last_status < 300:
                return response, {
                    "attempts": attempts,
                    "response_payload_bytes": total_payload_bytes,
                }
            last_error = _response_content(response)[:512].decode(
                "utf-8", errors="replace"
            )
            if last_status not in retryable:
                break
        except requests.RequestException as error:
            last_error = str(error)
        if attempt + 1 < retry.max_attempts:
            delay = min(
                retry.maximum_delay_seconds,
                retry.initial_delay_seconds * (2**attempt),
            )
            sleep(delay)
    raise DatasetViewerTransportError(
        f"{method} {url} failed after {attempts} attempts "
        f"(last_status={last_status}): {last_error}"
    )


def _parquet_tree_path(url: str) -> str:
    path = unquote(urlsplit(url).path)
    marker = "/resolve/refs/convert/parquet/"
    if marker not in path:
        raise DatasetViewerTransportError(
            f"Dataset Viewer Parquet URL is not on refs/convert/parquet: {url}"
        )
    relative = path.split(marker, 1)[1].strip("/")
    if not relative or ".." in relative.split("/"):
        raise DatasetViewerTransportError(f"unsafe Parquet export path: {relative!r}")
    return relative


def _source_identity(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        int(row.get("size_bytes") or row.get("file_size_bytes") or 0),
        _require_sha256(row.get("lfs_sha256"), field="source lfs_sha256"),
        _require_sha256(row.get("xet_hash"), field="source xet_hash"),
    )


def _tree_identity(row: Mapping[str, Any]) -> tuple[int, str, str]:
    lfs = row.get("lfs") or {}
    return (
        int(row.get("size") or lfs.get("size") or 0),
        _require_sha256(lfs.get("oid"), field="convert-tree lfs oid"),
        _require_sha256(row.get("xetHash"), field="convert-tree xetHash"),
    )


def _validated_source_shard(
    row: Mapping[str, Any],
    *,
    dataset: str,
    source_revision: str,
    hf_split: str,
) -> dict[str, Any]:
    if str(row.get("hf_dataset")) != dataset:
        raise DatasetViewerTransportError("source shard dataset mismatch")
    if str(row.get("hf_revision")) != source_revision:
        raise DatasetViewerTransportError("source shard revision mismatch")
    if str(row.get("hf_split")) != hf_split:
        raise DatasetViewerTransportError("source shard split mismatch")
    url = str(row.get("parquet_url") or "")
    if not url:
        raise DatasetViewerTransportError("source shard has no parquet_url")
    row_groups = list(row.get("row_groups") or [])
    if not row_groups:
        raise DatasetViewerTransportError(f"source shard has no row groups: {url}")
    group_rows: list[int] = []
    for expected_index, group in enumerate(row_groups):
        if int(group.get("index", -1)) != expected_index:
            raise DatasetViewerTransportError(
                f"non-contiguous row-group indexes in {url}"
            )
        count = int(group.get("num_rows") or 0)
        if count <= 0:
            raise DatasetViewerTransportError(f"invalid row-group size in {url}")
        group_rows.append(count)
    num_rows = int(row.get("num_rows") or 0)
    if sum(group_rows) != num_rows:
        raise DatasetViewerTransportError(
            f"row-group counts do not sum to num_rows in {url}"
        )
    size, lfs_sha256, xet_hash = _source_identity(row)
    if size <= 0:
        raise DatasetViewerTransportError(f"invalid source shard size: {url}")
    return {
        "source_route": str(row.get("source_route") or ""),
        "source_shard_id": str(row.get("shard_id") or ""),
        "source_parquet_url": url,
        "source_size_bytes": size,
        "source_lfs_sha256": lfs_sha256,
        "source_xet_hash": xet_hash,
        "source_num_rows": num_rows,
        "source_row_group_rows": group_rows,
    }


def build_split_binding(
    *,
    dataset: str,
    source_revision: str,
    config: str,
    viewer_split: str,
    hf_split: str,
    source_shards: Sequence[Mapping[str, Any]],
    parquet_document: Mapping[str, Any],
    parquet_response_revision: str,
    parquet_response_sha256: str,
    convert_commit: str,
    convert_tree_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Bind source shards to the official Dataset Viewer file order.

    ``source_shards`` must contain every physical shard for this route/split,
    not only shards that contain selected rows.  This is necessary to derive
    globally correct Dataset Viewer row offsets.
    """

    if not dataset or not source_revision or not config or not viewer_split:
        raise DatasetViewerTransportError("empty split binding identity")
    if parquet_response_revision != source_revision:
        raise DatasetViewerTransportError(
            "Dataset Viewer /parquet x-revision does not match pinned revision"
        )
    _require_sha256(parquet_response_sha256, field="parquet response SHA-256")
    if parquet_document.get("partial") is not False:
        raise DatasetViewerTransportError("Dataset Viewer Parquet export is partial")
    if parquet_document.get("pending"):
        raise DatasetViewerTransportError("Dataset Viewer Parquet export is pending")
    if parquet_document.get("failed"):
        raise DatasetViewerTransportError("Dataset Viewer Parquet export has failures")

    official_files = [
        dict(row)
        for row in parquet_document.get("parquet_files", [])
        if str(row.get("dataset")) == dataset
        and str(row.get("config")) == config
        and str(row.get("split")) == viewer_split
    ]
    if not official_files:
        raise DatasetViewerTransportError(
            f"no /parquet files for {dataset}/{config}/{viewer_split}"
        )
    official_paths = [_parquet_tree_path(str(row.get("url") or "")) for row in official_files]
    if len(set(official_paths)) != len(official_paths):
        raise DatasetViewerTransportError("duplicate files in /parquet split order")

    tree_by_path: dict[str, Mapping[str, Any]] = {}
    for row in convert_tree_entries:
        if str(row.get("type")) != "file":
            continue
        path = str(row.get("path") or "").strip("/")
        if path in tree_by_path:
            raise DatasetViewerTransportError(f"duplicate convert-tree path: {path}")
        tree_by_path[path] = row

    source_by_identity: dict[tuple[int, str, str], dict[str, Any]] = {}
    source_by_url: set[str] = set()
    source_routes: set[str] = set()
    for raw in source_shards:
        source = _validated_source_shard(
            raw,
            dataset=dataset,
            source_revision=source_revision,
            hf_split=hf_split,
        )
        identity = (
            int(source["source_size_bytes"]),
            str(source["source_lfs_sha256"]),
            str(source["source_xet_hash"]),
        )
        if identity in source_by_identity:
            raise DatasetViewerTransportError("duplicate source shard content identity")
        if source["source_parquet_url"] in source_by_url:
            raise DatasetViewerTransportError("duplicate source shard URL")
        source_by_identity[identity] = source
        source_by_url.add(str(source["source_parquet_url"]))
        source_routes.add(str(source["source_route"]))
    if len(source_routes) != 1 or "" in source_routes:
        raise DatasetViewerTransportError("binding must contain one non-empty source route")
    if len(source_by_identity) != len(official_files):
        raise DatasetViewerTransportError(
            "source shard count does not match official /parquet file count"
        )

    files: list[dict[str, Any]] = []
    matched_source_urls: set[str] = set()
    global_start = 0
    for ordinal, (official, path) in enumerate(zip(official_files, official_paths)):
        tree = tree_by_path.get(path)
        if tree is None:
            raise DatasetViewerTransportError(
                f"official /parquet file absent from pinned convert tree: {path}"
            )
        identity = _tree_identity(tree)
        if int(official.get("size") or 0) != identity[0]:
            raise DatasetViewerTransportError(
                f"/parquet size disagrees with convert tree for {path}"
            )
        source = source_by_identity.get(identity)
        if source is None:
            raise DatasetViewerTransportError(
                f"convert export is not content-identical to a source shard: {path}"
            )
        source_url = str(source["source_parquet_url"])
        if source_url in matched_source_urls:
            raise DatasetViewerTransportError("one source shard matched multiple exports")
        matched_source_urls.add(source_url)
        file_row = {
            "viewer_file_ordinal": ordinal,
            "viewer_file_url": str(official["url"]),
            "viewer_file_path": path,
            "viewer_file_size_bytes": identity[0],
            "viewer_file_lfs_sha256": identity[1],
            "viewer_file_xet_hash": identity[2],
            "global_row_start": global_start,
            **source,
        }
        files.append(file_row)
        global_start += int(source["source_num_rows"])
    if matched_source_urls != source_by_url:
        raise DatasetViewerTransportError("not every source shard matched exactly once")

    output: dict[str, Any] = {
        "format": BINDING_FORMAT,
        "dataset": dataset,
        "source_revision": source_revision,
        "source_route": next(iter(source_routes)),
        "config": config,
        "viewer_split": viewer_split,
        "hf_split": hf_split,
        "parquet_response_sha256": parquet_response_sha256,
        "convert_commit": str(convert_commit),
        "file_count": len(files),
        "total_rows": global_start,
        "files": files,
    }
    output["binding_sha256"] = _sha256(_canonical_bytes(output))
    return output


def resolve_row_location(
    binding: Mapping[str, Any],
    availability_entry: Mapping[str, Any],
) -> dict[str, Any]:
    if str(binding.get("format")) != BINDING_FORMAT:
        raise DatasetViewerTransportError("invalid Dataset Viewer binding format")
    claimed_hash = _require_sha256(
        binding.get("binding_sha256"), field="binding_sha256"
    )
    unsigned = dict(binding)
    unsigned.pop("binding_sha256", None)
    if _sha256(_canonical_bytes(unsigned)) != claimed_hash:
        raise DatasetViewerTransportError("Dataset Viewer binding hash mismatch")
    if str(availability_entry.get("hf_dataset")) != str(binding.get("dataset")):
        raise DatasetViewerTransportError("availability dataset mismatch")
    if str(availability_entry.get("hf_revision")) != str(
        binding.get("source_revision")
    ):
        raise DatasetViewerTransportError("availability revision mismatch")
    if str(availability_entry.get("hf_split")) != str(binding.get("hf_split")):
        raise DatasetViewerTransportError("availability split mismatch")
    if str(availability_entry.get("source_route")) != str(binding.get("source_route")):
        raise DatasetViewerTransportError("availability source route mismatch")

    source_url = str(availability_entry.get("parquet_url") or "")
    matches = [
        row for row in binding.get("files", []) if row.get("source_parquet_url") == source_url
    ]
    if len(matches) != 1:
        raise DatasetViewerTransportError(
            f"availability shard has {len(matches)} binding matches"
        )
    file_row = dict(matches[0])
    provenance = availability_entry.get("shard_provenance") or {}
    expected_identity = (
        int(file_row["source_size_bytes"]),
        str(file_row["source_lfs_sha256"]),
        str(file_row["source_xet_hash"]),
    )
    actual_identity = (
        int(provenance.get("file_size_bytes") or 0),
        _require_sha256(provenance.get("lfs_sha256"), field="availability lfs_sha256"),
        _require_sha256(provenance.get("xet_hash"), field="availability xet_hash"),
    )
    if actual_identity != expected_identity:
        raise DatasetViewerTransportError("availability shard provenance mismatch")

    row_group = int(availability_entry.get("row_group", -1))
    row_index = int(availability_entry.get("row_index", -1))
    group_rows = [int(value) for value in file_row["source_row_group_rows"]]
    if row_group < 0 or row_group >= len(group_rows):
        raise DatasetViewerTransportError("availability row group is out of range")
    if row_index < 0 or row_index >= group_rows[row_group]:
        raise DatasetViewerTransportError("availability row index is out of range")
    global_row = (
        int(file_row["global_row_start"])
        + sum(group_rows[:row_group])
        + row_index
    )
    video_id = str(availability_entry.get("video_id") or "")
    if not video_id:
        raise DatasetViewerTransportError("availability video_id is empty")
    labels = _canonical_strings(
        availability_entry.get("labels"),
        field="availability labels",
        require_nonempty=True,
    )
    output = {
        "format": LOCATION_FORMAT,
        "binding_sha256": claimed_hash,
        "dataset": str(binding["dataset"]),
        "source_revision": str(binding["source_revision"]),
        "source_route": str(binding["source_route"]),
        "config": str(binding["config"]),
        "viewer_split": str(binding["viewer_split"]),
        "hf_split": str(binding["hf_split"]),
        "global_row": global_row,
        "video_id": video_id,
        "labels": labels,
        "human_labels": _canonical_strings(
            availability_entry.get("human_labels"),
            field="availability human_labels",
            require_nonempty=False,
        ),
        "row_group": row_group,
        "row_index": row_index,
        "source_parquet_url": source_url,
        "source_size_bytes": expected_identity[0],
        "source_lfs_sha256": expected_identity[1],
        "source_xet_hash": expected_identity[2],
        "viewer_file_ordinal": int(file_row["viewer_file_ordinal"]),
        "viewer_file_url": str(file_row["viewer_file_url"]),
        "convert_commit": str(binding["convert_commit"]),
    }
    output["location_sha256"] = _sha256(_canonical_bytes(output))
    return output


def _extract_asset_url(audio_cell: Any) -> str:
    if isinstance(audio_cell, list):
        if len(audio_cell) != 1:
            raise DatasetViewerTransportError(
                f"expected exactly one audio asset, got {len(audio_cell)}"
            )
        audio_cell = audio_cell[0]
    if not isinstance(audio_cell, Mapping):
        raise DatasetViewerTransportError(
            f"unexpected Dataset Viewer audio cell: {type(audio_cell)}"
        )
    source = str(audio_cell.get("src") or "")
    if not source:
        raise DatasetViewerTransportError("Dataset Viewer audio cell has no src")
    return source


def _validate_asset_url(source: str, location: Mapping[str, Any]) -> str:
    parsed = urlsplit(source)
    if parsed.scheme != "https" or parsed.netloc != "datasets-server.huggingface.co":
        raise DatasetViewerTransportError("audio asset is not hosted by datasets-server")
    path = unquote(parsed.path)
    expected = (
        f"/cached-assets/{location['dataset']}/--/{location['source_revision']}/--/"
        f"{location['config']}/{location['viewer_split']}/{int(location['global_row'])}/audio/"
    )
    if not path.startswith(expected):
        raise DatasetViewerTransportError(
            f"cached asset path does not bind the requested row: {path}"
        )
    query_names = {part.split("=", 1)[0] for part in parsed.query.split("&") if part}
    if not {"Expires", "Signature", "Key-Pair-Id"}.issubset(query_names):
        raise DatasetViewerTransportError("cached asset URL is not signed")
    return path


def _decoded_audio_receipt(data: bytes) -> dict[str, Any]:
    try:
        with sf.SoundFile(io.BytesIO(data)) as handle:
            sample_rate = int(handle.samplerate)
            frames = int(handle.frames)
            channels = int(handle.channels)
            audio_format = str(handle.format)
            subtype = str(handle.subtype)
            pcm = handle.read(dtype="float32", always_2d=True)
    except Exception as error:
        raise DatasetViewerTransportError(
            f"soundfile rejected Dataset Viewer asset: {error}"
        ) from error
    if sample_rate <= 0 or frames <= 0 or channels <= 0 or pcm.shape != (frames, channels):
        raise DatasetViewerTransportError("invalid decoded Dataset Viewer audio")
    pcm_le = np.asarray(pcm, dtype="<f4", order="C")
    pcm_hash_payload = struct.pack("<QQQ", sample_rate, frames, channels) + pcm_le.tobytes()
    return {
        "encoded_audio_sha256": _sha256(data),
        "encoded_audio_bytes": len(data),
        "decoded_pcm_f32le_sha256": _sha256(pcm_hash_payload),
        "sample_rate": sample_rate,
        "frames": frames,
        "channels": channels,
        "duration_seconds": frames / sample_rate,
        "audio_format": audio_format,
        "audio_subtype": subtype,
    }


def fetch_row_audio(
    location: Mapping[str, Any],
    *,
    session: Any | None = None,
    token: str | None = None,
    rows_endpoint: str = DEFAULT_ROWS_ENDPOINT,
    retry: ViewerRetryPolicy = ViewerRetryPolicy(),
    timeout_seconds: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[bytes, dict[str, Any]]:
    """Fetch and validate one exact audio cell.

    Returns encoded audio bytes and a transfer/provenance receipt.  HTTP header
    and TLS overhead are not observable through ``requests`` and are therefore
    explicitly excluded from the byte counter.
    """

    if str(location.get("format")) != LOCATION_FORMAT:
        raise DatasetViewerTransportError("invalid row location format")
    claimed_hash = _require_sha256(
        location.get("location_sha256"), field="location_sha256"
    )
    unsigned = dict(location)
    unsigned.pop("location_sha256", None)
    if _sha256(_canonical_bytes(unsigned)) != claimed_hash:
        raise DatasetViewerTransportError("row location hash mismatch")

    client = session or requests.Session()
    headers: MutableMapping[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {
        "dataset": str(location["dataset"]),
        "config": str(location["config"]),
        "split": str(location["viewer_split"]),
        "offset": int(location["global_row"]),
        "length": 1,
    }
    rows_response, rows_transfer = _request_with_retry(
        client,
        "GET",
        rows_endpoint,
        params=params,
        headers=dict(headers),
        retry=retry,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
    )
    response_revision = _header(rows_response, "x-revision")
    if response_revision != str(location["source_revision"]):
        raise DatasetViewerTransportError(
            "Dataset Viewer /rows x-revision does not match pinned revision"
        )
    document = _response_json(rows_response)
    if document.get("partial") is not False:
        raise DatasetViewerTransportError("Dataset Viewer /rows response is partial")
    rows = list(document.get("rows") or [])
    if len(rows) != 1:
        raise DatasetViewerTransportError(
            f"Dataset Viewer returned {len(rows)} rows instead of one"
        )
    wrapper = rows[0]
    if int(wrapper.get("row_idx", -1)) != int(location["global_row"]):
        raise DatasetViewerTransportError("Dataset Viewer row_idx mismatch")
    row = wrapper.get("row") or {}
    if str(row.get("video_id") or "") != str(location["video_id"]):
        raise DatasetViewerTransportError("Dataset Viewer video_id mismatch")
    actual_labels = _canonical_strings(
        row.get("labels"),
        field="Dataset Viewer labels",
        require_nonempty=True,
    )
    if actual_labels != _canonical_strings(
        location["labels"], field="location labels", require_nonempty=True
    ):
        raise DatasetViewerTransportError("Dataset Viewer labels mismatch")
    expected_human = _canonical_strings(
        location.get("human_labels"),
        field="location human_labels",
        require_nonempty=False,
    )
    if expected_human:
        actual_human = _canonical_strings(
            row.get("human_labels"),
            field="Dataset Viewer human_labels",
            require_nonempty=False,
        )
        if actual_human != expected_human:
            raise DatasetViewerTransportError("Dataset Viewer human_labels mismatch")
    if wrapper.get("truncated_cells"):
        raise DatasetViewerTransportError("Dataset Viewer row contains truncated cells")

    asset_source = _extract_asset_url(row.get("audio"))
    asset_path = _validate_asset_url(asset_source, location)
    asset_response, asset_transfer = _request_with_retry(
        client,
        "GET",
        asset_source,
        headers=dict(headers),
        retry=retry,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
    )
    audio_bytes = _response_content(asset_response)
    declared_length = _header(asset_response, "content-length")
    if declared_length and int(declared_length) != len(audio_bytes):
        raise DatasetViewerTransportError("cached asset Content-Length mismatch")
    decoded = _decoded_audio_receipt(audio_bytes)
    stable_row = {
        "row_idx": int(wrapper["row_idx"]),
        "video_id": str(row["video_id"]),
        "labels": actual_labels,
        "human_labels": _canonical_strings(
            row.get("human_labels"),
            field="Dataset Viewer human_labels",
            require_nonempty=False,
        ),
        "asset_path": asset_path,
    }
    rows_payload = int(rows_transfer["response_payload_bytes"])
    asset_payload = int(asset_transfer["response_payload_bytes"])
    receipt = {
        "format": FETCH_FORMAT,
        "transport": "hf_dataset_viewer_row",
        "binding_sha256": str(location["binding_sha256"]),
        "location_sha256": claimed_hash,
        "dataset": str(location["dataset"]),
        "source_revision": str(location["source_revision"]),
        "convert_commit": str(location["convert_commit"]),
        "config": str(location["config"]),
        "viewer_split": str(location["viewer_split"]),
        "global_row": int(location["global_row"]),
        "video_id": str(location["video_id"]),
        "labels": actual_labels,
        "asset_url_path_without_signature": asset_path,
        "stable_row_sha256": _sha256(_canonical_bytes(stable_row)),
        "rows_response_revision": response_revision,
        "rows_attempts": int(rows_transfer["attempts"]),
        "asset_attempts": int(asset_transfer["attempts"]),
        "rows_response_payload_bytes": rows_payload,
        "asset_response_payload_bytes": asset_payload,
        "total_response_payload_bytes": rows_payload + asset_payload,
        "transfer_bytes_exclude": [
            "HTTP headers",
            "TLS and protocol overhead",
            "client/network intermediary cache effects",
        ],
        **decoded,
    }
    receipt["receipt_sha256"] = _sha256(_canonical_bytes(receipt))
    return audio_bytes, receipt


def fetch_parquet_document(
    *,
    dataset: str,
    source_revision: str,
    session: Any | None = None,
    token: str | None = None,
    endpoint: str = DEFAULT_PARQUET_ENDPOINT,
    retry: ViewerRetryPolicy = ViewerRetryPolicy(),
    timeout_seconds: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch the official file-order document and pin its response revision."""

    client = session or requests.Session()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    response, transfer = _request_with_retry(
        client,
        "GET",
        endpoint,
        params={"dataset": dataset},
        headers=headers,
        retry=retry,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
    )
    revision = _header(response, "x-revision")
    if revision != source_revision:
        raise DatasetViewerTransportError(
            "Dataset Viewer /parquet x-revision does not match pinned revision"
        )
    payload = _response_content(response)
    document = _response_json(response)
    if not isinstance(document, Mapping):
        raise DatasetViewerTransportError("invalid /parquet response document")
    return dict(document), {
        "response_revision": revision,
        "response_sha256": _sha256(payload),
        "response_payload_bytes": int(transfer["response_payload_bytes"]),
        "attempts": int(transfer["attempts"]),
    }


def _next_link(value: str) -> str:
    for part in value.split(","):
        bits = [item.strip() for item in part.split(";")]
        if len(bits) >= 2 and any(item == 'rel="next"' for item in bits[1:]):
            target = bits[0]
            if target.startswith("<") and target.endswith(">"):
                return target[1:-1]
    return ""


def fetch_convert_tree(
    *,
    dataset: str,
    config: str,
    viewer_split: str,
    session: Any | None = None,
    token: str | None = None,
    hub_endpoint: str = DEFAULT_HUB_ENDPOINT,
    retry: ViewerRetryPolicy = ViewerRetryPolicy(),
    timeout_seconds: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Fetch a content-pinned ``refs/convert/parquet`` tree for one split."""

    client = session or requests.Session()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    dataset_path = "/".join(quote(part, safe="") for part in dataset.split("/"))
    info_url = (
        f"{hub_endpoint.rstrip('/')}/api/datasets/{dataset_path}/revision/"
        "refs%2Fconvert%2Fparquet"
    )
    info_response, info_transfer = _request_with_retry(
        client,
        "GET",
        info_url,
        headers=headers,
        retry=retry,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
    )
    info = _response_json(info_response)
    convert_commit = str(info.get("sha") or "")
    if len(convert_commit) != 40:
        raise DatasetViewerTransportError("invalid convert/parquet commit")

    relative_path = quote(f"{config}/{viewer_split}", safe="")
    next_url = (
        f"{hub_endpoint.rstrip('/')}/api/datasets/{dataset_path}/tree/"
        f"{convert_commit}/{relative_path}"
    )
    params: Mapping[str, Any] | None = {
        "recursive": "true",
        "expand": "true",
        # The Hub tree API currently rejects values above 100.  Pagination is
        # content-pinned to ``convert_commit`` and followed through Link.
        "limit": 100,
    }
    entries: list[dict[str, Any]] = []
    payload_bytes = int(info_transfer["response_payload_bytes"])
    attempts = int(info_transfer["attempts"])
    while next_url:
        response, transfer = _request_with_retry(
            client,
            "GET",
            next_url,
            params=params,
            headers=headers,
            retry=retry,
            timeout_seconds=timeout_seconds,
            sleep=sleep,
        )
        params = None
        document = _response_json(response)
        if not isinstance(document, list):
            raise DatasetViewerTransportError("invalid convert-tree response")
        entries.extend(dict(row) for row in document)
        payload_bytes += int(transfer["response_payload_bytes"])
        attempts += int(transfer["attempts"])
        next_url = _next_link(_header(response, "link"))
    return convert_commit, entries, {
        "response_payload_bytes": payload_bytes,
        "attempts": attempts,
        "tree_entry_count": len(entries),
    }


def build_split_binding_online(
    *,
    dataset: str,
    source_revision: str,
    config: str,
    viewer_split: str,
    hf_split: str,
    source_shards: Sequence[Mapping[str, Any]],
    session: Any | None = None,
    token: str | None = None,
    parquet_endpoint: str = DEFAULT_PARQUET_ENDPOINT,
    hub_endpoint: str = DEFAULT_HUB_ENDPOINT,
    retry: ViewerRetryPolicy = ViewerRetryPolicy(),
    timeout_seconds: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a complete split binding from pinned online metadata.

    This operation downloads JSON metadata only.  It does not request a row or
    an audio asset.  The same client session is shared so authenticated/private
    datasets and connection pooling behave consistently.
    """

    client = session or requests.Session()
    parquet_document, parquet_receipt = fetch_parquet_document(
        dataset=dataset,
        source_revision=source_revision,
        session=client,
        token=token,
        endpoint=parquet_endpoint,
        retry=retry,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
    )
    convert_commit, tree_entries, tree_receipt = fetch_convert_tree(
        dataset=dataset,
        config=config,
        viewer_split=viewer_split,
        session=client,
        token=token,
        hub_endpoint=hub_endpoint,
        retry=retry,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
    )
    binding = build_split_binding(
        dataset=dataset,
        source_revision=source_revision,
        config=config,
        viewer_split=viewer_split,
        hf_split=hf_split,
        source_shards=source_shards,
        parquet_document=parquet_document,
        parquet_response_revision=str(parquet_receipt["response_revision"]),
        parquet_response_sha256=str(parquet_receipt["response_sha256"]),
        convert_commit=convert_commit,
        convert_tree_entries=tree_entries,
    )
    receipt = {
        "binding_sha256": binding["binding_sha256"],
        "dataset": dataset,
        "source_revision": source_revision,
        "config": config,
        "viewer_split": viewer_split,
        "hf_split": hf_split,
        "parquet_metadata": parquet_receipt,
        "convert_tree_metadata": tree_receipt,
        "metadata_response_payload_bytes": (
            int(parquet_receipt["response_payload_bytes"])
            + int(tree_receipt["response_payload_bytes"])
        ),
        "audio_payload_bytes": 0,
    }
    receipt["receipt_sha256"] = _sha256(_canonical_bytes(receipt))
    return binding, receipt
