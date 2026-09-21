"""Resumable full-clip adapter for the fail-closed Dataset Viewer backend.

The emitted ``source_audio_manifest.jsonl`` is intentionally compatible with
``audioset_plan_materializer --existing-manifest`` in requested-crop mode.
Thus the row-level network transport can be smoke-tested independently while
the existing Parquet materializer and all of its default behaviour remain
untouched.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import soundfile as sf

from mixi_understanding.qces.audioset_dataset_viewer_backend import (
    BINDING_FORMAT,
    FETCH_FORMAT,
    DatasetViewerTransportError,
    ViewerRetryPolicy,
    fetch_row_audio,
    resolve_row_location,
)


MATERIALIZER_FORMAT = "qces_audioset_dataset_viewer_materializer_v1"
TRANSACTION_FORMAT = "qces_audioset_dataset_viewer_transaction_v1"
MANIFEST_FORMAT = "qces_audioset_dataset_viewer_source_manifest_v1"


@dataclass(frozen=True)
class ViewerMaterializationConfig:
    binding_path: Path
    availability_paths: tuple[Path, ...]
    output_dir: Path
    max_new_rows: int = 0
    minimum_free_disk_bytes: int = 10 * (1 << 30)
    token: str | None = None
    retry: ViewerRetryPolicy = field(default_factory=ViewerRetryPolicy)
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if not self.availability_paths:
            raise ValueError("at least one availability path is required")
        if self.max_new_rows < 0:
            raise ValueError("max_new_rows must be >= 0")
        if self.minimum_free_disk_bytes < 0:
            raise ValueError("minimum_free_disk_bytes must be >= 0")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise DatasetViewerTransportError(f"invalid JSON file {path}: {error}") from error


def _availability_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DatasetViewerTransportError(f"availability input does not exist: {path}")
    rows: list[Any]
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as error:
                raise DatasetViewerTransportError(
                    f"invalid JSONL at {path}:{line_number}"
                ) from error
    else:
        document = _load_json(path)
        if isinstance(document, Mapping) and isinstance(document.get("entries"), list):
            rows = list(document["entries"])
        elif isinstance(document, list):
            rows = list(document)
        elif isinstance(document, Mapping):
            rows = [document]
        else:
            raise DatasetViewerTransportError(f"unsupported availability JSON: {path}")
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise DatasetViewerTransportError(f"non-object availability row in {path}")
        nested = row.get("availability_location")
        output.append(dict(nested if isinstance(nested, Mapping) else row))
    return output


def load_availability_entries(paths: Sequence[Path]) -> list[dict[str, Any]]:
    by_video: dict[str, dict[str, Any]] = {}
    canonical: dict[str, bytes] = {}
    for path in paths:
        for row in _availability_rows(path.resolve()):
            video_id = str(row.get("video_id") or "")
            if not video_id:
                raise DatasetViewerTransportError("availability row has empty video_id")
            encoded = _canonical_bytes(row)
            if video_id in by_video and canonical[video_id] != encoded:
                raise DatasetViewerTransportError(
                    f"availability inputs disagree for video_id={video_id}"
                )
            by_video[video_id] = row
            canonical[video_id] = encoded
    return [by_video[key] for key in sorted(by_video)]


def _transaction_key(video_id: str, location_sha256: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise DatasetViewerTransportError(f"unsafe AudioSet video_id: {video_id!r}")
    return f"{video_id}-{location_sha256[:16]}"


def _audio_extension(fetch_receipt: Mapping[str, Any]) -> str:
    value = str(fetch_receipt.get("audio_format") or "").upper()
    return {"FLAC": ".flac", "WAV": ".wav", "OGG": ".ogg"}.get(value, ".audio")


def _validate_transaction(path: Path, *, location_sha256: str) -> dict[str, Any]:
    receipt_path = path / "transaction.json"
    if not receipt_path.is_file():
        raise DatasetViewerTransportError(f"transaction receipt missing: {path}")
    transaction = _load_json(receipt_path)
    if str(transaction.get("format")) != TRANSACTION_FORMAT:
        raise DatasetViewerTransportError(f"invalid transaction format: {path}")
    claimed = str(transaction.get("transaction_sha256") or "")
    unsigned = dict(transaction)
    unsigned.pop("transaction_sha256", None)
    if _sha256(_canonical_bytes(unsigned)) != claimed:
        raise DatasetViewerTransportError(f"transaction hash mismatch: {path}")
    if str(transaction.get("location_sha256")) != location_sha256:
        raise DatasetViewerTransportError(f"transaction location mismatch: {path}")
    audio_path = path / str(transaction.get("audio_filename") or "")
    if not audio_path.is_file():
        raise DatasetViewerTransportError(f"transaction audio missing: {path}")
    if _sha256_file(audio_path) != str(transaction.get("audio_sha256")):
        raise DatasetViewerTransportError(f"transaction audio hash mismatch: {path}")
    try:
        info = sf.info(audio_path)
    except Exception as error:
        raise DatasetViewerTransportError(f"transaction audio is not decodable: {path}") from error
    fetch = transaction.get("fetch_receipt") or {}
    if int(info.samplerate) != int(fetch.get("sample_rate") or 0):
        raise DatasetViewerTransportError(f"transaction sample rate mismatch: {path}")
    if int(info.frames) != int(fetch.get("frames") or 0):
        raise DatasetViewerTransportError(f"transaction frame count mismatch: {path}")
    return transaction


def _commit_transaction(
    output_dir: Path,
    *,
    location: Mapping[str, Any],
    audio_bytes: bytes,
    fetch_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    key = _transaction_key(str(location["video_id"]), str(location["location_sha256"]))
    transactions_dir = output_dir / "transactions"
    transactions_dir.mkdir(parents=True, exist_ok=True)
    destination = transactions_dir / key
    if destination.exists():
        return _validate_transaction(
            destination, location_sha256=str(location["location_sha256"])
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=transactions_dir))
    try:
        audio_filename = f"audio{_audio_extension(fetch_receipt)}"
        audio_path = temporary / audio_filename
        with audio_path.open("wb") as handle:
            handle.write(audio_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        audio_hash = _sha256_file(audio_path)
        if audio_hash != str(fetch_receipt.get("encoded_audio_sha256")):
            raise DatasetViewerTransportError("fetched audio hash changed before commit")
        transaction: dict[str, Any] = {
            "format": TRANSACTION_FORMAT,
            "video_id": str(location["video_id"]),
            "location_sha256": str(location["location_sha256"]),
            "binding_sha256": str(location["binding_sha256"]),
            "audio_filename": audio_filename,
            "audio_sha256": audio_hash,
            "audio_bytes": len(audio_bytes),
            "fetch_receipt": dict(fetch_receipt),
        }
        transaction["transaction_sha256"] = _sha256(_canonical_bytes(transaction))
        _atomic_json(temporary / "transaction.json", transaction)
        os.replace(temporary, destination)
        return _validate_transaction(
            destination, location_sha256=str(location["location_sha256"])
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _manifest_row(
    *,
    output_dir: Path,
    transaction_path: Path,
    transaction: Mapping[str, Any],
    location: Mapping[str, Any],
) -> dict[str, Any]:
    fetch = transaction["fetch_receipt"]
    audio_path = transaction_path / str(transaction["audio_filename"])
    relative = audio_path.relative_to(output_dir).as_posix()
    return {
        "format": MANIFEST_FORMAT,
        "video_id": str(location["video_id"]),
        "hf_dataset": str(location["dataset"]),
        "hf_revision": str(location["source_revision"]),
        "hf_split": str(location["hf_split"]),
        "protocol_upstream_split": str(location["hf_split"]),
        "source_route": str(location["source_route"]),
        "transport": "hf_dataset_viewer_row",
        # Both names are supported by the existing crop-source adapter.
        "audio_path": relative,
        "mixture_path": relative,
        "audio_sha256": str(transaction["audio_sha256"]),
        "audio_bytes": int(transaction["audio_bytes"]),
        "sample_rate": int(fetch["sample_rate"]),
        "duration_seconds": float(fetch["duration_seconds"]),
        "decoded_pcm_f32le_sha256": str(fetch["decoded_pcm_f32le_sha256"]),
        "binding_sha256": str(location["binding_sha256"]),
        "location_sha256": str(location["location_sha256"]),
        "fetch_receipt_sha256": str(fetch["receipt_sha256"]),
    }


def run_viewer_materialization(
    config: ViewerMaterializationConfig,
    *,
    fetcher: Callable[..., tuple[bytes, dict[str, Any]]] = fetch_row_audio,
) -> dict[str, Any]:
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".dataset_viewer_materializer.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise DatasetViewerTransportError(
                f"another Dataset Viewer materializer holds {lock_path}"
            ) from error
        return _run_locked(config, output_dir=output_dir, fetcher=fetcher)


def _run_locked(
    config: ViewerMaterializationConfig,
    *,
    output_dir: Path,
    fetcher: Callable[..., tuple[bytes, dict[str, Any]]],
) -> dict[str, Any]:
    binding_path = config.binding_path.resolve()
    binding = _load_json(binding_path)
    if str(binding.get("format")) != BINDING_FORMAT:
        raise DatasetViewerTransportError("invalid binding input format")
    entries = load_availability_entries(config.availability_paths)
    locations = [resolve_row_location(binding, row) for row in entries]
    by_video = {str(row["video_id"]): row for row in locations}
    if len(by_video) != len(locations):
        raise DatasetViewerTransportError("resolved locations contain duplicate videos")

    completed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for location in locations:
        video_id = str(location["video_id"])
        key = _transaction_key(video_id, str(location["location_sha256"]))
        transaction_path = output_dir / "transactions" / key
        if transaction_path.exists():
            completed[video_id] = (
                transaction_path,
                _validate_transaction(
                    transaction_path,
                    location_sha256=str(location["location_sha256"]),
                ),
            )

    new_rows = 0
    for location in locations:
        video_id = str(location["video_id"])
        if video_id in completed:
            continue
        if config.max_new_rows and new_rows >= config.max_new_rows:
            break
        free = shutil.disk_usage(output_dir).free
        if free < config.minimum_free_disk_bytes:
            raise DatasetViewerTransportError(
                f"free disk {free} is below safety floor {config.minimum_free_disk_bytes}"
            )
        audio_bytes, fetch_receipt = fetcher(
            location,
            token=config.token,
            retry=config.retry,
            timeout_seconds=config.timeout_seconds,
        )
        if str(fetch_receipt.get("format")) != FETCH_FORMAT:
            raise DatasetViewerTransportError("fetcher returned an invalid receipt")
        transaction = _commit_transaction(
            output_dir,
            location=location,
            audio_bytes=audio_bytes,
            fetch_receipt=fetch_receipt,
        )
        key = _transaction_key(video_id, str(location["location_sha256"]))
        completed[video_id] = (output_dir / "transactions" / key, transaction)
        new_rows += 1

    manifest_rows = [
        _manifest_row(
            output_dir=output_dir,
            transaction_path=completed[video_id][0],
            transaction=completed[video_id][1],
            location=by_video[video_id],
        )
        for video_id in sorted(completed)
        if video_id in by_video
    ]
    manifest_path = output_dir / "source_audio_manifest.jsonl"
    _atomic_jsonl(manifest_path, manifest_rows)
    completed_requested = len(manifest_rows)
    receipt: dict[str, Any] = {
        "format": MATERIALIZER_FORMAT,
        "binding_path": str(binding_path),
        "binding_file_sha256": _sha256_file(binding_path),
        "binding_sha256": str(binding["binding_sha256"]),
        "availability_paths": [str(path.resolve()) for path in config.availability_paths],
        "requested_rows": len(locations),
        "completed_rows": completed_requested,
        "new_rows_this_run": new_rows,
        "remaining_rows": len(locations) - completed_requested,
        "complete": completed_requested == len(locations),
        "max_new_rows": config.max_new_rows,
        "source_audio_manifest": str(manifest_path),
        "source_audio_manifest_sha256": _sha256_file(manifest_path),
        "total_audio_bytes": sum(int(row["audio_bytes"]) for row in manifest_rows),
        "total_response_payload_bytes": sum(
            int(completed[video_id][1]["fetch_receipt"]["total_response_payload_bytes"])
            for video_id in completed
            if video_id in by_video
        ),
        "no_silent_fallback": True,
        "existing_parquet_materializer_modified": False,
    }
    receipt["receipt_sha256"] = _sha256(_canonical_bytes(receipt))
    _atomic_json(output_dir / "materialization_receipt.json", receipt)
    return receipt

