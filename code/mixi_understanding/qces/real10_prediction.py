"""Leakage-safe rendering contract for QCES-Real-10.

This module is deliberately separated from the synthetic QCES evaluators.
It accepts only the strict label-free ``qces_real10_inference_v2`` view and a
frozen AudioSep-CLAP feature cache.  Human answers and temporal annotations are
joined later by a scorer; they are neither accepted nor opened here.

The renderer writes sample-exact IEEE-float evidence and residual WAV files,
one strict prediction row per question, and a receipt binding every model,
backend, cache, input-audio, and runtime-code identity.  Real-test rendering is
fail-closed behind a content-bound method-freeze receipt.
"""

from __future__ import annotations

import hashlib
import io
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import struct
import tempfile
import ctypes
import errno
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, MutableMapping, Sequence

import numpy as np
import soundfile as sf
import torch

from mixi_understanding.data.qces_real10_schema import (
    DURATION_SECONDS,
    INFERENCE_SCHEMA_VERSION,
    NUM_CHANNELS,
    NUM_SAMPLES,
    QCESReal10InferenceRecord,
    SAMPLE_RATE,
    canonical_inference_manifest_fingerprint,
    canonical_json_sha256,
    inference_record_fingerprint,
    parse_inference_manifest,
    parse_inference_record,
    resolve_manifest_mixture_path,
)
from mixi_understanding.qces.composer import ROLE_ANCHOR, ROLE_ANSWER
from mixi_understanding.qces.config import (
    AUDIOSEP_CLAP_FOUNDATION_FEATURES,
    DUAL_ROLE_SEMANTIC_MODE,
    LEGACY_TEMPORAL_ROLE_MODE,
    OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    QCESConfig,
    UNION_SINGLE_SEMANTIC_MODE,
)
from mixi_understanding.qces.model import load_qces_checkpoint
from mixi_understanding.qces.separators import AudioSepConditionedAdapter
from mixi_understanding.qces.tokenization import StableHashTokenizer
from mixi_understanding.scripts.cache_audiosep_clap_features import (
    EFFECTIVE_FRAMES,
    EXPECTED_QUERY_STATE_TENSORS,
    FEATURE_SPACE,
    FINE_DIM,
    JOINT_DIM,
    QUESTION_CACHE_FORMAT,
    RAW_FINE_FRAMES,
    RECEIPT_FORMAT as FOUNDATION_CACHE_RECEIPT_FORMAT,
    REPEAT_RATIO,
    SCENE_CACHE_FORMAT,
    file_identity,
    source_tree_identity,
    validate_question_payload,
    validate_scene_payload,
)


PREDICTION_SCHEMA_VERSION = "qces_real10_prediction_v1"
PREDICTION_MANIFEST_FINGERPRINT_FORMAT = (
    "qces_real10_prediction_manifest_fingerprint_v1"
)
RENDER_RECEIPT_FORMAT = "qces_real10_render_receipt_v1"
RUN_SPEC_FORMAT = "qces_real10_render_run_spec_v1"
METHOD_IDENTITY_FORMAT = "qces_real10_method_identity_v1"
METHOD_FREEZE_RECEIPT_FORMAT = "qces_real10_method_freeze_receipt_v1"
RUNTIME_SOURCE_IDENTITY_FORMAT = "qces_real10_runtime_source_identity_v1"
STEM_AGGREGATE_FORMAT = "qces_real10_stem_aggregate_v1"

PREDICTION_MANIFEST_FILENAME = "prediction_manifest.jsonl"
RENDER_RECEIPT_FILENAME = "render_receipt.json"
CANONICAL_WAV_FORMAT = "WAV"
CANONICAL_WAV_SUBTYPE = "FLOAT"
VALID_CUBLAS_WORKSPACE_CONFIGS = frozenset({":4096:8", ":16:8"})
MAX_RECONSTRUCTION_ABS_ERROR = 1e-6

_SHA256_HEX = frozenset("0123456789abcdef")
_RUNTIME_SOURCE_FILES = (
    "qces/real10_prediction.py",
    "qces/model.py",
    "qces/composer.py",
    "qces/config.py",
    "qces/separators.py",
    "qces/signal.py",
    "qces/tokenization.py",
    "data/qces_real10_schema.py",
    "scripts/cache_audiosep_clap_features.py",
    "scripts/render_qces_real10.py",
)


class Real10RenderError(RuntimeError):
    """Raised when an input, prediction, or provenance gate fails closed."""


@dataclass(frozen=True)
class CanonicalMixture:
    """One fully decoded and content-bound canonical scene waveform."""

    scene_id: str
    manifest_path: str
    path: Path
    samples: np.ndarray
    file_identity: Mapping[str, Any]
    pcm_f32le_sha256: str


@dataclass(frozen=True)
class Real10FoundationFeatureCache:
    """Validated Real-10 frozen features with ID-only routing."""

    question_features: Mapping[str, torch.Tensor]
    scene_features: Mapping[str, torch.Tensor]
    sample_to_scene: Mapping[str, str]
    identity: Mapping[str, Any]


@dataclass(frozen=True)
class PreparedRenderInputs:
    """Validated label-free inputs shared by freeze and render operations."""

    manifest_path: Path
    manifest_identity: Mapping[str, Any]
    records: tuple[QCESReal10InferenceRecord, ...]
    selected_records: tuple[QCESReal10InferenceRecord, ...]
    full_manifest_fingerprint: str
    selected_manifest_fingerprint: str
    mixtures: Mapping[str, CanonicalMixture]
    foundation_cache: Real10FoundationFeatureCache
    foundation_cache_contract: Mapping[str, Any]
    foundation_cache_identity: Mapping[str, Any]
    qces_checkpoint_path: Path
    qces_checkpoint_identity: Mapping[str, Any]
    qces_checkpoint_payload: Mapping[str, Any]
    qces_config: QCESConfig
    audiosep_root: Path
    audiosep_source_identity: Mapping[str, Any]
    audiosep_config_path: Path
    audiosep_config_identity: Mapping[str, Any]
    audiosep_checkpoint_path: Path
    audiosep_checkpoint_identity: Mapping[str, Any]
    runtime_source_identity: Mapping[str, Any]


@dataclass(frozen=True)
class RenderSettings:
    """Frozen inference decisions which affect a prediction."""

    split: str = "real_dev"
    role_threshold: float = 0.5
    no_evidence_threshold: float = 0.5
    seed: int = 2026
    device_type: str = "cuda"

    def validate(self) -> None:
        if self.split not in {"real_dev", "real_test"}:
            raise Real10RenderError("split must be real_dev or real_test")
        for name, value in (
            ("role_threshold", self.role_threshold),
            ("no_evidence_threshold", self.no_evidence_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) < 1.0
            ):
                raise Real10RenderError(f"{name} must lie strictly in (0, 1)")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise Real10RenderError("seed must be an integer")
        if self.device_type not in {"cpu", "cuda"}:
            raise Real10RenderError("device_type must be cpu or cuda")


@dataclass(frozen=True)
class RenderResult:
    output_dir: Path
    receipt: Mapping[str, Any]
    resumed: bool


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and set(value).issubset(_SHA256_HEX)
    )


def _exact_keys(value: Any, expected: set[str], context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Real10RenderError(f"{context} must be an object")
    actual = set(value)
    if actual != expected:
        raise Real10RenderError(
            f"{context} fields mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Real10RenderError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise Real10RenderError(f"{context} must be finite")
    return result


def _probability(value: Any, context: str) -> float:
    result = _finite_number(value, context)
    if not 0.0 <= result <= 1.0:
        raise Real10RenderError(f"{context} must lie in [0, 1]")
    return result


def _positive_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Real10RenderError(f"{context} must be a positive integer")
    return value


def _safe_relative_wav(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise Real10RenderError(f"{context} must be non-empty text")
    if "\\" in value:
        raise Real10RenderError(f"{context} must use POSIX separators")
    parts = value.split("/")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.suffix.casefold() != ".wav"
        or any(part in {"", ".", ".."} or ":" in part for part in parts)
    ):
        raise Real10RenderError(f"{context} must be a safe relative WAV path")
    return value


def _portable_file_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    digest = identity.get("sha256")
    size = identity.get("size_bytes")
    if not _is_sha256(digest) or isinstance(size, bool) or not isinstance(size, int):
        raise Real10RenderError("invalid file identity")
    if size <= 0:
        raise Real10RenderError("file identity has a non-positive size")
    return {"sha256": digest, "size_bytes": size}


def _same_file_content(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return _portable_file_identity(first) == _portable_file_identity(second)


def _pcm_f32le_bytes(samples: np.ndarray) -> bytes:
    array = np.asarray(samples)
    if array.ndim != 1 or array.shape != (NUM_SAMPLES,):
        raise Real10RenderError(
            f"waveform must have shape ({NUM_SAMPLES},), got {array.shape}"
        )
    if array.dtype != np.float32:
        raise Real10RenderError(f"waveform must be float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise Real10RenderError("waveform contains NaN or Inf")
    return np.ascontiguousarray(array, dtype=np.dtype("<f4")).tobytes(order="C")


def pcm_f32le_sha256(samples: np.ndarray) -> str:
    """Hash the canonical little-endian float32 sample sequence."""

    return hashlib.sha256(_pcm_f32le_bytes(samples)).hexdigest()


def _canonical_float_wav_bytes(samples: np.ndarray) -> bytes:
    """Return a deterministic minimal mono IEEE-float RIFF/WAV."""

    pcm = _pcm_f32le_bytes(samples)
    fmt_chunk = b"fmt " + struct.pack(
        "<IHHIIHH",
        16,
        3,
        NUM_CHANNELS,
        SAMPLE_RATE,
        SAMPLE_RATE * 4,
        4,
        32,
    )
    fact_chunk = b"fact" + struct.pack("<II", 4, NUM_SAMPLES)
    data_header = b"data" + struct.pack("<I", len(pcm))
    riff_size = 4 + len(fmt_chunk) + len(fact_chunk) + len(data_header) + len(pcm)
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVE"
        + fmt_chunk
        + fact_chunk
        + data_header
        + pcm
    )


def _write_bytes_fsync(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically install a staged directory without replacing any target."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise Real10RenderError(
            "atomic exclusive output installation requires Linux renameat2"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number,
            "refusing to replace output created by another process",
            str(destination),
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        _fsync_directory(directory)
    _fsync_directory(root)


def _decode_float_wav_exact(path: Path, *, allow_silence: bool) -> np.ndarray:
    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as error:
        raise Real10RenderError(f"cannot inspect WAV {path}: {error}") from error
    if (
        info.format != CANONICAL_WAV_FORMAT
        or info.subtype != CANONICAL_WAV_SUBTYPE
        or info.samplerate != SAMPLE_RATE
        or info.channels != NUM_CHANNELS
        or info.frames != NUM_SAMPLES
    ):
        raise Real10RenderError(
            "WAV must be exact mono IEEE-float 32 kHz / 320000 samples: "
            f"format={info.format}, subtype={info.subtype}, "
            f"sr={info.samplerate}, channels={info.channels}, frames={info.frames}"
        )
    try:
        samples, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as error:
        raise Real10RenderError(f"cannot decode WAV {path}: {error}") from error
    array = np.asarray(samples)
    if sample_rate != SAMPLE_RATE or array.shape != (NUM_SAMPLES,):
        raise Real10RenderError("decoded WAV changed the canonical audio shape")
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise Real10RenderError("decoded WAV is not finite float32")
    if not allow_silence and float(np.max(np.abs(array))) <= 1e-7:
        raise Real10RenderError("canonical mixture is unexpectedly silent")
    return np.ascontiguousarray(array)


def _reject_symlink_components(path: Path, *, root: Path | None = None) -> None:
    if root is None:
        absolute = path.absolute()
        parts = absolute.parts
        current = Path(parts[0])
        remaining = parts[1:]
    else:
        current = root.resolve()
        try:
            remaining = path.absolute().relative_to(current).parts
        except ValueError as error:
            raise Real10RenderError("path escapes its declared root") from error
    for part in remaining:
        current = current / part
        if current.is_symlink():
            raise Real10RenderError(f"input paths must not use symlinks: {current}")


def _decode_bound_mixture_bytes(path: Path, expected_sha256: str) -> np.ndarray:
    """Hash and decode the same immutable byte buffer without audio conversion."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise Real10RenderError(f"cannot read canonical mixture: {path}") from error
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise Real10RenderError("canonical mixture bytes differ from manifest SHA256")
    try:
        info = sf.info(io.BytesIO(payload))
        samples, sample_rate = sf.read(
            io.BytesIO(payload), dtype="float32", always_2d=False
        )
    except (OSError, RuntimeError) as error:
        raise Real10RenderError(f"cannot decode canonical mixture: {path}") from error
    if (
        info.format != CANONICAL_WAV_FORMAT
        or info.subtype != CANONICAL_WAV_SUBTYPE
        or info.samplerate != SAMPLE_RATE
        or info.channels != NUM_CHANNELS
        or info.frames != NUM_SAMPLES
        or sample_rate != SAMPLE_RATE
    ):
        raise Real10RenderError(
            "canonical mixture must be exact mono IEEE-float 32 kHz / "
            "320000 samples; resampling and downmixing are forbidden"
        )
    array = np.asarray(samples)
    if array.shape != (NUM_SAMPLES,) or array.dtype != np.float32:
        raise Real10RenderError("canonical mixture decoder changed shape or dtype")
    if not np.isfinite(array).all():
        raise Real10RenderError("canonical mixture contains NaN or Inf")
    if float(np.max(np.abs(array))) <= 1e-7:
        raise Real10RenderError("canonical mixture is unexpectedly silent")
    return np.ascontiguousarray(array)


def write_deterministic_float_wav(path: Path, samples: np.ndarray) -> dict[str, Any]:
    """Write and revalidate a deterministic float32 stem, including silence."""

    canonical = np.ascontiguousarray(samples, dtype=np.float32)
    expected_pcm_sha = pcm_f32le_sha256(canonical)
    _write_bytes_fsync(path, _canonical_float_wav_bytes(canonical))
    decoded = _decode_float_wav_exact(path, allow_silence=True)
    if not np.array_equal(decoded.view(np.uint32), canonical.view(np.uint32)):
        raise Real10RenderError(f"WAV round trip changed float32 samples: {path}")
    actual_pcm_sha = pcm_f32le_sha256(decoded)
    if actual_pcm_sha != expected_pcm_sha:
        raise Real10RenderError(f"WAV PCM hash mismatch after write: {path}")
    identity = file_identity(path)
    return {
        "path": "",  # The caller fills a staging-independent relative path.
        "file_sha256": identity["sha256"],
        "pcm_f32le_sha256": actual_pcm_sha,
        "size_bytes": identity["size_bytes"],
        "sample_rate": SAMPLE_RATE,
        "num_channels": NUM_CHANNELS,
        "num_samples": NUM_SAMPLES,
        "duration_seconds": DURATION_SECONDS,
        "dtype": "float32",
        "wav_subtype": CANONICAL_WAV_SUBTYPE,
    }


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Real10RenderError(f"cannot read {context}: {path}") from error
    if not isinstance(payload, Mapping):
        raise Real10RenderError(f"{context} must contain a JSON object")
    return payload


def read_inference_manifest(
    path: Path,
) -> tuple[tuple[QCESReal10InferenceRecord, ...], Mapping[str, Any], str]:
    """Read an exact inference-only JSONL manifest with a stable identity."""

    resolved = path.resolve()
    _reject_symlink_components(path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    before = file_identity(resolved)
    rows: list[Mapping[str, Any]] = []
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise Real10RenderError(
                        f"manifest line {line_number} lacks a final newline"
                    )
                if not line.strip():
                    raise Real10RenderError(f"manifest line {line_number} is blank")
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise Real10RenderError(
                        f"manifest line {line_number} is not an object"
                    )
                # This exact parser rejects the scoring view and every gold field.
                parse_inference_record(row)
                rows.append(row)
    except json.JSONDecodeError as error:
        raise Real10RenderError(
            f"invalid JSON at manifest line {error.lineno}"
        ) from error
    try:
        records = parse_inference_manifest(rows)
    except (TypeError, ValueError) as error:
        raise Real10RenderError(f"invalid inference manifest: {error}") from error
    after = file_identity(resolved)
    if not _same_file_content(before, after):
        raise Real10RenderError("inference manifest changed while being read")
    return records, before, canonical_inference_manifest_fingerprint(records)


def _select_records(
    records: Sequence[QCESReal10InferenceRecord], split: str
) -> tuple[QCESReal10InferenceRecord, ...]:
    selected = tuple(
        sorted(
            (row for row in records if row.split == split),
            key=lambda row: row.sample_id,
        )
    )
    if not selected:
        raise Real10RenderError(f"manifest has no {split} records")
    return selected


def load_canonical_mixtures(
    manifest_path: Path,
    records: Sequence[QCESReal10InferenceRecord],
) -> dict[str, CanonicalMixture]:
    """Decode every unique scene exactly once without resampling or downmixing."""

    first_by_scene: dict[str, QCESReal10InferenceRecord] = {}
    for record in records:
        first_by_scene.setdefault(record.scene_id, record)
    result: dict[str, CanonicalMixture] = {}
    for scene_id in sorted(first_by_scene):
        record = first_by_scene[scene_id]
        manifest_root = manifest_path.resolve().parent
        lexical_path = manifest_root.joinpath(*Path(record.mixture_path).parts)
        _reject_symlink_components(lexical_path, root=manifest_root)
        path = resolve_manifest_mixture_path(manifest_path, record)
        if not path.is_file():
            raise FileNotFoundError(path)
        before = file_identity(path)
        if before["sha256"] != record.mixture_sha256:
            raise Real10RenderError(f"canonical mixture SHA256 mismatch for {scene_id}")
        samples = _decode_bound_mixture_bytes(path, record.mixture_sha256)
        after = file_identity(path)
        if not _same_file_content(before, after):
            raise Real10RenderError(
                f"canonical mixture changed while decoding: {scene_id}"
            )
        result[scene_id] = CanonicalMixture(
            scene_id=scene_id,
            manifest_path=record.mixture_path,
            path=path,
            samples=samples,
            file_identity=before,
            pcm_f32le_sha256=pcm_f32le_sha256(samples),
        )
    return result


def runtime_source_identity() -> dict[str, Any]:
    """Hash only executable files on the QCES real-render inference path."""

    root = Path(__file__).resolve().parent.parent
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for filename in _RUNTIME_SOURCE_FILES:
        path = root / filename
        identity = file_identity(path)
        relative = path.relative_to(root).as_posix()
        portable = _portable_file_identity(identity)
        rows.append({"path": relative, **portable})
        for value in (relative, portable["sha256"], portable["size_bytes"]):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return {
        "format": RUNTIME_SOURCE_IDENTITY_FORMAT,
        "aggregate_sha256": digest.hexdigest(),
        "files": rows,
    }


def _load_and_validate_qces_checkpoint(
    path: Path,
    *,
    audiosep_config_identity: Mapping[str, Any],
    audiosep_checkpoint_identity: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any], QCESConfig]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    before = file_identity(resolved)
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except Exception as error:
        raise Real10RenderError(f"cannot load QCES checkpoint: {resolved}") from error
    after = file_identity(resolved)
    if not _same_file_content(before, after):
        raise Real10RenderError("QCES checkpoint changed while loading")
    if not isinstance(payload, Mapping):
        raise Real10RenderError("QCES checkpoint must contain an object")
    if payload.get("format") != "qces_v1" or payload.get("backend") != "audiosep":
        raise Real10RenderError(
            "QCES-Real-10 renderer requires a qces_v1 AudioSep checkpoint"
        )
    config_payload = payload.get("config")
    composer_state = payload.get("composer_state_dict")
    if not isinstance(config_payload, dict) or not isinstance(composer_state, dict):
        raise Real10RenderError("QCES checkpoint lacks config or composer state")
    try:
        config = QCESConfig.from_dict(config_payload)
    except (TypeError, ValueError) as error:
        raise Real10RenderError(f"invalid QCES checkpoint config: {error}") from error
    if config.sample_rate != SAMPLE_RATE:
        raise Real10RenderError(f"QCES checkpoint sample rate must be {SAMPLE_RATE}")
    if config.foundation_feature_mode != AUDIOSEP_CLAP_FOUNDATION_FEATURES:
        raise Real10RenderError(
            "QCES-Real-10 renderer requires frozen AudioSep-CLAP features"
        )
    extra = payload.get("extra")
    if not isinstance(extra, Mapping):
        raise Real10RenderError("QCES checkpoint lacks training provenance")
    declared_audiosep = extra.get("audiosep")
    if not isinstance(declared_audiosep, Mapping):
        raise Real10RenderError("QCES checkpoint lacks AudioSep provenance")
    if (
        declared_audiosep.get("frozen_backbone") is not True
        or declared_audiosep.get("backbone_trainable_parameter_count") != 0
    ):
        raise Real10RenderError("QCES checkpoint does not declare a frozen AudioSep")
    declared_config = declared_audiosep.get("config")
    declared_checkpoint = declared_audiosep.get("checkpoint")
    if not isinstance(declared_config, Mapping) or not isinstance(
        declared_checkpoint, Mapping
    ):
        raise Real10RenderError("QCES checkpoint has incomplete AudioSep identities")
    if not _same_file_content(declared_config, audiosep_config_identity):
        raise Real10RenderError(
            "QCES checkpoint was trained with a different AudioSep config"
        )
    if not _same_file_content(declared_checkpoint, audiosep_checkpoint_identity):
        raise Real10RenderError(
            "QCES checkpoint was trained with a different AudioSep checkpoint"
        )
    return payload, before, config


def _aggregate_cache_mixtures(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        for key in ("scene_id", "manifest_path", "sha256", "size_bytes"):
            encoded = str(row[key]).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _foundation_contract_from_receipt(
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return a sanitized but content-bound public feature contract."""

    required = (
        "format",
        "canonical_audio",
        "features",
        "query_encoder_state",
        "execution",
        "privacy_contract",
        "qces_real10_inference_contract",
    )
    if any(field not in receipt for field in required):
        raise Real10RenderError("foundation cache receipt lacks its Real-10 contract")
    if receipt["format"] != FOUNDATION_CACHE_RECEIPT_FORMAT:
        raise Real10RenderError("unsupported foundation cache receipt format")
    real10_contract = receipt["qces_real10_inference_contract"]
    privacy = receipt["privacy_contract"]
    # The source receipt contains explicit negative assertions. Hash those
    # assertions without copying tempting label-related field names into a
    # prediction receipt.
    contract = {
        "format": receipt["format"],
        "canonical_audio": receipt["canonical_audio"],
        "features": receipt["features"],
        "query_encoder_state": receipt["query_encoder_state"],
        "feature_extraction_execution": receipt["execution"],
        "privacy_proof_sha256": canonical_json_sha256(privacy),
        "real10_schema_version": (
            real10_contract.get("schema_version")
            if isinstance(real10_contract, Mapping)
            else None
        ),
        "real10_sealed_input_policy": (
            "require_canonical_manifest_fingerprint_and_per_scene_wav_sha256"
        ),
    }
    try:
        return json.loads(json.dumps(contract, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise Real10RenderError(
            "foundation cache contract is not canonical JSON"
        ) from error


def load_real10_foundation_feature_cache(
    cache_dir: Path,
    manifest: Path,
    records: Sequence[QCESReal10InferenceRecord],
    *,
    expected_manifest_fingerprint: str,
    mixtures: Mapping[str, CanonicalMixture],
    audiosep_checkpoint_identity: Mapping[str, Any],
    audiosep_source_identity: Mapping[str, Any],
) -> tuple[Real10FoundationFeatureCache, Mapping[str, Any]]:
    """Load the explicit hardened Real-10 cache branch.

    The legacy training loader intentionally rejects the additional sealed
    Real-10 receipt field.  This validator keeps that behavior unchanged and
    independently requires the exact inference fingerprint and per-scene WAV
    hashes before exposing any tensor.
    """

    _reject_symlink_components(cache_dir)
    resolved = cache_dir.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise Real10RenderError(f"invalid Real-10 foundation cache: {resolved}")
    receipt_path = resolved / "cache_receipt.json"
    question_path = resolved / "question_features.pt"
    scene_path = resolved / "scene_audio_features.pt"
    for path in (receipt_path, question_path, scene_path):
        if not path.is_file() or path.is_symlink():
            raise Real10RenderError(f"missing or symlinked cache artifact: {path}")
    receipt_identity_before = file_identity(receipt_path)
    question_identity_before = file_identity(question_path)
    scene_identity_before = file_identity(scene_path)
    receipt = _read_json(receipt_path, "Real-10 foundation cache receipt")
    top = _exact_keys(
        receipt,
        {
            "format",
            "purpose",
            "manifest",
            "schema_versions",
            "audiosep_checkpoint",
            "audiosep_source_tree",
            "query_encoder_state",
            "canonical_audio",
            "features",
            "counts",
            "sample_ids",
            "scene_ids",
            "mixtures",
            "mixtures_aggregate_sha256",
            "execution",
            "privacy_contract",
            "artifacts",
            "qces_real10_inference_contract",
        },
        "Real-10 foundation cache receipt",
    )
    if (
        top["format"] != FOUNDATION_CACHE_RECEIPT_FORMAT
        or top["purpose"] != "frozen_controller_inputs_without_oracle_labels"
    ):
        raise Real10RenderError("invalid Real-10 foundation cache purpose")

    current_manifest_identity = file_identity(manifest)
    declared_manifest = _exact_keys(
        top["manifest"],
        {"path", "sha256", "size_bytes"},
        "cache manifest identity",
    )
    if not _same_file_content(declared_manifest, current_manifest_identity):
        raise Real10RenderError("foundation cache differs from inference manifest")
    if top["schema_versions"] != [INFERENCE_SCHEMA_VERSION]:
        raise Real10RenderError("foundation cache is not inference-v2-only")

    real10 = _exact_keys(
        top["qces_real10_inference_contract"],
        {
            "schema_version",
            "canonical_inference_manifest_fingerprint",
            "canonical_wav_sha256_by_scene",
            "contains_event_answer_or_oracle_inputs",
        },
        "cache Real-10 inference contract",
    )
    if (
        real10["schema_version"] != INFERENCE_SCHEMA_VERSION
        or real10["canonical_inference_manifest_fingerprint"]
        != expected_manifest_fingerprint
        or real10["contains_event_answer_or_oracle_inputs"] is not False
    ):
        raise Real10RenderError("cache Real-10 inference binding is invalid")
    expected_hashes = {
        scene_id: mixture.file_identity["sha256"]
        for scene_id, mixture in sorted(mixtures.items())
    }
    if real10["canonical_wav_sha256_by_scene"] != expected_hashes:
        raise Real10RenderError("cache canonical-WAV binding differs from manifest")

    declared_checkpoint = _exact_keys(
        top["audiosep_checkpoint"],
        {"path", "sha256", "size_bytes"},
        "cache AudioSep checkpoint identity",
    )
    if not _same_file_content(declared_checkpoint, audiosep_checkpoint_identity):
        raise Real10RenderError("cache uses a different AudioSep checkpoint")
    declared_source = _exact_keys(
        top["audiosep_source_tree"],
        {"path", "sha256", "hashed_file_count", "included_suffixes"},
        "cache AudioSep source identity",
    )
    if (
        declared_source.get("sha256") != audiosep_source_identity.get("sha256")
        or declared_source.get("hashed_file_count")
        != audiosep_source_identity.get("hashed_file_count")
        or declared_source.get("included_suffixes")
        != audiosep_source_identity.get("included_suffixes")
    ):
        raise Real10RenderError("cache uses a different AudioSep source tree")

    canonical = _exact_keys(
        top["canonical_audio"],
        {
            "sample_rate",
            "num_samples",
            "duration_seconds",
            "num_channels",
            "clap_sample_rate",
            "clap_num_samples",
        },
        "cache canonical audio",
    )
    if dict(canonical) != {
        "sample_rate": SAMPLE_RATE,
        "num_samples": NUM_SAMPLES,
        "duration_seconds": DURATION_SECONDS,
        "num_channels": NUM_CHANNELS,
        "clap_sample_rate": 48_000,
        "clap_num_samples": 480_000,
    }:
        raise Real10RenderError("cache canonical audio contract drifted")
    features = _exact_keys(
        top["features"],
        {
            "feature_space",
            "question_shape",
            "question_dtype",
            "raw_audio_shape",
            "effective_audio_shape",
            "effective_audio_dtype",
            "raw_to_effective_repeat_ratio",
            "audio_normalization",
        },
        "cache feature contract",
    )
    if dict(features) != {
        "feature_space": FEATURE_SPACE,
        "question_shape": [JOINT_DIM],
        "question_dtype": "float32",
        "raw_audio_shape": [RAW_FINE_FRAMES, FINE_DIM],
        "effective_audio_shape": [EFFECTIVE_FRAMES, JOINT_DIM],
        "effective_audio_dtype": "float16",
        "raw_to_effective_repeat_ratio": REPEAT_RATIO,
        "audio_normalization": "per_frame_l2_before_fp16_storage",
    }:
        raise Real10RenderError("cache frozen-feature declaration drifted")
    query_state = _exact_keys(
        top["query_encoder_state"],
        {
            "extracted_tensor_count",
            "loaded_tensor_count",
            "effective_missing_keys",
            "effective_unexpected_keys",
            "ignored_nonpersistent_checkpoint_keys",
            "fusion_enabled",
        },
        "cache query-encoder state",
    )
    ignored = query_state["ignored_nonpersistent_checkpoint_keys"]
    if (
        query_state["extracted_tensor_count"] != EXPECTED_QUERY_STATE_TENSORS
        or isinstance(query_state["loaded_tensor_count"], bool)
        or not isinstance(query_state["loaded_tensor_count"], int)
        or query_state["loaded_tensor_count"] <= 0
        or query_state["effective_missing_keys"] != []
        or query_state["effective_unexpected_keys"] != []
        or not isinstance(ignored, list)
        or any(
            not isinstance(key, str) or not key.endswith("embeddings.position_ids")
            for key in ignored
        )
        or query_state["fusion_enabled"] is not False
    ):
        raise Real10RenderError("cache query-encoder state is incompatible")
    privacy = _exact_keys(
        top["privacy_contract"],
        {
            "question_text_stored",
            "event_labels_stored",
            "answer_labels_stored",
            "evidence_annotations_stored",
            "allowed_keys",
            "contains_event_answer_or_oracle_inputs",
        },
        "cache privacy contract",
    )
    if dict(privacy) != {
        "question_text_stored": False,
        "event_labels_stored": False,
        "answer_labels_stored": False,
        "evidence_annotations_stored": False,
        "allowed_keys": "sample_id_and_scene_id_only",
        "contains_event_answer_or_oracle_inputs": False,
    }:
        raise Real10RenderError("cache privacy contract is not inference-safe")

    sample_to_scene = {record.sample_id: record.scene_id for record in records}
    expected_sample_ids = sorted(sample_to_scene)
    expected_scene_ids = sorted(mixtures)
    if len(sample_to_scene) != len(records):
        raise Real10RenderError("inference manifest has duplicate sample IDs")
    if (
        top["sample_ids"] != expected_sample_ids
        or top["scene_ids"] != expected_scene_ids
    ):
        raise Real10RenderError("cache IDs differ from inference manifest")
    counts = _exact_keys(
        top["counts"],
        {
            "sample_ids",
            "scene_ids",
            "unique_full_question_texts",
            "physical_mixture_encodes",
        },
        "cache counts",
    )
    if (
        counts["sample_ids"] != len(expected_sample_ids)
        or counts["scene_ids"] != len(expected_scene_ids)
        or counts["physical_mixture_encodes"] != len(expected_scene_ids)
        or isinstance(counts["unique_full_question_texts"], bool)
        or not isinstance(counts["unique_full_question_texts"], int)
        or not 1 <= counts["unique_full_question_texts"] <= len(expected_sample_ids)
    ):
        raise Real10RenderError("cache counts are inconsistent")

    mixture_rows = top["mixtures"]
    if not isinstance(mixture_rows, list) or len(mixture_rows) != len(
        expected_scene_ids
    ):
        raise Real10RenderError("cache mixture provenance rows are invalid")
    validated_mixture_rows: list[Mapping[str, Any]] = []
    for raw in mixture_rows:
        row = _exact_keys(
            raw,
            {
                "scene_id",
                "manifest_path",
                "sha256",
                "size_bytes",
                "sample_rate",
                "num_samples",
                "duration_seconds",
                "feature_shape",
                "feature_dtype",
                "declared_mixture_sha256",
            },
            "cache mixture identity",
        )
        scene_id = row["scene_id"]
        if not isinstance(scene_id, str) or scene_id not in mixtures:
            raise Real10RenderError("cache mixture has an unknown scene ID")
        expected = mixtures[scene_id]
        if (
            row["manifest_path"] != expected.manifest_path
            or row["sha256"] != expected.file_identity["sha256"]
            or row["declared_mixture_sha256"] != expected.file_identity["sha256"]
            or row["size_bytes"] != expected.file_identity["size_bytes"]
            or row["sample_rate"] != SAMPLE_RATE
            or row["num_samples"] != NUM_SAMPLES
            or row["duration_seconds"] != DURATION_SECONDS
            or row["feature_shape"] != [EFFECTIVE_FRAMES, JOINT_DIM]
            or row["feature_dtype"] != "float16"
        ):
            raise Real10RenderError(f"cache mixture identity drifted: {scene_id}")
        validated_mixture_rows.append(row)
    if sorted(row["scene_id"] for row in validated_mixture_rows) != expected_scene_ids:
        raise Real10RenderError("cache mixture scene IDs are not exact")
    if top["mixtures_aggregate_sha256"] != _aggregate_cache_mixtures(
        validated_mixture_rows
    ):
        raise Real10RenderError("cache mixture aggregate hash is invalid")

    execution = _exact_keys(
        top["execution"],
        {"device", "text_batch_size", "determinism", "software"},
        "cache execution",
    )
    if (
        not isinstance(execution["device"], str)
        or isinstance(execution["text_batch_size"], bool)
        or not isinstance(execution["text_batch_size"], int)
        or execution["text_batch_size"] <= 0
        or not isinstance(execution["software"], Mapping)
    ):
        raise Real10RenderError("cache execution provenance is invalid")
    determinism = _exact_keys(
        execution["determinism"],
        {
            "seed",
            "torch_deterministic_algorithms",
            "cublas_workspace_config",
            "cudnn_benchmark",
            "cudnn_deterministic",
            "cuda_matmul_allow_tf32",
            "cudnn_allow_tf32",
        },
        "cache determinism",
    )
    if (
        isinstance(determinism["seed"], bool)
        or not isinstance(determinism["seed"], int)
        or determinism["torch_deterministic_algorithms"] is not True
        or determinism["cudnn_benchmark"] is not False
        or determinism["cudnn_deterministic"] is not True
        or determinism["cuda_matmul_allow_tf32"] is not False
        or determinism["cudnn_allow_tf32"] is not False
    ):
        raise Real10RenderError("cache was not generated deterministically")

    artifacts = _exact_keys(
        top["artifacts"],
        {"question_features", "scene_audio_features"},
        "cache artifacts",
    )
    for name, path, expected_filename, before in (
        (
            "question_features",
            question_path,
            "question_features.pt",
            question_identity_before,
        ),
        (
            "scene_audio_features",
            scene_path,
            "scene_audio_features.pt",
            scene_identity_before,
        ),
    ):
        declared = _exact_keys(
            artifacts[name],
            {"filename", "sha256", "size_bytes"},
            f"cache {name} artifact",
        )
        if declared["filename"] != expected_filename or not _same_file_content(
            declared, before
        ):
            raise Real10RenderError(f"cache {name} artifact hash mismatch")
    try:
        question_payload = torch.load(
            question_path, map_location="cpu", weights_only=True
        )
        scene_payload = torch.load(scene_path, map_location="cpu", weights_only=True)
        validate_question_payload(question_payload)
        validate_scene_payload(scene_payload)
    except Exception as error:
        raise Real10RenderError("cache feature tensor payloads are invalid") from error
    if (
        question_payload.get("format") != QUESTION_CACHE_FORMAT
        or scene_payload.get("format") != SCENE_CACHE_FORMAT
    ):
        raise Real10RenderError("cache feature payload formats are invalid")
    question_features = question_payload["features"]
    scene_features = scene_payload["features"]
    if set(question_features) != set(expected_sample_ids) or set(scene_features) != set(
        expected_scene_ids
    ):
        raise Real10RenderError("cache tensor IDs differ from receipt")
    question_norms = torch.linalg.vector_norm(
        torch.stack([question_features[key] for key in expected_sample_ids]).float(),
        dim=-1,
    )
    scene_norms = torch.linalg.vector_norm(
        torch.stack([scene_features[key] for key in expected_scene_ids]).float(),
        dim=-1,
    )
    if not torch.allclose(
        question_norms,
        torch.ones_like(question_norms),
        rtol=1e-3,
        atol=1e-3,
    ) or not torch.allclose(
        scene_norms,
        torch.ones_like(scene_norms),
        rtol=2e-3,
        atol=2e-3,
    ):
        raise Real10RenderError("cache features are not L2 normalized")

    if not _same_file_content(receipt_identity_before, file_identity(receipt_path)):
        raise Real10RenderError("cache receipt changed while loading")
    if not _same_file_content(question_identity_before, file_identity(question_path)):
        raise Real10RenderError("cache question features changed while loading")
    if not _same_file_content(scene_identity_before, file_identity(scene_path)):
        raise Real10RenderError("cache scene features changed while loading")
    contract = _foundation_contract_from_receipt(top)
    identity = {
        "format": FOUNDATION_CACHE_RECEIPT_FORMAT,
        "directory": str(resolved),
        "receipt": receipt_identity_before,
        "question_feature_artifact": question_identity_before,
        "scene_feature_artifact": scene_identity_before,
        "manifest_binding": {
            "cache_declared": dict(declared_manifest),
            "run_expected": current_manifest_identity,
            "canonical_fingerprint": expected_manifest_fingerprint,
        },
        "audiosep_checkpoint_binding": {
            "cache_declared": dict(declared_checkpoint),
            "run_expected": dict(audiosep_checkpoint_identity),
        },
        "audiosep_source_binding": {
            "cache_declared": dict(declared_source),
            "run_expected": dict(audiosep_source_identity),
        },
        "sample_count": len(expected_sample_ids),
        "scene_count": len(expected_scene_ids),
        "contains_gold_label_inputs": False,
    }
    return (
        Real10FoundationFeatureCache(
            question_features=question_features,
            scene_features=scene_features,
            sample_to_scene=sample_to_scene,
            identity=identity,
        ),
        contract,
    )


def _summarize_foundation_identity(
    identity: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    required = (
        "receipt",
        "question_feature_artifact",
        "scene_feature_artifact",
        "manifest_binding",
        "audiosep_checkpoint_binding",
        "audiosep_source_binding",
        "sample_count",
        "scene_count",
        "contains_gold_label_inputs",
    )
    if any(field not in identity for field in required):
        raise Real10RenderError("foundation cache identity is incomplete")
    if identity["contains_gold_label_inputs"] is not False:
        raise Real10RenderError("foundation cache is not inference-safe")
    return {
        "receipt": identity["receipt"],
        "question_feature_artifact": identity["question_feature_artifact"],
        "scene_feature_artifact": identity["scene_feature_artifact"],
        "manifest_binding": identity["manifest_binding"],
        "audiosep_checkpoint_binding": identity["audiosep_checkpoint_binding"],
        "audiosep_source_binding": identity["audiosep_source_binding"],
        "sample_count": identity["sample_count"],
        "scene_count": identity["scene_count"],
        "contract_sha256": canonical_json_sha256(contract),
        "contains_gold_label_inputs": False,
    }


def prepare_render_inputs(
    *,
    manifest_path: Path,
    foundation_cache_dir: Path,
    qces_checkpoint_path: Path,
    audiosep_root: Path,
    audiosep_config_path: Path,
    audiosep_checkpoint_path: Path,
    settings: RenderSettings,
    foundation_loader: Callable[
        ..., tuple[Real10FoundationFeatureCache, Mapping[str, Any]]
    ] = (load_real10_foundation_feature_cache),
) -> PreparedRenderInputs:
    """Validate every label-free input and bind it before model construction."""

    settings.validate()
    records, manifest_identity, full_fingerprint = read_inference_manifest(
        manifest_path
    )
    selected = _select_records(records, settings.split)
    selected_fingerprint = canonical_inference_manifest_fingerprint(selected)
    mixtures = load_canonical_mixtures(manifest_path.resolve(), records)

    resolved_audiosep_root = audiosep_root.resolve()
    resolved_audiosep_config = audiosep_config_path.resolve()
    resolved_audiosep_checkpoint = audiosep_checkpoint_path.resolve()
    if not resolved_audiosep_root.is_dir():
        raise FileNotFoundError(resolved_audiosep_root)
    audiosep_source = source_tree_identity(resolved_audiosep_root)
    audiosep_config = file_identity(resolved_audiosep_config)
    audiosep_checkpoint = file_identity(resolved_audiosep_checkpoint)
    payload, qces_identity, config = _load_and_validate_qces_checkpoint(
        qces_checkpoint_path,
        audiosep_config_identity=audiosep_config,
        audiosep_checkpoint_identity=audiosep_checkpoint,
    )
    try:
        cache, contract = foundation_loader(
            foundation_cache_dir,
            manifest_path.resolve(),
            records,
            expected_manifest_fingerprint=full_fingerprint,
            mixtures=mixtures,
            audiosep_checkpoint_identity=audiosep_checkpoint,
            audiosep_source_identity=audiosep_source,
        )
    except SystemExit as error:  # Defensive: injected loaders may use CLI errors.
        raise Real10RenderError(str(error)) from error
    cache_identity = _summarize_foundation_identity(cache.identity, contract)
    return PreparedRenderInputs(
        manifest_path=manifest_path.resolve(),
        manifest_identity=manifest_identity,
        records=records,
        selected_records=selected,
        full_manifest_fingerprint=full_fingerprint,
        selected_manifest_fingerprint=selected_fingerprint,
        mixtures=mixtures,
        foundation_cache=cache,
        foundation_cache_contract=contract,
        foundation_cache_identity=cache_identity,
        qces_checkpoint_path=qces_checkpoint_path.resolve(),
        qces_checkpoint_identity=qces_identity,
        qces_checkpoint_payload=payload,
        qces_config=config,
        audiosep_root=resolved_audiosep_root,
        audiosep_source_identity=audiosep_source,
        audiosep_config_path=resolved_audiosep_config,
        audiosep_config_identity=audiosep_config,
        audiosep_checkpoint_path=resolved_audiosep_checkpoint,
        audiosep_checkpoint_identity=audiosep_checkpoint,
        runtime_source_identity=runtime_source_identity(),
    )


def build_method_identity(
    prepared: PreparedRenderInputs, settings: RenderSettings
) -> dict[str, Any]:
    """Build the split-independent method identity frozen before real-test."""

    settings.validate()
    config_payload = prepared.qces_config.to_dict()
    return {
        "format": METHOD_IDENTITY_FORMAT,
        "qces_checkpoint": _portable_file_identity(prepared.qces_checkpoint_identity),
        "qces_checkpoint_format": "qces_v1",
        "qces_backend": "audiosep",
        "qces_config": config_payload,
        "qces_config_sha256": canonical_json_sha256(config_payload),
        "audiosep_source_tree": {
            "sha256": prepared.audiosep_source_identity["sha256"],
            "hashed_file_count": prepared.audiosep_source_identity["hashed_file_count"],
            "included_suffixes": prepared.audiosep_source_identity["included_suffixes"],
        },
        "audiosep_config": _portable_file_identity(prepared.audiosep_config_identity),
        "audiosep_checkpoint": _portable_file_identity(
            prepared.audiosep_checkpoint_identity
        ),
        "foundation_cache_contract": prepared.foundation_cache_contract,
        "foundation_question_payload_format": QUESTION_CACHE_FORMAT,
        "foundation_scene_payload_format": SCENE_CACHE_FORMAT,
        "runtime_source": prepared.runtime_source_identity,
        "software_contract": _software_identity(),
        "inference_schema_version": INFERENCE_SCHEMA_VERSION,
        "prediction_schema_version": PREDICTION_SCHEMA_VERSION,
        "role_threshold": float(settings.role_threshold),
        "no_evidence_threshold": float(settings.no_evidence_threshold),
        "seed": settings.seed,
        "device_type": settings.device_type,
        "batch_size": 1,
        "inference_precision": "float32_no_autocast",
        "audio_io": {
            "sample_rate": SAMPLE_RATE,
            "num_channels": NUM_CHANNELS,
            "num_samples": NUM_SAMPLES,
            "duration_seconds": DURATION_SECONDS,
            "input_policy": "no_resample_no_downmix",
            "output_wav_subtype": CANONICAL_WAV_SUBTYPE,
        },
    }


def _canonical_json_bytes(payload: Mapping[str, Any], *, pretty: bool) -> bytes:
    if pretty:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def _assert_public_payload_has_no_invalid_claim_fields(
    value: Any, *, context: str
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise Real10RenderError(f"{context} has a non-string JSON key")
            normalized = key.casefold().replace("-", "_")
            if "oracle" in normalized or "sdr" in normalized:
                raise Real10RenderError(
                    f"{context} contains a prohibited claim field: {key!r}"
                )
            _assert_public_payload_has_no_invalid_claim_fields(item, context=context)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_public_payload_has_no_invalid_claim_fields(item, context=context)


def write_method_freeze_receipt(
    path: Path,
    *,
    prepared: PreparedRenderInputs,
    settings: RenderSettings,
) -> Mapping[str, Any]:
    """Exclusively freeze one dev-selected method before real-test access."""

    if settings.split != "real_dev":
        raise Real10RenderError(
            "a method-freeze receipt can only be created from real_dev"
        )
    if not prepared.selected_records or any(
        record.split != "real_dev" for record in prepared.selected_records
    ):
        raise Real10RenderError(
            "method-freeze inputs must be an actual real_dev inference view"
        )
    method = build_method_identity(prepared, settings)
    receipt = {
        "format": METHOD_FREEZE_RECEIPT_FORMAT,
        "purpose": "freeze_method_before_single_real_test_render",
        "source_split": "real_dev",
        "method": method,
        "method_identity_sha256": canonical_json_sha256(method),
        "freeze_context": {
            "source_inference_manifest": {
                **dict(prepared.manifest_identity),
                "canonical_fingerprint": prepared.full_manifest_fingerprint,
            },
            "selected_real_dev_view": {
                "canonical_fingerprint": prepared.selected_manifest_fingerprint,
                "record_count": len(prepared.selected_records),
                "scene_count": len(
                    {record.scene_id for record in prepared.selected_records}
                ),
            },
            "foundation_cache_receipt": prepared.foundation_cache_identity["receipt"],
        },
        "declaration": {
            "method_selected_without_real_test_metrics": True,
            "real_test_model_selection_prohibited": True,
            "single_real_test_render_after_freeze": True,
            "real_test_render_count_before_freeze_↓": 0,
        },
    }
    _assert_public_payload_has_no_invalid_claim_fields(
        receipt, context="method-freeze receipt"
    )
    payload = _canonical_json_bytes(receipt, pretty=True)
    _reject_symlink_components(path)
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() or resolved.is_symlink():
        try:
            existing = resolved.read_bytes()
        except OSError as error:
            raise Real10RenderError(
                f"cannot inspect existing method-freeze receipt: {resolved}"
            ) from error
        if existing != payload:
            raise FileExistsError(
                f"refusing to overwrite different method-freeze receipt: {resolved}"
            )
        return receipt
    _write_bytes_fsync(resolved, payload)
    return receipt


def validate_method_freeze_receipt(
    path: Path,
    *,
    expected_method: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Verify a real-test gate against the exact current method identity."""

    _reject_symlink_components(path)
    resolved = path.resolve()
    identity_before = file_identity(resolved)
    receipt = _read_json(resolved, "method-freeze receipt")
    _exact_keys(
        receipt,
        {
            "format",
            "purpose",
            "source_split",
            "method",
            "method_identity_sha256",
            "freeze_context",
            "declaration",
        },
        "method-freeze receipt",
    )
    if (
        receipt["format"] != METHOD_FREEZE_RECEIPT_FORMAT
        or receipt["purpose"] != "freeze_method_before_single_real_test_render"
        or receipt["source_split"] != "real_dev"
    ):
        raise Real10RenderError("invalid method-freeze receipt purpose")
    declared_method = receipt["method"]
    if not isinstance(declared_method, Mapping):
        raise Real10RenderError("method-freeze receipt method must be an object")
    declared_hash = receipt["method_identity_sha256"]
    if not _is_sha256(declared_hash) or declared_hash != canonical_json_sha256(
        declared_method
    ):
        raise Real10RenderError("method-freeze receipt hash is invalid")
    if dict(declared_method) != dict(expected_method):
        raise Real10RenderError(
            "current method differs from the method frozen before real-test"
        )
    freeze_context = _exact_keys(
        receipt["freeze_context"],
        {
            "source_inference_manifest",
            "selected_real_dev_view",
            "foundation_cache_receipt",
        },
        "method-freeze context",
    )
    source_manifest = freeze_context["source_inference_manifest"]
    selected_view = freeze_context["selected_real_dev_view"]
    cache_receipt = freeze_context["foundation_cache_receipt"]
    if (
        not isinstance(source_manifest, Mapping)
        or not _is_sha256(source_manifest.get("sha256"))
        or not _is_sha256(source_manifest.get("canonical_fingerprint"))
        or not isinstance(selected_view, Mapping)
        or not _is_sha256(selected_view.get("canonical_fingerprint"))
        or isinstance(selected_view.get("record_count"), bool)
        or not isinstance(selected_view.get("record_count"), int)
        or selected_view["record_count"] <= 0
        or isinstance(selected_view.get("scene_count"), bool)
        or not isinstance(selected_view.get("scene_count"), int)
        or selected_view["scene_count"] <= 0
        or not isinstance(cache_receipt, Mapping)
        or not _is_sha256(cache_receipt.get("sha256"))
    ):
        raise Real10RenderError("method-freeze context is invalid")
    declaration = _exact_keys(
        receipt["declaration"],
        {
            "method_selected_without_real_test_metrics",
            "real_test_model_selection_prohibited",
            "single_real_test_render_after_freeze",
            "real_test_render_count_before_freeze_↓",
        },
        "method-freeze declaration",
    )
    if dict(declaration) != {
        "method_selected_without_real_test_metrics": True,
        "real_test_model_selection_prohibited": True,
        "single_real_test_render_after_freeze": True,
        "real_test_render_count_before_freeze_↓": 0,
    }:
        raise Real10RenderError("method-freeze declaration is not fail-closed")
    identity_after = file_identity(resolved)
    if not _same_file_content(identity_before, identity_after):
        raise Real10RenderError("method-freeze receipt changed while loading")
    return receipt, identity_before


def _parse_intervals(value: Any, context: str) -> list[list[float]]:
    if not isinstance(value, list):
        raise Real10RenderError(f"{context} must be a list")
    result: list[list[float]] = []
    previous_end = -math.inf
    for index, raw in enumerate(value):
        if not isinstance(raw, list) or len(raw) != 2:
            raise Real10RenderError(f"{context}[{index}] must be [onset, offset]")
        onset = _finite_number(raw[0], f"{context}[{index}][0]")
        offset = _finite_number(raw[1], f"{context}[{index}][1]")
        if not 0.0 <= onset < offset <= DURATION_SECONDS:
            raise Real10RenderError(f"{context}[{index}] lies outside [0, 10]")
        if onset < previous_end:
            raise Real10RenderError(f"{context} must be sorted and non-overlapping")
        result.append([onset, offset])
        previous_end = offset
    return result


def _union_intervals(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> list[list[float]]:
    ordered = sorted(
        ([float(item[0]), float(item[1])] for item in (*first, *second)),
        key=lambda item: (item[0], item[1]),
    )
    merged: list[list[float]] = []
    for onset, offset in ordered:
        if not merged or onset > merged[-1][1]:
            merged.append([onset, offset])
        else:
            merged[-1][1] = max(merged[-1][1], offset)
    return merged


def _parse_audio_identity(value: Any, context: str) -> Mapping[str, Any]:
    identity = _exact_keys(
        value,
        {
            "path",
            "file_sha256",
            "pcm_f32le_sha256",
            "size_bytes",
            "sample_rate",
            "num_channels",
            "num_samples",
            "duration_seconds",
            "dtype",
            "wav_subtype",
        },
        context,
    )
    _safe_relative_wav(identity["path"], f"{context}.path")
    if not _is_sha256(identity["file_sha256"]) or not _is_sha256(
        identity["pcm_f32le_sha256"]
    ):
        raise Real10RenderError(f"{context} has an invalid SHA256")
    if (
        isinstance(identity["size_bytes"], bool)
        or not isinstance(identity["size_bytes"], int)
        or identity["size_bytes"] <= 0
        or identity["sample_rate"] != SAMPLE_RATE
        or identity["num_channels"] != NUM_CHANNELS
        or identity["num_samples"] != NUM_SAMPLES
        or identity["duration_seconds"] != DURATION_SECONDS
        or identity["dtype"] != "float32"
        or identity["wav_subtype"] != CANONICAL_WAV_SUBTYPE
    ):
        raise Real10RenderError(f"{context} is not canonical float32 audio")
    return identity


def validate_prediction_record(value: Any) -> Mapping[str, Any]:
    """Validate the strict prediction row consumed by the later scorer."""

    row = _exact_keys(
        value,
        {
            "schema_version",
            "id",
            "scene_id",
            "split",
            "question_index",
            "relation",
            "inference_record_sha256",
            "mixture",
            "prediction",
            "stems",
            "separator_calls",
            "max_evidence_plus_residual_minus_mixture_abs_error",
        },
        "QCES-Real-10 prediction record",
    )
    if row["schema_version"] != PREDICTION_SCHEMA_VERSION:
        raise Real10RenderError("unsupported prediction schema version")
    for field in ("id", "scene_id"):
        if not isinstance(row[field], str) or not row[field]:
            raise Real10RenderError(f"prediction {field} must be non-empty text")
    if row["split"] not in {"real_dev", "real_test"}:
        raise Real10RenderError("prediction split is invalid")
    if (
        isinstance(row["question_index"], bool)
        or not isinstance(row["question_index"], int)
        or row["question_index"] < 0
    ):
        raise Real10RenderError("prediction question_index is invalid")
    if row["relation"] not in {"after", "before", "first"}:
        raise Real10RenderError("prediction relation is invalid")
    if not _is_sha256(row["inference_record_sha256"]):
        raise Real10RenderError("prediction inference binding is invalid")

    mixture = _exact_keys(
        row["mixture"],
        {
            "manifest_path",
            "file_sha256",
            "pcm_f32le_sha256",
            "size_bytes",
            "sample_rate",
            "num_channels",
            "num_samples",
            "duration_seconds",
        },
        "prediction mixture",
    )
    _safe_relative_wav(mixture["manifest_path"], "mixture.manifest_path")
    if not _is_sha256(mixture["file_sha256"]) or not _is_sha256(
        mixture["pcm_f32le_sha256"]
    ):
        raise Real10RenderError("prediction mixture hash is invalid")
    if (
        isinstance(mixture["size_bytes"], bool)
        or not isinstance(mixture["size_bytes"], int)
        or mixture["size_bytes"] <= 0
        or mixture["sample_rate"] != SAMPLE_RATE
        or mixture["num_channels"] != NUM_CHANNELS
        or mixture["num_samples"] != NUM_SAMPLES
        or mixture["duration_seconds"] != DURATION_SECONDS
    ):
        raise Real10RenderError("prediction mixture is not canonical")

    prediction = _exact_keys(
        row["prediction"],
        {
            "anchor_intervals",
            "answer_intervals",
            "union_intervals",
            "no_evidence_probability",
            "no_evidence_prediction",
            "same_semantic_probability",
            "role_threshold",
            "no_evidence_threshold",
            "temporal_role_mode",
            "semantic_separation_mode",
        },
        "prediction outputs",
    )
    anchor = _parse_intervals(prediction["anchor_intervals"], "anchor_intervals")
    answer = _parse_intervals(prediction["answer_intervals"], "answer_intervals")
    union = _parse_intervals(prediction["union_intervals"], "union_intervals")
    if union != _union_intervals(anchor, answer):
        raise Real10RenderError("union_intervals must equal anchor-answer union")
    no_evidence_probability = _probability(
        prediction["no_evidence_probability"], "no_evidence_probability"
    )
    no_evidence_threshold = _probability(
        prediction["no_evidence_threshold"], "no_evidence_threshold"
    )
    role_threshold = _probability(prediction["role_threshold"], "role_threshold")
    if not 0.0 < no_evidence_threshold < 1.0 or not 0.0 < role_threshold < 1.0:
        raise Real10RenderError("prediction thresholds must lie in (0, 1)")
    if not isinstance(prediction["no_evidence_prediction"], bool) or prediction[
        "no_evidence_prediction"
    ] != (no_evidence_probability >= no_evidence_threshold):
        raise Real10RenderError("no-evidence decision disagrees with its threshold")
    same_probability = prediction["same_semantic_probability"]
    if same_probability is not None:
        _probability(same_probability, "same_semantic_probability")
    if prediction["temporal_role_mode"] not in {
        LEGACY_TEMPORAL_ROLE_MODE,
        OVERLAP_AWARE_TEMPORAL_ROLE_MODE,
    }:
        raise Real10RenderError("unsupported temporal role mode in prediction")
    semantic_mode = prediction["semantic_separation_mode"]
    if semantic_mode not in {
        UNION_SINGLE_SEMANTIC_MODE,
        DUAL_ROLE_SEMANTIC_MODE,
    }:
        raise Real10RenderError("unsupported semantic separation mode")
    if semantic_mode == UNION_SINGLE_SEMANTIC_MODE and same_probability is not None:
        raise Real10RenderError(
            "union-single prediction cannot emit same-target routing"
        )
    if semantic_mode == DUAL_ROLE_SEMANTIC_MODE and same_probability is None:
        raise Real10RenderError("dual-role prediction requires same-target routing")
    if (
        semantic_mode == DUAL_ROLE_SEMANTIC_MODE
        and prediction["temporal_role_mode"] != OVERLAP_AWARE_TEMPORAL_ROLE_MODE
    ):
        raise Real10RenderError("dual-role prediction requires overlap-aware roles")

    stems = _exact_keys(row["stems"], {"evidence", "residual"}, "prediction stems")
    _parse_audio_identity(stems["evidence"], "evidence stem")
    _parse_audio_identity(stems["residual"], "residual stem")
    if stems["evidence"]["path"] != _stem_relative_path(row["id"], "evidence"):
        raise Real10RenderError("evidence stem path does not match its sample ID")
    if stems["residual"]["path"] != _stem_relative_path(row["id"], "residual"):
        raise Real10RenderError("residual stem path does not match its sample ID")
    calls = _exact_keys(
        row["separator_calls"],
        {"physical_forwards", "effective_evaluations"},
        "separator calls",
    )
    physical_calls = _positive_integer(
        calls["physical_forwards"], "physical separator forwards"
    )
    effective_calls = _positive_integer(
        calls["effective_evaluations"], "effective evaluations"
    )
    expected_calls = (1, 1) if semantic_mode == UNION_SINGLE_SEMANTIC_MODE else (1, 2)
    if (physical_calls, effective_calls) != expected_calls:
        raise Real10RenderError(
            "separator call counts disagree with semantic separation mode"
        )
    error = _finite_number(
        row["max_evidence_plus_residual_minus_mixture_abs_error"],
        "maximum mixture reconstruction error",
    )
    if error < 0.0:
        raise Real10RenderError("maximum mixture reconstruction error is negative")
    return row


def prediction_manifest_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    validated = [dict(validate_prediction_record(row)) for row in rows]
    ids = [row["id"] for row in validated]
    if not ids or len(ids) != len(set(ids)):
        raise Real10RenderError("prediction manifest is empty or has duplicate IDs")
    return canonical_json_sha256(
        {
            "format": PREDICTION_MANIFEST_FINGERPRINT_FORMAT,
            "records": sorted(validated, key=lambda row: row["id"]),
        }
    )


def _boolean_intervals(active: torch.Tensor, *, hop_samples: int) -> list[list[float]]:
    if active.ndim != 1 or active.dtype != torch.bool:
        raise Real10RenderError("role activity must be a one-dimensional bool tensor")
    if (
        isinstance(hop_samples, bool)
        or not isinstance(hop_samples, int)
        or hop_samples <= 0
    ):
        raise Real10RenderError("frame hop must be a positive integer")
    values = active.detach().cpu().tolist()
    result: list[list[float]] = []
    start: int | None = None
    for index, value in enumerate([*values, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            start_sample = min(start * hop_samples, NUM_SAMPLES)
            end_sample = min(index * hop_samples, NUM_SAMPLES)
            if end_sample > start_sample:
                result.append(
                    [
                        round(start_sample / SAMPLE_RATE, 6),
                        round(end_sample / SAMPLE_RATE, 6),
                    ]
                )
            start = None
    return result


def _role_intervals(
    composition: Any, *, role_threshold: float
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    role_logits = getattr(composition, "role_logits", None)
    role_probabilities = getattr(composition, "role_probabilities", None)
    temporal_mode = getattr(composition, "temporal_role_mode", None)
    hop_samples = getattr(composition, "frame_hop_samples", None)
    if not isinstance(role_logits, torch.Tensor) or not isinstance(
        role_probabilities, torch.Tensor
    ):
        raise Real10RenderError("model output lacks temporal role tensors")
    if role_logits.ndim != 3 or role_logits.size(0) != 1 or role_logits.size(-1) != 3:
        raise Real10RenderError("role logits must have shape [1, frames, 3]")
    if role_probabilities.shape != role_logits.shape:
        raise Real10RenderError("role probabilities do not match role logits")
    if not bool(torch.isfinite(role_logits).all()) or not bool(
        torch.isfinite(role_probabilities).all()
    ):
        raise Real10RenderError("temporal role output contains NaN or Inf")
    if temporal_mode == LEGACY_TEMPORAL_ROLE_MODE:
        role_ids = role_logits[0].argmax(dim=-1)
        anchor_active = role_ids == ROLE_ANCHOR
        answer_active = role_ids == ROLE_ANSWER
    elif temporal_mode == OVERLAP_AWARE_TEMPORAL_ROLE_MODE:
        anchor_active = role_probabilities[0, :, ROLE_ANCHOR] >= role_threshold
        answer_active = role_probabilities[0, :, ROLE_ANSWER] >= role_threshold
    else:
        raise Real10RenderError(f"unsupported temporal role mode: {temporal_mode!r}")
    anchor = _boolean_intervals(anchor_active, hop_samples=hop_samples)
    answer = _boolean_intervals(answer_active, hop_samples=hop_samples)
    return anchor, answer, _union_intervals(anchor, answer)


def _stem_relative_path(sample_id: str, stem_name: str) -> str:
    # The inference schema already constrains IDs, but keep the output boundary local.
    if not sample_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in sample_id
    ):
        raise Real10RenderError(f"unsafe sample ID for output path: {sample_id!r}")
    if stem_name not in {"evidence", "residual"}:
        raise Real10RenderError("unknown stem name")
    return f"stems/{sample_id}/{stem_name}.wav"


def _render_prediction_rows(
    *,
    staging: Path,
    prepared: PreparedRenderInputs,
    settings: RenderSettings,
    model: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = StableHashTokenizer(
        model.config.vocab_size, model.config.max_question_tokens
    )
    rows: list[dict[str, Any]] = []
    physical_total = 0
    effective_total = 0
    maximum_error = 0.0
    for record in prepared.selected_records:
        mixture_spec = prepared.mixtures[record.scene_id]
        mixture = torch.from_numpy(mixture_spec.samples).unsqueeze(0)
        mixture = mixture.to(torch.device(settings.device_type), dtype=torch.float32)
        question_feature = prepared.foundation_cache.question_features.get(
            record.sample_id
        )
        scene_feature = prepared.foundation_cache.scene_features.get(record.scene_id)
        if not isinstance(question_feature, torch.Tensor) or not isinstance(
            scene_feature, torch.Tensor
        ):
            raise Real10RenderError(
                f"foundation cache lacks features for {record.sample_id}"
            )
        question_clap = question_feature.unsqueeze(0).to(mixture.device)
        scene_clap = scene_feature.unsqueeze(0).to(mixture.device)
        with torch.inference_mode():
            output = model.forward_questions(
                mixture,
                [record.question],
                tokenizer,
                question_clap=question_clap,
                scene_clap=scene_clap,
            )
        evidence_tensor = getattr(output, "evidence", None)
        residual_tensor = getattr(output, "residual", None)
        if not isinstance(evidence_tensor, torch.Tensor) or not isinstance(
            residual_tensor, torch.Tensor
        ):
            raise Real10RenderError("model output lacks evidence or residual")
        if evidence_tensor.shape != (1, NUM_SAMPLES) or residual_tensor.shape != (
            1,
            NUM_SAMPLES,
        ):
            raise Real10RenderError("model stems must have shape [1, 320000]")
        evidence = np.ascontiguousarray(
            evidence_tensor[0].detach().float().cpu().numpy(), dtype=np.float32
        )
        residual = np.ascontiguousarray(
            residual_tensor[0].detach().float().cpu().numpy(), dtype=np.float32
        )
        _pcm_f32le_bytes(evidence)
        _pcm_f32le_bytes(residual)
        reconstruction = evidence + residual - mixture_spec.samples
        if not np.isfinite(reconstruction).all():
            raise Real10RenderError("stem reconstruction contains NaN or Inf")
        max_error = float(np.max(np.abs(reconstruction)))
        if max_error > MAX_RECONSTRUCTION_ABS_ERROR:
            raise Real10RenderError(
                "evidence/residual reconstruction gate failed for "
                f"{record.sample_id}: {max_error} > "
                f"{MAX_RECONSTRUCTION_ABS_ERROR}"
            )
        maximum_error = max(maximum_error, max_error)

        evidence_relative = _stem_relative_path(record.sample_id, "evidence")
        residual_relative = _stem_relative_path(record.sample_id, "residual")
        evidence_identity = write_deterministic_float_wav(
            staging / PurePosixPath(evidence_relative), evidence
        )
        residual_identity = write_deterministic_float_wav(
            staging / PurePosixPath(residual_relative), residual
        )
        evidence_identity["path"] = evidence_relative
        residual_identity["path"] = residual_relative

        composition = getattr(output, "composition", None)
        separation = getattr(output, "separation", None)
        if composition is None or separation is None:
            raise Real10RenderError("model output lacks composition or separation")
        if composition.temporal_role_mode != model.config.temporal_role_mode:
            raise Real10RenderError(
                "model output temporal role mode differs from checkpoint config"
            )
        if separation.semantic_separation_mode != model.config.semantic_separation_mode:
            raise Real10RenderError(
                "model output semantic separation mode differs from checkpoint config"
            )
        anchor, answer, union = _role_intervals(
            composition, role_threshold=settings.role_threshold
        )
        no_evidence_logit = getattr(composition, "no_evidence_logit", None)
        if not isinstance(
            no_evidence_logit, torch.Tensor
        ) or no_evidence_logit.shape != (1,):
            raise Real10RenderError("model output lacks one no-evidence logit")
        no_evidence_probability = float(
            no_evidence_logit[0].detach().float().sigmoid().cpu()
        )
        same_probability_tensor = getattr(
            composition, "same_semantic_probability", None
        )
        same_probability: float | None
        if same_probability_tensor is None:
            same_probability = None
        else:
            if not isinstance(
                same_probability_tensor, torch.Tensor
            ) or same_probability_tensor.shape != (1,):
                raise Real10RenderError("same-semantic probability has wrong shape")
            same_probability = float(same_probability_tensor[0].detach().float().cpu())

        physical = _positive_integer(
            getattr(separation, "physical_separator_forwards_per_batch", None),
            "physical separator forwards",
        )
        effective = _positive_integer(
            getattr(separation, "effective_separator_evaluations_per_record", None),
            "effective separator evaluations",
        )
        physical_total += physical
        effective_total += effective
        row: dict[str, Any] = {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "id": record.sample_id,
            "scene_id": record.scene_id,
            "split": record.split,
            "question_index": record.question_index,
            "relation": record.relation,
            "inference_record_sha256": inference_record_fingerprint(record),
            "mixture": {
                "manifest_path": mixture_spec.manifest_path,
                "file_sha256": mixture_spec.file_identity["sha256"],
                "pcm_f32le_sha256": mixture_spec.pcm_f32le_sha256,
                "size_bytes": mixture_spec.file_identity["size_bytes"],
                "sample_rate": SAMPLE_RATE,
                "num_channels": NUM_CHANNELS,
                "num_samples": NUM_SAMPLES,
                "duration_seconds": DURATION_SECONDS,
            },
            "prediction": {
                "anchor_intervals": anchor,
                "answer_intervals": answer,
                "union_intervals": union,
                "no_evidence_probability": no_evidence_probability,
                "no_evidence_prediction": (
                    no_evidence_probability >= settings.no_evidence_threshold
                ),
                "same_semantic_probability": same_probability,
                "role_threshold": float(settings.role_threshold),
                "no_evidence_threshold": float(settings.no_evidence_threshold),
                "temporal_role_mode": composition.temporal_role_mode,
                "semantic_separation_mode": separation.semantic_separation_mode,
            },
            "stems": {
                "evidence": evidence_identity,
                "residual": residual_identity,
            },
            "separator_calls": {
                "physical_forwards": physical,
                "effective_evaluations": effective,
            },
            "max_evidence_plus_residual_minus_mixture_abs_error": max_error,
        }
        validate_prediction_record(row)
        rows.append(row)
    return rows, {
        "physical_separator_forwards_↓": physical_total,
        "effective_separator_evaluations_↓": effective_total,
        "maximum_evidence_plus_residual_minus_mixture_abs_error_↓": maximum_error,
    }


def _write_prediction_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    chunks = [
        _canonical_json_bytes(dict(validate_prediction_record(row)), pretty=False)
        for row in rows
    ]
    _write_bytes_fsync(path, b"".join(chunks))


def _software_identity() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for package in ("numpy", "soundfile", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": platform.python_version(),
        **versions,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_runtime": torch.backends.cudnn.version(),
    }


def configure_deterministic_inference(settings: RenderSettings) -> dict[str, Any]:
    """Set deterministic FP32 inference and reject an unusable CUDA request."""

    settings.validate()
    if settings.device_type == "cuda":
        if not torch.cuda.is_available():
            raise Real10RenderError("CUDA was requested but is unavailable")
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in VALID_CUBLAS_WORKSPACE_CONFIGS:
            raise Real10RenderError(
                "deterministic CUDA rendering requires CUBLAS_WORKSPACE_CONFIG "
                "to be :4096:8 or :16:8 before process start"
            )
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if settings.device_type == "cuda":
        torch.cuda.manual_seed_all(settings.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": settings.seed,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


def _load_model(prepared: PreparedRenderInputs, settings: RenderSettings) -> Any:
    try:
        model = load_qces_checkpoint(
            dict(prepared.qces_checkpoint_payload),
            map_location=torch.device(settings.device_type),
            audiosep_repository_root=str(prepared.audiosep_root),
            audiosep_config_path=str(prepared.audiosep_config_path),
            audiosep_checkpoint_path=str(prepared.audiosep_checkpoint_path),
        ).eval()
    except Exception as error:
        raise Real10RenderError(f"failed to construct QCES model: {error}") from error
    if not isinstance(model.separator, AudioSepConditionedAdapter):
        raise Real10RenderError("loaded QCES model does not use AudioSep")
    if any(
        parameter.requires_grad for parameter in model.separator.ss_model.parameters()
    ):
        raise Real10RenderError("loaded AudioSep backbone is unexpectedly trainable")
    if any(
        parameter.is_floating_point() and parameter.dtype != torch.float32
        for parameter in model.parameters()
    ):
        raise Real10RenderError("renderer requires float32 model parameters")
    learned_modules = [model.composer]
    refiner = model.separator.separator_aware_refiner
    if refiner is not None:
        learned_modules.append(refiner)
    for module in learned_modules:
        for tensor in (*module.parameters(), *module.buffers()):
            if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                raise Real10RenderError("learned QCES state contains NaN or Inf")
    if model.config.to_dict() != prepared.qces_config.to_dict():
        raise Real10RenderError("loaded QCES config differs from checkpoint audit")
    return model


def _selected_mixture_rows(prepared: PreparedRenderInputs) -> list[dict[str, Any]]:
    selected_scene_ids = sorted(
        {record.scene_id for record in prepared.selected_records}
    )
    return [
        {
            "scene_id": scene_id,
            "manifest_path": prepared.mixtures[scene_id].manifest_path,
            "file_sha256": prepared.mixtures[scene_id].file_identity["sha256"],
            "pcm_f32le_sha256": prepared.mixtures[scene_id].pcm_f32le_sha256,
            "size_bytes": prepared.mixtures[scene_id].file_identity["size_bytes"],
        }
        for scene_id in selected_scene_ids
    ]


def build_run_spec(
    prepared: PreparedRenderInputs,
    settings: RenderSettings,
    *,
    method_freeze_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    method = build_method_identity(prepared, settings)
    return {
        "format": RUN_SPEC_FORMAT,
        "split": settings.split,
        "source_inference_manifest": {
            **dict(prepared.manifest_identity),
            "canonical_fingerprint": prepared.full_manifest_fingerprint,
            "record_count": len(prepared.records),
        },
        "selected_inference_view": {
            "canonical_fingerprint": prepared.selected_manifest_fingerprint,
            "record_count": len(prepared.selected_records),
            "scene_count": len(
                {record.scene_id for record in prepared.selected_records}
            ),
            "sample_ids": [record.sample_id for record in prepared.selected_records],
        },
        "selected_mixtures": _selected_mixture_rows(prepared),
        "foundation_cache": prepared.foundation_cache_identity,
        "method": method,
        "method_identity_sha256": canonical_json_sha256(method),
        "method_freeze_receipt": (
            dict(method_freeze_identity) if method_freeze_identity is not None else None
        ),
    }


def _assert_inputs_unchanged(prepared: PreparedRenderInputs) -> None:
    checks = (
        (
            "inference manifest",
            prepared.manifest_identity,
            file_identity(prepared.manifest_path),
        ),
        (
            "QCES checkpoint",
            prepared.qces_checkpoint_identity,
            file_identity(prepared.qces_checkpoint_path),
        ),
        (
            "AudioSep config",
            prepared.audiosep_config_identity,
            file_identity(prepared.audiosep_config_path),
        ),
        (
            "AudioSep checkpoint",
            prepared.audiosep_checkpoint_identity,
            file_identity(prepared.audiosep_checkpoint_path),
        ),
    )
    for name, before, after in checks:
        if not _same_file_content(before, after):
            raise Real10RenderError(f"{name} changed during rendering")
    if dict(prepared.audiosep_source_identity) != source_tree_identity(
        prepared.audiosep_root
    ):
        raise Real10RenderError("AudioSep source tree changed during rendering")
    if dict(prepared.runtime_source_identity) != runtime_source_identity():
        raise Real10RenderError("QCES runtime source changed during rendering")
    for scene_id, mixture in prepared.mixtures.items():
        if not _same_file_content(mixture.file_identity, file_identity(mixture.path)):
            raise Real10RenderError(
                f"canonical mixture changed during rendering: {scene_id}"
            )
    cache = prepared.foundation_cache_identity
    cache_paths = (
        ("receipt", Path(cache["receipt"]["path"]), cache["receipt"]),
        (
            "question features",
            Path(cache["question_feature_artifact"]["path"]),
            cache["question_feature_artifact"],
        ),
        (
            "scene features",
            Path(cache["scene_feature_artifact"]["path"]),
            cache["scene_feature_artifact"],
        ),
    )
    for name, path, before in cache_paths:
        if not _same_file_content(before, file_identity(path)):
            raise Real10RenderError(f"foundation cache {name} changed during rendering")


def _stem_aggregate(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = []
    for row in sorted(rows, key=lambda item: item["id"]):
        payload.append(
            {
                "id": row["id"],
                "evidence": row["stems"]["evidence"],
                "residual": row["stems"]["residual"],
            }
        )
    return canonical_json_sha256(
        {"format": STEM_AGGREGATE_FORMAT, "artifacts": payload}
    )


def _read_prediction_manifest(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise Real10RenderError(
                        f"prediction manifest line {line_number} is malformed"
                    )
                row = json.loads(line)
                rows.append(validate_prediction_record(row))
    except json.JSONDecodeError as error:
        raise Real10RenderError("prediction manifest contains invalid JSON") from error
    if not rows:
        raise Real10RenderError("prediction manifest is empty")
    return rows


def _verify_stem_artifact(output_dir: Path, identity: Mapping[str, Any]) -> np.ndarray:
    parsed = _parse_audio_identity(identity, "stored stem")
    path = (output_dir / PurePosixPath(parsed["path"])).resolve()
    try:
        path.relative_to(output_dir.resolve())
    except ValueError as error:
        raise Real10RenderError("stored stem path escapes output directory") from error
    actual = file_identity(path)
    if (
        actual["sha256"] != parsed["file_sha256"]
        or actual["size_bytes"] != parsed["size_bytes"]
    ):
        raise Real10RenderError(f"stored stem file identity mismatch: {path}")
    samples = _decode_float_wav_exact(path, allow_silence=True)
    if pcm_f32le_sha256(samples) != parsed["pcm_f32le_sha256"]:
        raise Real10RenderError(f"stored stem PCM identity mismatch: {path}")
    return samples


def _expected_output_paths(
    output_dir: Path, rows: Sequence[Mapping[str, Any]]
) -> set[Path]:
    expected = {
        (output_dir / PREDICTION_MANIFEST_FILENAME).resolve(),
        (output_dir / RENDER_RECEIPT_FILENAME).resolve(),
    }
    for row in rows:
        for stem_name in ("evidence", "residual"):
            expected.add(
                (output_dir / PurePosixPath(row["stems"][stem_name]["path"])).resolve()
            )
    return expected


def validate_existing_render(
    output_dir: Path,
    *,
    expected_run_spec: Mapping[str, Any],
    selected_records: Sequence[QCESReal10InferenceRecord],
    mixtures: Mapping[str, CanonicalMixture],
) -> Mapping[str, Any]:
    """Resume only when every receipt, row, and WAV bit identity is exact."""

    output = output_dir.resolve()
    if not output.is_dir() or output.is_symlink():
        raise Real10RenderError("resume output is not a regular directory")
    receipt = _read_json(output / RENDER_RECEIPT_FILENAME, "render receipt")
    _exact_keys(
        receipt,
        {
            "format",
            "purpose",
            "run_spec",
            "run_identity_sha256",
            "execution",
            "counts",
            "gates",
            "artifacts",
            "input_boundary",
        },
        "render receipt",
    )
    if receipt.get("format") != RENDER_RECEIPT_FORMAT:
        raise Real10RenderError("resume output has an unsupported receipt")
    expected_run_hash = canonical_json_sha256(expected_run_spec)
    if (
        receipt.get("run_spec") != expected_run_spec
        or receipt.get("run_identity_sha256") != expected_run_hash
    ):
        raise Real10RenderError("resume output belongs to a different exact run")
    rows = _read_prediction_manifest(output / PREDICTION_MANIFEST_FILENAME)
    expected_by_id = {record.sample_id: record for record in selected_records}
    if set(row["id"] for row in rows) != set(expected_by_id):
        raise Real10RenderError("resume prediction IDs differ from selected manifest")
    physical_total = 0
    effective_total = 0
    reconstructed_errors: list[float] = []
    for row in rows:
        record = expected_by_id[row["id"]]
        if (
            row["scene_id"] != record.scene_id
            or row["split"] != record.split
            or row["question_index"] != record.question_index
            or row["relation"] != record.relation
            or row["inference_record_sha256"] != inference_record_fingerprint(record)
        ):
            raise Real10RenderError("resume prediction/inference binding mismatch")
        mixture = mixtures.get(record.scene_id)
        if mixture is None:
            raise Real10RenderError("resume input lacks its canonical mixture")
        expected_mixture_row = {
            "manifest_path": mixture.manifest_path,
            "file_sha256": mixture.file_identity["sha256"],
            "pcm_f32le_sha256": mixture.pcm_f32le_sha256,
            "size_bytes": mixture.file_identity["size_bytes"],
            "sample_rate": SAMPLE_RATE,
            "num_channels": NUM_CHANNELS,
            "num_samples": NUM_SAMPLES,
            "duration_seconds": DURATION_SECONDS,
        }
        if row["mixture"] != expected_mixture_row:
            raise Real10RenderError("resume prediction mixture binding mismatch")
        evidence = _verify_stem_artifact(output, row["stems"]["evidence"])
        residual = _verify_stem_artifact(output, row["stems"]["residual"])
        reconstructed_error = float(
            np.max(np.abs(evidence + residual - mixture.samples))
        )
        if (
            reconstructed_error
            != row["max_evidence_plus_residual_minus_mixture_abs_error"]
        ):
            raise Real10RenderError("resume reconstruction scalar is not reproducible")
        reconstructed_errors.append(reconstructed_error)
        physical_total += row["separator_calls"]["physical_forwards"]
        effective_total += row["separator_calls"]["effective_evaluations"]
        if reconstructed_error > MAX_RECONSTRUCTION_ABS_ERROR:
            raise Real10RenderError("resume output fails reconstruction gate")
    observed_maximum = max(reconstructed_errors)
    if receipt["gates"] != {
        "maximum_evidence_plus_residual_minus_mixture_abs_error_threshold_↓": (
            MAX_RECONSTRUCTION_ABS_ERROR
        ),
        "observed_maximum_↓": observed_maximum,
        "reconstruction_gate_passed": True,
    }:
        raise Real10RenderError("resume receipt reconstruction gate is invalid")
    manifest_identity = file_identity(output / PREDICTION_MANIFEST_FILENAME)
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise Real10RenderError("resume receipt lacks artifacts")
    declared_manifest = artifacts.get("prediction_manifest")
    if not isinstance(declared_manifest, Mapping) or (
        declared_manifest.get("sha256") != manifest_identity["sha256"]
        or declared_manifest.get("size_bytes") != manifest_identity["size_bytes"]
        or declared_manifest.get("canonical_fingerprint")
        != prediction_manifest_fingerprint(rows)
    ):
        raise Real10RenderError("resume prediction manifest identity mismatch")
    if artifacts.get("stem_aggregate_sha256") != _stem_aggregate(rows):
        raise Real10RenderError("resume stem aggregate identity mismatch")
    expected_counts = {
        "prediction_records_↑": len(rows),
        "unique_scenes_↑": len({row["scene_id"] for row in rows}),
        "physical_separator_forwards_↓": physical_total,
        "effective_separator_evaluations_↓": effective_total,
        "maximum_evidence_plus_residual_minus_mixture_abs_error_↓": observed_maximum,
    }
    if receipt["counts"] != expected_counts:
        raise Real10RenderError("resume receipt counts are not reproducible")
    if receipt["purpose"] != "label_free_qces_real10_evidence_and_residual_render":
        raise Real10RenderError("resume receipt purpose is invalid")
    if receipt["input_boundary"] != {
        "accepted_manifest_schema": INFERENCE_SCHEMA_VERSION,
        "scoring_or_gold_manifest_opened": False,
        "human_answer_fields_consumed": False,
        "human_temporal_fields_consumed": False,
        "clean_reference_metrics_emitted": False,
    }:
        raise Real10RenderError("resume receipt input boundary is invalid")
    execution = _exact_keys(
        receipt["execution"],
        {"device", "batch_size", "precision", "determinism", "software"},
        "resume execution",
    )
    method = expected_run_spec["method"]
    if (
        execution["device"] != method["device_type"]
        or execution["batch_size"] != 1
        or execution["precision"] != "float32_no_autocast"
        or not isinstance(execution["determinism"], Mapping)
        or not isinstance(execution["software"], Mapping)
    ):
        raise Real10RenderError("resume execution provenance is invalid")
    actual_files = {path.resolve() for path in output.rglob("*") if path.is_file()}
    expected_files = _expected_output_paths(output, rows)
    if actual_files != expected_files:
        raise Real10RenderError("resume output has missing or undeclared files")
    actual_directories = {path.resolve() for path in output.rglob("*") if path.is_dir()}
    expected_directories = {path.parent for path in expected_files}
    expected_directories.discard(output)
    expected_directories.add((output / "stems").resolve())
    if actual_directories != expected_directories or any(
        path.is_symlink() for path in output.rglob("*")
    ):
        raise Real10RenderError("resume output has undeclared directories or symlinks")
    return receipt


def render_prediction_set(
    *,
    prepared: PreparedRenderInputs,
    settings: RenderSettings,
    output_dir: Path,
    allow_real_test: bool = False,
    method_freeze_receipt_path: Path | None = None,
    resume: bool = False,
    model_loader: Callable[[PreparedRenderInputs, RenderSettings], Any] = _load_model,
) -> RenderResult:
    """Render one split atomically, or verify and reuse an exact prior result."""

    settings.validate()
    freeze_identity: Mapping[str, Any] | None = None
    if settings.split == "real_test":
        if not allow_real_test or method_freeze_receipt_path is None:
            raise Real10RenderError(
                "real_test requires --allow-real-test and a method-freeze receipt"
            )
        expected_method = build_method_identity(prepared, settings)
        _, freeze_identity = validate_method_freeze_receipt(
            method_freeze_receipt_path,
            expected_method=expected_method,
        )
    elif allow_real_test or method_freeze_receipt_path is not None:
        raise Real10RenderError(
            "real-test authorization flags are invalid for a real_dev render"
        )
    run_spec = build_run_spec(
        prepared,
        settings,
        method_freeze_identity=freeze_identity,
    )
    _reject_symlink_components(output_dir)
    output = output_dir.resolve()
    if output.exists() or output.is_symlink():
        if not resume:
            raise FileExistsError(f"refusing to overwrite output: {output}")
        receipt = validate_existing_render(
            output,
            expected_run_spec=run_spec,
            selected_records=prepared.selected_records,
            mixtures=prepared.mixtures,
        )
        _assert_inputs_unchanged(prepared)
        return RenderResult(output_dir=output, receipt=receipt, resumed=True)
    if resume:
        raise Real10RenderError("--resume requires an existing exact output")

    determinism = configure_deterministic_inference(settings)
    model = model_loader(prepared, settings)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.building-", dir=output.parent)
    )
    try:
        rows, metrics = _render_prediction_rows(
            staging=staging,
            prepared=prepared,
            settings=settings,
            model=model,
        )
        prediction_path = staging / PREDICTION_MANIFEST_FILENAME
        _write_prediction_manifest(prediction_path, rows)
        _assert_inputs_unchanged(prepared)
        prediction_identity = file_identity(prediction_path)
        receipt: MutableMapping[str, Any] = {
            "format": RENDER_RECEIPT_FORMAT,
            "purpose": "label_free_qces_real10_evidence_and_residual_render",
            "run_spec": run_spec,
            "run_identity_sha256": canonical_json_sha256(run_spec),
            "execution": {
                "device": settings.device_type,
                "batch_size": 1,
                "precision": "float32_no_autocast",
                "determinism": determinism,
                "software": _software_identity(),
            },
            "counts": {
                "prediction_records_↑": len(rows),
                "unique_scenes_↑": len({row["scene_id"] for row in rows}),
                **metrics,
            },
            "gates": {
                "maximum_evidence_plus_residual_minus_mixture_abs_error_threshold_↓": (
                    MAX_RECONSTRUCTION_ABS_ERROR
                ),
                "observed_maximum_↓": metrics[
                    "maximum_evidence_plus_residual_minus_mixture_abs_error_↓"
                ],
                "reconstruction_gate_passed": True,
            },
            "artifacts": {
                "prediction_manifest": {
                    "filename": PREDICTION_MANIFEST_FILENAME,
                    "sha256": prediction_identity["sha256"],
                    "size_bytes": prediction_identity["size_bytes"],
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "canonical_fingerprint": prediction_manifest_fingerprint(rows),
                },
                "stem_aggregate_sha256": _stem_aggregate(rows),
            },
            "input_boundary": {
                "accepted_manifest_schema": INFERENCE_SCHEMA_VERSION,
                "scoring_or_gold_manifest_opened": False,
                "human_answer_fields_consumed": False,
                "human_temporal_fields_consumed": False,
                "clean_reference_metrics_emitted": False,
            },
        }
        _assert_public_payload_has_no_invalid_claim_fields(
            receipt, context="render receipt"
        )
        _write_bytes_fsync(
            staging / RENDER_RECEIPT_FILENAME,
            _canonical_json_bytes(receipt, pretty=True),
        )
        _fsync_directory_tree(staging)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output appeared during render: {output}")
        _rename_directory_noreplace(staging, output)
        _fsync_directory(output.parent)
        return RenderResult(output_dir=output, receipt=receipt, resumed=False)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        del model
        if settings.device_type == "cuda":
            torch.cuda.empty_cache()


__all__ = [
    "CanonicalMixture",
    "MAX_RECONSTRUCTION_ABS_ERROR",
    "METHOD_FREEZE_RECEIPT_FORMAT",
    "PREDICTION_MANIFEST_FILENAME",
    "PREDICTION_SCHEMA_VERSION",
    "PreparedRenderInputs",
    "RENDER_RECEIPT_FILENAME",
    "RENDER_RECEIPT_FORMAT",
    "Real10FoundationFeatureCache",
    "Real10RenderError",
    "RenderResult",
    "RenderSettings",
    "build_method_identity",
    "build_run_spec",
    "configure_deterministic_inference",
    "load_canonical_mixtures",
    "load_real10_foundation_feature_cache",
    "pcm_f32le_sha256",
    "prediction_manifest_fingerprint",
    "prepare_render_inputs",
    "read_inference_manifest",
    "render_prediction_set",
    "runtime_source_identity",
    "validate_existing_render",
    "validate_method_freeze_receipt",
    "validate_prediction_record",
    "write_deterministic_float_wav",
    "write_method_freeze_receipt",
]
